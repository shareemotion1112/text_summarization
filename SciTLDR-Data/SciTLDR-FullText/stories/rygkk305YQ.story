This paper proposes a neural end-to-end text-to-speech (TTS) model which can control latent attributes in the generated speech that are rarely annotated in the training data, such as speaking style, accent, background noise, and recording conditions.

The model is formulated as a conditional generative model with two levels of hierarchical latent variables.

The first level is a categorical variable, which represents attribute groups (e.g. clean/noisy) and provides interpretability.

The second level, conditioned on the first, is a multivariate Gaussian variable, which characterizes specific attribute configurations (e.g. noise level, speaking rate) and enables disentangled fine-grained control over these attributes.

This amounts to using a Gaussian mixture model (GMM) for the latent distribution.

Extensive evaluation demonstrates its ability to control the aforementioned attributes.

In particular, it is capable of consistently synthesizing high-quality clean speech regardless of the quality of the training data for the target speaker.

Recent development of neural sequence-to-sequence TTS models has shown promising results in generating high fidelity speech without the need of handcrafted linguistic features BID30 BID37 BID2 .

These models rely heavily on a encoderdecoder neural network structure BID31 BID5 that maps a text sequence to a sequence of speech frames.

Extensions to these models have shown that attributes such as speaker identity can be controlled by conditioning the decoder on additional attribute labels BID3 .There are many speech attributes aside from speaker identity that are difficult to annotate, such as speaking style, prosody, recording channel, and noise levels. ; model such latent attributes through conditional auto-encoding, by extending the decoder inputs to include a vector inferred from the target speech which aims to capture the residual attributes that are not specified by other input streams, in addition to text and a speaker label.

These models have shown convincing results in synthesizing speech that resembles the prosody or the noise conditions of the reference speech, which may not have the same text or speaker identity as the target speech.

Nevertheless, the presence of multiple latent attributes is common in crowdsourced data such as BID26 , in which prosody, speaker, and noise conditions all vary simultaneously.

Using such data, simply copying the latent attributes from a reference is insufficient if one desires to synthesize speech that mimics the prosody of the reference, but is in the same noise condition as another.

If the latent representation were disentangled, these generating factors could be controlled independently.

Furthermore, it is can useful to construct a systematic method for synthesizing speech with random latent attributes, which would facilitate data augmentation BID33 BID12 BID8 by generating diverse examples.

These properties were not explicitly addressed in the previous studies, which model variation of a single latent attribute.

Motivated by the applications of sampling, inferring, and independently controlling individual attributes, we build off of and extend Tacotron 2 to model two separate latent spaces: one for labeled (i.e. related to speaker identity) and another for unlabeled attributes.

Each latent variable is modeled in a variational autoencoding BID22 ) framework using Gaussian mixture priors.

The resulting latent spaces (1) learn disentangled attribute representations, where each dimension controls a different generating factor; (2) discover a set of interpretable clusters, each of which corresponds to a representative mode in the training data (e.g., one cluster for clean speech and another for noisy speech); and (3) provide a systematic sampling mechanism from the learned prior.

The proposed model is extensively evaluated on four datasets with subjective and objective quantitative metrics, as well as comprehensive qualitative studies.

Experiments confirm that the proposed model is capable of controlling speaker, noise, and style independently, even when variation of all attributes is present but unannotated in the train set.

Our main contributions are as follows:??? We propose a principled probabilistic hierarchical generative model, which improves (1) sampling stability and disentangled attribute control compared to e.g. the GST model of , and (2) interpretability and quality compared to e.g. BID0 .???

The model formulation explicitly factors the latent encoding by using two mixture distributions to separately model supervised speaker attributes and latent attributes in a disentangled fashion.

This makes it straightforward to condition the model output on speaker and latent encodings inferred from different reference utterances.??? To the best of our knowledge, this work is the first to train a high-quality controllable textto-speech system on real found data containing significant variation in recording condition, speaker identity, as well as prosody and style.

Previous results on similar data focused on speaker modeling , and did not explicitly address modeling of prosody and background noise.

Leveraging disentangled speaker and latent attribute encodings, the proposed model is capable of inferring the speaker attribute representation from a noisy utterance spoken by a previously unseen speaker, and using it to synthesize high-quality clean speech that approximates the voice of that speaker.

Tacotron-like TTS systems take a text sequence Y t and an optional observed categorical label (e.g. speaker identity)

y o as input, and use an autoregressive decoder to predict a sequence of acoustic features X frame by frame.

Training such a system to minimize a mean squared error reconstruction loss can be regarded as fitting a probabilistic model p(X | Y t , y o ) = n p(x n | x 1 , x 2 , . . .

, x n???1 , Y t , y o ) that maximizes the likelihood of generating the training data, where the conditional distribution of each frame x n is modeled as fixed-variance isotropic Gaussian whose mean is predicted by the decoder at step n. Such a model effectively integrates out other unlabeled latent attributes like prosody, and produces a conditional distribution with higher variance.

As a result, the model would opaquely produce speech with unpredictable latent attributes.

To enable control of those attributes, we adopt a graphical model with hierarchical latent variables, which captures such attributes.

Below we explain how the formulation leads to interpretability and disentanglement, supports sampling, and propose efficient inference and training methods.

Two latent variables y l and z l are introduced in addition to the observed variables, X, Y t , and y o , as shown in the graphical model in the left of FIG0 .

y l is a K-way categorical discrete variable, named latent attribute class, and z l is a D-dimensional continuous variable, named latent attribute representation.

Throughout the paper, we use y * and z * to denote discrete and continuous variables, respectively.

To generate speech X conditioned on the text Y t and observed attribute y o , y l is first sampled from its prior, p(y l ), then a latent attribute representation z l is sampled from the conditional distribution p(z l | y l ).

Finally, a sequence of speech frames is drawn from p(X | Y t , y o , z l ), parameterized by the synthesizer neural network.

The joint probability can be written as: Specifically, it is assumed that p(y l ) = K ???1 to be a non-informative prior to encourage every component to be used, and p(z l | y l ) = N (?? y l , diag(?? y l )) to be diagonal-covariance Gaussian with learnable means and variances.

As a result, the marginal prior of z l becomes a GMM with diagonal covariances and equal mixture weights.

We hope this GMM latent model can better capture the complexity of unseen attributes.

Furthermore, in the presence of natural clusters of unseen attributes, the proposed model can achieve interpretability by learning to assign instances from different clusters to different mixture components.

The covariance matrix of each mixture component is constrained to be diagonal to encourage each dimension to capture a statistically uncorrelated factor.

DISPLAYFORM0

The conditional output distribution p(X | Y t , y o , z l ) is parameterized with a neural network.

Following the VAE framework of BID22 , a variational distribution q(y l | X) q(z l | X) is used to approximate the posterior p(y l , z l | X, Y t , y o ), which assumes that the posterior of unseen attributes is independent of the text and observed attributes.

The approximated posterior for z l , q(z l | X), is modeled as a Gaussian distribution with diagonal covariance matrix, whose mean and variance are parameterized by a neural network.

For q(y l | X), instead of introducing another neural network, we configure it to be an approximation of p(y l | X) that reuses q(z l | X) as follows: DISPLAYFORM0 which enjoys the closed-form solution of Gaussian mixture posteriors, p(y l | z l ).

Similar to VAE, the model is trained by maximizing its evidence lower bound (ELBO), as follows: DISPLAYFORM1 where q(z l | X) is estimated via Monte Carlo sampling, and all components are differentiable thanks to reparameterization.

Details can be found in Appendix A.

Categorical observed labels, such as speaker identity, can often be seen as a categorization from a continuous attribute space, which for example could model a speaker's characteristic F 0 range and vocal tract shape.

Given an observed label, there may still be some variation of these attributes.

We are interested in learning this continuous attribute space for modeling within-class variation and inferring a representation from an instance of an unseen class for one-shot learning.

To achieve this, a continuous latent variable, z o , named the observed attribute representation, is introduced between the categorical observed label y o and speech X, as shown on the right of FIG0 .

Each observed class (e.g. each speaker) forms a mixture component in this continuous space, whose conditional distribution is a diagonal-covariance Gaussian p(z o | y o ) = N (?? yo , diag(?? yo )).

With this formulation, speech from an observed class y o is now generated by conditioning on Y t , z l , and a sample z o drawn from p(z o | y o ).

As before, a variational distribution q(z o | X), parameterized by a neural network, is used to approximate the true posterior, where the ELBO becomes: The model is comprised of three modules: a synthesizer, a latent encoder, and an observed encoder.

DISPLAYFORM0 To encourage z o to disentangle observed attributes from latent attributes, the variances of p(z o | y o ) are initialized to be smaller than those of p(z l | y l ).

The intuition is that this space should capture variation of attributes that are highly correlated with the observed labels, so the conditional distribution of all dimensions should have relatively small variance for each mixture component.

Experimental results verify the effectiveness, and similar design is used in BID11 .

In the extreme case where the variance is fixed and approaches zero, this formulation converges to using an lookup table.

We parameterize three distributions: DISPLAYFORM0 , and q(z l |X) with neural networks, referred to in FIG1 as the synthesizer, observed encoder, and latent encoder, respectively.

The synthesizer is based on the Tacotron 2 architecture , which consists of a text encoder and an autoregressive speech decoder.

The former maps Y t to a sequence of text encodings Z t , and the latter predicts the mean of p( DISPLAYFORM1 We inject the latent variables z l and y o (or z o ) into the decoder by concatenating them to the decoder input at each step.

Text Y t and speech X are represented as a sequence of phonemes and a sequence of mel-scale filterbank coefficients, respectively.

To speed up inference, we use a WaveRNN-based neural vocoder BID16 instead of WaveNet BID35 , to invert the predicted mel-spectrogram to a time-domain waveform.

The two posteriors, q(z l | X) and q(z o | X), are both parameterized by a recurrent encoder that maps a variable-length mel-spectrogram to two fixed-dimensional vectors, corresponding to the posterior mean and log variance, respectively.

Full architecture details can be found in Appendix B.

The proposed GMVAE-Tacotron model is most related to , , BID9 , which introduce a reference embedding to model prosody or noise.

The first uses an autoencoder to extract a prosody embedding from a reference speech spectrogram.

The second Global Style Token (GST) model constrains a reference embedding to be a weighted combination of a fixed set of learned vectors, while the third further restricts the weights to be one-hot, and is built on a conventional parametric speech synthesizer BID40 ).

The main focus of these approaches was style transfer from a reference audio example.

They provide neither a systematic sampling mechanism nor disentangled representations as we show in Section 4.3.1.Similar to these approaches, BID0 extend VoiceLoop ) with a latent reference embedding generated by a VAE, using a centered fixed-variance isotropic Gaussian prior for the latent attributes.

This provides a principled mechanism for sampling from the latent distribution, but does not provide interpretability.

In contrast, GMVAE-Tacotron models latent attributes using a mixture distribution, which allows automatic discovery of latent attribute clusters.

This structure makes it easier to interpret the underlying latent space.

Specifically, we show in Section 4.1 that the mixture parameters can be analyzed to understand what each component corresponds to, similar to GST.

In addition, the most distinctive dimensions of the latent space can be identified using an inter-/intra-component variance ratio, which e.g. can identify the dimension controlling the background noise level as shown in Section 4.2.2.Finally, the extension described in Section 2.3 adds a second mixture distribution to additionally models speaker attributes.

This formulation learns disentangled speaker and latent attribute represen-tations, which can be used to approximate the voice of speakers previously unseen during training.

This speaker model is related to , which controls the output speaker identity using speaker embeddings, and trains a separate regression model to predict them from the audio.

This can be regarded as a special case of the proposed model where the variance of z o is set to be almost zero, such that a speaker always generates a fixed representation; meanwhile, the posterior model q(z o | X) corresponds to their embedding predictor, because it now aims to predict a fixed embedding for each speaker.

Using a mixture distribution for latent variables in a VAE was explored in Dilokthanakul et al. FORMULA0 ; BID25 BID15 for unconditional image generation and text topic modeling.

These models correspond to the sub-graph y l ??? z l ??? X in FIG0 .

The proposed model provides extra flexibility to model both latent and observed attributes in a conditional generation scenario.

BID11

The proposed GMVAE-Tacotron was evaluated on four datasets, spanning a wide degree of variations in speaker, recording channel conditions, background noise, prosody, and speaking styles.

For all experiments, y o was an observed categorical variable whose cardinality is the number of speakers in the training set if used, y l was configured to be a 10-way categorical variable (K = 10), and z l and z o (if used) were configured to be 16-dimensional variables (D = 16).

Tacotron 2 with a speaker embedding table was used as the baseline for all experiments.

For all other variants (e.g., GST), the reference encoder follows .

Each model was trained for at least 200k steps to maximize the ELBO in equation 3 or equation 4 using the Adam optimizer.

A list of detailed hyperparameter settings can be found in Appendix C. Quantitative subjective evaluations relied on crowd-sourced mean opinion scores (MOS) rating the naturalness of the synthesized speech by native speakers using headphones, with scores ranging from 1 to 5 in increments of 0.5.

For single speaker datasets each sample was rated by 6 raters, while for other datasets each sample was rated by a single rater.

We strongly encourage readers to listen to the samples on the demo page.

To evaluate the ability of GMVAE-Tacotron to model speaker variation and discover meaningful speaker clusters, we used a proprietary dataset of 385 hours of high-quality English speech from 84 professional voice talents with accents from the United States (US), Great Britain (GB), Australia (AU), and Singapore (SG).

Speaker labels were not seen during training (y o and z o were unused), and were only used for evaluation.

To probe the interpretability of the model, we computed the distribution of mixture components y l for utterances of a particular accent or gender.

Specifically, we collected at most 100 utterances from each of the 44 speakers with at least 20 test utterances (2,332 in total), and assigned each utterance to the component with the highest posterior probability: arg max y l q(y l |X).Figure 3 plots the assignment distributions for each gender and accent in this set.

Most components were only used to model speakers from one gender.

Each component which modeled both genders (0, 2, and 9) only represented a subset of accents (US, US, and AU/GB, respectively).

We also found that the several components which modeled US female speakers (3, 5, and 6) actually modeled groups of speakers with distinct characteristics, e.g. different F 0 ranges as shown in Appendix E. To quantify the association between speaker and mixture components, we computed the assignment consistency w.r.t.

speaker: .

The resulting consistency was 92.9%, suggesting that the components group utterances by speaker and group speakers by gender or accent.

We also explored what each dimension of z l controlled by decoding with different values of the target dimension, keeping all other factors fixed.

We discovered that there were individual dimensions which controlled F 0 , speaking rate, accent, length of starting silence, etc., demonstrating the disentangled nature of the learned latent attribute representation.

Appendix E contains visualization of attribute control and additional quantitative evaluation of using z l for gender/accent/speaker classification.

High quality data can be both expensive and time consuming to record.

Vast amounts of rich real-life expressive speech are often noisy and difficult to label.

In this section we demonstrate that our model can synthesize clean speech directly from noisy data by disentangling the background noise level from other attributes, allowing it to be controlled independently.

As a first experiment, we artificially generated training sets using a room simulator BID18 to add background noise and reverberation to clean speech from the multi-speaker English corpus used in the previous section.

We used music and ambient noise sampled from YouTube and recordings of "daily life" environments as noise signals, mixed at signal-to-noise ratios (SNRs) ranging from 5-25dB. The reverberation time varied between 100 and 900ms.

Noise was added to a random selection of 50% of utterances by each speaker, holding out two speakers (one male and one female) for whom noise was added to all of their utterances.

This construction was used to evaluate the ability of the model to synthesize clean speech for speakers whose training utterances were all corrupted by noise.

In this experiment, we provided speaker labels y o as input to the decoder, and only expect the latent attribute representations z l to capture the acoustic condition of each utterance.

Unlike clustering speakers, we expected that latent attributes would naturally divide into two categories: clean and noisy.

To verify this hypothesis, we plotted the Euclidean distance between means of each pair of components on the left of FIG4 , which clearly form two distinct clusters.

The right two plots in FIG4 show the mel-spectrograms of two synthesized utterances of the same text and speaker, conditioned on the means of two different components, one from each group.

It clearly presents the samples (in fact, all the samples) drawn from components in group one were noisy, while the samples drawn from the other components were clean.

See Appendix F for more examples.

We next explored if the level of noise was dominated by a single latent dimension, and whether we could determine such a dimension automatically.

For this purpose, we adopted a per-dimension LDA, which computed a between and within-mixture scattering ratio: DISPLAYFORM0 , where ?? y l ,d and ?? y l ,d are the d-th dimension mean and variance of mixture component y l , and?? l,d is the d-th dimension mean of the marginal dis- DISPLAYFORM1 .

This is a scale-invariant metric of the degree of separation between components in each latent dimension.

We discovered that the most discriminative dimension had a scattering ratio r 13 = 21.5, far larger than the second largest r 11 = 0.6.

Drawing samples and traversing values along the target dimension DISPLAYFORM2 Target dimension value Noisy component (0) dim 0 dim 1 dim 2 dim 3 dim 4 dim 5 dim 6 dim 7 dim 8 dim 9 dim 10 dim 11 dim 12 dim 13 dim 14 dim 15Figure 5: SNR as a function of the value in each latent dimension, comparing clean (left) and noisy (right) components.

while keeping others fixed demonstrates that dimension's effect on the output.

To determine the effective range of the target dimension, we approximate the multimodal distribution as a Gaussian and evaluate values spanning four standard deviations?? l,d around the mean.

To quantify the effect on noise level, we estimate the SNR without a reference clean signal following BID17 .

The results show that the noise level was clearly controlled by manipulating the 13th dimension, and remains nearly constant as the other dimensions vary, verifying that control has been isolated to the identified dimension.

The small degree of variation, e.g. in dimensions 2 and 4, occurs because some of those dimensions control attributes which directly affect the synthesized noise, such as type of noise (musical/white noise) and initial background noise offset, and therefore also affect the estimated SNR.

Appendix F.2 contains an additional spectrogram demonstration of noise level control by manipulating the identified dimension.

In this section, we evaluated synthesis quality for the two held out noisy speakers.

Evaluation metrics included subjective naturalness MOS ratings and an objective SNR metric.

TAB1 compares the proposed model with a baseline, a 16-token GST, and a VAE variant which replaces the GMM prior with an isotropic Gaussian.

To encourage synthesis of clean audio under each model we manually selected the cleanest token (weight=0.15) for GST, used the Gaussian prior mean (i.e. a zero vector) for VAE, and the mean of a clean component for GMVAE.

For the VAE model, the mean captured the average condition, which still exhibited a moderate level of noise, resulting in a lower SNR and MOS.

The generated speech from the GST was cleaner, however raters sometimes found its prosody to be unnatural.

Note that it is possible that another token would obtain a different trade-off between prosody and SNR, and using multiple tokens could improve both.

Finally, the proposed model synthesized both natural and high-quality speech, with the highest MOS and SNR.

Prosody and speaking style is another important factor for human speech other than speaker and noise.

Control of these aspects of the synthesize speech is essential to building an expressive TTS system.

In this section, we evaluated the ability of the proposed model to sample and control speaking styles.

A single speaker US English audiobook dataset of 147 hours, recorded by professional speaker, Catherine Byers, from the 2013 Blizzard Challenge BID20 ) is used for training.

The data incorporated a wide range of prosody variation.

We used an evaluation set of 150 audiobook sentences, including many long phrases.

TAB2 shows the naturalness MOS between baseline and proposed model conditioning on the same z l , set to the mean of a selected y l , for all utterances.

The results show that the prior already captured a common prosody, which could be used to synthesize more naturally sounding speech with a lower variance compared to the baseline.

Figure 7: (a) Mel-spectrograms of two unnatural GST samples when setting the weight for one token -0.1: first with tremolo at the end, and second with abnormally long duration for the first syllable.

(b) F 0 tracks and spectrograms from GMVAE-Tacotron using different values for the "speed" dimension.

Compared to GST, one primary advantage of the proposed model is that it supports random sampling of natural speech from the prior.

Figure 6 illustrates such samples, where the same text is synthesized with wide variation in speaking rate, rhythm, and F 0 .

In contrast, the GST model does not define a prior for normalized token weights, requiring weights to be chosen heuristically or by fitting a distribution after training.

Empirically we found that the GST weight simplex was not fully exploited during training and that careful tuning was required to find a stable sampling region.

An additional advantage of GMVAE-Tacotron is that it learns a representation which disentangles these attributes, enabling them to be controlled independently.

Specifically, latent dimensions in the proposed model are conditionally independent, while token weights of GST are in fact correlated.

Figure 7 (b) contains an example of the proposed model traversing the "speed" dimension with three values: DISPLAYFORM0 are the marginal distribution mean and standard deviation, respectively, of that dimension.

Their F 0 tracks, obtained using the YIN (De Cheveign?? & Kawahara, 2002) F 0 tracker, are shown on the left.

From these we can observe that the shape of the F 0 contours did not change much.

They were simply stretched horizontally, indicating that only the speed was manipulated.

In contrast, the style control of GST is more entangled, as shown in FIG3 ), where the F 0 also changed while controlling speed.

Appendix G contains a quantitative analysis of disentangled latent attribute control, and additional evaluation of style transfer, demonstrating the ability of the proposed the model to synthesize speech that resembles the prosody of a reference utterance.

We used an audiobook dataset 2 derived from the same subset of LibriVox audiobooks used for the LibriSpeech corpus BID26 , but sampled at 24kHz and segmented differently, making it appropriate for TTS instead of speech recognition.

The corpus contains recordings from thousands of speakers, with wide variation in recording conditions and speaking style.

Speaker identity is often highly correlated with the recording channel and background noise level, since many speakers tended to use the same microphone in a consistent recording environment.

The ability to disentangle and control these attributes independently is essential to synthesizing high-quality speech for all speakers.

We augmented the model with the z o layer described in Section 2.3 to learn a continuous speaker representation and an inference model for it.

The train-clean-{100,360} partitions were used for training, which spans 1,172 unique speakers and, despite the name, includes many noisy recordings.

As in previous experiments, by traversing each dimension of z l we found that different latent dimensions independently control different attributes of the generated speech.

Moreover, this representation was disentangled from speaker identity, i.e. modifying z l did not affect the generated speaker identity if z o was fixed.

In addition, we discovered that the mean of one mixture component corresponded to a narrative speaking style in a clean recording condition.

Demonstrations of latent attribute control are shown in Appendix H and the demo page.

We demonstrate the ability of GMVAE-Tacotron to consistently generate high-quality speech by conditioning on a value of z l associated with clean output.

We considered two approaches: (1) using the mean of the identified clean component, which can be seen as a preset configuration with a fixed channel and style; (2) inferring a latent attribute representation z l from reference speech and denoising it by modifying dimensions 3 associated with the noise level to predetermined values.

We evaluated a set of eight "seen clean" (SC) speakers and a set of nine "seen noisy" (SN) speakers from the training set, a set of ten "unseen noisy" (UN) speakers from a held-out set with no overlapping speakers, and the set of ten unseen speakers used in , denoted as "unseen clean" (UC).

For consistency, we always used an inferred z o from an utterance from the target speaker, regardless of whether that speaker was seen or unseen.

As a baseline we used a Tacotron model conditioned on a 128-dimensional speaker embedding learned for each speaker seen during training.

TAB3 shows the SNR of the original audio, audio synthesized by the baseline, and by the GMVAETacotron using the two proposed approaches, denoted as mean and latent-dn, respectively, on all speaker sets whenever possible.

In addition, to see the effectiveness of the denoising operation, the table also includes the results of using inferred z l directly, denoted as latent.

The results show that the inferred z l followed the same SNR trend as the original audio, indicating that z l captured the variation in acoustic condition.

The high SNR values of mean and latent-dn verifies the effectiveness of using a preset and denoising arbitrary inferred latent features, both of which outperformed the baseline by a large margin, and produced better quality than the original noisy audio.

TAB4 compares the proposed model using denoised z l to the baseline in a subjective side-byside preference test.

TAB5 further compares subjective naturalness MOS of the proposed model using the mean of the clean component to the baseline on the two seen speaker sets, and to the d-vector model on the two unseen speaker sets.

Specifically, we consider another stronger baseline model to compare on the SN set, which is trained on denoised data using spectral subtraction BID7 , denoted as "+ denoise."

Both results indicate that raters preferred the proposed model to the baselines.

Moreover, the MOS evaluation shows that the proposed model delivered similar level of naturalness under all conditions, seen or unseen, clean or noisy.

We evaluate whether the synthesized speech resembles the identity of the reference speaker, by pairing each synthesized utterance with the reference utterance for subjective MOS evaluation of speaker similarity, following .

TAB6 compares the proposed model using denoised latent attribute representations to baseline systems on the two seen speaker sets, and to d-vector systems on the unseen clean speaker set.

The d-vector systems used a separately trained speaker encoder model to extract speaker representations for TTS conditioning as in .

We considered two speaker encoder models, one trained on the same train-clean partition as the proposed model, and another trained on a larger scale dataset containing 18K speakers.

We denote these two systems as d-vector and d-vector (large).On the seen clean speaker set, the proposed model achieved similar speaker similarity scores to the baseline.

However, on the seen noisy speaker set, both the proposed model and the baseline trained on denoised speech performed significantly worse than the baseline.

We hypothesize that similarity of the acoustic conditions between the paired utterances biased the speaker similarity ratings.

To confirm this hypothesis, we additionally evaluated speaker similarity of the ground truth utterances from a speaker whose recordings contained significant variation in acoustic conditions.

As shown in TAB6 , these ground truth utterances were also rated with a significantly lower MOS than the baseline, but were close to the proposed model and the denoised baseline.

This result implies that this subjective speaker similarity test may not be reliable in the presence of noise and channel variation, requiring additional work to design a speaker similarity test that is unbiased to such nuisance factors.

Finally, on the unseen clean speaker set, the proposed model achieved significantly better speaker similarity scores than the d-vector system whose speaker representation extractor was trained on the same set as the proposed model, but worse than the d-vector (large) system which was trained on over 15 times more speakers.

However, we emphasize that: (1) this is not a fair comparison as the two models are trained on datasets of different sizes, and (2) our proposed model is complementary to d-vector systems.

Incorporating the high quality speaker transfer from the d-vector model with the strong controllability of the GMVAE is a promising direction for future work.

We describe GMVAE-Tacotron, a TTS model which learns an interpretable and disentangled latent representation to enable fine-grained control of latent attributes and provides a systematic sampling scheme for them.

If speaker labels are available, we demonstrate an extension of the model that learns a continuous space that captures speaker attributes, along with an inference model which enables one-shot learning of speaker attributes from unseen reference utterances.

The proposed model was extensively evaluated on tasks spanning a wide range of signal variation.

We demonstrated that it can independently control many latent attributes, and is able to cluster them without supervision.

In particular, we verified using both subjective and objective tests that the model could synthesize high-quality clean speech for a target speaker even if the quality of data for that speaker does not meet high standard.

These experimental results demonstrated the effectiveness of the model for training high-quality controllable TTS systems on large scale training data with rich styles by learning to factorize and independently control latent attributes underlying the speech signal.

This section gives detailed derivation of the evidence lower bound (ELBO) estimation used for training.

We first present a differentiable Monte Carlo estimation of the posterior q(y l | X), and then derive an ELBO for each of the graphical models in FIG0 , which differ in whether an additional observed attribute representation z o is used.

As shown in equation 2, we approximate the posterior over latent attribute class y l with DISPLAYFORM0 where q(z l | X) is a diagonal-covariance Gaussian, and p(y l | z l ) is the probability of z l being drawn from the y l -th Gaussian mixture component.

We first denote the mean vector and the diagonal elements of the covariance matrix of the y l -th component as ?? l,y l and ?? 2 l,y l , and write the posterior over mixture components given a latent attribute representation, p(y l | z l ): DISPLAYFORM1 with DISPLAYFORM2 where D is the dimensionality of z l , and K is the number of classes for y l .Finally, we denote the posterior mean and variance of q(z l | X) by?? l and?? 2 l , and compute a Monte Carlo estimate of the expectation in equation 5 after reparameterization: DISPLAYFORM3 DISPLAYFORM4 :=q(y l | X), DISPLAYFORM5 is a random sample, drawn from a standard Gaussian distribution using (n) ??? N (0, I), and N is the number of samples used for the Monte Carlo estimation.

The resulting estimateq(y l | X) is differentiable w.r.t.

the parameters of p(z l | y l ) and q(z l | X).

We next derive the ELBO L(p, q; X, Y t , y o ) and rewrite it as a Monte Carlo estimate used for training: DISPLAYFORM0 DISPLAYFORM1 : DISPLAYFORM2 DISPLAYFORM3 is the estimator used for training.

Similarly, N is the number of samples used for the Monte Carlo estimate.

In this section, we derive the ELBO L o (p, q; X, Y t , y o ) when using an additional observed attribute representation, z o , as described in Section 2.3, and rewrite it with a Monte Carlo estimation used for training.

As before, we denote the posterior mean and variance of q( DISPLAYFORM0 : DISPLAYFORM1 where the continuous latent variables are reparameterized asz DISPLAYFORM2 , with auxiliary noise variables DISPLAYFORM3 is used for training.

N and N are the numbers of samples used for the Monte Carlo estimate.

The synthesizer is an attention-based sequence-to-sequence network which generates a mel spectrogram as a function of an input text sequence and conditioning signal generated by the auxiliary encoder networks.

It closely follows the network architecture of Tacotron 2 .

The input text sequence is encoded by three convolutional layers, which contains 512 filters with shape 5 ?? 1, followed by a bidirectional long short-term memory (LSTM) of 256 units for each direction.

The resulting text encodings are accessed by the decoder through a location sensitive attention mechanism (Chorowski et al., 2015) , which takes attention history into account when computing a normalized weight vector for aggregation.

The base Tacotron 2 autoregressive decoder network takes as input the attention-aggregated text encoding, and the bottlenecked previous frame (processed by a pre-net comprised of two fullyconnected layers of 256 units) at each step.

In this work, to condition the output on additional attribute representations, the decoder is extended to consume z l and z o (or y o ) by concatenating them with the original decoder input at each step.

The concatenated vector forms the new decoder input, which is passed through a stack of two uni-directional LSTM layers with 1024 units.

The output from the stacked LSTM is concatenated with the new decoder input (as a residual connection), and linearly projected to predict the mel spectrum of the current frame, as well as an end-of-sentence token.

Finally, the predicted spectrogram frames are passed to a post-net, which predicts a residual that is added to the initial decoded sequence of spectrogram frames, to better model detail in the spectrogram and reduce the overall mean squared error.

Similar to Tacotron 2, we separately train a neural vocoder to invert a mel spectrograms to a timedomain waveform.

In contrast to that work, we replace the WaveNet (van den BID35 vocoder with one based on the recently proposed WaveRNN BID16 architecture, which is more efficient during inference.

Both the latent encoder and the observed encoder map a mel spectrogram from a reference speech utterance to two vectors of the same dimension, representing the posterior mean and log variance of the corresponding latent variable.

We design both encoders to have exactly the same architecture, whose outputs are conditioned by the decoder in a symmetric way.

Disentangling of latent attributes and observed attributes is therefore achieved by optimizing different KL-divergence objectives.

For each encoder, a mel spectrogram is first passed through two convolutional layers, which contains 512 filters with shape 3 ?? 1.

The output of these convolutional layers is then fed to a stack of two bidirectional LSTM layers with 256 cells at each direction.

A mean pooling layer is used to summarize the LSTM outputs across time, followed by a linear projection layer to predict the posterior mean and log variance.

The network is trained using the Adam optimizer BID21 , configured with an initial learning rate 10 ???3 , and an exponential decay that halved the learning rate every 12.5k steps, beginning after 50k steps.

Parameters of the network are initialized using Xavier initialization (Glorot & Bengio, 2010) .

A batch size of 256 is used for all experiments.

Following the common practice in the VAE literature BID22 , we set the number of samples used for the Monte Carlo estimate to 1, since we train the model with a large batch size.

TAB7 details the list of prior hyperparameters used for each of the four datasets described in Section 4: multi-speaker English data (multi-spk), noisified multi-speaker English data (noisy-multispk), single-speaker story-telling data (audiobooks), and crowd-sourced audiobook data (crowdsourced).

To ensure numerical stability we set a minimum value allowed for the standard deviation of the conditional distribution p(z l | y l ).

We initially set the lower bound to e ???1 ; however, with the exception of the multi-speaker English data, the trained standard deviation reached the lower bound for all mixture components for all dimensions.

We therefore lowered the minimum standard deviation to e ???2 , and found that it left sufficient range to capture the amount of variation.

As shown in TAB1 , we found that increasing the dimensionality of z l from 16 to 32 improves reconstruction quality; however, it also increases the difficulty of interpreting each dimension.

On the other hand, reducing the dimensionality too much can result in insufficient modeling capacity for latent attributes, however we have not carefully explored this lower bound.

Empirically, we found 16-dimensional z l to be appropriate for capturing the salient attributes one would like to control in the four datasets we experimented with.

When evaluating the meaning of each dimensions, we find the majority of the dimensions to be interpretable, and the number of dummy dimensions which do not affect the model output varied across datasets, as each of them inherently has variation across a different number of unlabeled attributes.

For example, the model trained on the multi-speaker English corpus (Section 4.1) has four dummy dimensions of z l that do not affect the output.

In contrast, for the model trained on the crowd-sourced audio book corpus (Section 4.4), which contains considerably more variation in style and prosody, we found only one dummy dimension of z l .

DISPLAYFORM0

There are two latent variables in our graphical model as shown in FIG0 (left): the latent attribute class y l (discrete) and latent attribute representation z l (continuous).

We discuss the potential for posterior collapse for each of them separately.

The continuous latent variable z l is used to directly condition the generation of X, along with two other observed variables, y o and Y t .

In our experiments, we observed that the latent variable z l is always used, i.e. the KL-divergence of z l never drops to zero, without applying any tricks such as KL-annealing (Bowman et al., 2016) .Previous studies report posterior-collapse of directly conditioned latent variables when using strong models (e.g. auto-regressive networks) to parameterize the conditional distribution of text (Bowman et al., 2016; BID19 .

This phenomenon arises from the competition between (1) increasing reconstruction performance by utilizing information provided by the latent variable, and (2) decreasing the KL-divergence by making the latent variable uninformative.

Auto-regressive models are more likely to converge to the second case during training because the improvement in reconstruction from utilizing the latent variable can be smaller than the increase in KL-divergence.

However, this does not always happen, because the amount of improvement resulted from utilizing the information provided by the latent variable depends on the type of data.

The reason that the posterior-collapse does not occur in our experiments is likely a consequence of the complexity of the speech sequence distribution, compared to text.

Even though we use an auto-regressive decoder, reconstruction performance can still be improved significantly by utilizing the information from z l , and such improvement overpowers the increase in KL-divergence.

The discrete latent variable y l indexes the mixture components in the space of latent attribute representation z l .

We did not observe the phenomenon of degenerate clusters mentioned in Dilokthanakul et al. FORMULA0 when training our model using the hyperparameters listed in TAB7 .

Below we identify the difference between our GMVAE and that in Dilokthanakul et al. FORMULA0 , which we will refer to as Dil-GMVAE, and explain why our formulation is less likely to suffer from similar posterior collapse issues.

In our model, the conditional distribution DISPLAYFORM0 Gaussian, parameterized by a mean and a covariance vector.

In contrast, in Dil-GMVAE, the conditional distribution of z l given y l is much more flexible, because it is parameterized using neural networks as: DISPLAYFORM1 where f ?? and f ?? 2 are neural networks that take y l and an auxiliary noise variable as input to predict the mean and variance of p(z l | y l , ), respectively.

The conditional distribution of each component in Dil-GMVAE can be seen as a mixture of infinitely many diagonal-covariance Gaussian distributions, which can model much more complex distributions, as shown in Dilokthanakul et al. (2016, Figure 2(d) ).

Compared with the GMVAE described in this paper, Dil-GMVAE can be regarded as having a much stronger stochastic decoder that maps y l to z l .Suppose the conditional distribution that maps z l to X can benefit from having a very complex, non-Gaussian marginal distribution over z l , denoted as p * (z l ).

To obtain such a marginal distribution of z l in Dil-GMVAE, the stochastic decoder that maps y l to z l can choose between (1) using the same p(z l | y l ) = p * (z l ) for all y l , or (2) having p(z l | y l ) model a different distribution for each y l , and DISPLAYFORM2 As noted in Dilokthanakul et al. (2016) , the KL-divergence term for y l in the ELBO prefers the former case of degenerate clusters that all model the same distribution.

As a result, the first option would be preferred with respect to the ELBO objective compared to the second one, because it does not compromise the expressiveness of z l while minimizing the KL-divergence on y l .

In contrast, our GMVAE formulation reduces to a single Gaussian when p(z l | y l ) is the same for all y l , and hence there is a trade-off between the expressiveness of p(z l ) and the KL-divergence on y l .In addition, we now explain the connection between posterior-collapse and hyperparameters of the conditional distribution p(z l | y l ) in our work.

In our GMVAE model, posterior-collapse of y l is equivalent to having the same conditional mean and variance for each mixture component.

In the ELBO derived from our model, there are two terms that are relevant to p(z l |y l ), which are (1) the expected KL-divergence on z l : E q(y l |X) [D KL (q(z l |X)||p(z l |y l ))] and (2) the KL-divergence on y l : D KL (q(y l |X)||p(y l )).

The second term encourages a uniform posterior q(y l |X), which effectively pulls the conditional distribution for each component to be close to each other, and promotes posterior collapse.

In contrast, the first term pulls each p(z l |y l ) to be close to q(z l |X) with a force proportional to the posterior of that component, q(y l |X).In one extreme, where the posterior q(y l |X) is close to uniform, each p(z l |y l ) is also pushed toward the same distribution, q(z l |X), which promotes posterior collapse.

In the other extreme, where the posterior q(y l |X) is close to one-hot with q(y l = k|X) ??? 1, only the conditional distribution of the assigned component p(z l |y l = k) is pushed toward q(z l |X).

As long as different X are assigned to different components, this term is anti-collapse.

Therefore, we can see that the effect of the first term on posterior-collapse depends on the entropy of q(y l |X), which is controlled by the scale of the variance when the means are not collapsed.

This variance is similar to the temperature parameter used in softmax: the smaller the variance is, the more spiky the posterior distribution over y is.

This is why we set the initial variance of each component to a smaller value at the beginning of training, which helps avoid posterior collapse

To quantify the difference between the three components that model US female speakers (3, 5, and 6), we draw 20 latent attribute encodings z l from each of the three components and decode the same set of 25 text sequences for each one.

TAB8 shows the average F 0 computed over 500 synthesized utterances for each component, demonstrating that each component models a different F 0 range. .

All examples use the same input text: "The fake lawyer from New Orleans is caught again."

The plots for dimension 0 (top row) and dimension 2 (second row) mainly show variation along the time axis.

The underlying F 0 contour values do not change, however dimension 0 controls the duration of the initial pause before the speech begins, and dimension 2 controls the overall speaking rate, with the F 0 track stretching in time (i.e. slowing down) when moving from the left column to the right.

Dimension nine (bottom row) mainly controls the degree of F 0 variation while maintaining the speed and starting offset.

Finally, we note that differences in accent controlled by dimension 3 (third row) are easier to recognize by listening to audio samples, which can be found at https://google.github.io/tacotron/publications/gmvae_ controllable_tts#multispk_en.control.

To quantify how well the learned representation captures useful speaker information, we experimented with training classifiers for speaker attributes on the latent features.

The test utterances were partitioned in a 9:1 ratio for training and evaluation, which contain 2,098 and 234 utterances, respectively.

Three linear discriminant analysis (LDA) classifiers were trained on the latent attribute representations z l to predict speaker identity, gender and accent.

To objectively evaluate the ability of the proposed GMVAE-Tacotron model to control individual attributes, we compute two metrics: (1) the average F 0 (fundamental frequency) in voiced frames, computed using YIN (De Cheveign?? & Kawahara, 2002) , and (2) the average speech duration.

These correspond to rough measures of the speaking rate and degree of pitch variation, respectively.

We randomly draw 10 samples of seed z l from the prior, deterministically set the target dimension DISPLAYFORM0 are the mean and standard deviation of the marginal distribution of the target dimension d. We then synthesize a set of the same 25 text sequences for each of the 30 resulting values of z * l .

For each value of the target dimension, we compute an average metric over 250 synthesized utterances (10 seed z l ?? 25 text inputs).

TAB1 .

For the "pitch" dimension, we can see that the measured F 0 varies substantially, while the speech remains constant.

Similarly, as the value in the "speaking rate" dimension varies, the measured duration varies substantially.

However, there is also a smaller inverse effect on the pitch, i.e. the pitch slightly increases with the speaking rate, which is consistent with natural speaking behavior BID1 BID6 .

These results are an indication that manipulating individual dimensions primarily controls the corresponding attribute.

We evaluated the ability of the proposed model to synthesize speech that resembled the prosody or style of a given reference utterance, by conditioning on a latent attribute representation inferred from the reference.

We adopted two metrics from to quantify style transfer performance: the mel-cepstral distortion(MCD 13 ), measuring the phonetic and timbral distortion, and F 0 frame error (FFE), which combines voicing decision error and F 0 error metrics to capture how well F 0 information, which encompasses much of the prosodic content, is retained.

Both metrics assume that the generated speech and the reference speech are frame aligned.

We therefore synthesized the same text content as the reference for this evaluation.

TAB1 compares the proposed model against the baseline and a 16-token GST model.

The proposed model with a 16-dimensional z l (D = 16) was better than the baseline but inferior to the GST model.

Because the GST model uses a four-head attention BID36 , it effectively has 60 degrees of freedom, which might explain why it performs better in replicating the reference style.

By increasing the dimension of z l to 32 (D = 32), the gap to the GST model is greatly reduced.

Note that the number of degrees of freedom of the latent space as well as the total number of parameters is still smaller than in the GST model.

As noted in , the ability of a model to reconstruct the reference speech frame-by-frame is highly correlated with the latent space capacity, and we can expect our proposed model to outperform GST on these two metrics if we further increase the dimensionality of z l ; however, this would likely reduce interpretability and generalization of the latent attribute control.

"By water in the midst of water!" "

And she began fancying the sort of thing that would happen:

Miss Alice!"

"She tasted a bite, and she read a word or two, and she sipped the amber wine and wiggled her toes in the silk stockings."Reference style 1 Reference style 2 Reference style 3Figure 13: Mel-spectrograms of reference and synthesized style transfer utterances.

The four reference utterances are shown on the top, and the four synthesized style transfer samples are shown below, where each row uses the same input text (shown above the spectrograms), and each column is conditioned on the z l inferred from the reference in the top row.

From left to right, the voices of the three reference utterances can be described as (1) tremulous and high-pitched, (2) rough, low-pitched, and terrifying, and (3) deep and masculine.

In all cases, the synthesized samples resemble the prosody and the speaking style of the reference.

For example, samples in the first column have the highest F 0 (positively correlated to the spacing between horizontal stripes) and more tremulous (vertical fluctuations), and spectrograms in the middle column are more blurred, related to roughness of a voice.

Audio samples can be found at https://google.github.io/tacotron/ publications/gmvae_controllable_tts#singlespk_audiobook.transfer FIG0 demonstrates that the GMVAE-Tacotron can also be applied in a non-parallel style transfer scenario to generate speech whose text content differs significantly from the reference.

(1) "We must burn the house down! said the Rabbit's voice.

", (2) "

And she began fancying the sort of thing that would happen: Miss Alice!", and (3) "She tasted a bite, and she read a word or two, and she sipped the amber wine and wiggled her toes in the silk stockings.

"

The five samples of z l encode different styles: the first sample has the fastest speaking rate, the third sample has the slowest speaking rate, and the fourth sample has the highest F 0 .

Audio samples can be found at https://google.github.io/ tacotron/publications/gmvae_controllable_tts#singlespk_audiobook.sample

Dimension 10: pause length Dimension 14: roughness FIG0 : Synthesized mel-spectrograms demonstrating independent control of speaking style and prosody.

The same input text is used for all samples: "He waited a little, in the vain hope that she would relent: she turned away from him."

In the top row, F 0 is controlled by setting different values for dimension eight.

F 0 tracks show that the F 0 range increases from left to right, while other attributes such as speed and rhythm do not change.

In the second row, the duration of pause before the phrase "she turned away from him." (red boxes) is varied.

The three spectrograms are very similar, except for the width of the red boxes, indicating that only the pause duration changed.

In the bottom row, the "roughness" of the voice is varied.

The same region of spectrograms is zoomed-in for clarity, where the spectrograms became less blurry and the harmonics becomes better defined from left to right.

FIG0 : Synthesized mel-spectrograms and F 0 tracks demonstrating independent control of attributes related to style, recording channel, and noise-condition.

The same text input was used for all the samples: '"Are you Italian?" asked Uncle John, regarding the young man critically.'

In each row we varied the value for a single dimension while holding other dimensions fixed.

In the top row, we controlled the F 0 by traversing dimension zero.

Note that the speaker identity did not change while traversing this dimension.

In the second row, the F 0 contours did change while traversing this dimension; however, it can be seen from the spectrograms that the leftmost one attenuated the energy in low-frequency bands, and the rightmost one attenuated energy in high-frequency bands.

This dimension appears to control the shape of a linear filter applied to the signal, perhaps corresponding to variation in microphone frequency response in the training data.

In the third row, the F 0 contours did not change, either.

However, the background noise level does vary while traversing this dimension, which can be heard on the demo page.

In the bottom row, variation in the speaking rate can be seen while other attributes remain constant.

Audio samples can be found at https://google.github.io/tacotron/publications/gmvae_ controllable_tts#crowdsourced_audiobook.control.

@highlight

Building a TTS model with Gaussian Mixture VAEs enables fine-grained control of speaking style, noise condition, and more.

@highlight

Describes the conditioned GAN model to generate speaker conditioned Mel spectra by augmenting the z-space corresponding to the identification

@highlight

This paper proposes a two layer latent variable model to obtain disentangled latent representation, thus facilitating fine-grained control over various attributes

@highlight

This paper proposes a model that can control non-annotated attributes such as speaking style, accent, background noise, etc.