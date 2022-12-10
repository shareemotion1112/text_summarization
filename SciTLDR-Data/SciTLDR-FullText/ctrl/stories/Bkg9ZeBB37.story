To leverage crowd-sourced data to train multi-speaker text-to-speech (TTS) models that can synthesize clean speech for all speakers, it is essential to learn disentangled representations which can independently control the speaker identity and background noise in generated signals.

However, learning such representations can be challenging, due to the lack of labels describing the recording conditions of each training example, and the fact that  speakers and recording conditions are often correlated, e.g. since users often make many recordings using the same equipment.

This paper proposes three components to address this problem by: (1) formulating a conditional generative model with factorized latent variables, (2) using data augmentation to add noise that is not correlated with speaker identity and whose label is known during training, and (3) using adversarial factorization to improve disentanglement.

Experimental results demonstrate that the proposed method can disentangle speaker and noise attributes even if they are correlated in the training data, and can be used to consistently synthesize clean speech for all speakers.

Ablation studies verify the importance of each proposed component.

Recent development of neural end-to-end TTS models BID26 BID1 enables control of both labelled and unlabelled speech attributes by conditioning synthesis on both text and learned attribute representations BID27 BID21 BID10 BID0 BID5 BID9 .

This opens the door to leveraging crowd-sourced speech recorded under various acoustic conditions BID18 to train a high-quality multi-speaker TTS model that is capable of consistently producing clean speech.

To achieve this, it is essential to learn disentangled representations that control speaker and acoustic conditions independently.

However, this can be challenging for two reasons.

First, the underlying acoustic conditions of an utterance, such as the type and level of background noise and reverberation, are difficult to annotate, and therefore such labels are often unavailable.

This hinders the use of direct conditioning on the acoustic condition labels in a way similar to conditioning on one-hot speaker labels BID1 .

Second, speaker identity can have strong correlations with recording conditions, since a speaker might make most of their recordings in the same location using the same device.

This makes it difficult to learn a disentangled representation by assuming statistical independence BID6 .We address this scenario by introducing three components: a conditional generative model with factorized latent variables to control different attributes, data augmentation by adding background noise to training utterances in order to counteract the inherent speaker-noise correlation and to create ground truth noisy acoustic condition labels, and adversarial training based on the generated labels to encourage disentanglement between latent variables.

We utilize the VCTK speech synthesis dataset BID23 , and background noise signals from the CHiME-4 challenge BID24 to synthesize a dataset containing correlated speaker and background noise conditions for controlled experiments.

We extensively evaluate disentanglement performance on the learned latent representations as well as the synthesized samples.

Experimental results identify the contribution of each component, and demonstrate the ability of the proposed model to disentangle noise from speakers and consistently synthesize clean speech for all speakers, despite the strong correlation in the training data.

We base our TTS model on Tacotron 2 BID20 , which takes a text sequence as input, and outputs a sequence of mel spectrogram frames.

To control speech attributes other than text, two additional latent variables, z s and z r , are introduced to condition the generative process, where the former models speaker identity, and the latter models residual unlabelled attributes (e.g. acoustic conditions).

Prior distributions for both variables are defined to be isotropic Gaussian.

The full TTS model can be written as a conditional generative model with two latent variables: p(speech | z s , z r , text).Two variational distributions are introduced: q(z s | speech) and q(z r | speech), to approximate the intractable posteriors of the latent variables, following the variational autoencoder (VAE) framework BID14 .

Each distribution is defined to be diagonal-covariance Gaussian, whose mean and variance are parameterized by a neural network encoder.

Note that z s , z r , and text are assumed to be conditionally independent given speech, in order to simplify inference.

In contrast to learning an embedding for each speaker, learning an inference model for z s can be used to infer speaker attributes for previously unseen speakers.

To factorize speaker and residual information, an auxiliary speaker classifier that takes z s as input is trained jointly with the TTS model.

This encourages information that is discriminative between speakers to be encoded in z s , and leaves residual information to z r .

A simple fully-connected network is used for the speaker classifier.

When acoustic conditions are correlated with speakers, information about e.g. background noise level can be used to discriminate between speakers, and therefore can be encoded into z s .

To counteract such behavior, one can decorrelate these factors by leveraging prior knowledge that adding noise should not affect speaker identity.

We propose to augment the original training set with a noisy copy that mixes each utterance with a randomly selected piece of background noise at a randomly sampled signal-to-noise ratio (SNR), but reuses the same transcript and speaker label as the original utterance.

This operation can be seen as flattening the SNR distribution of each speaker, in order to make SNRs less discriminative about speakers.

To increase the degree of disentanglement, it is also useful to proactively discourage z s from encoding acoustic condition information.

If the ground truth acoustic condition labels are available, domain adversarial training BID3 can be applied directly to encourage z s not to be informative about the acoustic condition.

Nevertheless, such labels are often unavailable in crowdsourced datasets such as BID18 .In order to utilize adversarial training in such a scenario, we propose to use the augmentation label (original/augmented) to replace the acoustic condition label (clean/noisy).

This augmentation label can be seen as a noisy acoustic condition label: an augmented utterance must be noisy, but an original one can be either.

If z s is invariant to acoustic conditions, then it is also invariant to augmentation labels, implying that the latter is a necessary condition for the former.

Following BID3 , invariance of z s to augmentation is measured using the empirical H-divergence between the z s distribution of the augmented data and that of the original data, given a hypothesis class H that is a set of binary classifiers.

The empirical H-divergence measures how well the best classifier in the hypothesis class can distinguish between samples drawn from different distributions.

However, it is generally hard to compute the empirical H-divergence.

Following BID2 BID3 , we approximate it with the Proxy A-distance: 2(1 − 2 ), where is a generalization error of an augmentation classifier trained to predict if z s is inferred from an augmented utterance.

A simple fully-connected network is used for the augmentation classifier.

The complete model is illustrated in FIG0 , composed of three modules: a synthesizer, p(speech | z s , z r , text), an inference network with two encoders, q(z s | speech) and q(z r | speech), and an adversarial factorization module with speaker and augmentation classifiers, p(y s | z s ) and p(y a | z r ), where y s and y a denotes speaker and augmentation labels.

The parameters of the synthesizer, the two encoders, the speaker classifier, and the augmentation classifiers are accordingly denoted as θ, φ s , φ r , ψ s , and ψ a , respectively.

Training of the proposed model aims to maximize the conditional likelihood and the information z s contains about speakers, while minimizing the H-divergence between the z s inferred from the original utterances and that from the augmented ones.

The H-divergence is approximated with the Proxy A-distance obtained from the augmentation classifier.

The objective function can be formulated as combining an evidence lower bound (ELBO) with a domain adversarial training BID3 objective: DISPLAYFORM0 where λ 1 , λ 2 > 0 are the loss weights for the two classifiers, and ELBO(θ, φ s , φ r ; speech, text) is formulated as: DISPLAYFORM1 Note that the augmentation classifier is optimized with a different objective than the rest of the model.

To train the entire model jointly, a gradient reversal layer BID3 is inserted after the input to the augmentation classifier, which scales the gradient by −λ 2 .

Our formulation of a TTS model with latent variables are closely related to BID27 BID21 BID0 BID5 BID9 , which focus on modeling unlabeled speech attributes.

In contrast to this work, BID27 BID21 BID0 BID5 do not address disentangling attributes to enable independent control when different attributes are highly correlated in the training data, while BID9 learns to disentangle speaker attributes from the rest by encoding those with small within-speaker variance to z s .The proposed augmentation-adversarial training combines data augmentation for speech BID11 with domain adversarial neural networks (DANNs) BID3 for disentangling correlated attributes.

These two methods have been mainly applied for training robust discriminative models BID7 BID22 BID24 BID19 , and are less studied in the context of building generative models.

In addition, our method provides two advantages.

First, while DANNs require domain labels, our proposed method enables adversarial training even when the ground truth domain labels are unavailable.

Second, domain adversarial training aims to remove domain information while preserving target attribute information; however, if domain and target attribute have very strong correlations, the two objectives conflict with each other, and one of the them will be compromised.

Our proposed method alleviates such issues by using data augmentation to decorrelate the two factors.

Learning disentangled representations for deep generative models has gain much interest recently BID8 BID28 .

Several studies also explored adversarial training for disentanglement, such as using maximum mean discrepancy BID15 and generative adversarial network BID17 .

We particularly emphasize disentangling statistically correlated attributes, and apply H-divergence based adversarial training on latent variables.

We artificially generated a noisy speech dataset with correlated speaker and noise conditions using the VCTK corpus BID23 and background noise from the CHiME-4 challenge BID24 .

The motivation here is to simulate real noisy data while evaluating the model under carefully controlled conditions.

VCTK contains 44 hours of clean English speech from 109 speakers.

We downsample the signals to 16 kHz to match the background noise sample rate, and split it into training and testing sets in a 9:1 ratio.

The CHiME-4 corpus contains 8.5 hours of background noise recorded in four different locations (bus, cafe, pedestrian area, and street junction), which we split into three partitions: train, test, and aug.

To simulate speaker-correlated noise, we randomly selected half the speakers to be noisy, and mixed all of their train and test utterances with noise sampled from train and test respectively, at SNRs ranging from 5 -25 dB. As described in Section 2.2, we generated an augmented set by mixing every (potentially noisy) training utterance with a noise signal sampled from aug at SNRs ranging from 5 -25 dB. Utterances in the augmented set are annotated with y a = 1, and those in the original noisy training set are annotated with y a = 0.

We strongly encourage readers to listen to the samples on the demo page.

The synthesizer network use the sequence-to-sequence Tacotron 2 architecture BID20 , with extra input z s and z r concatenated and passed to the decoder at each step.

If not otherwise mentioned, z s is 64-dim and z r is 8-dim.

The generated speech is represented as a sequence of 80-dim mel-scale filterbank frames, computed from 50ms windows shifted by 12.5ms.

We represent input text as a sequence of phonemes, since learning pronunciations from text is not our focus.

The speaker and the residual encoders both use the same architecture which closely follow the attribute encoder in BID9 .

Each encoder maps a variable length mel spectrogram to two vectors parameterizing the mean and log variance of the Gaussian posterior.

Both classifiers are fully-connected networks with one 256 unit hidden layer followed by a softmax layer to predict the speaker or augmentation posterior.

The synthesizer, encoders, and speaker classifier are trained to maximize Eq (1) with λ 1 = λ 2 = 1, and the augmentation classifier is trained to maximize Eq (2).

The entire model is trained jointly with a batch size of 256, using the Adam optimizer BID13 , configured with an initial learning rate of 10 −3 , and an exponential decay that halves the learning rate every 12.5k steps, starting at 50k steps.

We quantify the degree of disentanglement by training speaker and noise classifiers on z s and z r separately.

The classification accuracy on a held-out set is used to measure how much information a latent variable contains about the prediction targets.

A simple linear discriminative analysis classifier is used for all four tasks.

If the classifier input contains no information about the target, the best a classifier can do is to predict the highest prior probability class.

Since the distributions of both speaker and acoustic conditions are close to uniform, a speaker-uninformative input should result in about 1% accuracy, and a noise-uninformative input should result in about 50%.Results are shown in TAB1 , comparing the full proposed model with two alternative models: one which removes adversarial training, denoted as "-adv,", and a second which further removes data augmentation, denoted as "-adv -aug." Without data augmentation and adversarial training, the second alternative completely fails to disentangle speaker from noise, i.e. its speaker encoding z s can infer both, while its residual encoding z s cannot infer either.

The first alternative learns to encode acoustic condition into z r , reaching 96.5% accuracy on noise prediction; however, part of such information still leaks to z s , as indicated by the 85% noise prediction accuracy.

The full proposed model achieves the highest noise prediction accuracy using z r , and the lowest accuracy using z s , implying the best allocation of acoustic information.

Nevertheless, adversarial training also results in slight degradation of speaker information allocation, where the speaker prediction accuracy using z r increases from 1.4% to 2.3%.

We further analyze the latent space of the proposed model by visualizing the learned speaker and residual representations using t-SNE BID16 , which is a technique for projecting high-dimensional vectors to a two-dimensional space.

Results are shown in Figure 2 , where each point corresponding to a projected z r (left column) or z s (right column) inferred from a single utterance.

Points are color-coded according to speaker, gender, and accent labels in each row.

In the left column, projected z r are clearly separated by acoustic condition, but not by gender or speaker.

In contrast, projected z s shown in the right column forms many small clusters, with one speaker each cluster; Moreover, as shown in the middle row, clusters of speakers are further separated according to their genders.

In the lower right panel, projected z s of noisy utterances and clean utterances are overlaid, demonstrating that z s have similar distributions conditioning on different acoustic conditions.

To evaluate how well the two latent variables, z s and z r , can control the synthesized speech, we sample five clean speakers and five noisy speakers, and select one testing utterance for each speaker with duration ≥ 3s.

For each of the ten utterances, the two latent variables are inferred using the corresponding encoders.

We construct an evaluation set of 100 phrases that does not overlap with the VCTK corpus, and synthesize them conditioned on each combination of z r and z s , including those inferred from different utterances.

The total 10,000 synthesized samples are divided into four groups, depending on the set of speakers (clean/noisy) z r and z s are inferred from.

To quantify the ability to control noise, we use waveform amplitude distribution analysis (WADA) BID12 to estimate an SNR without a clean reference signal.

We compare to a baseline multi-speaker Tacotron model, which removes the residual encoder and replaces the speaker encoder with a lookup table of 64-D speaker embeddings.

The upper half of TAB2 presents the estimated SNRs of synthesized speech using this baseline, conditioning on the same five clean speakers and the five noisy speakers mentioned above.

The difference in SNR between clean and noisy speakers indicates that the acoustic condition is tied to speaker identity in this baseline model.

Results of the proposed model and the two alternatives mentioned in Section 4.2 are shown in the lower half of TAB2 .

By conditioning on z r inferred from clean utterances, the proposed model is able to synthesize clean speech even for noisy speakers whose training utterances all had background noise.

Moreover, when conditioning on the same set of z r , the proposed achieves the smallest discrepancy in SNR between different z s sets.

On the other hand, the "-adv" variant has a larger discrepancy between different z s sets, indicating worse disentanglement comparing to the full model, while the "-adv-aug" variant fails to control noise through z r .

These results are in line with the noise prediction results using z s and z r shown in TAB1 .

Figure 3 illustrates synthesized samples for a noisy speaker, comparing the baseline to our proposed model.

Our model is capable of controlling noise using z r , and can generate clean speech for the noisy speaker, while the baseline output always contains background noise.

We next examine if z s can control the speaker identity of synthesized speech, using a text-independent speaker verification system BID25 to is compute speaker discriminative embeddings, called d-vectors BID4 , from the reference and synthesized speech samples.

The system is trained to optimize a generalized end-to-end speaker verification loss, so that the embeddings of two utterances are close to each other if they are from the same speaker, and far way if from different speakers.

We build a nearest-neighbor classifier, which assigns an input signal the speaker label of the reference signal whose d-vector is closest to that of the input, measured using Euclidean distance.

To prevent background noise from affecting d-vector quality, we only evaluate synthesized samples conditioned on z r from clean utterances.

TAB3 shows that the synthesized samples closely resemble the speaker characteristics of their corresponding reference samples, regardless of z r used for conditioning.

The results indicate that speaker identity is controlled by z s , while being invariant to change in z r .

To quantify fidelity, we rely on crowd-sourced mean opinion scores (MOS), which rates the naturalness of the synthesized samples by natives speakers using headphones, with scores ranging from 1 to 5 in 0.5 increments.

To quantify fidelity, we rely on crowd-sourced mean opinion scores (MOS), which rates the naturalness of the synthesized samples by natives speakers using headphones, with scores ranging from 1 to 5 in 0.5 increments.

Results shown in TAB4 compares the baseline and the proposed model conditioning on z r from clean utterances.

When conditioning on z r from clean utterances, the proposed model achieves a higher MOS score than the baseline.

In contrast, the MOS drops significantly when conditioning on z r inferred from noisy utterances.

The results indicate that disentangling speaker and noise improves the naturalness of the generated speech, and the proposed model can synthesize more natural speech with less background noise than the baseline when conditioning on z r inferred from clean signals.

Finally, we study the sensitivity of disentanglement performance with respect to the choice of speaker encoding dimensions.

As shown in the previous two sections, good latent space disentanglement translates to good performance in terms of control of speaker identity and acoustic conditions for synthesis.

In this section, we only evaluate latent space disentanglement when changing the dimension of z s TAB5 compares performance of the proposed model when the dimensionality of z s is 32, 64, 128, and 256.

Variants without data augmentation or adversarial training fail to disentangle in all configurations.

When the dimension of z s increases, both the proposed model and "-adv" report worse separation of information, as indicated by increased noise prediction accuracy using z s .

Specifically, the "-adv" fails to encode noise information in z r when z s has 128 dimensions, which could result from a bad initialization of model parameters; however, such a behavior also indicates that when adversarial training is not applied, the disentanglement performance may rely heavily on the model initialization.

On the other hand, the proposed model is least sensitive to the change of z s dimensionality.

It always achieves the highest noise prediction accuracy using z r , and the lowest noise prediction accuracy using z s .

We build a neural network TTS model which incorporates conditional generative modeling, data augmentation, and adversarial training to learn disentangled representations of correlated and partially unlabeled attributes, which can be used to independently control different aspects of the synthesized speech.

Extensive studies on a synthetic dataset verify the effectiveness of each element of the proposed solution, and demonstrate the robustness to the choice of hyperparameters.

The proposed methods for disentangling correlated attributes is general, and can potentially be applied to other pairs of correlated factors, such as reverberation and speaker, or to other modalities, such as controllable text-to-image generation.

In addition, for future work, we would also like to investigate the capability of the proposed method to disentangle pairs of attributes which are both unsupervised.6 Acknowledgement

<|TLDR|>

@highlight

Data augmentation and adversarial training are very effective for disentangling correlated speaker and noise, enabling independent control of each attribute for text-to-speech synthesis.