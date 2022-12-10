We present a meta-learning approach for adaptive text-to-speech (TTS) with few data.

During training, we learn a multi-speaker model using a shared conditional WaveNet core and independent learned embeddings for each speaker.

The aim of training is not to produce a neural network with fixed weights, which is then deployed as a TTS system.

Instead, the aim is to produce a network that requires few data at deployment time to rapidly adapt to new speakers.

We introduce and benchmark three strategies: (i) learning the speaker embedding while keeping the WaveNet core fixed, (ii) fine-tuning the entire architecture with stochastic gradient descent, and (iii) predicting the speaker embedding with a trained neural network encoder.

The experiments show that these approaches are successful at adapting the multi-speaker neural network to new speakers, obtaining state-of-the-art results in both sample naturalness and voice similarity with merely a few minutes of audio data from new speakers.

Training a large model with lots of data and subsequently deploying this model to carry out classification or regression is an important and common methodology in machine learning.

It has been particularly successful in speech recognition , machine translation BID1 and image recognition BID2 BID3 .

In this textto-speech (TTS) work, we are instead interested in few-shot meta-learning.

Here the objective of training with many data is not to learn a fixed-parameter classifier, but rather to learn a "prior" neural network.

This prior TTS network can be adapted rapidly, using few data, to produce TTS systems for new speakers at deployment time.

That is, the intention is not to learn a fixed final model, but rather to learn a model prior that harnesses few data at deployment time to learn new behaviours rapidly.

The output of training is not longer a fixed model, but rather a fast learner.

Biology provides motivation for this line of research.

It may be argued that evolution is a slow adaptation process that has resulted in biological machines with the ability to adapt rapidly to new data during their lifetimes.

These machines are born with strong priors that facilitate rapid learning.

We consider a meta-learning approach where the model has two types of parameters: task-dependent parameters and task-independent parameters.

During training, we learn all of these parameters but discard the task-dependent parameters for deployment.

The goal is to use few data to learn the task-dependent parameters for new tasks rapidly.

Task-dependent parameters play a similar role to latent variables in classical probabilistic graphical models.

Intuitively, these variables introduce flexibility, thus making it easier to learn the taskindependent parameters.

For example, in classical HMMs, knowing the latent variables results in a simple learning problem of estimating the parameters of an exponential-family distribution.

In neural networks, this approach also facilitates learning when there is clear data diversity and categorization.

We show this for adaptive TTS BID4 BID5 .

In this setting, speakers correspond to tasks.

During training we have many speakers, and it is therefore helpful to have task-dependent parameters to capture speaker-specific voice styles.

At the same time, it is useful to have a large model with shared parameters to capture the generic process of mapping text to speech.

To this end, we employ the WaveNet model.

WaveNet BID6 is an autoregressive generative model for audio waveforms that has yielded state-of-art performance in speech synthesis.

This model was later modified for real-time speech generation via probability density distillation into a feed-forward model BID7 .

A fundamental limitation of WaveNet is the need for hours of training data for each speaker.

In this paper we describe a new WaveNet training procedure that facilitates adaptation to new speakers, allowing the synthesis of new voices from no more than 10 minutes of data with high sample quality.

We propose several extensions of WaveNet for sample-efficient adaptive TTS.

First, we present two non-parametric adaptation methods that involve fine-tuning either the speaker embeddings only or all the model parameters given few data from a new speaker.

Second, we present a parametric textindependent approach whereby an auxiliary network is trained to predict new speaker embeddings.

The experiments will show that all the proposed approaches, when provided with just a few seconds or minutes of recording, can generate high-fidelity utterances that closely resemble the vocal tract characteristics of a demonstration speaker, particularly when the entire model is fine-tuned end-to-end.

When fine-tuning by first estimating the speaker embedding and subsequently fine-tuning the entire model, we achieve state-of-the-art results in terms of sample naturalness and voice similarity to target speakers.

These results are robust across speech datasets recorded under different conditions and, moreover, we demonstrate that the generated samples are capable of confusing the state-of-the-art text-independent speaker verification system BID8 .TTS techniques require hours of high-quality recordings, collected in controlled environments, for each new voice style.

Given this high cost, reducing the length of the training dataset could be valuable.

For example, it is likely to be very beneficial when attempting to restore the voices of patients who suffer from voice-impairing medical conditions.

In these cases, long high quality recordings are scarce.

WaveNet is an autoregressive model that factorizes the joint probability distribution of a waveform, x = {x 1 , . . .

, x T }, into a product of conditional distributions using the probabilistic chain rule: DISPLAYFORM0 p(x t |x 1:t−1 , h; w), where x t is the t-th timestep sample, and h and w are respectively the conditioning inputs and parameters of the model.

To train a multi-speaker WaveNet, the conditioning inputs h consist of the speaker identity s, the linguistic features l, and the logarithmic fundamental frequency f 0 values.

l encodes the sequence of phonemes derived from the input text, and f 0 controls the dynamics of the pitch in the generated utterance.

Given the speaker identity s for each utterance in the dataset, the Figure 2: Training (slow, lots of data), adaptation (fast, few data) and inference stages for the SEA-ALL architecture.

The components with bold pink outlines are fine-tuned during the adaptation phase.

The purpose of training is to produce a prior.

This prior is combined with few data during adaptation to solve a new task.

This adapted model is then deployed in the final inference stage.

model is expressed as: DISPLAYFORM1 where a table of speaker embedding vectors e s (Embedding in FIG0 ) is learned alongside the standard WaveNet parameters.

These vectors capture salient voice characteristics across individual speakers, and provide a convenient mechanism for generalizing WaveNet to the few-shot adaptation setting in this paper.

The linguistic features l and fundamental frequency values f 0 are both time-series with a lower sampling frequency than the waveform.

Thus, to be used as local conditioning variables they are upsampled by a transposed convolutional network.

During training, l and f 0 are extracted by signal processing methods from pairs of training utterance and transcript, and during testing, those values are predicted from text by existing models .

In recent years, a large body of literature uses large datasets to train models to learn an input-output mapping that is then used for inference.

In contrast, few-shot meta-learning introduces an additional step, adaptation.

In this meta-learning setting, the purpose of training becomes to learn a prior.

During adaptation, this prior is combined with few data to rapidly learn a new skill; in this case adapting to a new speakers' voice style.

Finally, the new skill is deployed, which in this paper we are referring to as inference.

These three stages -training, adaptation and inference -are illustrated in Figure 2 .We present two multi-speaker WaveNet extensions for few-shot voice adaptation.

First, we introduce a non-parametric model fine-tuning approach, which involves adapting either the speaker embeddings or all the model parameters using held-aside demonstration data.

Second, and for comparison purposes, we use a parametric approach whereby an auxiliary network is trained to predict the embedding vector of a new speaker using the demonstration data.

Inspired by few-shot learning we first pre-train a multi-speaker conditional WaveNet model on a large and diverse dataset, as described in Section 2.

Subsequently, we fine-tune the model parameters by retraining with respect to held-aside adaptation data.

Training this WaveNet model to maximize the conditional log-likelihood of the generated audio jointly optimizes both the set of speaker parameters {e s } and the shared WaveNet core parameters w. Next, we extend this method to a new speaker by extracting the l and f 0 features from their adaptation data and randomly initializing a new embedding vector e.

We then optimize e such that the demonstration waveforms, {x DISPLAYFORM0 0,demo )}, are likely under the model with w fixed (SEA-EMB): DISPLAYFORM1 Alternatively, all of the model parameters may be additionally fine-tuned (SEA-ALL): DISPLAYFORM2 0,demo ; e, w).Both methods are non-parametric approaches to few-shot voice adaptation as the number of embedding vectors scales with the number of speakers.

However, the training processes are slightly different.

Because the SEA-EMB method optimizes only a low-dimensional vector, it is far less prone to overfitting, and we are therefore able to retrain the model to convergence even with mere seconds of adaptation data.

By contrast, the SEA-ALL has many more parameters that might overfit to the adaptation data.

We therefore hold out 10% of our demonstration data for calculating a standard early termination criterion.

We also initialize e with the optimal value from the SEA-EMB method, and we find this initialization significantly improves the generalization performance even with a few seconds of adaptation data.

In contrast to the non-parametric approach, whereby a different embedding vector is fitted for each speaker, one can train an auxiliary encoder network to predict an embedding vector for a new speaker given their demonstration data.

Specifically, we model: DISPLAYFORM0 where for each training example, we include a randomly selected demonstration utterance from that speaker in addition to the regular conditioning inputs.

The full WaveNet model and the encoder network e(·) are trained together from scratch.

We refer the reader to the Appendix for further architectural details.

This approach (SEA-ENC) exhibits the advantage of being trained in a transcriptindependent setting given only the input waveform, e(x demo ), and requires negligible computation at adaptation time.

However, the learned encoder can also introduce bias when fitting an embedding due to its limited network capacity.

As an example, demonstrated a typical scenario whereby speaker identity information can be very quickly extracted with deep models from audio signals.

Nonetheless, that the model is less capable of effectively leveraging additional training than approaches based on statistical methods.

The linguistic features and fundamental frequencies which are used as inputs contain information specific to an individual speaker.

As an example, the average voice pitch in the fundamental frequency sequence is highly speaker-dependent.

Instead, we would like these features to be as speaker-independent as possible such that identity is modeled via global conditioning on the speaker embedding.

To achieve this, we normalize the fundamental frequency values to have zero mean and unit variance separately for each speaker during training, denoted asf 0 : DISPLAYFORM0 As mentioned earlier, at test time, we use an existing model to predict (l,f 0 ).

Few-shot learning to build models, where one can rapidly learn using only a small amount of available data, is one of the most important open challenges in machine learning.

Recent studies have attempted to address the problem of few-shot learning by using deep neural networks, and they have shown promising results on classification tasks in vision BID11 BID12 and language .

Few-shot learning can also be leveraged in reinforcement learning, such as by imitating human Atari gameplay from a single recorded action sequence BID14 or online video BID15 .Meta-learning offers a sound framework for addressing few-shot learning.

Here, an expensive learning process results in machines with the ability to learn rapidly from few data.

Meta-learning has a long history BID16 BID17 , and recent studies include efforts to learn optimization processes BID18 that have been shown to extend naturally to the few-shot setting BID20 ).

An alternative approach is model-agnostic meta learning (MAML) BID21 , which differs by using a fixed optimizer and learning a set of base parameters that can be adapted to minimize any task loss by few steps of gradient descent.

This method has shown promise in robotics BID22 BID23 .In generative modeling, few-shot learning has been addressed from several perspectives, including matching networks BID24 and variable inference for memory addressing BID25 .

BID26 developed a sequential generative model that extended the Deep Recurrent Attention Writer (DRAW) model BID27 , and BID28 extended PixelCNN (Van Oord et al., 2016) with neural attention for few-shot auto-regressive density modeling.

BID30 presented a gated linear model able to model complex densities from a single pass of a limited dataset.

Early attempts of few-shot adaptation involved the attention models of BID28 and MAML BID21 ), but we found both of these strategies failed to learn informative speaker embedding in our preliminary experiments.

There is growing interest in developing neural TTS models that can be trained end-to-end without the need for hand-crafted representations.

In this study we focus on extending the autoregressive WaveNet model BID6 to the few-shot learning setting to adapt to speakers that were not presented at training time.

Other recent neural TTS models include Tacotron 2 (SkerryRyan et al., 2018) (building on ) which uses WaveNet as a vocoder to invert mel-spectrograms generated by an attentive sequence-to-sequence model.

DeepVoice 2 ) (building on BID34 ) introduced a multi-speaker variation of Tacotron that learns a low-dimensional embedding for each speaker, which was further extended in DeepVoice 3 to a 2,400 multi-speaker scenario.

Unlike WaveNet and DeepVoice, the Char2Wav BID36 and VoiceLoop ) models produce World Vocoder Features BID38 instead of generating raw audio signals.

Although many of these systems have produced high-quality samples for speakers present in the training set, generalizing to new speakers given only a few seconds of audio remains a challenge.

There have been several concurrent works to address this few-shot learning problem.

The VoiceLoop model introduced a novel memory-based architecture that was extended by to few-shot voice style adaptation, by introducing an auxiliary fitting network that predicts the embedding of a new speaker.

BID40 extended the Tacotron model for one-shot speaker adaptation by conditioning on a speaker embedding vector extracted from a pretrained speaker identity model of BID8 .

The most similar approached to our work was proposed by for the DeepVoice 3 model.

They considered both predicting the embedding with an encoding network and fitting the embedding based on a small amount of adaptation data, but the adaptation was applied to a prediction model for mel-spectrograms with a fixed vocoder.

In this section, we evaluate the quality of samples of SEA-ALL, SEA-EMB and SEA-ENC.

We first measure the naturalness of the generated utterances using the standard Mean Opinion Score (MOS) procedure.

Then, we evaluate the similarity of generated and real samples using the subjective MOS test and objectively using a speaker verification system BID8 .

Finally, we study these results varying the size of the adaptation dataset.

We train a WaveNet model for each of our three methods using the same dataset, which combines the high-quality LibriSpeech audiobook corpus BID42 3.61 ± 0.06 3.56 ± 0.06 3.65 ± 0.06 3.58 ± 0.06 Table 1 : Naturalness of the adapted voices using a 5-scale MOS score (higher is better) with 95% confidence interval on the LibriSpeech and VCTK held-out adaptation datasets.

Numbers in bold are the best few-shot learning results on each dataset without statistically significant difference.

van den Oord et al. (2016) was trained with 24-hour production quality data, Nachmani et al. (2018) Our few-shot model performance is evaluated using two hold-out datasets.

First, the LibriSpeech test corpus consists of 39 speakers, with an average of approximately 52 utterances and 5 minutes of audio per speaker.

For every test speaker, we randomly split their demonstration utterances into an adaptation set for adapting our WaveNet models and a test set for evaluation.

The subset of utterances used for early termination in Section 3.1 is chosen from the adaptation set.

There are about 4.2 utterances on average per speaker in the test set and the rest in the adaptation set.

Second, we consider a subset of the CSTR VCTK corpus BID43 consisting of 21 American English speakers, with approximately 368 utterances and 12 minutes of audio per speaker.

We also apply the adaptation/test split with 10 utterances per speaker for test.

We emphasize that no data from VCTK was presented to the model at training time.

Since our underlying WaveNet model was trained on data largely from LibriSpeech (which was recorded under noisier conditions than VCTK), one might expect that the generated samples on the VCTK dataset contain characteristic artifacts that make generated samples easier to distinguish from real utterances.

However, our evaluation using VCTK indicates that our model generalizes effectively and that such artifacts are not detectable.

Synthetic utterances are provided on our demo webpage 1 .It is worth mentioning, that SEA-ENC requires no adaptation time.

Where for SEA-EMB, it takes 5 ∼ 10k optimizing steps to fit the embedding vector, and an additional 100 ∼ 200 steps to fine-tune the entire model using early stopping for SEA-ALL.

We measure the quality of the generated samples by conducting a MOS test, whereby subjects are asked to rate the naturalness of generated utterances on a five-point Likert Scale (1: Bad, 2: Poor, 3: Fair, 4: Good, 5: Excellent).

Furthermore, we compare with other published few-shot TTS systems systems, that were developed in parallel to this work.

However, the literature uses varying combinations of training data and evaluation splits making comparison difficult.

The results presented are from the closest experimental setups to ours.

Table 1 presents MOS for the adaptation models compared to real utterances.

Two different adaptation dataset sizes are considered; T = 10 seconds, and T ≤ 5 minutes for LibriSpeech (T ≤ 10 minutes for VCTK).

For reference on 16 kHz data, WaveNet trained on a 24-hour production quality speech dataset (van den Oord et al., 2016) achieves a score of 4.21, while for LibriSpeech our best few-shot model attains an MOS score of 4.13 using only 5 minutes of data given a pre-trained multi-speaker model.

We note that both fine-tuning models produce overall "good" samples for both the LibriSpeech and VCTK test sets, with SEA-ALL outperforming SEA-EMB in all cases.

SEA-ALL is on par 3.41 ± 0.10 3.75 ± 0.09 3.51 ± 0.10 3.97 ± 0.09 SEA-EMB (ours)3.42 ± 0.10 3.56 ± 0.10 3.07 ± 0.10 3.18 ± 0.10 SEA-ENC (ours)2.47 ± 0.09 2.59 ± 0.09 2.07 ± 0.08 2.19 ± 0.09 Table 2 : Voice similarity of generated voices using a 5-scale MOS score (higher is better) with 95% confidence interval on the LibriSpeech and VCTK held-out adaptation datasets.with the state-of-the-art performance on both datasets.

The addition of extra adaptation data beyond 10 seconds of audio helps performance on LibriSpeech but not VCTK, and the gap between our best model and the real utterance is also wider on VCTK, possibly due to the different recording conditions.

Beside naturalness, we also measure the similarity of the generated and real voices.

The quality of similarity is the main evaluation metric for the voice adaptation problem.

We first follow the experiment setup of BID40 to run a MOS test for a subjective assessment and then use a speaker verification model for objective evaluation in the next section.

In every trial of this test a subject is presented with a pair of utterances consisting of a real utterance and another real or generated utterance from the same speaker, and is asked to rate the similarity in voice identity using a five-scale score (1: Not at all similar, 2: Slightly similar, 3: Moderately similar, 4: Very similar, 5: Extremely similar).

Table 2 shows the MOS for real utterances and all the adaptation models under two adaptation data time settings on both datasets.

Again, the SEA-ALL model outperforms the other two models, and the improvement over SEA-EMB scales with the amount of adaptation data.

Particularly, the learned voices on the VCTK dataset achieve an average score of 3.97, demonstrating the generalization performance on a different dataset.

As a rough comparisson, because of varying training setups, the state of the art system of BID40 achieves scores of 3.03 for LibriSpeech and 2.77 for VCTK when trained on LibriSpeech.

Their model computes the embedding based on the d-vector, similar to our SEA-ENC approach, and performs competitively for the one-shot learning setting, but its performance saturates with 5 seconds of adaptation data, as explained in Section 3.2.

We note the gap of similarity scores between SEA-ALL and real utterances, which suggests that although the generated samples sound similar to the target speakers, humans can still tell the difference from real utterances.

We also apply the state-of-the-art text independent speaker verification (TI-SV) model of BID8 to objectively assess whether the generated samples preserve the acoustic features of the speakers.

We calculate the TI-SV d-vector embeddings for generated and real voices.

In Figure 3 , we visualize the 2-dimensional projection of the d-vectors for a SEA-ALL model trained on T ≤ 5 minutes of data on the LibriSpeech dataset, and T ≤ 10 minutes on VCTK.

There are clear clusters on both datasets, with a strikingly large inter-cluster distance and low intra-cluster separation.

This shows both (1) an ease of correctly identifying the speaker associated with a given generated utterance, and (2) the difficulty in differentiating real from synthetic samples.

A similar figure is presented in BID40 , but there the generated and real samples do not overlap.

This indicates that the method presented in this paper generates voices that are more indistinguishable from real ones, when measured with the same verification system.

In the following subsections, we further analyze these results.

BID8 .

The utterances were generated using T ≤ 5 and T ≤ 10 minute samples from LibriSpeech and VCTK respectively.

EER is marked with a dot.

We first quantify whether generated utterances are attributed to the correct speaker.

Following common practice in speaker verification BID8 , we select the hold-out test set of real utterances from test speakers as the enrollment set and compute the centroid of the d-vectors for each speaker c i .

We then use the adaptation set of test speakers as the verification set.

For every verification utterance, we compute the cosine similarity between its d-vector v and a randomly chosen centroid c i .

The utterance is accepted as one from speaker i if the similarity is exceeds a given threshold.

We repeat the experiments with the same enrollment set and replace the verification set with samples generated by each adaptation method under different data size settings.

Figure 5 : Cosine similarity of real and generated utterances to the real enrollment set.

In our setup we fix the enrollment set together with the speaker verification model from BID8 , and study the performance of different verification sets that are either from real utterances or generated by a TTS system.

TAB4 lists the equal error rate (EER) of the verification model with real and generated verification utterances, and Figure 4 shows the detection error trade-off (DET) curves for a more thorough inspection.

Figure 4 only shows the adaptation models with the maximum data size setting (T ≤ 5 minutes for LibriSpeech and ≤ 10 minutes for VCTK).

The results for other data sizes are provided in Appendix B.We find that SEA-ALL outperforms the other two approaches, and the error rate decreases clearly with the size of demonstration data.

Noticeably, the EER of SEA-ALL is even lower than the real utterance on the LibriSpeech dataset with sufficient adaptation data.

A possible explanation is that the generated samples might be concentrated closer to the centroid of a speaker's embeddings than real speech with larger variance across utterances.

Our SEA-EMB model performs better than SEA-ENC.

Additionally, the benefit of more demonstration data is less significant than for SEA-ALL in both of these models.

In this section, we compare the generated samples and the real utterances of the speaker being imitated.

Figure 5 shows the box-plot of the cosine similarity between the embedding centroids of test speakers' enrollment set and (1) real utterances from the same speaker, (2) real utterances from a different speaker, and (3) generated utterances adapted to the same speaker.

Consistent with the observations from the previous subsection, SEA-ALL performs best.

We further consider an adversarial scenario for speaker verification.

In contrast to the previous standard speaker verification setup where we now select a verification utterance with either a real utterance from the same speaker or a synthetic sample from a model adapted to the same speaker.

Under this setup, the speaker verification system is challenged by synthetic samples and acts as a classifier for real versus generated utterances.

The ROC curve of this setup is shown in FIG3 and the models are using the maximum data size setting.

Other data size settings can be found in Appendix C. If the generated samples are indistinguishable from real utterances, the ROC curve approaches the diagonal line (that is, the verification system fails to separate real and generated voices).

Importantly, SEA-ALL manages to confuse the verification system especially for the VCTK dataset where the ROC curve is almost inline with the diagonal line with an AUC of 0.56.

This paper studied three variants of meta-learning for sample efficient adaptive TTS.

The adaptation method that fine-tunes the entire model, with the speaker embedding vector first optimized, shows impressive performance even with only 10 seconds of audio from new speakers.

When adapted with a few minutes of data, our model matches the state-of-the-art performance in sample naturalness.

Moreover, it outperforms other recent works in matching the new speaker's voice.

We also demon- : ROC curve for real versus generated utterance detection.

The utterances were generated using models with 5 and 10 minutes of training data per speaker from LibriSpeech and VCTK respectively.

Lower curve indicate that the verification system is having a harder time distinguishing real from generated samples.strated that the generated samples achieved a similar level of voice similarity to real utterances from the same speaker, when measured by a text independent speaker verification model.

Our paper considers the adaptation to new voices with clean, high-quality training data collected in a controlled environment.

The few-shot learning of voices with noisy data is beyond the scope of this paper and remains a challenging open research problem.

A requirement for less training data to adapt the model, however, increases the potential for both beneficial and harmful applications of text-to-speech technologies such as the creation of synthesized media.

While the requirements for this particular model (including the high-quality training data collected in a controlled environment and equally high quality data from the speakers to which we adapt, as described in Section 5.1) present barriers to misuse, more research must be conducted to mitigate and detect instances of misuse of text-to-speech technologies in general.

Our encoding network is illustrated as the summation of two sub-network outputs in FIG4 .

The first subnetwork is a pre-trained speaker verification model (TI-SV) BID8 , comprising 3 LSTM layers and a single linear layer.

This model maps a waveform sequence of arbitrary length to a fixed 256-dimensional d-vector with a sliding window, and is trained from approximately 36M utterances from 18K speakers extracted from anonymized voice search logs.

On top of this we add a shallow MLP to project the output d-vector to the speaker embedding space.

The second sub-network comprises 16 1-D convolutional layers.

This network reduces the temporal resolution to 256 ms per frame (for 16 kHz audio), then averages across time and projects into the speaker embedding space.

The purpose of this network is to extract residual speaker information present in the demonstration waveforms but not captured by the pre-trained TI-SV model.

Here we provide the DET curves of speaker verification problem for models with different training data sizes in addition to those shown in Section 5.4.1.

Figure 8 : Detection error trade-off (DET) curve for speaker verification, using the TI-SV speaker verification model BID8 .

The utterances were generated using 1 minute or 10 seconds of utterance from LibriSpeech and VCTK.

EER is marked with a dot.

We provide the ROC curves of the speaker verification problem with adversarial examples from adaptation models with different training data sizes in addition to those shown in Section 5.4.2. : ROC curve for real vs. generated utterance detection.

The utterances were generated using 1 minute or 10 seconds of utterance from LibriSpeech and VCTK.

Lower curve suggests harder to distinguish real from generated samples.

@highlight

Sample efficient algorithms to adapt a text-to-speech model to a new voice style with the state-of-the-art performance.