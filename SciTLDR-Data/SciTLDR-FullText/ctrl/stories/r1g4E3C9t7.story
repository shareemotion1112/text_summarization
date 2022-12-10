Recent studies have highlighted adversarial examples as a ubiquitous threat to different neural network models and many downstream  applications.

Nonetheless, as unique data properties have inspired distinct and powerful learning principles, this paper aims to explore their potentials towards mitigating adversarial inputs.

In particular, our results reveal the importance of using the temporal dependency in audio data to gain discriminate power against adversarial examples.

Tested on the automatic speech recognition (ASR) tasks and three recent audio adversarial attacks, we find that (i) input transformation developed from image adversarial defense provides limited robustness improvement and is subtle to advanced attacks; (ii) temporal dependency can be exploited to gain discriminative power against audio adversarial examples and is resistant to adaptive attacks considered in our experiments.

Our results not only show promising means of improving the robustness of ASR systems, but also offer novel insights in exploiting domain-specific data properties to mitigate negative effects of adversarial examples.

Deep Neural Networks (DNNs) have been widely adopted in a variety of machine learning applications BID18 BID20 .

However, recent work has demonstrated that DNNs are vulnerable to adversarial perturbations BID32 BID10 .

An adversary can add negligible perturbations to inputs and generate adversarial examples to mislead DNNs, first found in image-based machine learning tasks BID10 BID2 BID21 BID7 a; BID30 .Beyond images, given the wide application of DNN-based audio recognition systems, such as Google Home and Amazon Alexa, audio adversarial examples have also been studied recently BID0 BID8 BID17 .

Comparing between image and audio learning tasks, although their state-of-the-art DNN architectures are quite different (i.e., convolutional v.s. recurrent neural networks), the attacking methodology towards generating adversarial examples is fundamentally unanimous -finding adversarial perturbations through the lens of maximizing the training loss or optimizing some designed attack objectives.

For example, the same attack loss function proposed in BID8 ) is used to generate adversarial examples in both visual and speech recognition models.

Nonetheless, different types of data usually possess unique or domain-specific properties that can potentially be used to gain discriminative power against adversarial inputs.

In particular, the temporal dependency in audio data is an innate characteristic that has already been widely adopted in the machine learning models.

However, in addition to improving learning performance on natural audio examples, it is still an open question on whether or not the temporal dependency can be exploited to help mitigate negative effects of adversarial examples.

The focus of this paper has two folds.

First, we investigate the robustness of automatic speech recognition (ASR) models under input transformation, a commonly used technique in the image domain to mitigate adversarial inputs.

Our experimental results show that four implemented transformation techniques on audio inputs, including waveform quantization, temporal smoothing, down-sampling and autoencoder reformation, provide limited robustness improvement against the recent attack method proposed in BID1 , which aims to circumvent the gradient obfuscation issue incurred by input transformations.

Second, we demonstrate that temporal dependency can be used to gain discriminative power against adversarial examples in ASR.

We perform the proposed temporal dependency method on both the LIBRIS BID11 and Mozilla Common Voice datasets against three state-of-the-art attack methods BID0 BID36 considered in our experiments and show that such an approach achieves promising identification of non-adaptive and adaptive attacks.

Moreover, we also verify that the proposed method can resist strong proposed adaptive attacks in which the defense implementations are known to an attacker.

Finally, we note that although this paper focuses on the case of audio adversarial examples, the methodology of leveraging unique data properties to improve model robustness could be naturally extended to different domains.

The promising results also shed new lights in designing adversarial defenses against attacks on various types of data.

Related work An adversarial example for a neural network is an input x adv that is similar to a natural input x but will yield different output after passing through the neural network.

Currently, there are two different types of attacks for generating audio adversarial examples: the Speech-toLabel attack and the Speech-to-Text attack.

The Speech-to-Label attack aims to find an adversarial example x adv close to the original audio x but yields a different (wrong) label.

To do so, Alzantot et al. proposed a genetic algorithm BID0 , and Cisse et al. proposed a probabilistic loss function BID8 .

The Speech-to-Text attack requires the transcribed output of the adversarial audio to be the same as the desired output, which has been made possible by BID16 .

Yuan et al. demonstrated the practical "wav-to-API" audio adversarial attacks BID36 .

Another line of research focuses on adversarial training or data augmentation to improve model robustness BID28 BID26 BID29 BID31 , which is beyond our scope.

Our proposed approach focuses on gaining the discriminative power against adversarial examples through embedded temporal dependency, which is compatible with any ASR model and does not require adversarial training or data augmentation.

TO AUDIO DOMAIN?

Although in recent years both image and audio learning tasks have witnessed significant breakthroughs accomplished by advanced neural networks, these two types of data have unique properties that lead to distinct learning principles.

In images, the pixels entail spatial correlations corresponding to hierarchical object associations and color descriptions, which are leveraged by the convolutional neural networks (CNN) for feature extraction.

In audios, the waveforms possess apparent temporal dependency, which is widely adopted by the recurrent neural networks (RNNs).

For the segmentation task in the image domain, spatial consistency has played an important role in improving model robustness BID22 .

However, it remains unknown whether temporal dependency can have a similar effect of improving model robustness against audio adversarial examples.

In this paper, we aim to address the following fundamental questions: (a) do lessons learned from image adversarial examples transfer to the audio domain?; and (b) can temporal dependency be used to discriminate audio adversarial examples?

Moreover, studying the discriminative power of temporal dependency in audios not only highlights the importance of using unique data properties towards building robust machine learning models but also aids in devising principles for investigating more complex data such as videos (spatial + temporal properties) or multimodal cases (e.g., images + texts).Here we summarize two primary findings concluded from our experimental results in Section 4.Audio input transformation is not effective against adversarial attacks Input transformation is a widely adopted defense technique in the image domain, owing to its low operating cost and easy integration with the existing network architecture BID24 BID33 BID9 .

Generally speaking, input transformation aims to perform certain feature transformation on the raw image in order to disrupt the adversarial perturbations before passing it to a neural network.

Popular approaches include bit quantization, image filtering, image reprocessing, and autoencoder reformation BID35 BID12 BID25 .

However, many existing methods are shown to be bypassed by subsequent or adaptive adversarial attacks BID3 BID13 BID4 BID23 .

Moreover, Athalye et al. BID1 has pointed out that input transformation may cause obfuscated gradients when generating adversarial examples and thus gives a false sense of robustness.

They also demonstrated that in many cases this gradient obfuscation issue can be circumvented, making input transformation still vulnerable to adversarial examples.

Similarly, in our experiments we find that audio input transformations based on waveform quantization, temporal filtering, signal downsampling or autoencoder reformation suffers from similar weakness: the tested model with input transformation becomes fragile to adversarial examples when one adopts the attack considering gradient obfuscation as in BID1 .Temporal dependency possesses strong discriminative power against adversarial examples in automatic speech recognition Instead of input transformation, in this paper, we propose to exploit the inherent temporal dependency in audio data to discriminate adversarial examples.

Tested on the automatic speech recognition (ASR) tasks, we find that the proposed methodology can effectively detect audio adversarial examples while minimally affecting the recognition performance on normal examples.

In addition, experimental results show that a considered adaptive adversarial attack, even when knowing every detail of the deployed temporal dependency method, cannot generate adversarial examples that bypass the proposed temporal dependency-based approach.

Combining these two primary findings, we conclude that the weakness of defense techniques identified in the image case is very likely to be transferred to the audio domain.

On the other hand, exploiting unique data properties to develop defense methods, such as using temporal dependency in ASR, can lead to promising defense approaches that can resist adaptive adversarial attacks.

In this section, we will introduce the effect of basic input transformations on audio adversarial examples, and analyze temporal dependency in audio data.

We will also show that such temporal dependency can be potentially leveraged to discriminate audio adversarial examples.

Inspired by image input transformation methods and as a first attempt, we applied some primitive signal processing transformations to audio inputs.

These transformations are useful, easy to implement, fast to operate and have delivered several interesting findings.

Quantization: By rounding the amplitude of audio sampled data into the nearest integer multiple of q, the adversarial perturbation could be disrupted since its amplitude is usually small in the input space.

We choose q = 128, 256, 512, 1024 as our parameters.

Local smoothing: We use a sliding window of a fixed length for local smoothing to reduce the adversarial perturbation.

For an audio sample x i , we consider the K − 1 samples before and after it, denoted by [x i−K+1 , . . .

, x i , . . .

, x i+K−1 ], as a local reference sequence and replace x i by the smoothed value (average, median, etc) of its reference sequence.

Downsampling: Based on sampling theory, it is possible to down-sample a band-limited audio file without sacrificing the quality of the recovered signal while mitigating the adversarial perturbations in the reconstruction phase.

In our experiments, we down-sample the original 16kHz audio data to 8kHz and then perform signal recovery.

Autoencoder: In adversarial image defending field, the MagNet defensive method BID25 is an effective way to remove adversarial noises: Implement an autoencoder to project the adversarial input distribution space into the benign distribution.

In our experiments, we implement a sequence-to-sequence autoencoder and the whole audio will be cut into frame-level pieces passing through the autoencoder and concatenate them in the final stage, while using the whole audio passing the autoencoder directly is proved to be ineffective and hard to utilize the underlying information.

Due to the fact that audio sequence has an explicit temporal dependency (e.g., correlations in consecutive waveform segments), here we aim to explore if such temporal dependency will be affected by adversarial perturbations.

The pipeline of the temporal dependency based method is shown in FIG0 .

Given an audio sequence, we propose to select the first k portion of it (i.e., the prefix of length k) as input for ASR to obtain transcribed results as S k .

We will also insert the whole sequence into ASR and select the prefix of length k of the transcribed result as S {whole,k} , which has the same length as S k .

We will then compare the consistency between S k and S {whole,k} in terms of temporal dependency distance.

Here we adopt the word error rate (WER) as the distance metric BID19 .

For normal/benign audio instance, S k and S {whole,k} should be similar since the ASR model is consistent for different sections of a given sequence due to its temporal dependency.

However, for audio adversarial examples, since the added perturbation aims to alter the ASR output toward the targeted transcription, it may fail to preserve the temporal information of the original sequence.

Therefore, due to the loss of temporal dependency, S k and S {whole,k} , in this case, will not be able to produce consistent results.

Based on such hypothesis, we leverage the prefix of length k of the transcribed results and the transcribed k portion to potentially recognize adversarial inputs.

The presentation flows of the experimental results are summarized as follows.

We will first introduce the datasets, target learning models, attack methods, and evaluation metrics for different defense/detection methods that we focus on.

We then discuss the defense/detection effectiveness for different methods against each attack respectively.

Finally, we evaluate strong adaptive attacks against these defense/detection methods.

We show that due to different data properties, the autoencoder based defense cannot effectively recover the ground truth for adversarial audios and may also have negative effects on benign instances as well.

Input transformation is less effective in defending adversarial audio than images.

In addition, even when some input transformation is effective for recovering some adversarial audio data, we find that it is easy to perform adaptive attacks against them.

The proposed TD method can effectively detect adversarial audios generated by different attacks targeting on various learning tasks (classification and speech-to-text translation).

In particular, we propose different types of strong adaptive attacks against the TD detection method.

We show that these strong adaptive attacks are not able to generate effective adversarial audio against TD and we provide some case studies to further understand the performance of TD.

In our experiments, we measure the effectiveness on several adversarial audio generation methods.

For audio classification attack, we used Speech Commands dataset.

For the speech-to-text attack, we benchmark each method on both LibriSpeech and Mozilla Common Voice dataset.

In particular, for the Commander Song attack BID36 , we measure the generated adversarial audios given by the authors.

Dataset LibriSpeech dataset: LibriSpeech BID27 is a corpus of approximately 1000 hours of 16Khz English speech derived from audiobooks from the LibriVox project.

We used samples from its test-clean dataset in their website and the average duration is 4.294s.

We generated adversarial examples using the attack method in .Mozilla Common Voice dataset: Common Voice is a large audio dataset provided by Mozilla.

This dataset is public and contains samples from human speaking audio files.

We used the 16Khz-sampled data released in , whose average duration is 3.998s.

The first 100 samples from its test dataset are used to mount attacks, which is the same attack experimental setup as in .Speech Commands dataset: Speech Commands dataset BID34 is an audio dataset contains 65000 audio files.

Each audio is just a single command lasting for one second.

Commands are "yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".Model and learning tasks For the speech-to-text task, we use DeepSpeech speech-to-text transcription network, which is a biRNN based model with beam search to decode text.

For audio classification task, we use a convolutional speech commands classification model.

For the Command Song attack, we evaluate the performance on Kaldi speech recognition platform.

Genetic algorithm based attack against audio classification (GA): For the audio classification task, we consider the state-of-the-art attack proposed in BID0 .

Here an audio classification model is attacked and the audio classes include "yes, no, up, down, etc.".

They aimed to attack such a network to misclassify an adversarial instance based on either targeted or untargeted attack.

Commander Song attack against speech-to-text translation (Commander): Commander Song BID36 ) is a speech-to-text targeted attack which can attack audio extracted from popular songs.

The adversarial audio can even be played over the air with its adversarial characteristics.

Since the Commander Song codes are not available, we measure the effectiveness of the generated adversarial audios given by the authors.

Optimization based attack against speech-to-text translation (Opt): We consider the targeted speechto-text attack proposed by , which uses CTC-loss in a speech recognition system as an objective function and solves the task of adversarial attack as an optimization problem.

Evaluation Metrics For defense method such as input transformation, since it aims to recover the ground truth (original instances) from adversarial instances, we use the word error rate (WER) and character error rate (CER) BID19 as evaluation metrics to measure the recovery efficiency.

WER and CER are commonly used metrics to measure the error between recovered text and the ground truth in word level or character level.

Generally speaking, the error rate (ER) is defined by ER = S+D+I N , where S, D, I is the number of substitutions, deletions and insertions calculated by dynamic string alignment, and N is the total number of word/character in the ground truth text.

To fairly evaluate the effectiveness of these transformations against speech-to-text attack, we also report the ratio of translation distance between instance and corresponding ground truth before and after transformation.

For instance, as a controlled experiment, given an audio instance x (adversarial instance is denoted as x adv ), its corresponding ground truth y, and the ASR function g(·), we calculate the effectiveness ratio for benign instances as DISPLAYFORM0 ,y) , where T (·) denotes the result of transformation and D(·, ·) characterizes the distance function (WER and CER in our case).

For adversarial audio, we calculate the similar efficiency ratio as R adv = D(g(T (x adv )),y) D(g(x adv ),y) .

For the detection method, the standard evaluation metric is the area under curve (AUC) score, aiming to evaluate the detection efficiency.

The proposed TD method is the first data-specific metric to detect adversarial audio, which focuses on how many adversarial instances are captured (true positive) without affecting benign instances (false positive).

Therefore, we follow the standard criteria and report AUC for TD.

For the proposed TD method, we compare the temporal dependency based on WER, CER, as well as the longest common prefix (LCP).

LCP is a commonly used metric to evaluate the similarity between two strings.

Given strings b 1 and b 2 , the corresponding LCP is defined as max In this section, we measured our defense method of autoencoder based defense and input transformation defense for classification attack (GA) and speech-to-text attack (Commander and Opt).

We summarize our work in TAB1 and list some basic results.

For Commander, due to unreleased training data, we are not able to train an autoencoder.

For GA and Opt we have sufficient data to train autoencoder.

Here we perform the primitive input transformation for audio classification targeted attacks and evaluate the corresponding effects.

Due to the space limitation, we defer the results of untargeted attacks to the supplemental materials.

GA We first evaluate our input transformation against the audio classification attack (GA) in BID0 .

We implemented their attack with 500 iterations and limit the magnitude of adversarial perturbation within 5 (smaller than the quantization we used in transformation) and generated 50 adversarial examples per attack task (more targets are shown in the supplementary material).

The attack success rate is 84% on average.

For the ease of illustration, we use Quantization-256 as our input transformation.

As observed in FIG2 , the attack success rates decreased to only 2.1%, and 63.8% of the adversarial instances have been converted back to their original (true) label.

We also measure the possible effects on original audio due to our transformation methods: the original audio classification accuracy without our transformation is 89.2%, and the rate slightly decreased to 89.0% after our transformation, which means the effects of input transformation on benign instances are negligible.

In addition, it also shows that for classification tasks, such input transformation is more effective in mitigating the negative effects of adversarial perturbation.

This potential reason could be that classification tasks do not rely on audio temporal dependency but focus on local features, while speech-to-text task will be harder to defend based on the tested input transformations.

Commander We also evaluate our input transformation method against the Commander Song attack BID36 , which implemented an Air-to-API adversarial attack.

In the paper, the authors reported 91% attack detection rate using their defense method.

We measured our Quan-256 input transformation on 25 adversarial examples obtained via personal communications.

Based on the same detection evaluation metric in BID36 1 , Quan-256 attains 100% detection rate for characterizing all the adversarial examples.

Opt Here we consider the state-of-the-art audio attack proposed in .

We separately choose 50 audio files from two audio datasets (Common Voice, LIBRIS) and generate attacks based on the CTC-loss.

We evaluate several primitive signal processing methods as input transformation under WER and CER metrics in TAB1 and A2.

We then also evaluate the WER and CER based effectiveness ratio we mentioned before to Quantify the effectiveness of transformation.

R benign are shown in the brackets for the first two columns in TAB1 and A2, while R adv is shown in the brackets of last two columns within those tables.

We compute our results using both ground truth and adversarial target "This is an adversarial example" as references.

Here small R benign which is close to 1 indicates that transformation has little effect on benign instances, small R adv represents transformation is effective recovering adversarial audio back to benign.

From Tables A1 and A2 we showed that most of the input transformations (e.g., Median-4, Downsampling and Quan-256) effectively reduce the adversarial perturbation without affecting the original audio too much.

Although these input transformations show certain effectiveness in defending against adversarial audios, we find that it is still possible to generate adversarial audios by adaptive attacks in Section 4.4.

Towards defending against (non-adaptive) adversarial images, MagNet BID25 has achieved promising performance by using an autoencoder to mitigate adversarial perturbation.

Inspired by it, here we apply a similar autoencoder structure for audio and test if such input transformation can be applied to defending against adversarial audio.

We apply a MagNet-like method for feature-extracted audio spectrum map: we build an encoder to compress the information of origin audio features into latent vector z, then use z for reconstruction by passing through another decoder network under frame level and combine them to obtain the transformed audio BID15 .

Here we analyzed the performance of Autoencoder transformation in both GA and Opt attack.

We find that MagNet which gained great effectiveness on defending adversarial images in the oblivious attack setting BID4 BID23 , has limited effect on the audio defense.

GA We presented our results in TAB1 that against classification attack, Autoencoder did not perform well by only reducing attack success rate to 8.2% defeat by other input transformation methods.

Since you can reduce the attack success rate to 10% by just destroying the origin audio data and altering to random guess, it's hard to say that Autoencoder method has good performance.

Opt We report that the autoencoder works not very well for transforming benign instances (57.6 WER in Common Voice compared to 27.5 WER without transformation, 30.0 WER in LIBRIS compared to 12.4 WER without transformation), also fails to recover adversarial audio (76.5 WER in Common Voice and 99.4 WER in LIBRIS) .

This shows that the non-adaptive additive adversarial perturbation can bypass the MagNet-like autoencoder on audio, which implies different robustness implications of image and audio data.

In this section, we will evaluate the proposed TD detection method on different attacks.

We will first report the AUC for detecting different attacks with TD to demonstrate the effectiveness, and we will provide some additional analysis and examples to help better understand TD.

We only evaluate our TD method on speech-to-text attacks (Commander and Opt) because of the audio in the Speech Commands dataset for classification attack is just a single command lasting for one second and thus its temporal dependency is not obvious.

Commander In Commander Song attack, we directly examine whether the generated adversarial audio is consistent with its prefix of length k or not.

We report that by using TD method with k = 1 2 , all the generated adversarial samples showed inconsistency and thus were successfully detected.

Opt Here we show the empirical performance of distinguishing adversarial audios by leveraging the temporal dependency of audio data.

In the experiments, we use these three metrics, WER, CER and LCP, to measure the inconsistency between S k and S {whole,k} .

As a baseline, we also directly train a one layer LSTM with 64 hidden feature dimensions based on the collected adversarial and benign audio instances for classification.

Some examples of translated results for benign and adversarial audios are shown in TAB2 .

Here we consider three types of adversarial targets: short -hey google; medium -this is an adversarial example; and long -hey google please cancel my medical appointment.

We report the AUC score for these detection results for k = 1/2 in TAB3 .

We can see that by using WER as the detection metric, the temporal dependency based method can achieve AUC as high as 0.936 on Common Voice and 0.93 on LIBRIS.

We also explore different values of k and we observe that the results do not vary too much (detailed results can be found in Table A6 in Appendix).

When k = 4/5, the AUC score based on CER can reach 0.969, which shows that such temporal dependency based method is indeed promising in terms of distinguishing adversarial instances.

Interestingly, these results suggest that the temporal dependency based method would suggest an easy-implemented but effective method for characterizing adversarial audio attacks.

In this section, we measured some adaptive attack against the defense and detection methods.

Since the autoencoder based defense almost fails to defend against different attacks, here we will focus on the input transformation based defense and TD detection.

Given that Opt is the strongest attack here, we will mainly apply Opt to perform adaptive attack against the speech-to-text translation task.

We list our experiments' structure in TAB4 .

For full results please refer to the Appendix.

Here we apply adaptive attacks against the preceding input transformations and therefore evaluate the robustness of the input transformation as defenses.

We implemented our adaptive attack based on three input transformation methods: Quantization, Local smoothing, and Downsampling.

For these transformations, we leverage a gradientmasking aware approach to generate adaptive attacks.

In the optimization based attack , the attack achieved by solving the optimization problem: min δ δ 2 2 + c · l(x + δ, t), where δ is referred to the perturbation, x the benign audio, t the target phrase, and l(·) the CTC-loss.

Parameter c is iterated to trade off the importance of being adversarial and remaining close to the original instance.

For quantization transformation, we assume the adversary knows the quantization parameter q. We then change our attack targeted optimization function to min δ qδ 2 2 + c · l(x + qδ, t).

After that, all the adversarial audios can be resistant against quantization transformations and it only increased a small magnitude of adversarial perturbation, which can be ignored by human ears.

When q is large enough, the distortion would increase but the transformation process is also ineffective due to too much information loss.

For downsampling transformation, the adaptive attack is conducted by performing the attack on the sampled elements of origin audio sequence.

Since the whole process is differentiable, we can do adaptive attack through gradient directly and all the adversarial audios are able to attack.

For local smoothing transformation, it is also differentiable in case of average smoothing transformation, so we can pass the gradient effectively.

To attack against median smoothing transformation, we can just convert the gradient back to the median and update its value, which is similar to the maxpooling layer's backpropagation process.

By implementing the adaptive attack, all the smoothing transformation is shown to be ineffective.

We chose our samples randomly from LIBRIS and Common Voice audio dataset with 50 audio samples each.

We implemented our adaptive attack on the samples and passed them through the corresponding input transformation.

We use down-sampling from 16kHZ to 8kHZ, median / average smoothing with one-sided sequence length K = 4, quantization method with q = 256 as our input transformation methods.

In , Decibels (a logarithmic scale that measures the relative loudness of an audio sample) is applied as the measurement of the magnitude of perturbation: dB(x) = max i 20 · log 10 (x i ), which x referred as adversarial audio sampled sequence.

The relative perturbation is calculated as dB x (δ) = dB(δ) − dB(x), where δ is the crafted adversarial noise.

We measured our adaptive attack based on the same criterion.

We show that all the adaptive attacks become effective with reasonable perturbation, as shown in Table 6 .

As suggested in , almost all the adversarial audios have distortion dB x (δ) from -15dB to -45dB which is tolerable to human ears.

From Table 6 , the added perturbation are mostly within this range.

Adaptive Attacks Against Temporal Dependency Based Method To thoroughly evaluate the robustness of temporal dependency based method, we also perform some strong adaptive attack against it.

Notably, even if the adversary knows k, the adaptive attack is hard to conduct due to the fact that this process is non-differentiable.

Therefore, we propose three types of strong adaptive attacks here aiming to explore the robustness of the temporal consistency based method.

Segment attack: Given the knowledge of k, we first split the audio into two parts: the prefix of length k of the audio S k and the rest S k − .

We then apply a similar attack to add perturbation to only S k .

We hope this audio can be attacked successfully without changing S k − since the second part would not receive gradient updates.

Therefore, when performing the temporal-based consistency check, T (S k ) would be translated consistently with T (S {whole,k} ).

To maximally leverage the information of k, here we propose two ways to attack both S k and S k − individually, and then concatenate them together.1.

the target of S k is the first k−portion of the adversarial target, and S k − is attacked to the rest.2.

the target of S k is the whole adversarial target, while we attack S k − to be silence, which means S k − transcribing nothing.

This is different from the segment attack where S k − is not modified at all.

Combination attack: To balance the attack success rate for both sections and the whole sentence against TD, we apply the attack objective function as min δ δ 2 2 + c · (l(x + δ, t) + l((x + δ) k , t k ), where x refers to the whole sentence.

For segment attack, we found that in most cases the attack cannot succeed, that the attack success rate remains at 2% for 50 samples in both LIBRIS and Common Voice datasets, and some of the -35.65 -20.91 -9.48 -23.42 -25.12 examples are shown in Appendix.

We conjecture the reasons as: 1.

S k alone is not enough to be attacked to the adversarial target due to the temporal dependency; 2.

the speech recognition results on S k − cannot be applied to the whole recognition process and therefore break the recognition process for S k .For concatenation attack, we also found that the attack itself fails.

That is, the transcribed result of adv(S k )+adv(S k − ) differs from the translation result of S k +S k − .

Some examples are shown in Appendix.

The failure of the concatenation adaptive attack more explicitly shows that the temporal dependency plays an important role in audio.

Even if the separate parts are successfully attacked into the target, the concatenated instance will again totally break the perturbation and therefore render the adaptive attack inefficient.

On the contrary, such concatenation will have negligible effects on benign audio instances, which provides a promising direction to detect adversarial audio.

For combination attack, we vary the section portion k D used by TD and evaluate the cases where the adaptive attacker uses the same/different section k A .

We define Rand(a,b) as uniformly sampling from [a,b] .

We consider a stronger attacker, for whom the k A can be a set containing random sections.

The detection results for different settings are shown in TAB5 .

From the results, we can see that when |k A | = 1, if the attacker uses the same k A as k D to perform the adaptive attack, the attack can achieve relatively good performance and if the attacker uses different k A , the attack will fail with AUC above 85%.

We also evaluate the case that defender randomly sample k D during the detection and find that it's very hard for the adaptive attacker to perform attacks, which can improve model robustness in practice.

For |k A | > 1, the attacker can achieve some attack success when the set contains k D .

But when |k A | increases, the attacker's performance becomes worse.

The complete results are given in the Appendix.

Notably, the random sample based TD appears to be robust in all cases.

This paper proposes to exploit the temporal dependency property in audio data to characterize audio adversarial examples.

Our experimental results show that while four primitive input transformations on audio fail to withstand adaptive adversarial attacks, temporal dependency is shown to be resistant to these attacks.

We also demonstrate the power of temporal dependency for characterizing adversarial examples generated by three state-of-the-art audio adversarial attacks.

The proposed method is easy to operate and does not require model retraining.

We believe our results shed new lights in exploiting unique data properties toward adversarial robustness.

This work is partially supported by DARPA grant 00009970.

Figure

<|TLDR|>

@highlight

Adversarial audio discrimination using temporal dependency