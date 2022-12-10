In this paper we introduce a new speech recognition system, leveraging a simple letter-based ConvNet acoustic model.

The acoustic model requires only audio transcription for training -- no alignment annotations, nor any forced alignment step is needed.

At inference, our decoder takes only a word list and a language model, and is fed with letter scores from the acoustic model -- no phonetic word lexicon is needed.

Key ingredients for the acoustic model are Gated Linear Units and high dropout.

We show near state-of-the-art results in word error rate on the LibriSpeech corpus with MFSC features, both on the clean and other configurations.

Top speech recognition systems are either complicated pipelines or using more data that is publicly available.

We set out to show that it is possible to train a nearly state of the art speech recognition system for read speech, with a public dataset (LibriSpeech), on a GPU-equipped workstation.

Thus, we present an end-to-end system for speech recognition, going from Mel-Frequency Spectral Coefficients (MFSCs) to the transcription in words.

The acoustic model is trained using letters (graphemes) directly, which take out the need for an intermediate (human or automatic) phonetic transcription.

The classical pipeline to build state of the art systems for speech recognition consists in first training an HMM/GMM model to force align the units on which the final acoustic model operates (most often context-dependent phone states).

This approach takes its roots in HMM/GMM training BID31 .

The improvements brought by deep neural networks (DNNs) and convolutional neural networks (CNNs) BID26 BID27 for acoustic modeling only extend this training pipeline.

The current state of the art on LibriSpeech belongs to this approach too BID16 BID19 , with an additional step of speaker adaptation BID22 BID18 .

Recently, BID25 proposed GMM-free training, but the approach still requires to generate a forced alignment.

An approach that cut ties with the HMM/GMM pipeline (and with forced alignment) was to train with a recurrent neural network (RNN) BID7 ) for phoneme transcription.

There are now competitive end-to-end approaches of acoustic models toppled with RNNs layers as in BID9 BID14 BID23 BID0 , trained with a sequence criterion BID8 .

However these models are computationally expensive, and thus often take a long time to train.

On conversational speech (that is not the topic of this paper), the state of the art is still held by complex ConvNets+RNNs acoustic models, coupled to domain-adapted language models BID32 BID24 .Compared to classical approaches that need phonetic annotation (often derived from a phonetic dictionary, rules, and generative training), we propose to train the model end-to-end, using graphemes directly.

Compared to sequence criterion based approaches that train directly from speech signal to graphemes BID14 , we propose an RNN-free architecture based on convolutional networks for the acoustic model, toppled with a simple sequence-level variant of CTC.We reach the clean speech performance of BID19 , but without performing speaker adaptation.

Our word-error-rate on clean speech is better than BID0 , while being worse on noisy speech, but they train on 11,900 hours while we only train on the 960h available in LibriSpeech's train set.

The rest of the paper is structured as follows: the next section presents the convolutional networks used for acoustic modeling, along with the automatic segmentation criterion and decoding approaches.

The last section shows experimental results on LibriSpeech.

Figure 1: Overview of our acoustic model, which computes MFSC features which are fed to a Gated ConvNet.

The ConvNet output one score for each letter in the dictionary, and for each MFSC frame.

At inference time, theses scores are fed to a decoder (see Section 2.4) to form the most likely sequence of words.

At training time, the scores are fed to the ASG criterion (see FIG1 ) which promotes sequences of letters leading to the transcrition sequence (here "c a t").

Our acoustic model (see an overview in Figure 1 ) is a Convolutional Neural Network (ConvNet) BID13 , with Gated Linear Units (GLUs) .

The model is fed with 40 MFSCs features, and is trained with a variant of the Connectionist Temporal Classification (CTC) criterion BID8 , which does not have blank labels but embarks a simple duration model through letter transitions scores BID2 .

During training, we use dropout on the neural network outputs.

At inference, the acoustic model is coupled with a decoder which performs a beam search, constrained with a count-based language model.

We detail each of these components in the following.

Our system relies on Mel-Frequency Spectral Coefficients (MFSCs), which are obtained by averaging spectrogram values with mel-scale filters.

MFSCs are the step preceding the cosine transform required to compute Mel-Frequency Cepstrum Coefficients (MFCCs), often found in classical HMM/GMM speech systems BID31 because of their dimensionality compression (13 coefficients are often enough to span speech frequencies).

Compared to spectrogram coefficients, MFSCs have the advantage to be more robust to small time-warping deformations.

Our acoustic model is fed with the MFSC frames, and output letter scores for each input frame.

At each time step, there is one score per letter in a given dictionary L. Words are separated by a special letter <sil>.The acoustic model architecture is based on a 1D Gated Convolutional Neural Network (Gated ConvNet) .

Gated ConvNets stack 1D convolutions with Gated Linear Units.

More formally, given an input sequence X ∈ R T ×d i with T frames of d-dimensional vectors, the i th layer of our network performs the following computation: DISPLAYFORM0 where * is the convolution operator, DISPLAYFORM1 are the learned parameters (with convolution kernel size k i ), σ(·) is the sigmoid function and ⊗ is the element-wise product between matrices.

Gated ConvNets have been shown to reduce the vanishing gradient problem, as they provide a linear path for the gradients while retaining non-linear capabilities, leading to state-of-the-art performance both for natural language modeling and machine translation tasks BID5 .

Each MFSC input sequence is normalized with mean 0 and variance 1.

Given an input sequence X ∈ R T ×d , a convolution with kernel size k will output T − k + 1 frames, due to border effects.

DISPLAYFORM0 .. To compensate those border effects, we pad the MFSC features X 0 with zeroed frames.

To take in account the whole network, the padding size is i (k i − 1), divided in two equal parts at the beginning and the end of the sequence.

Most large labeled speech databases provide only a text transcription for each audio file.

In a classification framework (and given our acoustic model produces letter predictions), one would need the segmentation of each letter in the transcription to train properly the model.

Manually labeling the segmentation of each letter would be tedious.

Several solutions have been explored in the speech community to alleviate this issue:1.

HMM/GMM models use an iterative EM procedure: during the Estimation step, the best segmentation is inferred according to the current model, during the Maximization step the model is optimized using the current inferred segmentation.

This approach is also often used to boostrap the training of neural network-based acoustic models.2.

In the context of hybrid HMM/NN systems, the MMI criterion BID1 maximizes the mutual information between the acoustic sequence and word sequences or the Minimum Bayes Risk (MBR) criterion BID6 .

Recent state-of-the-art systems leverage the MMI criterion BID20 .3.

Standalone neural network architectures have also been trained using the Connectionist Temporal Classification (CTC), which jointly infers the segmentation of the transcription while increase the overall score of the right transcription BID8 .

In BID0 it has been shown that letter-based acoustic models trained with CTC could compete with existing phone-based systems, assuming enough training data is provided.

In this paper, we chose a variant of the Connectionist Temporal Classification.

CTC considers all possible sequence sub-word units (e.g. letters), which can lead to the correct transcription.

It also allow a special "blank" state to be optionally inserted between each sub-word unit.

The rational behind the blank state is two-folds: (i) modeling "garbage" frames which might occur between each letter and (ii) identifying the separation between two identical consecutive sub-word unit in a transcription.

FIG1 shows the CTC graph describing all the possible sequences of letters leading to the word "cat", over 6 frames.

We denote G ctc (θ, T ) the CTC acceptance graph over T frames for a given transcription θ, and π = π 1 , . . .

, π T ∈ G ctc (θ, T ) a path in this graph representing a (valid) sequence of letters for this transcription.

CTC assumes that the network output probability scores, normalized at the frame level.

At each time step t, each node of the graph is assigned with its corresponding log-probability letter i (that we denote f t i (X)) output by the acoustic model (given an acoustic sequence X).

CTC minimizes the Forward score over the graph G ctc (θ, T ): DISPLAYFORM0 where the "logadd" operation (also called "log-sum-exp") is defined as logadd(a, b) = log(exp(a) + exp(b)).

This overall score can be efficiently computed with the Forward algorithm.

Blank labels introduce complexity when decoding letters into words.

Indeed, with blank labels "ø", a word gets many entries in the sub-word unit transcription dictionary (e.g. the word "cat" can be represented as "c a t", "c ø a t", "c ø a t", "c ø a ø t", etc... -instead of only "c a t").

We replace the blank label by special letters modeling repetitions of preceding letters.

For example "caterpillar" can be written as "caterpil1ar", where "1" is a label to represent one repetition of the previous letter.

Removing blank labels from the CTC acceptance graph G ctc (θ, T ) (shown in FIG1 ) leads to a simpler graph that we denote G asg (θ, T ) (shown in FIG1 ).

Unfortunately, in practice we observed that most models do not train with this simplification of CTC.

Adding unormalized transition scores g i,j (·) on each edge of the graph, when moving from label i to label j fix the issue.

We observed in practice that normalized transitions led to similar issue that not having transitions.

Considering unnormalized transition scores implies implementing a sequence-level normalization, to avoid the model to diverge (represented by the graph G asg (θ, T ), as shown in FIG1 ).

This leads to the following criterion, dubbed ASG for "Auto SeGmentation": DISPLAYFORM0 (3) The left-hand part in Equation (3) promotes the score of sequences letters leading to the right transcription (as in Equation FORMULA3 for CTC), and the right-hand part demotes the score of all sequences of letters (as does the frame-level normalization -that is the softmax on the acoustic model -for CTC).

As for CTC, these two parts can be efficiently computed with the Forward algorithm.

When removing transitions in Equation (3), the sequence-level normalization becomes equivalent to the frame-level normalization and the ASG criterion is mathematically equivalent to CTC with no blank labels.

We apply dropout at the output to all layers of the acoustic model.

Dropout retains each output with a probability p, by applying a multiplication with a Bernoulli random variable taking value 1/p with probability p and 0 otherwise BID28 .Following the original implementation of Gated ConvNets (Dauphin et al., 2017), we found that using both weight normalization BID21 and gradient clipping BID17 were speeding up training convergence.

The clipping we implemented performs: DISPLAYFORM0 where C is either the CTC or ASG criterion, and is some hyper-parameters which controls the maximum amplitude of the gradients.

We wrote our own one-pass decoder, which performs a simple beam-search with beam threholding, histogram pruning and language model smearing BID29 .

We kept the decoder as simple as possible (under 1000 lines of C code).

We did not implement any sort of model adaptation before decoding, nor any word graph rescoring.

Our decoder relies on KenLM BID10 for the language modeling part.

It also accepts unnormalized acoustic scores (transitions and emissions from the acoustic model) as input.

The decoder attempts to maximize the following: DISPLAYFORM0 where P lm (θ) is the probability of the language model given a transcription θ, α, β, and γ are three hyper-parameters which control the weight of the language model, the word insertion penalty, and the silence insertion penalty, respectively.

The beam of the decoder tracks paths with highest scores according to Equation FORMULA6 , by bookkeeping pair of (language model, lexicon) states, as it goes through time.

The language model state corresponds the (n − 1)-gram history of the n-gram language model, while the lexicon state is the sub-word unit position in the current word hypothesis.

To maintain diversity in the beam, paths with identical (language model, lexicon) states are merged.

Note that traditional decoders combine the scores of the merge paths with a max(·) operation (as in a Viterbi beam-search) -which would correspond to a max(·) operation in Equation (5) instead of logadd(·).

We consider instead the logadd(·) operation, as it takes in account the contribution of all the paths leading to the same transcription, in the same way we do during training (see Equation (3) ).

In Section 3.1, we show that this leads to better accuracy in practice.

We benchmarked our system on LibriSpeech, a large speech database freely available for download BID16 .

We kept the original 16 kHz sampling rate.

We considered the two available setups in LibriSpeech: CLEAN data and OTHER.

We picked all the available data (about 960h of audio files) for training, and the available development sets (both for CLEAN, and OTHER) for tuning all the hyper-parameters of our system.

Test sets were used only for the final evaluations.

The letter vocabulary L contains 30 graphemes: the standard English alphabet plus the apostrophe, silence (<SIL>), and two special "repetition" graphemes which encode the duplication (once or twice) of the previous letter (see Section 2.3.1).

Decoding is achieved with our own decoder (see Section 2.4), with the standard 4-gram language model provided with LibriSpeech 1 , which contains 200, 000 words.

In the following, we either report letter-error-rates (LERs) or word-error-rates (WERs).MFSC features are computed with 40 coefficients, a 25 ms sliding window and 10 ms stride.

We implemented everything using TORCH7 2 .

The ASG criterion as well as the decoder were implemented in C (and then interfaced into TORCH).1 http://www.openslr.org/11.

2 http://www.torch.ch.

We tuned our acoustic model architectures by grid search, validating on the dev sets.

We consider here two architectures, with low and high amount of dropout (see the parameter p in Section 2.3.2).

TAB0 reports the details of our architectures.

The amount of dropout, number of hidden units, as well as the convolution kernel width are increased linearly with the depth of the neural network.

Note that as we use Gated Linear Units (see Section 2.2), each layer is duplicated as stated in Equation (1).Convolutions are followed by a fully connected layer, before the final layer which outputs 30 scores (one for each letter in the dictionary).

This leads to about 130M and 208M of trainable parameters for the LOW DROPOUT and HIGH DROPOUT architectures, respectively.

FIG2 shows the LER and WER on the LibriSpeech development sets, for the first 40 training epochs of our LOW DROPOUT architecture.

LER and WER appear surpringly well correlated, both on the "clean" and "other" version of the dataset.

In TAB1 , we report WERs on the LibriSpeech development sets, both for our LOW DROPOUT and HIGH DROPOUT architectures.

Increasing dropout regularize the acoustic model in a way which impacts significantly generalization, the effect being stronger on noisy speech.

We also report the WER for the decoder ran with the max(·) operation (instead of logadd(·) for other results) used to aggregate paths in the beam with identical (language model, lexicon) states.

It appears advantageous (as there is no complexity increase in the decoder) to use the logadd(·) aggregation.

In TAB2 , we compare our system with several of the best systems on LibriSpeech reported in the literature.

We highlighted the acoustic model architectures, as well as the type of underlying sub-word unit.

Note that phone-based acoustic models output in general senomes; senomes are carefully selected through a complicated procedure involving a phonetic-context-based decision tree built from another GMM/HMM system.

Phone-based system also require an additional word lexicon which translates words into a sequence of phones.

Most systems also perform speaker adaptation; iVectors compute a speaker embedding capturing both speaker and environment information (Xue BID30 .

We also report extra information (besides word transcriptions) which might be used by each system, including speaker adaptation, or any other domain-specific data.

Acoustic Model Sub-word Spkr Adapt.

Extra Resources BID16 HMM+DNN+pNorm phone fMLLR phone lexicon BID0 2D-CNN+RNN letter none 11.9Kh train set, Common Crawl LM BID19 test-clean test-other BID16 5.5 14.0 BID0 5.3 13.3 BID19 4.8 - BID20 4.3 - BID12 -12.5 this paper 4.8 14.5 this paper (no decoder) 6.7 20.8 et al., 2014), while fMMLR is a two-pass decoder technique which computes a speaker transform in the first pass BID4 .

DEEP SPEECH 2 BID0 is the system which is the most related to ours.

In contrast to other systems which combine a Hidden Markov Model (HMM) with a ConvNet, DEEP SPEECH 2 is a standalone neural network.

In contrast to our system, DEEP SPEECH 2 embarks a more complicated acoustic model composed of a ConvNet and a Recurrent Neural Network (RNN), while our system is a simple ConvNet.

Both Deep Speech 2 and our system rely on letters for acoustic modeling, alleviating the need of a phone-based word lexicon.

DEEP SPEECH 2 relies on a lot of speech data (combined with a very large 5-gram language model) to make the letter-base approach competitive , while we limited ourselves to the available data in the LibriSpeech benchmark.

In TAB4 , we report a comparison in WER performance for all systems introduced in TAB2 .

Our system is very competitive with existing approaches.

DEEP SPEECH 2 -which is also a letter-based system -is outperformed on clean data, even though our system has been trained with an order of magnitude less data.

We report also the WER with no decoder, that is taking the raw output of the neural network, with no alterations.

The Gated ConvNet appears very good at modeling true words.

Using a single GPU (no batching), our HIGH DROPOUT Gated ConvNet goes over the CLEAN (5.4h) and OTHER (5.1h) test sets in 4min26s and 4min43s, respectively.

The decoder runs over the CLEAN and OTHER sets in 3min56s and 30min5s, using only one CPU thread -which (considering the decoder alone) corresponds to a .01 and 0.1 Real Time Factor (RTF), respectively.

We have introduced a simple end-to-end automatic speech recognition system, which combines a large (208M parameters) but efficient ConvNet acoustic model, an easy sequence criterion which can infer the segmentation, and a simple beam-search decoder.

The decoding results are competitive on the LibriSpeech corpus (4.8% WER dev-clean).

Our approach breaks free from HMM/GMM pre-training and forced alignment, as well as not being as computationally intensive as RNN-based approaches BID0 .

We based all our work on a publicly available (free) dataset, all of which should make it easier to reproduce.

Further work should include leveraging speaker identity, training from the raw waveform, data augmentation, training with more data, better language models.

<|TLDR|>

@highlight

A letter-based ConvNet acoustic model leads to a simple and competitive speech recognition pipeline.

@highlight

This paper applies gated convolutional neural networks to speech recognition, using the training criterion ASG.