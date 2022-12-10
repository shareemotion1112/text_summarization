Sequence-to-sequence attention-based models are a promising approach for end-to-end speech recognition.

The increased model power makes the training procedure more difficult, and analyzing failure modes of these models becomes harder because of the end-to-end nature.

In this work, we present various analyses to better understand training and model properties.

We investigate on pretraining variants such as growing in depth and width, and their impact on the final performance, which leads to over 8% relative improvement in word error rate.

For a better understanding of how the attention process works, we study the encoder output and the attention energies and weights.

Our experiments were performed on Switchboard, LibriSpeech and Wall Street Journal.

The encoder-decoder framework with attention BID34 BID60 has been successfully applied to automatic speech recognition (ASR) BID26 BID61 BID58 BID47 and is a promising end-to-end approach.

The model outputs are words, sub-words or characters, and training the model can be done from scratch without any prerequisites except the training data in terms of audio features with corresponding transcriptions.

In contrast to the conventional hybrid hidden Markov models (HMM) / neural network (NN) approach BID8 Morgan, 1994, Robinson, 1994] , the encoder-decoder model does not model the alignment explicitly.

In the hybrid HMM/NN approach, a latent variable of hidden states is introduced, which model the phone state for any given time position.

Thus by searching for the most probable sequence of hidden states, we get an explicit alignment.

There is no such hidden latent variable in the encoder decoder model.

Instead there is the attention process which can be interpreted as an implicit soft alignment.

As this is only implicit and soft, it is harder to enforce constraints such as monotonicity, i.e. that the attention of future label outputs will focus also only to future time frames.

Also, the interpretation of the attention weights as a soft alignment might not be completely valid, as the encoder itself can shift around and reorder evidence, i.e. the neural network could learn to pass over information in any possible way.

E.g. the encoder could compress all the information of the input into a single frame and the decoder can learn to just attend on this single frame.

We observed this behavior in early stages of the training.

Thus, studying the temporal "alignment" behavior of the attention model becomes more difficult.

Other end-to-end models such as connectionist temporal classification BID21 has often been applied to ASR in the past BID20 BID23 BID35 BID1 BID51 BID2 BID26 BID63 BID67 .

Other approaches are e.g. the inverted hidden Markov / segmental encoder-decoder model BID5 , the recurrent transducer BID4 BID41 , or the recurrent neural aligner .

Depending on the interpretation, these can all be seen as variants of the encoder decoder approach.

In some of these models, the attention process is not soft, but a hard decision.

This hard decision can also become a latent variable such that we include several choices in the beam search.

This is also referred to as hard attention.

Examples of directly applying this idea on the usual attention approach are given by BID43 , BID0 , , BID33 , BID27 .We study recurrent NN (RNN) encoder decoder models in this work, which use long short-term memory (LSTM) units BID24 .

Recently the transformer model BID57 gained attention, which only uses feed-forward and self-attention layers, and the only recurrence is the label feedback in the decoder.

As this does not include any temporal information, some positional encoding is added.

This is not necessary for a RNN model, as it can learn such encoding by itself, which we demonstrate later for our attention encoder.

We study attention models in more detail here.

We are interested in when, why and how they fail and do an analysis on the search errors and relative error positions.

We study the implicit alignment behavior via the attention weights and energies.

We also analyze the encoder output representation and find that it contains information about the relative position and that it specially marks frames which should not be attended to, which correspond to silence.2 Related work BID25 analyzes individual neuron activations of a RNN language model and finds a neuron which becomes sensitive to the position in line.

BID7 analyzed the hidden activations of the DeepSpeech 2 BID1 ] CTC end-to-end system and shows their correlation to a phoneme frame alignment.

BID36 analyzed the encoder state and the attention weights of an attention model and makes similar observations as we do.

Attention plots were used before to understand the behaviour of the model BID15 .

BID6 performed a comparison of the alignment behavior between hybrid HMM/NN models, the inverted HMM and attention models.

BID42 investigate the effects of varying block sizes, attention types, and sub-word units.

Understanding the inner working of a speech recognition system is also subject in , where the authors examine activation distribution and temporal patterns, focussing on the comparison between LSTM and GRU systems.

A number of saliency methods BID50 BID32 BID52 are used for interpreting model decisions.

In all cases, we use the RETURNN framework BID65 for neural network training and inference, which is based on TensorFlow BID54 and contains some custom CUDA kernels.

In case of the attention models, we also use RETURNN for decoding.

All experiments are performed on single GPUs, we did not take advantage of multi-GPU training.

In some cases, the feature extraction, and in the hybrid case the decoding, is performed with RASR BID59 .

All used configs as well as used source code are published.

The Switchboard corpus BID19 ] consists of English telephone speech.

We use the 300h train dataset (LDC97S62), and a 90% subset for training, and a small part for cross validation, which is used for learning rate scheduling and to select a few models for decoding.

We decode and report WER on Hub5'00 and Hub5'01.

We use Hub5'00 to select the best model which we report the numbers on.

Our hybrid HMM/NN model uses a deep bidirectional LSTM as described by BID64 .

Our baseline has 6 layers with 500 nodes in each direction.

It uses dropout of 10% on the non-recurrent input of each LSTM layer, gradient noise with standard deviation of 0.3, Adam with Nesterov momentum (Nadam) BID18 , Newbob learning rate scheduling BID64 , and focal loss BID28 ].Our attention model uses byte pair encoding BID49 as subword units.

We follow the baseline with about 1000 BPE units as described by .

All our baselines and a comparison to results from the literature are summarized in TAB0 .

The LibriSpeech dataset BID37 are read audio books and consists of about 1000h of speech.

A subset of the training data is used for cross-validation, to perform learning rate scheduling and to select a number of models for full decoding.

We use the dev-other set for selecting the final best model.

The end-to-end attention model uses byte pair encoding (BPE) BID49 as subword units with a vocabulary of 10k BPE units.

We follow the baseline as described by .

A comparison of our baselines and other models are in TAB1 .

The Wall Street Journal (WSJ) dataset BID39 is read text from the WSJ.

We use 90% of si284 for training, the remaining for cross validation and learning rate scheduling, dev93 for validation and selection of the final model, and eval92 for the final evaluation.

We trained an end-to-end attention model using BPE subword units, with a vocabulary size of about 1000 BPE units.

Our preliminary results are shown in TAB2 .

Our attention model is based on the improved pretraining scheme as described in Section 5.

the search errors where the models recognized sentence (via beam search) has a worse model score than the ground truth sentence.

We observe that we do only very few search errors, and the amount of search errors seems independent from the final WER performance.

Thus we conclude that we mostly have a problem in the model.

We also were interested in the score difference between the best recognized sentence and the ground truth sentence.

The results are in Fig. 2 .

We can see that they concentrate on the lower side, around 10%, which is an indicator why a low beam size seems to be sufficient.

It has been observed that pretraining can be substantial for good performance, and sometimes to get a converging model at all [Zeyer et al., 2018a,b] .

We provide a study on cases with attention-based models where pretraining benefits convergence, and compare the performance with and without pretraining.

8 >100 >100 >100 >100 6 8 >100 >100 >100 >100The pretraining variant of the Switchboard baseline (6 layers, time reduction 8 after pretraining) consists of these steps: 1. starts with 2 layers (layer 1 and 6), time reduction 32, and dropout as well as label smoothing disabled; 2. enable dropout; 3.

3 layers (layer 1, 2 and 6); 4.

4 layers (layer 1, 2, 3 and 6); 5. 5 layers (layer 1, 2, 3, 4 and 6); 6. all 6 layers; 7. decrease time reduction to 8; 8.

final model, enable label smoothing.

Each pretrain step is repeated for 5 epochs, where one epoch corresponds to 1/6 of the whole train corpus.

In addition, a linear learning rate warm-up is performed from 1e-4 to 1e-3 in 10 epochs.

We have to start with 2 layers as we want to have the time pooling in between the LSTM layers.

In TAB3 , performed on Switchboard, we varied the number of encoder layers and encoder LSTM units, both with and without pretraining.

We observe that the overall best model is with 4 layers without the pretraining variant.

I.e. we showed that we can directly start with 4 layers and time reduction 8 and yield very good results.

We even can start directly with 6 layer with a reduced learning rate.

This was surprising to us, as this was not possible in earlier experiments.

This might be due to a reduced and improved BPE vocabulary.

We note that overall all the pretraining experiments seems to run more stable.

We also can see that with 6 layers (and also more), pretraining yields better results than no pretraining.

These results motivated us to perform further investigations into different variants of pretraining.

It seems that pretraining allows to train deeper model, however using too much pretraining can also hurt.

We showed that we can directly start with a deeper encoder and lower time reduction.

In TAB4 , we analyzed the optimal initial number of layers, and the initial time reduction.

We observed that starting with a deeper network improves the overall performance, but also it still helps to then go deeper during pretraining, and starting too deep does not work well.

We also observed that directly starting with time reduction 8 also works and further improves the final performance, but it seems that this makes the training slightly more unstable.

In further experiments, we directly start with 4 layers and time reduction 8.

We were also interested in the optimal number of repetitions of each pretrain step, i.e. how much epochs to train with each pretrain step; the baseline had 5 repetitions.

We collected the results in TAB5 .

In further experiments, we keep 5 repetitions as the default.

It has already been shown by that a lower final time reduction performed better.

So far, the lowest time reduction was 8 in our experiments.

By having a pool size of 3 in the first time max pooling layer, we achieve a better-performing model with time reduction factor of 6 as shown in TAB6 .

So far we always kept the top layer (layer 6) during pretraining as our intuition was that it might help to get always the same time reduction factor as an input to this layer.

When directly starting with the low time reduction, we do not need this scheme anymore, and we can always add a new layer on top.

Comparisons are collected in TAB6 .

We can conclude that this simpler scheme to add layers on top performs better.

We also did experiments with growing the encoder width / number of LSTM units during pretraining.

We do this orthogonal to the growing in depth / number of layers.

As before, our final number of LSTM units in each direction of the bidirectional deep LSTM encoder is 1024.

Initially, we start with 50% of the final width, i.e. with 512 units.

In each step, we linearly increase the number of units such that we have the final number of units in the last step.

We keep the weights of existing units, and weights from/to newly added units are randomly initialized.

We also decrease the dropout rate by the same factor.

We can see that this width growing scheme performs better.

This leads us to our current best model.

Our findings are that pretraining is in general more stable, esp.

for deep models.

However, the pretraining scheme is important, and less pretraining can improve the performance, although it becomes more unstable.

We also used the same improved pretraining scheme and time reduction 6 for WSJ as well as LibriSpeech and observed similar improvements, compare TAB1 .

We have observed that training attention models can be unstable, and careful tuning of initial learning rate, warm-up and pretraining is important.

Related to that, we observe a high training variance.

I.e. with the same configuration but different random seeds, we get some variance in the final WER performance.

We observed this even for the same random seed, which we suspect stems from non-deterministic behaviour in TensorFlow operations such as tf.reduce_sum based on kernels 3-19.9, 19.6, 0.24 25.6-26.6, 26.1, 0.38 12.8-13.3, 13.1, 0.17 19.0-19.7, 19.4, 0.20 attention 5 runs 19.1-19.6, 19.3, 0.22 25.3-26.3, 25.8, 0.40 12.7-13.0, 12.9, 0.12 18.9-19.6, 19.2, 0.27 hybrid 5 seeds 14.3-14.5, 14.4, 0.08 19.0-19.3, 19.1, 0.12 9.6-9.8, 9.7, 0.06 14.3-14.7, 14.5, 0.16 hybrid 5 runs 14.3-14.5, 14.4, 0.07 19.0-19.2, 19.1, 0.09 9.6-9.8, 9.7, 0.08 14.4-14.6, 14.5 This training variance seems to be about the same as due to random seeds, which is higher than we expected.

Note that it also depends a lot on other hyper parameters.

For example, in a previous iteration of the model using a larger BPE vocabulary, we have observed more unstable training with higher variance, and even sometimes some models diverge while others converge with the same settings.

We also compare that to hybrid HMM/LSTM models.

It can be observed that it is lower compared to the attention model.

We argue that is due to the more difficult optimization problem, and also due to the much bigger model.

All the results can be seen in TAB7 .

The encoder creates a high-level representation of the input.

It also arguably represents further information needed for the decoder to know where to attend to.

We try to analyze the output of the encoder and identify and examine the learned function.

In FIG6 , we plotted the encoder output and the attention weights, as well as the word positions in the audio.

One hypothesis for an important function of the encoder is the detection of frames which should not be attended on by the decoder, e.g. which are silent or non-speech.

Such a pattern can be observed in FIG6 .

By performing a dimensionality reduction (PCA) on the encoder output, we can identify the most important distinct information, which we identify as silence detection and encoder time position, compare FIG3 .

Similar behavior was shown by BID36 .

We further try to identify individual cells in the LSTM which encodes the positional information.

By qualitatively inspecting the different neurons activations, we have identified multiple neurons which perform the hypothesized function as shown in FIG4 We also observed that the attention weights are always very local in the encoder frames, and often focus mostly on a single encoder frame, compare FIG6 .

The sharp behavior in the converged attention weight distribution has been observed before BID6 BID36 .

We conclude that the information about the label also needs to be well-localized in the encoder output.

To support this observation, we performed experiments where we explicitly allowed only a local fixed-size window of non-zero attention weights around the arg max of the attention energies, to understand how much we can restrict the local context.

The results can be seen in TAB9 .

This confirms the hypothesis that the information is localized in the encoder.

We explain the gap in performance with decoder frames where the model is unsure to attend, and where a global attention helps the decoder to gather information from multiple frames at once.

We observed that in such case, there is sometimes some relatively large attention weight on the very first and/or very last frame.

We provided an overview of our recent attention models results on Switchboard, LibriSpeech and WSJ.

We performed an analysis on the beam search errors.

By our improved pretraining scheme, we improved our Switchboard baseline by over 8% relative in WER.

We pointed out the high training variance of attention models compared to hybrid HMM/NN models.

We analyzed the encoder output and identified the representation of the relative input position, both clearly visible in the PCA reduction of the encoder but even represented by individual neurons.

Also we found indications that the encoder marks frames which can be skipped by decoder, which correlate to silence.

<|TLDR|>

@highlight

improved pretraining, and analysing encoder output and attention