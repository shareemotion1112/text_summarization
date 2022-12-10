End-to-end automatic speech recognition (ASR) commonly transcribes audio signals into sequences of characters while its performance is evaluated by measuring the word-error rate (WER).

This suggests that predicting sequences of words directly may be helpful instead.

However, training with word-level supervision can be more difficult due to the sparsity of examples per label class.

In this paper we analyze an end-to-end ASR model that combines a word-and-character representation in a multi-task learning (MTL) framework.

We show that it improves on the WER and study how the word-level model can benefit from character-level supervision by analyzing the learned inductive preference bias of each model component empirically.

We find that by adding character-level supervision, the MTL model interpolates between recognizing more frequent words (preferred by the word-level model) and shorter words (preferred by the character-level model).

End-to-end automatic speech recognition (ASR) allows for learning a direct mapping from audio signals to character outputs.

Usually, a language model re-scores the predicted transcripts during inference to correct spelling mistakes BID16 .

If we map the audio input directly to words, we can use a simpler decoding mechanism and reduce the prediction time.

Unfortunately, word-level models can only be trained on known words.

Out-of-vocabulary (OOV) words have to be mapped to an unknown token.

Furthermore, decomposing transcripts into sequences of words decreases the available number of examples per label class.

These shortcomings make it difficult to train on the word-level BID2 .Recent works have shown that multi-task learning (MTL) BID8 on the word-and character-level can improve the word-error rate (WER) of common end-to-end speech recognition architectures BID2 BID3 BID18 BID21 BID22 BID24 BID29 .

MTL can be interpreted as learning an inductive bias with favorable generalization properties BID6 .

In this work we aim at characterizing the nature of this inductive bias in word-character-level MTL models by analyzing the distribution of words that they recognize.

Thereby, we seek to shed light on the learning process and possibly inform the design of better models.

We will focus on connectionist temporal classification (CTC) BID15 .

However, the analysis can also prove beneficial to other modeling paradigms, such as RNN Transducers BID14 or Encoder-Decoder models, e.g., BID5 BID9 .Contributions.

We show that, contrary to earlier negative results BID2 BID27 , it is in fact possible to train a word-level model from scratch on a relatively small dataset and that its performance can be further improved by adding character-level supervision.

Through an empirical analysis we show that the resulting MTL model combines the preference biases of word-and character-level models.

We hypothesize that this can partially explain why word-character MTL improves on only using a single decomposition, such as phonemes, characters or words.

Several works have explored using words instead of characters or phonemes as outputs of the end-toend ASR model BID2 BID27 .

Soltau et al. BID27 found that in order to solve the problem of observing only few labels per word, they needed to use a large dataset of 120, 000 hours to train a word-level model directly.

Accordingly, Audhkhasi et al. BID2 reported difficulty to train a model on words from scratch and instead fine-tuned a pre-trained character-level model after replacing the last dense layer with a word embedding.

MTL enables a straightforward joint training procedure to integrate transcript information on multiple levels of granularity.

Treating word-and character-level transcription as two distinct tasks allows for combining their losses in a parallel BID21 BID22 BID28 BID29 or hierarchical structure BID13 BID20 BID24 .

Augmenting the commonly-used CTC loss with an attention mechanism can help with aligning the predictions on both character-and word-level BID3 BID12 BID22 .

All these MTL methods improve a standard CTC baseline.

Finding the right granularity of the word decomposition is in itself a difficult problem.

While Li et al. BID22 used different fixed decompositions of words, sub-words and characters, it is also possible to optimize over alignments and decompositions jointly BID23 .

Orthogonal to these works different authors have explored how to minimize WER directly by computing approximate gradients BID25 BID32 .When and why does MTL work?

Earlier theoretical work argued that the auxiliary task provides a favorable inductive bias to the main task BID6 .

Within natural language processing on text several works verified empirically that this inductive bias is favorable if there is a certain notion of relatedness between the tasks BID4 BID7 BID26 .

Here, we investigate how to characterize the inductive bias learned via MTL for speech recognition.

The CTC loss is defined as follows BID15 : DISPLAYFORM0 where x is the audio input, commonly a spectrogram, and π is a path that corresponds to the groundtruth transcript z. The squashing function B maps a path π to the output z by first merging repetitions and then deleting so-called blank tokens.

The gradient of the CTC loss can be computed efficiently using a modified forward-backward algorithm.

Typically, π t is a categorical random variable over the corresponding output alphabet A = {a, b, c, ..., }.

Here, is the blank token which encodes the empty string.

This output representation enables the model to be able to transcribe any word possible without a specified alignment.

Character-level CTC models are often supplemented by an external language model that can significantly improve the accuracy of the ASR.

This is because these models still make spelling mistakes despite being trained on large amounts of data BID0 .By using an alphabet of words one can ensure that there are no misspellings.

The alphabet could contain, for example, the most common words found in the training set.

This has the advantage that any word is guaranteed to be spelled correctly and that costly re-scoring on a character-level is avoided.

However, by using a word-level decoding, we can no longer predict rare or new words.

In this case the model has to be content with outputting an unknown token.

Another challenge when using a word-level model is label sparsity.

While we will observe many examples of a single character, there will be fewer for a single word, making overfitting more likely.

We aim at counter-acting these shortcomings by making use of character-level information during training, similar to Audhkhasi et al. BID2 .In this work we combine word-and character-level models via an MTL loss and denote this a word-character-level model.

We treat each output-level prediction as a separate task and form a linear combination of the losses.

The MTL loss is then defined as DISPLAYFORM1 where λ ≥ 0 defines a hyperparameter to weight the influence of the character-level CTC loss L char against the word-level CTC loss L word .

In our experiments we set it to 1, giving equal contribution to both loss terms, but other choices may improve the performance.

Alternatively, one could try to estimate this weight based on the uncertainty BID17 or gradient norm BID10 of each loss term.

We experimented with these approaches, but did not observe any significant improvement in performance over the equally-weighted loss.

We trained our models using a convolutional architecture which is based on Wav2Letter BID11 .

Details can be found in the appendix.

Compared to recurrent neural networks, convolutional neural networks avoid iterative computation over time and suffer less from the vanishing/exploding gradient problem.

They achieve comparable performance in terms of WER BID11 BID31 .

We performed all experiments on read news articles from the Wall Street Journal (WSJ) BID30 .

This dataset has relatively little background noise and allows us to focus on the influence of word frequency and word length.

We used the si284 subset for training, and dev92 for validation.

For the character-level model we used 32 different characters which include the space-character and a blank token.

To define the output alphabet for the word-level model, we included all words that appeared at least 5 times in the training set in addition to a blank and an unknown token.

This corresponds to an alphabet of 9411 units with an OOV rate of 9 % on the training set, and 10 % on the validation set, which represents a lower bound for the achievable WER of a word-level model.

For the MTL model we let word-and character-level model share every layer but the last.

To decode the output on the character-and word-level, we used greedy decoding.

In order to get rid of unknown tokens in our prediction, we employed the following heuristic BID21 : For each unknown token predicted on the word-level, we substituted the corresponding word on the character-level that was defined at the same time step.

To compare our results we also trained word-and character-only models.

For optimization we used the Adam-optimizer BID19 with a learning rate of 5e−4 and a batch size of 16 to fit the whole model into the memory of one GPU.

We applied batch normalization and dropout.

For the input data, we transformed each utterance into spectrograms over 20 ms with a shift of 10 ms using 40 log-mel coefficients, standardized per spectrogram.

We ran each experiment for 100 epochs, corresponding to 233, 838 updates.

MTL performance.

The results of our experiments can be found in FIG0 .

It shows the learning curve for the word-and character-level components by measuring the WER on the validation set.

The dashed line shows the achieved WER using a character-level model without joint word-level training.

We observe that MTL converges faster and to a lower WER of 23 %, which is 5 percentage points lower than the character-level component of the MTL network, or the single-task character-level baseline.

Using a beam search decoder with a lexicon constraint on the character-level model reduces the WER from 28 % to a WER of 24 %, which is still higher than our MTL error.

This shows that MTL performs favorably even without a language model.

A word-level-only model achieved the same performance as the character-level baseline on this dataset.

Contrary to the findings of Audhkhasi et al. BID2 , this shows that it is indeed possible to train a word-level model from scratch, even without a large amount of training data.

While the combined decoding only gives an improvement of 0.7 percentage points in terms of WER, it eliminates unknown-token predictions which might make transcripts more readable.

Characterizing the inductive bias.

Arpit et al. BID1 have shown that a neural network trained with stochastic gradient descent learns easier examples first.

We argue that we can characterize the preference bias of our model and learning algorithm by showing which examples are easy to classify in the particular representation that each of the models is learning.

Since ASR models are usually evaluated in terms of WER, we consider which words each model is learning.

To this end we chose a relatively clean dataset and considered the attributes frequency and length to describe a word.

We trained each model for 4 epochs and recorded the distribution of the recognized words during training.

Since we are not given a perfect alignment between speech and ground-truth transcript, we define a word as being recognized if it is both present in the greedy prediction on the validation set and the corresponding ground-truth transcript.

FIG1 shows how the distribution of recognized words changes during training.

We see that the word-level model is biased towards recognizing the most common words and slowly learns less frequent words over time.

This makes sense since more weight is given to the corresponding examples.

While the same effect is present in the character-level model, it covers the complete support of the word frequency distribution in the same number of steps.

On the other hand for the length distribution, we see that the word-level model covers all words independent of its length within the beginning of training.

The character-level model focuses strongly on shorter words before it covers the whole range of the word length distribution.

If we compare the learning dynamics of both models, we find that each model learns words with different characteristics more easily.

If we take a look at the MTL model, we see that it combines both biases and arrives at learning a distribution that is much more uniform across both word frequency and word length.

We hypothesize that putting more emphasis on the tail of each of these distributions combines the strengths of the two models and makes them perform better, especially in distributions that follow a power law such as word frequency rank.

In contrast to earlier studies in the literature, we found that, even on a relatively small dataset, training on a word-level can be feasible.

Furthermore, we found that combining a word-level model with character-level supervision in MTL can improve results noticeably.

To gain a better understanding of this, we characterized the inductive bias of word-character MTL in ASR by comparing the distributions of recognized words at the beginning of training.

We found that adding character-level supervision to a word-level interpolates between recognizing more frequent words (preferred by the word-level model) and shorter words (preferred by the character-level model).

This effect could be even more pronounced on harder datasets than WSJ, such as medical communication data where many long words are infrequent, but very important.

Further analysis of word distributions in terms of pitch, noise and acoustic variability could provide additional insight.

<|TLDR|>

@highlight

Multi-task learning improves word-and-character-level speech recognition by interpolating the preference biases of its components: frequency- and word length-preference.