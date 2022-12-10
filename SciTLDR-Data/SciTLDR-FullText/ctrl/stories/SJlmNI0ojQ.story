End-to-end acoustic-to-word speech recognition models have recently gained popularity because they are easy to train, scale well to large amounts of training data, and do not require a lexicon.

In addition, word models may also be easier to integrate with downstream tasks such as spoken language understanding, because inference (search) is much simplified compared to phoneme, character or any other sort of sub-word units.

In this paper, we describe methods to construct contextual acoustic word embeddings directly from a supervised sequence-to-sequence acoustic-to-word speech recognition model using the learned attention distribution.

On a suite of 16 standard sentence evaluation tasks, our embeddings show competitive performance against a word2vec model trained on the speech transcriptions.

In addition, we evaluate these embeddings on a spoken language understanding task and observe that our embeddings match the performance of text-based embeddings in a pipeline of first performing speech recognition and then constructing word embeddings from transcriptions.

The task of learning fixed-size representations for variable length data like words or sentences, either text or speech-based, is an interesting problem and a focus of much current research.

In the natural language processing community, methods like word2vec BID0 , GLoVE BID1 , CoVe BID2 and ELMo BID3 have become increasingly popular, due to their utility in several natural language processing tasks.

Similar research has progressed in the speech recognition community, where however the input is a sequence of short-term audio features, rather than words or characters.

Therefore, the variability in speakers, acoustics or microphones for different occurrences of the same word or sentence adds to the challenge.

Prior work towards the problem of learning word representations from variable length acoustic frames involved either providing word boundaries to align speech and text BID4 , or chunking ("chopping" or "padding") input speech into fixed-length segments that usually span only one word BID5 BID6 BID7 BID8 .

Since these techniques learn acoustic word embeddings from audio fragment and word pairs obtained via a given segmentation of the audio data, they ignore the specific audio context associated with a particular word.

So the resulting word embeddings do not capture the contextual dependencies in speech.

In contrast, our work constructs individual acoustic word embeddings grounded in utterance-level acoustics.

In this paper, we present different methods of obtaining acoustic word embeddings from an attention-based sequence-to-sequence * Equal contribution model BID9 BID10 BID11 trained for direct Acoustic-to-Word (A2W) speech recognition BID12 .

Using this model, we jointly learn to automatically segment and classify input speech into individual words, hence getting rid of the problem of chunking or requiring pre-defined word boundaries.

As our A2W model is trained at the utterance level, we show that we can not only learn acoustic word embeddings, but also learn them in the proper context of their containing sentence.

We also evaluate our contextual acoustic word embeddings on a spoken language understanding task, demonstrating that they can be useful in non-transcription downstream tasks.

Our main contributions in this paper are the following: 1.

We demonstrate the usability of attention not only for aligning words to acoustic frames without any forced alignment but also for constructing Contextual Acoustic Word Embeddings (CAWE).

2.

We demonstrate that our methods to construct word representations (CAWE) directly from a speech recognition model are highly competitive with the text-based word2vec embeddings BID0 , as evaluated on 16 standard sentence evaluation benchmarks.

3.

We demonstrate the utility of CAWE on a speech-based downstream task of Spoken Language Understanding showing that pretrained speech models could be used for transfer learning similar to VGG in vision BID13 or CoVe in natural language understanding BID2 .

A2W modeling has been largely pursued using Connectionist Temporal Classification (CTC) models BID14 and Sequence-to-Sequence (S2S) models BID9 .

Prior work shows the need for large amounts of training data for these models (thousands of hours of speech) with large word vocabularies of frequently occurring words BID15 BID16 BID17 BID18 BID19 .

Progress in the field showed the possibility of training these models with smaller amount of data (300 hours Switchboard corpus BID20 ) but restricting the vocabulary to words occurring atleast 5 or 10 times BID21 BID22 .

The solutions to generate out-of-vocabulary words have revolved around backing off to smaller units like characters or sub-words BID21 BID16 BID17 BID22 BID19 .

While this solves the problem of rare word generation, the models are no longer pure-word models.[24] present one of the first S2S models for pure-word large vocabulary A2W recognition with the 300 hour Switchboard corpus with a vocabulary of about 30,000 words.

BID24 BID12 build upon their work and improve the training of these models for the large vocabulary task.

BID12 is one of our previous works where we show that the direct A2W model is also able to automatically learn word boundaries without any supervision and is the current best pure-word S2S model.

We use the same model in this work and expand it towards learning acoustic embeddings.

BID4 BID5 BID7 BID6 BID25 BID8 all explore ways to learn acoustic word embeddings.

All above methods except BID6 use unsupervised learning based methods to obtain these embeddings where they do not use the transcripts or do not perform speech recognition.

BID6 use a supervised Convolutional Neural Network based speech recognition model but with short speech frames as input that usually correspond to a single word.

This is the common practice in most prior work that simplifies training but prevents the models to scale to learn contextual word embeddings grounded in utterance level acoustics.

BID4 propose an unsupervised method to learn speech embeddings using a fixed context of words in the past and future.

The drawbacks of their method are the fixed context and need for forced alignment between speech and words for training.

Learning text-based word embeddings is also a rich area of research with well established techniques such as BID0 BID1 .

Research has further progressed into learning contextualized word embeddings BID2 BID3 that are useful in many text-based downstream tasks BID26 .

BID2 learns contextual word embeddings from a fully trained machine translation model and depict re-use of their encoder in other downstream tasks.

Our work ties A2W speech recognition model with learning contextual word embeddings from speech.

Our S2S model is similar in structure to the Listen, Attend and Spell model BID10 which consists of 3 components: the encoder network, a decoder network and an attention model.

The encoder maps the input acoustic features vectors a = (a1, a2, ..., aT ) where ai ∈ R d , into a sequence of higher-level features h = (h1, h2, ..., h T ).

The encoder is a pyramidal (sub-sampling) multi-layer bi-directional Long Short Term Memory (BLSTM) network.

The decoder network is also an LSTM network that learns to model the output distribution over the next target conditioned on sequence of previous predictions i.e. P (y l |y * DISPLAYFORM0 ) is the ground-truth label sequence.

In this work, y * i ∈ U is from a word vocabulary.

This decoder generates targets y from h using an attention mechanism.

We use the location-aware attention mechanism BID11 that enforces monotonicity in the alignments by applying a convolution across time to the attention of previous time step.

This convolved attention feature is used for calculating the attention for the current time step which leads to a peaky distribution BID11 BID12 .

Our model follows the same experimental setup and model hyper-parameters as the word-based models described in our previous work BID12 with the difference of learning 300 dimensional acoustic feature vectors instead of 320 dimensional.

We now describe our method to obtain the acoustic word embeddings from the end-to-end trained speech recognition system described in Section 3.

The model is as shown in Figure 1 where the embeddings are constructed using the hidden representations obtained from the encoder and the attention weights from the decoder.

Our method of constructing "contextual" acoustic word embeddings is similar to a method proposed for text embeddings, CoVe BID2 .

The main challenge that separates our method from CoVe BID2 in learning embeddings from a supervised task, is the problem of alignment between input speech and output words.

We use the location-aware attention mechanism that has the property to assign higher probability to certain frames leading to a peaky attention distribution.

We exploit this property of location-aware attention in an A2W model to automatically segment continuous speech into words as shown in our previous work BID12 , and then use this segmentation to obtain word embeddings.

In the next two subsections, we formalize this process of constructing contextual acoustic word embeddings.

Intuitively, attention weights on the acoustic frames hidden representations reflect their importance in classifying a particular word.

They thereby provide a correspondence between the frame and the word within a given acoustic context.

We can thus construct word representations by weighing the hidden representations of these acoustic frames in terms of their importance to the word i.e. the attention weight.

We show this in the Figure 1 wherein the hidden representations and their attention weights are colored according to their correspondence with a particular word.

Given that aj represents the acoustic frame j, let encoder(aj) represent the higher-level features obtained for the frame aj ( i.e. encoder(aj) = h = (h1, h2, ..., h T ), as explained in Section 3).

Then, for the i th word wi our model first obtains the mappings of wi to acoustic frames aK where K is the set such that ∀k ∈ K k = arg max j (attention(aj)) over all utterances U containing the word wi in the training set.

Below we describe three different ways of using attention to obtain acoustic word embeddings for a word wi (here, n(K) represents the cardinality of the set K): DISPLAYFORM0 Therefore, unweighted Average (U-AVG, Equation 1) is just the unweighted combination of all the hidden representations of acoustic frames mapped to a particular word.

Attention weighted Average (CAWE-W, Equation 2) is the weighted average of the hidden representations of all acoustic frames using the attention weights for a given word.

Finally, maximum attention (CAWE-M, Equation 3) is the hidden representation of the acoustic frame with the highest attention score for a given word across all utterances in the training data.

We call the attention-weighted average and the maximum attention based techniques as Contextual Acoustic Word Embeddings (CAWE) since they are contextual owing to the use of attention scores (over all acoustic frames for a given word).

We use a commonly used speech recognition setup, the 300 hour Switchboard corpus (LDC97S62) BID20 which consists of 2,430 twosided telephonic conversations between 500 different speakers and contains 3 million words of text.

Our second dataset is a 300 hour subset of the How2 BID27 dataset of instructional videos, which contains planned, but free speech, often outdoor and recorded with distant microphones, as opposed to the indoor, telephony, conversational speech of Switchboard.

There are 13,662 videos with a total of 3.5 million words in this corpus.

The A2W obtains a word error rate of 22.2% on Switchboard and 36.6% on CallHome set from the Switchboard Eval2000 test set and 24.3% on dev5 test set of How2.

Datasets for Downstream Tasks: We evaluate our embeddings by using them as features for 16 benchmark sentence evaluation tasks that cover Semantic Textual Similarity (STS 2012-2016 and STS B), classification: Movie Review (MR), product review (CJ), sentiment analysis (SST, SST-FG), question type (TREC), Subjectivity/Objectivity (SUBJ), and opinion polarity (MPQA), entailment and semantic relatedness using the SICK dataset for SICK-E (entailment) and SICK-R (relatedness) and paraphrase detection (MRPC).

The STS and SICK-R tasks measure Spearman's coefficient of correlation between embedding based similarity and human scores, hence the scores range from [−1, 1] where higher number denotes high correlation.

All the remaining tasks are measured on test classification accuracies.

We use the SentEval toolkit BID26 to evaluate.

Training Details: In all downstream evaluations involving classification tasks, we have used a simple logistic regression for classification since a better representation should lead to better scores without using complicated models (hence abstracting away model complexities from our evaluations).

This also means that we can use the concatenation of CAWE and CBOW as features to the logistic regression model without adding tunable embedding parameters.

Discussion: From the results in TAB1 we see that CAWE-M outperforms U-AVG by 34% and 13% and CAWE-W by 33.9% and 12% on Switchboard and How2 datasets respectively in terms of average performance on STS tasks and leads to better or slightly worse performance on the classification tasks.

We observe that CAWE-W usually performs worse than CAWE-M which could be attributed to a noisy estimation of the word embeddings on the account of taking even the less confident attention scores while constructing the embedding.

In contrast, CAWE-M is constructed using the most confident attention score obtained over all the occurrences of the acoustic frames corresponding to a particular word.

We also observe that U-AVG performs worse than CAWE-W on STS and SICK-R tasks since it is constructed using an even noisier process in which all encoder hidden representations are weighted equally irrespective of their attention scores.

Datasets for Downstream Tasks: The datasets are the same as described in Section 5.1.Training Details:

In all the following comparisons, we compare embeddings obtained only from the training set of the speech recognition model, while the text-based word embeddings are obtained by training Continuous Bag-of-Words (CBOW) word2vec model on all the transcripts (train, validation and test).

This was done to ensure a fair comparison between our supervised technique and the unsupervised word2vec method.

This naturally leads to a smaller vocabulary for CAWE.

Further, one of the drawbacks of A2W speech recognition model is that it fails to capture entire vocabulary, recognizing only 3044 words out of 29874 (out of which 18800 words occur less than 5 times) and 4287 out of 14242 total vocabulary for Switchboard and How2 respectively.

Despite this fact, the performance of CAWE is very competitive with word2vec CBOW which does not TAB2 , we see that our embeddings perform as well as the text-embeddings.

Evaluations using CAWE-M extracted from Switchboard based training show that the acoustic embeddings when concatenated with the text embeddings outperform the word2vec embeddings on 10 out of 16 tasks.

This concatenated embedding shows that we add more information with CAWE-M that improves the CBOW embedding as well.

The gains are more prominent in Switchboard as compared to the How2 dataset since How2 is planned instructional speech whereas Switchboard is spontaneous conversational speech (thereby making the How2 characteristics closer to text leading to a stronger CBOW model).

Dataset: In addition to generic sentence-level evaluations, we also evaluate CAWE on the widely used ATIS dataset BID28 for Spoken Language Understanding (SLU).

ATIS dataset is comprised of spoken language queries for airline reservations that have intent and named entities.

Hence, it is similar in domain to Switchboard, making it a useful test bed for evaluating CAWE on a speech-based downstream evaluation task.

Training Details:

For this task, our model is similar to the simple Recurrent Neural Network (RNN) based model architecture as investigated in BID29 .

Our architecture is comprised of an embedding layer, a single layer RNN-variant (Simple RNN, Gated Recurrent Unit (GRU)) along with a dense layer and softmax.

In each instance, we train our model for 10 epochs with RMSProp (learning rate 0.001).

We train each model 3 times with different seed values and report average performance.

Discussion: BID29 concluded that text-based word embeddings trained on large text corpora consistently lead to better performance on the ATIS dataset.

We demonstrate that direct speech-based word embeddings could lead to matching performance when compared to text-based word embeddings in this speech-based downstream task, thus highlighting the utility of our speech based embeddings.

Specifically, we compare the test scores obtained by initializing the model with CAWE-M, CAWE-W and CBOW embeddings and fine-tuning them based on the task.

We present a method to learn contextual acoustic word embeddings from a sequence-to-sequence acoustic-to-word speech recognition model that learns to jointly segment and classify speech.

We analyze the role of attention in constructing contextual acoustic word embeddings, and find our acoustic embeddings to be highly competitive with word2vec (CBOW) text embeddings.

We discuss two variants of such contextual acoustic word embeddings which outperform the simple unweighted average method by upto 34% on semantic textual similarity tasks.

The embeddings also matched the performance of text-based embeddings in spoken language understanding, showing the use of this model as a pre-trained model for other speech-based downstream tasks.

We surmise that contextual audio embeddings will generalize and improve downstream tasks in a way that is similar to their text counterparts, despite the additional complexity presented by noisy audio input.

In the future, we will explore ways to scale our model to larger corpora, larger vocabularies and compare with non-contextual acoustic word embedding methods.

This work was supported by the Center for Machine Learning and Health (CMLH) at Carnegie Mellon University and by Facebook.

<|TLDR|>

@highlight

Methods to learn contextual acoustic word embeddings from an end-to-end speech recognition model that perform competitively with text-based word embeddings.