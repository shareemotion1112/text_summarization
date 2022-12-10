Context information plays an important role in human language understanding, and it is also useful for machines to learn vector representations of language.

In this paper, we explore an asymmetric encoder-decoder structure for unsupervised context-based sentence representation learning.

As a result, we build an encoder-decoder architecture with an RNN encoder and a CNN decoder, and we show that neither an autoregressive decoder nor an RNN decoder is required.

We further combine a suite of effective designs to significantly improve model efficiency while also achieving better performance.

Our model is trained on two different large unlabeled corpora, and in both cases transferability is evaluated on a set of downstream language understanding tasks.

We empirically show that our model is simple and fast while producing rich sentence representations that excel in downstream tasks.

Learning distributed representations of sentences is an important and hard topic in both the deep learning and natural language processing communities, since it requires machines to encode a sentence with rich language content into a fixed-dimension vector filled with continuous values.

We are interested in learning to build a distributed sentence encoder in an unsupervised fashion by exploiting the structure and relationship in a large unlabeled corpus.

Since humans interpret sentences by composing from the meanings of the words, we decompose the task of learning a sentence encoder into two essential components: learning distributed word representations, and learning how to compose a sentence representation from the representations of words in the given sentence.

Numerous studies in human language processing have claimed that the context in which words and sentences are understood plays an important role in human language understanding BID1 BID4 .

The idea of learning from the context information BID35 was recently successfully applied to vector representation learning for words in ; BID30 .

BID8 proposed a unified framework for learning language representation from the unlabeled data, and it is able to generalize to various NLP tasks.

Inspired by the prior work on incorporating context information into representation learning, proposed the Skipthought model, which is an encoder-decoder model for unsupervised sentence representation learning.

The paper exploits the semantic similarity within a tuple of adjacent sentences as supervision, and successfully built a generic, distributed sentence encoder.

Rather than applying the conventional autoencoder model, the skip-thought model tries to reconstruct the surrounding 2 sentences instead of the input sentence.

The learned sentence representation encoder outperforms previous unsupervised pretrained models on the evaluation tasks with no finetuning, and the results are comparable to the models which were trained directly on the datasets in a supervised fashion.

The usage of 2 independent decoders in Skip-thought model matches our intuition that, given the current sentence, inferring the previous sentence and inferring the next one should be different.

Figure 1: Our proposed model is composed of an RNN encoder, and a CNN decoder.

During training, a batch of sentences are sent to the model, and the RNN encoder computes a vector representation for each of sentences; then the CNN decoder needs to reconstruct the paired target sequence, which contains 30 contiguous words right after the input sentence, given the vector representation.

300 is the dimension of word vectors.

D is the dimension of sentence representation, and it varies along with the change of the RNN encoder size.

(Better view in color.)

DISPLAYFORM0 Representation: We aim to provide a model with faster training speed with better transferability than existing algorithms, thus we choose to apply a parameter-free composition function, which is a concatenation of the outputs from a global mean pooling over time and a global max pooling over time, on the computed sequence of hidden states.

The composition function can be represented as DISPLAYFORM1 where max H d· is the max operation on the d-th row of the matrix H, which outputs a scalar.

Thus the representation z has a dimension of 2d.

Decoder: The decoder is a 3-layer CNN to reconstruct the paired target sequence t, which needs to expand z from length 1 to the length of t. Intuitively, the decoder could be a stack of deconvolution layers.

For fast training speed, we optimized the architecture to make it plausible to use fullyconnected layers and convolution layers in the decoder, since generally, convolution layers run faster than deconvolution layers in modern deep learning frameworks.

Suppose that the target sequence t has N words, the first layer of deconvolution will expand z, which could be considered as a sequence with length 1, into a feature map with length N .

It can be easily implemented as a concatenation of outputs from N linear transformations in parallel.

Then the second and third layer are 1D-convolution layers with kernel size 3 and 1, respectively.

The output feature DISPLAYFORM2 , where v ∈ R e , and e is dimension of the word vectors.

Note that our decoder is not an autoregressive model, and it brings us high training efficiency.

We will discuss the reason of choosing this decoder which we call a predict-all-words CNN decoder.

Objective: A softmax layer is applied after the decoder to produce a probability distribution over words at each position, softmax(Ev n ), and the training objective is to minimize the sum of the negative log-likelihood over all positions in the target sequence t: DISPLAYFORM3 The loss function L is summed over all sentences in the training corpus.

We follow the idea of an encoder-decoder model with using the context information for learning sentence representations in an unsupervised fashion.

Since the decoder won't be used after training, and the quality of the generated sequences is not our main focus, it is important to study the design of the decoder.

Generally, a fast training algorithm is preferred, thus proposing a new decoder with high training efficiency and also strong transferability is crucial for an encoder-decoder model.

Our design of the decoder is basically a 3-layer ConvNet, and it predicts all words in the next sequence all at once.

In contrast, existing work, such as Skip-thought , and CNN-LSTM ,use autoregressive RNNs as the decoders.

An autoregressive model is good at generating sequences with high quality, such as language and speech.

However, an autoregressive decoder seems to be unnecessary in an encoder-decoder model for learning sentence representations, since it won't be used after training, and it runs quite slow during training.

Therefore, we conducted experiments to test the necessity of using an autoregressive decoder in learning sentence representations, and we had 2 findings.

Table 1 : The models here all have a bi-directional GRU as the encoder (dimensionality 300 in each direction).

The default way of producing the representation is a concatenation of outputs from a global mean-pooling and a global max-pooling, while "·-Max" refers to the model with only global maxpooling.

Bold numbers are the best results among all presented models.

We found that 1) inputting correct words to an autoregressive decoder is not necessary; 2) predict-all-words decoders work roughly the same as autoregressive decoders; 3) mean+max pooling provides stronger transferability than the max-pooling alone does.

The table supports our choice of the predict-all-words CNN decoder and the way of producing vector representations from the bi-directional RNN encoder.

The experimental design was inspired by BID3 .

The model we designed for the experiment has a bi-directional GRU as the encoder, and an autoregressive decoder, including both RNN and CNN.

We started by analyzing the effect of different sampling strategies of the input words on learning an auto-regressive decoder.

We compared 3 autoregressive decoding settings: 1) using ground-truth words (Baseline), 2) using previously predicted words (Always Sampling), and 3) using uniformly sampled words from the dictionary (Uniform Sampling).

The 3 decoding settings were named by BID3 .

The results are presented in the Table 1.

Generally, the three different decoding settings didn't make much of a difference in terms of the performance on selected downstream tasks, with RNN or CNN as the decoder.

The results tell us that, in terms of learning good sentence representations, the autoregressive decoder doesn't require the correct ground-truth words as the inputs.

Finding II: The model with an autoregressive decoder works roughly the same as the model with a predict-all-words decoder.

With Finding I, we noticed that the correct ground-truth input words to the autoregressive decoder is not necessary in terms of learning sentence representations.

Therefore, it makes sense to test whether we need an autoregressive model at all.

In our model, the CNN decoder predicts all words at once during training, which is different from autoregressive decoders, and we call it a predict-all-words CNN decoder.

We want to compare the performance of the predict-all-words decoders and that of the autoregressive decoders separate from the RNN/CNN distinction, thus we designed a predict-all-words CNN decoder and RNN decoder.

The predict-all-words CNN decoder is described in Section 2, which is a stack of 3 convolutional layers, and all words are predicted once at the output of the decoder.

The predict-all-words RNN decoder is built based on our CNN decoder.

To keep the number of parameters roughly the same, we replaced the last 2 convolutional layers with a bidirectional GRU.The results are also presented in the Table 1 .

The performance of the predict-all-words RNN decoder does not significantly differ from that of any one of the autoregressive RNN decoders, and the same observation was observed in CNN decoders.

These two findings actually support our choice of using a predict-all-words CNN as the decoder, and it brings the model higher training efficiency and strong transferability.

Since the encoder is a bi-directional RNN in our model, we have multiple ways to select/compute on the generated hidden states to produce a sentence representation.

In Skip-thought and SDAE BID13 , only the hidden state at the last time step produced by the RNN encoder is regarded as the vector representation for a given sentence, which may not be the most expressive vector for representing the input sentence.

We followed the idea proposed in BID6 .

They built a model for supervised SNLI task BID5 that concatenates the outputs from a global mean pooling and a global max pooling to serve as a sentence representation, and showed a performance boost on the SNLI dataset.

Also, found that the model with global max pooling function has stronger transferability than the model with a global mean pooling function after supervised training on SNLI.In our proposed RNN-CNN model, we empirically show that the mean+max pooling provides stronger transferability than the max pooling does, and the results are presented in Table 1 .

The concatenation of a mean-pooling and a max pooling function is actually a parameter-free composition function, and the computation load is negligible compared to heavy matrix multiplications.

Also, the non-linearity of the max pooling function augments the mean pooling function for building a representation that captures a more complex composition of the syntactic information.

We choose to share the parameters in the word embedding layer in RNN encoder and the word prediction layer in CNN decoder.

The tying was proposed in both BID31 and BID16 , and it generally helps to learn a better language model.

In our model, the tying also drastically reduces the number of parameters, which could prevent overfitting.

Furthermore, we initialize the word embeddings with pretrained word vectors, such as word2vec and GloVe BID30 , since it has been shown that these pretrained word vectors can serve as good initialization for deep learning models, and more likely lead to better results than random samples from a uniform distribution.

We studied hyperparameters in our model design based on 3 out of 10 downstream tasks, including SICK-r, SICK-E BID23 , and STS14 BID0 ).

The first model we created, which is reported in Section 2, is a decent design, and the following variations didn't give us much performance change except small improvements with increasing the dimensionality of the encoder.

However, we think it is worth mentioning the effect of hyperparameters in our model design.

We present the Table in the supplementary material and we summarize it as follows:1.

Decoding the next sentence worked similarly as decoding the subsequent contiguous words.2.

Decoding subsequent 30 words, which was adopted from the Skip-thought training code 1 , gave us a reasonable good performance.

More words for decoding didn't give us a significant performance gain, while it took longer to train.3.

Adding more layers into the decoder and enlarging the dimension of the convolutional layers indeed sightly improved the performance on the 3 downstream tasks, but as training efficiency is one of our main concerns, we decided it wasn't worth sacrificing training efficiency for the minor performance improvement.4.

Increasing the dimensionality of the RNN encoder improved the model performance, and the additional training time brought by it was less than that by adding more layers and enlarging the dimension of the convolutional layers in the CNN decoder.

We reported results from both smallest and largest models in Table 2 .

The large corpus we used for unsupervised training is the BookCorpus dataset , which contains 74 million sentences from 7000 books in total.

For stable training, we use ADAM BID18 algorithm for optimization, and gradient clipping BID29 when the norm of gradient exceeds a certain value.

Since we didn't find any significant difference between word2vec and GloVe as initialization in terms of the performance, we stick to using the word vectors from word2vec to initialize the word embedding layer in our models.

The vocabulary for unsupervised training contains the top 20k most frequent words in BookCorpus.

In order to generalize the model trained with a relatively small, fixed vocabulary to the much larger set of all possible English words, Kiros et al. FORMULA1 proposed a word expansion method that learns a linear projection from the pretrained word embeddings word2vec to the learned RNN word embeddings.

Thus, the model benefits from the generalization ability of the pretrained word embeddings.

The downstream tasks for evaluation include semantic relatedness (SICK) BID23 , paraphrase detection (MSRP) BID10 , question-type classification (TREC) BID22 , and 5 benchmark sentiment and subjective datasets, which includes movie review sentiment (MR, SST) BID28 BID33 , customer product reviews (CR) BID15 , subjectivity/objectivity classification (SUBJ) BID27 , opinion polarity (MPQA) BID37 , and semantic textual similarity (STS14) BID0 .

After unsupervised training on the BookCorpus dataset, we fix the parameters in the encoder, and apply it as a sentence representation extractor on the 10 tasks.

In order to compare the effect of different corpora, we also trained 2 models on Amazon Book Review dataset (without ratings) which is the largest subset of the Amazon Review dataset BID24 with 142 million sentences after tokenization, about twice as large as BookCorpus.

Both training and evaluation of our models were conducted in PyTorch 2 , and we used SentEval 3 provided by to evaluate the transferability of models with different settings.

All the models were trained for the same number of iterations with the same batch size, and the performance was measured at the end of training for each of the models.

Table 2 presented the results on 10 evaluation tasks of our proposed RNN-CNN models, and related work.

"small RNN-CNN" refers to the model with the dimension of representation as 1200, and "large RNN-CNN" refers to that as 4800.

The results of our model on SNLI can be found in Table 3 .

Our work was inspired by analyzing the Skip-thought model .

Skip-thought model successfully applied this form of learning from the context information into unsupervised representation learning for sentences, in which the model learns to encode the current sentence and decode the surrounding 2 sentences, and then, augmented the LSTM with proposed layer-normalization (Skip-thought+LN), which improved the skip-thought model generally on all downstream tasks.

Instead of applying RNNs in the model, BID13 proposed the FastSent model which only learns source and target word embeddings, and it is a generalization of CBOW to sentence-level learning, and the composition function over word embeddings is a summation operation.

Later on, applied a CNN as the encoder, which is called the CNN-LSTM model.

The proposed composition model follows the idea of encoding the current sentence and predicting itself and the next sentence; the proposed hierarchical model leverages the context information from both sentence-level and paragraph-level, while learning to encode the current sentence and predict the next one, the model has another RNN to process the sentence representation one at a time at paragraph-level.

Table 2 : Related Work and Comparison.

As presented in the table, our designed asymmetric RNN-CNN model has strong transferability, and is overall better than existing unsupervised models in terms of fast training speed and good performance on evaluation tasks.

The table presents the model comparison.

" †"s refer to our models, and "small/large" refers to the dimension of representation as 1200/4800. " ‡" indicates that DiscSent model was trained with additional data from Wikipedia and the Gutenberg project.

Bold numbers are the best ones among the models with same training and transferring setting, and underlined numbers are best results among all unsupervised representation learning models.

For STS14, the performance measures are Pearson's and Spearman's score.

For MSRP, the performance measures are accuracy and F1 score.

DISPLAYFORM0 Our model falls in the same category as it is an encoder-decoder model.

However, we aim to propose an efficient and effective model.

Instead of decoding the surrounding 2 sentences as in Skip-thought, FastSent and the compositional CNN-LSTM, our model only decodes the subsequent sequence with a fixed length.

Compared with hierarchical CNN-LSTM, our model showed that, with a proper model design, this next-words context information is sufficient in learning sentence representations.

Particularly, our proposed small RNN-CNN model runs roughly 3 times faster than our implemented Skip-thought model on the same GPU machine during training.

Another unsupervised approach is to learn a discriminative model by distinguishing whether a target sentence is in the context of the source sentence, and also the discourse information.

DiscSent BID17 proposed to learn a classifier on top of the representations, which judges 1) whether the two sentences are adjacent to each other, 2) whether the two sentences are in the correct order, and 3) whether the second sentence starts with a conjunction phrase.

DisSent BID26 pointed out that human annotated explicit discourse relations is also good for learning sentence representations.

It is a very promising research direction since the proposed models are generally computational efficient and have clear intuition.

However, the performance on the downstream tasks is still worse than encoder-decoder models.

Proposed by BID32 , BYTE m-LSTM model uses a multiplicative LSTM unit BID20 to learn a language model on Amazon Review data BID24 .

The model works reasonably well on the downstream tasks, since the RNNs are able to produce a distributed representation for the given left-context information, such as a sentence or a document.

In our experiment, we also trained our RNN-CNN model on the Amazon Book review, which is the largest subset of the Amazon review dataset, and indeed, we had a performance gain on all single-sentence classification tasks.

The performance gain in our experiment and also in BYTE m-LSTM was brought by the matching between the corpus domain and the domain of downstream tasks, and it raises 2 questions 1) which corpus is good for learning sentence representations, and 2) whether the downstream tasks are comprehensive to cover sufficient aspects of a sentence.

Previously mentioned models are learned from ordered sentences, but unordered sentences can also be used for learning representations of sentences.

ParagraphVec (Le & Mikolov, 2014) learns a fixed-dimension vector for each sentence by predicting the words within the given sentence.

However, after training, the representation for a new sentence is hard to derive, since it requires optimizing the sentence representation towards an objective.

SDAE BID13 learns the sentence representations with a denoising auto-encoder model.

The noise was added in the encoder by replacing words with a fixed token, and swapping two words, both with a specific probability.

Our proposed RNN-CNN model trains faster than SDAE does, since the CNN decoder runs faster than the RNN decoder in SDAE, and since we utilized the sentence-level continuity as a supervision which SDAE doesn't, our model largely performs better than SDAE.

Unsupervised Transfer Learning Skip-thought (Vendrov et al.) 81.5 large RNN-CNN BookCorpus 81.7 large RNN-CNN Amazon 81.5Supervised Training ESIM (Chen et al.) 86.7 DIIN (Gong et al.) 88.9 Table 3 : We implemented the same classifier as mentioned in BID36 on top of the features computed by our model.

Our proposed RNN-CNN model gets similar result on SNLI as Skip-thought, but with much less training time.

Supervised transfer learning is also promising when we are able to get large enough labeled data.

applied a bi-directional LSTM as the sentence encoder with multiple fully-connected layers to deal with both SNLI BID5 , and MultiNLI BID38 ).

The trained model demonstrates a very impressive transferability on all downstream tasks, including both supervised and unsupervised.

The direct and discriminative training signal pushes the RNN encoder to focus on the semantics of a given sentence, which learns to a boost in performance, and beats all other methods.

Our RNN-CNN model trained on Amazon Book Review data has better results on supervised classification tasks than BiLSTM-Max does, while the performance of ours on semantic relatedness tasks is inferior to BiLSTM-Max.

We argue that labeling a large amount of training data is time-consuming and costly; unsupervised learning could potentially provide a great initial point for human labeling making it less costly and more efficient.

Inspired by learning to exploit the contextual information present in adjacent sentences, we proposed an asymmetric encoder-decoder model with a suite of techniques for improving context-based unsupervised sentence representation learning.

Since we believe that a simple model will be faster in training and easier to analyze, we opt to use simple techniques in our proposed model, including 1) an RNN as the encoder, and a predict-all-words CNN as the decoder, 2) learning by inferring next contiguous words, 3) mean+max pooling, and 4) tying word vectors with word prediction.

With thorough discussion and extensive evaluation, we justify our decision making for each component in our RNN-CNN model.

In terms of the performance and the efficiency of training, we justify that our model is a fast and simple algorithm for learning generic sentence representations from unlabeled corpora.

Further research will focus on how to maximize the utility of the context information, and how to design simple architectures to best make use of it.

9 , and 12) works better than other asymmetric models (CNN-LSTM, row 11), and models with symmetric structure (RNN-RNN, row 5 and 10).

In addition, with larger encoder size, our model demonstrates stronger transferability.

The default setting for our CNN decoder is that it learns to reconstruct 30 words right next to every input sentence.

"CNN(10)" represents a CNN decoder with the length of outputs as 10, and "CNN(50)" represents it with the length of outputs as 50.

" †" indicates that the CNN decoder learns to reconstruct next sentence.

" ‡" indicates the results reported in Gan et al. as future predictor.

The CNN encoder in our experiment, noted as " §", was based on AdaSent in Zhao et al. and Conneau et al..

Bold numbers are best results among models at same dimension, and underlined numbers are best results among all models.

For STS14, the performance measures are Pearson's and Spearman's score.

For MSRP, the performance measures are accuracy and F1 score.

@highlight

We proposed an RNN-CNN encoder-decoder model for fast unsupervised sentence representation learning.

@highlight

Modifications to the skip-thought framework for learning sentence embeddings.

@highlight

This paper presents a new RNN encoder–CNN decoder hybrid design for use in pretraining, which does not require an autoregressive decoder when pretraining encoders.

@highlight

The authors extend Skip-thought by decoding only one target sentence using a CNN decoder.