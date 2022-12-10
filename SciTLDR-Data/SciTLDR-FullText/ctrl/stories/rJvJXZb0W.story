In this work we propose a simple and efficient framework for learning sentence representations from unlabelled data.

Drawing inspiration from the distributional hypothesis and recent work on learning sentence representations, we reformulate the problem of predicting the context in which a sentence appears as a classification problem.

Given a sentence and the context in which it appears, a classifier distinguishes context sentences from other contrastive sentences based on their vector representations.

This allows us to efficiently learn different types of encoding functions, and we show that the model learns high-quality sentence representations.

We demonstrate that our sentence representations outperform state-of-the-art unsupervised and supervised representation learning methods on several downstream NLP tasks that involve understanding sentence semantics while achieving an order of magnitude speedup in training time.

Methods for learning meaningful representations of data have received widespread attention in recent years.

It has become common practice to exploit these representations trained on large corpora for downstream tasks since they capture a lot of prior knowlege about the domain of interest and lead to improved performance.

This is especially attractive in a transfer learning setting where only a small amount of labelled data is available for supervision.

Unsupervised learning allows us to learn useful representations from large unlabelled corpora.

The idea of self-supervision has recently become popular where representations are learned by designing learning objectives that exploit labels that are freely available with the data.

Tasks such as predicting the relative spatial location of nearby image patches BID6 , inpainting BID30 and solving image jigsaw puzzles BID27 have been successfully used for learning visual feature representations.

In the language domain, the distributional hypothesis has been integral in the development of learning methods for obtaining semantic vector representations of words BID24 .

This is the assumption that the meaning of a word is characterized by the word-contexts in which it appears.

Neural approaches based on this assumption have been successful at learning high quality representations from large text corpora.

Recent methods have applied similar ideas for learning sentence representations Hill et al., 2016; BID8 .

These are encoder-decoder models that learn to predict/reconstruct the context sentences of a given sentence.

Despite their success, several modelling issues exist in these methods.

There are numerous ways of expressing an idea in the form of a sentence.

The ideal semantic representation is insensitive to the form in which meaning is expressed.

Existing models are trained to reconstruct the surface form of a sentence, which forces the model to not only predict its semantics, but aspects that are irrelevant to the meaning of the sentence as well.

The other problem associated with these models is computational cost.

These methods have a word level reconstruction objective that involves sequentially decoding the words of target sentences.

Training with an output softmax layer over the entire vocabulary is a significant source of slowdown in the training process.

This further limits the size of the vocabulary and the model (Variations of the softmax layer such as hierarchical softmax BID26 , sampling based softmax BID9 and sub-word representations BID33 can help alleviate this issue).We circumvent these problems by proposing an objective that operates directly in the space of sentence embeddings.

The generation objective is replaced by a discriminative approximation where the model attempts to identify the embedding of a correct target sentence given a set of sentence candidates.

In this context, we interpret the 'meaning' of a sentence as the information in a sentence that allows it to predict and be predictable from the information in context sentences.

We name our approach quick thoughts (QT), to mean efficient learning of thought vectors.

Our key contributions in this work are the following:• We propose a simple and general framework for learning sentence representations efficiently.

We train widely used encoder architectures an order of magnitude faster than previous methods, achieving better performance at the same time.• We establish a new state-of-the-art for unsupervised sentence representation learning methods across several downstream tasks that involve understanding sentence semantics.

The pre-trained encoders will be made publicly available.

We discuss prior approaches to learning sentence representations from labelled and unlabelled data.

Learning from Unlabelled corpora.

BID18 proposed the paragraph vector (PV) model to embed variable-length text.

Models are trained to predict a word given it's context or words appearing in a small window based on a vector representation of the source document.

Unlike most other methods, in this work sentences are considered as atomic units instead of as a compositional function of its words.

Encoder-decoder models have been successful at learning semantic representations.

proposed the skip-thought vectors model, which consists of an encoder RNN that produces a vector representation of the source sentence and a decoder RNN that sequentially predicts the words of adjacent sentences.

Drawing inspiration from this model, BID8 explore the use of convolutional neural network (CNN) encoders.

The base model uses a CNN encoder and reconstructs the input sentence as well as neighboring sentences using an RNN.

They also consider a hierarchical version of the model which sequentially reconstructs sentences within a larger context.

Autoencoder models have been explored for representation learning in a wide variety of data domains.

An advantage of autoencoders over context prediction models is that they do not require ordered sentences for learning.

BID34 proposed recursive autoencoders which encode an input sentence using a recursive encoder and a decoder reconstructs the hidden states of the encoder.

Hill et al. (2016) considered a de-noising autoencoder model (SDAE) where noise is introduced in a sentence by deleting words and swapping bigrams and the decoder is required to reconstruct the original sentence.

BID3 proposed a generative model of sentences based on a variational autoencoder.

BID13 learn bag-of-words (BoW) representations of sentences by considering a conceptually similar task of identifying context sentences from candidates and evaluate their representations on sentence similarity tasks.

Hill et al. (2016) introduced the FastSent model which uses a BoW representation of the input sentence and predicts the words appearing in context (and optionally, the source) sentences.

The model is trained to predict whether a word appears in the target sentences.

BID2 consider a weighted BoW model followed by simple post-processing and show that it performs better than BoW models trained on paraphrase data.

BID10 use paragraph level coherence as a learning signal to learn representations.

The following related task is considered in their work.

Given the first three sentences of a paragraph, choose the next sentence from five sentences later in the paragraph.

Related to our objective is the local coherence model of BID19 where a binary classifier is trained to identify coherent/incoherent sentence windows.

In contrast, we only encourage observed contexts to be more plausible than contrastive ones and formulate it as a multi-class classification problem.

We experimentally found that this relaxed constraint helps learn better representations.

Encoder-decoder based sequence models are known to work well, but they are slow to train on large amounts of data.

On the other hand, bag-of-words models train efficiently by ignoring word order.

Spring had come.

And yet his crops didn't grow.

Spring had come.

They were so black.

And yet his crops didn't grow.

He had blue eyes.

Enc (g) Classifier We incorporate the best of both worlds by retaining flexibility of the encoder architecture, while still being able to to train efficiently.

There have been attempts to use labeled/structured data to learn sentence representations.

Hill et al. (2016) learn to map words to their dictionary definitions using a max margin loss that encourages the encoded representation of a definition to be similar to the corresponding word.

BID41 and use paraphrase data to learn an encoder that maps synonymous phrases to similar embeddings using a margin loss.

Hermann & Blunsom (2013) consider a similar objective of minimizing the inner product between paired sentences in different languages.

explore the use of machine translation to obtain more paraphrase data via back-translation and use it for learning paraphrastic embeddings.

BID5 consider the supervised task of Natural language inference (NLI) as a means of learning generic sentence representations.

The task involves identifying one of three relationships between two given sentences -entailment, neutral and contradiction.

The training strategy consists of learning a classifier on top of the embeddings of the input pair of sentences.

The authors show that sentence encoders trained for this task perform strongly on downstream transfer tasks.

The distributional hypothesis has been operationalized by prior work in different ways.

A common approach is illustrated in FIG0 , where an encoding function computes a vector representation of an input sentence, and then a decoding function attempts to generate the words of a target sentence conditioned on this representation.

In the skip-thought model, the target sentences are those that appear in the neighborhood of the input sentence.

There have been variations on the decoder such as autoencoder models which predict the input sentence instead of neighboring sentences (Hill et al., 2016) and predicting properties of a window of words in the input sentence BID18 .Instead of training a model to reconstruct the surface form of the input sentence or its neighbors, we take the following approach.

Use the meaning of the current sentence to predict the meanings of adjacent sentences, where meaning is represented by an embedding of the sentence computed from an encoding function.

Despite the simplicity of the modeling approach, we show that it facilitates learning rich representations.

Our approach is illustrated in FIG0 .

Given an input sentence, it is encoded as before using some function.

But instead of generating the target sentence, the model chooses the correct target sentence from a set of candidate sentences.

Viewing generation as choosing a sentence from all possible sentences, this can be seen as a discriminative approximation to the generation problem.

A key difference between these two approaches is that in FIG0 , the model can choose to ignore aspects of the sentence that are irrelevant in constructing a semantic embedding space.

Loss functions defined in a feature space as opposed to the raw data space have been found to be more attractive in recent work for similar reasons BID17 BID31 .Formally described, let f and g be parametrized functions that take a sentence as input and encode it into a fixed length vector.

Let s be a given sentence.

Let S ctxt be the set of sentences appearing in the context of s (for a particular context size) in the training data.

Let S cand be the set of candidate sentences considered for a given context sentence s ctxt ∈ S ctxt .

In other words, S cand contains a valid context sentence s ctxt (ground truth) and many other non-context sentences, and is used for the classification objective as described below.

For a given sentence position in the context of s (e.g., the next sentence), the probability that a candidate sentence s cand ∈ S cand is the correct sentence (i.e., appearing in the context of s) for that position is given by DISPLAYFORM0 where c is a scoring function/classifier.

The training objective maximizes the probability of identifying the correct context sentences for each sentence in the training data D. DISPLAYFORM1 The modeling approach encapsulates the Skip-gram approach of BID24 when words play the role of sentences.

In this case the encoding functions are simple lookup tables considering words to be atomic units, and the training objective maximizes the similarity between the source word and a target word in its context given a set of negative samples.

Alternatively, we considered an objective function similar to the negative sampling approach of BID24 .

This takes the form of a binary classifier which takes a sentence window as input and classifies them as plausible and implausible context windows.

We found objective (2) to work better, presumably due to the relaxed constraint it imposes.

Instead of requiring context windows to be classified as positive/negative, it only requires ground-truth contexts to be more plausible than contrastive contexts.

This objective also performed empirically better than a maxmargin loss.

In our experiments, c is simply defined to be an inner product c(u, v) = u T v. This was motivated by considering pathological solutions where the model learns poor sentence encoders and a rich classifier to compensate for it.

This is undesirable since the classifier will be discarded and only the sentence encoders will be used to extract features for downstream tasks.

Minimizing the number of parameters in the classifier encourages the encoders to learn disentangled and useful representations.

We consider f , g to have different parameters, although they were motivated from the perspective of modeling sentence meaning.

Another motivation comes from word representation learning methods which use different sets of input and output parameters.

Parameter sharing is further not a significant concern since these models are trained on large corpora.

At test time, for a given sentence s we consider its representation to be the concatenation of the outputs of the two encoders [f (s) g(s)].Our framework allows flexible encoding functions to be used.

We use RNNs as f and g as they have been widely used in recent sentence representation learning methods.

The words of the sentence are sequentially fed as input to the RNN and the final hidden state is interpreted as a representation of the sentence.

We use gated recurrent units (GRU) BID4 as the RNN cell similar to .

We evaluate our sentence representations by using them as feature representations for downstream NLP tasks.

Alternative fine-grained evaluation tasks such as identifying word appearance and word order were proposed in BID0 .

Although this provides some useful insight about the representations, these tasks focus on the syntactic aspects of a sentence.

We are more interested in assessing how well representations capture sentence semantics.

Although limitations of these evaluations have been pointed out, we stick to the traditional approach of evaluating using downstream tasks.

Models were trained on the 7000 novels of the BookCorpus dataset .

The dataset consists of about 45M ordered sentences.

We also consider a larger corpus for training: the UMBC corpus (Han et al., 2013) , a dataset of 100M web pages crawled from the internet, preprocessed and tokenized into paragraphs.

The dataset has 129M sentences, about three times larger than BookCorpus.

For models trained from scratch, we used case-sensitive vocabularies of sizes 50k and 100k for the two datasets respectively.

A minibatch is constructed using a contiguous sets of sentences in the corpus.

For each sentence, all the sentences in the minibatch are considered to be the candidate pool S cand of sentences for classification.

This simple scheme for picking contrastive sentences performed as well as other schemes such as random sampling and picking nearest neighbors of the input sentence.

Hyperparameters including batch size, learning rate, prediction context size were obtained using prediction accuracies (accuracy of predicting context sentences) on the validation set.

A context size of 3 was used, i.e., predicting the previous and next sentences given the current sentence.

We used a batch size of 400 and learning rate of 5e-4 with the Adam optimizer for all experiments.

All our RNN-based models are single-layered and use GRU cells.

Weights of the GRU are initialized using uniform Xavier initialization and gate biases are initialized to 1.

Word embeddings are initialized from U [−0.1, 0.1].

Tasks We evaluate the sentence representations on tasks that require understanding sentence semantics.

The following classification benchmarks are commonly used: movie review sentiment (MR) BID29 , product reviews (CR) (Hu & Liu, 2004) , subjectivity classification (SUBJ) BID28 , opinion polarity (MPQA) BID39 , question type classification (TREC) BID38 and paraphrase identification (MSRP) BID7 ).

The semantic relatedness task on the SICK dataset BID22 involves predicting relatedness scores for a given pair of sentences that correlate well with human judgements.

The MR, CR, SUBJ, MPQA tasks are binary classification tasks.

10-fold cross validation is used in reporting test performance for these tasks.

The other tasks come with train/dev/test splits and the dev set is used for choosing the regularization parameter.

We follow the evaluation scheme of where feature representations of sentences are obtained from the trained encoders and a logistic/softmax classifier is trained on top of the embeddings for each task while keeping the sentence embeddings fixed.

Kiros et al.'s scripts are used for evaluation.

Table 1 compares our work against representations from prior methods that learn from unlabelled data.

The dimensionality of sentence representations and training time are also indicated.

For our RNN based encoder we consider variations that are analogous to the skip-thought model.

The uni-QT model uses uni-directional RNNs as the sentence encoders f and g. In the bi-QT model, the concatenation of the final hidden states of two RNNs represent f and g, each processing the sentence in a different (forward/backward) direction.

The combine-QT model concatenates the representations (at test time) learned by the uni-QT and bi-QT models.

Models trained from scratch on BookCorpus.

While the FastSent model is efficient to train (training time of 2h), this efficiency stems from using a bag-of-words encoder.

Bag of words provides a strong baseline because of its ability to preserves word identity information.

However, the model performs poorly compared to most of the other methods.

Bag-of-words is also conceptually less attractive as a representation scheme since it ignores word order, which is a key aspect of meaning.

The de-noising autoencoder (SDAE) performs strongly on the paraphrase detection task (MSRP).

This is attributable to the reconstruction (autoencoding) loss which encourages word identity and order information to be encoded in the representation.

However, it fails to perform well in other tasks that require higher level sentence understanding and is also inefficient to train.

Our uni/bi/combine-QT variations perform comparably (and in most cases, better) to the skipthought model and the CNN-based variation of BID8 in all tasks despite requiring much less training time.

Since these models were trained from scratch, this also shows that the model learns good word representations as well.

MultiChannel-QT.

Next, we consider using pre-trained word vectors to train the model.

The MultiChannel-QT model (MC-QT) is defined as the concatenation of two bi-directional RNNs.

One of these uses fixed pre-trained word embeddings coming from a large vocabulary (∼ 3M) as input.

While the other uses tunable word embeddings trained from scratch (from a smaller vocabulary ∼ 50k).

This model was inspired by the multi-channel CNN model of BID14 which considered two sets of embeddings.

With different input representations, the two models discover less redundant features, as opposed to the uni and bi variations suggested in .

We use GloVe vectors BID32 as pre-trained word embeddings.

The MC-QT model outperforms all previous methods, including the variation of BID8 which uses pre-trained word embeddings.

UMBC data.

Because our framework is efficient to train, we also experimented on a larger dataset of documents.

Results for models trained on BookCorpus and UMBC corpus pooled together (∼ 174M sentences) are shown at the bottom of the table.

We observe strict improvements on a majority of the tasks compared to our BookCorpus models.

This shows that we can exploit huge corpora to obtain better models while keeping the training time practically feasible.

Computational efficiency.

Our models are implemented in Tensorflow.

Experiments were performed using cuda 8.0 and cuDNN 6.0 libraries on a GTX Titan X GPU.

Our best BookCorpus model (MC-QT) trains in just under 11hrs (On both the Titan X and GTX 1080).

Training time for the skip-thoughts model is mentioned as 2 weeks in and a more recent Tensorflow implementation 1 reports a training time of 9 days on a GTX 1080.

On the augmented dataset our models take about a day to train, and we observe monotonic improvements in all tasks except the TREC task.

Our framework allows training with much larger vocabulary sizes than most previous models.

Our approach is also memory efficient.

The paragraph vector model has a big memory footprint since it has to store vectors of documents used for training.

Softmax computations over the vocabulary in the skip-thought and other models with word-level reconstruction objectives incur heavy memory consumption.

Our RNN based implementation (with the indicated hyperparamters and batch size) fits within 3GB of GPU memory, a majority of it consumed by the word embeddings.

MR Table 3 : Comparison against task-specific supervised models.

The models are AdaSent BID44 , CNN BID14 , TF-KLD BID11 and Dependency-Tree LSTM BID36 .

Note that our performance values correspond to a linear classifier trained on fixed pre-trained embeddings, while the task-specific methods are tuned end-to-end.

TAB3 compares our approach against methods that learn from labelled/structured data.

The CaptionRep, DictRep and NMT models are from Hill et al. (2016) which are trained respectively on the tasks of matching images and captions, mapping words to their dictionary definitions and machine translation.

The InferSent model of BID5 is trained on the NLI task.

In addition to the benchmarks considered before, we additionally also include the sentiment analysis binary classification task on Stanford Sentiment Treebank (SST) BID35 ).

DISPLAYFORM0 The Infersent model has strong performance on the tasks.

Our multichannel model trained on the (BookCorpus + UMBC) data outperforms InferSent in most of the tasks, with most significant margins in the SST and TREC tasks.

Infersent is strong in the SICK task presumably due to the following reasons.

The model gets to observes near paraphrases (entailment relationship) and sentences that are not-paraphrases (contradiction relationship) at training time.

Furthermore, it considers difference features (|u − v|) and multiplicative features (u * v) of the input pair of sentences u, v during training.

This is identical to the feature transformations used in the SICK evaluation as well.

Ensemble We consider ensembling to exploit the strengths of different types of encoders.

Since our models are efficient to train, we are able to feasibly train many models.

We consider a subset of the following model variations for the ensemble.• Table 4 : Image-caption retrieval.

The purely supervised models are respectively from BID12 , BID16 , BID21 and BID37 .

Best pre-trained representations and best task-specific methods are highlighted.

Models are combined using a weighted average of the predicted log-probabilities of individual models, the weights being normalized validation set performance scores.

Results are presented in table 3.

Performance of the best purely supervised task-specific methods are shown at the bottom for reference.

Note that these numbers are not directly comparable with the unsupervised methods since the sentence embeddings are not fine-tuned.

We observe that the ensemble model closely approaches the performance of the best supervised task-specific methods, outperforming them in 3 out of the 8 tasks.

The image-to-caption and caption-to-image retrieval tasks have been commonly used to evaluate sentence representations in a multi-modal setting.

The task requires retrieving an image matching a given text description and vice versa.

The evaluation setting is identical to .

Images and captions are represented as vectors.

Given a matching image-caption pair (I, C) a scoring function f determines the compatibility of the corresponding vector representations v I , v C .

The scoring function is trained using a margin loss which encourages matching pairs to have higher compatibility than mismatching pairs.

DISPLAYFORM0 As in prior work, we use VGG-Net features (4096-dimensional) as the image representation.

Sentences are represented as vectors using the representation learning method to be evaluated.

These representations are held fixed during training.

The scoring function used in prior work is f (x, y) = (U x) T (V y) where U, V are projection matrices which project down the image and sentence vectors to the same dimensionality.

The MSCOCO dataset BID20 has been traditionally used for this task.

We use the train/val/test split proposed in BID12 .

The training, validation and test sets respectively consist of 113,287, 5000, 5000 images, each annotated with 5 captions.

Performance is reported as an average over 5 splits of 1000 image-caption pairs each from the test set.

Results are presented in table 3.

We outperform previous unsupervised pre-training methods by a significant margin, strictly improving the median retrieval rank for both the annotation and search tasks.

We also outperform some of the purely supervised task specific methods by some metrics.

Our model and the skip-thought model have conceptually similar objective functions.

This suggests examining properties of the embedding spaces to better understand how they encode semantics.

We consider a nearest neighbor retrieval experiment to compare the embedding spaces.

We use a pool of 1M sentences from a Wikipedia dump for this experiment.

For a given query sentence, the best neighbor determined by cosine distance in the embedding space is retrieved.

TAB5 shows a random sample of query sentences from the dataset and the corresponding retrieved sentences.

These examples show that our retrievals are often more related to the query sentence compared to the skip-thought model.

It is interesting to see in the first example that the model identifies a sentence with similar meaning even though the main clause and conditional clause are in a different order.

This is in line with our goal of learning representations that are less sensitive to the form in which meaning is expressed.

Query Seizures may occur as the glucose falls further .

It may also occur during an excessively rapid entry into autorotation .

QT When brain glucose levels are sufficiently low , seizures may result .

Query This evidence was only made public after both enquiries were completed .

This visa was provided for under Republic Act No .

QT These evidence were made public by the United States but concealed the names of sources .

Query He kept both medals in a biscuit tin .

He kept wicket for Middlesex in two first-class cricket matches during the 1891 County Championship .

He won a three medals at four Winter Olympics .

Query The American alligator is the only known natural predator of the panther .

Their mascot is the panther .

The American alligator is a fairly large species of crocodilian .

Query Several of them died prematurely : Carmen and Toms very young , while Carlos and Pablo both died .

At the age of 13 , Ahmed Sher died .

Many of them died in prison .

Query Music for " Expo 2068 " originated from the same studio session .

His 1994 work " Dialogue " was premiered at the Merkin Concert Hall in New York City .

Music from " Korra " and " Avatar " was also played in concert at the PlayFest festival in Mlaga , Spain in September 2014 .

Query Mohammad Ali Jinnah yielded to the demands of refugees from the Indian states of Bihar and Uttar Pradesh , who insisted that Urdu be Pakistan 's official language .

ST Georges Charachidz , a historian and linguist of Georgian origin under Dumzil 's tutelage , became a noted specialist of the Caucasian cultures and aided Dumzil in the reconstruction of the Ubykh language .

QT Wali Mohammed Wali 's visit thus stimulated the growth and development of Urdu Ghazal in Delhi .

Query The PCC , together with the retrosplenial cortex , forms the retrosplenial gyrus .

The Macro domain from human , macroH2A1.1 , binds an NAD metabolite O-acetyl-ADP-ribose .

The PCC forms a part of the posteromedial cortex , along with the retrosplenial cortex ( Brodmann areas 29 and 30 ) and precuneus ( located posterior and superior to the PCC ) .

Query With the exception of what are known as the Douglas Treaties , negotiated by Sir James Douglas with the native people of the Victoria area , no treaties were signed in British Columbia until 1998 .

All the assets of the Natal Railway Company , including its locomotive fleet of three , were purchased for the sum of 40,000 by the Natal Colonial Government in 1876 .

With few exceptions ( the Douglas Treaties of Fort Rupert and southern Vancouver Island ) no treaties were signed .

We proposed a framework to learn generic sentence representations efficiently from large unlabelled text corpora.

Our simple approach learns richer representations than prior unsupervised and supervised methods, consuming an order of magnitude less training time.

We establish a new state-of-the-art for unsupervised sentence representation learning methods on several downstream tasks.

We believe that exploring scalable approaches to learn data representations is key to exploit unlabelled data available in abundance.

In this experiment we compare the ability of our model and skip-thought vectors to reason about analogies in the sentence embedding space.

The analogy task has been widely used for evaluating word representations.

The task involves answering questions of the type A : B :: C :? where the answer word shares a relationship to word C that is identical to the relationship between words A and B. We consider an analogous task at the sentence level and formulate it as a retrieval task where the query vector v(C) + v(B) − v(A) is used to identify the closest sentence vector v(D) from a pool of candidates.

This evaluation favors models that produce meaningful dimensions.

Guu et al. (2017) exploit word analogy datasets to construct sentence tuples with analogical relationships.

They mine sentence pairs (s 1 , s 2 ) from the Yelp dataset (Yelp, 2017) which approximately differ by a single word, and use these pairs to construct sentence analogy tuples based on known word analogy tuples.

The dataset has 1300 tuples of sentences collected in this fashion.

For each sentence tuple we derive 4 questions by considering three of the sentences to form the query vector.

The candidate pool for sentence retrieval consists of all sentences in this dataset and 1M other sentences from the Yelp dataset.

TAB7 compares the retrieval performance of our representations and skip-thought vectors on the above task.

Results are classified under word-pair categories in the Google and Microsoft word analogy datasets BID23 BID20 .

Our model outperforms skip-thoughts across several categories and has good performance in the family and verb transformation categories .

TAB8 shows some qualitative retrieval results.

Each row of the table shows three sentences that form the query and the answer identified by the model.

The last row shows an example where the model fails.

This is a common failure case of both methods where the model assumes that A and B are identical in a question A : B :: C :? and retrieves sentence C as the answer.

These experiments show that the our representations possess better linearity properties.

The transformations evaluated here are mostly syntactic transformations involving a few words.

It would be interesting to explore other high-level transformations such as switching sentiment polarity and analogical relationships that involve several words in future work.

In this section we assess the representations learned by our encoders on semantic similarity tasks.

The STS14 datasets BID1 consist of pairs of sentences annotated by humans with similarity scores.

Representations are evaluated by measuring the correlation between human judgments and the cosine similarity of vector representations for a given pair of sentences.

We consider two types of encoders trained using our objective -RNN encoders and BoW encoders.

Models were trained from scratch on the BookCorpus data.

The RNN version is the same as the combine-QT model in Table 1 .

We describe the BoW encoder training below.

We train a BoW encoder using our training objective.

Hyperparameter choices for the embedding size ({100, 300, 500}), number of contrastive sentences ({500, 1000, 1500, 2000}) and context size ({3, 5, 7}) were made based on the validation set (optimal choices highlighted in bold).

Training this model on the BookCorpus dataset takes 2 hours on a Titan X GPU.

Similar to the RNN encoders, the representation of a sentence is obtained by concatenating the outputs of the input and output sentence encoders.

TAB10 compares different unsupervised representation learning methods trained on the BookCorpus data from scratch.

Methods are categorized as sequence models and bag-of-words models.

Our RNN-based encoder performs strongly compared to other sequence encoders.

Bag-of-words models are known to perform strongly in this task as they are better able to encode word identity information.

Our BoW variation performs comparably to prior BoW based models.

and Siamese CBOW BID13 .

QT (RNN) and QT (BoW) are our models trained with RNN and BoW encoders, respectively.

To better assess the training efficiency of our models, we perform the following experiment.

We train the same encoder architecture using our objective and the skip-thought (ST) objective and compare the performance after a certain number of hours of training.

Since training the ST objective with large embedding sizes takes many days, we consider a lower dimensional sentence encoder for this experiment.

We chose the encoder architecture to be a single-layer GRU Recurrent neural net with hidden state size H = 1000.

The word embedding size was set to W = 300 and a vocabulary size of V = 20, 000 words was used.

Both models are initialized randomly from the same distribution.

The models are trained on the same data for 1 epoch using the Adam optimizer with learning rate 5e-4 and batch size 400.

For the low dimensional model considered, the model trained with our objective and ST objective take 6.5 hrs and 31 hrs, respectively.

The number of parameters for the two objectives are Only the input side encoder parameters (≈ 9.9M parameters) are used for the evaluation.

The 1000-dimensional sentence embeddings are used for evaluation.

Evaluation follows the same protocol as in section 4.4.

FIG4 compares the performance of the two models on downstream tasks after x number of training hours.

The speed benefits of our training objective is apparent from these comparisons.

The overall training speedup observed for our objective is 4.8x.

Note that the output encoder was discarded for our model, unlike the experiments in the main text where the representations from the input and output encoders are concatenated.

Further speedups can be achieved by training with encoders half the size and concatenating them (This is also parameter efficient).

We explore the trade-off between training efficiency and the quality of representations by varying the representation size.

We trained models with different representation sizes and evaluate them on the downstream tasks.

The multi-channel model (MC-QT) was used for these experiments.

Models were trained on the BookCorpus dataset.

Table 9 shows the training time and the performance corresponding to different embedding sizes.

The training times listed here assume that the two component models in MC-QT are trained in parallel.

The reported performance is an average over all the classification benchmarks (MSRP, TREC, MR, CR, SUBJ, MPQA).

We note that the classifiers trained on top of the embeddings for downstream tasks differ in size for each embedding size.

So it is difficult to make any strong conclusions about the quality of embeddings for the different sizes.

However, we are able to reduce the embedding size and train the models more efficiently, at the expense of marginal loss in performance in most cases.

The 4800-dimensional Skip-thought model and Combine-CNN model BID8 achieve mean accuracies of 83.75 and 85.33 respectively.

We note that our 1600-dimensional model and 3200-dimensional model are respectively better than these models, in terms of the mean performance across the benchmarks (We acknowledge that the Skip-thought model did not use pre-trained word embeddings).

This suggests that high-quality models can be obtained even more efficiently by training lower-dimensional models on large amounts of data using our objective.

Table 9 : Training time and performance for different embedding sizes.

The reported performance is the mean accuracy over the classification benchmarks (MSRP, TREC, MR, CR, SUBJ, MPQA).

<|TLDR|>

@highlight

A framework for learning high-quality sentence representations efficiently.

@highlight

Proposes a faster algorithm for learning SkipThought-style sentence representations from corpora of ordered sentences that swaps the word-level decoder for a contrastive classification loss.

@highlight

This paper proposes a framework for unsupervised learning of sentence representations by maximizing a model of the probability of true context sentences relative to random candidate sentences