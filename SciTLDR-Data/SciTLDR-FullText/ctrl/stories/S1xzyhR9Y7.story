Multi-view learning can provide self-supervision when different views are available of the same data.

Distributional hypothesis provides another form of useful self-supervision from adjacent sentences which are plentiful in large unlabelled corpora.

Motivated by the asymmetry in the two hemispheres of the human brain as well as the observation that different learning architectures tend to emphasise different aspects of sentence meaning, we present two multi-view frameworks for learning sentence representations in an unsupervised fashion.

One framework uses a generative objective and the other a discriminative one.

In both frameworks, the final representation is an ensemble of two views, in which, one view encodes the input sentence with a Recurrent Neural Network (RNN), and the other view encodes it with a simple linear model.

We show that, after learning, the vectors produced by our multi-view frameworks provide improved representations over their single-view learnt counterparts, and the combination of different views gives representational improvement over each view and demonstrates solid transferability on standard downstream tasks.

Multi-view learning methods provide the ability to extract information from different views of the data and enable self-supervised learning of useful features for future prediction when annotated data is not available BID16 .

Minimising the disagreement among multiple views helps the model to learn rich feature representations of the data and, also after learning, the ensemble of the feature vectors from multiple views can provide an even stronger generalisation ability.

Distributional hypothesis BID22 noted that words that occur in similar contexts tend to have similar meaning BID51 , and distributional similarity BID19 consolidated this idea by stating that the meaning of a word can be determined by the company it has.

The hypothesis has been widely used in machine learning community to learn vector representations of human languages.

Models built upon distributional similarity don't explicitly require humanannotated training data; the supervision comes from the semantic continuity of the language data.

Large quantities of annotated data are usually hard and costly to obtain, thus it is important to study unsupervised and self-supervised learning.

Our goal is to propose learning algorithms built upon the ideas of multi-view learning and distributional hypothesis to learn from unlabelled data.

We draw inspiration from the lateralisation and asymmetry in information processing of the two hemispheres of the human brain where, for most adults, sequential processing dominates the left hemisphere, and the right hemisphere has a focus on parallel processing BID9 , but both hemispheres have been shown to have roles in literal and non-literal language comprehension BID15 BID14 .Our proposed multi-view frameworks aim to leverage the functionality of both RNN-based models, which have been widely applied in sentiment analysis tasks BID57 , and the linear/loglinear models, which have excelled at capturing attributional similarities of words and sentences BID5 BID24 BID51 for learning sentence representations.

Previous work on unsupervised sentence representation learning based on distributional hypothesis can be roughly categorised into two types:Generative objective: These models generally follow the encoder-decoder structure.

The encoder learns to produce a vector representation for the current input, and the decoder learns to generate sentences in the adjacent context given the produced vector BID24 BID20 BID50 .

The idea is straightforward, yet its scalability for very large corpora is hindered by the slow decoding process that dominates training time, and also the decoder in each model is discarded after learning as the quality of generated sequences is not the main concern, which is a waste of parameters and learning effort.

Our first multi-view framework has a generative objective and uses an RNN as the encoder and an invertible linear projection as the decoder.

The training time is drastically reduced as the decoder is simple, and the decoder is also utilised after learning.

A regularisation is applied on the linear decoder to enforce invertibility, so that after learning, the inverse of the decoder can be applied as a linear encoder in addition to the RNN encoder.

Discriminative Objective: In these models, a classifier is learnt on top of the encoders to distinguish adjacent sentences from those that are not BID31 BID26 BID40 BID33 ; these models make a prediction using a predefined differentiable similarity function on the representations of the input sentence pairs or triplets.

Our second multi-view framework has a discriminative objective and uses an RNN encoder and a linear encoder; it learns to maximise agreement among adjacent sentences.

Compared to earlier work on multi-view learning BID16 BID17 BID52 that takes data from various sources or splits data into disjoint populations, our framework processes the exact same data in two distinctive ways.

The two distinctive information processing views tend to encode different aspects of an input sentence; forcing agreement/alignment between these views encourages each view to be a better representation, and is beneficial to the future use of the learnt representations.

Our contribution is threefold:??? Two multi-view frameworks for learning sentence representations are proposed, in which one framework uses a generative objective and the other one adopts a discriminative objective.

Two encoding functions, an RNN and a linear model, are learnt in both frameworks.??? The results show that in both frameworks, aligning representations from two views gives improved performance of each individual view on all evaluation tasks compared to their single-view trained counterparts, and furthermore ensures that the ensemble of two views provides even better results than each improved view alone.??? Models trained under our proposed frameworks achieve good performance on the unsupervised tasks, and overall outperform existing unsupervised learning models, and armed with various pooling functions, they also show solid results on supervised tasks, which are either comparable to or better than those of the best unsupervised transfer model.

It is shown BID24 that the consistency between supervised and unsupervised evaluation tasks is much lower than that within either supervised or unsupervised evaluation tasks alone and that a model that performs well on supervised evaluation tasks may fail on unsupervised tasks.

It is subsequently showed BID13 BID48 ) that, with large-scale labelled training corpora, the resulting representations of the sentences from the trained model excel in both supervised and unsupervised tasks, while the labelling process is costly.

Our model is able to achieve good results on both groups of tasks without labelled information.

Our goal is to marry RNN-based sentence encoder and the avg-on-word-vectors sentence encoder into multi-view frameworks with simple objectives.

The motivation for the idea is that, RNN-based encoders process the sentences sequentially, and are able to capture complex syntactic interactions, while the avg-on-word-vectors encoder has been shown to be good at capturing the coarse meaning of a sentence which could be useful for finding paradigmatic parallels BID51 .We present two multi-view frameworks, each of which learns two different sentence encoders; after learning, the vectors produced from two encoders of the same input sentence are used to compose the sentence representation.

The details of our learning frameworks are described as follows:

In our multi-view frameworks, we first introduce two encoders that, after learning, can be used to build sentence representations.

One encoder is a bi-directional Gated Recurrent Unit BID10 f (s; ??), where s is the input sentence and ?? is the parameter vector in the GRU.

During learning, only hidden state at the last time step is sent to the next stage in learning.

The other encoder is a linear avg-on-word-vectors model g(s; W ), which basically transforms word vectors in a sentence by a learnable weight matrix W and outputs an averaged vector.

Given the finding BID50 ) that neither an autoregressive nor an RNN decoder is necessary for learning sentence representations that excel on downstream tasks, our learning framework only learns to predict words in the next sentence.

The framework has an RNN encoder f , and a linear decoder h. Given an input sentence s i , the encoder produces a vector z f i = f (s i ; ??), and the decoder h projects the vector to DISPLAYFORM0 , which has the same dimension as the word vectors v w .

Negative sampling is applied to calculate the likelihood of generating the j-th word in the (i + 1)-th sentence, shown in Eq. 1.

DISPLAYFORM1 where v w k are pretrained word vectors for w k , the empirical distribution P e (w) is the unigram distribution raised to power 0.75 , and K is the number of negative samples.

The learning objective is to maximise the likelihood for words in all sentences in the training corpus.

Ideally, the inverse of h should be easy to compute so that during testing we can set g = h ???1 .

As h is a linear projection, the simplest situation is when U is an orthogonal matrix and its inverse is equal to its transpose.

Often, as the dimensionality of vector z f i doesn't necessarily need to match that of word vectors v w , U is not a square matrix 1 .

To enforce invertibility on U , a row-wise orthonormal regularisation on U is applied during training, which leads to U U = I, where I is the identity matrix, thus the inverse function is simply h ???1 (x) = U x, which is easily computed.

The regularisation formula is ||U U ??? I|| F , where || ?? || F is the Frobenius norm.

Specifically, the update rule BID11 for the regularisation is: DISPLAYFORM2 where ?? is set to 0.01.

After learning, we set W = U , then the inverse of the decoder h becomes the encoder g. Compared to prior work with generative objective, our framework reuses the decoding function rather than ignoring it for building sentence representations after learning, thus information encoded in the decoder is also utilised.

Our multi-view framework with discriminative objective learns to maximise the agreement between the representations of a sentence pair across two views if one sentence in the pair is in the neighbourhood of the other one.

An RNN encoder f (s; ??) and a linear avg-on-word-vectors g(s; W ) produce a vector representation z f i and z g i for i-th sentence respectively.

The agreement between two views of a sentence pair (s i , s j ) is defined as a ij = a ji = cos(z DISPLAYFORM0 The training objective is to minimise the loss function: DISPLAYFORM1 where DISPLAYFORM2 where ?? is the trainable temperature term, which is essential for exaggerating the difference between adjacent sentences and those that are not.

The neighbourhood/context window c, and the batch size N are hyperparameters.

The choice of cosine similarity based loss is based on the observations BID51 that, of word vectors derived from distributional similarity, vector length tends to correlate with frequency of words, thus angular distance captures more important meaning-related information.

Also, since our model is unsupervised/self-supervised, whatever similarity there is between neighbouring sentences is what is learnt as important for meaning.

The postprocessing step BID6 , which removes the top principal component of a batch of representations, is applied on produced representations from f and g respectively after learning with a final l 2 normalisation.

In addition, in our multi-view framework with discriminative objective, in order to reduce the discrepancy between training and testing, the top principal component is estimated by the power iteration method BID39 and removed during learning.

Three unlabelled corpora from different genres are used in our experiments, including BookCorpus , UMBC News BID21 and Amazon Book Review 2 BID35 ; six models are trained separately on each of three corpora with each of two objectives.

The summary statistics of the three corpora can be found in TAB6 .

Adam optimiser BID27 and gradient clipping BID43 are applied for stable training.

Pretrained word vectors, fastText , are used in our frameworks and fixed during learning.

TAB6 : Summary statistics of the three corpora used in our experiments.

For simplicity, the three corpora will be referred to as 1, 2 and 3 in the following tables respectively.

Table 2 : Representation pooling in testing phase.

"max(??)", "mean(??)", and "min(??)" refer to global max-, mean-, and min-pooling over time, which result in a single vector.

The table also presents the diversity of the way that a single sentence representation can be calculated.

X i refers to word vectors in i-th sentence, and H i refers to hidden states at all time steps produced by f .

DISPLAYFORM0 All of our experiments including training and testing are done in PyTorch BID44 .

The modified SentEval BID12 package with the step that removes the first principal component is used to evaluate our models on the downstream tasks.

Hyperparameters, including negative samples K in the framework with generative objective, context window c in the one with discriminative objective, are tuned only on the averaged performance on STS14 of the model trained on the BookCorpus; STS14/G1 and STS14/D1 results are thus marked with a in TAB1 to indicate possible overfitting on that dataset/model only.

Batch size N and dimension d in both frameworks are set to be the same for fair comparison.

Hyperparameters are summarised in supplementary material.

Representation: For a given sentence input s with M words, suggested by BID45 BID30 , the representation is calculated as z = ??? f +??? g /2, where??? refers to the post-processed and normalised vector, and is mentioned in Table 2 .Tasks: The unsupervised tasks include five tasks from SemEval Semantic Textual Similarity (STS) in 2012-2016 BID0 BID60 and the SemEval2014 Semantic Relatedness task (SICK-R) BID34 .

We compare our models with: ??? Unsupervised learning: We selected models with strong results from related work, including fastText, fastText+WR.??? Semi-supervised learning: The word vectors are pretrained on each task BID55 without label information, and word vectors are averaged to serve as the vector representation for a given sentence BID6 .???

Supervised learning: ParaNMT BID54 ) is included as a supervised learning method as the data collection requires a neural machine translation system trained in supervised fashion.

The InferSent 3 BID13 ) trained on SNLI BID8 and MultiNLI BID56 ) is included as well.

The results are presented in TAB1 .

Since the performance of FastSent BID24 and QT BID33 were only evaluated on STS14, we compare to their results in TAB2 .All six models trained with our learning frameworks outperform other unsupervised and semisupervised learning methods, and the model trained on the UMBC News Corpus with discriminative objective gives the best performance likely because the STS tasks contain multiple news-and headlines-related datasets which is well matched by the domain of the UMBC News Corpus.

The evaluation on these tasks involves learning a linear model on top of the learnt sentence representations produced by the model.

Since a linear model is capable of selecting the most relevant dimensions in the feature vectors to make predictions, it is preferred to concatenate various types of representations to form a richer, and possibly more redundant feature vector, which allows the Table 5 : Supervised evaluation tasks.

Bold numbers are the best results among unsupervised transfer models, and underlined numbers are the best ones among all models.

" ???" refers to an ensemble of two models.

" ???" indicates that additional labelled discourse information is required.

Our models perform similarly or better than existing methods, but with higher training efficiency.

Representation: Inspired by prior work BID36 BID46 , the representation z f is calculated by concatenating the outputs from the global mean-, max-and min-pooling on top of the hidden states H, and the last hidden state, and z g is calculated with three pooling functions as well.

The post-processing and the normalisation step is applied individually.

These two representations are concatenated to form a final sentence representation.

Table 2 presents the details.

Tasks: Semantic relatedness (SICK) BID34 , paraphrase detection (MRPC) BID18 , question-type classification (TREC) BID32 , movie review sentiment (MR) BID42 , Stanford Sentiment Treebank (SST) BID47 , customer product reviews (CR) BID25 , subjectivity/objectivity classification (SUBJ) BID41 , opinion polarity (MPQA) BID53 .

The results are presented in Table 5 .Comparison: Our results as well as related results of supervised task-dependent training models, supervised learning models, and unsupervised learning models are presented in Table 5 .

Note that, for fair comparison, we collect the results of the best single model of MC-QT BID33 ) trained on BookCorpus.

Six models trained with our learning frameworks either outperform other existing methods, or achieve similar results on some tasks.

The model trained on the Amazon Book Review gives the best performance on sentiment analysis tasks, since the corpus conveys strong sentiment information.

DISPLAYFORM0 Multi-view with g 1 and g 2 : a ij = cos(z DISPLAYFORM1

In both frameworks, RNN encoder and linear encoder perform well on all tasks, and generative objective and discriminative objective give similar performance.

The orthonormal regularisation applied on the linear decoder to enforce invertibility in our multiview framework encourages the vector representations produced by f and those by h ???1 , which is g in testing, to agree/align with each other.

A direct comparison is to train our multi-view framework without the invertible constraint, and still directly use U as an additional encoder in testing.

The results of our framework with and without the invertible constraint are presented in TAB4 .The ensemble method of two views, f and g, on unsupervised evaluation tasks (STS12-16 and SICK14) is averaging, which benefits from aligning representations from f and g by applying invertible constraint, and the RNN encoder f gets improved on unsupervised tasks by learning to align with g. On supervised evaluation tasks, as the ensemble method is concatenation and a linear model is applied on top of the concatenated representations, as long as the encoders in two views process sentences distinctively, the linear classifier is capable of picking relevant feature dimensions from both views to make good predictions, thus there is no significant difference between our multi-view framework with and without invertible constraint.

In order to determine if the multi-view framework with two different views/encoding functions is helping the learning, we compare our framework with discriminative objective to other reasonable variants, including the multi-view model with two functions of the same type but parametrised independently, either two f -s or two g-s, and the single-view model with only one f or g. TAB4 presents the results of the models trained on UMBC News Corpus.

As specifically emphasised in previous work BID24 , linear/log-linear models, which include g in our model, produce better representations for unsupervised evaluation tasks than RNNbased models do.

This can also be observed in TAB4 as well, where g consistently provides better results on unsupervised tasks than f .

In addition, as expected, multi-view learning with f and g, improves the resulting performance of f on unsupervised tasks, also improves the resulting g on supervised evaluation tasks.

Provided the results of models with generative and discriminative objective in TAB4 , we confidently show that, in our multi-view frameworks with f and g, the two encoding functions improve each other's view.

In general, aligning the representations generated from two distinct encoding functions ensures that the ensemble of them performs better.

The two encoding functions f and g encode the input sentence with emphasis on different aspects, and the subsequently trained linear model for each of the supervised downstream tasks benefits from this diversity leading to better predictions.

However, on unsupervised evaluation tasks, simply averaging representations from two views without aligning them during learning leads to poor performance and it is worse than g (linear) encoding function solely.

Our multi-view frameworks ensure that the ensemble of two views provides better performance on both supervised and unsupervised evaluation tasks.

Compared with the ensemble of two multi-view models, each with two encoding functions of the same type, our multi-view framework with f and g provides slightly better results on unsupervised tasks, and similar results on supervised evaluation tasks, while our model has much higher training efficiency.

Compared with the ensemble of two single-view models, each with only one encoding function, the matching between f and g in our multi-view model produces better results.

We proposed multi-view sentence representation learning frameworks with generative and discriminative objectives; each framework combines an RNN-based encoder and an average-on-wordvectors linear encoder and can be efficiently trained within a few hours on a large unlabelled corpus.

The experiments were conducted on three large unlabelled corpora, and meaningful comparisons were made to demonstrate the generalisation ability and transferability of our learning frameworks and consolidate our claim.

The produced sentence representations outperform existing unsupervised transfer methods on unsupervised evaluation tasks, and match the performance of the best unsupervised model on supervised evaluation tasks.

Our experimental results support the finding BID24 ) that linear/log-linear models (g in our frameworks) tend to work better on the unsupervised tasks, while RNN-based models (f in our frameworks) generally perform better on the supervised tasks.

As presented in our experiments, multi-view learning helps align f and g to produce better individual representations than when they are learned separately.

In addition, the ensemble of both views leveraged the advantages of both, and provides rich semantic information of the input sentence.

Future work should explore the impact of having various encoding architectures and learning under the multi-view framework.

Our multi-view learning frameworks were inspired by the asymmetric information processing in the two hemispheres of the human brain, in which the left hemisphere is thought to emphasise sequential processing and the right one more parallel processing BID9 .

Our experimental results raise an intriguing hypothesis about how these two types of information processing may complementarily help learning.

The details, including size of each dataset and number of classes, about the evaluation tasks are presented below 1 .

The Power Iteration was proposed in BID39 , and it is an efficient algorithm for estimating the top eigenvector of a given covariance matrix.

Here, it is used to estimate the top principal component from the representations produced from f and g separately.

We omit the superscription here, since the same step is applied to both f and g.

Suppose there is a batch of representations Z = [z 1 , z 2 ..., z N ] ??? R 2d??N from either f or g, the Power Iteration method is applied here to estimate the top eigenvector of the covariance matrix 2 : C = ZZ , and it is described in Algorithm 1:Algorithm 1 Estimating the First Principal Component BID39 Input: Covariance matrix C ??? R 2d??2d , number of iterations T Output: First principal component u ??? R u ??? u ||u||In our experiments, T is set to be 5.1 Provided by https://github.com/facebookresearch/SentEval 2 In practice, often N is less than 2d, thus we estimate the top eigenvector of Z Z ??? R N ??N .

The hyperparameters we need to tune include the batch size N , the dimension of the GRU encoder d, and the context window c, and the number of negative samples K. The results we presented in this paper is based on the model trained with N = 512, d = 1024.

Specifically, in discriminative objective, the context window is set c = 3, and in generative objective, the number of negative samples is set K = 5.

It takes up to 8GB on a GTX 1080Ti GPU.The initial learning rate is 5 ?? 10 ???4 , and we didn't anneal the learning rate through the training.

All weights in the model are initialised using the method proposed in BID23 , and all gates in the bi-GRU are initialised to 1, and all biases in the single-layer neural network are zeroed before training.

The word vectors are fixed to be those in the FastText , and we don't finetune them.

Words that are not in the FastText's vocabulary are fixed to 0 vectors through training.

The temperature term is initialised as 1, and is tuned by the gradient descent during training.

The temperature term is used to convert the agreement a ij to a probability distribution p ij in Eq. 1 in the main paper.

In our experiments, ?? is a trainable parameter initialised to 1 that decreased consistently through training.

Another model trained with fixed ?? set to the final value performed similarly.4 EFFECT OF POST-PROCESSING STEP Table 2 : Effect of the Post-processing Step. 'WR' refers to the post-processing step BID6 which removes the principal component of a set of learnt vectors.

The postprocessing step overall improves the performance of our models on unsupervised evaluation tasks, and also improves the models with generative objective on supervised sentence similarity tasks.

However, it doesn't have a significant impact on single sentence classification tasks.

TAB1 .As presented in the table, no further improvement against models with only one objective is shown.

In our understanding, the inverse of the linear decoder in generative objective behaves similarly to the linear encoder in the discriminative objective, which is presented in TAB4 in the main paper.

Therefore, combining two objectives doesn't perform better than only one of them.

TAB1 : Our multi-view framework with both generative and discriminative objective. 'GD1' refers to a model with both generative and discriminative objectives trained on BookCorpus.

The results here don't show significant difference against the model trained with only one objective.

<|TLDR|>

@highlight

Multi-view learning improves unsupervised sentence representation learning

@highlight

Approach uses different, complementary encoders of the input sentence and consensus maximization.

@highlight

The paper presents a multi-view framework for improving sentence representation in NLP tasks using generative and discriminative objective architectures.

@highlight

This paper shows that multi-view frameworks are more effective than using individual encoders for learning sentence representations.