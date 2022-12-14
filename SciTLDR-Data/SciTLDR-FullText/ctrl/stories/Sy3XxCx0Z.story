Modeling informal inference in natural language is very challenging.

With the recent availability of large annotated data, it has become feasible to train complex models such as neural networks to perform natural language inference (NLI), which have achieved state-of-the-art performance.

Although there exist relatively large annotated data, can machines learn all knowledge needed to perform NLI from the data?

If not, how can NLI models benefit from external knowledge and how to build NLI models to leverage it?

In this paper, we aim to answer these questions by enriching the state-of-the-art neural natural language inference models with external knowledge.

We demonstrate that the proposed models with external knowledge further improve the state of the art on the Stanford Natural Language Inference (SNLI) dataset.

Reasoning and inference are central to both human and artificial intelligence.

Natural language inference (NLI) is concerned with determining whether a natural-language hypothesis h can be inferred from a natural-language premise p. Modeling inference in human language is very challenging but is a basic problem towards true natural language understanding -NLI is regarded as a necessary (if not sufficient) condition for true natural language understanding BID20 .The most recent years have seen advances in modeling natural language inference.

An important contribution is the creation of much larger annotated datasets such as SNLI BID5 and MultiNLI BID37 .

This makes it feasible to train more complex inference models.

Neural network models, which often need relatively large amounts of annotated data to estimate their parameters, have shown to achieve the state of the art on SNLI and MultiNLI BID5 BID25 BID27 BID30 BID26 BID8 BID1 .

While these neural networks have shown to be very effective in estimating the underlying inference functions by leveraging large training data to achieve the best results, they have focused on end-toend training, where all inference knowledge is assumed to be learnable from the provided training data.

In this paper, we relax this assumption, by exploring whether external knowledge can further help the best reported models, for which we propose models to leverage external knowledge in major components of NLI.

Consider an example from the SNLI dataset:??? p: An African person standing in a wheat field.??? h: A person standing in a corn field.

If the machine cannot learn useful or plenty information to distinguish the relationship between wheat and corn from the large annotated data, it is difficult for a model to predict that the premise contradicts the hypothesis.

In this paper, we propose neural network-based NLI models that can benefit from external knowledge.

Although in many tasks learning tabula rasa achieved state-of-the-art performance, we believe complicated NLP problems such as NLI would benefit from leveraging knowledge accumulated by humans, at least in a foreseeable future when machines are unable to learn that with limited data.

A typical neural-network-based NLI model consists of roughly four components -encoding the input sentences, performing co-attention across premise and hypothesis, collecting and computing local inference, and performing sentence-level inference judgment by aggregating or composing local information information.

In this paper, we propose models that are capable of leveraging external knowledge in co-attention, local inference collection, and inference composition components.

We demonstrate that utilizing external knowledge in neural network models outperforms the previously reported best models.

The advantage of using external knowledge is more significant when the size of training data is restricted, suggesting that if more knowledge can be obtained, it may yielding more benefit.

Specifically, this study shows that external semantic knowledge helps mostly in attaining more accurate local inference information, but also benefits co-attention and aggregation of local inference.

Early work on natural language inference (also called recognizing textual textual) has been performed on quite small datasets with conventional methods, such as shallow methods BID13 , natural logic methods BID20 , among others.

These work already shows the usefulness of external knowledge, such as WordNet BID22 , FrameNet BID1 , and so on.

More recently, the large-scale dataset SNLI was made available, which made it possible to train more complicated neural networks.

These models fall into two kind of approaches: sentence encodingbased models and inter-sentence attention-based models.

Sentence encoding-based models use Siamese architecture BID7 -the parameter-tied neural networks are applied to encode both the premise and hypothesis.

Then a neural network classifier (i.e., multilayer perceptron) is applied to decide the relationship between the two sentence representations.

Different neural networks have been utilized as sentence encoders, such as LSTM BID5 , GRU BID33 , CNN BID23 , BiLSTM and its variants BID17 BID9 , and more complicated neural networks BID6 BID24 BID1 .

The advantage of encoding-based models is that the encoders transform sentences into fixed-length vector representations, which can help a wide range of transfer tasks BID12 .

However, this architecture ignores the local interaction between two sentences, which is necessary in traditional natural language inference procedure BID20 .Therefore, inter-sentence attention-based models were proposed to relieve this problem.

In this framework, local inference information is collected by the attention mechanism and then fed into neural networks to compose as fixed-sized vectors before the final classification.

Many related works follow this route BID29 BID34 BID11 BID27 BID8 .

Among them, BID29 were the first to propose neural attention-based models for NLI.

BID8 proposed an enhanced sequential inference model (ESIM), which is one of the best models so far and regarded as the baseline in this paper.

In general, external knowledge have been shown to be effective in a wide range of NLP tasks, including machine translation BID32 , language modeling BID0 , and dialogue system .

For NLI, to the best of our knowledge, we are the first to utilize external knowledge together with neural networks.

In this paper, we first show that a neural network equipped with external knowledge obtains further improvement over the already strong baseline, and achieves an accuracy of 88.6% on the SNLI benchmark.

Furthermore, we show that the gain is more significant when using less training samples.

External knowledge needs to be converted to a numerical representation for enriching natural language inference model.

One of approaches to represent external knowledge is using knowledge graph embeddings, such as TransE BID4 , TransH BID35 , TransG BID38 , and so on.

However, these kind of approaches usually need to train a knowledge-graph embedding beforehand.

In this paper, we propose to use relation features to describe relationship between the words in any word pair, which can be easily obtained from various knowledge graphs, such as WordNet BID22 , and Freebase BID2 .

Specifically, we use WordNet to measure the semantic relatedness of the word in a pair using various relation types, including synonymy, antonymy, hypernymy, and so on.

Each of these features is a real number on the interval [0, 1] .

The definition and instances of pair features derived from WordNet are indicated in TAB0 .

The setting of features refers to MacCartney (2009), but we add a new feature same hypernym, which improve the result significantly in our experiments.

Intuitively, the synonymy, hypernymy and hyponymy features help model the entailment of word pairs; the antonymy and same hypernym features help model contradiction in word pairs.

We regard the vector r ??? R Dr as the relation feature derived from external knowledge, where D r is 5 in our experiments.

The r will be enriched in the neural inference model to capture external semantic knowledge.

TAB1 reports some key statistics of the relation features from WordNet.

We present here our natural inference models which are composed of the following major components: input encoding, knowledge enriched co-attention, knowledge enriched local inference collection, and knowledge enriched inference composition.

Figure 1 shows a high-level view of the architecture.

First, the premise and hypothesis are encoded by the input encoding components as context-dependent representations.

Second, co-attention is calculated to obtain word-level softalignment between two sentences.

Third, local inference information is collected to prepare for final prediction.

Fourth, the inference composition component applies aggregation of the whole sentences and makes final prediction based on the fixed-size vector.

Among them, external knowledge is regard as the auxiliary component to improve the ability of (1) calculating co-attention, (2) collecting local inference information and (3) composing inference.

Figure 1: A high-level view of our neural inference networks.

Given two sentence, i.e., the premise "The child is getting a pedicure", and the hypothesis "The kid is getting a manicure", the model needs to predict the relationship among them: entailment, contradiction, or neutral.

Given the word sequences of the premise a = (a 1 , . . .

, a M ) and the hypothesis b = (b 1 , . . .

, b N ), where M and N are the lengths of the sentences, the final objective is to predict a label y that indicates the logic relationship between a and b. The formula is y = arg max DISPLAYFORM0 Specifically, "<BOS>" and "<EOS>" are inserted as the first and last token, respectively.

First, a and b are embedded into a D e -dimensional vectors [E(a 1 ), . . .

, E(a M )] and [E(b 1 ), . . .

, E(b N )] using an embedding matrix E ??? R De??V , where V is the vocabulary size and E can be initialized with some pre-trained word embeddings from a universal corpus.

To represent the words of the premise and hypothesis in a context-dependent way, the two sentences are fed into the encoders to obtain context-dependent hidden states a s and b s .

The formula is DISPLAYFORM1 We employ bidirectional LSTMs (BiLSTMs) BID15 as encoders, which is a common choice for natural language.

A BiLSTM runs a forward and a backward LSTM on a sequence starting from the left and the right end, respectively.

The hidden states generated by these two LSTMs at each time step are concatenated to represent that time step and its context: DISPLAYFORM2 The hidden states of the unidirectional LSTM (h ??? t or h ??? t ) is calculated as follows: DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 where ?? is the sigmoid function, is the element-wise multiplication of two vectors.

DISPLAYFORM9 are parameters to be learned.

D is the dimension of the hidden states in the LSTM.

The LSTM utilizes a set of gating functions for each input vector x t , i.e., the input gate i t , forget gate f t , and output gate o t , together with a memory cell c t to generate a hidden state h t .

In this component, we acquire soft-alignment of word pairs between the premise and hypothesis based on our knowledge-enriched co-attention mechanism.

Given the relation features r ij ??? R Dr between the premise's i-th word and the hypothesis's j-th word from the external knowledge, the co-attention is calculated as e ij = (a DISPLAYFORM0 The function F can be any non-linear or linear function.

Here we use F (r ij ) = ??1(r ij ), where ?? is a hyper-parameter tuned on the development set and 1 is the indication function.1(r ij ) = 1 if r ij is not zero vector ; 0 if r ij is zero vector .Intuitively, the word pairs with semantic relationship in various features are probably aligned together.

Soft-alignment is determined by the co-attention matrix e ??? R M ??N computed in Equation (9), which is used to obtain the local relevance between the premise and hypothesis.

For the hidden state of a word in a premise, i.e., a s i (already encoding the word itself and its context), the relevant semantics in the hypothesis is identified into a context vector a c i using e ij , more specifically with Equation (11).

DISPLAYFORM1 , a DISPLAYFORM2 DISPLAYFORM3 where ?? ??? R M ??N and ?? ??? R M ??N are the normalized attention weight matrices with respect to the 2-axis and 1-axis.

The same calculation is performed for each word in the hypothesis, i.e., b DISPLAYFORM4 DISPLAYFORM5 where a heuristic matching trick with difference and element-wise product is used BID23 BID8 .

The last term in Equation FORMULA0 aims to obtain the local inference relationship between the original vectors (a

In this component, we introduce knowledge-enriched inference composition.

To determine the overall inference relationship between a premise and a hypothesis, we need to explore a composition layer to compose the local inference vectors (a m and b m ) collected above.

The formula is DISPLAYFORM0 Here, we also use BiLSTMs as building blocks for the composition layer.

The BiLSTMs read local inference vectors (a m and b m ) and learn to judge the type of local inference relationship and distinguish crucial local inference vectors for overall sentence-level inference relationship.

The responsibility of BiLSTMs in the inference composition layer is completely different from the BiLSTMs in the input encoding layer.

Our inference model converts the output hidden vectors of BiLSTMs to a fixed-length vector with pooling operations and puts it into the final classifier to determine the overall inference class.

Particularly, besides using mean pooling and max pooling similarly to ESIM BID8 , we propose to use weighted pooling based on external knowledge to obtain a fixed-length vector as in Equation FORMULA0 .

Intuitively, the final prediction is mostly determined by those word pairs appearing in the external knowledge.

BID9 uses a similar idea called gated-attention but they do not use external knowledge.

DISPLAYFORM1 In our experiments, we regard the function H as a 1-layer feed-forward neural network with ReLU activation function.

We concatenate all pooling vectors, i.e., mean, max, and weighted pooling, into a fixed-length vector and then put the vector into a final multilayer perceptron (MLP) classifier.

The MLP has a hidden layer with tanh activation and softmax output layer in our experiments.

The entire model is trained end-to-end, through minimizing the cross-entropy loss.

The Stanford Natural Language Inference (SNLI) dataset BID5 focuses on three basic relationships between a premise and a potential hypothesis: the premise entails the hypothesis (entailment), they contradict each other (contradiction), or they are not related (neutral).

We use the same data split as in previous work, and use classification accuracy as the evaluation metric, as in related work.

WordNet 3.0 BID22 is used to extract semantic relation features between words, as described in Section 3.1.

The words are lemmatized using Stanford CoreNLP 3.7.0 (Manning et al., 2014) to match words in WordNet, but the input word sequences for the input encoding layer are only tokenized, without lemmatization.

We release our code at [xxx] to make it replicatibility purposes.

The models are selected on the development set.

Some of our training details are as follows: the dimension of the hidden states of LSTMs and word embeddings are 300.

The word embeddings are initialized by 300D GloVe 840B BID28 , and out-of-vocabulary words among them are initialized randomly.

All word embeddings are updated during training.

Adam BID16 ) is used for optimization with an initial learning rate 0.0004.

The mini-batch size is set to 32.

Dropout with a keep rate of 0.5 and early stopping with patience of 7 are used to avoid overfitting.

The gradient is clipped with a maximum L2-norm 10.

The trade-off ?? for calculating co-attention in Equation FORMULA10 is selected in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50] based on the development set.

LSTM BID5 80.6 GRU BID33 81.4 Tree CNN BID23 82.1 SPINN-PI BID6 83.2 NTI BID25 83.4 Intra-Att BiLSTM 84.2 Self-Att BiLSTM BID17 84.2 NSE BID24 84.6 Gated-Att BiLSTM BID9 85.5 DiSAN BID31 85.6 LSTM Att BID29 83.5 mLSTM BID34 86.1 LSTMN BID11 86.3 Decomposable Att BID27 86.8 NTI BID25 87.3 Re-read LSTM BID30 87.5 BiMPM 87.5 btree-LSTM BID26 87.6 DIM BID14 88.0 ESIM BID8 88.0 KIM 88.6 HIM (ESIM+Syntactic TreeLSTM) BID8 88.6 BiMPM (Ensemble) 88.8 DIIN (Ensemble) 88.9 KIM (Ensemble) 89.1 TAB2 shows the results of different models on the SNLI dataset.

The first group of models use sentence-encoding based approaches.

BID5 employs LSTMs as encoders for both the premise and hypothesis into two fixed-size sentence vectors.

Then the sentence representation is put into a MLP classifier to predict the final inference relationship.

The accuracy on the test set is 80.6%.

Many related works follow this framework, using different neural networks as encoders.

Their performances are also listed in the first group in TAB2 .

Among them, gated-Att BiLSTM BID9 achieves an accuracy of 85.5%, which is state of the art for sentenceencoding based approaches.

The second group of models uses a cross-sentence attention mechanism, which can obtain softalignment information between cross-sentence word pairs.

BID34 proposes a matching-LSTM to compare the inference information of locally-aligned words, and obtains a higher accuracy of 86.1%, even better than the state-of-the-art sentence-encoding models.

Other related models are also listed in the second group in TAB2 .

Among them, ESIM BID8 ) is the previous state-of-the-art system, whose accuracy in test set is 88.0%.

The proposed model, namely Knowledge-based Inference Model (KIM), which enriches ESIM with external knowledge, obtains an accuracy of 88.6%.

The difference between ESIM and KIM is statistically significant under the one-tailed paired t-test at the 99% significance level.

To be best of our knowledge, this is a new state of the art.

Our ensemble model, which averages the probability distributions from ten individual single KIMs with different initialization, achieves an even higher accuracy, 89.1%.

To compare the importance of external knowledge under different training data scales, we randomly sample different ratio of the whole training set, i.e., 0.8%, 4%, 20% and 100%.

"A" indicates adding external knowledge in calculating the co-attention matrix as in Equation FORMULA10 , "I" indicates adding external knowledge in collecting local inference information as in Equation FORMULA0 , and "C" indicates adding external knowledge in composing inference as in Equation (16).

When we only have restricted training data, i.e., 0.8% training set (about 4,000 samples), our baseline ESIM has a poor accuracy of 62.4%.

When we only add external knowledge in calculating co-attention ("A"), the accuracy increases to 66.6% (+ absolute 4.2%).

When we only utilize external knowledge in collecting local inference information ("I"), the accuracy has a significant gain, to 70.3% (+ absolute 7.9%).

When we only add external knowledge in inference composition ("C"), the accuracy gets a smaller gain to 63.4% (+ absolute 1.0%).

The comparison indicates that "I" plays the most important role among the three components in using external knowledge.

Moreover, when we compose the three components ("A,I,C"), we obtain the best result of 72.6% (+ absolute 10.2%).

When we use more training data, i.e., 4%, 20%, 100% of the training set, only utilizing external knowledge in local inference information collected ("I") achieves a significant gain, but "A" or "C" do not bring any significant improvement.

The results indicate that external semantic knowledge only helps in co-attention and composition when there is limited training data, but always helps in collecting local inference information.

Meanwhile, for less training data, ?? is usually set to a larger value.

For example, the optimal ?? tuned on the development set is 20 for 0.8% training set, 2 for the 4% training set, 1 for the 20% training set and 0.2 for the 100% training set.

Figure 3 displays the results of using different ratio of external knowledge for different training data size.

Note that here we only use external knowledge in collecting local inference information, because it always works well for different scale of the training set.

Better accuracies are achieved when using more external knowledge.

Especially under the condition of restricted training data (0.8%), the model obtains a large gain when using more than half of the external knowledge.

Our enriched neural network-based model for natural language inference with external knowledge, namely KIM, achieves a new state-of-the-art accuracy on the SNLI dataset.

The model is equipped with external knowledge in the major informal inference components, specifically, in calculating co-attention, collecting local inference, and composing inference.

The proposed models of infusing neural networks with external knowledge may also help shed some light on tasks other than NLI, such as question answering and machine translation.

<|TLDR|>

@highlight

the proposed models with external knowledge further improve the state of the art on the SNLI dataset.