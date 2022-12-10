There is mounting evidence that pretraining can be valuable for neural network language understanding models, but we do not yet have a clear understanding of how the choice of pretraining objective affects the type of linguistic information that models learn.

With this in mind, we compare four objectives---language modeling, translation, skip-thought, and autoencoding---on their ability to induce syntactic and part-of-speech information, holding constant the genre and quantity of training data.

We find that representations from language models consistently perform best on our syntactic auxiliary prediction tasks, even when trained on relatively small amounts of data, which suggests that language modeling may be the best data-rich pretraining task for transfer learning applications requiring syntactic information.

We also find that a randomly-initialized, frozen model can perform strikingly well on our auxiliary tasks, but that this effect disappears when the amount of training data for the auxiliary tasks is reduced.

Representation learning with deep recurrent neural networks has revolutionized natural language processing and replaced many of the expert-designed, linguistic features previously used.

Recently, researchers have begun to investigate the properties of representations learned by networks by training auxiliary classifiers that use the hidden states of frozen pretrained models to perform other tasks.

These investigations have shown that when deep LSTM RNNs (Hochreiter and Schmidhuber, 1997) are trained on tasks like machine translation, they latently identify substantial syntactic and semantic information about their input sentences, including part-of-speech (Shi et al., 2016; Belinkov et al., 2017a,b; Blevins et al., 2018) .These intriguing findings lead us to ask the following questions:1.

How does the training task affect how well models latently learn syntactic properties?

Which tasks are better at inducing these properties?2.

How does the amount of data the model is trained on affect these results?

When does training on more data help?We investigate these questions by holding the data source and model architecture constant, while varying both the training task and the amount of training data.

Specifically, we examine models trained on English-German (En-De) translation, language modeling, skip-thought (Kiros et al., 2015) , and autoencoding, in addition to an untrained baseline model.

We control for the data domain by exclusively training on datasets from the 2016 Conference on Machine Translation (WMT; Bojar et al., 2016) .

We train models on all tasks using the parallel En-De corpus and a small subset of that corpus, which allows us to make a fair comparison across all five models.

Additionally, we augment the parallel dataset with a large monolingual corpus from WMT to examine how the performance of the unsupervised tasks (all but translation) scale with more data.

Throughout our work, we focus on the syntactic evaluation tasks of part-of-speech (POS) tagging and Combinatorial Categorical Grammar (CCG) supertagging.

Supertagging is a building block for parsing as these tags constrain the ways in which words can compose, largely determining the parse of the sentence.

CCG supertagging thus allows us to measure the degree to which models learn syntactic structure above the word.

We focus our analysis on representations learned by language models and by the encoders of sequence-to-sequence models, as translation encoders have been found to learn richer representations of POS and morphological information than translation decoders (Belinkov et al., 2017a) .We find that for POS and CCG tagging, bidirectional language models (BiLMs)-created by separately training forward and backward language models, and concatenating their hidden statesoutperform models trained on all other tasks.

Even BiLMs trained on relatively small amounts of data (1 million sentences) outperform translation and skip-thought models trained on larger datasets (5 million and 63 million sentences respectively).Our inclusion of an untrained LSTM baseline allows us to study the effect of training on state representations.

We find, surprisingly, that randomly initialized LSTMs underperform our best trained models by only a few percentage points when we use all of the available labeled data to train classifiers for our auxiliary tasks.

When we reduce the amount of classifier training data though, the performance of the randomly initialized LSTM model drops far below those of trained models.

We hypothesize that this occurs because training the classifiers on large amounts of auxiliary task data allows them to memorize configurations of words seen in the training set and their associated tags.

We test this hypothesis by training classifiers to predict the identity of neighboring words from a given hidden state, and find that randomly initialized models outperform all trained models on this task.

Our findings demonstrate that our best trained models do well on the tagging tasks because they are truly learning representations that conform to our notions of POS and CCG tagging, and not because the classifiers we train are able to recover neighboring word identity information well.

Evaluating Latent Representations Adi et al. (2016) introduce the idea of examining sentence vector representations by training auxiliary classifiers to take sentence encodings and predict attributes like word order.

Belinkov et al. (2017a) build on this work by using classifiers to examine the hidden states of machine translation models in terms of what they learn about POS and morphology.

They find that translating into morphologically poorer languages leads to a slight improvement in encoder representations.

However, this effect is small, and we expect that our study of English-German translation will nonetheless provide a reasonable overall picture of the representations that can be learned in data-rich translation.

Beyond translation, Blevins et al. (2018) find that deep LSTMs latently learn hierarchical syntax when trained on a variety of tasks-including semantic role labeling, language modeling, and dependency parsing.

We build on this work by controlling for model size, and the quantity and genre of the training data, which allows us to make direct comparisons between different tasks on their ability to induce syntactic information.

Transfer Learning of Representations Much of the work on sentence-level pretraining has focused on sentence-to-vector models and evaluating learned representations on how well they can be used to perform sentence-level classification tasks.

Skip-thought (Kiros et al., 2015) -the technique of training a sequence-to-sequence model to predict the sentence preceding and following each sentence in a running text-represents a prominent early success in this area with unlabeled data, and InferSent (Conneau et al., 2017) -the technique of pretraining encoders on natural language inference data-yields strikingly better performance when such labeled data is available.

Work in this area has recently moved beyond strict sentence-to-vector mapping.

Newer models that incorporate LSTMs pretrained on datarich tasks, like translation and language modeling, have achieved state-of-the-art results on many tasks-including semantic role labeling and coreference resolution (Peters et al., 2018; McCann et al., 2017; Howard and Ruder, 2018) .

Although comparisons have previously been made between translation and language modeling as pretraining tasks (Peters et al., 2018) , we investigate this issue more thoroughly by controlling for the quantity and content of the training data.

Training Dataset Size The performance of neural models depends immensely on the amount of training data used.

Koehn and Knowles (2017) find that when training machine translation models on corpora with fewer than 15 million words (English side), statistical machine translation approaches outperform neural ones.

Similarly, Hestness et al. (2017) study data volume dependence on several tasks-including translation and image classification-and find that for small amounts of data, neural models perform about as well as , 2017) .

Our method of training auxiliary classifiers on randomly initialized RNNs builds on the tradition of reservoir computing, in which randomly initialized networks or "reservoirs" are fixed and only "read-out" classifier networks are trained (Lukoševičius and Jaeger, 2009).

Echo state networks-reservoir computing with recurrent models-have been used for tasks like speech recognition, language modeling, and time series prediction (Verstraeten et al., 2006; Tong et al., 2007; Sun et al., 2017) .

We use the parallel English-German (En-De) dataset from the 2016 ACL Conference on Machine Translation (WMT) shared task on news translation (Bojar et al., 2016) .

This dataset contains 5 million ordered sentence translation pairs.

We also use the 2015 English monolingual news dataset from the same WMT shared task, which contains approximately 58 million ordered sentences.

To examine how the volume of training data affects learned representations, we use four corpus sizes: 1, 5, 15, and 63 million sentences (translation is only trained on the smaller two sizes).

We create the 1 million sentence corpora from the 5 million sentence dataset by sampling (i) sentence pairs for translation, (ii) English sentences for autoencoders, and (iii) ordered English sentence pairs for skip-thought and language models.

Note, we initialize language model LSTM states with the final state after reading the previous sentence.

Similarly, we create 15 million sentence corpora for the unsupervised tasks by sampling sentences from the entire corpus of 63 million sentences.

We use word-level representations throughout and use the Moses package (Koehn et al., 2007) to tokenize and truecase our data.

Finally, we limit both the English and German vocabularies to the 50k most frequent tokens in the training set.

We train all our models using OpenNMT-py (Klein et al., 2017) and use the default options for model sizes, hyperparameters, and training procedure-except that we increase the size of the LSTMs, make the encoders bidirectional, and use validation-based learning rate decay instead of a fixed schedule.

Specifically, all our models (except language models) are 1000D, twolayer encoder-decoder LSTMs with bidirectional encoders (500D LSTMs in each direction) and 500D embeddings; we train models both with and without attention (Bahdanau et al., 2015) .

For language models, we train a single 1000D forward language model and a bidirectional language model-two 500D language models (one forward, one backward) trained separately, whose hidden states are concatenated.

All models are randomly initialized with a uniform distribution between −0.1 and 0.1, the default in OpenNMT-py.

We use the same training procedure for all our models.

We evaluate on the validation set every epoch when training on the 1 and 5 million sentence datasets, and evaluate approximately every 5 million sentences when training on the larger datasets.

We use SGD with an initial learning rate of 1.

Whenever a model's validation loss increases relative to the previous evaluation, we halve the learning rate and stop training when the learning rate reaches 0.5 15 .

For each training task and dataset size, we select the model with the lowest validation perplexity to perform auxiliary task evaluations on.

We report model performance in terms of perplexity and BLEU (Papineni et al., 2002) in Table 1 .

For translation we use beam search (B = 5) when decoding.

For CCG supertagging, we use CCG Bank (Hockenmaier and Steedman, 2007) , which is based on PTB WSJ.

CCG supertagging provides fine-grained information about the role of each word in its larger syntactic context and is considered almost parsing, since sequences of tags map sentences to small subsets of possible parses.

The entire dataset contains approximately 50k sentences and 1327 tag types.

We display POS and CCG tags for an example sentence in FIG2 To study the impact of auxiliary task data volume, for both datasets we create smaller classifier training sets by sampling 10% and 1% of the sentences.

We truecase both datasets using the same truecase model trained on WMT and restrict the vocabularies to the 50k tokens used in our LSTM models.

We use the word-conditional most frequent class as a baseline, which is the most frequently assigned tag class for each word in the training set; for this baseline we restrict the vocabulary to that of our encoder models (we map all out-of-vocabulary words to a single UNK token).

Note that while PTB and WMT are both drawn from news text, there is slight genre mismatch.

Word Identity For this task, the classifier takes a single LSTM hidden state as input and predicts the identity of the word at a different time step, for example, three steps previous (shift of -3).

We use the WSJ data for this task.

Following Conneau et al. (2018) , we take all words that occur between 100 and 1000 times (about 1000 words total) as the possible targets for neighboring word prediction.

Classifier Training Procedure We train multilayer perceptron (MLP) classifiers that take an LSTM hidden state (from one time step and one layer) and output a distribution over the possible labels (tags or word identities).

The MLPs we train have a single 1000D hidden layer with a ReLU activation.

For classifier training, we use the same training and learning rate decay procedure used for pretraining the encoders.

In this section, we discuss the main POS and CCG tagging results displayed in FIG3 .

Overall, POS and CCG tagging accuracies tend to increase with the amount of data the LSTM encoders are trained on.

However, the amount of this improvement is generally small, especially when encoders are already trained on large amounts of data.

Language Modeling and Translation For all pretraining dataset sizes and tasks, bidirectional language model (BiLM) and translation encoder representations perform best in terms of both POS and CCG tagging.

Translation encoders, however, slightly underperform BiLMs, even when both models are trained on the same amount of data.

Interestingly, even BiLMs trained on the smallest amount of data (1 million sentences) outperform models trained on all other tasks and dataset sizes (up to 5 million sentences for translation, and 63 million sentences for skip-thought and autoencoding).

The consistent superior performance of BiLMs-along with the fact that language models do not require aligned data-suggests that for transfer learning of syntactic information, BiLMs are superior to translation encoders.

For all amounts of training data, the BiLMs significantly outperform the 1000D forward-only language models.

The gap in performance between bidirectional and forward language models is greater for CCG supertagging than for POS tagging.

When using all available auxiliary training data, there is a 2 and 8 percentage point performance gap on POS and CCG tagging respectively.

This difference in relative performance suggests that bidirectional context information is more important when identifying syntactic structure than when identifying part of speech.

Figure 2 also illustrates how the best performing BiLMs and translation models tend to be more robust to decreases in classifier data than models trained on other tasks.

When training on less auxiliary task data, POS tagging performance tends to drop less than CCG supertagging performance.

For the best model (BiLM trained on 63 million sentences), when using 1% rather than all of the auxiliary task training data, CCG accuracy drops 9 percentage points, while POS accuracy only drops 2 points.

Further examinations of the effect of classifier data volume are displayed in FIG5 .Skip-Thought Although skip-thought encoders consistently underperform both BiLMs and translation encoders in all data regimes we examine, skip-thought models improve the most when increasing the amount of pretraining data, and are the only models whose performance does not seem to have plateaued by 63 million training sentences.

Skip-thought models without attention are very similar to language models-the main difference is that while skip-thought models have separate encoder and decoder weights (and a bidirectional encoder), language models share weights between the encoder and decoder.

Thus language models can be interpreted as regularized versions of skip-thought.

The increased model capacity of skip-thought, compared to language models, could explain the difference in learned representation quality-especially when these models are trained on smaller amounts of data.

Random Initialization For our randomly initialized, untrained LSTM encoders we use the default weight initialization technique in OpenNMTpy, a uniform distribution between -0.1 and 0.1; the only change we make is to set all biases to zero.

We find that this baseline performs quite well when using all auxiliary data, and is only 3 and 8 percentage points behind the BiLM on POS and CCG tagging, respectively.

We find that decreasing the amount of classifier data leads to a significantly greater drop in the randomly initialized encoder performance compared to trained models.

In the 1% classifier data regime, the performance of untrained encoders on both tasks drops below that of all trained models and below even the wordconditional most-frequent class baseline.

We hypothesize that the randomly initialized baseline is able to perform well on tagging tasks with large amounts of auxiliary task training data, because the classifier can learn the identity of neighboring words from a given time step's hidden state, and simply memorize word configurations and their associated tags from the training data.

We test this hypothesis directly in Section 6 and find that untrained LSTM representations are in fact better at capturing neighboring word identity information than any trained model.

Autoencoder Models trained on autoencoding are the only ones that do not consistently improve with the amount of training data, which is unsurprising as unregularized autoencoders are prone to learning identity mappings (Vincent et al., 2008) .

When training on 10% and 1% of the auxiliary task data, autoencoders outperform randomly initialized encoders and match the word-conditional most frequent class baseline.

When training on all the auxiliary data though, untrained encoders outperform autoencoders.

These results suggest that autoencoders learn some useful structure that Figure 4: POS and CCG tagging accuracies in terms of percentage points over the word-conditional most frequent class baseline.

We display results for the best performing models for each task.is useful in the low auxiliary data regime.

However, the representations autoencoders learn do not capture syntactically rich features, since random encoders outperform them in the high auxiliary data regime.

This conclusion is further supported by the extremely poor performance of the second layer of an autoencoder without attention on POS tagging (almost 10 percentage points below the most frequent class baseline), as seen in Figure 4a .

Embeddings (Layer 0) We find that randomly initialized embeddings consistently perform as well as the word-conditional most frequent class baseline on POS and CCG tagging, which serves as an upper bound on performance for the embedding layer.

As these are untrained, the auxiliary classifiers are learning to memorize and classify the random vectors.

When using all the auxiliary classifier data, there is no significant difference in the performance of trained and untrained embeddings on the tagging tasks.

Only for smaller amounts of classifier data do trained embeddings consistently outperform randomly initialized ones.

Belinkov et al. (2017a) find that, for translation models, the first layer consistently outperforms the second on POS tagging.

We find that this pattern holds for all our models, except in BiLMs, where the first and second layers perform equivalently.

The pattern holds even for untrained models, suggesting that POS information is stored on the lower layer, not because the training task encourages this, but because of fundamental properties of the deep LSTM architecture.

We also find that for CCG supertagging, the first layer also outperforms the second layer on untrained models.

For the trained models though, behavior is mixed, with the second layer performing better in some cases.

Which layer performs best appears to be independent of absolute performance on the supertagging task.

Our layer analysis results are displayed in Figure 4 .

Our results on word identity prediction are summarized in FIG7 and given in more detail in Appendix A. We find that randomly initialized LSTMs outperform all trained models.

We hypothesize this occurs because a kind of useful forgetting occurs during training, as the LSTMs learn that information about certain word patterns are more important to remember than others.

In this regard, randomly initialized models are less biased and process inputs more uniformly.

The fact that untrained encoders outperform trained ones on word identity prediction, but underperform trained models on POS and CCG tagging, confirms that trained models genuinely capture substantial syntactic features, beyond mere word identity, that the auxiliary classifiers can use.

Effect of Depth We find that for both trained and untrained models, the first layer outperforms the second layer when predicting the identity of the immediate neighbors of a word.

However, the second layer tends to outperform the first at predicting the identity of more distant neighboring words.

This effect is especially apparent for the randomly initialized encoders.

Our findings suggest that, as is the case for convolutional neural networks, depth in recur-

By controlling for the genre and quantity of the training data, we make fair comparisons between several data-rich training tasks in their ability to induce syntactic information.

We find that bidirectional language models (BiLMs) do better than translation and skip-thought encoders at extracting useful features for POS tagging and CCG supertagging.

Moreover, this improvement holds even when the BiLMs are trained on substantially less data than competing models.

Although, due to limited parallel data, we could not compare BiLMs and translation encoders on more than 5 million sentences, our results suggest that for syntactic information, there is no need to compare these two models trained on more data, as BiLMs consistently outperform translation encoders in all data regimes.

We also find that randomly initialized encoders extract usable features for POS and CCG tagging, at least when the auxiliary POS and CCG classifiers are themselves trained on reasonably large amounts of data.

However, the performance of untrained models drops sharply relative to trained ones when using smaller amounts of the classifier data.

We investigate further and find that untrained models outperform trained ones on the task of neighboring word identity prediction, which confirms that trained encoders do not perform well on tagging tasks because the classifiers are simply memorizing word identity information.

We also find that both trained and untrained LSTMs store more local neighboring word identity information in lower layers and more distant word identity information in upper layers, which suggests that depth in LSTMs allow them to capture larger context information.

Our results suggest that for transfer learning, bidirectional language models like ELMo (Peters et al., 2018) capture more useful features than translation encoders-and that this holds even on genres or languages for which data is not abundant.

However, the scope of our experiments is limited, and we still know little about the representations of models trained on other supervised tasks, or precisely how the choice of training task affects the type of syntactic information that is learned.

Our work also highlights the interesting behavior of randomly initialized LSTMs, which show an ability to preserve the contents of their inputs significantly better than trained models.

Figure 6: Here we display results for the word identity prediction task with randomly initialized LSTM encoders with up to 4 layers.

Lower layers have a more peaked shape and upper layers a more flat shape, meaning that the lower layers encode relatively more nearby neighboring word information, while upper layers encode relatively more distant neighboring word information.

Table 4 : Here we display results for training on 1% of auxiliary task data.

Word-conditional most frequent class baselines for this amount of training data are 81.8% for POS tagging and 62.3% for CCG supertagging.

For each task, we underline the best performance for each training dataset size and bold the best overall performance.

<|TLDR|>

@highlight

Representations from language models consistently perform better than translation encoders on syntactic auxiliary prediction tasks.