Neural conversational models are widely used in applications like personal assistants and chat bots.

These models seem to give better performance when operating on word level.

However, for fusion languages like French, Russian and Polish vocabulary size sometimes become infeasible since most of the words have lots of word forms.

We propose a neural network architecture for transforming normalized text into a grammatically correct one.

Our model efficiently employs correspondence between normalized and target words and significantly outperforms character-level models while being 2x faster in training and 20\% faster at evaluation.

We also propose a new pipeline for building conversational models: first generate a normalized answer and then transform it into a grammatically correct one using our network.

The proposed pipeline gives better performance than character-level conversational models according to assessor testing.

Neural conversational models BID18 are used in a large number of applications: from technical support and chat bots to personal assistants.

While being a powerful framework, they often suffer from high computational costs.

The main computational and memory bottleneck occurs at the vocabulary part of the model.

Vocabulary is used to map a sequence of input tokens to embedding vectors: one embedding vector is stored for each word in vocabulary.

English is de-facto a standard language for training conversational models, mostly for a large number of speakers and simple grammar.

In english, words usually have only a few word forms.

For example, verbs may occur in present and past tenses, nouns can have singular and plural forms.

For many other languages, however, some words may have tens of word forms.

This is the case for Polish, Russian, French and many other languages.

For these languages storing all forms of frequent words in a vocabulary significantly increase computational costs.

To reduce vocabulary size, we propose to normalize input and output sentences by putting them into a standard form.

Generated texts can then be converted into grammatically correct ones by solving morphological agreement task.

This can be efficiently done by a model proposed in this work.

Our contribution is two-fold:• We propose a neural network architecture for performing morphological agreement in fusion languages such as French, Polish and Russian (Section 2).• We introduce a new approach to building conversational models: generating normalized text and then performing morphological agreement with proposed model (Section 3);

In this section we propose a neural network architecture for solving morphological agreement problem.

We start by formally defining the morphological agreement task.

Consider a grammatically correct sentence with words [a 1 , a 2 , . . .

, a K ].

Let S(a) be a function that maps any word to its standard form.

For example, S("went") = "go".

Goal of morphological agreement task is to learn a mapping from normalized sentence [S(a 1 ), S(a 2 ), . . .

, S(a K )] = [a n 1 , ,a n 2 , . . .

, a n K ] to initial sentence [a 1 , a 2 , . . . , a K ].

Interestingly, reverse mapping may be performed for each word independently using specialized dictionaries.

Original task, however, needs to consider dependencies between words in sequence in order to output a coherent text.

An important property of this task is that the number of words, their order and meaning are explicitly contained in input sequence.

To employ this knowledge, we propose a specific neural network architecture illustrated in FIG0 .

Network operates as follows: first all normalized words are embedded using the same character-level LSTM encoder.

The goal of the next step is to incorporate global information about other words to embedding of each word.

To do so we pass word embedding sequence through a bidirectional LSTM BID7 .

This allows new embeddings to get information from all other words: information about previous words is brought by forward LSTM and information about subsequent words is taken from backward LSTM.

Finally, new embeddings are decoded with character-level LSTM decoders.

At this stage we also added attention BID1 over input characters of corresponding words for better performance.

High-level overview of this model resembles sequence-to-sequence BID17 network: model learns to map input characters to output characters for each word using encoder-decoder sceme.

The difference is that bidirectional neural network is used to distribute information between different words.

Unlike simple character-level sequence-to-sequence model, our architecture allows for much faster evaluation, since encoding and decoding phases can be done in parallel for each word.

Also, one of the main advantages of our approach is that information paths from inputs to outputs are much shorter which leads to slower performance degradation as input length increases (see Section 5.3).

As discussed above, we propose a two stage approach to build a neural conversational model: first generate normalized answer using normalized question and then apply Concorde model to obtain grammatically correct response.

In this section we discuss a modification of Concorde model for conditioning its output on question's morphological features.

A modification of Concorde model that we call Q-Concorde uses two sources of input: question and normalized answer.

Question is first embedded into a single vector with character-level RNN.

This vector may carry important morphological information such as time, case and plurality of questions.

Question embedding is then mixed with answer embeddings using linear mapping.

The final model is shown in FIG2 .

DISPLAYFORM0

Most frequently used models for sequence to sequence mapping rely on generating embedding vector that contains all information about input sequence.

Reconstruction solely from this vector usually results in worse performance as length of the output sequence increases.

Attention BID19 , BID14 , BID1 ) partially fixes this problem, though information bottleneck of embedding vector is still high.

Encoder-decoder models have mostly been applied to tasks like speech recognition BID6 , BID7 , BID5 ), machine translation BID1 , BID17 , BID2 ) and neural conversational models BID18 , BID16 ).Some works have tried to perform decomposition of input sequence to obtain shorter information paths.

In (Johansen et al. (2016) ) input is first processed character-wise and then embeddings that correspond to word endings are used for sequence-to-sequence model.

This modification makes input-output information paths shorter which leads to better performance.

Word inflection is a most similar task to ours, but in this task model is asked to generate a specific word form while we want our model to automatically select desired word forms.

BID3 proposed a supervised approach to predicting the set of all word forms by generating transformation rules from known inflection tables.

They also propose to use Conditional Random Fields for unseen base forms.

Some authors have also tried to apply neural networks for this problem.

BID0 and BID4 propose to use bidirectional LSTM to encode the word.

Then BID4 uses different decoders for different word forms, while BID0 suggests to have one decoder and to attach morphological features to its input.

Besides recurrent networks, there has been an attempt to use convolutional networks.

BID15 based his work on BID4 and proposed to first pass raw data through convolutional layers and then to pass them through recurrent encoder.

BID9 BID11 library.

We leave first 20 words from each sentence to reduce computational costs.

Concorde and Q-Concorde models consist of 2-layer LSTM encoder and decoder with hidden size 512.

We compare our model to three baselines: unigram charRNN, bigram charRNN and hierarchical model.

Unigram and bigram models are standard sequence-to-sequence models with attention BID14 that operate with characters or pairs of characters as tokens.

We use 2-layer LSTM as an encoder.

Decoder consists of 2-layer LSTM followed by attention layer and another recurrent layer.

The third baseline is a hierarchical model motivated by Johansen et al. (2016) : we first embed each word using recurrent encoder and then compute sentence embedding by running word-level encoder on these embeddings.

For baselines we use layer size of 768 which results in a comparable number of parameters for all models.

We train models with Adam BID10 optimizer in batch size 16 with learning rate 0.0002 that halves after each 50k updates.

We terminate training after 300k updates which is enough for all models to converge.

To evaluate our model we used French, Russian an Polish corpuses from OpenSubtitles 2 database.

We performed morphological agreement for each subtitle line independently.

We estimated potential vocabulary size reduction from normalization by selecting words that appeared more than 10 times in first 10M examples.

This lead to 2.5 times reduction for Polish language, 2.4 for Russian, and 1.8 for French.

We evaluated our model in two metrics: word and sentence accuracies.

Word accuracy shows a fraction of words that were correctly generated.

Sentence accuracy corresponds to the fraction of sentences that were transformed without any mistake.

Results are reported in TAB1 .

From four models that we compared, our model gave the best performance among all datasets, while second best model was hierarchical model.

We inspected our model to show some examples where it was able to infer plural form and gender for unseen words TAB2 ).

For Russian language we found out that the model was able to learn some rare rules like changing the letter я to й when going to plural form in some words: один заяц , два зайца (one rabbit, two rabbits).We can also see that our model can infer gender from words.

To show that we chose feminine, masculine and neuter words and asked the model to perform agreement with word "one".

This word changes its spelling for different genders in French, Polish, and Russian.

Results presented in TAB3 suggest that model can indeed correctly solve this task by setting correct gender to the numeral.

TAB4 we show results on full sentences.

Interestingly, on quite a complex Russian example our model was able to perform agreement.

To select the correct form of word соседнем (neighbouring), network had to use multiple markers from different parts of a sentence: gender from подъезд (entrance) and case from в (in).

As a motivation for our model we argued that making shorter input-output paths may reduce information load of the embedding vector.

To check this hypothesis we computed average sentence accuracy for different input lengths and reported results in FIG2 .We can clearly see that all baseline models perform worse as the input length increases.

However, this is not the case for our model -while character-level models perform with almost 0% accuracy when input is 100 characters long, our model still gives similar performance as for short sentences.

This result can be explained by the way in which models use embedding vectors.

Baseline models have to share embedding capacity between all words in a sentence.

Our model, however, has a separate embedding for each word and does not require the whole sentence to be squeezed into one vector.

It is also clear that character-level models perform better for short sequences (about 33% of the test set).

This may be the case since the capacity of the embedding vector is not fully used for them.

Despite being worse for short inputs, our model can still handles many important cases quite well including those discussed in Section 5.2.

To evaluate our conversation model we constructed a corpus of question-answer pairs from web site Otvet.mail.ru -general topic Russian service for questions and answers (analogue of Quora.com).

The uniqueness of this corpus is that it contains general knowledge questions that allow the trained model to answer questions about movies, capitals, etc.

This requires many rare entity-related words to be in the vocabulary which makes it extremely large without normalization.

First we compared Q-Concorde and Concorde models to show that Q-Concorde can indeed grasp important morphological features form a context.

We also trained baseline models with context concatenated to input sentence (with a special delimiter in between).

word and sentence accuracies are reported in TAB5 .

Again, Concorde model was able to outperform baselines even though it didn't have access to the context.

Also, Q-Concorde model was able to improve Concorde's performance.

We inspected cases on which Q-Concorde model showed better performance than Concorde TAB6 ).

In example 1 of this table, question was asked about a single object.

Q-Concorde model used singular form, while Concorde used plural.

Q-Concorde was also able to successfully carry correct case (example 2) and time (example 3) from the question.

Some mistakes made by Q-Concorde model are shown in TAB7 .

In example 1, Q-Concorde wasn't able to decide whether to use polite form or not and used one word in a less polite form than another.

An important property of Q-Concorde model is that it can generate different texts depending on lexical features of a question.

For example, in TAB8 we changed question's tense from present simple to past simple ("what do you do?" and "what did you do?").

The model correctly generated answer in corresponding tense.

We also tried to change gender of a word "did" in a question: from masculine делал ) to feminine далала .

Our model used the correct gender to generate the answer.

While model generated grammatically correct answers in all three cases, in the third case (past simple, masculine) model answered in less common form with a meaning that differs from expected one.

Finally, we apply Q-Concorde model to a proposed pipeline for training conversational models.

We compare our model with a 3-layer character-level sequence-to-sequence model which was trained on grammatically correct sentences.

For generating diverse answers we train two models: one to predict answer given question and the other to predict question given answer, as suggested in BID16 .

This allows us to discard answers that are too general.

To compare two models we set an experiment environment where assessors were asked to select one of two possible answers to the given question: one was generated by character-wise model and another was generated by our pipeline.

Assessors did not know the order in which cases were shown and so they did not know which model generated the text.

In 62.1% cases assessors selected the proposed model, and in the remaining 37.9% assessors preferred character-wise model.

We noticed that time for processing one batch is much higher for character-level models since they need to process longer sequences sequentially.

In TAB9 we report time for forward and backward pass of one batch (16 objects) and other important computational characteristics.

We measured this time on GeForce GTX TITAN X graphic card.

It turns out that proposed models have comparable evaluation time, but train faster than unigram and hierarchical models.

In this paper we proposed a neural network model that can efficiently employ relationship between input and output words in morphological agreement task.

We also proposed a modification for this model that uses context sentence.

We apply this model for neural conversational model in a new pipeline: we use normalized question to generate normalized answer and then apply proposed model to obtain grammatically correct response.

This model showed better performance than character level neural conversational model based on assessors responses.

We achieved significant improvement comparing to character-level, bigram and hierarchical sequenceto-sequence models on morphological agreement task for Russian, French and Polish languages.

Trained models seem to understand main grammatical rules and notions such as tenses, cases and pluralities.

@highlight

Proposed architecture to solve morphological agreement task