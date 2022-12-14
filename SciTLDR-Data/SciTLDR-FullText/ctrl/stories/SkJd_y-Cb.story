Word embeddings extract semantic features of words from large datasets of text.

Most embedding methods rely on a log-bilinear model to predict the occurrence of a word in a context of other words.

Here we propose word2net, a method that replaces their linear parametrization with neural networks.

For each term in the vocabulary, word2net posits a neural network that takes the context as input and outputs a probability of occurrence.

Further, word2net can use the hierarchical organization of its word networks to incorporate additional meta-data, such as syntactic features, into the embedding model.

For example, we show how to share parameters across word networks to develop an embedding model that includes part-of-speech information.

We study word2net with two datasets, a collection of Wikipedia articles and a corpus of U.S. Senate speeches.

Quantitatively, we found that word2net outperforms popular embedding methods on predicting held- out words and that sharing parameters based on part of speech further boosts performance.

Qualitatively, word2net learns interpretable semantic representations and, compared to vector-based methods, better incorporates syntactic information.

Word embeddings extract semantic features of words from large datasets of text.

Most embedding methods rely on a log-bilinear model to predict the occurrence of a word in a context of other words.

Here we propose word2net, a method that replaces their linear parametrization with neural networks.

For each term in the vocabulary, word2net posits a neural network that takes the context as input and outputs a probability of occurrence.

Further, word2net can use the hierarchical organization of its word networks to incorporate additional meta-data, such as syntactic features, into the embedding model.

For example, we show how to share parameters across word networks to develop an embedding model that includes part-of-speech information.

We study word2net with two datasets, a collection of Wikipedia articles and a corpus of U.S. Senate speeches.

Quantitatively, we found that word2net outperforms popular embedding methods on predicting heldout words and that sharing parameters based on part of speech further boosts performance.

Qualitatively, word2net learns interpretable semantic representations and, compared to vector-based methods, better incorporates syntactic information.

Word embeddings are an important statistical tool for analyzing language, processing large datasets of text to learn meaningful vector representations of the vocabulary BID0 BID1 BID11 BID16 .

Word embeddings rely on the distributional hypothesis, that words used in the same contexts tend to have similar meanings BID4 .

More informally (but equally accurate), a word is defined by the company it keeps BID2 .While there are many extensions and variants of embeddings, most rely on a log-bilinear model.

This model posits that each term is associated with an embedding vector and a context vector.

Given a corpus of text, these vectors are fit to maximize an objective function that involves the inner product of each observed word's embedding with the sum of the context vectors of its surrounding words.

With useful ways to handle large vocabularies, such as negative sampling BID10 or Bernoulli embeddings BID19 , the word embedding objective resembles a bank of coupled linear binary classifiers.

Here we introduce word2net, a word embedding method that relaxes this linear assumption.

Word2net still posits a context vector for each term, but it replaces each word vector with a term-specific neural network.

This word network takes in the sum of the surrounding context vectors and outputs the occurrence probability of the word.

The word2net objective involves the output of each word's network evaluated with its surrounding words as input.

The word2net objective resembles a bank of coupled non-linear binary classifiers.

How does word2net build on classical word embeddings?

The main difference is that the word networks can capture non-linear interaction effects between co-occurring words; this leads to a better model of language.

Furthermore, the word networks enable us to share per-term parameters based on word-level meta-data, such as syntactic information.

Here we study word2net models that share parameters based on part-of-speech ( ) tags, where the parameters of certain layers of each network are shared by all terms tagged with the same tag.

FIG0 illustrates the intuition behind word2net.

Consider the term .

The top of the figure shows one observation of the word, i.e., one of the places in which it appears in the data.

(This excerpt is from U.S. Senate speeches.)

From this observation, the word2net objective contains the probability of a binary variable w n; conditional on its context (i.e., the sum of the context vectors of the surrounding words).

This variable is whether occurred at position n.... were opposed to the increase of the circulation of ...

Under review as a conference paper at ICLR 2018 DISPLAYFORM0 cut n doubling v amount n increases v saving n supply v limit n cost v amounts n estimates v decreases n cut v increases n raising v decrease n amount v declines n half adj amounting n The most similar word/tag pairs for the former include mostly verbs such as "decrease," while for the latter the most similar results are nouns such as "cut." neural network that outputs the probability of that word ( FIG0 ).

If we are given the tags of the words, we may use parameter sharing instead in order to form a per-word per-tag neural network ( FIG0 .

Finally, we also propose a method for computing similarities between the neural network representations of the words and demonstrate that they capture semantic (and even syntactic) similarities FIG0 ).In our empirical study, we show that parameter sharing in word2net performs better than applying word2vec or standard Benoulli embeddings on the augmented vocabulary of word/tag pairs.

We also demonstrate that deep Bernoulli embeddings provide better predictive log-likelihood when compared to word2vec or standard Bernoulli embeddings.

R fjrr: moved this to the introduction, needs rewriting here Word embedding models learn semantic features of words by exploiting the co-occurrence patterns of words in a collection of documents.

There are many extensions and variants of word embeddings BID0 BID1 BID13 BID10 b; c; BID16 BID15 BID14 BID7 BID22 Barkan, 2016; Bamler & Mandt, 2017) .

Most of these approaches rely on a log-bilinear model, in which the emission probabilities depend on a dot product of the word embedding vectors and the context vectors, as opposed to the deep neural network architectures proposed by BID0 BID1 and BID13 .

Our model di ers from these deep neural network architectures in two ways.

First, we have a separate network for each vocabulary word, instead of a single network that outputs the logits for all words in the vocabulary.

Our perspective of a bank of parallel binary classification problems allows for faster optimization of the networks.

Second, our architecture enables incorporating side information (such as part of speech tags) in specific layers of the network.

Recall that word embeddings (without any further structure) tend to capture semantic properties of the words, and the syntactic properties they encode are typically redundant (Andreas & Klein, 2014) , so there is room for improvement with a model that allows for additional syntactic structure.

We adopt the perspective of exponential family embeddings BID19 , which extend word embeddings to datasets beyond text.

There are also some variants and extensions of exponential family embeddings BID8 BID8 , but they all have in common an exponential family likelihood whose natural parameter is determined The idea behind word2net is that the conditional probability of w n;is the output of a multi-layer network that takes the context as input.

Each layer of the network transforms the context into a new hidden representation, reweighting the latent features according to their relevance for predicting the occurrence of .

Note that not illustrated are the 0-variables, i.e., the negative samples, which correspond to words that are not at position n. In word2net, their probabilities also come from their corresponding word networks.

Now suppose we have tagged the corpus with .

FIG0 shows how to incorporate this syntactic information into word2net.

The network is specific to as a noun (as opposed to a verb).

The parameters of the first layer (orange) are shared among all nouns in the collection; the other layers (blue) are specific to .

Thus, the networks for / and / differ in how the first layer promotes the latent aspects of the context, i.e., according to which context features are more relevant for each tag.

This model further lets us consider these two tags separately.

FIG0 shows the most similar words to each sense of ; the method correctly picks out tagged words related to the verb and related to the noun.

Below, we develop the details of word2net and study its performance with two datasets, a collection of Wikipedia articles and a corpus of U.S. Senate speeches.

We found that word2net outperforms popular embedding methods on predicting held-out words, and that sharing parameters based on further boosts performance.

Qualitatively, word2net learns interpretable semantic representations and, compared to vector-based methods, better incorporates syntactic information.

Related work.

Word2net builds on word embeddings methods.

Though originally designed as deep neural network architectures BID0 BID1 BID13 , most applications of word embeddings now rely on log-bilinear models BID10 b; c; BID16 BID15 BID14 BID7 BID22 Barkan, 2016; Bamler & Mandt, 2017) .

The key innovation behind word2net is that it represents words with functions, instead of vectors BID21 or distributions BID22 .

Word2net keeps context vectors, but it replaces the embedding vector with a neural network.

Previous work has also used deep neural networks for word embeddings BID0 BID1 BID13 ; these methods use a single network that outputs the unnormalized log probabilities for all words in the vocabulary.

Word2net takes a different strategy: it has a separate network for each vocabulary word.

Unlike the previous methods, word2net's approach helps maintain the objective as a bank of binary classifiers, which allows for faster optimization of the networks.

To develop word2net, we adopt the perspective of exponential family embeddings BID19 , which extend word embeddings to data beyond text.

There are several extensions to exponential family embeddings BID8 , but they all have in common an exponential family likelihood whose natural parameter has a log-bilinear form.

Word2net extends this framework to allow for non-linear relationships.

Here we focus on Bernoulli embeddings, which are related to word embeddings with negative sampling, but our approach easily generalizes to other exponential family distributions (e.g., Poisson).Finally, word embeddings can capture semantic properties of the word, but they tend to neglect most of the syntactic information (Andreas & Klein, 2014) .

Word2net introduces a simple way to leverage the syntactic information to improve the quality of the word representations.

In this section we develop word2net as a novel extension of Bernoulli embeddings BID19 .

Bernoulli embeddings are a conditional model of text, closely related to word2vec.

Specifically, they are related to continuous bag-of-words ( ) with negative sampling.

for each unique term in the vocabulary, v D 1; : : : ; V .

These vectors encode the semantic properties of words, and they are used to parameterize the conditional probability of a word given its context.

Specifically, let w n be the V -length one-hot vector indicating the word at location n, such that w nv D 1 for one term (vocabulary word) v, and let c n be the indices of the words in a fixed-sized window centered at location n (i.e., the indices of the context words).

Exponential family embeddings parameterize the conditional probability of the target word given its context via a linear combination of the embedding vector and the context vectors, DISPLAYFORM0 Here, .x/ D 1 1Ce x is the sigmoid function, and we have introduced the notation??n for the sum of the context vectors at location n. Note that Eq. 1 does not impose the constraint that the sum over the vocabulary words P v p.w nv D 1 j c n / must be 1.

This significantly alleviates the computational complexity BID11 BID19 .

This type of exponential family embedding is called Bernoulli embedding, named for its conditional distribution.

In Bernoulli embeddings, our goal is to learn the embedding vectors v and the context vectors??v from the text by maximizing the log probability of words given their contexts.

The data contains N pairs .w n ; c n / of words and their contexts, and thus we can form the objective function L. ;??/ as the sum of log p.w nv j c n / for all instances and vocabulary words.

The resulting objective can be seen as a bank of V binary classifiers, where V is the vocabulary size.

To see that, we make use of Eq. 1 and express the objective L. ;??/ as a sum over vocabulary words, DISPLAYFORM1 If we hold all the context vectors??v fixed, then Eq. 2 is the objective of V independent logistic regressors, each predicting whether a word appears in a given context or it does not.

The positive examples are those where word v actually appeared in a given context; the negative examples are those where v did not appear.

It is the context vectors that couple the V binary classifiers together.

In practice, we need to either downweight the contribution of the zeros in Eq. 2, or subsample the set of negative examples for each n BID19 .

We follow the latter case here, which leads to negative sampling BID11 DISPLAYFORM2 where f .

I??v/ W R K !

R is a feed-forward neural network with parameters (i.e., weights and intercepts)??v.

The number of neurons of the input layer is K, equal to the length of the context vectors??v.

Essentially, we have replaced the per-term embedding vectors v with a per-term neural network??v.

We refer to the per-term neural networks as word networks.

The word2net objective is the sum of the log conditionals, DISPLAYFORM3 where we choose the function f .

I??v/ to be a three-layer neural network, DISPLAYFORM4 Replacing vectors with neural networks has several implications.

First, the bank of binary classifiers has additional model capacity to capture nonlinear relationships between the context and the cooccurrence probabilities.

Specifically, each layer consecutively transforms the context to a different representation until the weight matrix at the last layer can linearly separate the real occurrences of the target word from the negative examples.

Second, for a fixed dimensionality K, the resulting model has more parameters.3 This increases the model capacity, but it also increases the risk of overfitting.

Indeed, we found that without extra regularization, the neural networks may easily overfit to the training data.

We regularize the networks via either weight decay or parameter sharing (see below).

In the empirical study of Section 3 we show that word2net fits text data better than its shallow counterparts and that it captures semantic similarities.

Even for infrequent words, the learned semantic representations are meaningful.

Third, we can exploit the hierarchical structure of the neural network representations via parameter sharing.

Specifically, we can share the parameters of a specific layer of the networks of different words.

This allows us to explicitly account for tags in our model (see below).Regularization through parameter sharing enables the use of tags.

One way to regularize word2net is through parameter sharing.

For parameter sharing, each word is assigned to one of T groups.

Importantly, different occurrences of a term may be associated to different groups.

We share specific layers of the word networks among words in the same group.

In this paper, all neural network representations have 3 layers.

We use index`2 f1; 2; 3g to denote the layer at which we apply the parameter sharing.

Then, for each occurrence of term v in group t we set??.`/ v D??.`/ t .

Consider now two extreme cases.

First, for T D 1 group, we have a strong form of regularization by forcing all word networks to share the parameters of layer`.

The number of parameters for layer`has been divided by the vocabulary size, which implies a reduction in model complexity that might help prevent overfitting.

This parameter sharing structure does not require side information and hence can be applied to any text corpus.

In the second extreme case, each word is in its own group and T D V .

This set-up recovers the model of Eqs. 4 and 5, which does not have parameter sharing.

When we have access to a corpus annotated with tags, parameter sharing lets us use the information to improve the capability of word2net by capturing the semantic structure of the data.

Andreas & Klein (2014) have shown that word embeddings do not necessarily encode much syntactic information, and it is still unclear how to use syntactic information to learn better word embeddings.

The main issue is that many words can appear with different tags; for example, can be both a and refer to the animal or a and refer to the activity of catching the animal.

On the one hand, both meanings are related.

On the other hand, they may have differing profiles of which contexts they appear in.

Ideally, embedding models should be able to capture the difference.

However, the simple approach of considering / and / as separate terms fails because there are few occurrences of each individual term/tag pair. (We show that empirically in Section 3.)Exploiting the hierarchical nature of the network representations of word2net, we incorporate information through parameter sharing as follows.

Assume that for location n in the text we have a one-hot vector s n 2 f0; 1g T indicating the tag.

To model the observation at position n, we use a neural network specific to that term/tag combination, DISPLAYFORM5 That is, the neural network parameters are combined to form a neural network in which layer`has parameters??.`/ t and the other layers have parameters??.:`/ v. Thus, we leverage the information about the tag t by replacing??.`/ v with??.`/ t in layer`, resulting in parameter sharing at that layer.

If the same term v appears at a different position n 0 with a different tag t 0 , at location n 0 we replace the parameters??.`/ v of layer`with??.`/ t 0 .

FIG0 illustrates parameter sharing at`D 1.Even though now we have a function f .

/ for each term/tag pair, the number of parameters does not scale with the product V T ; indeed the number of parameters of the network with information is smaller than the number of parameters of the network without side information (Eq. 5).

The reason is that the number of parameters necessary to describe one of the layers has been reduced from V to T due to parameter sharing (the other layers remain unchanged).

Finally, note that we have some flexibility in choosing which layer is tag-specific and which layers are word-specific.

We explore different combinations in Section 3, where we show that word2net with information improves the performance of word2net.

The parameter sharing approach extends to side information beyond tags, as long as the words can be divided into groups, but we focus on parameter sharing across all words (T D 1) or across tags.

Semantic similarity of word networks.

In standard word embeddings, the default choice to compute semantic similarities between words is by cosine distances between the word vectors.

Since word2net replaces the word vectors with word networks, we can no longer apply this default choice.

We next describe the procedure that we use to compute semantic similarities between word networks.

After fitting word2net, each word is represented by a neural network.

Given that these networks parameterize functions, we design a metric that accounts for the fact that two functions are similar if they map similar inputs to similar outputs.

So the intuition behind our procedure is as follows: we consider a set of K-dimensional inputs, we evaluate the output of each neural network on this set of inputs, and then we compare the outputs across networks.

For the inputs, we choose the V context vectors, which we stack together into a matrix??2 R V K. We evaluate each network f .

/ row-wise on??(i.e., feeding each??v as a K-dimensional input to obtain a scalar output), obtaining a V -dimensional summary of where the network f .

/ maps the inputs.

Finally, we use the cosine distance of the outputs to compare the outputs across networks.

In summary, we obtain the similarity of two words w and v as DISPLAYFORM6 If we are using parameter sharing, we can also compare -tagged words; e.g., we may ask how similar is / to / .

The two combinations will have different representations under the word2net method trained with -tag sharing.

Assuming that layer`is the shared layer, we compute the semantic similarity between the word/tag pair OEw; t and the pair OEv; s as dist.

OEw; t ; OEv; s/ D f .??I??.:`/

w ;??.`/ t / > f .??I??.:`/ v ;??.`/ s / jjf .??I??.:`/ w ;??.`/ t /jj 2 jjf .??I??. DISPLAYFORM7

In this section we study the performance of word2net on two datasets, Wikipedia articles and Senate speeches.

We show that word2net fits held-out data better than existing models and that the learned network representations capture semantic similarities.

Our results also show that word2net is superior at incorporating syntactic information into the model, which improves both the predictions and the quality of the word representations.

Data.

We use word2net to study two data sets, both with and without tags:Wikipedia: The text8 corpus is a collection of Wikipedia articles, containing 17M words.

We form a vocabulary with the 15K most common terms, replacing less frequent terms with the token.

We annotate text8 using the tagger and the universal tagset.4 Table 7 in Appendix C shows a description of the tagset.

We also form a tagged dataset in which each term/tag combination has a unique token, resulting in a vocabulary of 49K tagged terms.

Senate speeches: These are the speeches given in the U.S. Senate in the years .

The data is a transcript of spoken language and contains 24M words.

Similarly as above, we form a vocabulary of 15K terms.

We annotate the text using the Stanford CoreNLP tagger , and we map the tags to the universal tagset.

We form a tagged dataset with 38K tagged terms.

TAB1 summarizes the information about both corpora.

We split each dataset into a training, a validation, and a test set, which respectively contain 90%, 5%, and 5% of the words.

Additional details on preprocessing are in Appendix C.

We compare word2net to its shallow counterpart, the model BID11 , which is equivalent to Bernoulli embeddings ( -) 5 BID19 .

We also compare with the skip-gram model.

DISPLAYFORM0 The shallow models have 2K parameters per term (the entries of the context and word vectors).

Since we want to compare models both in terms of context dimension K and in terms of total parameters, we fit the methods with K 2 f20; 165; 100; 1260g.

We experiment with context sizes jc n j 2 f2; 4; 8g and we train all methods using stochastic gradient descent ( ) BID17 with jS n j D 10 negative samples on the Wikipedia data and with jS n j D 20 negative samples on the Senate speeches.

We use regularization with standard deviation 10 for the word and context vectors, as well as weight decay for the neural networks.

We use Adam BID6 with Tensorflow's default settings (Abadi et al., 2016) to train all methods for up to 30000 iterations, using a minibatch size of 4069 or 1024.

We assess convergence by monitoring the loss on a held-out validation set every 50 iterations, and we stop training when the average validation loss starts increasing.

We initialize and freeze the context vectors of the word2net methods with the context vectors from a pretrained Bernoulli embedding with the same context dimension K. Network parameters are initialized according to standard initialization schemes of tags.

We compare models with the same context dimension K and the same total number of parameters p=V for different context sizes (cs). (Results on more configurations are in Appendix A.) For word2net, we study different parameter sharing schemes, and the color coding indicates which layer is shared and how, as in FIG0 .

Parameter sharing improves the performance of word2net, especially with tags.

feed-forward neural networks BID3 , i.e., the weights are initialized from a uniform distribution with bounds??p6= p H in C H out .

Quantitative results: Word2net has better predictive performance.

We compute the predictive log-likelihood of the words in the test set, log p.w nv j c n /.

For skip-gram, which was trained to predict the context words from the target, we average the context vectors??v for a fair comparison.

7 TAB2 shows the results for the Wikipedia dataset.

We explore different model sizes: with the same number of parameters as word2net, and with the same dimensionality K of the context vectors.

For word2net, we explore different parameter sharing approaches.

TAB5 in Appendix A shows the results for other model sizes (including K D 100).

In both tables, word2net without parameter sharing performs at least as good as the shallow models.

Importantly, the performance of word2net improves with parameters sharing, and it outperforms the other methods.

Tables 2 and 5 also show that -/ and skip-gram perform poorly when we incorporate information by considering an augmented vocabulary of tagged words.

The reason is that each term becomes less frequent, and these approaches would require more data to capture the cooccurrence patterns of tagged words.

In contrast, word2net with parameter sharing provides the best predictions across all methods (including other versions of word2net).Finally, TAB6 in Appendix A shows the predictive performance for the U.S. Senate speeches.

On this corpus, skip-gram performs better than -/ and word2net without parameter sharing; however, word2net with sharing also provides the best predictions across all methods.

Qualitative results: Word2net captures similarities and leverages syntactic information.

TAB3 displays the similarity between word networks (trained on Wikipedia with parameter sharing at layer D 1), compared to the similarities captured by word embeddings ( -/ ).

For each query word, we list the three most similar terms, according to the learned representations.

The word vectors are compared using cosine similarity, while the word networks are compared using Eq. 7.

The table shows that word2net can capture latent semantics, even for less frequent words such as .

TAB4 shows similarities of models trained on the Senate speeches.

In particular, the table compares:-/ without information, -/ trained on the augmented vocabulary of tagged words, and word2net with parameter sharing at the input layer (`D 1).

We use Eq. 8 to compute the similarity across word networks with sharing.

We can see that word2net is superior at incorporating syntactic information into the learned representations.

For example, the most similar 7 If we do not average, the held-out likelihood of skip-gram becomes worse.

networks to the pronoun are other pronouns such as , , and .

Word networks are often similar to other word networks with the same tag, but we also see some variation.

One such example is in FIG0 , which shows that the list of the 10 most similar words to the verb contains the adjective .

We have presented word2net, a method for learning neural network representations of words.

The word networks are used to predict the occurrence of words in small context windows and improve prediction accuracy over existing log-bilinear models.

We combine the context vectors additively, but this opens the door for future research directions in which we explore other ways of combining the context information, such as accounting for the order of the context words and their tags.

We have also introduced parameter sharing as a way to share statistical strength across groups of words and we have shown empirically that it improves the performance of word2net.

Another opportunity for future work is to explore other types of parameter sharing besides sharing, such as sharing layers across documents or learning a latent group structure together with the word networks.

For completeness, we show here some additional results that we did not include in the main text for space constraints.

In particular, TAB5 compares the test log-likelihood of word2net with the competing modelsnamely, skip-gram and -/ .

All methods are trained with negative sampling, as described in the main text.

This table shows the results for the Wikipedia dataset, similarly to TAB2 , but it includes other model sizes (i.e., another value of K).

In this table, word2net with no parameter sharing performs similarly to -/ with the same number of parameters, but its performance can be further improved with part-of-speech ( ) parameter sharing.

TAB6 shows the test log-likelihood for the U.S. Senate speeches.

Here, skip-gram is the best method that does not use tags, but it is outperformed by word2net with parameter sharing.

Word2vec BID11 ) is one of the most widely used method for learning vector representations of words.

There are multiple ways to implement word2vec.

First, there is a choice of the objective.

Second, there are several ways of how to approximate the objective to get a scalable algorithm.

In this section, we describe the two objectives, continuous bag-of-words ( ) and skip-gram, and we focus on negative sampling as the method of choice to achieve scalability.

We describe the similarities and differences between Bernoulli embeddings BID19 and these two objectives.

In summary, under certain assumptions Bernoulli embeddings are equivalent to with negative sampling, and are related to skip-gram through Jensen's inequality.-?? (negative sampling)First we explain how Bernoulli embeddings and with negative sampling are related.

Consider the Bernoulli embedding full objective, DISPLAYFORM0 log .

DISPLAYFORM1 In most cases, the summation over negative examples (w nv D 0) is computationally expensive to compute.

To address that, we form an unbiased estimate of that term by subsampling a random set S n Here, we have introduced an auxiliary coefficient .

The estimate is unbiased only for D 1; however, BID19 showed that downweighting the contribution of the zeros works better in practice.

There are two more subtle theoretical differences between both.

The first difference is that Bernoulli embeddings include a regularization term for the embedding vectors, whereas does not.

The second difference is that, in Bernoulli embeddings, we need to draw a new set of negative samples S n at each iteration of the gradient ascent algorithm (because we form a noisy estimator of the downweighted objective).

In contrast, in with negative sampling, the samples S n are drawn once in advance and then hold fixed.

In practice, for large datasets, we have not observed significant differences in the performance of both approaches.

For simplicity, we draw the negative samples S n only once.(negative sampling) skip-gram (negative sampling) Now we show how and skip-gram are related (considering negative sampling for both).

Recall that the objective of is to predict a target word from its context, while the skip-gram objective is to predict the context from the target word.

Negative sampling breaks the multi-class constraint that the sum of the probability of each word must equal one, and instead models probabilities of the individual entries of the one-hot vectors representing the words.

When we apply negative sampling, the objective becomes Eq. 11.

The skip-gram objective is given by

<|TLDR|>

@highlight

Word2net is a novel method for learning neural network representations of words that can use syntactic information to learn better semantic features.

@highlight

This paper extends SGNS with an architectural change from a bag-of-words model to a feedforward model, and contributes a new form of regularization by tying a subset of layers between different associated networks.

@highlight

A method to use non-linear combination of context vectors for learning vector representation of words, where the main idea is to replace each word embedding by a neural network.