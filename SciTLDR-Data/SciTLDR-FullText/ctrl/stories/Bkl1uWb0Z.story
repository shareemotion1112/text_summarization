Previous work has demonstrated the benefits of incorporating additional linguistic annotations such as syntactic trees into neural machine translation.

However the cost of obtaining those syntactic annotations is expensive for many languages and the quality of unsupervised learning linguistic structures is too poor to be helpful.

In this work, we aim to improve neural machine translation via source side dependency syntax but without explicit annotation.

We propose a set of models that learn to induce dependency trees on the source side and learn to use that information on the target side.

Importantly, we also show that our dependency trees capture important syntactic features of language and improve translation quality on two language pairs En-De and En-Ru.

Sequence to sequence (seq2seq) models have exploded in popularity due to their apparent simplicity and yet surprising modeling strength.

The basic architecture cleanly extends the standard machine learning paradigm wherein some function f is learned to map inputs to outputs x → y to the case where x and y are natural language strings.

In its most basic form, an input is summarized by a recurrent neural network into a summary vector and then decoded into a sequence of observations.

These models have been strengthened with attention mechanisms BID0 , and variational dropout BID9 , in addition to important advances in expressivity via gating like Long Short-Term Memory (LSTM) cells BID11 and advanced gradient optimizers like Adam BID15 .Despite these impressive advances, the community has still largely been at a loss to explain how these models are so successful at a wide range of linguistic tasks.

Recent work has shown that the LSTM captures a surprising amount of syntax BID21 , but this is evaluated via downstream tasks designed to test the model's abilities not its representation.

Simultaneously, recent research in neural machine translation (NMT) has shown the benefit of modeling syntax explicitly using parse trees BID1 BID19 BID8 rather than assuming the model will automatically discover and encode it.

BID19 present a mixed encoding of words and a linearized constituency-based parse tree of the source sentence.

BID1 propose to use Graph Convolution to encode source sentences given their dependency links and attachment labels.

In this work, we attempt to contribute to both modeling syntax and investigating a more interpretable interface for testing the syntactic content of a new seq2seq model's internal representation and attention.

We achieve this by augmenting seq2seq with a gate that allows the model to decide between syntactic and semantic objectives.

The syntactic objective is encoded via a syntactic structured attention (Section §3) from which we can extract dependency trees.

Our goal is to have a model which reaps the benefits of syntactic information (i.e. parse trees) without requiring explicit annotation.

In this way, learning the internal representation of our model is a cousin to work done in unsupervised grammar induction except that by focusing on translation we require both syntactic and semantic knowledge.

The semantic objective is the word translation prediction.

It is often captured by attention, as an analogy to word-alignment model in phrase-based MT BID17 .

The syntactic objective is captured implicitly in the decoder because it ensures the fluency of the translation.

For grammar induction, the translation objective is provides more guidance than the marginal likelihood typically used in unsupervised learning.

However, we note that the quality of the induced grammar also depends on the choice of the target language ( §6).The boy sitting next to the girls ordered a Miso ramen .Figure 1: Here we show a simple dependency tree.

For the sake of understanding this paper, we draw the reader's eyes to two distinct classes of dependency: semantic roles (verb ordered → subject/object boy, ramen) and syntactic rules (noun girls → determiner the).To motivate our work and the importance of structure in translation, consider the process of translating the sentence "The boy sitting next to the girls ordered a Miso ramen." from English to German.

The dependency tree of the sentence is given in figure 1.

In German, translating the verb "ordered", requires knowledge of its subject "boy" to correctly predict the verb's number "bestellte" instead of "bestellten" if the model wrongly identifies "girls" as the subject.

This is a case where syntactic agreement requires long-distance information transfer.

On the other hand, translating the word "next" can be done in isolation without knowledge of neither its head nor child dependencies.

While its true the decoder can, in principle, utilize previously predicted words (e.g. the translation of the "boy") to reason about subject-verb agreement, in practice LSTMs still struggle with long-distance dependencies.

Moreover, BID2 showed that using attention reduces the capacity of the decoder to learn target side syntax.

Based on the insights from examples like the one above, we have designed a model with the following properties:1.

It can induce syntactic relations in the source sentences; 2.

It can decide when and which syntactic information from the source to use for generating target words.

Previous work seems to imply that syntactic dependencies on the source side can be modeled via a self-attention layer BID29 because self-attention allows direct interactions amongst source words.

However, we will show that this is not always the case (section §6).

We achieve our first requirement (1) by means of a syntactic attention layer ( §3.1) that imposes non-projective dependency structure over the source sentence.

To meet our second requirement (2) we use a simple gating mechanism ( §3.2) that learns when to use the source side syntax.

As noted previously, in addition to demonstrating improvements in translation quality with our proposed models, we are also interested in analyzing the aforementioned non-projective dependency trees learned by the models.

Recent work has begun analyzing task-specific latent trees BID30 .

It has been shown that incorporating hierarchical structures leads to better task performance.

Unlike the previous work that induced latent trees explicitly for semantic tasks, we present the first results on learning latent trees with a joint syntactic-semantic objective.

We do this in the service of machine translation which inherently requires access to both aspects of a sentence.

In summary, in this work we make the following contributions:• We propose a new NMT model that learns the latent structure of the encoder and how to use it during decoding.

Our model is language independent and straightforward to apply with BytePair Encoding (BPE) inputs.

We show that our model obtains a significant improvement 0.6 BLEU (German→English) and 0.8 BLEU (English→German) over a strong baseline.• We perform an in-depth analysis of the learned structures on the source side and investigate where the target decoder decides syntax is required.

The rest of the paper is organized as follow: We describe our NMT baseline in section §2.

Our proposed models are detailed in section §3.

We present the experimental setups and translation results in section §4.

In section §5 we analyze models' behavior by means of visualization which pairs with our analysis of the latent trees induced by our model in section §6.

We conclude our work in the last section.

DISPLAYFORM0 Given a training pair of source and target sentences (x, y) of length n and m respectively, Neural Machine Translation (NMT) is a conditional probabilistic model p(y | x) implemented using neural networks DISPLAYFORM1 where θ is the model's parameters.

We will omit the parameters θ herein for readability.

The NMT system used in this work is a seq2seq model that consists of a bidirectional LSTM encoder and an LSTM decoder coupled with an attention mechanism BID0 BID23 .

Our system is based on a PyTorch implementation 1 of OpenNMT BID16 .

Let DISPLAYFORM2 be the output of the encoder DISPLAYFORM3 Here we use S = [s 1 ; . . . ; s n ] ∈ R d×n as a concatenation of {s i }.

The decoder is composed of stacked LSTMs with input-feeding.

Specifically, the inputs of the decoder at time step t are the previous hidden state h t−1 , a concatenation of the embedding of previous generated word y t−1 and a vector u t−1 : DISPLAYFORM4 where g is a one layer feed-forward network and c t−1 is a context vector computed by an attention mechanism DISPLAYFORM5 DISPLAYFORM6 where W a ∈ R d×d is a trainable parameter.

Finally a single layer feed-forward network f takes u t as input and returns a multinomial distribution over all the target words DISPLAYFORM7 Previous work on incorporating source-side syntax in NMT often focuses on modifying the standard recurrent encoder such that the encoder is explicitly made aware of the syntactic structure of the source sentence.

Given a sentence of length n, syntax encoders of this type return a set of n annotation vectors each compressing semantic and syntactic relations defined by the given parse tree of the input.

The attention module then accesses these annotations during the generation of the target.

We argue that this approach puts a lot of burden on the encoder as it has to balance the influence of semantics and syntax at every step regardless of the target words that are being generated.

Here, we propose a simple alternative approach where we let the encoder output two sets of vectors: content annotations and syntactic annotations ( FIG1 ).

The content annotations are the outputs of a standard BiLSTM while the syntactic annotations are produced by a structured attention layer ( §3.1).

Having two set of annotations, first we compute attention weights α using the decoder's hidden state h, then we compute the context vector c (eq. 4) as in standard NMT system ( FIG1 ).

We then calculate syntactic vector d by taking a weighted average between α and the syntactic annotations ( FIG1 ).

Finally, we allow the decoder to decide how much syntax it needs for making a prediction given the decoder's current state by using a gating mechanism to control syntactic information.

Apart from lifting the burden otherwise placed on the encoder and tightly coupling the syntactic encoding to the (a) Structured Attention Encoder: the first layer is a standard BiLSTM, the top layer is a syntactic attention network.

decoder, the gating mechanism also allows us to inspect the decoder state and answer the question "

When does source side syntax matter?" in section §5.Inspired by structured attention networks , we present a syntactic attention layer that aims to discovery and convey source side dependency information to the decoder.

The syntactic attention model consists of two parts:1.

A syntactic attention layer for head word selection in the encoder;2.

An attention with gating mechanism to control the amount of syntax needed for generating a target word at each time step.

The head word selection layer learns to select a soft head word for each source word via structured attention.

This layer does not have access to any dependency labels from the source.

The head word selection layer transforms S into a matrix M that encodes implicit dependency structure of x using self-structured-attention.

First we apply three trainable weight matrices W q , W k , W v ∈ R d×d to map S to query, key, and value matrices S q , S k , S v ∈ R d×n : DISPLAYFORM0 Then we compute structured attention probabilities β relying on a function sattn that we will describe in detail shortly.

DISPLAYFORM1 The structured attention function sattn is inspired by the work of but differs in two important ways.

First we model non-projective dependency trees.

Second, we ultilize Kirchhoff's Matrix-Tree Theorem BID28 instead of sum-product algorithm presented in for fast evaluation of the attention probabilities.

We note that BID22 first propose using the Matrix-Tree Theorem for evaluating the marginals in end to end training of neural networks.

Their work however focuses on semantic objectives rather than a joint semantic and syntactic objectives such as machine translation.

Additionally, in this work, we will evaluate structured attention component on datasets that are two orders of magnitude larger than the datasets studied in BID22 .Let z ∈ {0, 1} n×n be an adjacency matrix encoding a source's dependency tree.

Let φ ∈ R n×n be a scoring matrix such that cell φ i,j scores how likely word x i is to be the head of word x j .

The matrix φ is obtained simply by DISPLAYFORM2 The probability of a dependency tree z is therefore given by DISPLAYFORM3 where Z(φ) is the partition function.

In the head selection model, we are interested in the marginal p(z i,j = 1 | x; φ) DISPLAYFORM4 We use the framework presented by BID18 to compute the marginal of non-projective dependency structures.

BID18 use the Kirchhoff's Matrix-Tree Theorem BID28 to compute p(z i,j = 1 | x; φ) as follow: DISPLAYFORM5 Now we construct a matrixL that accounts for root selection DISPLAYFORM6 The marginals β are then DISPLAYFORM7 where δ i,j is the Kronecker delta.

For the root node, the marginals are given by DISPLAYFORM8 The computation of the marginals is fully differentiable, thus we can train the model in an end-to-end fashion by maximizing the conditional likelihood of the translation.

We encourage the decoder to use syntactic annotations by means of attention.

Essentially, if the model attends to a particular source word x i when generating the next target word, we also want the model to attend to the head word of x i .

We implement this idea using a new shared attention layer from decoder's state h to encoder's annotations S and M. First, a we compute a standard attention weights α t−1 = softmax(h T t−1 W a S) as in equation 3.

We then compute a weighted syntactic vector: DISPLAYFORM0 Note that the syntactic vector d t−1 and the context vector c t−1 share the same attention weights α t−1 at time step t. By sharing the attention weights α t−1 we hope that if the model picks a source word x i to translate with the highest probability α t−1 [i], the contribution of x i 's head in the syntactic vector d t−1 is also highest.

It is not always useful or necessary to access the syntactic context d t−1 every time step t. Ideally, we should let the model decide whether it needs to use this information.

For example, the model might decide when it needs to resolve long distance dependencies in the source side.

To control the amount of source side syntactic information we introduce a gating mechanism: DISPLAYFORM1 The vector u t−1 from equation 2 now becomes Figure 3: A pictorial illustration of having two separate attention (3b) and shared attention (3a) from the decoder to the encoder.

The blue text represents the content vectors of the sentence and the purple text represents the syntactic vectors.

The number corresponding to each word is the probability mass from decoder-to-encoder attention layer(s).

Note, the reallocation of mass to both the subject and object.

DISPLAYFORM2 An alternative to incorporate syntactic annotation M to the decoder is to use a separate attention layer to compute the syntactic vector d t−1 at time step t: Figure 3 illustrates the difference between shared attention and separate attention when the decoder is translating the english word "ordered".

The source words are in blue and their corresponding head words are in purple.

As can be seen, shared attention now helps the decoder pick the right number for the verb by taking into account the subject "boy".

DISPLAYFORM3 DISPLAYFORM4

Finally, we include an experiment with hard structured attention.

The main motivation of this experiment is twofold.

First, we want to simulate the scenario where the model has access to a decoded parse tree.

Obviously, we do not expect this model to perform best overall in NMT as it only has access to an induced tree rather than a gold one.

Conversely, forcing the model to make hard decisions during training mirrors the intermediary extraction and conditioning on a dependency tree ( §6.1), we therefore hope this technique will improve the performance on grammar induction.

Recall the marginal β i,j gives us the probability that word x i is the head of word x j .

We convert these soft weights to hard onesβ bȳ DISPLAYFORM0 We train this model using the straight-through estimator BID3 .

Note that in this setup, each word has a parent but there is no guarantee that the structure given by hard attention will result in a tree (i.e. it may contain cycle).

A more principle way to enforce tree structure is to decode the best tree T using the maximum spanning tree algorithm BID5 Edmonds, 1967) and to setβ k,j = 1 if the edge (x k → x j ) ∈ T .

Unfortunately, maximum spanning tree decoding can be prohibitively slow as the Chu-Liu-Edmonds algorithm is not GPU friendly.

We therefore resort to greedily picking a parent word for each word x j in the sentence using equation 21.

This is actually a principled simplification as greedily assigning a parent for each word is the first step in Chu-Liu-Edmonds algorithm.

Next we will discuss our experimental setup and report results for English↔German (En↔De) and English↔Russian (Ru↔En) translation models.

We use WMT17 2 data in our experiments.

TAB1 shows the statistics of the data.

For En-De, we use a concatenation of Europarl, Common Crawl, Rapid corpus of EU press releases, and News Commentary v12.

We use newstest2015 for development and newstest2016, newstest2017 for test.

For En-Ru, we use Common Crawl, News Commentary v12, and Yandex Corpus.

The development data comes from newstest2016 and newstest2017 and is reserved for testing.

We use BPE BID26 with 32,000 merge operations.

We run BPE for each language instead of using BPE for the concatenation of both source and target languages.

Our baseline is an NMT model with input-feeding ( §2).

As we will be making several modifications from the basic architecture in our proposed models, we will verify each choice in our architecture design empirically.

First we validate the structured attention module by comparing it to a selfattention module BID20 BID29 .

Since self-attention does not assume any hierarchical structure over the source sentence, we refer it as flat-attention (FA).

Second, we validate the benefit of using two sets of annotations in the encoder.

We combine the hidden states of the encoder h with syntactic context d to obtain a single set of annotation using the following equation DISPLAYFORM0 Here we first down weight the syntactic context d i before adding it to s i .

We refer to this baseline as SA-NMT-1set.

Note that in this baseline, there is only one attention layer from the target to the source.

In all the models, we share the weights of target word embeddings and the output layer as suggested by Inan et al. FORMULA3 ; BID24 .

For all the models, we set the word embedding size to 1024, the number of LSTM layers to 2, and the dropout rate to 0.3.

Parameters are initialized uniformly in (−0.04, 0.04).

We use the Adam optimizer with an initial learning rate 0.001.

We evaluate our models on development data every 10,000 updates for De-En and 5,000 updates for Ru-En.

If the validation perplexity increases, we decay the learning rate by 0.5.

We stop training after decaying the learning rate five times as suggested by BID6 .

The mini-batch size is 32 in all the experiments.

We report the BLEU scores using the multi-bleu.perl script.

TAB2 shows the BLEU scores in our experiments.

We test statistical significance using bootstrap resampling BID25 .

Statisical significance are marked as † p < 0.05 and ‡ p < 0.01 when compared against the baselines.

Additionally, we also report statistical significance p < 0.05 and p < 0.01 when compared against the FA-NMT models that have two separate attention layers from the decoder to the encoder.

Overall, the SA-NMT (shared) model performs the best gaining more than 0.5 BLEU De→En on wmt16, up to 0.82 BLEU on En→De wmt17 and 0.64 BLEU En→Ru direction over a competitive NMT baseline.

The results show that structured attention is useful when translating from English to languages that have long-distance dependencies and complex morphological agreements.

We also see that the gain is marginal compared to selfattention models (shared-FA-NMT-shared) and not significant.

Within FA-NMT models, sharing attention is helpful.

Our results also confirm the advantage of having two separate sets of annotations in the encoder when modeling syntax.

The hard structured attention model (SA-NMT-hard) performs comparable to the baseline.

While this is a somewhat expected result from the hard attention model, we will show in the next section ( §6) that the quality of induced trees from hard attention is far better than the soft ones.

Darker color means higher attention weights.

As can be seen here, while both models agree on some basic elements of the underlying grammar, the attention's mass tends to concentrate on fewer tokens in hard structured attention.

For some tokens, hard-attention, before binarization by eq 21, does not show a strong favor towards any head.

Perhaps this explains the poor performance of SA-NMT-hard in translation because hard attention has to pick one head word among all equally probable heads.

FIG3 shows a sample visualization of structured attention models trained on En→De data.

It is worth noting that the shared SA-NMT model FIG3 ) and the hard SA-NMT model FIG3 ) capture similar structures of the source sentence.

We hypothesize that when the objective function requires syntax, the induced trees are more consistent unlike those discovered by a semantic objective BID30 .

Both models correctly identify that the verb is the head of pronoun (hope→I, said→she).

While intuitively it is clearly beneficial to know the subject of the verb when translating from English into German, the model attention is still somewhat surprising because long distance dependency phenomena are less common in English, so we would expect that a simple content based addressing (i.e. standard attention mechanism) would be sufficient in this translation.

Finally, in addition to attention weight visualization, we provide sample trees induced by our models in Figure We now turn to the question of when does the target LSTM need to access source side syntax.

We investigate this by analyzing the gate activations of our best model, SA-NMT (shared).

At time step t, when the model is about to predict the target word y t , we compute the norm of the gate activations

The activation norm z t allows us to see how much syntactic information flows into the decoder.

We observe that z t has its highest value when the decoder is about to generate a verb while it has its lowest value when the end of sentence token </s> is predicted.

FIG6 shows some examples of German target sentences.

The darker colors represent higher activation norms and bold words indicate the highest activation norms when those words are being predicted.

It is clear that translating verbs requires knowledge of syntax.

We also see that after verbs, the gate activation norms are highest at nouns Zeit (time), Mut (courage), Dach (roof ) and then tail off as we move to function words which require less context to disambiguate.

Below are the frequencies with which the highest activation norm in a sentence is applied to a given part-of-speech tag on newstest2016.

We include the top 10 most common activations.

It is important to note that this distribution is dramatically different than a simple frequency baseline.

NLP has longed assumed hierarchical structured representations were important to understanding language.

In this work, we have borrowed that intuition to inform the construction of our model (as previously discussed).

We feel it is important to take a step beyond a comparison of aggregate model performance and investigate whether the internal latent representations discovered by our models share properties previously identified within linguistics and if not, what important differences exist.

We investigate the interpretability of our model's representations by: 1) A quantitative attachment accuracy and 2) A qualitative comparison of the underlying grammars.

Our results both corroborate and refute previous work BID10 BID30 .

We agree and provide stronger evidence that syntactic information can be discovered via latent structured attention, but we also present preliminary results that indicate that conventional definitions of syntax may be at odds with task specific performance.

For extracting non-projective dependency trees, we use Chu-Liu-Edmonds algorithm BID5 Edmonds, 1967) .

First, we must collapse BPE segments into words.

Assume the k-th word corresponds to BPE tokens from index u to v. We obtain a new matrixφ by summing over φ i,j that are the corresponding BPE segments.

DISPLAYFORM0

We compute unlabeled directed and undirected attachment accuracies of our predicted trees on gold annotations from Universal Dependencies (UD version 2) dataset 3 .

Our five model settings in addition to left and right branching baselines are presented in Table 3 .

The results indicate that the target language effects the source encoder's induction performance and several settings are competitive with branching baselines for determining headedness.

We see performance gains from hard attention and several models outperform baselines for undirected dependency metrics (UA).

Whether hard attention helps is unclear.

Its appears to help for German and not with Russian.

Successfully extracting linguistic structure with hard attention indicates that models can capture interesting structures beyond semantic co-occurrence via discrete actions.

This corroborates previous work BID4 BID32 which has shown that non-trivial structures are learned by using REINFORCE BID31 or Gumbel-softmax trick BID13 to backprop through discrete units.

Our approach also outperforms that of BID10 despite our model lacking access to additional resources like part-of-speech tags.

Dependency Accuracies While SA-NMT-hard model gives the best directed attachment scores on both German and English, the BLEU scores of this model are below other SA-NMT models as shown in TAB2 .

The lack of correlation between syntactic performance and NMT contradicts the intuition of previous work and actually suggests that useful structures learned in service of a task might not necessarily benefit from or correspond to known linguistic formalisms.

Table 3 : Directed and Undirected (DA/UA) accuracies of our models on both English and German data as compared to branching baselines.

Punctuation is removed during the evaluation.

Our results show an intriguing effect of the target language on grammar induction.

We observe a huge boost in DA/UA scores in FA-NMT and SA-NMT-shared models when the target language is morphologically rich (Russian).

In comparison to previous work BID2 BID27 on the encoder's ability to capture source side syntax, we show a stronger result that even when the encoders are designed to capture syntax explicitly, the choice of the target language has a great influence on the amount of syntax learned by the encoder.

Qualitative Grammar Analysis We should obviously note that the model's strength shows up in the directed but not the undirected attention.

This begs the question as to whether there are basic structural elements the grammar has decided not to attend to or if all constructions are just generally weak.

We qualitatively analyzed the learned grammars as a function of dependency productions between universal part-of-speech tags in TAB5 .

Here, we extract the grammar of the language as if it were a CFG and compare the gold production frequencies and compare them to our models' predictions.

In other words, how often does tag i generate tag j in the treebank and how closely did our models uncover those statistics.

This finer grained analysis gives us insight into the model's surprising ability to often verb based syntax when translating, but simultaneously favoring noun based constructions.

This is particularly noticeable in the SA model for German.

We have proposed a structured attention encoder for NMT.

Our models show significant gains in performance over a strong baseline on standard WMT benchmarks.

The models presented here do not access any external information such as parse-trees or part-of-speech tags.

We show that our models induce dependency trees over the source sentences that systematically outperform baseline branching and previous work.

We find that the quality of induced trees (compared against gold standard annotations) is not correlated with the translation quality.

<|TLDR|>

@highlight

improve NMT with latent trees

@highlight

This paper describes a method to induce source-side dependency structures in service to neural machine translation.