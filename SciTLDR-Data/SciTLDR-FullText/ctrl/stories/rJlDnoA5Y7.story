The Softmax function is used in the final layer of nearly all existing sequence-to-sequence models for language generation.

However, it is usually the slowest layer to compute which limits the vocabulary size to a subset of most frequent types; and it has a large memory footprint.

We propose a general technique for replacing the softmax layer with a continuous embedding layer.

Our primary innovations are a novel probabilistic loss, and a training and inference procedure in which we generate a probability distribution over pre-trained word embeddings, instead of a multinomial distribution over the vocabulary obtained via softmax.

We evaluate this new class of sequence-to-sequence models with continuous outputs on the task of neural machine translation.

We show that our models obtain upto 2.5x speed-up in training time while performing on par with the state-of-the-art models in terms of translation quality.

These models are capable of handling very large vocabularies without compromising on translation quality.

They also produce more meaningful errors than in the softmax-based models, as these errors typically lie in a subspace of the vector space of the reference translations.

Due to the power law distribution of word frequencies, rare words are extremely common in any language BID45 ).

Yet, the majority of language generation tasks-including machine translation BID39 BID1 BID24 , summarization BID36 BID37 BID30 , dialogue generation BID40 , question answering BID44 , speech recognition BID13 Xiong et al., 2017) , and others-generate words by sampling from a multinomial distribution over a closed output vocabulary.

This is done by computing scores for each candidate word and normalizing them to probabilities using a softmax layer.

Since softmax is computationally expensive, current systems limit their output vocabulary to a few tens of thousands of most frequent words, sacrificing linguistic diversity by replacing the long tail of rare words by the unknown word token, unk .

Unsurprisingly, at test time this leads to an inferior performance when generating rare or out-of-vocabulary words.

Despite the fixed output vocabulary, softmax is computationally the slowest layer.

Moreover, its computation follows a large matrix multiplication to compute scores over the candidate words; this makes softmax expensive in terms of memory requirements and the number of parameters to learn BID26 BID27 BID8 .

Several alternatives have been proposed for alleviating these problems, including sampling-based approximations of the softmax function BID2 BID26 , approaches proposing a hierarchical structure of the softmax layer BID27 BID7 , and changing the vocabulary to frequent subword units, thereby reducing the vocabulary size BID38 .We propose a novel technique to generate low-dimensional continuous word representations, or word embeddings BID25 BID31 BID4 instead of a probability distribution over the vocabulary at each output step.

We train sequence-to-sequence models with continuous outputs by minimizing the distance between the output vector and the pretrained word embedding of the reference word.

At test time, the model generates a vector and then searches for its nearest neighbor in the target embedding space to generate the corresponding word.

This general architecture can in principle be used for any language generation (or any recurrent regression) task.

In this work, we experiment with neural machine translation, implemented using recurrent sequence-to-sequence models BID39 with attention BID1 BID24 .To the best of our knowledge, this is the first work that uses word embeddings-rather than the softmax layer-as outputs in language generation tasks.

While this idea is simple and intuitive, in practice, it does not yield competitive performance with standard regression losses like 2 .

This is because 2 loss implicitly assumes a Gaussian distribution of the output space which is likely false for embeddings.

In order to correctly predict the outputs corresponding to new inputs, we must model the correct probability distribution of the target vector conditioned on the input BID3 .

A major contribution of this work is a new loss function based on defining such a probability distribution over the word embedding space and minimizing its negative log likelihood ( §3).We evaluate our proposed model with the new loss function on the task of machine translation, including on datasets with huge vocabulary sizes, in two language pairs, and in two data domains ( §4).

In §5 we show that our models can be trained up to 2.5x faster than softmax-based models while performing on par with state-of-the-art systems in terms of generation quality.

Error analysis ( §6) reveals that the models with continuous outputs are better at correctly generating rare words and make errors that are close to the reference texts in the embedding space and are often semantically-related to the reference translation.

Traditionally, all sequence to sequence language generation models use one-hot representations for each word in the output vocabulary V. More formally, each word w is represented as a unique vector o(w) ∈ {0, 1} V , where V is the size of the output vocabulary and only one entry id(w) (corresponding the word ID of w in the vocabulary) in o(w) is 1 and the rest are set to 0.

The models produce a distribution p t over the output vocabulary at every step t using the softmax function: DISPLAYFORM0 where, s w = W hw h t + b w is the score of the word w given the hidden state h produced by the LSTM cell BID15 at time step t. W ∈ R V xH and b ∈ R v are trainable parameters.

H is the size of the hidden layer h. These parameters are trained by minimizing the negative log-likelihood (aka cross-entropy) of this distribution by treating o(w) as the target distribution.

The loss function is defined as follows: DISPLAYFORM1 This loss computation involves a normalization proportional to the size of the output vocabulary V .

This becomes a bottleneck in natural language generation tasks where the vocabulary size is typically tens of thousands of words.

We propose to address this bottleneck by representing words as continuous word vectors instead of one-hot representations and introducing a novel probabilistic loss to train these models as described in §3.2 Here, we briefly summarize prior work that aimed at alleviating the sofmax bottleneck problem.

We briefly summarize existing modifications to the sofmax layer, capitalizing on conceptually different approaches.

Sampling-Based Approximations Sampling-based approaches completely do away with computing the normalization term of softmax by considering only a small subset of possible outputs.

These include approximations like Importance Sampling BID2 , Noise Constrastive Estimation BID26 , Negative Sampling BID25 , and Blackout BID16 .

These alternatives significantly speed-up training time but degrade generation quality.

BID27 replace the flat softmax layer with a hierarchical layer in the form of a binary tree where words are at the leaves.

This alleviates the problem of expensive normalization, but these gains are only obtained at training time.

At test time, the hierarchical approximations lead to a drop in performance compared to softmax both in time efficiency and in accuracy.

BID7 propose to divide the vocabulary into clusters based on their frequencies.

Each word is produced by a different part of the hidden layer making the output embedding matrix much sparser.

This leads to performance improvement both in training and decoding.

However, it assigns fewer parameters to rare words which leads to inferior performance in predicting them BID34 .

BID0 ; BID11 add additional terms to the training loss which makes the normalization factor close to 1, obviating the need to explicitly normalize.

The evaluation of certain words can be done much faster than in softmax based models which is extremely useful for tasks like language modeling.

However, for generation tasks, it is necessary to ensure that the normalization factor is exactly 1 which might not always be the case, and thus it might require explicit normalization.

BID18 introduce character-based methods to reduce vocabulary size.

While character-based models lead to significant decrease in vocabulary size, they often differentiate poorly between similarly spelled words with different meanings.

BID38 find a middle ground between characters and words based on sub-word units obtained using Byte Pair Encoding (BPE).

Despite its limitations BID28 , BPE achieves good performance while also making the model truly open vocabulary.

BPE is the state-of-the art approach currently used in machine translation.

We thus use this as a baseline in our experiments.

In our proposed model, each word type in the output vocabulary is represented by a continuous vector e(w) ∈ R m where m V .

This representation can be obtained by training a word embedding model on a large monolingual corpus BID25 BID31 BID4 .At each generation step, the decoder of our model produces a continuous vectorê ∈ R m .

The output word is then predicted by searching for the nearest neighbor ofê in the embedding space: DISPLAYFORM0 where V is the output vocabulary, d is a distance function.

In other words, the embedding space could be considered to be quantized into V components and the generated continuous vector is mapped to a word based on the quanta in which it lies.

The mapped word is then passed to the next step of the decoder BID14 .

While training this model, we know the target vector e(w), and minimize its distance from the output vectorê.

With this formulation, our model is directly trained to optimize towards the information encoded by the embeddings.

For example, if the embeddings are primarily semantic, as in BID25 or BID4 , the model would tend to output words in a semantic space, that is produced words would either be correct or close synonyms (which we see in our analysis in §6), or if we use synactico-semantic embeddings BID22 BID23 , we might be able to also control for syntatic forms.

We propose a novel probabilistic loss function-a probabilistic variant of cosine loss-which gives a theoretically grounded regression loss for sequence generation and addresses the limitations of existing empirical losses (described in §4.2).

Cosine loss measures the closeness between vector directions.

A natural choice for estimating directional distributions is von Mises-Fisher (vMF) defined over a hypersphere of unit norm.

That is, a vector close to the mean direction will have high probability.

VMF is considered the directional equivalent of Gaussian distribution 3 .

Given a target word w, its density function is given as follows: DISPLAYFORM1 where µ and e(w) are vectors of dimension m with unit norm, κ is a positive scalar, also called the concentration parameter.

κ = 0 defines a uniform distribution over the hypersphere and κ = ∞ defines a point distribution at µ. C m (κ) is the normalization term: DISPLAYFORM2 where I v is called modified Bessel function of the first kind of order v. The output of the model at each step is a vectorê of dimension m. We use κ = ê .

Thus the density function becomes: DISPLAYFORM3 (2) It is noteworthy that equation 2 is very similar to softmax computation (except that e(w) is a unit vector), the main difference being that normalization is not done by summing over the vocabulary, which makes it much faster than the softmax computation.

More details about it's computation are given in the appendix.

The negative log-likelihood of the vMF distribution, which at each output step is given by: DISPLAYFORM4 Regularization of NLLvMF In practice, we observe that the NLLvMF loss puts too much weight on increasing ê , making the second term in the loss function decrease rapidly without significant decrease in the cosine distance.

To account for this, we add a regularization term.

We experiment with two variants of regularization.

NLLvMF reg1 : We add λ 1 ê to the loss function, where λ 1 is a scalar hyperparameter.

4 This makes intuitive sense in that the length of the output vector should not increase too much.

The regularized loss function is as follows: DISPLAYFORM5 We modify the previous loss function as follows: DISPLAYFORM6 − log C m ( ê ) decreases slowly as ê increases as compared the second term.

Adding a λ 2 < 1 the second term controls for how fast it can decrease.

We modify the standard seq2seq models in OpenNMT 6 in PyTorch 7 BID19 ) to implement the architecture described in §3.

This model has a bidirectional LSTM encoder with an attentionbased decoder BID24 .

The encoder has one layer whereas the decoder has 2 layers of 3 A natural choice for many regression tasks would be to use a loss function based on Gaussian distribution itself which is a probabilistic version of 2 loss.

But as we describe in §4.2, 2 is not considered a suitable loss for regression on embedding spaces 4 We empirically set λ1 = 0.02 in all our experiments 5 We use λ2 = 0.1 in all our experiments 6 http://opennmt.net/ 7 https://pytorch.org/ size 1024 with the input word embedding size of 512.

For the baseline systems, the output at each decoder step multiplies a weight matrix (H × V ) followed by softmax.

This model is trained until convergence on the validation perplexity.

For our proposed models, we replace the softmax layer with the continuous output layer (H × m) where the outputs are m dimensional.

We empirically choose m = 300 for all our experiments.

Additional hyperparameter settings can be found in the appendix.

These models are trained until convergence on the validation loss.

Out of vocabulary words are mapped to an unk token 8 .

We assign unk an embedding equal to the average of embeddings of all the words which are not present in the target vocabulary of the training set but are present in vocabulary on which the word embeddings are trained.

Following BID10 , after decoding a post-processing step replaces the unk token using a dictionary look-up of the word with highest attention score.

If the word does not exist in the dictionary, we back off to copying the source word itself.

Bilingual dictionaries are automatically extracted from our parallel training corpus using word alignment BID12 9 .

We evaluate all the models on the test data using the BLEU score BID29 .We evaluate our systems on standard machine translation datasets from IWSLT'16 BID6 , on two target languages, English: German→English, French→English and a morphologically richer language French: English→French.

The training sets for each of the language pairs contain around 220,000 parallel sentences.

We use TED Test 2013+2014 (2,300 sentence pairs) as developments sets and TED Test 2015+2016 (2,200 sentence pairs) as test sets respectively for all the language pairs.

All mentioned setups have a total vocabulary size of around 55,000 in the target language of which we choose top 50,000 words by frequency as the target vocabulary 10 .We also experiment with a much larger WMT'16 German→English BID5 task whose training set contains around 4.5M sentence pairs with the target vocabulary size of around 800,000.

We use newstest2015 and newstest2016 as development and test data respectively.

Since with continuous outputs we do not need to perform a time consuming softmax computation, we can train the proposed model with very large target vocabulary without any change in training time per batch.

We perform this experiment with WMT'16 de-en dataset with a target vocabulary size of 300,000 (basically all the words in the target vocabulary for which we had trained embeddings).

But to able to produce these words, the source vocabulary also needs to be increased to have their translations in the inputs, which would lead to a huge increase in the number of trainable parameters.

Instead, we use sub-words computed using BPE as source vocabulary.

We use 100,000 merge operations to compute the source vocabulary as we observe using a smaller number leads to too small (and less meaningful) sub-word units which are difficult to align with target words.

Both of these datasets contain examples from vastly different domains, while IWSLT'16 contains less formal spoken language, WMT'16 contains data primarily from news.

We train target word embeddings for English and French on corpora constructed using WMT'16 BID5 monolingual datasets containing data from Europarl, News Commentary, News Crawl from 2007 to 2015 and News Discussion (everything except Common Crawl due to its large memory requirements).

These corpora consist of 4B+ tokens for English and 2B+ tokens for French.

We experiment with two embedding models: word2vec BID25 and fasttext Bojanowski et al. FORMULA0 which were trained using the hyper-parameters recommended by the authors.

We compare our proposed loss function with standard loss functions used in multivariate regression.

Squared Error ( 2 ) is the most common distance function used when the model outputs are continuous BID21 .

For each target word w, it is given as L 2 = ê − e(w) 2 2 penalizes large errors more strongly and therefore is sensitive to outliers.

To avoid this we use a square rooted version of 2 loss.

But it has been argued that there is a mismatch between the objective function used to learn word representations (maximum likelihood based on inner product), the distance measure for word vectors (cosine similarity), and 2 distance as the objective function 8 Although the proposed model can make decoding open vocabulary, there could still be unknown words, e.g., words for which we do not have pre-trained embeddings; we need unk token to represent these words 9 https://github.com/clab/fast_align 10 Removing the bottom 5,000 words did not make a significant difference in terms of translation quality to learn transformations of word vectors BID41 .

This argument prompts us to look at cosine loss.

ê .

e(w) .

This loss minimizes the distance between the directions of output and target vectors while disregarding their magnitudes.

The target embedding space in this case becomes a set of points on a hypersphere of dimension m with unit radius.

BID20 argue that using pairwise losses like 2 or cosine distance for learning vectors in high dimensional spaces leads to hubness: word vectors of a subset of words appear as nearest neighbors of many points in the output vector space.

To alleviate this, we experiment with a margin-based ranking loss (which has been shown to reduce hubness) to train the model to rank the word vector predictionê for target vector e(w) higher than any other word vector e(w ) in the embedding space.

L mm = w ∈V,w =w max{0, γ + cos(ê, e(w )) − cos(ê, e(w))} where, γ is a hyperparameter 11 representing the margin and w denotes negative examples.

We use only one informative negative example as described in BID20 which is closest tô e and farthest from the target word vector e(w).

But, searching for this negative example requires iterating over the vocabulary which brings back the problem of slow loss computation.

In the case of empirical losses, we output the word whose target embedding is the nearest neighbor to the vector in terms of the distance (loss) defined.

In the case of NLLvMF, we predict the word whose target embedding has the highest value of vMF probability density wrt to the output vector.

This predicted word is fed as the input for the next time step.

Our nearest-neighbor decoding scheme is equivalent to a greedy decoding; we thus compare to baseline models with beam size of 1.

Until now we discussed the embeddings in the output layer.

Additionally, decoder in a sequenceto-sequence model has an input embedding matrix as the previous output word is fed as an input to the decoder.

Much of the size of the trainable parameters in all the models is occupied by these input embedding weights.

We experiment with keeping this embedding layer fixed and tied with pre-trained target output embeddings BID33 .

This leads to significant reduction in the number of parameters in our model.

TAB8 shows the BLEU scores on the test sets for several baseline systems, and various configurations including the types of losses, types of inputs/outputs used (word, BPE, or embedding) 12 and whether the model used tied embeddings in the decoder or not.

Since we represent each target word by its embedding, the quality of embeddings should have an impact on the translation quality.

We measure this by training our best model with fasttext embeddings BID4 , which leads to > 1 BLEU improvement.

Tied embeddings are the most effective setups: they not only achieve highest translation quality, but also dramatically reduce parameters requirements and the speed of convergence.

11 We use γ = 0.5 in our experiments.

12 Note that we do not experiment with subword embeddings since the number of merge operations for BPE usually depend on the choice of a language pair which would require the embeddings to be retrained for every language pair.

TAB3 shows the average training time per batch.

In FIG1 (left), we show how many samples per second our proposed model can process at training time compared to the baseline.

As we increase the batch size, the gap between the baseline and the proposed models increases.

Our proposed models can process large mini-batches while still training much faster than the baseline models.

The largest mini-batch size with which we can train our model is 512, compared to 184 in the baseline model.

Using max-margin loss leads to a slight increase in the training time compared to NLLvMF.

This is because its computation needs a negative example which requires iterating over the entire vocabulary.

Since our model requires look-up of nearest neighbors in the target embedding table while testing, it currently takes similar time as that of softmax-based models.

In future work, approximate nearest neighbors algorithms BID17 can be used to improve translation time.

We also compare the speed of convergence, using BLEU scores on dev data.

In FIG1 , we plot the BLEU scores against the number of epochs.

Our model convergences much faster than the baseline models leading to an even larger improvement in overall training time (Similar figures for more datasets can be found in the appendix).

As a result, as shown in table 3, the total training time of our proposed model (until convergence) is less than up-to 2.5x of the total training time of the baseline models.

Memory Requirements As shown in TAB3 our best performing model requires less than 1% of the number of parameters in input and output layers, compared to BPE-based baselines.

Softmax BPE Emb w/ NLL-vMF fr-en 4h 4.5h 1.9h de-en 3h 3.5h 1.5h en-fr 1.8h 2.8h 1.3 WMT de-en 4.3d 4.5d 1.6d Table 3 : Total convergence times in hours(h)/days(d).

Translation of Rare Words We evaluate the translation accuracy of words in the test set based on their frequency in the training corpus.

Table 5 shows how the F 1 score varies with the word frequency.

F 1 score gives a balance between recall (the fraction of words in the reference that the predicted sentence produces right) and precision (the fraction of produced words that are in reference).

We show substantial improvements over softmax and BPE baselines in translating less frequent and rare words, which we hypothesize is due to having learned good embeddings of such words from the monolingual target corpus where these words are not as rare.

Moreover, in BPE based models, rare words on the source side are split in smaller units which are in some cases not properly translated in subword units on the target side if transparent alignments don't exist.

For example, the word saboter in French is translated to sab+ot+tate by the BPE model whereas correctly translated as sabotage by our model.

Also, a rare word retraite in French in translated to pension by both Softmax and BPE models (pension is a related word but less rare in the corpus) instead of the expected translation retirement which our model gets right.

We conducted a thorough analysis of outputs across our experimental setups.

Few examples are shown in the appendix.

Interestingly, there are many examples where our models do not exactly match the reference translations (so they do not benefit from in terms of BLEU scores) but produce meaningful translations.

This is likely because the model produces nearby words of the target words or paraphrases instead of the target word (which are many times synonyms).Since we are predicting embeddings instead of actual words, the model tends to be weaker sometimes and does not follow a good language model and leads to ungrammatical outputs in cases where the baseline model would perform well.

Integrating a pre-trained language model within the decoding framework is one potential avenue for our future work.

Another reason for this type of errors could be our choice of target embeddings which are not modeled to (explicitly) capture syntactic relationships.

Using syntactically inspired embeddings BID22 BID23 might help reduce these errors.

However, such fluency errors are not uncommon also in softmax and BPE-based models either.

Table 5 : Test set unigram F 1 scores of occurrence in the predicted sentences based on their frequencies in the training corpus for different models for fr-en.

This work makes several contributions.

We introduce a novel framework of sequence to sequence learning for language generation using word embeddings as outputs.

We propose new probabilistic loss functions based on vMF distribution for learning in this framework.

We then show that the proposed model trained on the task of machine translation leads to reduction in trainable parameters, to faster convergence, and a dramatic speed-up, up to 2.5x in training time over standard benchmarks.

TAB5 visualizes a comparison between different types of softmax approximations and our proposed method.

State-of-the-art results in softmax-based models are highly optimized after a few years on research in neural machine translation.

The results that we report are comparable or slightly lower than the strongest baselines, but these setups are only an initial investigation of translation with the continuous output layer.

There are numerous possible directions to explore and improve the proposed setups.

What are additional loss functions?

How to setup beam search?

Should we use scheduled sampling?

What types of embeddings to use?

How to translate with the embedding output into morphologically-rich languages?

Can low-resource neural machine translation benefit from translation with continuous outputs if large monolingual corpora are available to pre-train strong target-side embeddings?

We will explore these questions in future work.

Furthermore, the proposed architecture and the probabilistic loss (NLLvMF) have the potential to benefit other applications which have sequences as outputs, e.g. speech recognition.

NLLvMF could be used as an objective function for problems which currently use cosine or 2 distance, such as learning multilingual word embeddings.

Since the outputs of our models are continuous (rather than class-based discrete symbols), these models can potentially simplify training of generative adversarial networks for language generation.

DISPLAYFORM0 where C m (κ) is given as: DISPLAYFORM1 The normalization constant is not directly differentiable because Bessel function cannot be written in a closed form.

The gradient of the first component (log (C m ê )) of the loss is given as DISPLAYFORM2 .

In Table 9 : Translation quality experiments using beam search with BPE based baseline models with a beam size of 5With our proposed models, in principle, it is possible to generate candidates for beam search by using K-Nearest Neighbors.

But how to rank the partially generated sequences is not trivial (one could use the loss values themselves to rank, but initial experiments with this setting did not result in significant gains).

In this work, we focus on enabling training with continuous outputs efficiently and accurately giving us huge gains in training time.

The question of decoding with beam search requires substantial investigation and we leave it for future work.

Une ducation est critique, mais rgler ce problme va ncessiter que chacun d'entre nous s'engage et soit un meilleur exemple pour les femmes et filles dans nos vies.

An education is critical, but tackling this problem is going to require each and everyone of us to step up and be better role models for the women and girls in our own lives.

Education is critical, but it's going to require that each of us will come in and if you do a better example for women and girls in our lives.

Education is critical , but to to do this is going to require that each of us of to engage and or a better example of the women and girls in our lives.

That's critical , but that's that it's going to require that each of us is going to take that the problem and they're going to if you're a better example for women and girls in our lives.

Predicted (MaxMargin) Education is critical, but that problem is going to require that every one of us is engaging and is a better example for women and girls in our lives.

Predicted (NLLvMF reg ) Education is critical , but fixed this problem is going to require that all of us engage and be a better example for women and girls in our lives.

TAB8 : Translation examples.

Red and blue colors highlight translation errors; red are bad and blue are outputs that are good translations, but are considered as errors by the BLEU metric.

Our systems tend to generate a lot of such "meaningful" errors.

Pourquoi ne sommes nous pas de simples robots qui traitent toutes ces donnes, produisent ces rsultats, sans faire l'exprience de ce film intrieur ?

Reference Why aren't we just robots who process all this input, produce all that output, without experiencing the inner movie at all?

Predicted (BPE) Why don't we have simple robots that are processing all of this data, produce these results, without doing the experience of that inner movie?

Why are we not that we do that that are technologized and that that that's all these results, that they're actually doing these results, without do the experience of this film inside?

Predicted (Cosine) Why are we not simple robots that all that data and produce these data without the experience of this film inside?

Predicted (MaxMargin) Why aren't we just simple robots that have all this data, make these results, without making the experience of this inside movie?

Predicted (NLLvMF reg ) Why are we not simple robots that treat all this data, produce these results, without having the experience of this inside film?

TAB8 : Example of fluency errors in the baseline model.

Red and blue colors highlight translation errors; red are bad and blue are outputs that are good translations, but are considered as errors by the BLEU metric.

<|TLDR|>

@highlight

Language generation using seq2seq models which produce word embeddings instead of a softmax based distribution over the vocabulary at each step enabling much faster training while maintaining generation quality