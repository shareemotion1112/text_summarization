Concerns about interpretability, computational resources, and principled inductive priors have motivated efforts to engineer sparse  neural  models for NLP tasks.

If sparsity is important for NLP, might well-trained neural models naturally become roughly sparse?

Using the Taxi-Euclidean norm to measure sparsity, we find that frequent input words are associated with concentrated or sparse activations, while frequent target words are associated with dispersed activations but concentrated gradients.

We find that  gradients associated with function words are more concentrated than the gradients of content words, even controlling for word frequency.

Researchers in NLP have long relied on engineering features to reflect the sparse structures underlying language.

Modern deep learning methods promised to relegate this practice to history, but have not eliminated the interest in sparse modeling for NLP.

Along with concerns about computational resources BID0 BID12 and interpretability BID10 BID21 , human intuitions continue to motivate sparse representations of language.

For example, some work applies assumptions of sparsity to model latent hard categories such as syntactic dependencies BID14 or phonemes BID1 .

BID13 found that a sparse attention mechanism outperformed dense methods on some NLP tasks; BID11 found sparsified versions of LMs that outperform dense originals.

Attempts to engineer sparsity rest on an unstated assumption that it doesn't arise naturally when neural models are learned.

Is this true?Using a simple measure of sparsity, we analyze how it arises in different layers of a neural language model in relation to word frequency.

We show that the sparsity of a word representation increases with exposure to that word during training.

We also find evidence of syntactic learning: gradient updates in backpropagation depend on whether a word's part of speech is open or closed class, even controlling for word frequency.

Language model.

Our LM is trained on a corpus of tokenized, lowercased English Wikipedia (70/10/20 train/dev/test split).

To reduce the number of unique words (mostly names) in the corpus, we excluded any sentence with a word which appears fewer than 100 times.

Those words which still appear fewer than 100 times after this filter are replaced with <UNK>. The resulting training set is over 227 million tokens of around 19.5K types.

We use a standard 2-layer LSTM LM trained with cross entropy loss for 50 epochs.

The pipeline from input x t−1 at time step t − 1 to predicted output distributionx for time t is described in Figure 1 , illustrating intermediate activations h e t , h 1 t , and h 2 t .

At training time, the network observes x t and backpropagates the gradient updates h e t ,h 1 t ,h 2 t , andx t .

The embeddings produced by the encoding layer are 200 units, and the recurrent layers have 200 hidden units each.

The batch size is set to forty, the maximum sequence length to 35, and the dropout ratio to 0.2.

The optimizer is standard SGD with clipped gradients at 2 = 0.25, where the learning rate begins at 20 and is quartered whenever loss fails to improve.

Measuring sparsity.

We measure the sparsity of a vector v using the reciprocal of the TaxicabEuclidean norm ratio BID17 .

This measurement has a long history as a measurement of sparsity in natural settings (Zibulevsky Figure 2: Average sparsity χ(h 2 t ) over all training epochs (x-axis), for target words x t occurring more than 100k times in training.

Target words are sorted from most frequent (bottom) to least frequent (top).and BID23 BID4 BID15 BID22 and is formally defined as χ(v) = v 2 / v 1 .

The relationship between sparsity and this ratio is illustrated in two dimensions in the image on the right, in which darker blue regions are more concentrated.

The pink circle shows the area where 2 ≤ 1 while the yellow diamond depicts 1 ≤ 1.

For sparse vectors 1, 0 or 0, 1 , the norms are identical so χ is 1, its maximum.

For a uniform vector like 1, 1 , χ is at its smallest.

In general, χ(v) is higher when most elements of v are close to 0; and lower when the elements are all similar in value.

Sparsity is closely related to the behavior of a model: If only a few units hold most of the mass of a representation, the activation vector will be highly concentrated.

If a neural network relies heavily on a small number of units in determining its predictions, the gradient will be highly concentrated.

A highly concentrated gradient is mainly modifying a few specific pathways.

For example, it might modify a neuron associated with particular inputs like parentheses BID5 , or properties like sentiment BID16 .Representations of Target Words.

Our first experiments look at the relationship of sparsity to target word x t .

Gradient updates triggered by the target are often used to identify units that are relevant to a prediction , and as shown in Figure 2 , gradient sparsity increases with both the frequency of a word in the corpus and the overall training time.

In other words, more exposure leads to sparser relevance.

Because the sparsity ofh 2 increases with target word frequency, we measure not sparsity itself but the Pearson correlation, over all words w, between word frequency and mean χ(h) over representations h where w is the target: DISPLAYFORM0 Here FIG0 we confirm that concentrated gradients are not a result of concentrated activations, as activation sparsity χ(h 2 ) is not correlated with target word frequency.

The correlation is strong and increasing only for ρ ← (h 2 ).

The sparse structure being applied is therefore particular to the gradient passed from the softmax to the top LSTM layer, related to how a word interacts with its context.

The Role of Part of Speech.

FIG1 shows that ρ ← (h 2 ) follows distinctly different trends for open POS classes 1 and closed classes 2 .

To associate words to POS, we tagged our training corpus with spacy 3 ; we associate a word to a POS only if the majority (at least 100) of its occurrences are tagged with that POS.

We see that initially, frequent words from closed classes are highly concentrated, but soon stabilize, while frequent words from open classes continue to become more concentrated throughout training.

Why?Closed class words clearly signal POS.

But open classes contain many ambiguous words, like "report", which can be a noun or verb.

Open classes also contain many more words in general.

We posit that early in training, closed classes reliably signal syntactic structure, and are essential for shaping network structure.

But open classes are essential for predicting specific words, so their importance in training continues to increase after part of speech tags are effectively learned.

The high sparsity of function word gradient may be surprising when compared with findings that content words have a greater influence on outputs BID7 .

However, those findings were based on the impact on the vector representation of an entire sentence after omitting the word.

BID6 found that content words have a longer window during which they are relevant, which may explain the results of BID7 .

Neither of these studies controlled for word frequency in their analyses contrasting content and function words, but we believe this oversight is alleviated in our work by measuring correlations rather than raw magnitude.

Because ρ ← (h 2 ) is higher when evaluated over more fre- Representations of Input Words.

We next looked at the vector representations of each step in the word sequence as a representation of the input word x t−1 that produced that step.

We measure the correlation with input word frequency: DISPLAYFORM1 Here FIG0 we find that the view across training sheds some light on the learning process.

While the lower recurrent layer quickly learns sparse representations of common input words, ρ → (h 1 ) increases more slowly later in training and is eventually surpassed by ρ → (h e ), while gradient sparsity never becomes significantly correlated with word frequency.

studied the activations of feedforward networks in terms of the importance of individual units by erasing a particular dimension and measuring the difference in log likelihood of the target class.

They found that importance is concentrated into a small number of units at the lowest layers in a neural network, and is more dispersed at higher layers.

Our findings suggest that this effect may be a natural result of the sparsity of the activations at lower layers.

We relate the trajectory over training to the Information Bottleneck Hypothesis of BID20 .

This theory, connected to language model training by BID19 , proposes that the earlier stages of training are dedicated to learning to effectively represent inputs, while later in training these representations are compressed and the optimizer removes input information extraneous to the task of predicting outputs.

If extraneous information is encoded in specific units, this compression would lead to the observed effect, in which the first time the optimizer rescales the step size, it begins an upward trend in ρ → as extraneous units are mitigated.

Why do common target words have such concentrated gradients with respect to the final LSTM layer?

A tempting explanation is that the amount of information we have about common words offers high confidence and stabilizes most of the weights, leading to generally smaller gradients.

If this were true, the denominator of sparsity, gradient 1 , should be strongly anti-correlated with word frequency.

In fact, it is only ever slightly anti-correlated (correlation > −.1).

Furthermore, the sparsity of the softmax gradient χ(x) does not exhibit the strong correlation seen in χ(h 2 ), so sparsity at the LSTM gradient is not a direct effect of sparse logits.

However, the model could still be "high confidence" in terms of how it assigns blame for error during common events, even if it is barely more confident overall in its predictions.

According to this hypothesis, a few specialized neurons might be responsible for the handling of such words.

Perhaps common words play a prototyping role that defines clusters of other words, and therefore have a larger impact on these clusters by acting as attractors within the representation space early on.

Such a process would be similar to how humans acquire language by learning to use words like 'dog' before similar but less prototypical words like 'canine' BID18 .

As a possible mechanism for prototyping with individual units, BID2 found that some neurons in a translation system specialized in particular word forms, such as verb inflection or comparative and superlative adjectives.

For example, a common comparative adjective like 'better' might be used as a reliable signal to shape the handling of comparatives by triggering specialized units, while rarer words have representations that are more distributed according to a small collection of specific contexts.

There may also be some other reason that common words interact more with specific substructures within the network.

For example, it could be related to the use of context.

Because rare words use more context than common words and content words use more context than function words BID6 , the gradient associated with a common word would be focused on interactions with the most recent words.

This would lead common word gradients to be more concentrated.

It is possible that frequent words have sparse activations because frequency is learned as a feature and thus is counted by a few dimensions of proportional magnitude, as posited by .

Understanding where natural sparsity emerges in dense networks could be a useful guide in deciding which layers we can apply sparsity constraints to without affecting model performance, for the purpose of interpretability or efficiency.

It might also explain why certain techniques are effective: for example, in some applications, summing representations together works quite well BID3 .

We hypothesize that this occurs when the summed representations are sparse so there is often little overlap.

Understanding sparsity could help identify cases where such simple ensembling approaches are likely to be effective.

Future work may develop ways of manipulating the training regime, as in curriculum learning, to accelerate the concentration of common words or incorporating concentration into the training objective as a regularizer.

We would also like to see how sparsity emerges in models designed for specific end tasks, and to see whether concentration is a useful measure for the information compression predicted by the Information Bottleneck.

<|TLDR|>

@highlight

We study the natural emergence of sparsity in the activations and gradients for some layers of a dense LSTM language model, over the course of training.