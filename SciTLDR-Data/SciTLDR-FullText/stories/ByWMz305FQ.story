Multilingual Neural Machine Translation (NMT) systems are capable of translating between multiple source and target languages within a single system.

An important indicator of generalization within these systems is the quality of zero-shot translation - translating between language pairs that the system has never seen during training.

However, until now, the zero-shot performance of multilingual models has lagged far behind the quality that can be achieved by using a two step translation process that pivots through an intermediate language (usually English).

In this work, we diagnose why multilingual models under-perform in zero shot settings.

We propose explicit language invariance losses that guide an NMT encoder towards learning language agnostic representations.

Our proposed strategies significantly improve zero-shot translation performance on WMT English-French-German and on the IWSLT 2017 shared task, and for the first time, match the performance of pivoting approaches while maintaining performance on supervised directions.

In recent years, the emergence of sequence to sequence models has revolutionized machine translation.

Neural models have reduced the need for pipelined components, in addition to significantly improving translation quality compared to their phrase based counterparts BID35 .

These models naturally decompose into an encoder and a decoder with a presumed separation of roles: The encoder encodes text in the source language into an intermediate latent representation, and the decoder generates the target language text conditioned on the encoder representation.

This framework allows us to easily extend translation to a multilingual setting, wherein a single system is able to translate between multiple languages BID11 BID28 .Multilingual NMT models have often been shown to improve translation quality over bilingual models, especially when evaluated on low resource language pairs BID14 BID20 .

Most strategies for training multilingual NMT models rely on some form of parameter sharing, and often differ only in terms of the architecture and the specific weights that are tied.

They allow specialization in either the encoder or the decoder, but tend to share parameters at their interface.

An underlying assumption of these parameter sharing strategies is that the model will automatically learn some kind of shared universally useful representation, or interlingua, resulting in a single model that can translate between multiple languages.

The existence of such a universal shared representation should naturally entail reasonable performance on zero-shot translation, where a model is evaluated on language pairs it has never seen together during training.

Apart from potential practical benefits like reduced latency costs, zero-shot translation performance is a strong indicator of generalization.

Enabling zero-shot translation with sufficient quality can significantly simplify translation systems, and pave the way towards a single multilingual model capable of translating between any two languages directly.

However, despite being a problem of interest for a lot of recent research, the quality of zero-shot translation has lagged behind pivoting through a common language by 8-10 BLEU points BID15 BID24 BID21 BID27 .

In this paper we ask the question, What is the missing ingredient that will allow us to bridge this gap?

Figure 1 : The proposed multilingual NMT model along with the two training objectives.

CE stands for the cross-entropy loss associated with maximum likelihood estimation for translation between English and other languages.

Align represents the source language invariance loss that we impose on the representations of the encoder.

While training on the translation objective, training samples (x, y) are drawn from the set of parallel sentences, D x,y .

For the invariance losses, (x, y) could be drawn from D x,y for the cosine loss, or independent data distributions for the adversarial loss.

Both losses are minimized simultaneously.

Since we have supervised data only to and from English, one of x or y is always in English.

In BID24 , it was hinted that the extent of separation between language representations was negatively correlated with zero-shot translation performance.

This is supported by theoretical and empirical observations in domain adaptation literature, where the extent of subspace alignment between the source and target domains is strongly associated with transfer performance BID7 BID8 BID17 .

Zero-shot translation is a special case of domain adaptation in multilingual models, where English is the source domain and other languages collectively form the target domain.

Following this thread of domain adaptation and subspace alignment, we hypothesize that aligning encoder representations of different languages with that of English might be the missing ingredient to improving zero-shot translation performance.

In this work, we develop auxiliary losses that can be applied to multilingual translation models during training, or as a fine-tuning step on a pre-trained model, to force encoder representations of different languages to align with English in a shared subspace.

Our experiments demonstrate significant improvements on zero-shot translation performance and, for the first time, match the performance of pivoting approaches on WMT English-French-German (en-fr-de) and the IWSLT 2017 shared task, in all zero shot directions, without any meaningful regression in the supervised directions.

We further analyze the model's representations in order to understand the effect of our explicit alignment losses.

Our analysis reveals that tying weights in the encoder, by itself, is not sufficient to ensure shared representations.

As a result, standard multilingual models overfit to the supervised directions, and enter a failure mode when translating between zero-shot languages.

Explicit alignment losses incentivize the model to use shared representations, resulting in better generalization.2 ALIGNMENT OF LATENT REPRESENTATIONS 2.1 MULTILINGUAL NEURAL MACHINE TRANSLATION Let x = (x 1 , x 2 ...x m ) be a sentence in the source language and y = (y 1 , y 2 , ...

y n ) be its translation in the target language.

For machine translation, our objective is to learn a model, p(y|x; ??).

In modern NMT, we use sequence-to-sequence models supplemented with an attention mechanism BID5 to learn this distribution.

These sequence-to-sequence models consist of an encoder, Enc(x) = z = (z 1 , z 2 , ...z m ) parameterized with ?? enc , and a decoder that learns to map from the latent representation z to y by modeling p(y|z; ?? dec ), again parameterized with ?? dec .

This model is trained to maximize the likelihood of the available parallel data, D x,y .

DISPLAYFORM0 In multilingual training we jointly train a single model BID26 to translate from many possible source languages to many potential target languages.

When only the decoder is informed about the desired target language, a special token to indicate the target language, < tl >, is input to the first step of the decoder.

In this case, D x,y is the union of all the parallel data for each of the supervised translation directions.

Note that either the source or the target is always English.

For zero-shot translation to work, the encoder needs to produce language invariant feature representations of a sentence.

Previous works learn these transferable features by using a weight sharing constraint and tying the weights of the encoders, the decoders, or the attentions across some or all languages BID11 BID24 BID27 BID14 .

They argue that sharing these layers across languages causes sentences that are translations of each other to cluster together in a common representation space.

However, when a model is trained on just the end-to-end translation objective, there is no explicit incentive for the model to discover language invariant representations; given enough capacity, it is possible for the model to partition its intrinsic dimensions and overfit to the supervised translation directions.

This would result in intermediate encoder representations that are specific to individual languages.

We now explore two classes of regularizers, ???, that explicitly force the model to make the representations in all other languages similar to their English counterparts.

We align the encoder representations of every language with English, since it is the only language that gets translated into all other languages during supervised training.

Thus, English representations now form an implicit pivot in the latent space.

The loss function we then minimize is: DISPLAYFORM0 where L CE is the cross-entropy loss and ?? is a hyper-parameter that controls the contribution of the alignment loss ???.

Here we view zero-shot translation through the lens of domain adaptation, wherein English is the source domain and the other languages together constitute the target domain.

BID7 and BID30 have shown that target risk can be bounded by the source risk plus a discrepancy metric between the source and target feature distribution.

Treating the encoder as a deterministic feature extractor, the source distribution is Enc(x en )p(x en ) and the target distribution is Enc(x t )p(x t ).

To enable zero-shot translation, our objective then is to minimize the discrepancy between these distributions by explicitly optimizing the following domain adversarial loss BID17 : DISPLAYFORM0 where Disc is the discriminator and is parametrized by ?? disc .

D En are English sentences and D T are the sentences of all the other languages.

Note that, unlike BID4 BID41 , who also train the encoder adversarially with a language detecting discriminator, we are trying to align the distribution of encoder representations of all other languages to that of English and vice-versa.

Our discriminator is just a binary predictor, independent of how many languages we are jointly training on.

Architecturally, the discriminator is a feed-forward network that acts on the temporally max-pooled representation of the encoder output.

We also experimented with a discriminator that made independent predictions for the encoder representation, z i , at each time-step i, but found the pooling based approach to work better.

More involved discriminators that consider the sequential nature of the encoder representations may be more effective, but we do not explore them in this work.

While adversarial approaches have the benefit of not needing parallel data, they only align the marginal distributions of the encoder's representations.

Further, adversarial approaches are hard to optimize and are often susceptible to mode collapse, especially when the distribution to be modeled is multi-modal.

Even if the discriminator is fully confused, there are no guarantees that the two learned distributions will be identical BID2 .To resolve these potential issues, we attempt to make use of the available parallel data, and enforce an instance level correspondence between the pairs (x, y) ??? D x,y , rather than just aligning the marginal distributions of Enc(x)p(x) and Enc(y)p(y) as in the case of domain-adversarial training.

Previous work on multi-modal and multi-view representation learning has shown that, when given paired data, transferable representations can be learned by improving some measure of similarity between the corresponding views from each mode.

Various similarity measures have been proposed such as Euclidean distance BID22 , cosine distance BID16 , correlation BID1 etc.

In our case, the different views correspond to equivalent sentences in different languages.

Note that Enc(x) and Enc(y) are actually a pair of sequences, and to compare them we would ideally have access to the word level correspondences between the two sentences.

In the absence of this information, we make a bag-of-words assumption and align the pooled representation similar to BID19 ; BID10 .

Empirically, we find that max pooling and minimizing the cosine distance between the representations of parallel sentences similar to works well.

We now minimize the distance function: DISPLAYFORM0

A multilingual model with a single encoder and a single decoder similar to BID24 is our baseline.

This setup maximally enforces the parameter sharing constraint that previous works rely on to promote cross-lingual transfer.

We first train our model solely on the translation loss until convergence, on all languages to and from English.

This is our baseline multilingual model.

We then fine-tune this model with the proposed alignment losses, in conjunction with the translation objective.

We then compare the performance of the baseline model against the aligned models on both the supervised and the zero-shot translation directions.

We also compare our zero-shot performance against the pivoting performance using the baseline model.

For our en???{fr, de} experiments, we train our models on the standard en???fr (39M) and en???de (4.5M) training datasets from WMT'14.

We pre-process the data by applying the standard Moses pre-processing 1 .

We swap the source and target to get parallel data for the fr???en and de???en directions.

The resulting datasets are merged by oversampling the German portion to match the size of the French portion.

This results in a total of 158M sentence pairs.

We get word counts and apply 32k BPE BID33 ) to obtain subwords.

The target language < tl > tokens are also added to the vocabulary.

We use newstest-2012 as the dev set and newstest-2013 as the test set.

Both of these sets are 3-way parallel and have 3003 and 3000 sentences respectively.

We run all our experiments with Transformers BID36 , using the TransformerBase config.

We train our model with a learning rate of 1.0 and 4000 warmup steps.

Input dropout is set to 0.1.

We use synchronized training with 16 Tesla P100 GPUs and train the model for 500k steps.

The model is instructed on which language to translate a given input sentence into, by feeding in a unique < tl > token per target language.

In our implementation, this token is pre-pended into the source sentence, but it could just as easily be fed into the decoder to the same effect.

For the alignment experiments, we fine-tune a pre-trained multilingual model by jointly training on both the alignment and translation losses.

For adversarial alignment, the discriminator is a feedforward network with 3 hidden layers of dimension 2048 using the leaky ReLU(?? = 0.1) nonlinearity.

?? was tuned to 1.0 for both the adversarial and the cosine alignment losses.

Simple fine-tuning with SGD using a learning rate of 1e-4 works well and we do not need to train from scratch.

We observe that the models converge within a few thousand updates.

de ??? f r f r ???

de en ??? f r en ???

de f r ???

Table 1 : Zero-shot results with baseline and aligned models compared against pivoting.

Zero-Shot results are marked zs.

Pivoting through English is performed using the baseline multilingual model.

Our results, in Table 1 , demonstrate that both our approaches to align representations result in large improvements in zero-shot translation quality for both directions, effectively closing the gap to the performance of the strong pivoting baseline.

We didn't notice any significant differences between the performance of the two proposed alignment methods.

Importantly, these improvements come at no cost to the quality in the supervised directions.

While both the proposed approaches aren't significantly different in terms of final quality, we noticed that the adversarial regularizer was very sensitive to the initialization scheme and the choice of hyper-parameters.

In comparison, the cosine distance loss was relatively stable, with ?? being the only hyper-parameter controlling the weight of the alignment loss with respect to the translation loss.

We further analyze the outputs of our baseline multilingual model in order to understand the effect of alignment on zero-shot performance.

We identify the major effects that contribute to the poor zeroshot performance in multilingual models, and investigate how an explicit alignment loss resolves these pathologies.

94% 0% f r references 4% 0% 96% de references 4% 96% 0% Table 2 : Percentage of sentences by language in reference translations and the sentences decoded using the baseline model (newstest2012)While investigating the high variance of the zero-shot translation score during multilingual training in the absence of alignment, we found that a significant fraction of the examples were not getting translated into the desired target language at all.

Instead, they were either translated to English or simply copied.

This phenomenon is likely a consequence of the fact that at training time, German and French source sentences were always translated into English.

Because of this, the model never learns to properly attribute the target language to the < tl > token, and simply changing the < tl > token at test time is not effective.

We count the number of sentences in each language using an automatic language identification tool and report the results in Table 2 .Further, we find that for a given sentence, all output tokens tend to be in the same language, and there is little to no code-switching.

This was also observed by BID24 , where it was explained as a cascading effect in the decoder: Once the decoder starts emitting tokens in one language, the conditional distribution p(y i |y i???1 , ..., y 1 ) is heavily biased towards that particular language.

With explicit alignment, we remove the target language information encoded into the source token representations.

In the absence of this confounding information, the < tl > target token gives us more control to set the translation direction.

Table 3 : BLEU on subset of examples predicted in the right language by the direct translation using the baseline system (newstest2012)Here we try to isolate the gains our system achieves due to improvements in the learning of transferable features, from those that can be attributed to decoding to the desired language.

We discount the errors that could be attributed to incorrect language errors and inspect the translation quality on the subset of examples where the baseline model decodes in the right language.

We re-evaluate the BLEU scores of all systems and show the results in Table 3 .

We find that the vanilla zero-shot translation system (Baseline) is much stronger than expected at first glance.

It only lags the pivoting baseline by 0.5 BLEU points on French to German and by 2.7 BLEU points on German to French.

We can now see that, even on this subset which was chosen to favor the baseline model, the representation alignment of our adapted model contributes to improving the quality of zero-shot translation by 0.7 and 2.2 BLEU points on French to German and German to French, respectively.

We design a simple experiment to determine whether representations learned while training a multilingual translation model are truly cross-lingual.

We probe our baseline and aligned multilingual models with 3-way aligned data to determine the extent to which their representations are functionally equivalent, during different stages in model training.

Because source languages can have different sequence lengths and word orders for equivalent sentences, it is not possible to directly compare encoder output representations.

However, it is possible to directly compare the representations extracted by the decoder from the encoder outputs for each language.

Suppose we want to compare representations of semantically equivalent English and German sentences when translating into French.

At time-step i in the decoder, we use the model to predict p(y i |Enc(x en ), y 1:(i???1) ) and p(y i |Enc(x de ), y 1:(i???1) ).

However, in the seq2seq with attention formulation, these problems reduce to predicting p(y i |c We use a randomly sampled set of 100 parallel en-de-fr sentences extracted from our dev set, newstest2012, to perform this analysis.

For each set of aligned sentences, we obtain the sequence of aligned context vectors (c en i , c de i ) and plot the mean cosine distances for our baseline training run, and the incremental runs with alignment losses in FIG1 .

Our results indicate that the vanilla multilingual model learns to align encoder representations over the course of training.

However, in the absence of an external incentive, the alignment process arrests as training progresses.

Incrementally training with the alignment losses results in a more language-agnostic representation, which contributes to the improvements in zero-shot performance.

Given the good results on WMT en-fr-de, we now extend our experiments, to test the scalability of our approach to multiple languages.

We work with the IWSLT-17 dataset which has transcripts of Ted talks in 5 languages: English (en), Dutch (nl), German (de), Italian (it), and Romanian (ro).

The original dataset is multi-way parallel with approximately 220 thousand sentences per language, but for the sake of our experiments we only use the to/from English directions for training.

The dev and test sets are also multi-way parallel and comprise around 900 and 1100 sentences per language pair respectively.

We again use the transformer base architecture.

We set the learning rate to 2.0 and the number of warmup steps to 8k.

A dropout rate of 0.2 was applied to all connections of the transformer.

We use the cosine loss with ?? set to 0.001 because of how easy it is to tune.

Our baseline model's scores on IWSLT-17 are suspiciously close to that of bridging, as seen in TAB3 .

We suspect this is because the data that we train on is multi-way parallel, and the English sentences are shared across the language pairs.

This may be helping the model learn shared representations with the English sentences acting as pivots.

Even so, we are able to gain 1 BLEU over the strong baseline system and demonstrate the applicability of our approach to larger groups of languages.5 RELATED WORK

Multilingual NMT models were first proposed by BID11 and have since been explored in BID14 ; BID9 and several other works.

While zero-shot translation was the direct goal of BID14 , they were only able to achieve 'zero-resource translation', by using their pre-trained multi-way multilingual model to generate pseudo-parallel data for fine-tuning.

BID24 were the first to show the possibility of zero-shot translation by proposing a model that shared all the components and used a token to indicate the target language.

BID32 propose a novel way to modulate the amount of sharing between languages, by using a parameter generator to generate the parameters for either the encoder or the decoder of the multilingual NMT system based on the source and target languages.

They also report higher zero-shot translation scores with this approach.

Learning coordinated representations with the use of parallel data has been explored thoroughly in the context of multi-view and multi-modal learning BID6 .

These often involve either auto-encoder like networks with a reconstruction objective, or paired feed-forward networks with a similarity based objective .

This function used to encourage similarity may be Euclidean distance BID22 , cosine distance BID16 , partial order BID37 , correlation BID1 , etc.

More recently a vast number of adversarial approaches have been proposed to learn domain invariant representations, by ensuring that they are indistinguishable by a discriminator network BID17 .The use of aligned parallel data to learn shared representations is common in the field of crosslingual or multilingual representations, where work falls into three main categories.

Obtaining representations from word level alignments -bilingual dictionaries or automatically generated word alignments -is the most popular approach BID13 BID42 .

The second category of methods try to leverage document level alignment, like parallel Wikipedia articles, to generate cross-lingual representations BID34 BID38 .

The final category of methods often use sentence level alignments, in the form of parallel translation data, to obtain cross-lingual representations BID23 BID18 BID29 BID0 .

Recent work by BID12 showed that the representations learned by a multilingual NMT system are widely applicable across tasks and languages.

Parameter sharing based approaches have also been tried in the context of unsupervised NMT, where learning a shared latent space BID3 was believed to improve translation quality.

Some approaches explore applying adversarial losses on the encoder, to ensure that the representations are language agnostic.

However, recent work has shown that enforcing a shared latent space is not important for unsupervised NMT BID25 , and the cycle consistency loss suffices by itself.

In this work we propose explicit alignment losses, as an additional constraint for multilingual NMT models, with the goal of improving zero-shot translation.

We view the zero-shot NMT problem in the light of subspace alignment for domain adaptation, and propose simple approaches to achieve this.

Our experiments demonstrate significantly improved zero-shot translation performance that are, for the first time, comparable to strong pivoting based approaches.

Through careful analyses we show how our proposed alignment losses result in better representations, and thereby better zeroshot performance, while still maintaining performance on the supervised directions.

Our proposed methods have been shown to work reliably on two public benchmarks datasets: WMT EnglishFrench-German and the IWSLT 2017 shared task.

@highlight

Simple similarity constraints on top of multilingual NMT enables high quality translation between unseen language pairs for the first time.