Most state-of-the-art neural machine translation systems, despite being different in architectural skeletons (e.g., recurrence, convolutional), share an indispensable feature: the Attention.

However, most existing attention methods are token-based and ignore the importance of phrasal alignments, the key ingredient for the success of phrase-based statistical machine translation.

In this paper, we propose novel phrase-based attention methods to model n-grams of tokens as attention entities.

We incorporate our phrase-based attentions into the recently proposed Transformer network, and demonstrate that our approach yields improvements of 1.3 BLEU for English-to-German and 0.5 BLEU for German-to-English translation tasks, and 1.75 and 1.35 BLEU points in English-to-Russian and Russian-to-English translation tasks  on WMT newstest2014 using WMT’16 training data.

Neural Machine Translation (NMT) has established breakthroughs in many different translation tasks, and has quickly become the standard approach to machine translation.

NMT offers a simple encoder-decoder architecture that is trained end-to-end.

Most NMT models (except a few like BID5 and BID4 ) possess attention mechanisms to perform alignments of the target tokens to the source tokens.

The attention module plays a role analogous to the word alignment model in Statistical Machine Translation or SMT BID9 .

In fact, the Transformer network introduced recently by BID19 achieves state-of-the-art performance in both speed and BLEU scores BID12 by using only attention modules.

On the other hand, phrasal interpretation is an important aspect for many language processing tasks, and forms the basis of Phrase-Based Machine Translation BID9 .

Phrasal alignments BID10 can model one-to-one, one-to-many, many-to-one, and many-to-many relations between source and target tokens, and use local context for translation.

They are also robust to non-compositional phrases.

Despite the advantages, the concept of phrasal attentions has largely been neglected in NMT, as most NMT models generate translations token-by-token autoregressively, and use the token-based attention method which is order invariant.

Therefore, the intuition of phrase-based translation is vague in existing NMT systems that solely depend on the underlying neural architectures (recurrent, convolutional, or self-attention) to incorporate contextual information.

However, the information aggregation strategies employed by the underlying neural architectures provide context-relevant clues only to represent the current token, and do not explicitly model phrasal alignments.

We argue that having an explicit inductive bias for phrases and phrasal alignments is necessary for NMT to exploit the strong correlation between source and target phrases.

In this paper, we propose phrase-based attention methods for phrase-level alignments in NMT.

Specifically, we propose two novel phrase-based attentions, namely CONVKV and QUERYK, designed to assign attention scores directly to phrases in the source and compute phrase-level attention vector for the target.

We also introduce three new attention structures, which apply these methods to conduct phrasal alignments.

Our homogeneous and heterogeneous attention structures perform token-to-token and token-to-phrase mappings, while the interleaved heterogeneous attention structure models all token-to-token, token-to-phrase, phrase-to-token, and phrase-to-phrase alignments.

To show the effectiveness of our approach, we apply our phrase-based attention methods to all multi-head attention layers of the Transformer.

Our experiments on WMT'14 translation tasks show improvements of up to 1.3 and 0.5 BLEU points for English-to-German and German-to-English respectively, and up to 1.75 and 1.35 BLEU points for English-to-Russian and Russian-to-English respectively, compared to the baseline Transformer network trained in identical settings.

Most NMT models adopt an encoder-decoder framework, where the encoder network first transforms an input sequence of symbols x = (x 1 , x 2 , . . .

, x n ) to a sequence of continuous representations Z = (z 1 , z 2 , . . .

, z n ), from which the decoder generates a target sequence of symbols y = (y 1 , y 2 , . . . , y m ) autoregressively, one element at a time.

Recurrent seq2seq models with diverse structures and complexity BID17 BID1 BID11 BID23 are the first to yield state-of-the-art results.

Convolutional seq2seq models BID7 BID3 alleviate the drawback of sequential computation of recurrent models and leverage parallel computation to reduce training time.

The recently proposed Transformer network BID19 structures the encoder and the decoder entirely with stacked self-attentions and cross-attentions (only in the decoder).

In particular, it uses a multi-headed, scaled multiplicative attention defined as follows: DISPLAYFORM0 DISPLAYFORM1 where S is the softmax function, Q, K, V are the matrices with query, key, and value vectors, respectively, d k is the dimension of the query/key vectors; DISPLAYFORM2 v are the head-specific weights for query, key, and value vectors, respectively; and W is the weight matrix that combines the outputs of the heads.

The attentions in the encoder and decoder are based on self-attention, where all of Q, K and V come from the output of the previous layer.

The decoder also has crossattention, where Q comes from the previous decoder layer, and the K-V pairs come from the encoder.

We refer readers to BID19 for further details of the network design.

One crucial issue with the attention mechanisms employed in the Transformer network as well as other NMT architectures BID11 BID3 ) is that they are order invariant locally and globally.

That is, changing the order of the vectors in Q, K and V does not change the resulted attention weights and vectors.

If this problem is not tackled properly, the model may not learn the sequential characteristics of the data.

RNN-based models BID1 BID11 tackle this issue with a recurrent encoder and decoder, CNN-based models like BID3 use position embeddings, while the Transformer uses positional encoding.

Another limitation is that these attention methods attend to tokens, and play a role analogous to word alignment models in traditional SMT.

It is, however, well admitted in SMT that phrases are better than words as translation units BID9 .

Without explicit attention to phrases, a particular attention function has to depend entirely on the token-level softmax scores of a phrase for phrasal alignment, which is not robust and reliable, thus making it more difficult for the model to learn the required mappings.

For example, the attention heatmaps of the Transformer BID19 show concentration of the scores on individual tokens even if it uses multiple heads concurrently in multiple layers.

Our main hypothesis is that in order to exploit the strong correlation between source and target phrases, the NMT models should have explicit inductive biases towards phrases.

There exists some research on phrase-based decoding in NMT framework.

For example, BID4 proposed a phrase-based decoding approach based on a soft reordering layer and a Sleep-WAke Network (SWAN), a segmentation-based sequence model proposed by BID21 .

Their decoder uses a recurrent architecture without any attention on the source.

BID18 and BID22 used an external phrase memory to decode phrases for a Chinese-toEnglish translation task.

In addition, hybrid search and PBMT were introduced to perform phrasal translation in BID2 .

Nevertheless, to the best of our knowledge, our work is the first to embed phrases into attention modules, which thus propagate the information throughout the entire end-to-end Transformer network, including the encoder, decoder, and the cross-attention.

In this section, we present our proposed methods to compute attention weights and vectors based on n-grams of queries, keys, and values.

We compare and discuss the pros and cons of these methods.

For simplicity, we describe them in the context of the Transformer network; however, it is straightforward to apply them to other architectures such as RNN-based or CNN-based seq2seq models.

In this subsection, we present two novel methods to achieve phrasal attention.

In Subsection 3.2, we present our methods for combining different types of n-gram attentions.

The key element in our methods is a temporal (or one-dimensional) convolutional operation that is applied to a sequence of vectors representing tokens.

Formally, we can define the convolutional operator applied to each token x t with corresponding vector representation x t ∈ IR d1 as: DISPLAYFORM0 where ⊕ denotes vector concatenation, w ∈ IR n×d1 is the weight vector (a.k.a.

kernel), and n is the window size.

We repeat this process with d 2 different weight vectors to get a d 2 -dimensional latent representation for each token x t .

We will use the notation Conv n (X, W ) to denote the convolution operation over an input sequence X with window size n and kernel weights W ∈ IR n×d1×d2 .

The intuition behind key-value convolution technique is to use trainable kernel parameters W k and W v to compute the latent representation of n-gram sequences using convolution operation over key and value vectors.

The attention function with key-value convolution is defined as: DISPLAYFORM0 where S is the softmax function, DISPLAYFORM1 n×dv×dv are the respective kernel weights for Q, K and V .

Throughout this paper, we will use S to denote the softmax function.

Note that in this convolution, the key and value sequences are left zero-padded so that the sequence length is preserved after the convolution (i.e., one latent representation per token).The CONVKV method can be interpreted as indirect query-key attention, in contrast to the direct query-key approach to be described next.

This means that the queries do not interact directly with the keys to learn the attention weights; instead the model relies on the kernel weights (W k ) to learn n-gram patterns.

In order to allow the queries to directly and dynamically influence the word order of phrasal keys and values, we introduce Query-as-Kernel attention method.

In this approach, when computing the attention weights, we use the query as kernel parameters in the convolution applied to the series of keys.

The attention output in this approach is given by: DISPLAYFORM0 where DISPLAYFORM1 n×dv×dv are trainable weights.

Notice that we include the window size n (phrase length) in the scaling factor to counteract the fact that there are n times more multiplicative operations in the convolution than the traditional matrix multiplication.

Having presented the two phrase-based attention methods in the previous subsection, we now introduce our extensions to the multi-headed attention framework of the Transformer to enable it to pay attention not only to tokens but also to phrases across many sub-spaces and locations.

In homogeneous n-gram attention, we distribute the attention heads to different n-gram types with each head attending to one particular n-gram type (n = 1, 2, . . .

, N ).

For instance, FIG0 shows a homogeneous structure, where the first four heads attend to unigrams, and the last four attend to bigrams.

A head can apply one of the phrasal attention methods described in Subsection 3.1.

The selection of which n-gram to assign to how many heads is considered as hyperparameters to the model.

Since all heads must have consistent sequence length, phrasal attention heads in the homogeneous setting require left-padding of keys and values before convolution.

Since each head attends to a subspace resulting from one type of n-gram, homogeneous attention learns the mappings in a distributed way.

However, the homogeneity restriction may limit the model to learn interactions between different n-gram types since the gradients for different n-gram types flow in parallel paths with no explicit interactions.

Furthermore, the homogeneous heads force the model to assign each query with attentions on all n-gram types (e.g., unigrams and bigrams) even when it does not need to do so, thus possibly inducing more noise into the model.

The heterogeneous n-gram attention relaxes the constraint of the homogeneous approach.

Instead of limiting each head's attention to a particular type of n-gram, it allows the query to freely attend to all types of n-grams simultaneously.

To achieve this, we first compute the attention logit for each n-gram type separately (i.e., for n = 1, 2, . . .

, N ), then we concatenate all the logits before passing them through the softmax layer to compute the attention weights over all n-gram types.

Similarly, the value vectors for the n-gram types are concatenated to produce the overall attention output.

FIG1 demonstrates the heterogeneous attention process for unigrams and bigrams.

For CONVKV technique in Equation 5, the attention output is given by: For QUERYK technique (Equation 6), the attention output is given as follows: DISPLAYFORM0 DISPLAYFORM1 Note that in heterogeneous attention, we do not need to pad the input sequences before the convolution operation to ensure identical sequence length.

Also, the key/value sequences that are shorter than the window size do not have any valid phrasal component to be attended.

All the methods presented above perform attention mappings from token-based queries to phrasebased key-value pairs.

In other words, they model token-to-token and token-to-phrase structures.

These types of attentions are beneficial when there exists a translation of a phrase in the source language (keys and values) to a single token in the target language (query).

However, these methods are not explicitly designed to work in the reverse direction when a phrase or a token in the source language should be translated to a phrase in the target language.

In this section, we present a novel approach to heterogeneous phrasal attention that allows phrasal queries to attend to tokens and phrases of keys and values (i.e., phrase-to-token and phrase-to-phrase mappings).We accomplish this with the QUERYK and CONVKV methods as follows.

We first apply convolutions Conv n (Q, W qn ) on the query sequence with window size n to obtain the n-gram hidden representations of the query.

Consider FIG2 , where we apply convolution on Q for n = 1 and n = 2 to generate the respective unigram and bigram queries.

1 These queries are then used to attend over unigram and bigram key-values to generate the heterogeneous attention vectors as follows.

DISPLAYFORM0 A2,ConvKV = S( DISPLAYFORM1 A1,QueryK = S([ DISPLAYFORM2 A2,QueryK = S([ DISPLAYFORM3 The result of these operations (Eq. 9 -12) is a sequence of unigram and bigram attention vectors A 1 = (u 1 , u 2 , . . . , u N ) and A 2 = (b 1 , b 2 , . . .

, b N −1 ) respectively, where N is the query length.

Note that each u i ∈ A 1 represents the attention vector for Q i unigram query, and each b i ∈ A 2 represents the attention vector for (Q i -Q i+1 ) bigram queries, and these vectors are generated by attending to n-grams (n = 1, 2, . . .

, N ) of keys and values.

In the next step, the phrase-level attentions in A 2 are interleaved with the unigram attentions in A 1 to form an interleaved attention sequence I such that the vectors are aligned.

For unigram and bigram queries, the interleaved vector sequences at the encoder and decoder are formed as DISPLAYFORM4 where I enc and I dec denote the interleaved sequence for self-attention at the encoder and decoder respectively, and I cross denotes the interleaved sequence for cross-attention between the encoder and the decoder.

Note that although I dec and I cross are computed using the same formula (Eq. 14), they are different entities, operating over different input sequences -the input to I dec comes from the self-attended features in the target side, whereas the input to I cross comes from the cross-attended features from source.

Also, to prevent information flow from the future in the decoder, the right connections are masked out in I dec and I cross (similar to the original Transformer).

The interleaving operation places the phrase-and token-based representations of a token next to each other.

The interleaved vectors are finally passed through a convolution layer (as opposed to a point-wise feedforward layer in the Transformer) to compute the overall representation for each token.

By doing so, each query is intertwined with the n-gram representations of the phrases containing itself, which enables the model to learn the query's correlation with neighboring tokens.

For unigram and bigram queries, the encoder uses a convolution layer with a window size of 3 and stride of 2 to allow the token to intertwine with its past and future phrase representations, while the ones in the decoder (self-and cross-attention) use a window size of 2 and stride of 2 to incorporate only the past phrase representations to preserve the autoregressive property.

More formally, DISPLAYFORM5

In this section, we present the training settings, experimental results and analysis of our models.

We preserve most of the training settings from BID19 to enable a fair comparison with the original Transformer.

Specifically, we use the Adam optimizer BID8 with β 1 = 0.9, β 2 = 0.98, and = 10 −9 .

We follow a similar learning rate schedule with warmup steps of 16000: LearningRate = 2 * d −0.5 * min(step num −0.5 , step num * warmup steps −1.5 ).

While BID19 trained their base and big models at a massive scale with 8 GPUs, we could train our models only on a single GPU because of limited GPU facilities.

We trained our models and the baseline Transformer on the same GPU for 500,000 steps.

The batches were formed by sentence pairs containing approximately 4096 source and 4096 target tokens.

Similar to BID19 , we also applied residual dropout with 0.1 probability and label smoothing with ls = 0.1.

Our models are implemented in the tensor2tensor 2 library , on top of the original Transformer codebase.

We conducted all the experiments with our models and the original Transformer in an identical setup for a fair comparison.

We trained our models on the standard WMT'16 English-German (En-De) and English-Russian (En-Ru) datasets constaining about 4.5 and 25 million sentence pairs, respectively.

We used WMT newstest2013 as our development sets and newstest2014 as our test sets for all the translation tasks.

We used byte-pair encoding BID14 with combined source and target vocabulary of 37,000 sub-words for English-German and 40,000 sub-words for English-Russian.

We took the average of the last 5 checkpoints (saved at 10,000-iteration intervals) for evaluation, and used a beam search size of 5 and length penalty of 0.6 BID23 .

Table 1 : BLEU (cased) scores on WMT'14 testsets for English-German (En-De) and EnglishRussian (En-Ru) language pairs (in both directions).

All models were trained with 1 GPU.

The # Parameters is shown in approximate terms.

For homogeneous models, the N-grams denote how we distribute the 8 heads to different n-gram types; e.g., '3/2/3' means 3 heads on unigrams, 2 on bigrams and 3 on trigrams.

For heterogeneous, the numbers indicate the phrase lengths of the collection of n-gram components jointly attended by each head; e.g., '1-2' means attention scores are computed across unigram and bigram logits.

Table 1 compares our model variants with the Transformer base and big models for En-De and EnRu translation tasks (both directions).

We notice that almost all of our models achieve higher BLEU scores than the Transformer base, showing the effectiveness of our approach.

On the En→De translation task, our best homogeneous model (QUERYK with 3/2/3 head distribution) achieves a BLEU of 26.86, already outperforming the Transformer base by about 0.8 points.

When we compare the results of the homogeneous models with those of the heterogeneous, we notice even higher scores for the heterogeneous models; the best heterogeneous model yields a BLEU of 27.15, which is about 1.1 points higher than the Transformer base.

Finally, we notice that our interleaved heterogeneous models surpass all aforementioned scores achieving up to 27.4 BLEU and establishing a 1.33 BLEU improvement over the Transformer base.

This demonstrates the existence of phrase-to-token and phrase-to-phrase mappings from target (De) to source (En) language.

Likewise, on De→En, our models achieve improvements compared to the Transformer, but the gain is not as high as in En→De.

Specifically, homogeneous and heterogeneous attentions perform comparably, giving up to +0.38 BLEU improvements compared to the Transformer base.

Our interleaved models show more improvements (up to 30.3 BLEU), outperforming the Transformer by about 0.5 points.

This again demonstrates the importance of phrase-level query representation in the target.

Similarly, on the En→Ru translation task, all of our models surpass the Transformer base.

Homogeneous models with CONVKV perform slightly better than their QUERYK counterparts.

In the heterogeneous family, all of the model variants experimented with achieved higher score than the homogeneous models.

The QUERYK heterogeneous model with 1-2-3-4 N-grams achieves the highest performance with 37.39 BLEU, 1.75 points higher than the Transformer base.

Interleaved homogeneous models with 1-2 N-grams also perform well on this task with a BLEU of 37.24, which is more than the heterogeneous models with 1-2-3 N-grams, having similar number of parameters.

On the Ru→En task, homogeneous models perform slightly better than the Transformer base, while heterogeneous models excel in performance.

In particular, CONVKV heterogeneous model with 1-2-3-4 N-grams achieves 35.91 BLEU, which outperforms the Transformer base by 1.35 BLEU points.

However, interleaved attention models do not provide significant improvements on this task.

We suspect this is because the morphological richness of Russian is not fully leveraged in the current interleaved models as they are limited to only 1-2 N-grams.

We put this issue in our future work.

Table 2 : BLEU scores on WMT'14 English-to-German translation task for different models with different training settings.

The first block presents reported results along with the number of GPUs, batch size, and number of update steps used from the respective papers.

The second and third block show results in our training setting with 1 GPU.Homogeneous vs. Heterogeneous.

Heterogeneous models generally perform better than their homogeneous counterparts.

This shows the effectiveness of relaxing the 'same n-gram type' attention constraint in the heterogeneous head, which allows it to attend to all n-gram types simultaneously, avoiding any forced attention to a particular n-gram type.

In fact, our experiments show that the models perform worse than the baseline if we remove token-level attentions.

Effect of Higher-Order N-grams.

We now analyze the effect of including higher-order n-grams.

For homogeneous models, including 3-gram components does not have any significant effect in performance, while it increases the number of parameters.

On the other hand, heterogeneous models benefit significantly from 3-gram and 4-gram components.

For En→De, 3-gram and 4-gram components offer 0.1 -0.2 improvements in BLEU.

Meanwhile, for En → Ru and Ru → En, they yield up to 0.64 BLEU improvements compared to the model using only uni-and bi-grams.

This supports the argument that the usefulness of 3-grams and 4-grams depends on the language pair, and our heterogeneous models are more robust in selecting any type of n-grams that are useful.

Comparison with State-of-the-art.

Table 2 places our BLEU scores with the state-of-the-art results reported on the En→De translation task.

Note that BID19 and others conducted their experiments at a massive scale with 8 GPUs, which we could not replicate due to limited number of GPUs.

Therefore, we conducted all experiments using a single GPU, and make comparisons in this 'low-resource' setting.

There have been many evidences that practical training of the Transformer networks (theirs and ours) is significantly susceptible to the batch size (which increases with the number of GPUs used), and training on a single GPU with lower batch size for sufficiently long does not produce similar results as in 8 or more GPU settings; see BID13 for details.

As can be noticed, in our training setting (1 GPU), the Transformer big performs only slightly better than the Transformer base, giving 26.63 BLEU in contrast to the reported 28.40 BLEU with 8 GPUs.

We believe, the difference in performance is due to the batch size, for which we could conduct trials only at 2048 tokens instead of 25000, although we trained the big model 10 times longer.

In order to verify if phrasal attention (which adds an additional convolution layer in parallel to the original Transformer layer), does indeed help more than just placing extra Transformer layers, we conducted additional experiments with the Transformer base with 10 layers, and with our heterogeneous model with 4 layers.

We notice that our heterogeneous model with 6 layers outperforms the Transformer base with 10 layers and the Transformer big, and with 4 layers, it performs on par, even though it has much less parameters.

These differences are even more significant for En→Ru, where the Transformer big underperforms the Transformer base with a BLEU of 34.64 (Table 1) .

On the other hand, our heterogeneous model achieves 37.39 BLEU (+ 2.75).

Therefore, our phrasal attention can achieve similar or higher results with a wider but shallower network that consumes less parameters compared to deep Transformer base or Transformer big.

This increases parallelizability.

Phrasal attentions are also more suitable for 'low-resource' (GPU) setup, and for complex language pairs like English-Russian, where our methods show larger improvements.

Model Interpretation.

To interpret our phrasal attention models, we now discuss how they learn the alignments.

FIG3 shows attention heatmaps for an En→De sample in newstest2014; figure 4a displays the heatmap in layer 3 (mid layer), while FIG3 shows the one in layer 6 (top layer) within a 6-layer Transformer based on our interleaved attention.

Each figure shows 4 quadrants representing token-to-token, token-to-phrase, phrase-to-token, phrase-to-phrase attentions, respectively.

We can see in this example that phrasal attentions are activated strongly in the mid-layers; particularly, the phrase-to-phrase attention is the most concentrated one.

On the other hand, token-to-token attention is activated the most in the top layer.

Although the distribution of attentions can be quite different depending on the model initialization, we observed a large portion of the attentions for phrases, as shown in TAB4 TAB4 show the distributions of phrase-and token-based attentions across different layers for two different random seeds.

Depending on the initial states, the allocations can be different, but phrasal attentions always play an important role in learning the mapping from source to target.

Layer token-to-token token-to-phrase phrase-to-token phrase-to-phrase Table 3 : For model trained with seed 100, percentage (%) of activations for different attention types in each layer of the Interleaved model for English-to-German translation task in newstest2014.Layer token-to-token token-to-phrase phrase-to-token phrase-to-phrase

@highlight

Phrase-based attention mechanisms to assign attention on phrases, achieving token-to-phrase, phrase-to-token, phrase-to-phrase attention alignments, in addition to existing token-to-token attentions.

@highlight

Paper presents an attention mechanism that computes a weighted sum over not only single tokens but ngrams(phrases).