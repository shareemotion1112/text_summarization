Experimental evidence indicates that simple models outperform complex deep networks on many unsupervised similarity tasks.

Introducing the concept of an optimal representation space, we provide a simple theoretical resolution to this apparent paradox.

In addition, we present a straightforward procedure that, without any retraining or architectural modifications, allows deep recurrent models to perform equally well (and sometimes better) when compared to shallow models.

To validate our analysis, we conduct a set of consistent empirical evaluations and introduce several new sentence embedding models in the process.

Even though this work is presented within the context of natural language processing, the insights are readily applicable to other domains that rely on distributed representations for transfer tasks.

Distributed representations have played a pivotal role in the current success of machine learning.

In contrast with the symbolic representations of classical AI, distributed representation spaces can encode rich notions of semantic similarity in their distance measures, allowing systems to generalise to novel inputs.

Methods to learn these representations have gained significant traction, in particular for modelling words BID30 .

They have since been successfully applied to many other domains, including images BID15 BID39 and graphs BID25 BID17 BID33 .Using unlabelled data to learn effective representations is at the forefront of modern machine learning research.

The Natural Language Processing (NLP) community in particular, has invested significant efforts in the construction BID30 BID37 BID10 BID21 , evaluation and theoretical analysis BID28 of distributed representations for words.

Recently, attention has shifted towards the unsupervised learning of representations for larger pieces of text, such as phrases BID50 BID51 , sentences BID22 BID43 BID19 BID7 , and entire paragraphs BID27 .

Some of this work simply sums or averages constituent word vectors to obtain a sentence representation BID32 BID31 BID48 BID7 , which is surprisingly effective but naturally cannot leverage any contextual information.

Another line of research has relied on a sentence-level distributional hypothesis BID38 , originally applied to words BID18 , which is an assumption that sentences which occur in similar contexts have a similar meaning.

Such models often use an encoder-decoder architecture BID12 to predict the adjacent sentences of any given sentence.

Examples of such models include SkipThought , which uses Recurrent Neural Networks (RNNs) for its encoder and decoders, and FastSent BID19 , which replaces the RNNs with simpler bagof-words (BOW) versions.

Models trained in an unsupervised manner on large text corpora are usually applied to supervised transfer tasks, where the representation for a sentence forms the input to a supervised classification problem, or to unsupervised similarity tasks, where the similarity (typically taken to be the cosine similarity) of two inputs is compared with corresponding human judgements of semantic similarity in order to inform some downstream process, such as information retrieval.

Interestingly, some researchers have observed that deep complex models like SkipThought tend to do well on supervised transfer tasks but relatively poorly on unsupervised similarity tasks, whereas for shallow log-linear models like FastSent the opposite is true BID19 BID13 .

It has been highlighted that this should be addressed by analysing the geometry of the representation space BID6 BID42 BID19 , however, to the best of our knowledge it has not been systematically attempted 1 .In this work we attempt to address the observed performance gap on unsupervised similarity tasks between representations produced by simple models and those produced by deep complex models.

Our main contributions are as follows:• We introduce the concept of an optimal representation space, in which the space has a similarity measure that is optimal with respect to the objective function.• We show that models with log-linear decoders are usually evaluated in their optimal space, while recurrent models are not.

This effectively explains the performance gap on unsupervised similarity tasks.•

We show that, when evaluated in their optimal space, recurrent models close that gap.

We also provide a procedure for extracting this optimal space using the decoder hidden states.•

We validate our findings with a series of consistent empirical evaluations utilising a single publicly available codebase.

We begin by considering a general problem of learning a conditional probability distribution P model (y | x) over the output symbols y ∈ Y given the input symbols x ∈ X .

Definition 1.

A space H combined with a similarity measure ρ : H × H → R in which semantically close symbols s i , s j ∈ S have representations h i , h j ∈ H that are close in ρ is called a distributed representation space BID16 .In general, a distributed representation of a symbol s is obtained via some function h s = f (s; θ f ), parametrised by weights θ f .

Distributed representations of the input symbols are typically found as the layer activations of a Deep Neural Network (DNN).

One can imagine running all possible x ∈ X through a DNN and using the activations h x of the n th layer as vectors in H x : DISPLAYFORM0 The distributed representation space of the output symbols H y can be obtained via some function h y = g(y; θ g ) that does not depend on the input symbol x, e.g. a row of the softmax projection matrix that corresponds to the output y.

In practice, although H obtained in such a manner with a reasonable vector similarity ρ (such as cosine or Euclidean distance) forms a distributed representation space, there is no a priori reason why an arbitrary choice of a similarity function would be appropriate given H and the model's objective.

There is no analytic guarantee, for arbitrarily chosen H and ρ, that small changes in semantic similarity of symbols correspond to small changes in similarity ρ between their vector representations in H and vice versa.

This motivates Definition 2.

A space H equipped with a similarity measure ρ such that log P model (y | x) ∝ ρ (h y , h x ) is called an optimal representation space.

In words, if a model has an optimal representation space, the conditional log-probability of an output symbol y given an input symbol x is proportional to the similarity ρ(h y , h x ) between their corresponding vector representations h y , h x ∈ H.For example, consider the following standard classification model DISPLAYFORM0 where u y is the y th row of the output projection matrix U.If H x = {DNN(x) | x ∈ X } and H y = {u y | y ∈ Y}, then H = H x ∪ H y equipped with ρ(h 1 , h 2 ) = h 1 · h 2 (the dot product) is an optimal representation space.

Note that if the exponents of Equation FORMULA1 contained Euclidean distance, then we would find log P model (y | x) ∝ ||u y − DNN(x)|| 2 .

The optimal representation space would then be equipped with Euclidean distance as its optimal distance measure ρ.

This easily extends to any other distance measures desired to be induced on the optimal representation space.

Let us elaborate on why Definition 2 is a reasonable definition of an optimal space.

Let x 1 , x 2 ∈ X be the input symbols and y 1 , y 2 ∈ Y their corresponding outputs.

Using DISPLAYFORM1 to denote that a and b are close under ρ, a reasonable model trained on a subset of (X , Y) will ensure that h x1 ρ ∼ h y1 and h x2 ρ ∼ h y2 .

If x 1 and x 2 are semantically close and assuming semantically close input symbols have similar outputs, we also have that h x1 ρ ∼ h y2 and h x2 ρ ∼ h y1 .

Therefore it follows that h x1 ρ ∼ h x2 (and h y1 ρ ∼ h y2 ).

Putting it differently, semantic similarity of input and output symbols translates into closeness of their distributed representations under ρ, in a way that is consistent with the model.

Note that any model P model (y | x) parametrised by a continuous function can be approximated by a function in the form of Equation (1).

It follows that any model that produces a probability distribution has an optimal representation space.

Also note that the optimal space for the inputs does not necessarily have to come from the final layer before the softmax projection but instead can be constructed from any layer, as we now demonstrate.

Let n be the index of the final activation before the softmax projection and let k ∈ {1, . . .

, n}. We split the network into three parts: DISPLAYFORM2 where G k contains first k layers, F n contains the remaining n − k layers and U is the softmax projection matrix.

Let the space for inputs H x be defined as DISPLAYFORM3 and the space for outputs H y defined as DISPLAYFORM4 where DISPLAYFORM5 is again an optimal representation space.

We will show a specific example where this holds in Section 3.3.

For the remainder of this paper, we focus on unsupervised models for learning distributed representations of sentences, an area of particular interest in NLP.

si consists of words from a pre-defined vocabulary V of size |V |.We transform the corpus into a set of pairs DISPLAYFORM0 , where s i ∈ S and c i is a context of s i .

The context usually (but not necessarily) contains some number of surrounding sentences of s i , e.g. c i = (s i−1 , s i+1 ).We are interested in modelling the probability of a context c given a sentence s.

In general DISPLAYFORM1 One popular way to model P (c | s) for sentence-level data is suggested by the encoder-decoder framework.

The encoder E produces a fixed-length vector representation h

We first consider encoder-decoder architectures with a log-linear BOW decoder for the context.

Let h i = E(s i ) be a sentence representation of s i produced by some encoder E. The nature of E is not important for our analysis; for concreteness, the reader can consider a model such as FastSent BID19 , where E is a BOW (sum) encoder.

In the case of the log-linear BOW decoder, words are conditionally independent of the previously occurring sequence, thus Equation (3) becomes DISPLAYFORM0 .(4) where u w ∈ R d is the output word embedding for a word w and h i is the encoder output. (Biases are omitted for brevity.)The objective is to maximise the model probability of contexts c i given sentences s i across the corpus D, which corresponds to finding the Maximum Likelihood Estimator (MLE) for the trainable parameters θ: DISPLAYFORM1 By switching to the negative log-likelihood and inserting the above expression, we arrive at the following optimisation problem: DISPLAYFORM2 Noticing that DISPLAYFORM3 we see that the objective in Equation (6) forces the sentence representation h i to be similar under dot product to its context representation c i , which is simply the sum of the output embeddings of the context words.

Simultaneously, output embeddings of words that do not appear in the context of a sentence are forced to be dissimilar to its representation.

Figure 1: Unrolling a RNN decoder at inference time.

The initial hidden state for the decoder is typically the encoder output, either the recurrent cell final state for a RNN encoder, or the sum of the input word embeddings for a BOW encoder.

At the first time step, a learned <GO> token is presented as the input.

In subsequent time steps, a probability-weighted sum over word vectors is used.

The decoder is then unrolled for a fixed number of steps.

The hidden states are then concatenated to produce the unrolled decoder embedding.

In the models evaluated in Section 4, this process is performed for the RNN corresponding to the previous and next sentences.

The sentence representation is then taken as the concatenation across both RNNs.

Using dot ∼ to denote close under dot product, we find that if two sentences s i and s j have similar DISPLAYFORM4 Putting it differently, sentences that occur in related contexts are assigned representations that are similar under the dot product.

Hence we see that the encoder output equipped with the dot product constitutes an optimal representation space as defined in Section 2.

Another common choice for the context decoder is an RNN decoder DISPLAYFORM0 where h i = E(s i ) is the encoder output.

The specific structure of E is again not important for our analysis. (When E is also an RNN, this is similar to SkipThought .)The time unrolled states of decoder are converted to probability distributions over the vocabulary, conditional on the sentence s i and all the previously occurring words.

Equation (3) becomes DISPLAYFORM1 Similarly to Equation FORMULA11 , MLE for the model parameters θ can be found as DISPLAYFORM2 Using ⊕ to denote vector concatenation, we note that DISPLAYFORM3 where the sentence representation h 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10Number of unroll steps Spearman correlation coefficient Figure 2 : Performance on the STS tasks depending on the number of unrolled hidden states of the decoders, using dot product as the similarity measure.

The top row presents results for the RNN encoder and the bottom row for the BOW encoder.

Red: Raw encoder output with BOW decoder.

Green: Raw encoder output with RNN decoder.

Blue: Unrolled RNN decoder output.

Independent of the encoder architecture, unrolling even a single state of the decoder always outperforms the raw encoder output with RNN decoder, and almost always outperforms the raw encoder output with BOW decoder for some number of unrolls.the context words.

Hence we can come to the same conclusion as in the log-linear case, except we have order-sensitive representations as opposed to unordered ones.

As before, h D i is forced to be similar to the context c i under dot product, and is made dissimilar to sequences of u w that do not appear in the context.

The "transitivity" argument from Section 3.2 remains intact, except the length of decoder hidden state sequences might differ from sentence to sentence.

To avoid this problem, we can formally treat them as infinite-dimensional vectors in 2 with only a finite number of initial components occupied by the sequence and the rest set to zero.

Alternatively, we can agree on the maximum sequence length, which in practice can be determined from the training corpus.

Regardless, the above space of unrolled concatenated decoder states, equipped with dot product, is the optimal representation space for models with recurrent decoders.

Consequently, this space could be a much better candidate for unsupervised similarity tasks.

We refer to the method of accessing the decoder states at every time step as unrolling the decoder, illustrated in Figure 1 .

Note that accessing the decoder output does not require re-architecting or retraining the model, yet gives a potential performance boost on unsupervised similarity tasks almost for free.

We will demonstrate the effectiveness of this technique empirically in Section 5.

We have seen in Section 2 that the optimal representation space for a given model depends on the choice of decoder architecture.

To support this theory, we train several encoder-decoder architectures for sentences with the decoder types analysed in Section 3, and evaluate them on downstream tasks using both their optimal space and the standard space of the encoder output as the sentence representations.

Models and training.

Each model has an encoder for the current sentence, and decoders for the previous and next sentences.

As our analysis is independent of encoder type, we train and evaluate models with BOW and RNN encoders, two common choices in the literature for sentence representation learners BID19 .

The BOW encoder is the sum of word vectors BID19 .

The RNN encoder and decoders are Gated Recurrent Units (GRUs) BID12 .

using dot product as the similarity measure.

On each task, the highest performing setup for each encoder type is highlighted in bold and the highest performing setup overall is underlined.

All reported values indicate Pearson/Spearman correlation coefficients for the task.

RNN encoder: Unrolling the RNN decoders using the concatenation of the decoder hidden states (RNN-concat) dramatically improves the performance across all tasks compared to using the raw encoder output (RNN-RNN) , validating the theoretical justification presented in Section 3.3.

BOW encoder: Unrolling the RNN decoders improves performance overall, however, the improvement is less drastic than that observed for the RNN encoder, which we discuss further in the main text.

Using the notation ENC-DEC, we train RNN-RNN, RNN-BOW, BOW-BOW, and BOW-RNN models.

For each encoder-decoder combination, we test several methods of extracting sentence representations to be used in the downstream tasks.

First, we use the standard choice of the final output of the encoder as the sentence representation.

In addition, for models that have RNN decoders, we unroll between 1 and 10 decoder hidden states.

Specifically, when we unroll n decoder hidden states, we take the first n hidden states from each of the decoders and concatenate them in order to get the resulting sentence representation.

We refer to these representations as *-RNN-concat.

All models are trained on the Toronto Books Corpus , a dataset of 70 million ordered sentences from over 7,000 books.

The sentences are pre-processed such that tokens are lower case and splittable by space.

Evaluation tasks.

We use the SentEval tool BID13 to benchmark sentence embeddings on both supervised and unsupervised transfer tasks.

The supervised tasks in SentEval include paraphrase identification (MSRP) BID14 , movie review sentiment (MR) BID36 , product review sentiment (CR), BID20 ), subjectivity (SUBJ) BID35 , opinion polarity (MPQA) BID46 , and question type (TREC) BID45 BID40 .

In addition, there are two supervised tasks on the SICK dataset, entailment and relatedness (denoted SICK-E and SICK-R) BID29 .

For the supervised tasks, SentEval trains a logistic regression model with 10-fold cross-validation using the model's embeddings as features.

The unsupervised Semantic Textual Similarity (STS) tasks are STS12-16 BID11 BID2 BID1 BID5 , which are scored in the same way as SICK-R but without training a new supervised model; in other words, the embeddings are used to directly compute similarity.

We use dot product to compute similarity as indicated by our analysis; results and discussion using cosine similarity, which is canonical in the literature, are presented in Appendix B. For more details on all tasks and the evaluation strategy, see BID13 .Implementation and hyperparameters.

Our goal is to study how different decoder types affect the performance of sentence embeddings on various tasks.

To this end, we use identical hyperparameters and architecture for each model (except encoder and decoder types), allowing for a fair headto-head comparison.

Specifically, for RNN encoders and decoders we use a single layer GRU with layer normalisation BID8 .

All the weights (including word embeddings) are initialised uniformly over [−0.1, 0.1] and trained with Adam without weight decay or dropout BID24 .

Sentence length is clipped or zero-padded to 30 tokens and end-of-sentence tokens are used throughout training and evaluation.

Following , we use a vocabulary size of 20k with vocabulary expansion, 620-dimensional word embeddings, and 2400 hidden units in all RNNs.

Performance of the unrolled models on the STS tasks is presented in Figure 2 .

We note that unrolling even a single state of the decoder always improves the performance over the raw encoder output with the RNN decoder, and nearly always does so for the BOW decoder for some number of unrolled hidden states.

We observe that the performance tends to peak around 2-3 hidden states and fall off afterwards.

In principle, one might expect the peak to be around the average sentence length of the corpus.

A possible explanation of this behaviour is the "softmax drifting effect".

As there is no context available at inference time, we generate the word embedding for the next time step using the softmax output from the previous time step (see Figure 1 ).

Given that for any sentence, there is no single correct context, the probability distribution over the next words in that context will be multi-modal.

This will flatten the softmax and produce inputs for the decoder that diverge from the inputs it expects (i.e. word vectors for the vocabulary).

Further work is needed to understand this and other possible causes in detail.

Performance across unsupervised similarity tasks is presented in Table 1 and performance across supervised transfer tasks is presented in TAB2 .

For the unrolled architectures, in these tables we report on the one that performs best on the STS tasks.

When the encoder is an RNN, the supervised transfer results validate our claims in Section 3.3.

The results are less conclusive when the encoder is a BOW.

We believe this is caused by the simplicity of the BOW encoder forcing its outputs to obey the sentence-level distributional hypothesis irrespective of decoder type, resulting in multiple candidates for the optimal representation space, but this should be investigated with a detailed analysis in future work.

In addition, see Appendix A for a comparison with the original SkipThought results from the literature, and Appendix B for results using cosine similarity rather than dot product as the similarity measure in STS tasks, as is the canonical choice.

When we look at the performance on supervised transfer in TAB2 , combined with the similarity results in Table 1 , we see that the notion that models cannot be good at both supervised transfer and unsupervised similarity tasks needs refining; for example, RNN-RNN achieves strong performance on supervised transfer, while RNN-RNN-concat achieves strong performance on unsupervised similarity.

In general, our results indicate that a single model may be able to perform well on different downstream tasks, provided that the representation spaces chosen for each task are allowed to differ.

Curiously, the unusual combination of a BOW encoder and concatenation of the RNN decoders leads to the best performance on most benchmarks, even slightly exceeding that of some supervised models on some tasks BID13 .

This architecture may be worth investigating.

In this work, we introduced the concept of an optimal representation space, where semantic similarity directly corresponds to distance in that space, in order to shed light on the performance gap between simple and complex architectures on downstream tasks.

In particular, we studied the space of initial hidden states to BOW and RNN decoders (typically the outputs of some encoder) and how that space relates to the training objective of the model.

For BOW decoders, the optimal representation space is precisely the initial hidden state of the decoder equipped with dot product, whereas for RNN decoders it is not.

Noting that it is precisely these spaces that have been used for BOW and RNN decoders has led us to a simple explanation for the observed performance gap between these architectures, namely that the former has been evaluated in its optimal representation space, whereas the latter has not.

Furthermore, we showed that any neural network that outputs a probability distribution has an optimal representation space.

Since a RNN does produce a probability distribution, we analysed its objective function which motivated a procedure of unrolling the decoder.

This simple method allowed us to extract representations that are provably optimal under dot product, without needing to retrain the model.

We then validated our claims by comparing the empirical performance of different architectures across transfer tasks.

In general, we observed that unrolling even a single state of the decoder always outperforms the raw encoder output with RNN decoder, and almost always outperforms the raw encoder output with BOW decoder for some number of unrolls.

This indicates different vector embeddings can be used for different downstream tasks depending on what type of representation space is most suitable, potentially yielding high performance on a variety of tasks from a single trained model.

Although our analysis of decoder architectures was restricted to BOW and RNN, others such as convolutional BID49 and graph BID25 decoders are more appropriate for many tasks.

Similarly, although we focus on Euclidean vector spaces, hyperbolic vector spaces BID34 , complex-valued vector spaces BID44 and spinor spaces BID23 all have beneficial modelling properties.

In each case, although an optimal representation space should exist, it is not clear if the intuitive space and similarity measure is the optimal one.

However, there should at least exist a mapping from the intuitive choice of space to the optimal space using a transformation provided by the network itself, as we showed with the RNN decoder.

Evaluating in this space should further improve performance of these models.

We leave this for future work.

Ultimately, a good representation is one that makes a subsequent learning task easier.

For unsupervised similarity tasks, this essentially reduces to how well the model separates objects in the chosen representation space, and how appropriately the similarity measure compares objects in that space.

Our findings lead us to the following practical advice: i) Use a simple model architecture where the optimal representation space is clear by construction, or ii) use an arbitrarily complex model architecture and analyse the objective function to reveal, for a chosen vector representation, an appropriate similarity metric.

We hope that future work will utilise a careful understanding of what similarity means and how it is linked to the objective function, and that our analysis can be applied to help boost the performance of other complex models.

1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10Number of unroll steps Spearman correlation coefficient Figure 3 : Performance on the STS tasks depending on the number of unrolled hidden states of the decoders, using cosine similarity as the similarity measure.

The top row presents results for the RNN encoder and the bottom row for the BOW encoder.

Red: Raw encoder output with BOW decoder.

Green: Raw encoder output with RNN decoder.

Blue: Unrolled RNN decoder output.

For both RNN and BOW encoders, unrolling the decoder strictly outperforms *-RNN for almost every number of unroll steps, and perform nearly as well as or better than *-BOW.A COMPARISON WITH SKIPTHOUGHT Table 3 : Performance of the SkipThought model, with and without layer normalisation BID8 , compared against the RNN-RNN model used in our experimental setup.

On each task, the highest performing model is highlighted in bold.

For SICK-R, we report the Pearson correlation, and for STS14 we report the Pearson/Spearman correlation with human-provided scores.

For all other tasks, reported values indicate test accuracy.

† indicates results taken from BID13 .

‡ indicates our results from running SentEval on the model downloaded from BID8 's publicly available codebase (https://github.com/ryankiros/layer-norm).

We attribute the discrepancies in performance to differences in experimental setup or implementation.

However, we expect our unrolling procedure to also boost SkipThought's performance on unsupervised similarity tasks, as we show for RNN-RNN in our fair singlecodebase comparisons in the main text.

As discussed in Section 3, the objective function is maximising the dot product between the BOW decoder/unrolled RNN-decoder and the context.

However, as other researchers in the field and the STS tasks specifically use cosine similarity by default, we present the results using cosine similarity in TAB4 and the results for different numbers of unrolled hidden decoder states in Figure 3 .Although the results in TAB4 are consistent with the dot product results in Table 1 , the overall performance across STS tasks is noticeably lower when dot product is used instead of cosine similarity to determine semantic similarity.

Switching from using cosine similarity to dot product transitions from considering only angle between two vectors, to also considering their length.

Empirical studies have indicated that the length of a word vector corresponds to how sure of its context the model that produces it is.

This is related to how often the model has seen the word, and how many different contexts it appears in (for example, the word vectors for "January" and "February" have similar norms, however, the word vector for "May" is noticeably smaller) BID41 .

Using the raw encoder output (RNN-RNN) achieves the lowest performance across all tasks.

Unrolling the RNN decoders dramatically improves the performance across all tasks compared to using the raw encoder RNN output, validating the theoretical justification presented in Section 3.3.

BOW encoder: We do not observe the same uplift in performance from unrolling the RNN encoder compared to the encoder output.

This is consistent with our findings when using dot product (see Table 1 ).

A corollary is that longer sentences on average have shorter norms, since they contain more words which, in turn, have appeared in more contexts BID0 .

During training, the corpus can induce differences in norms in a way that strongly penalises sentences potentially containing multiple contexts, and consequently will disfavour these sentences as similar to other sentences under the dot product.

This induces a noise that potentially renders the dot product a less useful metric to choose for STS tasks than cosine similarity, which is unaffected by this issue.using dot product as the similarity measure.

On each task, the highest performing setup for each encoder type is highlighted in bold and the highest performing setup overall is underlined.

A practical downside of the unrolling procedure described in Section 3.3 is that concatenating hidden states of the decoder leads to very high dimensional vectors, which might be undesirable due to memory or other practical constraints.

An alternative is to instead average the hidden states, which also corresponds to a representation space in which the training objective optimises the dot product as a measure of similarity between a sentence and its context.

We refer to this model choice as *-RNN-mean.

Results on similarity and transfer tasks for BOW-RNN-mean and RNN-RNN-mean are presented in TAB7 respectively, with results for the other models from Section 5 included for completeness.

While the strong performance of RNN-RNN-mean relative to RNN-RNN is consistent with our theory, exploring why it is able to outperform RNN-concat experimentally on STS tasks is left to future work.

@highlight

By introducing the notion of an optimal representation space, we provide a theoretical argument and experimental validation that an unsupervised model for sentences can perform well on both supervised similarity and unsupervised transfer tasks.