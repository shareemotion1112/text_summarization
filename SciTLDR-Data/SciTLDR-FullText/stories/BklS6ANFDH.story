Semantic dependency parsing, which aims to find rich bi-lexical relationships, allows words to have multiple dependency heads, resulting in graph-structured representations.

We propose an approach to semi-supervised learning of semantic dependency parsers based on the CRF autoencoder framework.

Our encoder is a discriminative neural semantic dependency parser that predicts the latent parse graph of the input sentence.

Our decoder is a generative neural model that reconstructs the input sentence conditioned on the latent parse graph.

Our model is arc-factored and therefore parsing and learning are both tractable.

Experiments show our model achieves significant and consistent improvement over the supervised baseline.

Semantic dependency parsing (SDP) is a task aiming at discovering sentence-internal linguistic information.

The focus of SDP is the identification of predicate-argument relationships for all content words inside a sentence (Oepen et al., 2014; .

Compared with syntactic dependencies, semantic dependencies are more general, allowing a word to be either unattached or the argument of multiple predicates.

The set of semantic dependencies within a sentence form a directed acyclic graph (DAG), distinguishing SDP from syntactic dependency parsing tasks, where dependencies are usually tree-structured.

Extraction of such high-level structured semantic information potentially benefits downstream NLP tasks (Reddy et al., 2017; Schuster et al., 2017) .

Several supervised SDP models are proposed in the recent years by modifying syntactic dependency parsers.

Their parsing mechanisms are either transition-based (Ribeyre et al., 2014; Kanerva et al., 2015; Wang et al., 2018) or graph-based (Martins & Almeida, 2014; Dozat & Manning, 2018; Wang et al., 2019) .

One limitation of supervised SDP is that labeled SDP data resources are limited in scale and diversity.

Due to the rich relationships in SDP, the annotation of semantic dependency graphs is expensive and difficult, calling for professional linguists to design rules and highly skilled annotators to annotate sentences.

This limitation becomes more severe with the rise of deep learning, because neural approaches are more data-hungry and susceptible to over-fitting when lacking training data.

To alleviate this limitation, we investigate semi-supervised SDP capable of learning from both labeled and unlabeled data.

While a lot of work has been done on supervised SDP, the research of unsupervised and semisupervised SDP is still lacking.

Since parsing results of semantic dependencies are DAGs without the tree-shape restriction, most existing successful unsupervised (Klein & Manning, 2004; I. Spitkovsky et al., 2010; Jiang et al., 2016; Cai et al., 2017) and semi-supervised (Koo et al., 2008; Druck et al., 2009; Suzuki et al., 2009; Corro & Titov, 2019) learning models for syntactic dependency parsing cannot be applied to SDP directly.

There also exist several unsupervised (Poon & Domingos, 2009; Titov & Klementiev, 2011) and semi-supervised (Das & Smith, 2011; Kočiskỳ et al., 2016; Yin et al., 2018) methods for semantic parsing, but these models are designed for semantic representations different from dependency graphs, making their adaptation to SDP difficult.

In this work, we propose an end-to-end neural semi-supervised model leveraging both labeled and unlabeled data to learn a dependency graph parser.

Our model employs the framework of Condi-tional Random Field Autoencoder (Ammar et al., 2014) , modeling the conditional reconstruction probability given the input sentence with its dependency graph as the latent variable.

Our encoder is the supervised model of Dozat & Manning (2018) , formulating an SDP task as labeling each arc in a directed graph with a simple neural network.

Analogous to a CRF model (Sutton et al., 2012) , our encoder is capable of computing the probability of a dependency graph conditioned on the input sentence.

The decoder is a generative model based on recurrent neural network language model (Mikolov et al., 2010) , which formulates the probability of generating the input sentence, but we take into account the information given by the dependency parse graphs when generating the input.

Our model is arc-factored, i.e., the encoding, decoding and reconstructing probabilities can all be factorized into the product of arc-specific quantities, making both learning and parsing tractable.

A unified learning objective is defined that takes advantage of both labeled and unlabeled data.

Compared with previous semi-supervised approaches based on Variational Autoencoder (Kingma & Welling, 2013) , our learning process does not involve sampling, promising better stability.

We evaluate our model on SemEval 2015 Task 18 Dataset (English) (Oepen et al., 2015) and find that our model consistently outperforms the state-of-the-art supervised baseline.

We also conduct detailed analysis showing the benefits of different amounts of unlabeled data.

Our model is based on the CRF autoencoder framework (Ammar et al., 2014) which provides a unified fashion for structured predictors to leverage both labeled and unlabeled data.

A CRF autoencoder aims to produce a reconstruction of the inputX from the original input X with an intermediate latent structure Y. It is trained to maximize the conditional reconstruction probability P (X = X|X) with the latent variable Y marginalized.

Ideally, successful reconstruction implies that the latent structure captures important information of the input.

We adopt the following notations when describing our model.

We represent a vector in lowercase bold, e.g., s, and use a superscript for indexing, e.g., s

i for the i-th vector.

We represent a scalar in lowercase italics, e.g., s, and use a subscript for indexing, e.g., s i for the i-th element of vector s. An uppercase italic letter such as Y denotes a matrix.

A lower case letter with a subscript pair such as y i,j refers to the element of matrix Y at row i and column j. An uppercase bold letter, e.g., U, stands for a tensor.

We maintain this convention when indexing, e.g., y i is the i-th row of matrix Y .

In our model, the input is a natural language sentence consisting of a sequence of words.

A sentence with m words is represented by s = (s 0 , s 1 , s 2 , . . .

, s m ), where s 0 is a special token TOP.

The latent variable produced by our encoder is a dependency parse graph of the input sentence, represented as a matrix of booleans Y ∈ {0, 1} (m+1)×(m+1) , where y i,j = 1 indicates that there exists an dependency arc pointing from word s i to word s j .

The reconstructed output generated by our decoder is a word sequenceŝ = (ŝ 1 ,ŝ 2 , . . .

,ŝ m ).

Our encoder with parameters Θ computes P Θ (Y |s), the probability of generating a dependency parse graph Y given a sentence s. Our decoder with parameters Λ computes P Λ (ŝ|Y ), the probability of reconstructing sentenceŝ conditioned on the parse graph Y .

The encoder and decoder in combination specify the following conditional distribution.

To compute the conditional probability P (ŝ|s), we sum out the latent variable Y .

where Y is the set of all possible dependency parse graphs of s. During training, we setŝ = s and maximize the conditional reconstruction probability P (ŝ|s).

Note that throughout our model, we only consider dependency arc predictions (i.e., whether an arc exists between each word pair).

Arc-labels will be learned separately as described in Section 3.

We leave the incorporation of arc-label prediction in our model for future work.

Our encoder can be any arc-factored discriminative SDP model.

Here we adopt the model of Dozat & Manning (2018) , which formulates the semantic dependency parsing task as independently labeling each arc in a directed complete graph.

To predict whether or not a directed arc (s i , s j ) exists, the model computes contextualized representations of s i and s j and feeds them into a binary classifier.

The architecture of our encoder is shown in Fig.1a .

Word, part-of-speech tag (for short, POS tag), and lemma embeddings 1 of each word in the input sentence are concatenated and fed into a multilayer bi-directional LSTM to get a contextualized representation of the word.

where e are notations for the word, POS tag and lemma embedding respectively, concatenated (⊕) to form an embedding x i for word s i .

Stacking x i for i = 0, 1, . . .

, m forms matrix X.

The contextualized word representation is then fed into two single-layer feedforward neural networks (FNN) with different parameters to produce two vectors: one for the representation of the word as a dependency head and the other for the representation of the word as a dependent.

They are denoted as h

Finally, a biaffine function is applied to every arc between word pairs (s i , s j ) to obtain an arcexistence score ψ i,j .

where W is a square matrix of size

, and b is a scalar.

The likelihood of every arc's presence given a sentence, P (y i,j = 1|s), can be computed by applying a sigmoid function on score ψ i,j .

The arc-absence probability P (y i,j = 0|s) is evidently 1 − P (y i,j = 1|s).

To conclude, the probability of producing a dependency parse graph Y from the encoder given an input sentence s can be computed as below.

1 Unless stated otherwise, our model makes use of lemma embeddings by default.

Our generative decoder is based on recurrent neural network language models (Mikolov et al., 2010 ), but we take dependency relationships into account during reconstruction.

Our inspiration sources from the decoder with a Graph Convolutional Network (GCN) used by Corro & Titov (2019) to incorporate tree-structured syntactic dependencies when generating sentences, but our decoder differs significantly from theirs in that ours handles parse graphs and is arc-factored.

As mentioned above, semantic dependency parsing allows a word to have multiple dependency heads.

If we generate a word conditioned on multiple heads, then it becomes difficult to make the decoder arc-factored and hence inference and learning becomes less tractable.

Instead, we propose to generate a word for multiple times, each time conditioned on a different head.

Specifically, we split dependency graph Y of a sentence s = (s 0 , s 1 , . . .

, s m ) with m words and a TOP token into m + 1 parts:

Each y i is the i-th row of Y , representing a sub-graph where arcs are rooted at the i-th word of the sentence s.

Mathematically, we have y i = {y i,j |j ∈ (1, 2, ..., m)}.

We then generate m+1 sentences (ŝ 0 ,ŝ 1 ,ŝ 2 , . . .

,ŝ m ) using m+1 neural generators.

The generation of sentenceŝ k is guided by the k-th sub-graph y k .

Each generator is a left-to-right LSTM language model and computes P Λ (ŝ k i |ŝ k 1:i−1 , y k,i ), the probability of generating each word conditioned on its preceding words and whether y k contains a dependency arc to the word.

We share parameters among all the m + 1 generators.

Fig.1b shows an example for computing the generative probability ofŝ k by the k-th generator (k ∈ {0, 1, . . .

, m}) that incorporates the information of the k-th sub-graph y k .

Recall that y k contains only dependencies rooted at s k .

Below we describe how to compute the generative probability of each wordŝ k i with and without the dependency arc (s k , s i ) respectively.

Generative probability with a dependency Suppose there is a dependency arc from s k to s i , we need to compute the generative probability P Λ (ŝ k i |ŝ k 1:i−1 , y k,i = 1).

The LSTM in the k-th generator takes the embedding of the previous word s i−1 computed through Eq.1 as its input and outputs the hidden state g i−1 , which is fed into an FNN to produce a representation m (pre) i−1 .

Meanwhile, the embedding of the k-th word (also computed through Eq.1) is fed into another FNN to get its representation m (head) k as a dependency head.

Here, U is a tensor of size i−1 .

To conserve parameters, the tensor U is diagonal (i.e., u i,k,j = 0 wherever i = j).

A softmax function can then be applied to φ k i , from which we pick the generative probability ofŝ k i .

Generative probability without a dependency Suppose there is no dependency arc from s k to s i .

In this case, reconstruction ofŝ

The generative probability P Λ (ŝ k i |ŝ k 1:i−1 , y k,i = 0) can then be computed by applying a softmax function onφ k i and selecting the corresponding probability ofŝ k i .

Since we simply reconstruct word s i without considering the dependency arc information, this probability is exactly the same in the m + 1 generators and only needs to be computed once.

To conclude the overall design of our decoder, it is worth noting that in m + 1 generation processes, parameters among all LSTMs are shared, as well as those among all FNNs 2 and FCs.

Still, embeddings in Eq.1 are shared among both encoder and decoder.

With P (ŝ k i |ŝ k 1:i−1 , y k,i ) computed for i = 1, . . .

, m, k = 0, 1, . . .

, m, the probability of generatinĝ s 0 ,ŝ 1 ,ŝ 2 , . . . ,ŝ m from dependency graph Y can be computed through:

In our model, we are only interested in the case where all the m + 1 sentences are the same.

In addition, to balance the influence of the encoder and the decoder, we take the geometric mean of the m + 1 probabilities.

The final decoding probability is defined as follows.

Note that this is not a properly normalized probability distribution, but in practice we find it sufficient for semi-supervised SDP.

Given parameters {Θ, Λ} of our encoder and decoder, we can parse a sentence s by finding a Y ∈ Y(s) which maximizes probability P (ŝ = s, Y |s), where Y(s) is the set of all parse graphs of sentence s.

= arg max

Since the probability is arc-factored, we can determine the existence of each dependency arc independently by picking the value of y i,j that maximizes the corresponding term.

The time complexity of our parsing algorithm is O(m 2 ) for a sentence with m words.

Since we want to train our model in a semi-supervised manner, we design loss functions for labeled and unlabeled data respectively.

For each training sentence s, the overall loss function is defined as a combination of supervised loss L l and unsupervised loss L u .

where an indicator ι(s) ∈ {0, 1} specifies whether training sentence s is labeled or not and a tunable constant ρ balances the two losses.

Supervised Loss For any labeled sentence (s, Y * ), where s stands for a sentence and Y * stands for a gold parse graph, we can compute the discriminative loss.

2 FNN (dec−pre) and FNN (dec−head) never share parameters between each other, since their usages are different.

Following the derivation of Eq.6, we have:

Gold parses also provide a label for each dependency.

We follow Dozat & Manning (2018) and model dependency labels with a purely supervised module on top of the BiLSTM layer of the encoder.

Its parameters are learned by optimizing a cross-entropy loss function.

Unsupervised Loss For any unlabeled sentence s, we maximize the conditional reconstruction probability P (ŝ = s|s).

The unsupervised loss is:

Derivations of Eq.10 are provided in Appendix A. Given a dataset containing both labeled and unlabeled sentences, our model can be trained end-to-end by optimizing the loss function Eq.7 over the combined dataset using any gradient based method.

Dataset We examine the performance of our model on the English corpus of the SDP 2014 & 2015: Broad Coverage Semantic Dependency Parsing dataset (Oepen et al., 2015) .

The corpus is composed of three distinct and parallel semantic dependency annotations (DM, PAS, PSD) of Sections 00-21 of the WSJ Corpus, as well as a balanced sample of twenty files from the Brown Corpus.

More information of this dataset is shown in Table 3 in Appendix B.

We evaluate the performance of models through two metrics: Unlabeled F1 score (UF1) and Labeled F1 score (LF1).

UF1 measures the accuracy of the binary classification of arc existence, while LF1 measures the correctness of each arc-label as well.

Network Configuration For our encoder, we adopt the hyper-parameters of Dozat & Manning (2018) .

Following Dozat & Manning (2018) , words or lemmas whose occurrences are less than 7 times within the training set are treated as UKN.

For our decoder, we set the number of layer(s) of uni-directional LSTM to 1, whose recurrent hidden size is 600.

For FNN (dec−head) and FNN (dec−pre) , the output sizes are both 400, activated by a tanh(·) function.

Learning Our loss function (Eq.7) is optimized by the Adam+AMSGrad optimizer (Reddi et al., 2018) , with hyper-parameters β 1 , β 2 kept the same as those of Dozat & Manning (2018) .

The interpolation constant ρ is tuned with the size of unlabeled data.

A detailed table of hyper-parameter values is provided in Appendix C. The training time for one batch with our autoencoder is 2-3 times of that of Dozat & Manning (2018) because of the extra decoder.

In our first experiment (with the DM annotations only), we fix the amount of labeled data and continuously incorporate more unlabeled data into the training set.

Specifically, we randomly sample 10% of the whole dataset as labeled data.

Unlabeled data are then sampled from the remaining part (with their gold parses removed), with a proportion increasing from 0% to 90% of the complete dataset.

For unlabeled data, we find that long sentences do not help in improving F1 scores and therefore in this and all the subsequent experiments we remove unlabeled sentences longer than 20 to reduce the running time and memory usage.

Experimental results are visualized in Fig.2 .

First, we observe that in the purely supervised setting (i.e., +0% unlabeled data), our model already outperforms the baseline (Dozat & Manning, 2018) (a) UF1 for in-domain tests.

Table 1 : Experimental results with varying proportions of labeled and unlabeled data.

D&M stands for the supervised model of Dozat & Manning (2018) trained on labeled data only.

Ours-Sup stands for our model trained on labeled data only.

Ours-Semi stands for our model trained on both labeled and unlabeled data.

and the advantage of our model is larger on the out-of-domain test set than on the in-domain test set.

Since our encoder is exactly the baseline model, this shows the benefit of adding the decoder for joint learning and parsing even in the supervised setting.

Second, with an increasing size of unlabeled dataset from 0% to 30%, we see a clear increase in performance of our model, suggesting the benefit of semi-supervised learning with our model.

However, we observe little improvement when the size of unlabeled data exceeds 40% (not shown in the figure) , indicating a possible upper bound of the effectiveness of unlabeled data.

In our second experiment (again with the DM annotations), we use the full training set and vary the proportion of labeled and unlabeled data.

Experimental results are shown in Table 1 .

Our semi-supervised model shows the largest advantage over the supervised models with the 0.1:9.9 proportion (which contains only 339 labeled sentences).

With the increased proportion of labeled data, the performance of all the models goes up, but the advantage of our semi-supervised model vanishes.

This demonstrates the diminishing effectiveness of unlabeled data when adding more labeled data.

Another worth-noting observation is that the superiority of our semi-supervised model is much stronger on the out-of-domain tests and does not vanish even with the 5:5 proportion.

This suggests good generalizability of our semi-supervised model.

In the previous two experiments, we evaluate our model on the DM representation.

Here we evaluate our model on all the three representations: DM, PAS and PSD.

We slightly tune the hyper-parameters based on the optimal values from the previous experiments of the DM representation.

We use 10% of the sentences as labeled data and the rest 90% of the sentences as unlabeled data.

For the completeness of our experiment, we follow Dozat & Manning (2018) and examine four different word representations: basic (i.e., using only word and POS tag embeddings), +Lemma (i.e., using word, Table 2 : Experimental results on all the three representations.

D&M stands for the supervised model of Dozat & Manning (2018) .

Ours-Sup stands for our model trained on labeled data only.

Ours-Semi stands for our model trained on both labeled and unlabeled data.

POS tag and lemma embeddings), +Char (i.e., using word, POS tag and character embeddings) and +Lemma+Char (i.e. using word, POS tag, lemma and character embeddings).

Table 2 shows the experimental results of +Lemma, the default word representation.

The results of the other word representations show very similar trends (see Appendix D).

We observe significant improvement of our semi-supervised model over the two supervised baselines on both DM and PSD representations.

However, it is surprising to find that on the PAS representation, our semi-supervised model exhibits little advantage over its supervised counterpart.

One possible explanation, as Dozat & Manning (2018) also noted, is that PAS is the easiest of the three representations (as can be seen by comparing the scores of the three representations in Table 2 ) and our supervised model may already reach the performance ceiling.

Work on unsupervised or semi-supervised dependency parsing, to the best of our knowledge, is dominated by tree-structured parsing (Koo et al., 2008; Druck et al., 2009; Suzuki et al., 2009) .

Recently, Corro & Titov (2019) introduced an approximate inference method with a Variational Autoencoder (Kingma et al., 2014) for semi-supervised syntactic dependency parsing.

Our decoder is inspired by their work, but differs from theirs in that our decoder handles parse graphs and is arc-factored.

Cai et al. (2017) used the framework of CRF Autoencoder (Ammar et al., 2014) to perform unsupervised syntactic dependency parsing.

The same framework has been used by Zhang et al. (2017) for semi-supervised sequence labeling.

Our work also adopts the CRF Autoencoder framework, but with both the encoder and the decoder redesigned for semantic dependency parsing.

Existing unsupervised and semi-supervised approaches to semantic parsing focused on semantic representations different from dependency graphs, e.g., general-purpose logic forms (Sondheimer & Nebel, 1986) and formal meaning representations (Bordes et al., 2012) .

Therefore, these approaches cannot be applied to SDP directly.

Poon & Domingos (2009) presented the first unsupervised semantic parser to transform dependency trees into quasi-logical forms with Markov logic.

Following this work, Titov & Klementiev (2011) proposed a non-parametric Bayesian model for unsupervised semantic parsing using hierarchical PitmanYor process (Teh, 2006) .

Das & Smith (2011) described a semi-supervised approach to frame-semantic parsing.

Kočiskỳ et al. (2016) proposed a semisupervised semantic parsing approach making use of unpaired logical forms.

Recently, Yin et al. (2018) proposed a variational autoencoding model for semi-supervised semantic parsing of treestructured semantic representations.

In this work, we proposed a semi-supervised learning model for semantic dependency parsing using CRF Autoencoders.

Our model is composed of a discriminative neural encoder producing a dependency graph conditioned on an input sentence, and a generative neural decoder for input reconstruction based on the dependency graph.

The model works in an arc-factored fashion, promising end-to-end learning and efficient parsing.

We evaluated our model under both full-supervision settings and semi-supervision settings.

Our model outperforms the baseline on multiple target representations.

By adding unlabeled data, our model exhibits further performance improvements.

Derivation of the marginalized probability over all possible dependency graphs of a sentence with m words for Eq.10 is shown below.

We extract WSJ Section 20 (1,692 sentences) from the train set for development purpose.

id stands for in-domain testing, while ood stands for out-of-domain testing.

Here we provide a summary of hyper-parameters in our experiments, as shown in Table 4 .

Complete experimental results on DM, PAS and PSD under the setting of Section 4.4 (i.e., basic, +Char, +Lemma and +Lemma+Char) are shown in Table 6 .

To test the stability of our model, we repeat the experiment of Section 4.4 on the DM annotation for three times, each time with a differently sampled dateset.

Table 5 : Experimental results on three randomly sampled datasets.

D&M stands for the supervised model of Dozat & Manning (2018) .

Ours-Sup stands for our model trained on labeled data only.

Ours-Semi stands for our model trained on both labeled and unlabeled data.

@highlight

We propose an approach to semi-supervised learning of semantic dependency parsers based on the CRF autoencoder framework.

@highlight

This paper focuses on semi-supervised semantic dependency parsing using the CRF-autoencoder to train the model in a semi-supervised style, indicating effectiveness on low resource labeled data tasks.