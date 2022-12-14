Multi-label classification (MLC) is the task of assigning a set of target labels for a given sample.

Modeling the combinatorial label interactions in MLC has been a long-haul challenge.

Recurrent neural network (RNN) based encoder-decoder models have shown state-of-the-art performance for solving MLC.

However, the sequential nature of modeling label dependencies through an RNN limits its ability in parallel computation, predicting dense labels, and providing interpretable results.

In this paper, we propose Message Passing Encoder-Decoder (MPED) Networks,  aiming to provide fast, accurate, and interpretable MLC.

MPED networks model the joint prediction of labels by replacing all RNNs in the encoder-decoder architecture with message passing mechanisms and dispense with autoregressive inference entirely.

The proposed models are simple, fast, accurate, interpretable, and structure-agnostic (can be used on known or unknown structured data).

Experiments on seven real-world MLC datasets show the proposed models outperform autoregressive RNN models across five different metrics with a significant speedup during training and testing time.

Multi-label classification (MLC) is receiving increasing attention in tasks such as text categorization and image classification.

Accurate and scalable MLC methods are in urgent need for applications like assigning topics to web articles, classifying objects in an image, or identifying binding proteins on DNA.

The most common and straightforward MLC method is the binary relevance (BR) approach that considers multiple target labels independently BID0 .

However, in many MLC tasks there is a clear dependency structure among labels, which BR methods ignore.

Accordingly, probabilistic classifier chain (PCC) models were proposed to model label dependencies and formulate MLC in an autoregressive sequential prediction manner BID1 .

One notable work in the PCC category was from which implemented a classifier chain using a recurrent neural network (RNN) based sequence to sequence (Seq2Seq) architecture, Seq2Seq MLC.

This model uses an encoder RNN encoding elements of an input sequence, a decoder RNN predicting output labels one after another, and beam search that computes the probability of the next T predictions of labels and then chooses the proposal with the max combined probability.

However, the main drawback of classifier chain models is that their inherently sequential nature precludes parallelization during training and inference.

This can be detrimental when there are a large number of positive labels as the classifier chain has to sequentially predict each label, and often requires beam search to obtain the optimal set.

Aside from time-cost disadvantages, PCC methods have several other drawbacks.

First, PCC methods require a defined ordering of labels for the sequential prediction, but MLC output labels are an unordered set, and the chosen order can lead to prediction instability .

Secondly, even if the optimal ordering is known, PCC methods struggle to accurately capture long-range dependencies among labels in cases where the number of positive labels is large (i.e., dense labels).

For example, the Delicious dataset has a median of 19 positive labels per sample, so it can be difficult to correctly predict the labels at the end of the prediction chain.

Lastly, many real-world applications prefer interpretable predictors.

For instance, in the task of predicting which proteins (labels) will bind to a DNA sequence based binding site, users care about how a prediction is made and how the interactions among labels influence the predictions 1 .Message Passing Neural Networks (MPNNs) BID3 introduce a class of methods that model joint dependencies of variables using neural message passing rather than an explicit representation such as a probabilistic classifier chain.

Message passing allows for efficient inference by modelling conditional independence where the same local update procedure is applied iteratively to propagate information across variables.

MPNNs provide a flexible method for modeling multiple variables jointly which have no explicit ordering (and can be modified to incorporate an order, as explained in section 3).

To handle the drawbacks of BR and PCC methods, we propose a modified version of MPNNs for MLC by modeling interactions between labels using neural message passing.

We introduce Message Passing Encoder-Decoder (MPED) Networks aiming to provide fast, accurate, and interpretable multi-label predictions.

The key idea is to replace RNNs and to rely on neural message passing entirely to draw global dependencies between input components, between labels and input components, and between labels.

The proposed MPED networks allow for significantly more parallelization in training and testing.

The main contributions of this paper are:??? Novel approach for MLC.

To the authors' best knowledge, MPED is the first work using neural message passing for MLC.??? Accurate MLC.

Our model achieves similar, or better performance compared to the previous state of the art across five different MLC metrics.

We validate our model on seven MLC datasets which cover a wide spectrum of input data structure: sequences (English text, DNA), tabular (bag-of-words), and graph (drug molecules), as well as output label structure: unknown and graph.??? Fast.

Empirically our model achieves an average 1.7x speedup over the autoregressive seq2seq MLC at training time and an average 5x speedup over its testing time.??? Interpretable.

Although deep-learning based systems have widely been viewed as "black boxes" due to their complexity, our attention based MPED models provide a straightforward way to explain label to label, input to label, and feature to feature dependencies.

Message Passing Neural Networks (MPNNs) BID3 are a generalization of graph neural networks (GNNs) BID4 BID5 , where variables are represented as nodes on a graph G and joint dependencies are modelled using message passing rather than explicit representations, which allows for efficient inference.

MPNNs model the joint dependencies using message function M t and node update function U t for T time steps, where t is the current time step.

The hidden state v t i ??? R d of node i ??? G is updated based on messages m t i from its neighboring nodes {v t j???N (i) } defined by neighborhood N (i): DISPLAYFORM0 DISPLAYFORM1 After T updates, a readout function R is used on the updated nodes for a prediction (e.g., node classification or graph classification) on the graph G.Many possibilities exist for functions M t and U t .

For example, one can pass messages using neural attention in which nodes are able to attend over their neighborhoods differentially BID6 .

This allows for the network to learn different weights for different nodes in a neighborhood, without depending on knowing the graph structure a priori.

In this formulation, messages for node v t i are obtained by a weighted sum of all neighboring nodes {v t j???N (i) } where the weights are obtained by attention BID7 .

In our implementation, we implement neural message passing with attention.

In the rest of the paper, we use "graph attention" and "neural message passing" interchangeably.

Neural message passing with attention works as follows.

DISPLAYFORM2 where e t ij represents the importance of node j for node i. DISPLAYFORM3 Attention coefficients e t ij are then normalized across all neighboring nodes of node i using a softmax function: DISPLAYFORM4 .In our method, we use a so called attention message function M t atn to produce the message from node j to node i using the learned attention weights ?? t ij and another transformation matrix W v ??? R d??d .

Then we compute the full message m t i by linearly combining messages from all nodes j ??? N (i) with a residual connection on the current v DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 It is important to note that matrices W are shared (i.e., separately applied) across all nodes.

This can be viewed as 1-dimensional convolution with kernel and stride sizes of 1.

Weight sharing across nodes is a key aspect of MPNNs, where node dependencies are learned in an order-invariant manner.

Notations: We define the following notations, used throughout the paper.

DISPLAYFORM0 be the set of data samples with inputs x ??? X and outputs y ??? Y .

Inputs x are a (possibly ordered) set of S components {x 1 , x 2 , ..., x S }, and outputs y are a set of L labels {y 1 , y 2 , ..., y L }.

MLC involves predicting the set of binary labels {y 1 , y 2 , ..., y L }, y i ??? {0, 1} given input x.

Input features are represented as embedded vectors {c DISPLAYFORM1 , where d is the embedding size, ?? is the vocabulary size, and t represents the 'state' of the embedding after t updates.

Similarly, labels are represented as an embedded vectors {h DISPLAYFORM2 where L is the number of labels, and t represents the 'state' of the embedding after t updates.

In MLC, each output label is determined by a joint probability of other labels and the input features.

Our goal is to achieve the performance of explicit joint probability methods such as PCCs (Eqs. 22 and 23) , at the test speed of BR methods (Eq. 21).We introduce Message Passing Encoder-Decoder (MPED) networks, where we formulate MLC using an encoder-decoder architecture.

In MPED Networks, input components are represented as nodes in encoder graph G ENC using embedding vectors {c t 1:S }, and labels are represented as nodes in decoder graph G DEC using embedding vectors {h t 1:L }.

MPED networks use three MPNN modules with attention to pass messages within G ENC , from G ENC to G DEC , and within G DEC to model the joint prediction of labels.

The first module, MPNN xx , is used to update input component nodes {c t 1:S } by passing messages within the encoder (between input nodes).

The second module, MPNN xy , is used to update output label nodes {h t 1:L } by passing messages from the encoder to decoder (from input nodes {c t 1:S } to output nodes {h t 1:L }).

The third module, MPNN yy , is used to update output label nodes {h t 1:L } by passing messages within the decoder (between label nodes).

Once messages have been passed to update input and label nodes, a readout function R is then used on the label nodes to make a binary classification prediction for each label, {?? 1 ,?? 2 , ...,?? L }.

An overview of our model is shown in Fig. 1 .

COMPONENT MESSAGE PASSING For a particular input x, we first assume that the input features {x 1:S } are nodes on a graph, we call G ENC .

G ENC = (V, E), V = {x 1:S }, and E includes all undirected pairwise edges connecting node c i and node c j .

MPNN xx , parameterized by W xx , is used to pass messages between the input embeddings in order to update their states.

x i can be any component of a particular input (e.g. words in a sentence, patches of an image, nodes of a known graph, or tabular features).Nodes on G ENC are represented as embedding vectors {c : DISPLAYFORM0 DISPLAYFORM1 If there exists a known G ENC graph, message m t i for node i is computed using its neighboring nodes j ??? N (i), where the neighbors N (i) are defined by the graph.

If there is no known graph, we assume a fully connected G ENC graph, which means N (i) = {j = i}. Inputs with a sequential ordering can be modelled as a fully connected graph using positional embeddings BID9 .

Similar to the input components in the encoder, we assume that the labels {y 1:L } are nodes on a decoder graph called G DEC .

Nodes on G DEC are represented as embedding vectors {h DISPLAYFORM0 where the initial states {h 0 1:L } are obtained using label embedding matrix W y .

The decoder MPNNs update the label embeddings {h t 1:L } by passing messages from the encoder to the decoder, and then pass messages within the decoder.

MPNN xy , is used to pass messages from input embeddings {c T 1:S } to label embeddings, and then MPNN yy is used to pass messages between label embeddings.

In order to update the label nodes given a particular input x, the decoder uses MPNN xy , parameterized by W xy , to pass messages from input x to labels y.

At the equation level, this module is identical to MPNN xx except that it updates the i th label node's embedding h i using the embeddings of all the components of an input.

That is, we update each h t i by using a weighted sum of all input embeddings {c T 1:S }, in which the weights represent how important an input component is to the i th label node and the weights are learned via attention.

Messages are only passed from the encoder nodes to the decoder nodes, and not vice versa (i.e. encoder to decoder message passing is directed).

DISPLAYFORM0 DISPLAYFORM1 The key advantage of input-to-label message passing with attention is that each label node can attend to different input nodes (e.g. different words in the sentence).

At this point, the decoder can make an independent prediction for each label conditioned on x. However, in order to make more accurate predictions, we model interactions between the label nodes {h t 1:L } using message passing and update them accordingly.

To do this we use a a third message passing module, MPNN yy .

At the equation level, this layer is identical to MPNN xx except that it replaces the input embeddings with label embeddings.

In other words, label embedding h t i is updated by a weighted combination through attention of all its neighbor label nodes {h DISPLAYFORM0 To update each label embedding h : DISPLAYFORM1 DISPLAYFORM2 If there exists a known G DEC graph, message m t i for node i is computed using its neighboring nodes j ??? N (i), where the neighbors N (i) are defined by the graph.

If there is no known G DEC graph, we assume a fully connected graph, which means N (i) = {j = i}.In our implementation, the label embeddings are updated by MPNN xy and MPNN xx for T time steps to produce {h DISPLAYFORM3

The last module of the decoder predicts each label DISPLAYFORM0 is the learned output vector for label i.

The calculated vector of size L ?? 1 is then fed through an element-wise sigmoid function to produce probabilities of each label being positive: DISPLAYFORM1 In MPED networks we use binary the mean cross entropy on the individual label predictions to train the model.

p(y i |{y j =i }, c T 1:S ; W) is approximated in MPED networks by jointly representing {y 1:L } using message passing from {c T 1:S } and from the embeddings of all neighboring labels {y j???N (i) }.

Multi-head Attention In order to allow a particular node to attend to multiple other nodes (or multiple groups of nodes) at once, MPED uses multiple attention heads.

Inspired by BID8 , we use K independent attention heads for each W ?? matrix during the message computation, where each matrix column W ??,k j is of dimension d/K. The generated representations are concatenated (denoted by ) and linearly transformed by matrix W z ??? R d??d .

Multi-head attention changes message passing function M atn , but update function U mlp stays the same.

DISPLAYFORM0 DISPLAYFORM1 Graph Time Steps To learn more complex relations among nodes, we compute T time steps of embedding updates.

This is essentially a stack of T MPNN layers.

Matrices BID10 .

DISPLAYFORM2

Speed.

In MPED models, the joint probability of labels isn't explicitly estimated using the chain rule.

This enables making predictions in parallel and decreases test time drastically, especially when the number of labels is large.

We model the joint probability implicitly using the MPED decoder, at the benefit of a substantial speedup.

Time complexities of different types of models are compared in Handling dense label predictions.

Motivated by the drawbacks of autoregressive models for MLC (Section 5.6), the proposed MPED model removes the dependencies on a chosen label ordering and beam search.

This is particularly beneficial when the number of positive output labels is large (i.e. dense).

MPED networks predict the output set of labels all at once, which is made possible by the fact that inference doesn't use a probabilistic chain, but there is still a representation of label dependencies via label to label attention.

As an additional benefit, as noted by , it may be useful to maintain 'soft' predictions for each label in MLC.

This is a major drawback of the PCC models which make 'hard' predictions of the positive labels, defaulting all other labels to 0.

DISPLAYFORM0 Flexibility Many input or output types are instances where the relational structure is not made explicit, and must be inferred or assumed (e.g., text corpora, or MLC labels) BID9 .

MPED networks allow for greater flexibility of input structures (known structure such as sequence or graph, or unknown such as tabular), or output structures (e.g., known graph vs unknown structure).Interpretability.

One advantage of MPED models is that interpretability is "built in" via neural attention.

Specifically, we can visualize 3 aspects: input-to-input attention (input dependencies), input-to-label attention (input/label dependencies), and label-to-label attention (label dependencies).

Structured Output Predictions The use of graph attention in MPED models is closely connected to the literature of structured output prediction for MLC.

BID12 used conditional random fields BID13 ) to model dependencies among labels and features for MLC by learning a distribution over pairs of labels to input features.

In another research direction, recently proposed SPENs (structured prediction energy network BID14 ) and Deep Value Networks BID15 tackled MLC by optimizing different variants of structured loss formulations.

In contrast to SPEN and related methods which use an iterative refinement of the output label predictions, our method is a simpler feed forward block to make predictions in one step, yet still models dependencies through attention mechanisms on embeddings.

However, we plan to expand MPED models by adding a structured loss formulation.

Graph Neural Networks (GNNs) Passing embedding messages from node to neighbor nodes connects to a large body of literature on graph neural networks BID9 ) and embedding models for structures BID16 .

The key idea is that instead of conducting probabilistic operations (e.g., product or re-normalization), the proposed models perform nonlinear function mappings in each step to learn feature representations of structured components.

Neural message passing networks BID3 , graph attention networks BID6 and neural relation models BID17 follow similar ideas to pass the embedding from node to neighbor nodes or neighbor edges.

There have been many recent works extending the basic GNN framework to update nodes using various message passing, update, and readout functions BID18 BID19 BID20 BID21 BID3 BID17 BID22 BID23 .

We refer the readers to BID9 ) for a survey.

However, none of these have used GNNs for MLC.

In the appendix, we have added the details of training/hyperparameters (5.2), datasets (5.1), evaluation metrics (5.3), and explained how we selected baseline models from previous work (5.8).

We explain our own MPED variations in 5.7 and previous work baselines in section 5.8.In short, we compare MPED to (1) MPED Prior G DEC (by using a known label graph), (2) MPED Edgeless G DEC (by removing label-to-label message passing), (3) MPED Autoregressive (by predicting labels sequentially), (4) MP BR (binary relevance MLP output used on the mean of MPNN xx embeddings), and the baselines reported in related works.

We compare our models on seven real world datasets, which vary in the number of samples, number of labels, input type (sequential, tabular, graph), and output type (unknown, known label graph), as summarized in TAB8 .Across all datasets, MPED outperforms or achieves similar results as the baseline models.

Most importantly, we show that autoregressive models are not crucial in MLC for most metrics, and non-autoregressive models result in a significant speedup at test time.

Table 2 shows the performance of different models across the 7 datasets.

Significance across models is shown in Appendix TAB9 .

For subset accuracy (ACC), autoregressive models perform the best, but at a small margin of increase.

However, autoregressive models that predict only positive labels are targeted at maximizing subset accuracy, but they perform poorly on other metrics.

For all other metrics, autoregressive models are not essential.

One important observation is that for most datasets, MPED outperforms the autoregressive models in both miF1 (frequent labels) and more importantly, maF1 (rare labels).

Since maF1 favors models which can predict the rare labels, this shows that autoregressive models with beam search often make the wrong predictions on the rare labels (which are ordered last in the sequence during training).

MPED is a solid choice across all metrics as it comes the closest to subset accuracy as the autoregressive models, but also performs well in other metrics.

While MPED does not explicitly model label dependencies as autoregressive or structured prediction models do, it seems as though the attention weights do learn some dependencies among labels (Visualizations TAB8 ).

This is indicated by the fact that MPED, which uses label-to-label attention, mostly outperforms the ones which don't, indicating that it is learning label dependencies.

Table 2 shows 3 time step models results, but a comparison of time steps is shown in Figure 2 .Speed Table 2 shows the per epoch train and test times for each model.

All models are trained and tested on the same GPU using a batch size of 32.

At test time, since the autoregressive model cannot be parallelized, MPED and other non-autoregressive models are significantly faster.

During training, the autoregressive model can be parallelized because the true labels are fed as the previous label.

Since the autoregressive models only predict the ?? positive labels, they can be faster at training time, whereas the MPED model is predicting the probability for all labels.

MPED results in a mean of 1.7x and 5.0x training and testing speedups, respectively, over Seq2Seq autoregressive models.

Interpretability We present visualizations of the input-to-label and label-to-label attention weights (averaged across the 4 attention heads) in the Appendix.

In the visualizations, we show the positive Table 2 : Results.

Across all 7 datasets, MPED produces similar or better average metric scores to baseline models.

MPED results in a mean of 1.7x and 5.0x training and testing speedups, respectively, over the previous state-of-the-art probabilistic MLC method, RNN Seq2Seq.

Speedups over RNN Seq2Seq model are shown in minutes per epoch in parentheses for the MPED model.

Bold numbers show the best performing method(s).

labels only, and the darker lines show higher attention weights to the corresponding label or word.

The attention weights clearly learn certain relationships between input-label pairs as well as the label-label pairs, which is all done in an unsupervised manner.

In future work, we plan to add a structured prediction loss function which will likely improve the attention mechanisms and the model's ability to estimate the joint probability.

In this work we present Message Passing Encoder-Decoder (MPED) Networks which achieve a significant speedup at close to the same performance as autoregressive models for MLC.

We open a new avenue of using neural message passing to model label dependencies in MLC tasks.

In addition, we show that our method is able to handle various input data types (sequence, tabular, graph), as well various output label structures (known vs unknown).

One of our future extensions is to adapt the current model to predict more dynamic outputs.

BID1 BID24

We test our method against baseline methods on seven different multi-label sequence classification datasets.

The datasets are summarized in TAB8 .

We use Reuters-21578, Bibtex (Tsoumakas et al., 2009) BID30 , which is side effects of drug molecules.

As shown in the table, each dataset has a varying number of samples, number of labels, positive labels per sample, and samples per label.

For BibTex and Delicious, we use 10% of the provided training set for validation.

For the TFBS dataset, we use 1 layer of convolution at the first layer to extract "words" from the DNA characters (A,C,G,T), as commonly done in deep learning models for DNA.For datasets which have sequential ordering of the input components (Reuters, RCV1), we add a positional encoding to the word embedding as used in BID8 (sine and cosine functions of different frequencies) to encode the location of each word in the sentence.

For datasets with no ordering or graph stucture (Bibtex, Delicious, Bookmarks, which use bag-of-word input representations) we do not use positional encodings.

For inputs with an explicit graph representation (SIDER), we use the known graph structer.

We validate our model on seven MLC datasets.

These datasets cover a wide spectrum of input data types, including: raw English text (sequential form), bag-of-words (tabular form), and drug molecules (graph form).For all 6 datasets except SIDER, we use the same MPED model with T =3 time steps, d = 512, and K=4 attention heads.

Since SIDER is significantly smaller, we use T =1 time step, d = 64, and K=4 attention heads.

We trained our models on an NVIDIA TITAN X Pascal with a batch size of 32.

We used Adam BID31 with betas= (0.9, 0.999), eps=1e-08, and a learning rate of 0.0002 for each dataset.

We used dropout of p = 0.2 for all models.

The MPED models also use layer normalization BID32 around each of the attention and feedforward layers.

The non-autoregressive models are trained with binary cross-entropy on each label and the autoregressive models are trained with cross entropy across all possible labels at each position.

Multi-label classification methods can be evaluated with many different metrics which each evaluate different strengths or weaknesses.

We use the same 5 evaluation metrics from .All of our autoregressive models predict only the positive labels before outputting a stop signal.

This is a special case of PCC models (explained as PCC+ in section 5.4), which have been shown to outperform the binary prediction of each label in terms of performance and speed.

These models use beam search at inference time with a beam size of 5.

For the non-autoregressive models, to convert the labels to {0, 1} we chose the best threshold on the validation set from the same set of thresholds used in BID14 .Example-based measures are defined by comparing the target vector y to the prediction vector??.

Subset Accuracy (ACC) requires an exact match of the predicted labels and the true labels: ACC(y,??) = I[y =??].Hamming Accuracy (HA) evaluates how many labels are correctly predicted in??: DISPLAYFORM0 Example-based F1 (ebF1) measures the ratio of correctly predicted labels to the sum of the total true and predicted labels: DISPLAYFORM1 Label-based measures treat each label yj as a separate two-class prediction problem, and compute the number of true positives (tpj), false positives (f pj), and false negatives (f nj) for a label.

Macro-averaged F1 (maF1) measures the label-based F1 averaged over all labels: DISPLAYFORM2 measures the label-based F1 averaged over each sample: DISPLAYFORM3 .

High maF1 scores usually indicate high performance on less frequent labels.

High miF1 scores usually indicate high performance on more frequent labels.

MLC has a rich history in text (McCallum; BID34 , images BID0 BID35 , bioinformatics BID0 BID35 , and many other domains.

MLC methods can roughly be broken into several groups, which are explained as follows.

Label powerset models (LP) BID36 BID37 , classify each input into one label combination from the set of all possible combinations Y = {{1}, {2}, ..., {1, 2, ..., L}}. LP explicitly models the joint distribution by predicting the one subset of all positive labels.

Since the label set Y grows exponentially in the number of total labels (2 L ), classifying each possible label set is intractable for a modest L. In addition, even in small L tasks, LP suffers from the "subset scarcity problem" where only a small amount of the label subsets are seen during training, leading to bad generalization.

Binary relevance (BR) methods predict each label separately as a logistic regression classfier for each label BID38 BID39 .

The na??ve approach to BR prediction is to predict all labels independently of one another, assuming no dependencies among labels.

That is, BR uses the following conditional probability parameterized by learned weights W : DISPLAYFORM0 Probabilistic classifier chain (PCC) methods BID40 BID1 ) are autoregressive models that estimate the true joint probability of output labels given the input by using the chain rule, predicting one label at a time: DISPLAYFORM1 Two issues with PCC models are that inference is very slow if L is large, and the errors propagate as L increases BID41 .

To mitigate the problems with both LP and PCC methods, one solution is to only predict the true labels in the LP subset.

In other words, only predicting the positive labels (total of ?? for a particular sample) and ignoring all other labels, which we call PCC+.

Similar to PCC, the joint probability of PCC+ can be computed as product of conditional probabilities, but unlike PCC, only ?? < L terms are predicted as positive: DISPLAYFORM2 This can be beneficial when the number of possible labels L is large, reducing the total number of prediction steps.

However, in both PCC and PCC+, inference is done using beam search, which is a costly dynamic programming step to find the optimal prediction.

MPED methods approximate the following factored formulation, where N (Yi) denotes the neighboring nodes of Yi.

DISPLAYFORM3

In machine translation (MT), sequence-to-sequence (Seq2Seq) models have proven to be the superior method, where an encoder RNN reads the source language sentence into an encoder hidden state, and a decoder RNN Figure 2 : Average Across Metrics for T =1, T =2, and T =3 GDEC time steps.

In these experiments, the encoder GENC is processed with a fixed T =3 time steps, and the decoder time steps are varied.

We do not compare time steps for the SIDER dataset since it is too small, and we only evaluate using T =1.translates the hidden state into a target sentence, predicting each word autoregressively BID42 .

BID7 improved this model by introducing "neural attention" which allows the decoder RNN to "attend" to every encoder word at each step of the autoregressive translation.

Recently, showed that, across several metrics, state-of-the-art MLC results could be achieved by using a recurrent neural network (RNN) based encoder-to-decoder framework for Equation 23 (PCC+).

They use a Seq2Seq RNN model (Seq2Seq Autoregressive) which uses one RNN to encode x, and a second RNN to predict each positive label sequentially, until it predicts a 'stop' signal.

This type of model seeks to maximize the 'subset accuracy', or correctly predict every label as its exact 0/1 value.

BID8 eliminated the need for the recurrent network in MT by introducing the Transformer.

Instead of using an RNN to model dependencies, the Transformer explicitly models pairwise dependencies among all of the features by using attention BID7 BID43 between signals.

This speeds up training time because RNNs can't be fully parallelized but, the transformer still uses an autoregressive decoder.

Autoregressive models have been proven effective for machine translation and MLC BID42 BID7 .

However, predictions must be made sequentially, eliminating parallelization.

Also, beam search is typically used at test time to find optimal predictions.

But beam search is limited by the time cost of large beams sizes, making it difficult to optimally predict many output labels BID44 .In addition to speed constraints, beam search for autoregressive inference introduces a second drawback: initial wrong predictions will propagate when using a modest beam size (e.g. most models use a beam size of 5).

This can lead to significant decreases in performance when the number of positive labels is large.

For example, the Delicious dataset has a median of 19 positive labels per sample, and it can be very difficult to correctly predict the labels at the end of the prediction chain.

Autoregressive models are well suited for machine translation because these models mimic the sequential decoding process of real translation.

However, for MLC, the output labels have no intrinsic ordering.

While the joint probability of the output labels is independent of the label ordering via autoregressive based inference, the chosen ordering can make a difference in practice BID45 .

Some ordering of labels must be used during training, and this chosen ordering can lead to unstable predictions at test time.

Our non-autoregressive version connects to BID46 who removed the autoregressive decoder in MT with the Non-Autoregressive Transformer.

In this model, the encoder makes a proxy prediction, called "fertilities", which are used by the decoder to predict all translated words at once.

The difference between their model and ours is that we have a constant label at each position, so we don't need to marginalize over all possible labels at each position.

In the full MPED model, we use 3 encoder time steps and 3 decoder time steps with node to node attention in both the encoder and decoder graphs, and K=4 attention heads.

@highlight

We propose Message Passing Encoder-Decode networks for a fast and accurate way of modelling label dependencies for multi-label classification.