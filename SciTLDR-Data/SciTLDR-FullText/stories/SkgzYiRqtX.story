Recently, progress has been made towards improving relational reasoning in machine learning field.

Among existing models, graph neural networks (GNNs) is one of the most effective approaches for multi-hop relational reasoning.

In fact, multi-hop relational reasoning is indispensable in many natural language processing tasks such as relation extraction.

In this paper, we propose to generate the parameters of graph neural networks (GP-GNNs) according to natural language sentences, which enables GNNs to process relational reasoning on unstructured text inputs.

We verify GP-GNNs in relation extraction from text.

Experimental results on a human-annotated dataset and two distantly supervised datasets show that our model achieves significant improvements compared to the baselines.

We also perform a qualitative analysis to demonstrate that our model could discover more accurate relations by multi-hop relational reasoning.

Recent years, graph neural networks (GNNs) have been applied to various fields of machine learning, including node classification BID10 , relation classification BID22 , molecular property prediction BID6 , few-shot learning BID5 , and achieve promising results on these tasks.

These works have demonstrated GNNs' strong power to process relational reasoning on graphs.

Relational reasoning aims to abstractly reason about entities/objects and their relations, which is an important part of human intelligence.

Besides graphs, relational reasoning is also of great importance in many natural language processing tasks such as question answering, relation extraction, summarization, etc.

Consider the example shown in Fig. 1 , existing relation extraction models could easily extract the facts that Luc Besson directed a film Léon:

The Professional and that the film is in English, but fail to infer the relationship between Luc Besson and English without multi-hop relational reasoning.

By considering the reasoning patterns, one can discover that Luc Besson could speak English following a reasoning logic that Luc Besson directed Léon:

The Professional and this film is in English indicates Luc Besson could speak English.

However, most existing GNNs can only process multi-hop relational reasoning on pre-defined graphs and cannot be directly applied in natural language relational reasoning.

Enabling multi-hop relational reasoning in natural languages remains an open problem.

To address this issue, in this paper, we propose graph neural networks with generated parameters (GP-GNNs), to adapt graph neural networks to solve the natural language relational reasoning task.

GP-GNNs first constructs a fully-connected graph with the entities in the sequence of text.

After that, it employs three modules to process relational reasoning: (1) an encoding module which enables edges to encode rich information from natural languages, (2) a propagation module which propagates relational information among various nodes, and (3) a classification module which makes predictions with node representations.

As compared to traditional GNNs, GP-GNNs could learn edges' parameters from natural languages, extending it from performing inferring on only non-relational graphs or graphs with a limited number of edge types to unstructured inputs such as texts.

In the experiments, we apply GP-GNNs to a classic natural language relational reasoning task: relation extraction from text.

We carry out experiments on Wikipedia corpus aligned with Wikidata knowledge base BID25 Figure 1 : An example of relation extraction from plain text.

Given a sentence with several entities marked, we model the interaction between these entities by generating the weights of graph neural networks.

Modeling the relationship between "Léon" and "English" as well as "Luc Besson" helps discover the relationship between "Luc Besson" and "English".model outperforms other state-of-the-art models on relation extraction task by considering multihop relational reasoning.

We also perform a qualitative analysis which shows that our model could discover more relations by reasoning more robustly as compared to baseline models.

Our main contributions are in two-fold:(1) We extend a novel graph neural network model with generated parameters, to enable relational message-passing with rich text information, which could be applied to process relational reasoning on unstructured inputs such as natural languages.(2) We verify our GP-GNNs in the task of relation extraction from text, which demonstrates its ability on multi-hop relational reasoning as compared to those models which extract relationships separately.

Moreover, we also present three datasets, which could help future researchers compare their models in different settings.

GNNs were first proposed in BID21 and are trained via the Almeida-Pineda algorithm BID1 .

Later the authors in BID12 replace the Almeida-Pineda algorithm with the more generic backpropagation and demonstrate its effectiveness empirically.

BID6 propose to apply GNNs to molecular property prediction tasks.

BID5 shows how to use GNNs to learn classifiers on image datasets in a few-shot manner.

BID6 study the effectiveness of message-passing in quantum chemistry.

BID4 apply message-passing on a graph constructed by coreference links to answer relational questions.

There are relatively fewer papers discussing how to adapt GNNs to natural language tasks.

For example, BID14 propose to apply GNNs to semantic role labeling and BID22 apply GNNs to knowledge base completion tasks.

BID31 apply GNNs to relation extraction by encoding dependency trees, and De Cao et al. FORMULA2 apply GNNs to multi-hop question answering by encoding co-occurence and co-reference relationships.

Although they also consider applying GNNs to natural language processing tasks, they still perform message-passing on predefined graphs.

Johnson (2017) introduces a novel neural architecture to generate a graph based on the textual input and dynamically update the relationship during the learning process.

In sharp contrast, this paper focuses on extracting relations from real-world relation datasets.

Relational reasoning has been explored in various fields.

For example, BID20 propose a simple neural network to reason the relationship of objects in a picture, BID26 build up a scene graph according to an image, and BID9 model the interaction of physical objects.

In this paper, we focus on the relational reasoning in natural language domain.

Existing works BID27 BID13 DISPLAYFORM0 Figure 2: Overall architecture: the encoding module takes a sequence of vector representations as inputs, and output a transition matrix as output; the propagation module propagates the hidden states from nodes to its neighbours with the generated transition matrix; the classification module provides task-related predictions according to nodes representations.the pair-wise relationship between entities in certain situations.

For example, BID27 ) is one of the earliest works that applies a simple CNN to this task, and BID28 further extends it with piece-wise max-pooling.

BID16 propose a multi-window version of CNN for relation extraction.

BID13 study an attention mechanism for relation extraction tasks.

BID18 predict n-ary relations of entities in different sentences with Graph LSTMs.

BID11 treat relations as latent variables which are capable of inducing the relations without any supervision signals.

BID29 show that the relation path has an important role in relation extraction.

BID15 show the effectiveness of LSTMs BID7 in relation extraction.

BID2 proposed a walk-based model to do relation extraction.

The most related work is BID24 , where the proposed model incorporates contextual relations with attention mechanism when predicting the relation of a target entity pair.

The drawback of existing approaches is that they could not make full use of the multi-hop inference patterns among multiple entity pairs and their relations within the sentence.

We first define the task of natural language relational reasoning.

Given a sequence of text with m entities, it aims to reason on both the text and entities and make a prediction of the labels of the entities or entity pairs.

In this section, we will introduce the general framework of GP-GNNs.

GP-GNNs first build a fullyconnected graph G = (V, E), where V is the set of entities, and each edge DISPLAYFORM0 l−1 extracted from the text.

After that, GP-GNNs employ three modules including (1) encoding module, (2) propagation module and (3) classification module to proceed relational reasoning, as shown in Fig. 2 .

The encoding module converts sequences into transition matrices corresponding to edges, i.e. the parameters of the propagation module, by DISPLAYFORM0 where f (·) could be any model that could encode sequential data, such as LSTMs, GRUs, CNNs, E(·) indicates an embedding function, and θ n e denotes the parameters of the encoding module of n-th layer.

The propagation module learns representations for nodes layer by layer.

The initial embeddings of nodes, i.e. the representations of layer 0, are task-related, which could be embeddings that encode features of nodes or just one-hot embeddings.

Given representations of layer n, the representations of layer n + 1 are calculated by DISPLAYFORM0 where N (v i ) denotes the neighbours of node v i in graph G and σ(·) denotes non-linear activation function.

Generally, the classification module takes node representations as inputs and outputs predictions.

Therefore, the loss of GP-GNNs could be calculated as DISPLAYFORM0 where θ c denotes the parameters of the classification module, K is the number of layers in propagation module and Y denotes the ground truth label.

The parameters in GP-GNNs are trained by gradient descent methods.

Relation extraction from text is a classic natural language relational reasoning task.

Given a sentence s = (x 0 , x 1 , . . . , x l−1 ), a set of relations R and a set of entities in this sentence V s = {v 1 , v 2 , . . .

, v |Vs| }, where each v i consists of one or a sequence of tokens, relation extraction from text is to identify the pairwise relationship r vi,vj ∈ R between each entity pair (v i , v j ).In this section, we will introduce how to apply GP-GNNs to relation extraction.

To encode the context of entity pairs (or edges in the graph), we first concatenate the position embeddings with word embeddings in the sentence: DISPLAYFORM0 where x t denotes the word embedding of word x t and p i,j t denotes the position embedding of word position t relative to the entity pair's position i, j (Details of these two embeddings are introduced in the next two paragraphs.)

After that, we feed the representations of entity pairs into encoder f (·) which contains a bi-directional LSTM and a multi-layer perceptron: DISPLAYFORM1 where n denotes the index of layer 1 , [·] means reshaping a vector as a matrix, BiLSTM encodes a sequence by concatenating tail hidden states of the forward LSTM and head hidden states of the backward LSTM together and MLP denotes a multi-layer perceptron with non-linear activation σ.

Word Representations We first map each token x t of sentence {x 0 , x 1 , . . .

, x l−1 } to a kdimensional embedding vector x t using a word embedding matrix W e ∈ R |V |×dw , where |V | is the size of the vocabulary.

Throughout this paper, we stick to 50-dimensional GloVe embeddings pre-trained on a 6 billion corpus BID19 .Position Embedding In this work, we consider a simple entity marking scheme 2 : we mark each token in the sentence as either belonging to the first entity v i , the second entity v j or to neither of those.

Each position marker is also mapped to a d p -dimensional vector by a position embedding matrix P ∈ R 3×dp .

We use notation p i,j t to represent the position embedding for x t corresponding to entity pair (v i , v j ).

Next, we use Eq. (2) to propagate information among nodes where the initial embeddings of nodes and number of layers are further specified as follows.

The Initial Embeddings of Nodes Suppose we are focusing on extracting the relationship between entity v i and entity v j , the initial embeddings of them are annotated as h (0) vi = a subject , and h (0) vj = a object , while the initial embeddings of other entities are set to all zeros.

We set special values for the head and tail entity's initial embeddings as a kind of "flag" messages which we expect to be passed through propagation.

Annotators a subject and a object could also carry the prior knowledge about subject entity and object entity.

In our experiments, we generalize the idea of Gated Graph Neural Networks BID12 by setting a subject = [1; 0] and a object = [0; 1] 3 .Number of Layers In general graphs, the number of layers K is chosen to be of the order of the graph diameter so that all nodes obtain information from the entire graph.

In our context, however, since the graph is densely connected, the depth is interpreted simply as giving the model more expressive power.

We treat K as a hyper-parameter, the effectiveness of which will be discussed in detail (Sect.

5.4).

The output module takes the embeddings of the target entity pair (v i , v j ) as input, which are first converted by: DISPLAYFORM0 where represents element-wise multiplication.

This could be used for classification: DISPLAYFORM1 where r vi,vj ∈ R, and MLP denotes a multi-layer perceptron module.

We use cross entropy here as the classification loss DISPLAYFORM2 where r vi,vj denotes the relation label for entity pair (v i , v j ) and S denotes the whole corpus.

In practice, we stack the embeddings for every target entity pairs together to infer the underlying relationship between each pair of entities.

We use PyTorch BID17 to implement our models.

To make it more efficient, we avoid using loop-based, scalar-oriented code by matrix and vector operations.

Our experiments mainly aim to: (1) showing that our best models could improve the performance of relation extraction under a variety of settings; (2) illustrating that how the number of layers affect the performance of our model; and (3) performing a qualitative investigation to highlight the difference between our models and baseline models.

In both part (1) and part (2), we do three subparts of experiments: (i) we will first show that our models could improve instance-level relation extraction on a human annotated test set, and (ii) then we will show that our models could also help enhance the performance of bag-level relation extraction on a distantly labeled test set 4 , and (iii) we also split a subset of distantly labeled test set, where the number of entities and edges is large.

Distantly labeled set BID24 have proposed a dataset with Wikipedia corpora.

There is a small difference between our task and theirs: our task is to extract the relationship between every pair of entities in the sentence, whereas their task is to extract the relationship between the given entity pair and the context entity pairs.

Therefore, we need to modify their dataset: (1) We added reversed edges if they are missing from a given triple, e.g. if triple (Earth, part of, Solar System) exists in the sentence, we add a reversed label, (Solar System, has a member, Earth), to it; (2) For all of the entity pairs with no relations, we added "NA" labels to them.

5 We use the same training set for all of the experiments.

Human annotated test set Based on the test set provided by BID24 , 5 annotators 6 are asked to label the dataset.

They are asked to decide whether or not the distant supervision is right for every pair of entities.

Only the instances accepted by all 5 annotators are incorporated into the human annotated test set.

There are 350 sentences and 1,230 triples in this test set.

Dense distantly labeled test set We further split a dense test set from the distantly labeled test set.

Our criteria are: (1) the number of entities should be strictly larger than 2; and (2) there must be at least one circle (with at least three entities) in the ground-truth label of the sentence BID0 .

This test set could be used to test our methods' performance on sentences with the complex interaction between entities.

There are 1,350 sentences and more than 17,915 triples and 7,906 relational facts in this test set.

We select the following models for comparison, the first four of which are our baseline models.

Context-Aware RE, proposed by BID24 .

This model utilizes attention mechanism to encode the context relations for predicting target relations.

It was the state-of-the-art models on Wikipedia dataset.

This baseline is implemented by ourselves based on authors' public repo 8 .Multi-Window CNN.

BID27 utilize convolutional neural networks to classify relations.

Different from the original version of CNN proposed in BID27 , our implementation, follows BID16 , concatenates features extracted by three different window sizes: 3, 5, 7.PCNN, proposed by BID28 .

This model divides the whole sentence into three pieces and applies max-pooling after convolution layer piece-wisely.

For CNN and following PCNN, the entity markers are the same as originally proposed in BID27 .LSTM or GP-GNN with K = 1 layer.

Bi-directional LSTM BID23 could be seen as an 1-layer variant of our model.

GP-GNN with K = 2 or K = 3 layerss.

These models are capable of performing 2-hop reasoning and 3-hop reasoning, respectively.

We select the best parameters for the validation set.

We select non-linear activation functions between relu and tanh, and select d n among {2, 4, 8, 12, 16} 9 .

We have also tried two forms of adjacent matrices: tied-weights (set A (n) = A (n+1) ) and untied-weights.

Table 1 shows our best hyper-parameter settings, which are used in all of our experiments.

Hyper-parameters Value learning rate 0.001 batch size 50 dropout ratio 0.5 hidden state size 256 non-linear activation σ relu embedding size for #layers = 1 8 embedding size for #layers = 2 and 3 12 adjacent matrices untied Table 1 : Hyper-parameters settings.

So far, we have only talked about the way to implement sentence-level relation extraction.

To evaluate our models and baseline models in bag-level, we utilize a bag of sentences with given entity pair to score the relations between them.

BID28 formalize the bag-level relation extraction as multi-instance learning.

Here, we follow their idea and define the score function of entity pair and its corresponding relation r as a max-one setting: TAB3 and 3, we can see that our best models outperform all the baseline models significantly on all three test sets.

These results indicate our model could successfully conduct reasoning on the fully-connected graph with generated parameters from natural language.

These results also indicate that our model not only performs well on sentence-level relation extraction but also improves on bag-level relation extraction.

Note that Context-Aware RE also incorporates context information to predict the relation of the target entity pair, however, we argue that Context-Aware RE only models the co-occurrence of various relations, ignoring whether the context relation participates in the reasoning process of relation extraction of the target entity pair.

Context-Aware RE may introduce more noise, for it may mistakenly increase the probability of a relation with the similar topic with the context relations.

We will give samples to illustrate this issue in Sect.

5.5.

Another interesting observation is that our #layers=1 version outperforms CNN and PCNN in these three datasets.

One probable reason is that sentences from Wikipedia corpus are always complex, which may be hard to model for CNN and PCNN.

Similar conclusions are also reached by BID30 .

Table 4 : Sample predictions from the baseline models and our GP-GNN model.

Ground truth graphs are the subgraph in Wikidata knowledge graph induced by the sets of entities in the sentences.

The models take sentences and entity markers as input and produce a graph containing entities (colored and bold) and relations between them.

Although "No Relation" is also be seen as a type of relation, we only show other relation types in the graphs.

DISPLAYFORM0

The number of layers represents the reasoning ability of our models.

A K-layer version has the ability to infer K-hop relations.

To demonstrate the effects of the number of layers, we also compare our models with different numbers of layers.

From TAB3 , we could see that on all three datasets, 3-layer version achieves the best.

We could also see from FIG0 that as the number of layers grows, the curves get higher and higher precision, indicating considering more hops in reasoning leads to better performance.

However, the improvement of the third layer is much smaller on the overall distantly supervised test set than the one on the dense subset.

This observation reveals that the reasoning mechanism could help us identify relations especially on sentences where there are more entities.

We could also see that on the human annotated test set 3-layer version to have a greater improvement over 2-layer version as compared with 2-layer version over 1-layer version.

It is probably due to the reason that bag-level relation extraction is much easier.

In real applications, different variants could be selected for different kind of sentences or we can also ensemble the prediction from different models.

We leave these explorations for future work.

− −−−− → z to find the fact (BankUnited Center, located in, English).

Note that (BankUnited Center, located in, English) is even not in Wikidata, but our model could identify this fact through reasoning.

We also find that Context-Aware RE tends to predict relations with similar topics.

For example, in the third case, share boarder with and located in are both relations about territory issues.

Consequently, Context-Aware RE makes a mistake by predicting (Kentucky, share boarder with, Ohio).

As we have discussed before, this is due to its mechanism to model co-occurrence of multiple relations.

However, in our model, since Ohio and Johnson County have no relationship, this wrong relation is not predicted.

We addressed the problem of utilizing GNNs to perform relational reasoning with natural languages.

Our proposed models, GP-GNNs, solves the relational message-passing task by encoding natural language as parameters and performing propagation from layer to layer.

Our model can also be considered as a more generic framework for graph generation problem with unstructured input other than text, e.g. images, videos, audios.

In this work, we demonstrate its effectiveness in predicting the relationship between entities in natural language and bag-level and show that by considering more hops in reasoning the performance of relation extraction could be significantly improved.

@highlight

A graph neural network model with parameters generated from natural languages, which can perform multi-hop reasoning. 