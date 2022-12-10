The problem of building a coherent and non-monotonous conversational agent with proper discourse and coverage is still an area of open research.

Current architectures only take care of semantic and contextual information for a given query and fail to completely account for syntactic and external knowledge which are crucial for generating responses in a chit-chat system.

To overcome this problem, we propose an end to end multi-stream deep learning architecture which learns unified embeddings for query-response pairs by leveraging contextual information from memory networks and syntactic information by incorporating Graph Convolution Networks (GCN) over their dependency parse.

A stream of this network also utilizes transfer learning by pre-training a bidirectional transformer to extract semantic representation for each input sentence and incorporates external knowledge through the neighbourhood of the entities from a Knowledge Base (KB).

We benchmark these embeddings on next sentence prediction task and significantly improve upon the existing techniques.

Furthermore, we use AMUSED to represent query and responses along with its context to develop a retrieval based conversational agent which has been validated by expert linguists to have comprehensive engagement with humans.

With significant advancements in Automatic speech recognition systems (Hinton et al., 2012; Kumar et al., 2018) and the field of natural language processing, conversational agents have become an important part of the current research.

It finds its usage in multiple domains ranging from self-driving cars (Chen et al., 2017b) to social robots and virtual assistants (Chen et al., 2017a) .

Conversational agents can be broadly classified into two categories: a task oriented chat bot and a chit-chat based system respectively.

The former works towards completion of a certain goal and are specifically designed for domain-specific needs such as restaurant reservations (Wen et al., 2017) , movie recommendation (Dhingra et al., 2017) , flight ticket booking systems ) among many others.

The latter is more of a personal companion and engages in human-computer interaction for entertainment or emotional companionship.

An ideal chit chat system should be able to perform non-monotonous interesting conversation with context and coherence.

Current chit chat systems are either generative (Vinyals & Le, 2015) or retrieval based in nature.

The generative ones tend to generate natural language sentences as responses and enjoy scalability to multiple domains without much change in the network.

Even though easier to train, they suffer from error-prone responses (Zhang et al., 2018b) .

IR based methods select the best response from a given set of answers which makes them error-free.

But, since the responses come from a specific dataset, they might suffer from distribution bias during the course of conversation.

A chit-chat system should capture semantic, syntactic, contextual and external knowledge in a conversation to model human like performance.

Recent work by Bordes et al. (2016) proposed a memory network based approach to encode contextual information for a query while performing generation and retrieval later.

Such networks can capture long term context but fail to encode relevant syntactic information through their model.

Things like anaphora resolution are properly taken care of if we incorporate syntax.

Our work improves upon previous architectures by creating enhanced representations of the conversation using multiple streams which includes Graph Convolution networks (Bruna et al., 2014) , Figure 1 : Overview of AMUSED.

AMUSED first encodes each sentence by concatenating embeddings (denoted by ⊕) from Bi-LSTM and Syntactic GCN for each token, followed by word attention.

The sentence embedding is then concatenated with the knowledge embedding from the Knowledge Module ( Figure 2 ).

The query embedding passes through the Memory Module ( Figure 3 ) before being trained using triplet loss.

Please see Section 4 for more details.

transformers (Vaswani et al., 2017) and memory networks (Bordes et al., 2016) in an end to end setting, where each component captures conversation relevant information from queries, subsequently leading to better responses.

Our contribution for this paper can be summarized as follows:

• We propose AMUSED, a novel multi stream deep learning model which learns rich unified embeddings for query response pairs using triplet loss as a training metric.

• We perform multi-head attention over query-response pairs which has proven to be much more effective than unidirectional or bi-directional attention.

• We use Graph Convolutions Networks in a chit-chat setting to incorporate the syntactical information in the dialogue using its dependency parse.

• Even with the lack of a concrete metric to judge a conversational agent, our embeddings have shown to perform interesting response retrieval on Persona-Chat dataset.

The task of building a conversational agent has gained much traction in the last decade with various techniques being tried to generate relevant human-like responses in a chit-chat setting.

Previous modular systems (Martin & Jurafsky, 2009 ) had a complex pipeline based structure containing various hand-crafted rules and features making them difficult to train.

This led to the need of simpler models which could be trained end to end and extended to multiple domains.

Vinyals & Le (2015) proposed a simple sequence to sequence model that could generate answers based on the current question, without needing extensive feature engineering and domain specificity.

However, the responses generated by this method lacked context.

To alleviate this problem, Sordoni et al. (2015) introduced a dynamic-context generative network which is shown to have improved performance on unstructured Twitter conversation dataset.

To model complex dependencies between sub-sequences in an utterance, Serban et al. (2017) proposed a hierarchical latent variable encoder-decoder model.

It is able to generate longer outputs while maintaining context at the same time.

Reinforcement learning based approaches have also been deployed to generate interesting responses (Zhang et al., 2018a) and tend to possess unique conversational styles (Asghar et al., 2017) .

With the emergence of a number of large datasets, retrieval methods have gained a lot of popularity.

Even though the set of responses are limited in this scenario, it doesn't suffer from the problem of generating meaningless responses.

A Sequential Matching Network proposed by Wu et al. (2017) performs word matching of responses with the context before passing their vectors to a RNN.

Addition of external information along with the current input sentence and context improves the system as is evident by incorporating a large common sense knowledge base into an end to end conversational agent (Young et al., 2018) .

To maintain diversity in the responses, Song et al. (2018) suggests a method to combine a probabilistic model defined on item-sets with a seq2seq model.

Responses like 'I am fine' can make conversations monotonous; a specificity controlled model (Zhang et al., 2018b) in conjunction with seq2seq architecture overcomes this problem.

These networks helps solve one or the other problem in isolation.

To maintain proper discourse in the conversation, context vectors are passed together with input query vector into a deep learning model (Sordoni et al., 2015) .

A context modelling approach which includes concatenation of dialogue history has also been tried (Martin & Jurafsky, 2009 ).

However, the success of memory networks on Question-Answering task (Sukhbaatar et al., 2015) opened the door for its further use in conversational agents.

Bordes et al. (2016) used the same in a task oriented setting for restaurant domain and reported accuracies close to 96% in a full dialogue scenario.

Zhang et al. (2018c) further used these networks in a chit chat setting on Persona-Chat dataset and came up with personalized responses.

In our network, we make use of Graph Convolution Networks (Kipf & Welling, 2017; Defferrard et al., 2016) , which have been found to be quite effective for encoding the syntactic information present in the dependency parse of sentences .

External Knowledge Bases (KBs) have been exploited in the past to improve the performances in various tasks (Vashishth et al., 2018a; b; Ling & Weld, 2012) .

The relation based strategy followed by Hixon et al. (2015) creates a KB from dialogue itself, which is later used to improve Question-Answering (Saha et al., 2018) .

Han et al. (2015) ; Ghazvininejad et al. (2018) have used KBs to generate more informative responses by using properties of entities in the graph. (Young et al., 2018) focused more on introducing knowledge from semantic-nets rather than general KBs.

GCN for undirected graph: For an undirected graph G = (V, E), where V is the set of n vertices and E is the set of edges, the representation of the node v is given by x v ∈ R m , ∀v ∈ V .

The output hidden representation h v ∈ R d of the node after one layer of GCN is obtained by considering only the immediate neighbors of the node as given by Kipf & Welling (2017) .

To capture the multi-hop representation, GCN layers can be stacked on top of each other.

GCN for labeled directed graph: For a directed graph G = (V, E), where V is the set of vertices we define the edge set E as a set of tuples (u, v, l(u, v) ) where there is an edge having label l(u, v) between nodes u and v. proposed the assumption that information doesn't necessarily propagate in certain directions in the directed edge, therefore, we add tuples having inverse edges (v, u, l(u, v) −1 ) as well as self loops (u, u, Ω), where Ω denotes self loops, to our edge set E to get an updated edge set E .

The representation of a node x v , after the k th layer is given as :

are trainable edge-label specific parameters for the layer k, N (v) denotes the set of all vertices that are immediate neighbors of v and f is any non-linear activation function (e.g., ReLU:

Since we are obtaining the dependency graph from Stanford CoreNLP , some edges can be erroneous.

Edgewise gating (Bastings et al., 2017; helps to alleviate this problem by decreasing the effects of such edges.

For this, each edge (u, v, l(u, v) ) is assigned a score which is given by : u,v) ) ∈ R are trained and σ denotes the sigmoid function.

Incorporating this, the final GCN embedding for a node v after n th layer is given as :

This section provides details of three main components of AMUSED which can broadly be classified into Syntactic, Knowledge and Memory Module.

We hypothesize that each module captures information relevant for learning representations, for a query-response pair in a chit-chat setting.

Suppose that we have a dataset D consisting of a set of conversations d 1 , d 2 , ..., d C where d c represents a single full length conversation consisting of multiple dialogues.

A conversation d c is given by a set of tuples (q 1 , r 1 ), (q 2 , r 2 ), ..., (q n , r n ) where a tuple (q i , r i ) denotes the query and response pair for a single turn.

The context for a given query q i ∀ i ≥ 2 is defined by a list of sentences l : [q 1 , r 1 , ..., q i−1 , r i−1 ].

We need to find the best response r i from the set of all responses, R. The training set D for AMUSED is defined by set of triplets (q i , r i , n i ) ∀ 1 ≤ i ≤ N where N is the total number of dialogues and n i is a negative response randomly chosen from set R.

Syntax information from dependency trees has been successfully exploited to improve a lot of Natural Language Processing (NLP) tasks (Vashishth et al., 2018a; Mintz et al., 2009) .

In dialog agents, where anaphora resolution as well as sentence structure influences the responses, it finds special usage.

A Bi-GRU followed by a syntactic GCN is used in this module.

Each sentence s from the input triplet is represented with a list of k-dimensional GloVe embedding (Pennington et al., 2014) corresponding to each of the m tokens in the sentence.

The sentence representation S ∈ R m×k is then passed to a Bi-GRU to obtain the representation S gru ∈ R m×dgru , where d gru is the dimension of the hidden state of Bi-GRU.

This contextual encoding (Graves et al., 2013) captures the local context really well, but fails to capture the long range dependencies that can be obtained from the dependency trees.

We use GCN to encode this syntactic information.

Stanford CoreNLP ) is used to obtain the dependency parse for the sentence s. Giving the input as S gru , we use GCN Equation 1, to obtain the syntactic embedding S gcn .

Following Nguyen & Grishman (2018) , we only use three edge labels, namely forward-edge, backward-edge and self-loop.

This is done because incorporating all the edge labels from the dependency graph heavily over-parameterizes the model.

The final token representation is obtained by concatenating the contextual Bi-GRU representation h gru and the syntactic GCN representation h gcn .

A sentence representation is then obtained by passing the tokens through a layer of word attention as used by (Vashishth et al., 2018b; Jat et al., 2018) , which is concatenated with the embedding obtained from the Knowledge Module (described in Section 4.2) to obtain the final sentence representation h concat .

The final sentence representation h concat of the query is then passed into Knowledge Module.

It is further subdivided into two components: a pre-trained Transformer model for next dialogue prediction problem and a component to incorporate information from external Knowledge Bases (KBs).

The next dialogue prediction task is described as follows: For each query-response pair in the dataset, we generate a positive sample (q, r) and a negative sample (q, n) where n is randomly chosen from the set of responses R in dataset D. Following Devlin et al. (2018) , a training example is defined by concatenating q and r which are separated by a delimiter || and is given by [q||r] .

The problem is to classify if the next sentence is a correct response or not.

A pre-trained BERT model is used to further train a binary classifier for the next dialogue prediction task as described above.

After the model is trained, the pre-final layer is considered and the vector from the special cls token is chosen as the sentence representation.

The representation thus obtained would have a tendency to be more inclined towards its correct positive responses.

Multi-head attention in the transformer network, along with positional embeddings during training, helps it to learn intra as well as inter sentence dependencies (Devlin et al., 2018; Vaswani et al., 2017) .

The input query sentence is then passed from this network to obtain the BERT embedding, h bert .

In our day-to-day conversations, to ask succinct questions, or to keep the conversation flowing, we make use of some background knowledge .

For example, if someone remarks that they like rock music, we can ask a question if they have listened to Nirvana.

It can be done only if we know that Nirvana plays rock music.

To incorporate such external information, we can make use of existing Knowledge Bases like Wikipedia, Freebase (Bollacker et al., 2008) and Wikidata (Vrandečić & Krötzsch, 2014) .

Entities in these KBs are linked to each other using relations.

We can expand the information we have about an entity by looking at its linked entities.

Multiple hops of the Knowledge Graph (KG) can be used to expand knowledge.

In AMUSED, we do this by passing the input query into Stanford CoreNLP to obtain entity linking information to Wikipedia.

Suppose the Wikipedia page of an entity e contains links to the set of entities E. We ignore relation information and only consider one-hop direct neighbors of e.

To obtain a KB-expanded embedding h kb of the input sentence, we take the average of GloVE embeddings of each entity in E. In place of Wikipedia, bigger knowledge bases like Wikidata, as well as relation information, can be used to improve KB embeddings.

We leave that for future work.

For effective conversations, it is imperative that we form a sense from the dialogues that have already happened.

A question about '

Who is the president of USA' followed by '

What about France' should be self-containing.

This dialogue context is encoded using a memory network (Sukhbaatar et al., 2015) .

The memory network helps to capture context of the conversation by storing dialogue history i.e. both question and responses.

The query representation, h concat is passed to the memory network, along with BERT embeddings h bert of the context, from the Knowledge Module (Section 4.2).

In AMUSED, memory network uses supporting memories to generate the final query representation (h concat ).

Supporting memories contains input (m i ) and output (c i ) memory cells (Sukhbaatar et al., 2015) .

The incoming query q i as well as the history of dialogue context l : [(q 1 , r 1 ), .., (q i−1 , r i−1 )] is fed as input.

The memory cells are populated using the BERT representations of context sentences l as follows:

Following Bordes et al. (2016) , the incoming query embedding along with input memories is used to compute relevance of context stories as a normalized vector of attention weights as a i = (< m i , h concat >) , where < a, b > represents the inner product of a and b. The response from the output memory, o, is then generated as : o = i a i c i .

The final output of the memory cell, u is obtained by adding o to h concat .

To capture context in an iterative manner, memory cells are stacked in layers (Sukhbaatar et al., 2015) which are called as hops.

The output of the memory cell after the k th hop is given by

The memory network performs k such hops and the final representation h concat is given by sum of o k and u k .

Triplet loss has been successfully used for face recognition (Schroff et al., 2015) .

Our insight is that traditional loss metrics might not be best suited for a retrieval-based task with a multitude of valid responses to choose from.

We define a Conversational Euclidean Space where the representation of a sentence is driven by its context in the dialogue along with its syntactic and semantic information.

We have used this loss to bring the query and response representations closer in the conversational space.

Questions with similar answers should be closer to each other and the correct response.

An individual data point is a triplet which consists of a query (q i ), its correct response (r i ) and a negative response (n i ) selected randomly.

We need to learn their embeddings φ(q i ) = h

where α is the margin hyper-parameter used to separate negative and positive pairs.

If I be the set of triplets, N the number of triplets and w the parameter set, then, triplet loss (L) is defined as :

We use this dataset to build and evaluate the chit-chat system.

Persona-Chat (Zhang et al., 2018c ) is an open domain dataset on personal conversations created by randomly pairing two humans on Amazon Mechanical Turk.

The paired crowd workers converse in a natural manner for 6 − 12 turns.

This made sure that the data mimic normal conversations between humans which is very crucial for building such a system.

This data is not limited to social media comments or movie dialogues.

It contains 9907 training conversations and 1000 conversations each for testing and validation.

There are a total of 131, 438 query-response pairs with a vocab size of 19262 in the dataset.

We use it for training AMUSED as it provides consistent conversations with proper context.

DSTC: Dialogue State Tracking Challenge dataset (Henderson et al., 2014) contains conversations for restaurant booking task.

Due to its task oriented nature, it doesn't need an external knowledge module, so we train it only using memory and syntactic module and test on an automated metric.

We further use Multi-Genre Natural Language Inference and Microsoft Research Paraphrase Corpus (Wang et al., 2019) to fine-tune parts of the network i.e; Knowledge Module.

It is done because these datasets resemble the core nature of our problem where in we want to predict the correctness of one sentence in response to a particular query.

Pre-training BERT: Before training AMUSED, the knowledge module is processed by pre-training a bidirectional transformer network and extracting one hop neighborhood entities from Wikipedia KB.

We use the approach for training as explained in Section 4.2.1.

There are 104224 positive training and 27214 validation query-response pairs from Persona Chat.

We perform three different operations: a) Equal sampling: Sample equal number of negative examples from dataset, b) Oversampling: Sample double the negatives to make training set biased towards negatives and c) Under sampling: Sample 70% of negatives to make training set biased towards positives.

Batch size and maximum sequence length are 32 and 128 respectively.

We fine-tune this next sentence prediction model with MRPC and MNLI datasets which improves the performance.

Training to learn Embeddings: AMUSED requires triplets to be trained using triplet loss.

A total of 131438 triplets of the form (q, r, n) are randomly split in 90:10 ratio to form training and validation set.

The network is trained with a batch size of 64 and dropout of 0.5.

Word embedding size is chosen to be 50.

Bi-GRU and GCN hidden state dimensions are chosen to be 192 and 32 respectively.

One layer of GCN is employed.

Validation loss is used as a metric to stop training which converges after 50 epochs using Adam optimizer at 0.001 learning rate.

As a retrieval based model, the system selects a response from the predefined answer set.

The retrieval unit extracts embedding (h concat ) for each answer sentence from the trained model and stores it in a representation matrix which will be utilized later during inference.

First, a candidate subset A is created by sub-sampling a set of responses having overlapping words with a given user query.

Then, the final output is retrieved on the basis of cosine similarity between query embedding h concat and the extracted set of potential responses (A).

The response with the highest score is then labelled as the final answer and the response embedding is further added into the memory to take care of context.

The model resulting from oversampling method beats its counterparts by more than 3% in accuracy.

It clearly indicates that a better model is one which learns to distinguish negative examples well.

The sentence embeddings obtained through this model is further used for lookup in the Knowledge Module (Section 4.2) in AMUSED.

We use two different automated metrics to check the effectiveness of the model and the queryresponse representations that we learnt.

Next Dialogue Prediction Task: Various components of AMUSED are analysed for their performance on next dialogue prediction task.

This task tell us that, given two sentences (a query and a response) and the context, whether second sentence is a valid response to the first sentence or not.

Embeddings for queries and responses are extracted from our trained network and then multiple operations which include a) Concatenation, b) Element wise min and c) Subtraction are performed on those before passing them to a binary classifier.

A training example consists of embeddings of two sentences from a (q, a) or (q, n) pair which are created in a similar fashion as in Section 4.2.1.

Accuracy on this binary classification problem has been used to select the best network.

Furthermore, we perform ablation studies using different modules to understand the effect of each component in the network.

A 4 layer neural network with ReLU activation in its hidden layers and softmax in the final layer is used as the classifier.

External knowledge in conjunction with memory and GCN module has the best accuracy when embeddings of query and response are concatenated together.

A detailed study of performance of various components over these operations is shown in Table 1 .

Precision@1: This is another metric used to judge the effectiveness of our network.

It is different from the next sentence prediction task accuracy.

It measures that for n trials, the number of times a relevant response is reported with the highest confidence value.

Table 2 reports a comparative study of this metric on 500 trials conducted for AMUSED along with results for other methods.

DSTC dataset is also evaluated on this metric without the knowledge module as explained in Section 5.1

Looking for exact answers might not be a great metric as many diverse answers might be valid for a particular question.

So, we must look for answers which are contextually relevant for that query.

Overall, we use next sentence prediction task accuracy to choose the final model before retrieval.

There is no concrete metric to evaluate the performance of an entire conversation in a chit-chat system.

Hence, human evaluation was conducted using expert linguists to check the quality of conversation.

They were asked to chat for 7 turns and rate the quality of responses on a scale of 1−10 where a higher score is better.

Similar to Zhang et al. (2018c) , there were multiple parameters to rate the chat based on coherence, context awareness and non-monotonicity to measure various factors that are essential for a natural dialogue.

By virtue of our network being retrieval based, we don't need to judge the responses based on their structural correctness as this will be implicit.

To monitor the effect of each neural component, we get it rated by experts either in isolation or in conjunction with other components.

Such a study helps us understand the impact of different modules on a human based conversation.

Dialogue system proposed by Zhang et al. (2018c) is also reproduced and reevaluated for comparison.

From Table 3 we can see that human evaluation follows a similar trend as the automated metric, with the best rating given to the combined architecture.

In the paper, we propose AMUSED, a multi-stream architecture which effectively encodes semantic information from the query while properly utilizing external knowledge for improving performance on natural dialogue.

It also employs GCN to capture long-range syntactic information and improves context-awareness in dialogue by incorporating memory network.

Through our experiments and results using different metrics, we demonstrate that learning these rich representations through smart training (using triplets) would improve the performance of chit-chat systems.

The ablation studies show the importance of different components for a better dialogue.

Our ideas can easily be extended to various conversational tasks which would benefit from such enhanced representations.

@highlight

This paper provides a multi -stream end to end approach to learn unified embeddings for query-response pairs in dialogue systems by leveraging contextual, syntactic, semantic and external information together.