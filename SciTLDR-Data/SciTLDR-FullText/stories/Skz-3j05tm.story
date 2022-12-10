Domain specific goal-oriented dialogue systems typically require modeling three types of inputs, viz., (i) the knowledge-base associated with the domain, (ii) the history of the conversation, which is a sequence of utterances and (iii) the current utterance for which the response needs to be generated.

While modeling these inputs, current state-of-the-art models such as Mem2Seq typically ignore the rich structure inherent in the knowledge graph and the sentences in the conversation context.

Inspired by the recent success of structure-aware Graph Convolutional Networks (GCNs) for various NLP tasks such as machine translation, semantic role labeling and document dating, we propose a memory augmented GCN for goal-oriented dialogues.

Our model exploits (i) the entity relation graph in a knowledge-base  and (ii) the dependency graph associated with an utterance to compute richer representations for words and entities.

Further, we take cognizance of the fact that in certain situations, such as, when the conversation is in a code-mixed language, dependency parsers may not be available.

We show that in such situations we could use the global word co-occurrence graph and use it to enrich the representations of utterances.

We experiment with the modified DSTC2 dataset and its recently released code-mixed versions in four languages and show that our method outperforms existing state-of-the-art methods, using a wide range of evaluation metrics.

Goal-oriented dialogue systems which can assist humans in various day-to-day activities have widespread applications in several domains such as e-commerce, entertainment, healthcare, etc.

For example, such systems can help humans in scheduling medical appointments, reserving restaurants, booking tickets, etc..

From a modeling perspective, one clear advantage of dealing with domain specific goal-oriented dialogues is that the vocabulary is typically limited, the utterances largely follow a fixed set of templates and there is an associated domain knowledge which can be exploited.

More specifically, there is some structure associated with the utterances as well as the knowledge base.

More formally, the task here is to generate the next response given (i) the previous utterances in the conversation history (ii) the current user utterance (known as the query) and (iii) the entities and relationships in the associated knowledge base.

Current state-of-the-art methods BID30 BID23 typically use variants of Recurrent Neural Network BID10 to encode the history and current utterance and an external memory network to store the entities in the knowledge base.

The encodings of the utterances and memory elements are then suitably combined using an attention network and fed to the decoder to generate the response, one word at a time.

However, these methods do not exploit the structure in the knowledge base as defined by entity-entity relations and the structure in the utterances as defined by a dependency parse.

Such structural information can be exploited to improve the performance of the system as demonstrated by recent works on syntax-aware neural machine translation BID13 BID2 BID4 , semantic role labeling and document dating BID35 which use GCNs BID8 BID9 BID19 to exploit sentence structure.

In this work, we propose to use such graph structures for goal-oriented dialogues.

In particular, we compute the dependency parse tree for each utterance in the conversation and use a GCN to capture the interactions between words.

This allows us to capture interactions between distant words in the sentence as long as they are connected by a dependency relation.

We also use GCNs to encode the entities of the KB where the entities are treated as nodes and the relations as edges of the graph.

Once we have a richer structure aware representation for the utterances and the entities, we use a sequential attention mechanism to compute an aggregated context representation from the GCN node vectors of the query, history and entities.

Further, we note that in certain situations, such as, when the conversation is in a code-mixed language or a language for which parsers are not available then it may not be possible to construct a dependency parse for the utterances.

To overcome this, we construct a co-occurrence matrix from the entire corpus and use this matrix to impose a graph structure on the utterances.

More specifically, we add an edge between two words in a sentence if they co-occur frequently in the corpus.

Our experiments suggest that this simple strategy acts as a reasonable substitute for dependency parse trees.

We perform experiments with the modified DSTC2 BID3 dataset which contains goal-oriented conversations for reserving restaurants.

We also use its recently released code-mixed versions BID1 which contain code-mixed conversations in four different languages, viz.

, Hindi, Bengali, Gujarati and Tamil.

We compare with recent state-of-the-art methods and show that on average the proposed model gives an improvement of 2.8 BLEU points and 2 ROUGE points.

Our contributions can be summarized as follows: (i) We use GCNs to incorporate structural information for encoding query, history and KB entities in goal-oriented dialogues (ii) We use a sequential attention mechanism to obtain query aware and history aware context representations (iii) We leverage co-occurrence frequencies and PPMI (positive-pointwise mutual information) values to construct contextual graphs for code-mixed utterances and (iv) We show that the proposed model obtains state-of-the-art results on the modified DSTC2 dataset and its recently released code-mixed versions.

In this section we review the previous work in goal-oriented dialogue systems and describe the introduction of GCNs in NLP.Goal-Oriented Dialogue System : Initial goal-oriented dialogue systems BID39 BID37 were based on dialogue state tracking BID38 BID15 b) and included pipelined modules for natural language understanding, dialogue state tracking, policy management and natural language generation.

used neural networks for these intermediate modules but still lacked absolute end-to-end trainability.

Such pipelined modules were restricted by the fixed slot-structure assumptions on the dialogue state and required per-module based labelling.

To mitigate this problem BID3 released a version of goal-oriented dialogue dataset that focuses on the development of end-to-end neural models.

Such models need to reason over the associated KB triples and generate responses directly from the utterances without any additional annotations.

For example, BID3 proposed a Memory Network BID34 based model to match the response candidates with the multi-hop attention weighted representation of the conversation history and the KB triples in memory.

BID22 further added highway BID33 and residual connections BID14 to the memory network in order to regulate the access to the memory blocks.

BID30 developed a variant of RNN cell which computes a refined representation of the query over multiple iterations before querying the memory.

However, all these approaches retrieve the response from a set of candidate responses and such a candidate set is not easy to obtain in any new domain of interest.

To account for this, ; Zhao et al. (2017) adapted RNN based encoder-decoder models to generate appropriate responses instead of retrieving them from a candidate set.

introduced a key-value memory network based generative model which integrates the underlying KB with RNN based encode-attend-decode models.

BID23 used memory networks on top of the RNN decoder to tightly integrate KB entities with the decoder to generate more infor-mative responses.

However, as opposed to our work, all these works ignore the underlying structure of the entity-relation graph of the KB and the syntactic structure of the utterances.

Recently, there has been an active interest in enriching existing encode-attenddecode models BID0 with structural information for various NLP tasks.

Such structure is typically obtained from the constituency and/or dependency parse of sentences.

The idea is to treat the output of a parser as a graph and use an appropriate network to capture the interactions between the nodes of this graph.

For example, BID13 and BID4 showed that incorporating such syntactical structures as Tree-LSTMs in the encoder can improve the performance of Neural Machine Translation (NMT).

BID29 use Graph-LSTMs to perform cross sentence n-ary relation extraction and show that their formulation is applicable to any graph structure and Tree-LSTMs can be thought of as a special case of it.

In parallel, Graph Convolutional Networks (GCNs) BID9 BID8 BID19 and their variants BID20 have emerged as state-of-the-art methods for computing representations of entities in a knowledge graph.

They provide a more flexible way of encoding such graph structures by capturing multi-hop relationships between nodes.

This has led to their adoption for various NLP tasks such as neural machine translation BID25 BID2 , semantic role labeling , document dating BID35 and question answering BID17 BID26 .To the best of our knowledge ours is the first work that uses GCNs to incorporate dependency structural information and the entity-entity graph structure in a single end-to-end neural model for goaloriented dialogue.

This is also the first work that incorporates contextual co-occurrence information for code-mixed utterances, for which no dependency structures are available.

In this section we describe Graph Convolutional Networks (GCN) BID19 for undirected graphs and then describe their syntactic versions which work with directed labeled edges of dependency parse trees.

Graph convolutional networks operate on a graph structure and compute representations for the nodes of the graph by looking at the neighbourhood of the node.

k layers of GCNs can be stacked to account for neighbours which are k-hops away from the current node.

Formally, let G = (V, E) be an undirected graph where V is the set of nodes (let |V| = n) and E is the set of edges.

Let X ∈ R n×m be the input feature matrix with n nodes and each node x u (u ∈ V) is represented by an m-dimensional feature vector.

The output of a 1-layer GCN is the hidden representation matrix H ∈ R n×d where each d-dimensional representation of a node captures the interactions with its 1-hop neighbour.

Each row of this matrix can be computed as: DISPLAYFORM0 Here W ∈ R d×m is the model parameter matrix, b ∈ R d is the bias vector and ReLU is the rectified linear unit activation function.

N (v) is the set of neighbours of node v and is assumed to also include the node v so that the previous representation of the node v is also considered while computing the new hidden representation.

To capture interactions with nodes which are multiple hops away, multiple layers of GCNs can be stacked together.

Specifically, the representation of node v after k th GCN layer can be formulated as: DISPLAYFORM1 where h k u is the representation of the u th node in the (k − 1) th GCN layer and h DISPLAYFORM2

In a directed labeled graph G = (V, E), each edge between nodes u and v is represented by a triple (u, v, L(u, v) ) where L(u, v) is the associated edge label.

modified GCNs to operate over directed labeled graphs, such as the dependency parse tree of a sentence.

For such a tree, in order to allow information to flow from head to dependents and vice-versa, they added inverse dependency edges from dependents to heads such as (v, u, L(u, v) ) to E and made the model parameters and biases label specific.

In their formulation, u,v) which are label specific.

Suppose there are L different labels, then this formulation will require L weights and biases per GCN layer resulting in a large number of parameters.

To avoid this, the authors use only three sets of weights and biases per GCN layer (as opposed to L) depending on the direction in which the information flows.

More specifically, u,v) , where dir(u, v) indicates whether information flows from u to v, v to u or u = v. In this work, we also make b DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 instead of having a separate bias per label.

The final GCN formulation can thus be described as: DISPLAYFORM3

We first formally define the task of end-to-end goal-oriented dialogue generation.

Each dialogue of t turns can be viewed as a succession of user utterances (U ) and system responses (S) and can be represented as: (U 1 , S 1 , U 2 , S 2 , ..U t , S t ).

Along with these utterances, each dialogue is also accompanied by e KB triples which are relevant to that dialogue and can be represented as: DISPLAYFORM0 Each triple is of the form: (entity 1 , relation, entity 2 ).

These triples can be represented in the form of a graph G k = (V k , E k ) where V is the set of all entities and each edge in E is of the form: (entity 1 , entity 2 , relation) where relation signifies the edge label.

At any dialogue turn i, given the (i) dialogue history H = (U 1 , S 1 , U 2 , ..S i−1 ), (ii) the current user utterance as the query Q = U i and (iii) the associated knowledge graph G k , the task is to generate the current response S i which leads to a completion of the goal.

As mentioned earlier, we exploit the graph structure in KB and the syntactic structure in the utterances to generate appropriate responses.

Towards this end we propose a model with the following components for encoding these three types of inputs.

The query Q = U i is the i th (current) utterance in the dialogue and contains |Q| tokens.

We denote the embedding of the i th token in the query as q i We first compute the contextual representations of these tokens by passing them through a bidirectional RNN: DISPLAYFORM0 Now, consider the dependency parse tree of the query sentence denoted by G Q = (V Q , E Q ).

We use a query specific GCN to operate on G Q , which takes DISPLAYFORM1 as the input to the 1 st GCN layer.

The node representation in the k th hop of the query specific GCN is computed as: DISPLAYFORM2 where u,v) DISPLAYFORM3 DISPLAYFORM4 Figure 1: Illustration of the GCN and RNN+GCN modules which are used as encoders in our model.

The notations are specific to the dialogue history encoder but both the encoders are same for the query.

The GCN encoder is same for the KB except the graph structure.

The history H of the dialogue contains |H| tokens and we denote the embedding of the i th token in the history by p i Once again, we first compute the hidden representations of these embeddings using a bidirectional RNN: DISPLAYFORM0 We now compute a dependency parse tree for each sentence in the history and collectively represent all the trees as a single graph G H = (V H , E H ).

Note that this graph will only contain edges between words belonging to the same sentence and there will be no edges between words across sentences.

We then use a history specific GCN to operate on G H which takes s t as the input to the 1 st layer.

The node representation in the k th hop of the history specific GCN is computed as: DISPLAYFORM1 where V k dir (u,v) and o k dir (u,v) are edge direction specific history-GCN weights and biases in the k th hop and a 1 u = s u .

Such an encoder with a single hop of GCN is illustrated in figure 1(b) and the encoder without the BiRNN is depicted in figure 1(a) .

As mentioned earlier, G K = (V K , E K ) is the graph capturing the interactions between the entities in the knowledge graph associated with the dialogue.

Let there be m such entities and we denote the embeddings of the node corresponding to the i th entity as e i We then operate a KB specific GCN on these entity representations to obtain refined representations which capture relations between entities.

The node representation in the k th hop of the KB specific GCN is computed as: DISPLAYFORM0 where U k dir (u,v) and z k dir (u,v) are edge direction specific KB-GCN weights and biases in k th hop and r 1 u = e u .

We also add inverse edges to E K similar to the case of syntactic GCNs in order to allow information flow in both the directions for an entity pair in the knowledge graph.

Dialogue History KB Entities Query DISPLAYFORM0

We use an RNN decoder to generate the tokens of the response and let the hidden states of the decoder be denoted as: DISPLAYFORM0 where T is the total number of decoder timesteps.

In order to obtain a single representation from the final layer (k = f ) of the query-GCN node vectors, we use an attention mechanism as described below: DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Here v 1 , W 1 , W 2 are parameters.

Further, at each decoder timestep, we obtain a query aware representation from the final layer of the history-GCN by computing an attention score for each node/token in the history based on the query context vector h Q t as shown below: DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 Here v 2 , W 3 , W 4 and W 5 are parameters.

Finally, we obtain a query and history aware representation of the KB by computing an attention score over all the nodes in the final layer of KB-GCN using h Q t and h H t as shown below: DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 Here v 3 , W 6 , W 7 , W 8 and W 9 are parameters.

This sequential attention mechanism is illustrated in FIG0 .

For simplicity, we depict the GCN and RNN+GCN encoders as blocks.

The internal structure of these blocks are shown in figure 1.

The decoder takes two inputs, viz., (i) the context which contains the history and the KB and (ii) the query which is the last/previous utterance in the dialogue.

We use an aggregator which learns the overall attention to be given to the history and KB components.

These attention scores: θ H t and θ K t are dependent on the respective context vectors and the previous decoder state d t−1 .

The final context vector is obtained as: DISPLAYFORM0 where [; ] denotes the concatenation operator.

At every timestep the decoder then computes a probability distribution over the vocabulary using the following equations: DISPLAYFORM1 where w t is the decoder input at time step t, V and b are parameters.

P vocab gives us a probability distribution over the entire vocabulary and the loss for time step t is l t = − log P vocab (w * t ), where w * t is the t th word in the ground truth response.

The total loss is an average of the per-time step losses.

For the dialogue history and query encoder, we used the dependency parse tree for capturing structural information in the encodings.

However, if the conversations occur in a language for which no dependency parsers exist, for example: code-mixed languages like Hinglish (Hindi-English) BID1 , then we need an alternate way of extracting a graph structure from the utterances.

One simple solution which worked well in practice was to create a word co-occurrence matrix from the entire corpus where the context window is an entire sentence.

Once we have such a co-occurrence matrix, for a given sentence we can connect an edge between two words if their co-occurrence frequency is above a threshold value.

The co-occurrence matrix can either contain co-occurrence frequency counts or positive-pointwise mutual information (PPMI) values BID6 BID7 BID27 .

In this section we describe the datasets used in our experiments, the various hyperparameters that we considered and the models that we compared.

The original DSTC2 dataset BID15 ) was based on the task of restaurant reservation and contains transcripts of real conversations between humans and bots.

The utterances were labeled with the dialogue state annotations like the semantic intent representation, requested slots and the constraints on the slot values.

We report our results on the modified DSTC2 dataset of BID3 where such annotations are removed and only the raw utterance-response pairs are present with an associated set of KB triples for each dialogue.

For our experiments with contextual graphs we reported our results on the code-mixed versions of modified DSTC2, which was recently released by BID1 1 .

This dataset has been collected by code-mixing the utterances of the English version of modified DSTC2 in four languages viz.

Hindi (Hi-DSTC2), Bengali (Be-DSTC2), Gujarati (Gu-DSTC2) and Tamil (Ta-DSTC2), via crowdsourcing.

Statistics about this dataset and example dialogues are shown in Appendix A.Model per-resp.

acc BLEU ROUGE Entity F1 1 2 L Rule-Based BID3 33.3 -----MEMNN BID3 41.1 -----QRN BID30 50.7 -----GMEMNN BID22 48.7 -----Seq2Seq-Attn BID0 46.0 57.3 67.2 56.0 64.9 67.1 Seq2Seq-Attn+Copy 47.3 55.4 ---71.6 HRED BID31 48.9 58.4 67.9 57.6 65.7 75.6 Mem2Seq BID23 45 Table 2 : Comparison of RNN+GCN-SeA, GCN-SeA with other models on all code-mixed datasets

We used the same train, test and validation splits as provided in the original versions of the datasets.

We minimized the cross entropy loss using the Adam optimizer BID18 and tuned the initial learning rates in the range of 0.0006 to 0.001.

For regularization we used an L2 norm of 0.001 in addition to a dropout BID32 of 0.1.

We used randomly initialized word embeddings of size 300.

The RNN and GCN hidden dimensions were also chosen to be 300.

We use GRU BID5 ) cells for the RNNs.

All parameters were initialized from a truncated normal distribution with a standard deviation of 0.1.

We compare the performance of the following models.(i) RNN+GCN-SeA vs GCN-SeA : We use RNN+GCN-SeA to refer to the model described in section 4.

Instead of using the hidden representations obtained from the bidirectional RNNs, we also experiment by providing the token embeddings directly to the GCNs i.e. c 1 u = q u in equation 6 and a 1 u = p u in equation 8.

We refer to this model as GCN-SeA.(ii) Cross edges between the GCNs: In addition to the dependency and contextual edges, we add edges between words in the dialogue history/query and KB entities if a history/query word exactly matches the KB entity.

Such edges create a single connected graph which is encoded using a single GCN encoder and then separated into different contexts to perform the sequential attention.

This model is referred to as RNN+CROSS-GCN-SeA.(iii) Frequency vs PPMI Contextual Graph : We experiment with the raw frequency cooccurrence graph structure and the PPMI graph structure for the code-mixed datasets, as explained in section 4.6.

We refer to these models as GCN-SeA+Freq and GCN-SeA+PPMI.

In both these models, the GCN takes inputs from a bidirectional RNN.(iv) GCN-SeA+Random vs GCN-SeA+Structure : We experiment with the model where the graph is constructed by randomly connecting edges between two words in a context.

We refer to this model as GCN-SeA+Random.

We refer to the model which either uses dependency or contextual graph instead of random graphs as GCN-SeA+Structure.

In this section we discuss the results of our experiments as summarized in tables 1,2, and 3.

We use BLEU BID28 and ROUGE BID21 metrics to evaluate the generation quality of responses.

We also report the per-response accuracy which computes the percentage of responses in which the generated response exactly matches the ground truth response.

In order to evaluate the model's capability of correctly injecting entities in the generated response, we report the entity F1 measure as defined in .Results on En-DSTC2 : We compare our model with the previous works on the English version of modified DSTC2 in table 1.

For most of the retrieval based models, the BLEU or ROUGE scores are not available as they select a candidate from a list of candidates as opposed to generating it.

Our model outperforms all of the retrieval and generation based models.

We obtain a gain of 0.7 in the per-response accuracy compared to the previous retrieval based state-of-the-art model of BID30 , which is a very strong baseline for our generation based model.

We call this a strong baseline because the candidate selection task of this model is easier than the response generation task of our model.

We also obtain a gain of 2.8 BLEU points, 2 ROUGE points and 2.5 entity F1 points compared to current state-of-the-art generation based models.

Results on code-mixed datasets and effect of using RNNs: The results of our experiments on the code-mixed datasets are reported in table 2.

Our model outperforms the baseline models on all the code-mixed languages.

One common observation from the results over all the languages (including En-DSTC2) is that RNN+GCN-SeA performs better than GCN-SeA. Similar observations were made by for the task of semantic role labeling.

Effect of using Hops: As we increased the number of hops of GCNs, we observed a decrease in the performance.

One reason for such a drop in performance could be that the average utterance length is very small (7.76 words).

Thus, there isn't much scope for capturing distant neighbourhood information and more hops can add noisy information.

Please refer to Appendix B for detailed results about the effect of varying the number of hops.

Frequency vs PPMI graphs: We observed that PPMI based contextual graphs were slightly better than frequency based contextual graphs (See Appendix C).

In particular, when using PPMI as opposed to frequency based contextual graph, we observed a gain of 0.95 in per-response accuracy, 0.45 in BLEU, 0.64 in ROUGE and 1.22 in entity F1 score when averaged across all the code-mixed languages.

Effect of using Random Graphs: GCN-SeA-Random and GCN-SeA-Structure take the token embeddings directly instead of passing them though an RNN.

This ensures that the difference in performance of the two models are not influenced by the RNN encodings.

The results are shown in table 3 and we observe a drop in performance for GCN-Random across all the languages.

This Table 3 : GCN-SeA with random graphs and frequency co-occurrence graphs on all DSTC2 datasets.shows that any random graph does not contribute to the performance gain of GCN-SeA and the dependency and contextual structures do play an important role.

Ablations : We experiment with replacing the sequential attention by the Bahdanau attention BID0 .

We also experiment with various combinations of RNNs and GCNs as encoders.

The results are shown in

We showed that structure aware representations are useful in goal-oriented dialogue and we obtain state-of-the art performance on the modified DSTC2 dataset and its recently released code-mixed versions.

We used GCNs to infuse structural information of dependency graphs and contextual graphs to enrich the representations of the dialogue context and KB.

We also proposed a sequential attention mechanism for combining the representations of (i) query (current utterance), (ii) conversation history and (ii) the KB.

Finally, we empirically showed that when dependency parsers are not available for certain languages such as code-mixed languages then we can use word co-occurrence frequencies and PPMI values to extract a contextual graph and use such a graph with GCNs for improved performance.

south part of town.

bot api call R cuisine south moderate api call R cuisine south moderate KB Triples: pizza hut cherry hinton R post code pizza hut cherry hinton post code pizza hut cherry hinton R cuisine italian pizza hut cherry hinton R location south pizza hut cherry hinton R phone pizza hut cherry hinton phone pizza hut cherry hinton R address pizza hut cherry hinton address pizza hut cherry hinton R price moderate pizza hut cherry hinton R rating 3 restaurant alimentum R post code restaurant alimentum post code restaurant alimentum R cuisine european restaurant alimentum R location south restaurant alimentum R phone restaurant alimentum phone restaurant alimentum R address restaurant alimentum address restaurant alimentum R price moderate restaurant alimentum R rating 10 user <SILENCE> <SILENCE> bot restaurant alimentum is a nice restaurant in the south of town serving modern european food.

restaurant alimentum ek bohot hi accha restaurant hai south town me jo modern european food serve karta hai user can i get the address ?

kya mujhe address mil sakta hai bot Sure, restaurant alimentum is on restaurant alimentum address.

Bilkul, restaurant alimentum restaurant alimentum address par hai user and whats the phone number for that ?

aur iska phone number kya hai ?

bot The phone number of restaurant alimentum is restaurant alimentum phone.restaurant alimentum ka phone number hai restaurant alimentum phone user okay, thank you, good bye. theek, thank you, good bye.

bot you are welcome.

welcome, alvida.

@highlight

We propose a Graph Convolutional Network based encoder-decoder model with sequential attention for goal-oriented dialogue systems.