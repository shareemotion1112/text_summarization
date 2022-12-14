Summarization of long sequences into a concise statement is a core problem in natural language processing, requiring non-trivial understanding of the input.

Based on the promising results of graph neural networks on highly structured data, we develop a framework to extend existing sequence encoders with a graph component that can reason about long-distance relationships in weakly structured data such as text.

In an extensive evaluation, we show that the resulting hybrid sequence-graph models outperform both pure sequence models as well as pure graph models on a range of summarization tasks.

Summarization, the task of condensing a large and complex input into a smaller representation that retains the core semantics of the input, is a classical task for natural language processing systems.

Automatic summarization requires a machine learning component to identify important entities and relationships between them, while ignoring redundancies and common concepts.

Current approaches to summarization are based on the sequence-to-sequence paradigm over the words of some text, with a sequence encoder -typically a recurrent neural network, but sometimes a 1D-CNN BID34 or using self-attention BID32 -processing the input and a sequence decoder generating the output.

Recent successful implementations of this paradigm have substantially improved performance by focusing on the decoder, extending it with an attention mechanism over the input sequence and copying facilities BID32 .

However, while standard encoders (e.g. bidirectional LSTMs) theoretically have the ability to handle arbitrary long-distance relationships, in practice they often fail to correctly handle long texts and are easily distracted by simple noise BID19 .In this work, we focus on an improvement of sequence encoders that is compatible with a wide range of decoder choices.

To mitigate the long-distance relationship problem, we draw inspiration from recent work on highly-structured objects BID23 BID20 BID14 BID12 .

In this line of work, highly-structured data such as entity relationships, molecules and programs is modelled using graphs.

Graph neural networks are then successfully applied to directly learn from these graph representations.

Here, we propose to extend this idea to weakly-structured data such as natural language.

Using existing tools, we can annotate (accepting some noise) such data with additional relationships (e.g. co-references) to obtain a graph.

However, the sequential aspect of the input data is still rich in meaning, and thus we propose a hybrid model in which a standard sequence encoder generates rich input for a graph neural network.

In our experiments, the resulting combination outperforms baselines that use pure sequence or pure graph-based representations.

Briefly, the contributions of our work are: 1.

A framework that extends standard sequence encoder models with a graph component that leverages additional structure in sequence data.

2.

Application of this extension to a range of existing sequence models and an extensive evaluation on three summarization tasks from the literature.

3.

We release all used code and data at https://github.com/CoderPat/structured-neural-summarization.

add a parameter to this dynamic parameter list BILSTM ??? LSTM: adds a new parameter to the specified parameter BILSTM+GNN ??? LSTM:creates a new instance of the dynamic type specified BILSTM+GNN ??? LSTM+POINTER: add a parameter to a list of parameters Figure 1 : An example from the dataset for the METHODDOC source code summarization task along with the outputs of a baseline and our models.

In the METHODNAMING dataset, this method appears as a sample requiring to predict the name Add as a subtoken sequence of length 1.

In this work, we consider three summarization tasks with different properties.

All tasks follow the common pattern of translating a long (structured) sequence into a shorter sequence while trying to preserve as much meaning as possible.

The first two tasks are related to the summarization of source code (Figure 1 ), which is highly structured and thus can profit most from models that can take advantage of this structure; the final task is a classical natural language task illustrating that hybrid sequence-graph models are applicable for less structured inputs as well.

METHODNAMING The aim of this task is to infer the name of a function (or method in objectoriented languages, such as Java, Python and C#) given its source code BID2 .

Although method names are a single token, they are usually composed of one or more subtokens (split using snake case or camelCase) and thus, the method naming task can be cast as predicting a sequence of subtokens.

Consequently, method names represent an "extreme" summary of the functionality of a given function (on average, the names in the Java dataset have only 2.9 subtokens).

Notably, the vocabulary of tokens used in names is very large (due to abbreviations and domainspecific jargon), but this is mitigated by the fact that 33% of subtokens in names can be copied directly from subtokens in the method's source code.

Finally, source code is highly structured input data with known semantics, which can be exploited to support name prediction.

METHODDOC Similar to the first task, the aim of this task is to predict a succinct description of the functionality of a method given its source code BID7 .

Such descriptions usually appear as documentation of methods (e.g. "docstrings" in Python or "JavaDocs" in Java).

While the task shares many characteristics with the METHODNAMING task, the target sequence is substantially longer (on average 19.1 tokens in our C# dataset) and only 19.4% of tokens in the documentation can be copied from the code.

While method documentation is nearer to standard natural language than method names, it mixes project-specific jargon, code segments and often describes non-functional aspects of the code, such as performance characteristics and design considerations.

NLSUMMARIZATION Finally, we consider the classic summarization of natural language as widely studied in NLP research.

Specifically, we are interested in abstractive summarization, where given some text input (e.g. a news article) a machine learning model produces a novel natural language summary.

Traditionally, NLP summarization methods treat text as a sequence of sentences and each one of them as a sequence of words (tokens).

The input data has less explicitly defined structure than our first two tasks.

However, we recast the task as a structured summarization problem by considering additional linguistic structure, including named entities and entity coreferences as inferred by existing NLP tools.

As discussed above, standard neural approaches to summarization follow the sequence-to-sequence framework.

In this setting, most decoders only require a representation h of the complete input sequence (e.g. the final state of an RNN) and per-token representations h ti for each input token t i .

These token representations are then used as the "memories" of an attention mechanism BID6 BID29 or a pointer network BID39 .In this work, we propose an extension of sequence encoders that allows us to leverage known (or inferred) relationships among elements in the input data.

To achieve that, we combine sequence encoders with graph neural networks (GNNs) BID23 BID14 BID20 .

For this, we first use a standard sequential encoder (e.g. bidirectional RNNs) to obtain a pertoken representation h ti , which we then feed into a GNN as the initial node representations.

The resulting per-node (i.e. per-token) representations h ti can then be used by an unmodified decoder.

Experimentally, we found this to surpass models that use either only the sequential structure or only the graph structure (see Sect.

4).

We now discuss the different parts of our model in detail.

Gated Graph Neural Networks To process graphs, we follow BID23 and briefly summarize the core concepts of GGNNs here.

A graph G = (V, E, X) is composed of a set of nodes V, node features X, and a list of directed edge sets E = (E 1 , . . .

, E K ) where K is the number of edge types.

Each v ??? V is associated with a real-valued vector x v representing the features of the node (e.g., the embedding of a string label of that node), which is used for the initial state h (0) v of a node.

Information is propagated through the graph using neural message passing BID14 .

For this, every node v sends messages to its neighbors by transforming its current representation h (i) v using an edge-type dependent function f k .

Here, f k can be an arbitrary function; we use a simple linear layer.

By computing all messages at the same time, all states can be updated simultaneously.

In particular, a new state for a node v is computed by aggregating all incoming messages as m DISPLAYFORM0 | there is an edge of type k from u to v}).

g is an aggregation function; we use elementwise summation for g. Given the aggregated message m DISPLAYFORM1 , where GRU is the recurrent cell function of a gated recurrent unit.

These dynamics are rolled out for a fixed number of timesteps T , and the state vectors resulting from the final step are used as output node representations, i.e., GNN((V, E, X)) = {h DISPLAYFORM2 Sequence GNNs We now explain our novel combination of GGNNs and standard sequence encoders.

As input, we take a sequence S = [s 1 . . .

s N ] and K binary relationships R 1 . . .

R K ??? S ?? S between elements of the sequence.

For example, R = could be the equality relationship {(s i , s j ) | s i = s j }.

The choice and construction of relationships is dataset-dependent, and will be discussed in detail in Sect.

4.

Given any sequence encoder SE that maps S to per-element representations [e 1 . . .

e N ] and a sequence representation e (e.g. a bidirectional RNN), we can construct the sequence GNN SE GN N by simply computing [e 1 . . .

DISPLAYFORM3 ).

To obtain a graph-level representation, we use the weighted averaging mechanism from BID14 .

Concretely, for each node v in the graph, we compute a weight ??(w(h (T ) v )) ??? [0, 1] using a learnable function w and the logistic sigmoid ?? and compute a graph-level representation a?? e = 1???i???N ??(w(e i )) ?? ???(e i ), where ??? is another learnable projection function.

We found that best results were achieved by computing the final e as W ?? (e??) for some learnable matrix W .This method can easily be extended to support additional nodes not present in the original sequence S after running SE (e.g., to accommodate meta-nodes representing sentences, or non-terminal nodes from a syntax tree).

The initial node representation for these additional nodes can come from other sources, such as a simple embedding of their label.

Implementation Details.

Processing large graphs of different shapes efficiently requires to overcome some engineering challenges.

For example, the CNN/DM corpus has (on average) about 900 nodes per graph.

To allow efficient computation, we use the trick of where all graphs in a minibatch are "flattened" into a single graph with multiple disconnected components.

The varying graph sizes also represent a problem for the attention and copying mechanisms in the decoder, as they require to compute a softmax over a variable-sized list of memories.

To handle this efficiently without padding, we associate each node in the (flattened) "batch" graph with the index of the sample in the minibatch from which the node originated.

Then, using TensorFlow's unsorted segment * operations, we can perform an efficient and numerically stable softmax over the variable number of representations of the nodes of each graph.

We evaluate Sequence GNNs on our three tasks by comparing them to models that use only sequence or graph information, as well as by comparing them to task-specific baselines.

We discuss the three tasks, their respective baselines and how we present the data to the models (including the relationships considered in the graph component) next before analyzing the results.

Datasets, Metrics, and Models.

We consider two datasets for the METHODNAMING task.

First, we consider the "Java (small)" dataset of BID4 , re-using the train-validation-test splits they have picked.

We additionally generated a new dataset from 23 open-source C# projects mined from GitHub (see below for the reasons for this second dataset), removing any duplicates.

More information about these datasets can be found in Appendix C. We follow earlier work on METHOD-NAMING BID2 BID4 and measure performance using the F1 score over the generated subtokens.

However, since the task can be viewed as a form of (extreme) summarization, we also report ROUGE-2 and ROUGE-L scores BID25 , which we believe to be additional useful indicators for the quality of results.

ROUGE-1 is omitted since it is equivalent to F1 score.

We note that there is no widely accepted metric for this task and further work identifying the most appropriate metric is required.

We compare to the current state of the art BID4 , as well as a sequence-to-sequence implementation from the OpenNMT project (Klein et al.) .

Concretely, we combine two encoders (a bidirectional LSTM encoder with 1 layer and 256 hidden units, and its sequence GNN extension with 128 hidden units unrolled over 8 timesteps) with two decoders (an LSTM decoder with 1 layer and 256 hidden units with attention over the input sequence, and an extension using a pointer network-style copying mechanism BID39 ).

Additionally, we consider self-attention as an alternative to RNN-based sequence encoding architectures.

For this, we use the Transformer BID38 implementation in OpenNMT (i.e., using self-attention both for the decoder and the encoder) as a baseline and compare it to a version whose encoder is extended with a GNN component.

Data Representation Following the work of BID2 ; BID4 , we break up all identifier tokens (i.e. variables, methods, classes, etc.) in the source code into subtokens by splitting them according to camelCase and pascal case heuristics.

This allows the models to extract information from the information-rich subtoken structure, and ensures that a copying mechanism in the decoder can directly copy relevant subtokens, something that we found to be very effective for this task.

All models are provided with all (sub)tokens belonging to the source code of a method, including its declaration, with the actual method name replaced by a placeholder symbol.

To construct a graph from the (sub)tokens, we implement a simplified form of the work of .

First, we introduce additional nodes for each (full) identifier token, and connect the constituent subtokens appearing in the input sequence using a INTOKEN edge; we additionally connect these nodes using a NEXTTOKEN edge.

We also add nodes for the parse tree and use edges to indicate that one node is a CHILD of another.

Finally, we add LASTLEXICALUSE edges to connect identifiers to their most (lexically) recent use in the source code.

Datasets, Metrics, and Models.

We tried to evaluate on the Python dataset of BID7 of the dataset and were only able to reach comparable results by substantially overfitting to the training data that overlapped with the test set.

We have documented details in subsection C.3 and in BID0 , and decided to instead evaluate on our new dataset of 23 open-source C# projects from above, again removing duplicates and methods without documentation.

Following Barone & Sennrich (2017) , we measure the BLEU score for all models.

However, we also report F1, ROUGE-2 and ROUGE-L scores, which should better reflect the summarization aspect of the task.

We consider the same models as for the METHODNAMING task, using the same configuration, and use the same data representation.

Datasets, Metrics, and Models.

We use the CNN/DM dataset BID17 using the exact data and split provided by .

The data is constructed from CNN and Daily Mail news articles along with a few sentences that summarize each article.

To measure performance, we use the standard ROUGE metrics.

We compare our model with the near-to-state-of-the-art work of , who use a sequence-to-sequence model with attention and copying as basis, but have additionally substantially improved the decoder component.

As our contribution is entirely on the encoder side and our model uses a standard sequence decoder, we are not expecting to outperform more recent models that introduce substantial novelty in the structure or training objective of the decoder BID10 BID34 .

Again, we evaluate our contribution using an OpenNMT-based encoder/decoder combination.

Concretely, we use a bidirectional LSTM encoder with 1 layer and 256 hidden units, and its sequence GNN extension with 128 hidden units unrolled over 8 timesteps.

As decoder, we use an LSTM with 1 layer and 256 hidden units with attention over the input sequence, and an extension using a pointer network-style copying mechanism.

Data Representation We use Stanford CoreNLP BID30 (version 3.9.1) to tokenize the text and provide the resulting tokens to the encoder.

For the graph construction FIG2 , we extract the named entities and run coreference resolution using CoreNLP.

We connect tokens using a NEXT edge and introduce additional super-nodes for each sentence, connecting each token to the corresponding sentence-node using a IN edge.

We also connect subsequent sentence-nodes using a NEXT edge.

Then, for each multi-token named entity we create a new node, labeling it with the type of the entity and connecting it with all tokens referring to that entity using an IN edge.

Finally, coreferences of entities are connected with a special REF edge.

FIG2 shows a partial graph for an article in the CNN/DM dataset.

The goal of this graph construction process is to explicitly annotate important relationships that can be useful for summarization.

We note that (a) in early efforts we experimented with adding dependency parse edges, but found that they do not provide significant benefits and (b) that since we retrieve the annotations from CoreNLP, they can contain errors and thus, the performance of the our method is influenced by the accuracy of the upstream annotators of named entities and coreferences.

We show all results in Tab.

1. Results for models from the literature are taken from the respective papers and repeated here.

Across all tasks, the results show the advantage of our hybrid sequence GNN encoders over pure sequence encoders.

On METHODNAMING, we can see that all GNN-augmented models are able to outperform the current specialized state of the art, requiring only simple graph structure that can easily be obtained using existing parsers for a programming language.

The results in performance between the different encoder and decoder configurations nicely show that their effects are largely orthogonal.

On METHODDOC, the unmodified SELFATT ??? SELFATT model already performs quite well, and the augmentation with graph data only improves the BLEU score and worsens the results on ROUGE.

Inspection of the results shows that this is due to the length of predictions.

Whereas the ground truth data has on average 19 tokens in each result, SELFATT ??? SELFATT predicts on average 11 tokens, and SELFATT+GNN ??? SELFATT 16 tokens.

Additionally, we experimented with an ablation in which a model is only using graph information, e.g., a setting comparable to a simplification of the architecture of .

For this, we configured the GNN to use 128-dimensional representations and unrolled it for 10 timesteps, keeping the decoder configuration as for the other models.

The results indicate that this configuration performs less well than a pure sequenced model.

We speculate that this is mainly due to the fact that 10 timesteps are insufficient to propagate infor- Finally, on NLSUMMARIZATION, our experiments show that the same model suitable for tasks on highly structured code is competitive with specialized models for natural language tasks.

While there is still a gap to the best configuration of (and an even larger one to more recent work in the area), we believe that this is entirely due to our simplistic decoder and training objective, and that our contribution can be combined with these advances.

In TAB1 we show some ablations for NLSUMMARIZATION.

As we use the same hyperparameters across all datasets and tasks, we additionally perform an experiment with the model of (as implemented in OpenNMT) but using our settings.

The results achieved by these baselines trend to be a bit worse than the results reported in the original paper, which we believe is due to a lack of hyperparameter optimization for this task.

We then evaluated how much the additional linguistic structure provided by CoreNLP helps.

First, we add the coreference and entity annotations to the baseline BILSTM ??? LSTM + POINTER model (by extending the embedding of tokens with an embedding of the entity information, and inserting fresh "??REF1??", . . .

tokens at the sources/targets of co-references) and observe only minimal improvements.

This suggests that our graph-based encoder is better-suited to exploit additional structured information compared to a biLSTM encoder.

We then drop all linguistic structure information from our model, keeping only the sentence edges/nodes.

This still improves on the baseline BILSTM ??? LSTM + POINTER model (in the ROUGE-2 score), suggesting that the GNN still yields improvements in the absence of linguistic structure.

Finally, we add long-range dependency edges by connecting tokens with equivalent string representations of their stems and observe further minor improvements, indicating that even using only purely syntactical information, without a semantic parse, can already provide gains.

We look at a few sample suggestions in our dataset across the tasks.

Here we highlight some observations we make that point out interesting aspects and failure cases of our model.

Input:

Arsenal , Newcastle United and Southampton have checked on Caen midfielder N'golo Kante .

Parisborn Kante is a defensive minded player who has impressed for Caen this season and they are willing to sell for around ?? 5million .

Marseille have been in constant contact with Caen over signing the 24-year-old who has similarities with Lassana Diarra and Claude Makelele in terms of stature and style .

N'Golo Kante is attracting interest from a host of Premier League clubs including Arsenal .

Caen would be willing to sell Kante for around ?? 5million .Reference: n'golo kante is wanted by arsenal , newcastle and southampton .

marseille are also keen on the ?? 5m rated midfielder .

kante has been compared to lassana diarra and claude makelele .

click here for the latest premier league news .

METHODDOC FIG3 illustrate typical results of baselines and our model on the METHODDOC task (see Appendix A for more examples).

The hardness of the task stems from the large number of distractors and the need to identify the most relevant parts of the input.

In Figure 1 , the token "parameter" and variations appears many times, and identifying the correct relationship is non-trivial, but is evidently eased by graph edges explicitly denoting these relationships.

Similarly, in FIG3 , many variables are passed around, and the semantics of the method require understanding how information flows between them.

FIG5 shows one sample summarization.

More samples for this task can be found in Appendix B. First, we notice that the model produces natural-looking summaries with no noticeable negative impact on the fluency of the language over existing methods.

Furthermore, the GNN-based model seems to capture the central named entity in the article and creates a summary centered around that entity.

We hypothesize that the GNN component that links long-distance relationships helps capture and maintain a better "global" view of the article, allowing for better identification of central entities.

Our model still suffers from repetition of information (see Appendix B), and so we believe that our model would also profit from advances such as taking coverage into account or optimizing for ROUGE-L scores directly via reinforcement learning BID10 BID34 .

Natural language processing research has studied summarization for a long time.

Most related is work on abstractive summarization, in which the core content of a given text (usually a news article) is summarized in a novel and concise sentence.

BID11 and BID33 use deep learning models with attention on the input text to guide a decoder that generates a summary.

and BID32 extend this idea with pointer networks BID39 to allow for copying tokens from the input text to the output summary.

These approaches treat text as a simple token sequences, not explicitly exposing additional structure.

In principle, deep sequence networks are known to be able to learn the inherent structure of natural language (e.g. in parsing BID40 and entity recognition BID22 ), but our experiments indicate that explicitly exposing this structure by separating concerns improves performance.

Recent work in summarization has proposed improved training objectives for summarization, such as tracking coverage of the input document or using reinforcement learning to directly identify actions in the decoder that improve target measures such as ROUGE-L BID10 BID34 .

These objectives are orthogonal to the graph-augmented encoder discussed in this work, and we are interested in combining these efforts in future work.

Exposing more language structure explicitly has been studied over the last years, with a focus on tree-based models BID37 .

Very recently, first uses of graphs in natural language processing have been explored.

BID31 use graph convolutional networks to encode single sentences and assist machine translation.

De Cao et al. (2018) create a graph over named entities over a set of documents to assist question answering.

Closer to our work is the work of BID26 , who use abstract meaning representation (AMR), in which the source document is first parsed into AMR graphs, before a summary graph is created, which is finally rendered in natural language.

In contrast to that work we do not use AMRs but directly encode relatively simple relationships directly on the tokenized text, and do not treat summarization as a graph rewrite problem.

Combining our encoder with AMRs to use richer graph structures may be a promising future direction.

Finally, summarization in source code has also been studied in the forms of method naming, comment and documentation prediction.

Method naming has been tackled with a series of models.

For example, BID1 use a log-bilinear network to predict method names from features, and later extend this idea to use a convolutional attention network over the tokens of a method to predict the subtokens of names BID2 .

BID35 and BID8 use CRFs for a range of tasks on source code, including the inference of names for variables and methods.

Recently, BID5 a) extract and encode paths from the syntax tree of a program, setting the state of the art in accuracy on method naming.

Linking text to code can have useful applications, such as code search BID15 , traceability BID16 , and detection of redundant method comments BID28 .

Most approaches on source code either treat it as natural language (i.e., a token sequence), or use a language parser to explicitly expose its tree structure.

For example, Barone & Sennrich (2017) use a simple sequenceto-sequence baseline, whereas BID18 summarize source code by linearizing the abstract syntax tree of the code and using a sequence-to-sequence model.

Wan et al. (2018) instead directly operate on the tree structure using tree recurrent neural networks BID37 .

The use of additional structure on related tasks on source code has been studied recently, for example in models that are conditioned on learned traversals of the syntax tree BID9 and in graph-based approaches BID12 .

However, as noted by BID24 , GNN-based approaches suffer from a tension between the ability to propagate information across large distances in a graph and the computational expense of the propagation function, which is linear in the number of graph edges per propagation step.

We presented a framework for extending sequence encoders with a graph component that can leverage rich additional structure.

In an evaluation on three different summarization tasks, we have shown that this augmentation improves the performance of a range of different sequence models across all tasks.

We are excited about this initial progress and look forward to deeper integration of mixed sequence-graph modeling in a wide range of tasks across both formal and natural languages.

The key insight, which we believe to be widely applicable, is that inductive biases induced by explicit relationship modeling are a simple way to boost the practical performance of existing deep learning systems.

We use the datasets and splits of BID4 provided by their website.

Upon scanning all methods in the dataset, the size of the corpora can be seen in Table 4 .

More information can be found at BID4 .

We use the dataset as split of BID7 provided by their GitHub repository.

Upon parsing the dataset, we get 106,065 training samples, 1,943 validation samples and 1,937 test samples.

We note that 16.9% of the documentation samples in the validation set and 15.3% of the samples in test set have a sample with the identical natural language documentation on the training set.

This Table 4 : The statistics of the extracted graphs from the Java method naming dataset of BID4

Below we present the data characteristics of the graphs we use across the datasets.

@highlight

One simple trick to improve sequence models: Compose them with a graph model

@highlight

This paper presents a structural summarization model with a graph-based encoder extended from RNN.

@highlight

This work combines Graph Neural Networks with a sequential approach to abstractive summarization, effective across all datasets in comparison to external baselines.