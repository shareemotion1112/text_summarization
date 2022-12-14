Real-world Question Answering (QA) tasks consist of thousands of words that often represent many facts and entities.

Existing models based on LSTMs require a large number of parameters to support external memory and do not generalize well for long sequence inputs.

Memory networks attempt to address these limitations by storing information to an external memory module but must examine all inputs in the memory.

Hence, for longer sequence inputs the intermediate memory components proportionally scale in size resulting in poor inference times and high computation costs.



In this paper, we present Adaptive Memory Networks (AMN) that process input question pairs to dynamically construct a network architecture optimized for lower inference times.

During inference, AMN parses input text into entities within different memory slots.

However, distinct from previous approaches, AMN is a dynamic network architecture that creates variable numbers of memory banks weighted by question relevance.

Thus, the decoder can select a variable number of memory banks to construct an answer using fewer banks, creating a runtime trade-off between accuracy and speed.



AMN is enabled by first, a novel bank controller that makes discrete decisions with high accuracy and second, the capabilities of a dynamic framework (such as PyTorch) that allow for dynamic network sizing and efficient variable mini-batching.

In our results, we demonstrate that our model learns to construct a varying number of memory banks based on task complexity and achieves faster inference times for standard bAbI tasks, and modified bAbI tasks.

We achieve state of the art accuracy over these tasks with an average 48% lower entities are examined during inference.

Question Answering (QA) tasks are gaining significance due to their widespread applicability to recent commercial applications such as chatbots, voice assistants and even medical diagnosis BID7 ).

Furthermore, many existing natural language tasks can also be re-phrased as QA tasks.

Providing faster inference times for QA tasks is crucial.

Consumer device based question-answer services have hard timeouts for answering questions.

For example, Amazon Alexa, a popular QA voice assistant, allows developers to extend the QA capabilities by adding new "Skills" as remote services BID0 ).

However, these service APIs are wrapped around hard-timeouts of 8 seconds which includes the time to transliterate the question to text on Amazon's servers and the round-trip transfer time of question and the answer from the remote service, and sending the response back to the device.

Furthermore, developers are encouraged to provide a list of questions ("utterances") apriori at each processing step to assist QA processing BID0 ).Modeling QA tasks with LSTMs can be computationally expensive which is undesirable especially during inference.

Memory networks, a class of deep networks with explicit addressable memory, have recently been used to achieve state of the art results on many QA tasks.

Unlike LSTMs, where the number of parameters grows exponentially with the size of memory, memory networks are comparably parameter efficient and can learn over longer input sequences.

However, they often require accessing all intermediate memory to answer a question.

Furthermore, using focus of attention over the intermediate state using a list of questions does not address this problem.

Soft attention based models compute a softmax over all states and hard attention models are not differentiable and can be difficult to train over a large state space.

Previous work on improving inference over memory networks has focused on using unsupervised clustering methods to reduce the search space BID2 ; BID19 ).

Here, the memory importance is not learned and the performance of nearest-neighbor style algorithms is often comparable to a softmax operation over memories.

To provide faster inference for long sequence-based inputs, we present Adaptive Memory Networks (AMN), that constructs a memory network on-the-fly based on the input.

Like past approaches to addressing external memory, AMN constructs the memory nodes dynamically.

However, distinct from past approaches, AMN constructs a memory architecture with network properties that are decided dynamically based on the input story.

Given a list of possible questions, our model computes and stores the entities from the input story in a memory bank.

The entities represent the hidden state of each word in the story while a memory bank is a collection of entities that are similar w.r.t the question.

As the number of entities grow, our network learns to construct new memory banks and copies entities that are more relevant towards a single bank.

Entities may reside in different bank depending on their distance from the question.

Hence, by limiting the decoding step to a dynamic number of constructed memory banks, AMN achieves lower inference times.

AMN is an end-to-end trained model with dynamic learned parameters for memory bank creation and movement of entities.

Figure 1 demonstrates a simple QA task where AMN constructs two memory banks based on the input.

During inference only the entities in the left bank are considered reducing inference times.

To realize its goals, AMN introduces a novel bank controller that uses reparameterization trick to make discrete decisions with high accuracy while maintaining differentiability.

Finally, AMN also models sentence structures on-the-fly and propagates update information for all entities that allows it to solve all 20 bAbI tasks.

Memory Networks: Memory networks store the entire input sequence in memory and perform a softmax over hidden states to update the controller BID27 ; BID23 ).

DMN+ connects memory to input tokens and updates them sequentially BID29 ).

For inputs that consist of large number of tokens or entities, these methods can be expensive during inference.

AMN stores entities with tied weights in different memory banks.

By controlling the number of memory banks, AMN achieves low inference times with reasonable accuracy.

Nearest neighbor methods have also been explored over memory networks.

For example, Hierarchical Memory Networks separates the input memory into groups using the MIPS algorithm BID2 ) .

However, using MIPS is as slow as a softmax operation, so the authors propose using an approximate MIPS that gives inferior performance.

In contrast, AMN is end to end differentiable, and reasons which entities are important and constructs a network with dynamic depth.

Neural Turing Machine (NTM) consists of a memory bank and a differentiable controller that learns to read and write to specific locations BID9 ).

In contrast to NTMs, AMN memory bank controller is more coarse grained and the network learns to store entities in memory banks instead of specific locations.

AMN uses a discrete bank controller that gives improved performance for bank controller actions over NTM's mechanisms.

However, like NTMs, our design is consistent with the modeling studies of working memory by BID11 ) where the brain performs robust memory maintenance and may maintain multiple working representations for individual working tasks.

Sparse access memory uses approximate nearest neighbors (ANN) to reduce memory usage in NTMs BID19 ).

However, ANNs are not differentiable.

AMN, uses a input specific memory organization that does not create sparse structures.

This limits access during inference to specific entities reducing inference times.

Graph-based networks, (GG-NNs, BID16 and GGT-NNs, Johnson (2017) ) use nodes with tied weights that are updated based on gated-graph state updates with shared weights over edges.

However, unlike AMN, they require strong supervision over the input and teacher forcing to learn the graph structure.

Furthermore, the cost of building and training these models is expensive and if every edge is considered at every time-step the amount of computation grows at the order of O(N 3 ) where N represents the number of nodes/entities.

AMN does not use strong supervision but can solve tasks that require transitive logic by modeling sentence walks on the fly.

EntNet constructs dynamic networks based on entities with tied weights for each entity BID12 ).

A key-value update system allows it to update relevant (learned) entities.

However, Entnet uses soft-attention during inference to attend to all entities that incur high inference costs.

To summarize, majority of the past work on memory networks uses softmax over memory nodes, where each node may represent input or an entity.

In contrast, AMN learns to organize memory into various memory banks and performs decode over fewer entities reducing inference times.

Conditional Computation & Efficient Inference: AMN is also related to the work on conditional computation which allows part of networks to be active during inference improving computational efficiency BID1 ).

Recently, this has been often accomplished using a gated mixture of experts BID4 ; BID22 ).

AMN conditionally attends to entities in initial banks during inference improving performance.

For faster inference using CNNs, pruning BID15 ; BID10 ), low rank approximations BID3 ), quantization and binarization BID20 ) and other tricks to improve GEMM performance BID25 ) have been explored.

For sequence based inputs, pruning and compression has been explored BID6 ; BID21 ).

However, compression results in irregular sparsity that reduces memory costs but may not reduce computation costs.

Adaptive computation time BID8 ) learns the number of steps required for inferring the output and this can also be used to reduce inference times BID5 ).

AMN uses memory networks with dynamic number of banks to reduce computation costs.

Dynamic networks: Dynamic neural networks that change structure during inference have recently been possible due to newer frameworks such as Dynet and PyTorch.

Existing work on pruning can be implemented using these frameworks to reduce inference times dynamically like dynamic deep network demonstrates BID17 ).

AMN utilizes the dynamic architecture abilities to construct an input dependent memory network of variable memory bank depth and the dynamic batching feature to process a variable number of entities.

Furthermore, unlike past work that requires an apriori number of fixed memory slots, AMN constructs them on-the-fly based on the input.

The learnable discrete decision-making process can be extended to other dynamic networks which often rely on REINFORCE to make such decisions BID17 ).Neuroscience: Our network construction is inspired by work on working memory representations.

There is sufficient evidence for multiple, working memory representations in the human brain (Hazy et al. FORMULA0 ).

Semantic memory BID24 ), describes a hierarchical organization starting with relevant facts at the lowest level and progressively more complex and distant concepts at higher levels.

AMN constructs entities from the input stories and stores the most relevant entities based on the question in the lowest level memory bank.

Progressively higher level memory banks represent distant concepts (and not necessarily higher level concepts for AMN).

Other work demonstrates organization of human memory in terms of "priority structure" where attention is a gate-keeper of working memory-guided by executive control's goals, plans, and intentions as in BID26 , similar in spirit to AMN's question guided network construction.

In this section, we describe the design process and motivation of our memory module.

Our memory network architecture is created during inference time for every story.

The architecture consists of different memory banks and each memory bank stores entities from the input story.

Hence, a memory entity represents the hidden state of each entity (each word in our case) from the input story while a memory bank is a collection of entities.

Intuitively, each memory bank stores entities that have a similar distance score from the question.

At a high level, entities are gradually and recurrently copied through memory banks to filter out irrelevant nodes such that in the final inference stage, fewer entities are considered by the decoder.

Note that the word filter implies a discrete decision and that recurrence implies time.

If we were to perform a strict cut off and remove entities that appear to be irrelevant at each time step, learning the reasoning logic that requires previous entities that were cut off would not be possible.

Thus, smoothed discretization is required.

We design filtering to be a two-stage pseudo-continuous process to simulate discrete cut offs (?? move , ?? new ), while keeping reference history.

The overall memory (M ) consists of multiple memory banks.

A memory bank is a collection or group of entities (m 0...l ), where m 0 denotes the initial and most general bank and m l denotes the most relevant bank.

Note that |l| is input dependent and learned.

First, entities are moved from m 0 gradually towards m l based off of their individual relevance to the question and second, if m l becomes too saturated, m l+1 is created.

Operations in the external memory allowing for such dynamic restructuring and entity updates are described below.

Note that these operations still maintain end to end differentiability.1.

Memory bank creation (?? new ), which creates a new memory bank depending on the current states of entities m i .

If the entropy, or information contained (explained below), of m i is too high, ?? new (m i ) will learn to create a new memory bank m i+1 to reduce entropy.2.

Moving entities across banks (?? move ), which determines which entities are relevant to the current question and move such entities to further (higher importance) memory banks.3.

Adding/Updating entities in a bank (?? au ), which adds entities that are not yet encountered to the first memory bank m 0 or if the entity is already in m 0 , the operation updates the entity state.4.

Propagating changes across entities (?? prop ), which updates the entity states in memory banks based on node current states ?? prop (M ) and their semantic relationships.

This is to communicate transitive logic.

Both ?? new , ?? move require a discrete decision (refer to section 4.2.1.), and in particular, for ?? new we introduce the notion of entropy.

That is to say if m i contains too many nodes (the entropy becomes too high), the memory module will learn to create a new bank m i+1 and move nodes to m i+1 to reduce entropy.

By creating more memory banks, the model spreads out the concentration of information which in turn better discretizes nodes according to relevance.

A high-level overview is shown in FIG1 , followed by a mathematical detail of the model's modules.

Our model adopts the encoder-decoder framework with an augmented adaptive memory module.

For an overview of the algorithm, refer to Section A.1.Notation and Problem Statement:

Given a story represented by N input sentences (or statements), i.e., (l 1 , ?? ?? ?? , l N ), and a question q, our goal is to generate an answer a. Each sentence l is a sequence of N words, denoted as (w 1 , ?? ?? ?? , w Ns ), and a question is a sequence of N q words denoted as (w 1 , ?? ?? ?? , w Nq ).

Throughout the model we refer to entities; these can be interpreted as a 3-tuple of e w = (word ID wi, hidden state w, question relevance strength s).

Scalars, vectors, matrices, and dot products are denoted by lower-case letters, boldface lower-case letters and boldface capital letters, and angled brackets respectively.

The input to the model, starting with the encoder, are story-question input pairs.

On a macro level, sentences l 1...

N are processed.

On a micro level, words w 1...

Ns are processed within sentences.

For each w i ??? l i , the encoder maps w i to a hidden representation and a question relevance strength ??? [0, 1].

The word ID of w i is passed through a standard embedding layer and then encoded through an accumulation GRU.

The accumulation GRU captures the entity states through time by adding the output of each GRU time step to its respective word, stored in a lookup matrix.

The initial states of e w are set to this GRU output.

Meanwhile, the question is also embedded and encoded in the same manner sans accumulation.

In the following, the subscripts i, j are used to iterate through the total number of words in a statement and question respectively, D stores the accumulation GRU output, and w i is a GRU encoding output.

The last output of the GRU will be referred to as w N , w Nq for statements and questions.

DISPLAYFORM0 DISPLAYFORM1 To compute the question relevance strength s ??? [0, 1] for each word, the model uses GRU-like equations.

The node strengths are first initialized to Xavier normal and the inputs are the current word states w in , the question state w Nq , and when applicable, the previous strength.

Sentences are processed each time step t. DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 In particular, equation FORMULA2 shows where the model learns to lower the strengths of nodes that are not related the question.

First, a dot product between the current word states and question state are computed for similarity (high correlation), then it is subtracted from a 1 to obtain the dissimilarity.

We refer to these operations as SGRU (Strength GRU) in Algorithm 1.

The adaptive memory module recurrently restructures entities in a question relevant manner so the decoder can then consider fewer entities (namely, the question relevant entities) to generate an answer.

The following operations are performed once per sentence.

As mentioned earlier, discrete decisions are difficult for neural networks to learn so we designed a specific memory bank controller ?? ctrl for binary decision making.

The model takes ideas from the reparameterization trick and uses custom backpropagation to maintain differentiability.

In particular, the adaptive memory module needs to make two discrete decisions on a {0, 1} basis, one in ?? new to create a new memory bank and the other in ?? move to move nodes to a different memory bank.

The model uses a scalar p ??? {0, 1} to parameterize a Bernoulli distribution where the realization H, is the decision the model makes.

However, backpropagation through a random node is intractable, so the model detaches H from the computation graph and introduces H as a new node.

Finally, H is used as a mask to zero out entities in the discrete decision.

Meanwhile, p is kept in the computation graph and has a special computed loss (Section 4.4).

The operations below will be denoted as ?? ctrl and has two instances: one for memory bank creation ?? new and one for moving entities across banks ?? move .

In equation 9, depending on what ?? ctrl is used for, q is a polymorphic function and will take on a different operation and * will be a different input.

Examples of such are given in their respective sections (4.2.2.1, 4.2.2.2).

DISPLAYFORM0 4.2.2 MEMORY BANK OPERATIONS 1.

Memory bank creation ?? new : To determine when a new memory bank is created, in other words, if the current memory bank becomes too saturated, the memory bank controller (4.2.1.) will make a discrete decision to create a new memory bank.

Here, q (eq 9) is a fully connected layer and the input is the concatenation of all the current memory bank m i 's entity states [w 0 ...w i ] ??? R 1,n|ew| .

Intuitively, q will learn a continuous decision that is later discretized by eq 10 based on entity states and the number of entities.

Note this is only performed for the last memory bank.

DISPLAYFORM1 2.

Moving entities through memory banks: Similar to ?? new , individual entities' relevance scores are passed into the bank controller to determine H as the input.

The relevance score is computed by multiplying an entity state by its respective relevance ??? R n,|ew| .

Here, q has a slight modification and is the identity function.

Note that this operation can only be performed if there is a memory bank to move nodes to, namely if m i+1 exists.

Additionally, each bank has a set property where it cannot contain duplicate nodes, but the same node can exist in two different memory banks.

DISPLAYFORM2 3.

Adding/Updating entities in a bank: Recall that entities are initially set to the output of D. However, as additional sentences are processed, new entities and their hidden states are observed.

In the case of a new entity e w , the entity is added to the first memory bank m 0 .

If the entity already exists in m 0 , then e w 's corresponding hidden state is updated through a GRU.

This procedure is done for all memory banks.

DISPLAYFORM3 4.

Propagating updates to related entities: So far, entities exist as a bag of words model and the sentence structure is not maintained.

This can make it difficult to solve tasks that require transitive reasoning over multiple entities.

To track sentence structure information, we model semantic relationships as a directed graph stored in adjacency matrix A. As sentences are processed word by word, a directed graph is drawn progressively from w 0 ...

w i ...w N .

If sentence l k 's path contains nodes already in the current directed graph, l k will include said nodes in its path.

After l k is added to A, the model propagates the new update hidden state information a i among all node states using a GRU.

a i for each node i is equal to the sum of the incoming edges' node hidden states.

Additionally, we add a particular emphasis on l k to simulate recency.

At face value, one propagation step of A will only have a reachability of its immediate neighbor, so to reach all nodes, A is raised to a consecutive power r to reach and update each intermediate node.

r can be either the longest path in A or a set parameter.

Again, this is done within a memory bank for all memory banks.

For entities that have migrated to another bank, the update for these entities is a no-op but propagation information as per the sentence structure is maintained.

A single iteration is shown below: DISPLAYFORM4 When nodes are transferred across banks, A is still preserved.

If intermediate nodes are removed from a path, a transitive closure is drawn if possible.

After these steps are finished at the end of a sentence, namely, the memory unit has reasoned through how large (number of memory banks) the memory should be and which entities are relevant at the current point in the story, all entities are passed through the strength modified GRU (4.1, eq 5-8) to recompute their question relevance (relevance score).

After all sentences l 1...N are ingested, the decode portion of the network learns to interpret the results from the memory banks.

The network iterates through the memory banks using a standard attention mechanism.

To force the network to understand the question importance weighting, the model uses an exponential function d to weight important memory banks higher.

C m are the hidden states contained in memory m, s m are the relevance strengths of memory bank m, w Nq is the question hidden state, ps is the attention score, r, h are learned weight masks, g are the accumulated states, and l is the final logits prediction.

During inference, fewer memory banks are considered.

DISPLAYFORM0

Loss is comprised of two parts, answer loss, which is computed from the given annotations, and secondary loss (from ?? new , ?? move ), which is computed from sentence and story features at each sentence time step l 0...N .

Answer loss is standard cross entropy at the end of the story after l N is processed.

DISPLAYFORM0 After each sentence l i , the node relevance s li is enforced by computing the expected relevance E[s li ].

E[s] is determined by nodes that are connected to the answer node a in a directed graph; words that are connected to a are relevant to a. They are then weighted with a deterministic function of distance from a. DISPLAYFORM1 Additionally, bank creation is kept in check by constraining p li w.r.t.

the expected number of memory banks.

The expected number of memory banks can be thought of as a geometric distribution ??? Geometric(p li ) parameterized byp li , a hyperparameter.

Typically, at each sentence stepp is raised to the inverse power of the current sentence step to reflect the amount of information ingested.

Intuitively, this loss ensures banks are created when a memory bank contains too many nodes.

On the other hand, the learned mask q (eq. 9) enables the model to weight certain nodes a higher entropy to prompt bank creation.

Through these two dependencies, the model is able to simulate bank creation as a function of the number of nodes and the type of nodes in a given memory bank.

DISPLAYFORM2 All components combined, the final loss is given in the following equation DISPLAYFORM3

In this section, we evaluate AMN accuracy and inference times on the bAbI dataset and extended bAbI tasks dataset.

We compare our performance with Entnet BID12 ), which recently achieved state of the art results on the bAbi dataset.

For accuracy measurements, we also compare with DMN+ and encoder-decoder methods.

Finally we discuss the time trade offs between AMN and current SOTA methods.

The portion regarding inference times are not inclusive of story ingestion.

We summarize our experiments results as follows:??? We are able to solve all bAbi tasks using AMN.

Furthermore, AMN is able to reason important entities and propagate them to the final memory bank allowing for 48% fewer entities examined during inference.??? We construct extended bAbI tasks to evaluate AMN behavior.

First, we extend Task 1 for multiple questions in order to gauge performance in a more robust manner.

For example, if a reasonable set of questions are asked (where reasonable means that collectively they do not require all entities to answer implying entities can be filtered out), will the model still sufficiently reason through entities.

We find that our network is able to reason useful entities for both tasks and store them in the final memory bank.

Furthermore, we also scale bAbI for a large number of entities and find that AMN provides additional benefits at scale since only relevant entities are stored in the final memory bank.

We implement our network in PyTorch BID18 ).

We initialize our model using Xavier initialization, and the word embeddings utilize random uniform initialization ranging from ??? ??? 3 to ??? 3.

The learning rate is set as 0.001 initially and updated with a learning rate scheduler.

E[s] contains nodes in the connected components of A containing the answer node a which has relevance scores sampled from a Gaussian distribution centered at 0.75 with a variance of 0.05 (capped at 1).

Nodes that are not in the connected component containing a are similarly sampled from a Gaussian centered from 0.3 with a variance of 0.1 (capped at 0).p li is initially set to 0.8 and ?? varies depending on the story length from 0.1 ??? ?? ??? 0.25.

Note that for transitive tasks,p li is set to 0.2.

We train our models using the Adam optimizer BID14 .

The bAbI task suite consists of 20 reasoning tasks that include deduction, induction, path finding etc.

Results are from the following parameters: ??? 200 epochs, best of 10 runs.

TAB1 shows the inference performance in terms of the number of entities examined.

A task is considered passed if the error rate is less than 5%.We find that AMN creates 1 ??? 6 memory banks for different tasks.

We also find that 8 tasks can be solved by looking at just one memory bank and 14 tasks can be solved with half the total number of memory banks.

Lastly, all tasks can be solved by examining less than or equal the total number of entities (e ??? M ??? |V | + )1 .

Tasks that cannot be solved in fewer than half the memory banks either require additional entities due to transitive logic or have multiple questions.

For transitive logic, additional banks could be required as an relevant nodes may be in a further bank.

However, this still avoids scanning all banks.

In the case of multiple questions, all nodes may become necessary to construct all answers.

We provide additional evaluation in Appendix to examine memory bank behavior for certain tasks.

TAB5 shows the number of banks created and required to solve a task, as well as the ratio of entities examined to solve the task.

TAB2 shows the complexity of AMN and other SOTA models.

Entnet uses an empirically selected parameter, typically set to the number of vocabulary words.

GGT-NN uses the number of vocabulary words and creates new k new nodes intermittently per sentence step.

For tasks where nodes are easily separable where nodes are clearly irrelevant to the question(s), AMN is able to successfully reduce the number of nodes examined.

However for tasks that require more information, such as counting (Task 7), the model is still able to obtain the correct answer Table 2 : Memory bank analysis of indicative tasks.without using all entities.

Lastly, transitive logic tasks where information is difficult to separate due to dependencies of entities, the model creates very few banks (1 or 2) and uses all nodes to correctly generate an answer.

We note that in the instance where the model only creates one bank, it is very sparse, containing only one or two entities.

Because variations in computation times in text are minute, the number of entities required to construct an answer are of more interest as they directly correspond to the number of computations required.

Additionally, due to various implementations of current models, their run times can significantly vary.

However, for the comparison of inference times, AMN's decoder and EntNet's decoder are highly similar and contain roughly the same number of operations.

We extend the bAbI tasks by adding additional entities and sentences and adding multiple questions for a single story, for Task 1.

We increase the the number of entities to 100 entities in the task generation system instead of existing 38.

We also extend the story length to 90 to ensure new entities are referenced.

We find that AMN creates 6 memory banks and the ratio of entities in the final banks versus the overall entities drops to 0.13 given the excess entities that are not referenced in the questions.

Multiple questions: We also augment the tasks with multiple questions to understand if AMN can handle when a story has multiple questions associated with it.

We extend our model to handle multiple questions at once to limit re-generating the network for every question.

To do so, we modify bAbi to generate several questions per story for tasks that do not currently have multiple questions.

For single supporting fact (Task 1), the model creates 3 banks and requires 1 bank to successfully pass the task.

Furthermore, the ratio of entities required to pass the task only increases by 0.16 for a total of 0.38.

In this paper, we present Adaptive Memory Network that learns to adaptively organize the memory to answer questions with lower inference times.

Unlike NTMs which learn to read and write at individual memory locations, Adaptive Memory Network demonstrates a novel design where the learned memory management is coarse-grained that is easier to train.

Through our experiments, we demonstrate that AMN can learn to reason, construct, and sort memory banks based on relevance over the question set.

AMN architecture is generic and can be extended to other types of tasks where the input sequence can be separated into different entities.

In the future, we plan to evaluate AMN over such tasks to evaluate AMN generality.

We also plan to experiment with larger scale datasets (beyond bAbI, such as a document with question pairs) that have a large number of entities to further explore scalability.

Method Complexity Entnet BID12 We describe our overall algorithm in pseudo-code in this section.

We follow the notation as described in the paper.

DISPLAYFORM0 Algorithm 1 AMN(S, q, a) DISPLAYFORM1 for word w ??? s do 4: DISPLAYFORM2 end for 6: DISPLAYFORM3 for memory bank m i ??? M do 8: DISPLAYFORM4 n mi ??? SGRU(D, n mi ) We compare the computations costs during the decode operation during inference for solving the extended bAbi task.

We compute the overheads for AMN Entnet BID12 ) and GGT-NN.

TAB2 gives the decode comparisons between AMN, Entnet and GGT-NN.

Here, |V | represents to the total number of entities for all networks.

GGT-NN can dynamically create nodes and k k is hyper parameter the new nodes created for S sentences in input story.

?? is the percent of entities stored in the final bank w.r.t to the total entities for AMN.We compare the wall clock execution times for three tasks within bAbI for 1000 examples/task.

We compare the wall-clock times for three tasks.

We compare the inference times of considering all banks (and entities) versus the just looking at the passing banks as required by AMN.

We find that AMN requires fewer banks and as a consequence fewer entities and saves inference times.

In this section, we understand memory bank behavior of AMN.

Figure 3 shows the memory banks and the entity creation for a single story example, for some of the tasks from bAbI. Depending upon the task, and distance from the question AMN creates variable number of memory banks.

The heatmap demonstrates how entities are copied across memory banks.

Grey blocks indicate absence of those banks.

Under review as a conference paper at ICLR 2018 Figure 4 shows how propagation happens after every time step.

The nodes represent entities corresponding to words in a sentence.

As sentences are processed word by word, a directed graph is drawn progressively from w 0 ...

w i ...w N .

If sentence l k 's path contains nodes already in the current directed graph, l k will include said nodes in the its path.

After l k is added to A, the model propagates the new update hidden state information a i among all node states using a GRU.

a i for each node i is equal to the sum of the incoming edges' node hidden states.

Additionally, we add a particular emphasis on l k to simulate recency.

At face value, one propagation step of A will only have a reachability of its immediate neighbor, so to reach all nodes, A is raised to a consecutive power r to reach and update each intermediate node.

r can be either the longest path in A or a set parameter.

@highlight

Memory networks with faster inference