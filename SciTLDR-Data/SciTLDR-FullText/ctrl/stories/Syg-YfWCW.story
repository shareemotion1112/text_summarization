Knowledge bases (KB), both automatically and manually constructed, are often incomplete --- many valid facts can be inferred from the KB by synthesizing existing information.

A popular approach to KB completion is to infer new relations by combinatory reasoning over the information found along other paths connecting a pair of entities.

Given the enormous size of KBs and the exponential number of paths, previous path-based models have considered only the problem of predicting a missing relation given two entities, or evaluating the truth of a proposed triple.

Additionally, these methods have traditionally used random paths between fixed entity pairs or more recently learned to pick paths between them.

We propose a new algorithm, MINERVA, which addresses the much more difficult and practical task of answering questions where the relation is known, but only one entity.

Since random walks are impractical in a setting with unknown destination and combinatorially many paths from a start node, we present a neural reinforcement learning approach which learns how to navigate the graph conditioned on the input query to find predictive paths.

On a comprehensive evaluation on seven knowledge base datasets, we found MINERVA to be competitive with many current state-of-the-art methods.

Automated reasoning, the ability of computing systems to make new inferences from observed evidence, has been a long-standing goal of artificial intelligence.

We are interested in automated reasoning on large knowledge bases (KB) with rich and diverse semantics BID44 BID1 BID5 .

KBs are highly incomplete BID26 , and facts not directly stored in a KB can often be inferred from those that are, creating exciting opportunities and challenges for automated reasoning.

For example, consider the small knowledge graph in Figure 1 .

We can answer the question "

Who did Malala Yousafzai share her Nobel Peace prize with?" from the following reasoning path: Malala Yousafzai →

WonAward → Nobel Peace Prize 2014 → AwardedTo → Kailash Satyarthi.

Our goal is to automatically learn such reasoning paths in KBs.

We frame the learning problem as one of query answering, that is to say, answering questions of the form (Malala Yousafzai, SharesNobelPrizeWith, ?).From its early days, the focus of automated reasoning approaches has been to build systems that can learn crisp symbolic logical rules BID24 BID34 .

Symbolic representations have also been integrated with machine learning especially in statistical relational learning BID29 BID15 BID21 BID22 , but due to poor generalization performance, these approaches have largely been superceded by distributed vector representations.

Learning embedding of entities and relations using tensor factorization or neural methods has been a popular approach BID31 BID2 Socher et al., 2013, inter alia) , but these methods cannot capture chains of reasoning expressed by KB paths.

Neural multi-hop models BID30 BID17 BID47 address the aforementioned problems to some extent by operating on KB paths embedded in vector space.

However, these models take as input a set of paths which are gathered by performing random walks Figure 1: A small fragment of a knowledge base represented as a knowledge graph.

Solid edges are observed and dashed edges are part of queries.

Note how each query relation (e.g. SharesNobelPrizeWith, Nationality, etc.) can be answered by traversing the graph via "logical" paths between entity 'Malala Yousafzai' and the corresponding answer.independent of the query relation.

Additionally, models such as those developed in BID30 ; BID9 use the same set of initially collected paths to answer a diverse set of query types (e.g. MarriedTo, Nationality, WorksIn etc.).This paper presents a method for efficiently searching the graph for answer-providing paths using reinforcement learning (RL) conditioned on the input question, eliminating any need for precomputed paths.

Given a massive knowledge graph, we learn a policy, which, given the query (entity 1 , relation, ?), starts from entity 1 and learns to walk to the answer node by choosing to take a labeled relation edge at each step, conditioning on the query relation and entire path history.

This formulates the query-answering task as a reinforcement learning (RL) problem where the goal is to take an optimal sequence of decisions (choices of relation edges) to maximize the expected reward (reaching the correct answer node).

We call the RL agent MINERVA for "Meandering In Networks of Entities to Reach Verisimilar Answers."Our RL-based formulation has many desirable properties.

First, MINERVA has the built-in flexibility to take paths of variable length, which is important for answering harder questions that require complex chains of reasoning BID42 .

Secondly, MINERVA needs no pretraining and trains on the knowledge graph from scratch with reinforcement learning; no other supervision or fine-tuning is required representing a significant advance over prior applications of RL in NLP.

Third, our path-based approach is computationally efficient, since by searching in a small neighborhood around the query entity it avoids ranking all entities in the KB as in prior work.

Finally, the reasoning paths found by our agent automatically form an interpretable provenance for its predictions.

The main contributions of the paper are: (a) We present agent MINERVA, which learns to do query answering by walking on a knowledge graph conditioned on an input query, stopping when it reaches the answer node.

The agent is trained using reinforcement learning, specifically policy gradients ( § 2).

(b) We evaluate MINERVA on several benchmark datasets and compare favorably to Neural Theorem Provers (NTP) BID39 and Neural LP , which do logical rule learning in KBs, and also state-of-the-art embedding based methods such as DistMult BID54 and ComplEx BID48 and ConvE BID12 .

(c) We also extend MINERVA to handle partially structured natural language queries and test it on the WikiMovies dataset ( § 3.3) BID25 .We also compare to DeepPath BID53 which uses reinforcement learning to pick paths between entity pairs.

The main difference is that the state of their RL agent includes the answer entity since it is designed for the simpler task of predicting if a fact is true or not.

As such their method cannot be applied directly to our more challenging query answering task where the second entity is unknown and must be inferred.

Nevertheless, MINERVA outperforms DeepPath on their benchmark NELL-995 dataset when compared in their experimental setting ( § 3.2.2).

We formally define the task of query answering in a KB.

Let E denote the set of entities and R denote the set of binary relations.

A KB is a collection of facts stored as triplets (e 1 , r, e 2 ) where e 1 , e 2 ∈ E and r ∈ R. From the KB, a knowledge graph G can be constructed where the entities e 1 , e 2 are represented as the nodes and relation r as labeled edge between them.

Formally, a knowledge graph is a directed labeled multigraph G = (V, E, R), where V and E denote the vertices and edges of the graph respectively.

Note that V = E and E ⊆ V × R ×V .

Also, following previous approaches BID2 BID30 BID53 , we add the inverse relation of every edge, i.e. for an edge (e 1 , r, e 2 ) ∈ E, we add the edge (e 2 , r −1 , e 1 ) to the graph. (If the set of binary relations R does not contain the inverse relation r −1 , it is added to R as well.)Since KBs have a lot of missing information, two natural tasks have emerged in the information extraction community -fact prediction and query answering.

Query answering seeks to answer questions of the form (e 1 , r, ?), e.g. Toronto, locatedIn, ?, whereas fact prediction involves predicting if a fact is true or not, e.g. (Toronto, locatedIn, Canada)?.

Algorithms for fact prediction can be used for query answering, but with significant computation overhead, since all candidate answer entities must be evaluated, making it prohibitively expensive for large KBs with millions of entities.

In this work, we present a query answering model, that learns to efficiently traverse the knowledge graph to find the correct answer to a query, eliminating the need to evaluate all entities.

Query answering reduces naturally to a finite horizon sequential decision making problem as follows:We begin by representing the environment as a deterministic partially observed Markov decision process on a knowledge graph G derived from the KB ( §2.1).

Our RL agent is given an input query of the form e 1q , r q , ? .

Starting from vertex corresponding to e 1q in G, the agent follows a path in the graph stopping at a node that it predicts as the answer ( § 2.2).

Using a training set of known facts, we train the agent using policy gradients more specifically by REINFORCE (Williams, 1992) with control variates ( § 2.3).

Let us begin by describing the environment.

Our environment is a finite horizon, deterministic partially observed Markov decision process that lies on the knowledge graph G derived from the KB.

On this graph we will now specify a deterministic partially observed Markov decision process, which is a 5-tuple (S, O, A, δ, R), each of which we elaborate below.

States.

The state space S consists of all valid combinations in E × E × R × E. Intuitively, we want a state to encode the query (e 1q , r q ), the answer (e 2q ), and a location of exploration e t (current location of the RL agent).

Thus overall a state S ∈ S is represented by S = (e t , e 1q , r q , e 2q ) and the state space consists of all valid combinations.

Observations.

The complete state of the environment is not observed.

Intuitively, the agent knows its current location (e t ) and (e 1q , r q ), but not the answer (e 2q ), which remains hidden.

Formally, the observation function O : S → E × E × R is defined as O(s = (e t , e 1q , r q , e 2q )) = (e t , e 1q , r q ).Actions.

The set of possible actions A S from a state S = (e t , e 1q , r q , e 2q ) consists of all outgoing edges of the vertex e t in G. Formally A S = {(e t , r, v) ∈ E : S = (e t , e 1q , r q , e 2q ), r ∈ R, v ∈ V } ∪ {(s, ∅, s)}. Basically, this means an agent at each state has option to select which outgoing edge it wishes to take having the knowledge of the label of the edge r and destination vertex v.

During implementation, we unroll the computation graph up to a fixed number of time steps T. We augment each node with a special action called 'NO OP' which goes from a node to itself.

Some questions are easier to answer and needs fewer steps of reasoning than others.

This design decision allows the agent to remain at a node for any number of time steps.

This is especially helpful when the agent has managed to reach a correct answer at a time step t < T and can continue to stay at the 'answer node' for the rest of the time steps.

Alternatively, we could have allowed the agent to take a special 'STOP' action, but we found the current setup to work sufficiently well.

As mentioned before, we also add the inverse relation of a triple, i.e. for the triple (e 1 , r, e 2 ), we add the triple (e 2 , r −1 , e 1 ) to the graph.

We found this important because this actually allows our agent to undo a potentially wrong decision.

Transition.

The environment evolves deterministically by just updating the state to the new vertex incident to the edge selected by the agent.

The query and answer remains the same.

Formally, the transition function is δ : S × A → S defined by δ(S, A) = (v, e 1q , r q , e 2q ), where S = (e t , e 1q , r q , e 2q ) and A = (e t , r, v)).Rewards.

We only have a terminal reward of +1 if the current location is the correct answer at the end and 0 otherwise.

To elaborate, if S T = (e t , e 1q , r q , e 2q ) is the final state, then we receive a reward of +1 if e t = e 2q else 0., i.e. R(S T ) = I{e t = e 2q }.

To solve the finite horizon deterministic partially observable Markov decision process described above, we design a randomized non-stationary history-dependent policy π = (d 1 , d 2 , ..., d T−1 ), where d t : H t → P(A S t ) and history H t = (H t−1 , A t−1 , O t ) is just the sequence of observations and actions taken.

We restrict ourselves to policies parameterized by long short-term memory network (LSTM) BID19 ).An agent based on LSTM encodes the history H t as a continuous vector h t ∈ R 2d .

We also have embedding matrix r ∈ R |R|×d and e ∈ R |E|×d for the binary relations and entities respectively.

The history embedding for H t = (H t−1 , A t−1 , O t ) is updated according to LSTM dynamics: DISPLAYFORM0 where a t−1 ∈ R d and o t ∈ R d denote the vector representation for action/relation at time t − 1 and observation/entity at time t respectively and [; ] denote vector concatenation.

To elucidate, a t−1 = r A t−1 , i.e. the embedding of the relation corresponding to label of the edge the agent chose at time t − 1 and o t = e e t if O t = (e t , e 1q , r q ) i.e. the embedding of the entity corresponding to vertex the agent is at time t.

Based on the history embedding h t , the policy network makes the decision to choose an action from all available actions (A S t ) conditioned on the query relation.

Recall that each possible action represents an outgoing edge with information of the edge relation label l and destination vertex/entity d. So embedding for each A ∈ A S t is [r l ; e d ], and stacking embeddings for all the outgoing edges we obtain the matrix A t .

The network taking these as inputs is parameterized as a two-layer feedforward network with ReLU nonlinearity which takes in the current history representation h t and the embedding for the query relation r q and outputs a probability distribution over the possible actions from which a discrete action is sampled.

In other words, DISPLAYFORM1 Note that the nodes in G do not have a fixed ordering or number of edges coming out from them.

The size of matrix A t is |A S t | × 2d, so the decision probabilities d t lies on simplex of size |A S t |.

Also the procedure above is invariant to order in which edges are presented as desired and falls in purview of neural networks designed to be permutation invariant BID56 .

Finally, to summarize, the parameters of the LSTM, the weights W 1 , W 2 , the corresponding biases (not shown above for brevity), and the embedding matrices form the parameters θ of the policy network.

For the policy network (π θ ) described above, we want to find parameters θ that maximize the expected reward: DISPLAYFORM0 where we assume there is a true underlying distribution (e 1 , r, e 2 ) ∼ D. To solve this optimization problem, we employ REINFORCE (Williams, 1992) as follows:• The first expectation is replaced with empirical average over the training dataset.• For the second expectation, we approximate by running multiple rollouts for each training example.

The number of rollouts is fixed and for all our experiments we set this number to 20.• For variance reduction, a common strategy is to use an additive control variate baseline BID18 BID14 BID13 .

We use a moving average of the cumulative discounted reward as the baseline.

We tune the weight of this moving average as a hyperparameter.

Note that in our experiments we found that using a learned baseline performed similarly, but we finally settled for cumulative discounted reward as the baseline owing to its simplicity.• To encourage diversity in the paths sampled by the policy at training time, we add an entropy regularization term to our cost function scaled by a constant (β).

We now present empirical studies for MINERVA in order to establish that (i) MINERVA is competitive for query answering on small (Sec. 3.1.1) as well as large KBs (Sec. 3.1.2), (ii) MINERVA is superior to a path based models that do not search the KB efficiently or train query specific models (Sec. 3.2), (iii) MINERVA can not only be used for well formed queries, but can also easily handle partially structured natural language queries (Sec 3.3), (iv) MINERVA is highly capable of reasoning over long chains, and (v) MINERVA is robust to train and has much faster inference time (Sec. 3.5).

To gauge the reasoning capability of MINERVA, we begin with task of query answering on KB, i.e. we want to answer queries of the form (e 1 , r, ?).

Note that, as mentioned in Sec. 2, this task is subtly different from fact checking in a KB.

Also, as most of the previous literature works in the regime of fact checking, their ranking includes variations of both (e 1 , r, x) and (x, r, e 2 ).

However, since we do not have access to e 2 in case of question answering scenario the same ranking procedure does not hold for us -we only need to rank on (e 1 , r, x).

This difference in ranking made it necessary for us to re-run all the implementations of previous work.

We used the implementation or the best pre-trained models (whenever available) of BID39 and BID12 .

For MINERVA to produce a ranking of answer entities during inference, we do a beam search with a beam width of 50 and rank entities by the probability of the trajectory the model took to reach the entity and remaining entities are given a rank of ∞.Method We compare MINERVA with various state-of-the-art models using HITS@1,3,10 and mean reciprocal rank (MRR), which are standard metrics for KB completion tasks.

In particular we compare against embedding based models -DistMult BID54 , ComplEx BID48 and ConvE BID12 .

For ConvE and ComplEx, we used the implementation released by BID12 1 on the best hyperparameter settings reported by them.

For DistMult, we use our highly tuned implementation (e.g. which performs better than the state-of-the-art results of BID46 ).

We also compare with two recent work in learning logical rules in KB namely Neural Theorem Provers (NTP) BID39 and NeuralLP .

BID39 also reports a NTP model which is trained with an additional objective function of ComplEx (NTP-λ).

For these models, we used the implementation released by corresponding authors 2 3 , again on the best hyperparameter settings reported by them.

Table 3 : Query answering results on KINSHIP and UMLS datasets.

Dataset We use three standard datasets: COUNTRIES BID3 , KINSHIP, and UMLS BID21 .

The COUNTRIES dataset ontains countries, regions, and subregions as entities and is carefully designed to explicitly test the logical rule learning and reasoning capabilities of link prediction models.

The queries are of the form LocatedIn(c, ?) and the answer is a region (e.g. LocatedIn(Egypt, ?) with the answer as Africa).

The dataset has 3 tasks (S1-3 in table 2) each requiring reasoning steps of increasing length and difficulty (see BID39 for more details about the tasks).

Following the design of the COUNTRIES dataset, for task S1 and S2, we set the maximum path length T = 2 and for S3, we set T = 3.

The Unified Medical Language System (UMLS) dataset, is from biomedicine.

The entities are biomedical concepts (e.g. disease, antibiotic) and relations are like treats and diagnoses.

The KINSHIP dataset contains kinship relationships among members of the Alyawarra tribe from Central Australia.

For these two task we use maximum path length T = 2.

Also, for MINERVA we turn off entity in (1) in these experiments.

Observations For the COUNTRIES dataset, in TAB1 we report a stronger metric -the area under the precision-recall curve -as is common in the literature.

We can see that MINERVA compares favorably or outperforms all the baseline models except on the task S2 of COUNTRIES, where the ensemble model NTP-λ and ConvE outperforms it, albeit with a higher variance across runs.

Our gains are much more prominent in task S3, which is the hardest among all the tasks.

The Kinship and UMLS datasets are small KB datasets with around 100 entities each and as we see from Table 3 , embedding based methods (ConvE, ComplEx and DistMult) perform much better than methods which aim to learn logical rules (NTP, NeuralLP and MINERVA).

On Kinship, MINERVA outperforms both NeuralLP and NTP and matches the HITS@10 performance of NTP on UMLS.

Unlike COUNTRIES, these datasets were not designed to test the logical rule learning ability of models and given the small size, embedding based models are able to get really high performance.

Combination of both methods gives a slight increase in performance as can be seen from the results of NTP-λ.

However, when we initialized MINERVA with pre-trained embeddings of ComplEx, we did not find a significant increase in performance.

Dataset Next we evaluate MINERVA on three large KG datasets -WN18RR, FB15K-237 and NELL-995.

The WN18RR BID12 and FB15K-237 BID46 datasets are created from the original WN18 and FB15K datasets respectively by removing various sources of test leakage, making the datasets more realistic and challenging.

The NELL-995 dataset released by BID53 has separate graphs for each query relation, where a graph for a query relation can have triples from the test set of another query relation.

For the query answering experiment, we combine all the graphs and removed all test triples (and the corresponding triples with inverse relations) from the graph.

We also noticed that several triples in the test set had an entity (source or target) that never appeared in the graph.

Since, there will be no trained embeddings for those entities, we removed them from the test set.

This reduced the size of test set from 3992 queries to 2818 queries.

We observe that on FB15K-237, however, embedding based methods dominate over MINERVA and NeuralLP.

Upon deeper inspection, we found that the query relation types of FB15K-237 knowledge graph differs significantly from others.

Analysis of query relations of FB15k-237: We analyzed the type of query relation types on the FB15K-237 dataset.

Following BID2 , we categorized the query relations into (M)any to 1, 1 to M or 1 to 1 relations.

An example of a M to 1 relation would be '/people/profession' (What is the profession of person 'X'?).

An example of 1 to M relation would be /music/instrument/instrumentalists ('Who plays the music instrument X?') or '/people/ethnicity/people' ('Who are people with ethnicity X?').

From a query answering point of view, the answer to these questions is a list of entities.

However, during evaluation time, the model is evaluated based on whether it is able to predict the one target entity which is in the query triple.

Also, since MINERVA outputs the end points of the paths as target entities, it is sometimes possible that the particular target entity of the triple does not have a path from the source entity (however there are paths to other 'correct' answer entities).

TAB10 (in appendix) shows few other examples of relations belonging to different classes.

Following BID2 , we classify a relation as 1-to-M if the ratio of cardinality of tail to head entities is greater than 1.5 and as M-to-1 if it is lesser than 0.67.

In the validation set of FB15K-237, 54% of the queries are 1-to-M, whereas only 26% are M-to-1.

Contrasting it with NELL-995, 27% are 1-to-M and 36% are M-to-1 or UMLS where only 18% are 1-to-M. Table 10 (in appendix) shows few relations from FB15K-237 dataset which have high tail-to-head ratio.

The average ratio for 1-TO-M relations in FB15K-237 is 13.39 (substantially higher than 1.5).

As explained before, the current evaluation scheme is not suited when it comes to 1-to-M relations and the high percentage of 1-to-M relations in FB15K-237 also explains the sub optimal performance of MINERVA.We also check the frequency of occurrence of various unique path types.

We define a path type as the sequence of relation types (ignoring the entities) in a path.

Intuitively, a predictive path which generalizes across queries will occur many number of times in the graph.

Figure 2 shows the plot.

As we can see, the characteristics of FB15K-237 is quite different from other datasets.

For example, in NELL-995, more than 1000 different path types occur more than 1000 times.

WN18RR has only 11 different relation types which means there are only 11 3 possible path types of length 3 and even fewer number of them would be predictive.

As can be seen, there are few path types which occur more than 10 4 times and around 50 of them occur more than 1000 times.

However in FB15K-237,

Figure 2: Count of number of unique path types of length 3 which occur more than 'x' times in various datasets.

For example, in NELL-995 there are more than 10 3 path types which occur more than 10 3 times.

However, for FB15k-237, we see a sharp decrease as 'x' becomes higher, suggesting that path types do not repeat often.which has the highest number of relation types, we observe a sharp decrease in the number of path types which occur a significant number of times.

Since MINERVA cannot find path types which repeat often, it finds it hard to learn path types that generalize.

In this experiment, we compare to a model which gathers path based on random walks and tries to predict the answer entity.

Neural multi-hop models BID30 BID47 , operate on paths between entity pairs in a KB.

However these methods need to know the target entity in order to pre-compute paths between entity pairs.

BID17 is an exception in this regard as they do random walks starting from a source entity 'e 1 ' and then using the path, they train a classifier to predict the target answer entity.

However, they only consider one path starting from a source entity.

In contrast, BID30 ; BID47 use information from multiple paths between the source and target entity.

We design a baseline model which combines the strength of both these approaches.

Starting from 'e 1 ' , the model samples (k = 100) random paths of up to a maximum length of T = 3.

Following BID30 , we encode each paths with an LSTM followed by a max-pooling operation to featurize the paths.

This feature is concatenated with the source entity and query relation vector which is then passed through a feed forward network which scores all possible target entities.

The network is trained with a multi-class cross entropy objective based on observed triples and during inference we rank target entities according to the model score.

The PATH-BASELINE column of table 4 shows the performance of this model on the three datasets.

As we can see MINERVA outperforms this baseline significantly.

This shows that a model which predicts based on a set of randomly sampled paths does not do as well as MINERVA because it either loses important paths during random walking or it fails to aggregate predictive features from all the k paths, many of which would be irrelevant to answer the given query.

The latter is akin to the problem with distant supervision BID27 , where important evidence gets lost amidst a plethora of irrelevant information.

However, by taking each step conditioned on the query relation, MINERVA can effectively reduce the search space and focus on paths relevant to answer the query.

We also compare MINERVA with DeepPath which uses RL to pick paths between entity pairs.

For a fair comparison, we only rank the answer entities against the negative examples in the dataset used in their experiments 5 and report the mean average precision (MAP) scores for each query relation.

DeepPath feeds the paths its agent gathers as input features to the path ranking algorithm (PRA) BID22 , which trains a per-relation classifier.

But unlike them, we train one model which learns for all query relations so as to enable our agent to leverage from correlations and more data.

If our agent is not able to reach the correct entity or one of the negative entities, the corresponding entities gets a score of negative infinity.

If MINERVA fails to reach any of the entities in the set of correct and negative entities.

then we fall back to a random ordering of the entities.

As show in Queries in KBs are structured in the form of triples.

However, this is unsatisfactory since for most real applications, the queries appear in natural language.

As a first step in this direction, we extend MINERVA to take in "partially structured" queries.

We use the WikiMovies dataset BID25 which contains questions in natural language albeit generated by templates created by human annotators.

An example question is "Which is a film written by Herb Freed?".

WikiMovies also has an accompanying KB which can be used to answer all the questions.

We link the entity occurring in the question to the KB via simple string matching.

To form the vector representation of the query relation, we design a simple question encoder which computes the average of the embeddings of the question words.

The word embeddings are learned from scratch and we do not use any pretrained embeddings.

We compare our results with those reported in TAB7 .

For this experiment, we found that T = 1 sufficed, suggesting that WikiMovies is not the best testbed for multihop reasoning, but this experiment is a promising first step towards the realistic setup of using KBs to answer natural language question.

While chains in KB need not be very long to get good empirical results BID30 BID9 , in principle MINERVA can be used to learn long reasoning chains.

To evaluate the same, we test our model on a synthetic 16-by-16 grid world dataset created by , where the task is to navigate to a particular cell (answer entity) starting from a random cell (start entity) by following a set of directions (query relation).

The KB consists of atomic triples of the form ((2,1), North, (1,1)) -entity (1,1) is north of entity (2,1).

The queries consists of a sequence of directions (e.g. North, SouthWest, East).

The queries are classified into classes based on the path lengths.

FIG2 shows the accuracy on varying path lengths.

Compared to Neural LP, MINERVA is much more robust to queries, which require longer path, showing minimal degradation in performance for even the longest path in the dataset.

Training time.

Figure 5 plots the HITS@10 scores on the development set against the training time comparing MINERVA with DistMult.

It can be seen that MINERVA converges to a higher score much faster than DistMult.

It is also interesting to note that even during the early stages of the training, MINERVA has much higher performance than that of DistMult, as during these initial stages, MINERVA would just be doing random walks in the neighborhood of the source entity (e 1 ).

This implies that MINERVA's approach of searching for an answer in the neighborhood of e 1 is a much more efficient and smarter strategy than ranking all entities in the knowledge graph (as done by DistMult and other related methods).Inference Time.

At test time, embedding based methods such as ConvE, ComplEx and DistMult rank all entities in the graph.

Hence, for a test-time query, the running time is always O (|E|) where R denotes the set of entities (= nodes) in the graph.

MINERVA, on the other hand is efficient at inference time since it has to essentially search for answer entities in its local neighborhood.

The many cost at inference time for MINERVA is to compute probabilities for all outgoing edges along the path.

Thus inference time of MINERVA only depends on degree distribution of the graph.

If we assume the knowledge graph to obey a power law degree distribution, like many natural graphs, then for MINERVA the average inference time can be shown to be O( α α−1 ), when the coefficient of the power law α > 1.

The median inference time for MINERVA is O(1) for all values of α.

Note that these quantities are independent of size of entities |E|.

For instance, on the test dataset of WN18RR, the wall clock inference time of MINERVA is 63s whereas that of a GPU implementation of DistMult, which is the simplest among the lot, is 211s.

Similarly the wall-clock inference time on the test set of NELL-995 for a GPU implementation of DistMult is 115s whereas that of MINERVA is 35s.

Query based Decision Making.

At each step before making a decision, our agent conditions on the query relation.

Figure 4 shows examples, where based on the query relation, the probabilities are peaked on different actions.

For example, when the query relation is WorksFor, MINERVA assigns a much higher probability of taking the edge CoachesTeam than AthletePlaysInLeague.

We also see similar behavior on the WikiMovies dataset where the query consists of words instead of fixed schema relation.

Model Robustness.

TAB8 also reports the mean and standard deviation across three independent runs of MINERVA.

We found it easy to obtain/reproduce the highest scores across several runs as can be seen from the low deviations in scores.

Similarly inverse relation gives the agent the ability to recover from a potentially wrong decision it has taken before.

Example (ii) shows such an example, where the agent took a incorrect decision at the first step but was able to revert the decision because of the presence of inverted edges.

Learning vector representations of entities and relations using tensor factorization BID31 BID32 BID2 BID38 BID33 BID54 or neural methods BID43 BID46 BID49 has been a popular approach to reasoning with a knowledge base.

However, these methods cannot capture more complex reasoning patterns such as those found by following inference paths in KBs.

Multi-hop link prediction approaches BID22 BID30 BID17 BID47 BID9 ) address the problems above, but the reasoning paths that they operate on are gathered by performing random walks independent of the type of query relation.

BID22 further filters paths from the set of sampled paths based on the restriction that the path must end at one of the target entities in the training set and are within a maximum length.

These constraints make them query (i) Can learn general rules: DISPLAYFORM0 (ii) Can learn shorter path: Richard F. Velky Published as a conference paper at ICLR 2018 dependent but they are heuristic in nature.

Our approach eliminates any necessity to pre-compute paths and learns to efficiently search the graph conditioned on the input query relation.

Inductive Logic Programming (ILP) BID29 aims to learn general purpose predicate rules from examples and background knowledge.

Early work in ILP such as FOIL BID36 , PROGOL BID28 are either rule-based or require negative examples which is often hard to find in KBs (by design, KBs store true facts).

Statistical relational learning methods BID15 BID21 BID41 along with probabilistic logic BID37 BID4 combine machine learning and logic but these approaches operate on symbols rather than vectors and hence do not enjoy the generalization properties of embedding based approaches.

There are few prior work which treat inference as search over the space of natural language.

BID35 propose a task (WikiNav) in which each the nodes in the graph are Wikipedia pages and the edges are hyperlinks to other wiki pages.

The entity is to be represented by the text in the page and hence the agent is required to reason over natural language space to navigate through the graph.

Similar to WikiNav is Wikispeedia BID51 in which an agent needs to learn to traverse to a given target entity node (wiki page) as quickly as possible.

BID0 propose natural logic inference in which they cast the inference as a search from a query to any valid premise.

At each step, the actions are one of the seven lexical relations introduced by BID23 .Neural Theorem Provers (NTP) BID39 and Neural LP are methods to learn logical rules that can be trained end-to-end with gradient based learning.

NTPs are constructed by Prolog's backward chaining inference method.

It operates on vectors rather than symbols, thereby providing a success score for each proof path.

However, since a score can be computed between any two vectors, the computation graph becomes quite large because of such soft-matching during substitution step of backward chaining.

For tractability, it resorts to heuristics such as only keeping the top-K scoring proof paths trading-off guarantees for exact gradients.

Also the efficacy of NTPs has yet to be shown on large KBs.

Neural LP introduces a differential rule learning system using operators defined in TensorLog BID7 .

It has a LSTM based controller with a differentiable memory component BID16 BID45 and the rule scores are calculated via attention.

Even though, differentiable memory allows end to end training, it necessitates accessing the entire memory, which can be computationally expensive.

RL approaches capable of hard selection of memory BID57 are computationally attractive.

MINERVA uses a similar hard selection of relation edges to walk on the graph.

More importantly, MINERVA outperforms both these methods on their respective benchmark datasets.

DeepPath BID53 uses RL based approaches to find paths in KBs.

However, the state of their MDP requires the target entity to be known in advance and hence their path finding strategy is dependent on knowing the answer entity.

MINERVA does not need any knowledge of the target entity and instead learns to find the answer entity among all entities.

DeepPath, additionally feeds its gathered paths to Path Ranking Algorithm BID22 , whereas MINERVA is a complete system trained to do query answering.

DeepPath also uses fixed pretrained embeddings for its entity and relations.

Lastly, on comparing MINERVA with DeepPath in their experimental setting on the NELL dataset, we match their performance or outperform them.

MINERVA is also similar to methods for learning to search for structured prediction BID8 BID10 BID11 BID40 BID6 .

These methods are based on imitating a reference policy (oracle) which make near-optimal decision at every step.

In our problem setting, it is unclear what a good reference policy would be.

For example, a shortest path oracle between two entities would be unideal, since the answer providing path should depend on the query relation.

We explored a new way of automated reasoning on large knowledge bases in which we use the knowledge graphs representation of the knowledge base and train an agent to walk to the answer node conditioned on the input query.

We achieve state-of-the-art results on multiple benchmark knowledge base completion tasks and we also show that our model is robust and can learn long chains-ofreasoning.

Moreover it needs no pretraining or initial supervision.

Future research directions include applying more sophisticated RL techniques and working directly on textual queries and documents.

Table 10 : Few example 1-to-M relations from FB15K-237 with high cardinality ratio of tail to head.

Experimental Details We choose the relation and embedding dimension size as 200.

The action embedding is formed by concatenating the entity and relation embedding.

We use a 3 layer LSTM with hidden size of 400.

The hidden layer size of MLP (weights W 1 and W 2 ) is set to 400.

We use Adam BID20 with the default parameters in REINFORCE for the update.

In our experiments, we tune our model over two hyper parameters, viz., β which is the entropy regularization constant and λ which is the moving average constant for the REINFORCE baseline.

The table 11 lists the best hyper parameters for all the datasets.

The NELL dataset released by BID53 includes two additional tasks for which the scores were not reported in the paper and so we were unable to compare them against DeepPath.

Nevertheless, we ran MINERVA on these tasks and report our results in

<|TLDR|>

@highlight

We present a RL agent MINERVA which learns to walk on a knowledge graph and answer queries