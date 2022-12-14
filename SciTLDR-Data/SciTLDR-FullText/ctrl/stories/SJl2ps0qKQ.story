As for knowledge-based question answering, a fundamental problem is to relax the assumption of answerable questions from simple questions to compound questions.

Traditional approaches firstly detect topic entity mentioned in questions, then traverse the knowledge graph to find relations as a multi-hop path to answers, while we propose a novel approach to leverage simple-question answerers to answer compound questions.

Our model consists of two parts: (i) a novel learning-to-decompose agent that learns a policy to decompose a compound question into simple questions and (ii) three independent simple-question answerers that classify the corresponding relations for each simple question.

Experiments demonstrate that our model learns complex rules of compositionality as stochastic policy, which benefits simple neural networks to achieve state-of-the-art results on WebQuestions and MetaQA.

We analyze the interpretable decomposition process as well as generated partitions.

Knowledge-Based Question Answering (KBQA) is one of the most interesting approaches of answering a question, which bridges a curated knowledge base of tremendous facts to answerable questions.

With question answering as a user-friendly interface, users can easily query a knowledge base through natural language, i.e., in their own words.

In the past few years, many systems BID5 BID2 Yih et al., 2015; BID11 BID13 have achieved remarkable improvements in various datasets, such as WebQuestions BID5 , SimpleQuestions BID6 and MetaQA .However, most of them BID31 BID6 BID10 BID34 BID36 assume that only simple questions are answerable.

Simple questions are questions that have only one relation from the topic entity to unknown tail entities (answers, usually substituted by an interrogative word) while compound questions are questions that have multiple 1 relations.

For example, "Who are the daughters of Barack Obama?" is a simple question and "Who is the mother of the daughters of Barack Obama?" is a compound question which can be decomposed into two simple questions.

In this paper, we aim to relax the assumption of answerable questions from simple questions to compound questions.

Figure 1 illustrates the process of answering compound questions.

Intuitively, to answer a compound question, traditional approaches firstly detect topic entity mentioned in the question, as the starting point for traversing the knowledge graph, then find a chain of multiple (??? 3) relations as a multi-hop 2 path to golden answers.

We propose a learning-to-decompose agent which assists simple-question answerers to solve compound questions directly.

Our agent learns a policy for decomposing compound question into simple ones in a meaningful way, guided by the feedback from the downstream simple-question answerers.

The goal of the agent is to produce partitions and compute the compositional structure of questions 1 We assume that the number of corresponding relations is at most three.

2 We are aware of the term multi-hop question in the literature.

We argue that compound question is a better fit for the context of KBQA since multi-hop characterizes a path, not a question.

As for document-based QA, multi-hop also refers to routing over multiple evidence to answers.

Figure 1: An example of answering compound questions.

Given a question Q, we first identify the topic entity e with entity linking.

By relation detection, a movie-to-actor relation f 1 , an actor-tomovie relation f 2 and a movie-to-writer relation f 3 forms a path to the answers W i .

Note that each relation f i corresponds to a part of the question.

If we decomposes the question in a different way, we may find a movie-to-movie relation g as a shortcut, and g(e) = f 2 (f 1 (e)) = (f 2 ??? f 1 )(e) holds.

Our model discovered such composite rules.

See section 4 for further discussion.with maximum information utilization.

The intuition is that encouraging the model to learn structural compositions of compound questions will bias the model toward better generalizations about how the meaning of a question is encoded in terms of compositional structures on sequences of words, leading to better performance on downstream question answering tasks.

We demonstrate that our agent captures the semantics of compound questions and generate interpretable decomposition.

Experimental results show that our novel approach achieves state-of-the-art performance in two challenging datasets (WebQuestions and MetaQA), without re-designing complex neural networks to answer compound questions.

For combinational generalization BID4 on the search space of knowledge graph, many approaches BID31 BID34 tackle KBQA in a tandem manner, i.e., topic entity linking followed by relation detection.

An important line of research focused on directly parsing the semantics of natural language questions to structured queries BID8 BID17 BID2 BID31 .

An intermediate meaning representation or logical form is generated for query construction.

It often requires pre-defined rules or grammars BID5 ) based on hand-crafted features.

By contrast, another line of research puts more emphasis on representing natural language questions instead of constructing knowledge graph queries.

Employing CNNs BID11 BID34 or RNNs BID10 BID36 , variable-length questions are compressed into their corresponding fix-length vector.

Most approaches in this line focus on solving simple questions because of the limited expression power of fix-length vector, consistent with observations BID23 BID0 in Seq2Seq task such as Neural Machine Translation.

Closely related to the second line of research, our proposed model learns to decompose compound question into simple questions, which eases the burden of learning vector representations for compound question.

Once the decomposition process is completed, a simple-question answerer directly decodes the vector representation of simple questions to an inference chain of relations with the desired order, which resolves the bottleneck of KBQA.

Many reinforcement learning approaches learn sentence representations in a bottom-up manner.

BID35 learn tree structures for the order of composing words into sentences using reinforcement learning with Tree-LSTM BID24 BID40 , while BID37 employ REINFORCE BID28 to select useful words sequentially.

Either in tree structure or sequence, the vector representation is built up from the words, which benefits the downstream natural language processing task such as text classification BID22 and natural language inference BID7 .

By contrast, from the top down, our proposed model learns to decompose compound questions into simple questions, which helps to tackle the bottleneck of KBQA piece by piece.

See section 3 for more details.

Natural question understanding has attracted the attention of different communities.

BID15 introduce SequentialQA task that requires to parse the text to SQL which locates table cells as answers.

The questions in SequentialQA are decomposed from selected questions of WikiTableQuestions dataset BID19 by crowdsourced workers while we train an agent to decompose questions using reinforcement learning.

BID25 propose a ComplexWebQuestion dataset that contains questions with compositional semantics while BID3 collects a dataset called ComplexQuestions focusing on multi-constrained knowledge-based question answering.

The closest idea to our work is BID25 which adopts a pointer network to decompose questions and a reading comprehension model to retrieve answers from the Web.

The main difference is that they leverage explicit supervisions to guide the pointer network to correctly decompose complex web questions based on human logic (e.g., conjunction or composition) while we allow the learning-to-decompose agent to discover good partition strategies that benefit downstream task.

Note that it is not necessarily consistent with human intuition or linguistic knowledge.

Without heavy feature engineering, semantic role labeling based on deep neural networks BID9 focus on capturing dependencies between predicates and arguments by learning to label semantic roles for each word. build an end-to-end system which takes only original text information as input features, showing that deep neural networks can outperform traditional approaches by a large margin without using any syntactic knowledge.

BID18 improve the role classifier by incorporating vector representations of both input sentences and predicates.

BID26 handle structural information and long-range dependencies with self-attention mechanism.

This line of work concentrates on improving role classifier.

It still requires rich supervisions for training the role classifier at the token level.

Our approach also requires to label an action for each word, which is similar to role labeling.

However, we train our approach at the sentence level which omits word-by-word annotations.

Our learning-to-decompose agent generates such annotations on the fly by exploring the search space of strategies and increases the probability of good annotations according to the feedback.

FIG0 illustrates an overview of our model and the data flow.

Our model consists of two parts: a learning-to-decompose agent that decomposes each input question into at most three partitions and three identical simple-question answerers that map each partition to its corresponding relation independently.

We refer to the learning-to-decompose agent as the agent and three simple-question answerers as the answerers in the rest of our paper for simplicity.

Our main idea is to best divide an input question into at most three partitions which each partition contains the necessary information for the downstream simple-question answerer.

Given an input A zoom-in version of the lower half of figure 2.

Our agent consists of two components: a Memory Unit and an Action Unit.

The Memory Unit observes current word at each time step t and updates the state of its own memory.

We use a feedforward neural network as policy network for the Action Unit.

question of N words 3 x = {x 1 , x 2 , . . .

, x N }, we assume that a sequence of words is essentially a partially observable environment and we can only observe the corresponding vector representation o t = x t ??? R D at time step t. FIG1 summarizes the process for generating decision of compound question decomposition.

The agent has a Long Short-Term Memory (LSTM; BID14 ) cell unrolling for each time step to memorize input history.

DISPLAYFORM0 where The state s t ??? R 2H of the agent is defined as DISPLAYFORM1 DISPLAYFORM2 which maintained by the above memory cell (Eq. 1) unrolling for each time step. [??, ??] denotes the concatenation of two vectors.

Action Unit The agent also has a stochastic policy network ??(??|s; W ?? ) where W ?? is the parameter of the network.

Specifically, we use a two-layer feedforward network that takes the agent's state s as its input: DISPLAYFORM3 where W(1) DISPLAYFORM4 ?? ??? R 3??H and b DISPLAYFORM5 Following the learned policy, the agent decomposes a question of length N by generating a sequence of actions ?? t ??? {1st, 2nd, 3rd}, t = 1, 2, . . .

, N .

Words under the same decision (e.g. 1st) will be appended into the same sub-sequence (e.g. the first partition).Formally, DISPLAYFORM6 denotes the partitions of a question.

Note that in a partition, words are not necessarily consecutive 4 .

The relative position of two words in original question is preserved.

t 1 + t 2 + t 3 = N holds for every question.

Reward The episodic reward R will be +1 if the agent helps all the answerers to get the golden answers after each episode, or ???1 otherwise.

There is another reward function R = ?? log P (Y * | X) that is widely used in the literature of using reinforcement learning for natural language processing task BID1 BID37 .

We choose the former as reward function for lower variance.

Each unique rollout (sequence of actions) corresponds to unique compound question decomposition.

We do not assume that any question should be divided into exactly three parts.

We allow our agent to explore the search space of partition strategies and to increase the probability of good ones.

The goal of our agent is to learn partition strategies that benefits the answerers the most.

With the help of the learning-to-decompose agent, simple-question answerers can answer compound questions.

Once the question is decomposed into partitions as simple questions, each answerer takes its partition DISPLAYFORM0 } as input and classifies it as the corresponding relation in knowledge graph.

For each partition x (k) , we use LSTM network to construct simple-question representation directly.

The partition embedding is the last hidden state of LSTM network, denoted by x (k) ??? R 2H .

We again use a two-layer feedforward neural network to make prediction, i.e. estimate the likelihood of golden relation r. DISPLAYFORM1 where DISPLAYFORM2 C is the number of classes.

Each answerer only processes its corresponding partition and outputs a predicted relation.

These three modules share no parameters except the embedding layer because our agent will generates conflict assignments for the same questions in different epoches.

If all the answerers share same parameters in different layers, data conflicts undermine the decision boundary and leads to unstable training.

Note that we use a classification network for sequential inputs that is as simple as possible.

In addition to facilitating the subsequent theoretical analysis, the simple-question answerers we proposed are much simpler than good baselines for simple question answering over knowledge graph, without modern architecture features such as bi-directional process, read-write memory BID6 , attention mechanism BID34 or residual connection BID36 .The main reason is that our agent learns to decompose input compound questions to the simplified version which is answerable for such simple classifiers.

This can be a strong evidence for validating the agent's ability on compound question decomposition.

The agent and the answerers share the same embeddings.

The agent can only observe word embeddings while the answerers are allowed to update them in the backward pass.

We train three simple-question answerers separately using Cross Entropy loss between the predicted relation and the golden relation.

These three answerers are independent of each other.

We do not use the pre-train trick for all the experiments since we have already observed consistent convergence on different task settings.

We reduce the variance of Monte-Carlo Policy Gradient estimator by taking multiple (??? 5) rollouts for each question and subtracting a baseline that estimates the expected future reward given the observation at each time step.

The Baseline We follow Ranzato et al. FORMULA0 which uses a linear regressor which takes the agent's memory state s t as input and minimizes the mean squared loss for training.

Such a loss signal is used for updating the parameters of baseline only.

The regressor is an unbiased estimator of expected future rewards since it only depends on the agent's memory states.

Our agent learns a optimal policy to decompose compound questions into simple ones using MonteCarlo Policy Gradient (MCPG) method.

The partitions of question is then feeded to corresponding simple-question answerers for policy evaluation.

The agent takes the final episodic reward in return.

The goal of our experiments is to evaluate our hypothesis that our model discovers useful question partitions and composition orders that benefit simple-question answerers to tackle compound question answering.

Our experiments are three-fold.

First, we trained the proposed model to master the order of arithmetic operators (e.g., + ??? ????) on an artificial dataset.

Second, we evaluate our method on the standard benchmark dataset MetaQA .

Finally, we discuss some interesting properties of our agent by case study.

The agent's ability of compound question decomposition can be viewed as the ability of priority assignment.

To validate the decomposition ability of our proposed model, we train our model to master the order of arithmetic operations.

We generate an artificial dataset of complex algebraic expressions. (e.g. 1 + 2 ??? 3 ?? 4 ?? 5 =? or 1 + (2 ??? 3) ?? 4 ?? 5).

The algebraic expression is essentially a question in math language which the corresponding answer is simply a real number.

Specifically, the complex algebraic expression is a sequence of arithmetic operators including +, ???, ??, ??, ( and ).

We randomly sample a symbol sequence of length N , with restriction of the legality of parentheses.

The number of parentheses is P (??? 2).

The number of symbols surrounding by parentheses is Q. The position of parentheses is randomly selected to increase the diversity of expression patterns.

For example, (+??)+(??) and +??(+??)????? are data points (1+2??3)+(4??5) and 1 + 2 ?? (3 + 4 ?? 5) ??? 6 ?? 7 with N = 8.This task aims to test the learning-to-decompose agent whether it can assign a feasible order of arithmetic operations.

We require the agent to assign higher priority for operations surrounding by parentheses and lower priority for the rest of operations.

We also require that our agent can learn a policy from short expressions (N ??? 8), which generalizes to long ones (13 ??? N ??? 16).We use 100-dimensional (D = 100) embeddings for symbols with Glorot initialization BID12 .

The dimension of hidden state and cell state of memory unit H is 128.

We use the RMSProp optimizer BID27 to train all the networks with the parameters recommended in the original paper except the learning rate ??.

The learning rate for the agent and the answerers is 0.00001 while the learning rate for the baseline is 0.0001.

We test the performance in different settings.

TAB0 summarizes the experiment results.

DISPLAYFORM0 99.21 N = 8, P = 1, Q = 3 N = 13, P = 1, Q = 3 93.37 N = 8, P = 1, Q = 3 N = 13, P = 1, Q = 7 66.42The first line indicates that our agent learns an arithmetic skill that multiplication and division have higher priority than addition and subtraction.

The second line indicates that our agent learns to discover the higher-priority expression between parentheses.

The third line, compared to the second line, indicates that increasing the distance between two parentheses could harm the performance.

We argue that this is because of the Long Short-Term Memory Unit of our agent suffers when carrying the information of left parenthesis for such a long distance.

We evaluate our proposed model on the test set of two challenging KBQA research dataset, i.e., WebQuestions BID5 and MetaQA .

Each question in both datasets is labeled with the golden topic entity and the inference chain of relations.

The statistics of MetaQA dataset is shown in table 2.

The number of compound questions in MetaQA is roughly twice that of simple questions.

The max length of training questions is 16.

The size of vocabulary in questions is 39,568.

The coupled knowledge graph contains 43,234 entities and 9 relations.

We also augmented the relation set with the inversed relations, as well as a "NO OP" relation as placeholder.

The total number of relations we used is 14 since some inversed relations are meaningless.

WebQuestions contains 2,834 questions for training, 944 questions for validation and 2,032 questions for testing respectively.

We use 602 relations for the relation classification task.

The number of compound questions in WebQuestions is roughly equal to that of simple questions.

Note that a compound question in WebQuestions is decomposed into two partitions since the maximum number of corresponding relations is two.

One can either assume topic entity of each question is linked or use a simple string matching heuristic like character trigrams matching to link topic entity to knowledge graph directly.

We use the former setting while the performance of the latter is reasonable good.

We tend to evaluate the relation detection performance directly.

For both datasets, we use 100-dimensional (D = 100) word embeddings with Glorot initialization BID12 .

The dimension of hidden state and cell state of memory unit H is 128.

We use the RMSProp optimizer BID27 to train the agent with the parameters recommended in the original paper except the learning rate.

We train the rest of our model using Adam BID16 with default parameters.

The learning rate for all the modules is 0.0001 no matter the optimizer it is.

We use four samples for Monte-Carlo Policy Gradient estimator of REINFORCE.

The metric for relation detection is overall accuracy that only cumulates itself if all relations of a compound question are correct.

TAB2 presents our results on MetaQA dataset.

The last column for total accuracy is the most representative for our model's performance since the only assumption we made about input questions is the number of corresponding relations is at most three.

BID36 on this dataset focus on leveraging information of the name of Freebase relation while we are only using question information for classification.

We assume that the compound question can be decomposed into at most three simple questions.

In practice, this generalized assumption of answerable questions is not necessary.

One example is that WebQuestions only contains compound questions corresponding to two but not three relations.

It indicates that people tend to ask less complicated questions more often.

So we conduct an ablation study for the hyperparameters of this central assumption in our paper.

We assume that all the questions in MetaQA dataset contain at most two corresponding relations.

We run the same code with the same hyperparameters except we only use two simple-question answerers.

The purpose of the evaluation is to prove that our model improves performance on 1-hop and 2-hop questions by giving up the ability to answer three-hop questions.

TAB4 presents our results on ablation test.

We can draw a conclusion that there exists a trade-off between answering more complex questions and achieving better performance by limiting the size of search space.

Figure 4 illustrates a continuous example of figure 1 for the case study, which is generated by our learning-to-decompose agent.

Assuming the topic entity e is detected and replaced by a placeholder, the agent may discover two different structures of the question that is consistent with human intuition.

Since the knowledge graph does not have a movie-to-movie relation named "share actor with", the lower partition can not help the answerers classify relations correctly.

However, the upper partition will be rewarded.

As a result, our agent optimizes its strategies such that it can decompose the original question in the way that benefits the downstream answerers the most.

We observe the fact that our model understands the concept of "share" as the behavior "take the inversed relation".

That is, "share actors" in a question is decomposed to "share" and "actors" in two partitions.

The corresponding formulation is g(e) = f 2 (f 1 (e)) = (f 2 ??? f 1 )(e).

We observe the same phenomenon on "share directors".

We believe it is a set of strong evidence for supporting our main claims.

Understanding compound questions, in terms of The Principle of Semantic Compositionality BID20 , require one to decompose the meaning of a whole into the meaning of parts.

While previous works focus on leveraging knowledge graph for generating a feasible path to answers, we Figure 4 : A continuous example of figure 1.

The hollow circle indicates the corresponding action the agent takes for each time step.

The upper half is the actual prediction while the lower half is a potential partition.

Since we do not allow a word to join two partitions, the agent learns to separate "share" and "actors" into different partitions to maximize information utilization.propose a novel approach making full use of question semantics efficiently, in terms of the Principle of Semantic Compositionality.

In other words, it is counterintuitive that compressing the whole meaning of a variable-length sentence to a fixed-length vector, which leaves the burden to the downstream relation classifier.

In contrast, we assume that a compound question can be decomposed into three simple questions at most.

Our model generates partitions by a learned policy given a question.

The vector representations of each partition are then fed into the downstream relation classifier.

While previous works focus on leveraging knowledge graph for generating a feasible path to answers, we propose a novel approach making full use of question semantics efficiently, in terms of the Principle of Semantic Compositionality.

Our learning-to-decompose agent can also serve as a plug-and-play module for other question answering task that requires to understand compound questions.

This paper is an example of how to help the simple-question answerers to understand compound questions.

The answerable question assumption must be relaxed in order to generalize question answering.

<|TLDR|>

@highlight

We propose a learning-to-decompose agent that helps simple-question answerers to answer compound question over knowledge graph.