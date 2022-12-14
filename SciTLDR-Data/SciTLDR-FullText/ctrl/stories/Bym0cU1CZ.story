Conventional methods model open domain dialogue generation as a black box through end-to-end learning from large scale conversation data.

In this work, we make the first step to open the black box by introducing dialogue acts into open domain dialogue generation.

The dialogue acts are generally designed and reveal how people engage in social chat.

Inspired by analysis on real data, we propose jointly modeling dialogue act selection and response generation, and perform learning with human-human conversations tagged with a dialogue act classifier and a reinforcement approach to further optimizing the model for long-term conversation.

With the dialogue acts, we not only achieve significant improvement over state-of-the-art methods on response quality for given contexts and long-term conversation in both machine-machine simulation and human-machine conversation, but also are capable of explaining why such achievements can be made.

Conversational agents are becoming ubiquitous recently.

Through human-machine conversation, such agents either help users complete specific tasks BID39 or engage them in social chat BID30 .

Depending on application scenarios, various conversational agents have been designed including chatbots, personal assistants, and automated customer service, etc.

Traditional research on conversational agents focuses on task-oriented dialogue systems BID39 where task specific dialogue acts are handcrafted in a form of slot-value pairs.

On the one hand, through slot-filling, the dialogue acts make conversations in such systems interpretable and controllable; on the other hand, they also hinder scaling such systems to new domains.

To escape from the limitation, recent interest of research moves to end-to-end dialogue learning without any assumptions on dialogue acts.

Most of the effort is paid to non-task-oriented chit-chat BID30 , and there are also a few studies on task-oriented dialogues BID1 BID4 .

Without dialogue acts, these work directly constructs a response by learning from large scale data with neural networks, and thus is easy to scale to new domains.

On the other hand, due to the absence of dialogue acts, it is hard to interpret the emergence of a response to a dialogue context and predict where the conversation will flow to.

In this work, we aim to achieve interpretability and controllability in non-task-oriented dialogues.

To this end, we introduce dialogue acts into open domain dialogue generation.

Open domain dialogue generation has been widely applied to chatbots which aim at engaging users by keeping conversation going.

Existing work concentrates on generating relevant and diverse responses for a static context.

However, it is not clear if relevance and diversity are sufficient to engagement in dynamic interactions.

Therefore, we investigate the following problems: (1) if we can properly design dialogue acts that can enable us to understand engagement in human-human open domain conversation; (2) how to learn a dialogue generation model with the dialogue acts; and (3) how the model performs in practice and if the performance can be explained by the dialogue acts.

To examine how people engage in social chat, we establish a general dialogue act taxonomy for open domain conversation by extending the existing work with high-level dialogue acts regarding to conversational context.

The taxonomy, when applied to real data, gives rise to an interesting finding that in addition to replying with relevance and diversity , people are used to driving their social chat by constantly switching to new contexts and properly asking questions.

Such behaviors are less explored before, and thus are difficult for the existing end-to-end learning methods to imitate.

To mimic human behaviors, we propose jointly modeling dialogue act selection and response generation in open domain dialogue generation.

The dialogue model is specified with neural networks.

We propose learning from human-human interactions by fitting the model to large scale real world dialogues tagged with a dialogue act classifier and further optimizing the policy of act selection for long-term conversation through a reinforcement learning approach.

Our model enjoys several advantages over the existing models: (1) the dialogue acts provide interpretation to response generation from a discourse perspective; (2) the dialogue acts enhance diversity of responses by expanding the search space from language to act ?? language; (3) the dialogue acts manage the flow of humanmachine conversations and thus enhance human engagement; and (4) the dialogue act selection is compatible with post-engineering work (e.g., combination with rules), and thus allows engineers to flexibly control their systems through picking responses from their desired dialogue acts.

Evaluation results on large scale test data indicate that our model can significantly outperform state-of-the-art methods in terms of quality of generated responses regarding to given contexts and lead to longterm conversation in both machine-machine simulation and human-machine conversation in a way similar to how human behave in their interactions.

Our contributions in this work include: (1) design of dialogue acts that represent human behavior regarding to conversational context and insights from analysis of human-human interactions with the design; (2) joint modeling of dialogue act selection and response generation in open domain dialogue generation; (3) proposal of a supervised learning approach and a reinforcement learning approach for model optimization; (4) empirical verification of the effectiveness of the model through automatic metrics, human annotations, machine-machine simulation, and human-machine conversation.

A response or an answer to the previous utterances in the current context.

"this summer." after "when are you going to Tokyo?".

Similar to CM.S, but the user or the bot tries to switch to a new context (e.g., topic) by bringing in new content."I plan to study English this summer.

" after "I plan to have a tour in Tokyo this summer.

".

Context Switch Question (CS.Q) A user or a bot tries to change the context of conversation by asking a question. "

When will your summer vacation start?

" after "I plan to have a tour in Tokyo this summer.

"

The utterance not only replies to the previous turn, but also starts a new topic.

"

I don't know because I have to get an A+ in my math exam." after "when are you going to Tokyo?".

Others (O) greetings, thanks, and requests, etc.. "thanks for your help."

We first define dialogue acts, and then describe the data for learning and the insights we obtain from the data.

Finally, we elaborate how we build the classifier with neural networks.

Our dialogue acts are inherited from the existing work on 1-on-1 live chats and twitter BID9 BID6 .

Similar to BID21 , we organize the 12 acts in BID6 which originate from the 42 tags BID8 BID29 based on the DAMSL annotation scheme (Core & Allen, 1997) into high-level dialogue acts: "statement" and "expressive" are merged as "statement"; "yes-no question" and "open question" are combined as "question"; "yes-answer", "no-answer", and "response-ack" are collapsed as "answer"; and other tags are treated as "others".

On top of these acts, we further define two high-level dialogue acts that describe how people behave regarding to conversational context in their interactions.

As will be seen later, the extension may bring us further insights on engagement in social chat.

Details of the dialogue acts are described in TAB0 .The high-level dialogue acts in TAB0 To resolve these problems, we build a data set.

We crawled 30 million dyadic dialogues (conversations between two people) from Baidu Tieba.

Baidu Tieba is the largest Reddit-like forum in China which allows users to post and comment on others' post.

Two people can communicate with each other through one posting a comment and the other one replying to the comment.

Data in Baidu Tieba covers a large variety of topics, and thus can be viewed as a simulation of open domain conversation in a chatbot.

We randomly sample 9 million dialogues as a training set, 90 thousand dialogues as a validation set, and 1000 dialogues as a test set.

These data are used to learn a dialogue generation model later.

We employ the Standford Chinese word segmenter 2 to tokenize utterances in the data.

TAB2 reports statistics of the data.

For dialogue act learning, we randomly sample 500 dialogues from the training set and recruit 3 native speakers to label dialogue acts 3 for each utterance according to the definitions in TAB0 .

TAB3 shows a labeling example from one annotator.

Each utterance receives 3 labels, and the Fleiss' kappa of the labeling work is 0.45, indicating moderate agreement among the labelers.

The frequencies of the dialogue acts in terms of percentages of the total number of utterances in the labeled data are CM.S 55.8%, CM.Q 11.7%, CM.A 12.2%, CS.S 12.4%, CS.Q 4.8%, CS.A 2%, and O 1.1%.

In addition to the numbers, we also get further insights from the data that are instructive to our dialogue generation learning:Context switch is a common skill to keep conversation going.

In fact, we find that 78.2% dialogues contain at least one CS.

* act.

The average number of turns of dialogues that contain at least one CS.

* is 8.4, while the average number of turns of dialogues that do not contain a CS.

* is 7.

When dialogues are shorter than 5 turns, only 47% of them contain a CS.

*, but when dialogues exceed 10 turns, more than 85% of them contain a CS.*. Because there are no specific goals in their conversations, people seldom stay long in one context.

The average number of turns before context switch is 3.39.

We also observed consecutive context switch in many dialogues (43.7%).

The numbers suggest dialogue generation with smooth context switch and moderate context maintenance.

Question is an important building block in open domain conversation.

In fact, 13.9% CM.

* are CM.Q and the percentage is even higher in CS.

* which is 20.27%.

People need to ask questions in order to maintain contexts.

The average number of turns of contexts with questions (i.e., consecutive CM.

* with at least one CM.Q) is 3.92, while the average number of turns of contexts without questions is only 2.95.

The observation indicates that a good dialogue model should be capable of asking questions properly, as suggested by BID15 .

A further step to study human's questioning behavior is to look into types and functions of questions.

We leave it as future work.

The observations raise new challenges that are difficult for the existing end-to-end methods to tackle (e.g., smoothly interleaving context blocks with switch actions), and thus encourage us to create a new model.

Note that these observations may relate to dialogue scenarios (e.g., chatting online instead of face-to-face) and cultures, but we ignore these factors and just study how to learn the conversational patterns from the data with a principled approach.

The learning approach is generally applicable to other data.

To perform learning, we need to build a classifier that can automatically tag large scale dialogues with dialogue acts.

We aim to learn a classifier c from DISPLAYFORM0 } represents a dialogue with u i,k the k-th utterance and a i,k the labeled dialogue act.

Given a new dialogue d = {u 1 , . . . , u n }, c can sequentially tag the utterances in d with dialouge acts by taking u i , u i???1 , and the predicted a i???1 as inputs and outputting a vector c(u i , u i???1 , a i???1 ) where the j-th element representing the probability of u i being tagged as the j-th dialogue act.

We parameterize c(??, ??, ??) using neural networks.

Specifically, u i and u i???1 are first processed by bidirectional recurrent neural networks with gated recurrent units (biGRUs) BID2 respectively.

Then the last hidden states of the two biGRUs are concatenated with an embedding of a i???1 and fed to a multi-layer perceptron (MLP) to calculate a dialogue act distribution.

Formally, suppose that u i = (w i,1 , . . .

, w i,n ) where w i,j is the embedding of the j-th word, then the j-th hidden state of the biGRU is given by DISPLAYFORM1 is the j-th state of a forward GRU, ??? ??? h i,j is the j-th state of a backward GRU, and DISPLAYFORM2 Similarly, we have h i???1,j as the j-th hidden state of u i???1 .

Let e(a i???1 ) be the embedding of a i???1 , then c(u i , u i???1 , a i???1 ) is defined by a two-layer MLP: DISPLAYFORM3 where we pad zeros for u 0 and a 0 in c(u 1 , u 0 , a 0 ).

We learn c(??, ??, ??) by minimizing cross entropy with D A .

Let p j (a i ) be the probability of a i being the j-th dialogue act and c(u i , u i???1 , a i???1 )[j] be the j-th element of c(u i , u i???1 , a i???1 ), then the objective function of learning is formulated as We randomly split the labeled dialogues as 400/30/70 dialogues with 3280/210/586 utterances for training/validation/test.

Details of learning are given in Appendix 7.2.

The learned classifier achieves an accuracy of 70.1% on the test data.

We employ it to tag the training, validation, and test sets in TAB2 .

DISPLAYFORM4

We present dialogue generation learning using large scale dialogues tagged with dialogue acts.

Then, we describe model optimization with reinforcement learning for long-term conversation.

We aim to learn a dialogue generation model FIG0 , . . .

, (u i,ni , a i,ni )} refers to a human-human dialogue with u i,k the k-th utterance and a i,k the dialogue act tagged by the classifier in Section 2.4.

Given s i = {(u 1 , a 1 ), . . . , (u i???1 , a i???1 )} as a new dialogue session, g(s i ) can generate a response as the next turn of the dialogue.

DISPLAYFORM0 Our dialogue model consists of a policy network and a generation network.

A dialogue act is first selected from the policy network according to the conversation history, and then a response is generated from the generation network based on the conversation history and the dialogue act.

Formally, the dialogue model can be formulated as DISPLAYFORM1 where a i = O(p a (a i |s i )) is the selected dialogue act for the i-th turn, and r i is the response of the i-th turn.

p a is the policy network and p r is the generation network.

O(??) refers to a dialogue act select operation according to the value of the policy network.

DISPLAYFORM2 where A is the space of dialogue acts.

One can also customize O(??) with more complicated rules to achieve controllability or further optimization (e.g., improving response diveristy by selecting multiple acts) of their systems.

Figure 1(b) shows the architecture of the policy network.

The utterance sequence and the act sequence are encoded with a hierarchical encoder and a GRU encoder respectively.

Then, the last hidden states of the two encoders are concatenated and fed to an MLP to calculate a probability distribution of dialogue acts for the next turn.

Formally, ???u j ??? s i , u j is first transformed to hidden DISPLAYFORM3 k=1 through a biGRU parameterized as Equation (1).

Then, {h DISPLAYFORM4 j=1 is processed by a GRU parameterized as DISPLAYFORM5 We build the generation network in a sequence-to-sequence framework.

Here, we simplify p r (r i |s i , a i ) as p r (r i |a i , u i???1 , u i???2 ) since decoding natural language responses from long conversation history is challenging.

FIG0 illustrates the architecture of the generation network.

The only difference from the standard encoder-decoder architecture with an attention mechanism is that in encoding, we concatenate u i???1 and u i???2 , and attach a i to the top of the long sentence as a special word.

The technique here is similar to that in zero-shot machine translation BID7 .

More formulation details can be found in Appendix 7.1.The dialogue model is then learned by minimizing the negative log likelihood of D: DISPLAYFORM6 where DISPLAYFORM7 Through supervised learning, we fit the dialogue model to human-human interactions in order to learn their conversational patterns.

However, supervised learning does not explicitly encourage long-term conversation (e.g., 45.35% dialogues in our training set are no more than 5 turns).

The policy network is learned by fitting to the existing conversation history, and it is not aware what is going to happen in the future when a dialogue act is selected.

This motivates us to further optimize the model through a reinforcement learning approach.

We aim to optimize the dialogue model by letting it know a possible result in the following conversation when an act and a response are generated.

To avoid exhausting and expensive online optimization, we choose self-play BID14 BID11 where we let two models learned with the supervised approach talk to each other in order to improve their performance.

In the simulation, a dialogue is initialized with a message sampled from the training set.

Then, the two models continue the dialogue by alternately taking the conversation history as an input and generating a response (top one in beam search) until T turns (T = 20 in our experiments).To speed up training and avoid generated responses diverging from human language, we fix the generation network and only optimize the policy network by reinforcement learning.

Thus, the policy in learning is naturally defined by the policy network p a (a i |s i ) with s i = {(u 1 , a 1 ), . . .

, (u i???1 , a i???1 )} a state and a i an action.

We define a reward function r(a i , s i ) as DISPLAYFORM0 where E[len(a i , s i )] is the expected conversation length after taking action a i under state s i , E[rel(a i , s i )] is the expected response relevance within the conversation, ?? = 0.67, and ?? = 0.33.

Through Equation FORMULA13 (N = 10 in our experiments) by sampling after (s i , a i ) with self-play.

???j, d i,j = (s i , u j,i+1 , . . . , u j,ni,j ) where ???k, u j,i+k is randomly sampled from the top 5 beam search results of p r conditioned on the most probable dialogue act given by p a for that turn.

Inspired by BID14 , we terminate a simulated dialogue if (1) cosine(e(u i???1 ), e(u i )) > 0.9 && cosine(e(u i )), e(u i+1 )) > 0.9, or (2) cosine(e(u i???1 ), e(u i+1 )) > 0.9, or (3) the length of the dialogue reaches T , where e(??) denotes the representation of an utterance given by the encoder of p r .

Condition (1) means three consecutive turns are (semantically) repetitive, and Condition (2) means one agent gives repetitive responses in two consecutive turns.

Both conditions indicate a high probability that the conversation falls into a bad infinite loop.

E[len(a i , s i )] and E[rel(a i , s i )] are then estimated by DISPLAYFORM1 where d i,j<k = (u 1 , . . .

, u j,k???1 ), and m(??, ??) is the dual LSTM model proposed in BID18 which measures the relevance between a response and a context.

We train m(??, ??) with the 30 million crawled data through negative sampling.

The objective of learning is to maximize the expected future reward: DISPLAYFORM2 The gradient of the objective is calculated by Reinforce algorithm BID35 : DISPLAYFORM3 where the baseline b t is empirically set as 1 |A| at???A r(a t , s t ).

Our experiments are conducted with the data in TAB2 .

The following methods are employed as baselines: (1) S2SA: sequence-to-sequence with attention BID0 in which utterances in contexts are concatenated as a long sentence.

We use the implementation with Blocks (https://github.com/mila-udem/blocks); (2) HRED: the hierarchical encoder-decoder model in implemented with the source code available at (https://github.com/julianser/hed-dlg-truncated); (3) VHRED: the hierarchical latent variable encoder-decoder model in BID26 implemented with the source code available at (https://github.com/julianser/hed-dlg-truncated); and (4) RL-S2S: dialogue generation with reinforcement learning BID14 .

We implement the algorihtm by finishing the code at (https://github.com/liuyuemaicha/ Deep-Reinforcement-Learning-for-Dialogue-Generation-in-tensorflow).Dull responses are defined as in BID14 and listed in Appendix 7.3.All baseline models are implemented with the recommended configurations in the existing literatures.

We denote our Dialogue Act aware Generation Model with only Supervised Learning as SL-DAGM, and the full model (supervised learning + reinforcement learning) as RL-DAGM.

Implementation details are given in Appendix 7.3.

The first experiment is to check if the proposed models can generate high-quality responses regarding to given contexts.

To this end, we take the last turn of each test dialogue as ground truth, and feed the previous turns as a context to different models for response generation.

Top one responses from beam search (beam size= 20) of different models are collected, randomly shuffled, and presented to 3 native speakers to judge their quality.

Each response is rated by the three annotators under the following criteria: 2: the response is not only relevant and natural, but also informative and interesting; 1: the response can be used as a reply, but might not be informative enough (e.g.,"Yes, I see" etc.); 0: the response makes no sense, is irrelevant, or is grammatically broken.

TAB5 (a) summarizes the annotation results.

Improvements from our models over the baseline methods are statistically significant (t-test, p-value < 0.01).

In addition to human annotations, we also compare different models using automatic metrics with the the ground truth.

These metrics include (1) BLEU BID22 which measures term overlap of two responses; (2) embedding based metrics BID17 such as Embedding Average (Average), Embedding Extrema (Extrema), and Embedding Greedy (Greedy) which measure similarity of two responses in a semantic space; and (3) ratios of distinct unigrams (distinct-1) and bigrams (distinct-2) in the generated responses which are employed in to measure response diversity.

TAB6 reports the automatic evaluation results.

We can see that one benefit brought by the dialogue acts is that diversity of responses is significantly improved.

This is supported by the much more 2 responses from the two models in TAB5 (a) and the significant improvement on distinct n-grams in TAB6 .

The reason is easy to understand: we search a response not only from a language space, but also from an act space.

The dimension of dialogue acts provides further variations to the generated responses.

On the other hand, due to the diversity, responses from our models may diverge from the ground truth sometimes.

This is why improvements on other automatic metrics are not significant.

To further explain the advantages of our models, we show an example in TAB7 .

In addition to responses from SL-DAGM and RL-DAGM which are selected from the dialogue acts obtained by Equation FORMULA7 , we also show responses from other reasonable but not selected acts.

With dialogue acts, responses from our models become really rich, from confirmation (CM.Q) to an open question (CS.Q) and then to a long informative statement (CS.S).

More importantly, the dialogue acts let us know why we have such responses: both SL-DAGM and RL-DAGM try to switch to new topics (e.g., Xiamen, noodle, and plan etc.) in order to continue the conversation.

One can also change the flow of the conversation by picking responses from other dialogue acts.

The example demonstrates that in addition to good performance, our models enjoy good interpretability and controllability as well.

We show more such examples in Appendix 7.4.

DISPLAYFORM0

Secondly, we study conversation engagement with the proposed models.

Experiments are conducted through machine-machine simulation and human-machine conversation.

In both experiments, we compare SL-DAGM and RL-DAGM with RL-S2S, as RL-S2S is the only baseline optimized for future success.

Responses from all models are randomly sampled from the top 5 beam search results.

Average length of dialogues is employed as an evaluation metric.

Machine-machine simulation is conducted in a way similar to BID14 ) in which we let two bots equipped with the same model talk with each other in 1000 simulated dialogues.

Each dialogue is initialized with the first utterance of a test example, and terminated according to the termination conditions for reward estimation in Section 3.2.

In human-machine conversation, we recruit 5 native speakers as testers and ask them to talk with the bots equipped with the three models.

Every time, a bot is randomly picked for a tester, and the tester does not know which model is behind.

Every tester finishes 100 dialogues with each bot.

To make a fair comparison, we let the bots start dialgoues.

A starting message in a dialogue is randomly sampled from the test data and copied 3 times for all the 3 bots (a tester can skip the message if he/she cannot understand it).

A dialogue is terminated if (1) the tester thinks the conversation cannot be continued (e.g., due to bad relevance or repetitive content etc.); or (2) the bot gives repetitive responses in two consecutive turns (measured by cosine(e(u i???1 ), e(u i+1 )) > 0.9).

Dialogue acts in human turns are tagged by the classifier in Section 2.4.

The evaluation metric is calculated with the total 500 dialogues for each model.

TAB5 (b) reports the evaluation results.

In both experiments, SL-DAGM and RL-DAGM can lead to longer conversations, and the improvements from both models over the baseline are statistically significant (t-test, p-value < 0.01).

Improvements in human-machine conversation are smaller than those in machine-machine simulation, indicating the gap between the simulation environment and the real conversation environment and encouraging us to consider online optimization in humanmachine conversations in the future.

RL-DAGM is better than SL-DAGM in both experiments, indicating the efficacy of reinforcement learning.

The reason that our models are better is that they captured conversational patterns in human-human interactions and obtained further optimization through reinforcement learning.

First, the models can pro-actively switch contexts in a smooth way.

In machine-machine simulation, 65.4% (SL) and 94.4% (RL) dialogues contain at least one CS.

*; and in human-machine conversation, the two percentages are 38.1% (SL) and 48.1% (RL) respectively.

More interestingly, in machine-machine simulation, average lengths of dialogues without CS.

* are only 4.78 (SL) and 2.67 (RL) respectively which are comparable with or even worse than RL-S2S, while average lengths of dialogues with CS.

* are 8.66 (SL) and 8.18 (RL) respectively.

The results demonstrate the importance of context switch for engagement in open domain conversation and one signficant effect of RL is promoting context switch in interactions for future engagment even with a little sacrifice on relevance of the current turn (e.g., more 0 responses than SL-DAGM in TAB5 (a)).

Second, the models can drive conversations by asking questions.

In machine-machine simulation, 36.5% (SL) and 32.4% (RL) dialogues contain at least one question.

The percentages in human-machine conversation are 17.7% (SL) and 22.3% (RL) respectively.

We give more analysis in Appendix 7.5.

Finally, we study how the generated responses are affected by the dialogue acts.

We collect generated responses from a specific dialogue act for the contexts of the test dialogues, and characterize the responses with the following metrics: (1) distinct-1 and distinct-2; (2) words out of context (OOC): ratio of words that are in the generated responses but not contained by the contexts; and (3) average length of the generated responses (Ave Len).

TAB9 reports the results 4 .

In general, responses generated from CS.

* are longer, more informative, and contain more new words than responses generated from CM.

*, which has been illustrated by the example in TAB7 .

Another interesting finding is that statements and answers are generally more informative than questions in both CS.

* and CM.*. In addition to these metrics, we also calculate BLEU scores and embedding based metrics, but do not observe significant difference among responses from different dialogue acts.

The reason might be that these metrics are based on comparsion of the generated responses and human responses, but human responses in the test set are inherently mixture of responses from different dialogue acts.

Existing dialogue models are either built for open domain conversation or for specific task completion.

Regarding to the former, a common practice is to learn a generation model in an end-to-end fashion.

On top of the basic sequence-to-sequence with attention architecture BID30 BID27 , various extensions have been proposed to tackle the "safe response" problem BID19 BID37 ; to model complicated structures of conversation contexts BID28 ; to bias responses to some specific persona or emotions BID13 ; and to pursue better optimization strategies BID16 BID14 .

On the other line of research, POMDP BID39 breaks down the development of task-oriented dialogue systems into natural language understanding BID38 BID5 , dialogue management BID20 , and response generation BID32 .

Recently, researchers also consider learning task-oriented dialogue models in an end-to-end way BID1 .

In this work, we introduce dialogue acts into open domain dialogue generation.

Although some previous work BID41 BID24 has leveraged dialogue acts as extra features, the dialogue acts in this work are generally designed for explaining engagement in social chat and modelled as policies to manage the flow of interactions.

To the best of our knowledge, we are the first who design dialogue acts to explain social interactions, control open domain response generation, and guide human-machine conversations.

Before us, some researchers have proposed analyzing open domain dialogues with dialogue acts BID9 BID21 BID6 BID31 BID36 BID23 .

These work, however, stops at performing utterance classification or clustering.

Our dialogue act design is inspired by these work, but we not only exploit the dialogue acts to interpret open domain dialogues, but also conduct dialogue generation with the dialogue acts.

We study open domain dialogue generation with generally designed dialogue acts that can describe human behavior in social interactions.

To mimic such behavior, we propose jointly modeling dialogue act selection and response generation, and perform both supervised learning with a learned dialogue act classifier and reinforcement learning for long-term conversation.

Empirical studies on response generation for given contexts, machine-machine simulation, and human-machine conversation show that the proposed models can significantly outperform state-of-the-art methods.

Suppose that x i = [a i ; u i???1 ; u i???2 ] = (w i,1 , . . .

, w i,n i ) where w i,k is the embedding of the k-th word, then the k-th hidden state of the encoder is given by DISPLAYFORM0 Positions of u ???1 and u 0 in x 1 and x 2 are padded with zeros.

Let r i = (w i,1 , . . .

, w i,T ), then in decoding the j-th word w i,j , {v i,1 , . . .

, v i,n i } is summarized as a context vector c i,j through an attention mechanism: DISPLAYFORM1 where v and W ?? are parameters, and v i,j???1 is the (j ??? 1)-th hidden state of the decoder GRU in which v i,j is calculated by DISPLAYFORM2 The generation probability of w i,j is then defined as DISPLAYFORM3 where I(w i,j ) is a vector with only one element 1 indicating the index of w i,j in the vocabulary.

DISPLAYFORM4

We randomly split the 500 labeled dialogues as 400, 30, and 70 dialogues for training, validation, and test respectively.

Utterances in the three sets are 3280, 210, and 586 respectively.

In training, we represent dialogue acts as probability distributions by averaging the labels given by the three annotators.

For example, if an utterance is labeled as "CM.S", "CM.S", and "CS.S", then the probability distribution is (0.67, 0, 0, 0.33, 0, 0, 0) .

In test, we predict the dialogue act of an utterance u i by arg max j g(u i , u i???1 , a i???1 ) [j] .

To avoid overfitting, we pre-train word embeddings using word2vec 5 with an embedding size of 200 on the 30 million data and fix them in training.

We set the embedding size of the dialogue acts and the hidden state size of the biGRUs as 100, and the dimensions of the first layer and the second layer of the MLP as 200 and 7 respectively.

We optimize the objective function (3) using back-propagation and the parameters are updated by stochastic gradient descent with AdaDelta algorithm BID40 .

The best performing model on the validation data is picked up for test.

In learning of the generation network, we set the size of word embedding as 620 and the size of hidden vectors as 1024 in both the encoder and the decoder.

Both the encoder vocabulary and the decoder vocabulary contain 30, 000 words.

Words out of the vocabularies are replaced by a special token "UNK".

We employ AdaDelta algorithm BID40 to train the generation network with a batch size 128.

We set the initial learning rate as 1.0 and reduce it by half if perplexity on validation begins to increase.

We stop training if the perplexity on validation keeps increasing in two successive epochs.

In learning of the policy network, we set the size of word embedding, the size of dialogue act, and the size of hidden states of the biGRU as 100.

There are 50 neurons in the first layer of the MLP and 7 neurons in the second layer of the MLP.

Vectors in the policy network have smaller sizes than 5 https://code.google.com/archive/p/word2vec/ those in the generation network because the complexity of dialogue act prediction is much lower than language generation.

In reinforcement learning, the size of mini-batch is 60 and learning rate is fixed as 0.05.

To estimate the reward, we train a dual LSTM BID18 with the size of word embedding and the size of hidden states as 100.

Responses from the simulated dialogues are generated with a beam size 20.In RL-S2S, we define 8 responses as dull responses according to the frequency of responses in the training set.

TAB10 gives the responses.

We compare SL-DAGM and RL-DAGM with baseline models in terms of response quality for given contexts with more examples in TAB11 .

TAB0 gives some examples on machine-machine simulation.

Unlike the dialogues from RL-S2S which quickly converge to loops, dialogues from our models smoothly move forward under the management of the dialogue acts.

The dialogue acts let us know why such responses are generated and make the simulated dialogues closer to human dialogues with moderate context continuation and jumping out of the contexts at proper timing.

TAB0 show some examples from the test of human-machine conversation.

We denote a machine turn as "M" and a human turn as "H".

After each example, we give the reason of termination in which "EOD-H" means the dialogue is terminated by the tester and "EOD-R" means the dialogue is terminated by the repetition check with the next generated turn attached.

Compared to dialogues with the baseline, dialogues with our models can go deeper with much richer content, although a side-effect is that sometimes responses from CS.

* might be nonsense (e.g., the first example of SL-DAGM).

This sheds light on our future direction to further improve the generation network with knowledge.

In addition to the qualitative results, we also show quantitative results of human-machine conversation test in terms of different testers in FIG1 .

Although there exists variance among the testers, the overall trend is consistent with the numbers in TAB5 (b).

TAB0 : Example 2 of human-machine conversation.

"M" means a machine turn, and "H" means a human turn.

<|TLDR|>

@highlight

open domain dialogue generation with dialogue acts

@highlight

The authors use a distant supervision technique to add dialogue act tags as a conditioning factor for generating responses in open-domain dialogues

@highlight

The paper describes a technique to incorporate dialog acts into neural conversational agents