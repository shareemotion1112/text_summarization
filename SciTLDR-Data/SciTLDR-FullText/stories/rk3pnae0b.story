Asking questions is an important ability for a chatbot.

This paper focuses on question generation.

Although there are existing works on question generation based on a piece of descriptive text, it remains to be a very challenging problem.

In the paper, we propose a new question generation problem, which also requires the input of a target topic in addition to a piece of descriptive text.

The key reason for proposing the new problem is that in practical applications, we found that useful questions need to be targeted toward some relevant topics.

One almost never asks a random question in a conversation.

Due to the fact that given a descriptive text, it is often possible to ask many types of questions, generating a question without knowing what it is about is of limited use.

To solve the problem, we propose a novel neural network that is able to generate topic-specific questions.

One major advantage of this model is that it can be trained directly using a question-answering corpus without requiring any additional annotations like annotating topics in the questions or answers.

Experimental results show that our model outperforms the state-of-the-art baseline.

The goal of question generation is to generate questions according to some given information (e.g., a sentence or a paragraph).

It has been applied in many scenarios, e.g., generating questions for reading comprehension BID8 and generating data for large scale question-answering training (Serban et al., 2016; BID7 .

Since questioning is an important communication skill, question generation plays an important role in both general-purpose chatbot systems and goal-oriented dialogue systems.

In the context of dialogue, many researchers have studied the problem BID15 BID1 .

The generated questions are mainly used to start a conversation or to obtain some specific information.

Earlier approaches to question generation mainly used human-crafted rules and patterns to transform a descriptive sentence to a related question BID14 BID2 .

However, human-crafted rules are limited.

They cannot effectively cover a large number of question generation scenarios.

Deep neural networks learned by end-to-end methods can help overcome this problem.

This approach has been successfully applied to many NLP tasks, e.g., neural machine translation BID0 BID19 , summarization BID9 , etc.

At the same time, some training optimization studies have further guaranteed the performance and stability of end-to-end networks BID13 BID4 .

For question generation, Serban et al. (2016) applied a neural network to the knowledge base Freebase to transduce facts to questions.

introduced an attention-based sequence learning model, which outperformed state-of-the-art rule-based systems.

Two approaches were proposed by BID7 .

One is retrieval-based, and the other is generation-based.

also studied question-worthy sentences in reading comprehension.

Much of the existing question generation studies mainly focused on the task of generating questions from sentences, paragraph, or structured data only.

In this paper, we argue that topic-based question generation is also very important.

That is, in addition to the given sentence or paragraph, it is also useful to specify a relevant topic contained in the text.

The main reason is that a sentence or paragraph often involves multiple topics or concepts that questions can be generated, only arbitrarily choose one or mixing them may be of limited use because we found that in practical applications, questions need to be targeted toward some topics related to the current conversation.

One almost never asks a random question in a conversation.

Generating a question without knowing what it is about is not very useful.

To solve the proposed problem, we propose a novel neural network that can generate topic-based questions.

One major advantage of our model is that it can be trained directly using a question-answering corpus without requiring any additional annotations like annotating the topics in the questions or answers.

In summary, this paper makes the following contributions:??? It proposes the new problem of question generation based on a given sentence and a topic (or concept) in the sentence.

To our knowledge, this topic-based question generation has not been studied before.

The model can also take a question type because for the same topic, different types of questions can be asked (see section 5.3).??? It proposes a novel neural network model to solve the problem.

A pre-decode mechanism is also explored to improve the model performance.??? The proposed model can be directly trained using a normal question-answering corpus without requiring additional labeling of topics in each input sentence.??? The proposed model is evaluated using the Amazon community question-answering corpus.

Experimental results show that our model is effective.

Our work is inspired by three lines of research: sequence to sequence models, question pattern prediction and question topic selection, multi-source sequence-to-sequence learning.

Sequence to sequence models have been successfully applied to many Natural Language Processing tasks especially Neural Machine Translation BID0 BID19 and Sequence Tagging BID12 , etc., tackled question generation using a conditional neural language model with a global attention mechanism and it outperformed a state-of-the-art rule-based system.

Their model is a typical sequence to sequence structure using a bidirectional LSTM as the encoder to encode a sentence and a LSTM as the decoder to generate the target question.

This structure does not rely on hand-crafted rules or a sophisticated NLP pipeline and can generate more fluent and grammatically correct questions.

However, this approach is not targeted and often generate poorly focused questions.

Question pattern prediction and question topic selection are two important components of the question generation engine proposed by BID7 .

In their approach, a question is formed by filling an automatically selected phrase Q t into the pattern predicted from pre-mined patterns, which are not done using deep learning.

This work illustrates one important hypothesis that it is reasonable to divide one question into two independent parts, i.e., question pattern and question topic.

Inspired by this, to control the question type and content in the generation process, we introduce two extra information into the sequence to sequence learning framework named respectively Topic (T) and Question Type (QT).Multi-source sequence-to-sequence learning aims to integrate information from multiple sources to boost learning.

BID22 built a multi-source machine translation model and reported up to +4.8 Bleu increases on top of a very strong attention-based neural translation model.

proposed a neural system combination framework leveraging multi-source NMT, which takes as input the outputs of NMT and SMT systems and produces the final translation.

The multi-source framework achieved significant improvement over the best single system output.

Inspired by this, we design independent encoders for question type, topic and answer, then a similar mechanism is used to integrate these pieces of information into decoder.

We now define and formulate the task of topic-based question generation.

Given an input sentence a, a topic tc and a question type qt (we use TQT to denote topic and question type together, use z to denote (qt, tc)), the goal is to generate a word sequence q, a question of arbitrary length, related to the TQT and the sentence.

qt is the vector represention of the question type, tc is the given topic and is represented as a sequence of tokens [tc 1 , ..., tc m ].

The topic-based question generation task is defined as finding the best questionq that maximizes the conditional likelihood given a and z: DISPLAYFORM0 where P (q|a) is modeled with a global attention mechanism (Section 4.3).

This section describes the proposed model in detail, which can be trained directly using a questionanswering corpus without requiring any additional annotations.

To generate topic-specific questions, we need the topic in a given answer (or descriptive text) and its question for training.

However, training data annotated with topics in the questions and answers are difficult to obtain as it requires a great deal of manual labeling.

Here, we propose a simple and effective method to identify topics shared in given question and answer pairs.

This method is based on the observation that the content (or tokens with similar meanings) shared by a given question and answer can be regarded as a topic.

Imagine a multi-round dialogue about tourism shown in the illustration 1, word 'America' is mentioned many times, and it's obvious that this is the topic of the conversation.

What did you go to America for?I went to the United States Based on this observation, we extract the topic from the given answer A according to its question Q. A and Q are represented as a sequence of tokens [a 1 , ..., a n ] and [q 1 , ..., q m ] respectively.

In the process of topic extraction, we consider all the tokens in A as candidates for the topic, and consider all the tokens in Q as voters voting for the candidates similar to q i .

Formally, after voting, candidate a i obtains votes DISPLAYFORM0 , where p ij can be calculated by: DISPLAYFORM1 where sim(a i , q j ) is the cosine similarity between a i and q j calculated using word embeddings of the two words.

Candidate a i is selected as part of the topic T if the average DISPLAYFORM2 Question type is another piece of information used in generating a topic-specific question.

For example, if one wants a specific type of answer, he/she needs to ask a specific type of question.

According to commonly used questioning methods, we divide question types into 7 categories, namely 'yes/no ','what','who','how','where','what', and 'others' (very infrequent) .

Many other types can be grouped into the existing types, e.g., 'whose' belongs to the category 'who'.

Since the question type can be fairly easily determined in a given question, we simply employ some keywords to categorize them, which achieves good results.

Section 4.1 gave a method to extract topics from a raw question-answering training corpus.

Given a model learned from the corpus and a topic from those existing topics in the corpus, it is easy to see that we can generate reasonable questions.

However, the problem is whether it is possible to generate a reasonable question given any topics, including those that have not appeared in the training corpus.

Under certain conditions, the answer is yes.

We explain this here.

Formally, the problem described above is whether our model can generate meaningful question q ij when given an answer a i and an arbitrary but related tqt (topic and question type) z j (to some of those tqts in the training corpus).

Here we first describe our question generation model and then use the model to answer the question.

In fact, the question generation task defined in section 3 can be written as a product of conditional probabilities in various ways.

This paper formulates the model in the following way: DISPLAYFORM0 where a i and z j are the internal vector representations for the given answer sentence and topic respectively.

We then use these two internal representations to generate the final question.

BID19 showed that the sentence with the similar meaning will be clustered together by clustering their LSTM hidden states (corresponding to our internal representations).

This indicates that similar sentences have similar internal representations, or it can be assumed that they are sampled from the same distribution.

As illustration 2 shows, our model uses two internal representations (can be assumed as sampled from ??(A l ) and ??(Z s ) respectively) to generate a question (can be assumed as sampled from ??(Q sl )) which learns a distribution mapping method DISPLAYFORM1 .

When given an answer a i and a tqt z i which never appeared as a pair in the training data, benefited from the distribution mapping, our model can generate a question similar to q j because a i and a j are from the same distribution ??(A l ) and, z i and z j are from the same distribution ??(Z s ).

However, this method requires enough training data to learn each distribution and the mapping relationship between them.

To generate topic-specific questions, our model takes two pieces of information, namely a topic and question type pair and an answer.

FIG3 illustrates the overall framework of our model, which has three major components: TQT Encoder, Answer Encoder and Decoder.

This framework contains a pre-decode mechanism.

We get our basic model if we use f t marked in the figure to predict words.

Below, we describe our model in detail.

Notation Description: To simplify the framework in the figure, we omit some connecting lines and attentions, but we use words or symbols to describe those omitted connections and attentions, e.g., word 'attention2' represents an attention similar to 'attention1'.Full Connection: All the 'full connection' components in FIG3 have the same structure (but different parameters) which can be formulated as: DISPLAYFORM0 where I is the input of the 'full connection' component, and W s and W t are parameters to be learned.

TQT Encoder: We also call it topic encoder.

It encodes the topic information into an internal vector representation.

For question types in tqt (topic and question type), we divide them into 7 categories as we discussed previously, and each category is represented by a type embedding e t i which can be learned in the training process.

The question type encoder uses a full connection function to transduce e t i to an internal vector representation qt i = F C 1 (e t i ).

We use a bidirectional LSTM to encode the topic, DISPLAYFORM1 where ??? ??? b t is the hidden state at time step t for the forward pass LSTM and ??? ??? b t for the backward pass.

We use their concatenation, i.e., [ ??? ??? b t ; ??? ??? b t ], as the hidden state b t at time step t. As for time step 0, we use F C 2 (e t i ) to initialize the LSTM hidden state since we have discussed in section 2 that question type as the first part in a sentence is appropriate for starting a sentence.

Answer Encoder: To encoder the answer information, we use another bidrectional LSTM.

The formula is similar to equation 5.

Denoting the hidden state of this bidrectional LSTM at time step t as ???

??? ae t and ??? ??? ae t for the forward and backward directions respectively, we use the concatenation of them as the hidden state ae t (= [ ??? ??? ae t ; ??? ??? ae t ]) at time t.

We use both the TQT encoder's and answer encoder's outputs for initialization of the decoder hidden state.

Firstly, we concatenate each encoder's last hidden state of the forward and backward pass, named respectively s c = [ DISPLAYFORM2 where aw is a sequence of answer tokens.

Then, we take s c and s a as the input of a full connection function to obtain the initialization of the decoder hidden state: DISPLAYFORM3 Decoder: Since we use a LSTM to decode the internal representation and generate questions, in our model, the conditional likelihood P (q t |a, z, q <t ) in equation 1 can be calculated by: DISPLAYFORM4 where h t is the hidden state of the decoder LSTM; qt t is the question type encoder's output; tc t and ac t have the same calculation method.

Here we only give the calculation formula for tc t : DISPLAYFORM5 where b i is the hidden state at time step i for the bidirectional LSTM in topic encoder, and its weight ?? i is calculated by: DISPLAYFORM6 where e i = h t W a b i scores how well h t and b i match.

Pre-decode Mechanism: The topic extraction approach introduced in section 4.1 is simple and effective.

However, the topic extracted by this approach still mixed with some noise.

To improve the model performance and filter out some noise, we explore a pre-decode mechanism.

The main idea of this mechanism is to use another LSTM to obtain a pre-decode result f t at time step t as a filter to influence the final prediction.

In the pre-decode method, the conditional likelihood P (q t |a, z, q <t ) formula in equation 6 needs to be rewritten as: DISPLAYFORM7 where h t is the hidden state of the pre-decoder LSTM; h t is the hidden state of the final decoder LSTM; c t = F C 4 ([qt t ; tc t ; ac t ]).

The weighted sum tc t in equation 7 needs to add the weight (or filter) ?? i calculated by the pre-decode result, denoted as tc t : DISPLAYFORM8 where DISPLAYFORM9 where e i = f t W a b i ; ?? i has the same calculation method as that for equation 8 except e i = h t W a b i .

The hidden state of the final decoder LSTM h t is obtained by combining the previous hidden state h t???1 , the word representations em t at the current time step, and the pre-decode result f t : DISPLAYFORM10

We now evaluate the proposed model for our topic-based question generation task.

As there is no existing method that can perform this new task, we compare with conventional question generation (which our model can do as well) without any given topic or question type.

We use the Amazon question/answer corpus 1 (AQAD) BID20 for training and testing.

The dataset contains questions and answers about products and services from Amazon.

It has around 1.4 million answered questions.

For pre-processing, we remove those question-answer pairs (using simple patterns) with answers that are impossible to be used to generate meaningful questions, like "yes/no", "yes, you can" and "no, you cannot," etc.

Since we focus on generation of a single question from one answer, we also remove those question-answer pairs with multiple questions or the question length with more than 50 words.

We are finally left with about 900k question-answer pairs for our experiments.

We use the approach proposed in section 4.1 to extract topics and decide the question type.

Pretrained word embedding 2 BID16 ) is employed to calculate the similarity in equation 2.

We set the topic selection threshold K to 0.4, and the word similarity threshold ?? to 0.3 (section 4.1).

For each question/answer pair, we select no more than 5 words as the topic.

We randomly divide the corpus into a training set, a development set (with about one thousand q/a pairs), and a test set (with about one thousand q/a pairs).

We report the results on the test set.

The hyper-parameters in our system are described as follows.

We share the word embedding between the answer and the question.

We limit the shared vocabulary to 30k in our experiments.

The number of hidden units is 600 and the word embedding dimension is 300.

We set the number of layers of LSTM to 1 in both encoder and decoder.

The network parameters are updated with the Adam algorithm (Kingma & Ba, 2014) with learning rate 0.0001.For the experiment of our system in the conventional question generation setting, since there is no topic given at the test time, we employ a sentence classification model BID10 to predict the question type from the given answer and a sequence labeling model BID12 to extract topics from the given answer.

Both models are trained using automatically extracted labels.

Automatic Evaluation: Following Du et al. FORMULA0 , we use the evaluation package released by BID3 .

The package includes BLEU 1 to 4, METEOR and ROUGE L .

We use the latest model from as our baseline, which is a conventional question generation system using a sequence-to-sequence model.

It significantly outperforms stateof-the-art rule-based systems.

Table 1 : Experimental results for different models based on automated evaluation.

The symbol string 'a-qy g-tc' in the table means using automatically generated question type and given topic to test.

Specifically, 'a' is for automatic, 'g' is for given (question type or topic), 'qt' is for question type, and 'tc' is for topic.

Ours* is our model with the pre-decode mechanism.

Table 1 shows the results.

The first three rows are for conventional question generation, and the last three rows are for different variations of the proposal topic-based question generation.

From the table, we can make the following observations:(1) Based the results from the first three rows, we can see that although our model is not designed for conventional question generation, with the pre-decode mechanism it outperforms the state-of-the-art baseline in four out of 6 test metrics.

Our model (row 3) did not achieve even better results due to the difficulty of automated sentence classification and topic labeling.

TAB3 shows their results.(2) Based on all the results, we can see that our models (our*) with the pre-decode mechanism significantly outperforms our basic model (row 2) which shows the validity of pre-decode mechanism.(3) When the topic is given, regardless whether the question type is given or not, our models (last three rows) outperform the baseline (row 1).

When the question type is also given, the results are markedly better.

Although these results are expected as our models are given more information, in practical applications, one seldom wants a random question.

A question should be generated on a topic that is related to the current conversation and with a desired type of answer.

Human Evaluation: We also perform human evaluation to measure the quality of questions generated by our system and the baseline.

We consider naturalness modality, which indicates the grammaticality, fluency and rationality.

We randomly sampled 100 sentence-question pairs and completely shuffled the order of tested systems.

We ask three Ph.

D.students to rate the pairs in terms of the modality on the 1 to 5 scale (5 for the best and the total score is scaled to the full score of 100).

Ours* a-qt a-tc a-qt g-tc g-qt a-tc g-qt g-tc Score 66.47 65.07 67.4 66.27 68 Table 3 : Experimental results for different models based on human evaluation.

The symbol string a-qt g-tc in the table has the same meaning as in Table 1 .

The perfect score is 100.

Table 3 shows that our model gets the highest score on the condition of 'a-qt g-tc' and 'g-qt gtc', which shows that our model generates more natural questions (i.e. grammaticality, fluency and rationality).

Our model (columns 2 and 4) did not achieve even better results due to the wrong topic and we'll explain this with examples in the Case Study section.

Case Study: In TAB5 , we present some sample questions generated by our model and the baseline.

From the first sample, we can see that the question generated by our model is more consistent with the given sentence.

By comparing our model output of different topics, we can find that our model can still generate a high quality question when given an inaccurate topic.

This means that our model has a certain anti-interference ability.

In the second sample, the baseline model generates a question that is irrelevant to the given sentence.

However, our model can generate question relevant to the given topic and sentence.

In this sample, our model generates an irrelevant question due to the wrong given topic 'tracks' and generate a natural and relevant question when given a reasonable topic.

It is for this reason that our model (when taking 'a-qt a-tc' as input) did not achieve even better results in human and automatic evaluations.

Table 5 : Questions generated from our model by given arbitrary question type or topic.

Table 5 shows several questions generated from a given sentence according to different question types and topics.

The first three rows are for given the same topic but different question types.

The results show that our model can change the questioning method according to the given question type.

Note that some question types, i.e. 'yes/no' and 'what' for the given sentence have not appeared in the training corpus.

The last three rows are for given arbitrary topics.

In this situation, questions generated by our model are still rational and consistent with the given topic, question type and the sentence.

In this paper, we proposed the new task of topic-based question generation, which has not been investigated before.

Based on our experiences, we believe this is a more useful question generation setting than the conventional setting without a given topic because a question without the knowledge of its topic is of limited use in actual conversations.

We then proposed a novel network architecture to perform the task, and discussed its generalization ability.

Experimental results showed that the proposed model performed slightly better than the state-of-the-art baseline in the conventional setting (although our model not designed for this setting), and performed markedly better in the proposed new setting.

@highlight

We propose a neural network that is able to generate topic-specific questions.

@highlight

Presents a neural network-based approach to generate topic-specific questions with the motivation that topical questions are more meaningful in practical applications.

@highlight

Proposes a topic-based generation method using an LSTM to extract topics using a two-stage encoding technique