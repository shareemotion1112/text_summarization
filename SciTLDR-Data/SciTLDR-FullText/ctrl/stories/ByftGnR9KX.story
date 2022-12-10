Conversational machine comprehension requires a deep understanding of the conversation history.

To enable traditional, single-turn models to encode the history comprehensively, we introduce Flow, a mechanism that can incorporate intermediate representations generated during the process of answering previous questions, through an alternating parallel processing structure.

Compared to shallow approaches that concatenate previous questions/answers as input, Flow integrates the latent semantics of the conversation history more deeply.

Our model, FlowQA, shows superior performance on two recently proposed conversational challenges (+7.2% F1 on CoQA and +4.0% on QuAC).

The effectiveness of Flow also shows in other tasks.

By reducing sequential instruction understanding to conversational machine comprehension, FlowQA outperforms the best models on all three domains in SCONE, with +1.8% to +4.4% improvement in accuracy.

Q3

The young girl and her dog set out a trip into the woods one day.

Upon entering the woods the girl and her dog found that the woods were dark and cold.

The girl was a little scared and was thinking of turning back, but yet they went on.

…

Figure 1: An illustration of conversational machine comprehension with an example from the Conversational Question Answering Challenge dataset (CoQA).Humans seek information in a conversational manner, by asking follow-up questions for additional information based on what they have already learned.

Recently proposed conversational machine comprehension (MC) datasets BID20 BID3 aim to enable models to assist in such information seeking dialogs.

They consist of a sequence of question/answer pairs where questions can only be understood along with the conversation history.

Figure 1 illustrates this new challenge.

Existing approaches take a single-turn MC model and augment the current question and context with the previous questions and answers BID3 BID20 .

However, this offers only a partial solution, ignoring previous reasoning 1 processes performed by the model.

We present FLOWQA, a model designed for conversational machine comprehension.

FLOWQA consists of two main components: a base neural model for single-turn MC and a FLOW mechanism that encodes the conversation history.

Instead of using the shallow history, i.e., previous questions and answers, we feed the model with the entire hidden representations generated during the process of answering previous questions.

These hidden representations potentially capture related information, such as phrases and facts in the context, for answering the previous questions, and hence provide additional clues on what the current conversation is revolving around.

This FLOW mechanism is also remarkably effective at tracking the world states for sequential instruction understanding BID15 : after mapping world states as context and instructions as questions, FLOWQA can interpret a sequence of inter-connected instructions and generate corresponding world state changes as answers.

The FLOW mechanism can be viewed as stacking single-turn QA models along the dialog progression (i.e., the question turns) and building information flow along the dialog.

This information transfer happens for each context word, allowing rich information in the reasoning process to flow.

The design is analogous to recurrent neural networks, where each single update unit is now an entire question answering process.

Because there are two recurrent structures in our modeling, one in the context for each question and the other in the conversation progression, a naive implementation leads to a highly unparallelizable structure.

To handle this issue, we propose an alternating parallel processing structure, which alternates between sequentially processing one dimension in parallel of the other dimension, and thus speeds up training significantly.

FLOWQA achieves strong empirical results on conversational machine comprehension tasks, and improves the state of the art on various datasets (from 67.8% to 75.0% on CoQA and 60.1% to 64.1% on QuAC).

While designed for conversational machine comprehension, FLOWQA also shows superior performance on a seemingly different task -understanding a sequence of natural language instructions (framed previously as a sequential semantic parsing problem).

When tested on SCONE BID15 , FLOWQA outperforms all existing systems in three domains, resulting in a range of accuracy improvement from +1.8% to +4.4%.

Our code can be found in https://github.com/momohuang/FlowQA.

In this section, we introduce the task formulations of machine comprehension in both single-turn and conversational settings, and discuss the main ideas of state-of-the-art MC models.

Given an evidence document (context) and a question, the task is to find the answer to the question based on the context.

The context C = {c 1 , c 2 , . . .

c m } is described as a sequence of m words and the question Q = {q 1 , q 2 . . .

q n } a sequence of n words.

In the extractive setting, the answer A must be a span in the context.

Conversational machine comprehension is a generalization of the singleturn setting: the agent needs to answer multiple, potentially inter-dependent questions in sequence.

The meaning of the current question may depend on the conversation history (e.g., in Fig. 1 , the third question such as '

Where?' cannot be answered in isolation).

Thus, previous conversational history (i.e., question/answer pairs) is provided as an input in addition to the context and the current question.

For single-turn MC, many top-performing models share a similar architecture, consisting of four major components: (1) question encoding, (2) context encoding, (3) reasoning, and finally (4) answer prediction.

Initially the word embeddings (e.g., BID18 BID19 of question tokens Q and context tokens C are taken as input and fed into contextual integration layers, such as LSTMs BID9 or self attentions BID32 , to encode the question and context.

Multiple integration layers provide contextualized representations of context, and are often inter-weaved with attention, which inject question information.

The context integration layers thus produce a series of query-aware hidden vectors for each word in the context.

Together, the context integration layers can be viewed as conducting implicit reasoning to find the answer span.

The final sequence of context vectors is fed into the answer prediction layer to select Figure 2: An illustration of the conversation flow and its importance.

As the current topic changes over time, the answer to the same question changes accordingly.the start and end position of answer span.

To adapt to the conversational setting, existing methods incorporate previous question/answer pairs into the current question and context encoding without modifying higher-level (reasoning and answer prediction) layers of the model.

Our model aims to incorporate the conversation history more comprehensively via a conceptually simple FLOW mechanism.

We first introduce the concept of FLOW (Section 3.1), propose the INTEGRATION-FLOW layers (Section 3.2), and present an end-to-end architecture for conversational machine comprehension, FLOWQA (Section 3.3).

Successful conversational MC models should grasp how the conversation flows.

This includes knowing the main topic currently being discussed, as well as the relevant events and facts.

Figure 2 shows a simplified CoQA example where such conversation flow is crucial.

As the conversation progresses, the topic being discussed changes over time.

Since the conversation is about the context C, we consider FLOW to be a sequence of latent representations based on the context tokens (the middle part of Fig 2) .

Depending on the current topic, the answer to the same question may differ significantly.

For example, when the dialog is about the author's father's funeral, the answer to the question What did he feel?

would be lonely, but when the conversation topic changes to five years after the death of the author's father, the answer becomes we saved each other.

A naive implementation of FLOW would pass the output hidden vectors from each integration layer during the (i − 1)-th question turn to the corresponding integration layer for Q i .

This is highly unparalleled, as the contexts have to be read in order, and the question turns have to be processed sequentially.

To achieve better parallelism, we alternate between them: context integration, processing sequentially in context, in parallel of question turns; and flow, processing sequentially in question turns, in parallel of context words (see Fig. 3 ).

This architecture significantly improves efficiency during training.

Below we describe the implementation of an INTEGRATION-FLOW (IF) layer, which is composed of a context integration layer and a FLOW component.

We pass the current context representation C h i for each question i into a BiLSTM layer.

All question i (1 ≤ i ≤ t) are processed in parallel during training.

DISPLAYFORM0 Q u e s t io n T u r n s C o n te x t W o rd s FLOW After the integration, we have t context sequences of length m, one for each question.

We reshape it to become m sequences of length t, one for each context word.

We then pass each sequence into a GRU 3 so the entire intermediate representation for answering the previous questions can be used when processing the current question.

We only consider the forward direction since we do not know the (i + 1)-th question when answering the i-th question.

All context word j (1 ≤ j ≤ m) are processed in parallel.

DISPLAYFORM1 We reshape the outputs from the FLOW layer to be sequential to context tokens, and concatenate them to the output of the integration layer.

DISPLAYFORM2 In summary, this process takes C h i and generates C h+1 i , which will be used for further contextualization to predict the start and end answer span tokens.

When FLOW is removed, the IF layer becomes a regular context integration layer and in this case, a single layer of BiLSTM.

We construct our conversation MC model, FLOWQA, based on the single-turn MC structure (Sec. 2.2) with fully-aware attention BID10 .

The full architecture is shown in Fig. 4 .

In this section, we describe its main components: initial encoding, reasoning and answer prediction.

We embed the context into a sequence of vectors, C = {c 1 , . . .

, c m } with pretrained GloVe BID18 , CoVE (McCann et al., 2017) and ELMo BID19 embeddings.

Similarly, each question at the i-th turn is embedded into a sequence of vectors Q i = {q i,1 , . . .

, q i,n }, where n is the maximum question length for all questions in the conversation.

Attention (on Question) Following DrQA BID1 , for each question, we compute attention in the word level to enhance context word embeddings with question.

The generated question-specific context input representation is denoted as C 0 i .

For completeness, a restatement of this representation can be found in Appendix C.1.Question Integration with QHierRNN Similar to many MC models, contextualized embeddings for the questions are obtained using multiple layers of BiLSTM (we used two layers).

DISPLAYFORM0 We build a pointer vector for each question to be used in the answer prediction layer by first taking a weighted sum of each word vectors in the question.

DISPLAYFORM1 where w is a trainable vector.

We then encode question history hierarchically with LSTMs to generate history-aware question vectors (QHierRNN).

DISPLAYFORM2 The final vectors, p 1 , . . .

, p t , will be used in the answer prediction layer.

The reasoning component has several IF layers on top of the context encoding, inter-weaved with attention (first on question, then on context itself).

We use fully-aware attention BID10 , which concatenates all layers of hidden vectors and uses S(x, y) = ReLU(Ux) T D ReLU(Uy) to compute the attention score between x, y, where U, D are trainable parameters and D is a diagonal matrix.

Below we give the details of each layer (from bottom to top).Integration-Flow ×2 First, we take the question-augmented context representation C 0 i and pass it to two IF layers.

DISPLAYFORM0 DISPLAYFORM1 Attention (on Question) After contextualizing the context representation, we perform fully-aware attention on the question for each context words.

DISPLAYFORM2 Integration-Flow We concatenate the output from the previous IF layer with the attended question vector, and pass it as an input.

DISPLAYFORM3 Attention (on Context) We apply fully-aware attention on the context itself (self-attention).

DISPLAYFORM4 Integration We concatenate the output from the the previous IF layer with the attention vector, and feed it to the last BiLSTM layer.

DISPLAYFORM5

We use the same answer span selection method BID29 BID10 to estimate the start and end probabilities P S i,j , P E i,j of the j-th context token for the i-th question.

Since there are unanswerable questions, we also calculate the no answer probabilities P ∅

i for the i-th question.

For completeness, the equations for answer span selection is in Appendix C.1.

BID3 .CoQA QuAC Prev.

SotA BID31

In this section, we evaluate FLOWQA on recently released conversational MC datasets.

We experiment with the QuAC BID3 and CoQA BID20 datasets.

While both datasets follow the conversational setting (Section 2.1), QuAC asked crowdworkers to highlight answer spans from the context and CoQA asked for free text as an answer to encourage natural dialog.

While this may call for a generation approach, BID31 shows that the an extractive approach which can handle Yes/No answers has a high upper-bound -97.8% F 1 .

Following this observation, we apply the extractive approach to CoQA.

We handle the Yes/No questions by computing P Y i , P N i , the probability for answering yes and no, using the same equation as P ∅ i (Eq. 17), and find a span in the context for other questions.

The main evaluation metric is F 1 , the harmonic mean of precision and recall at the word level.5 In CoQA, we report the performance for each context domain (children's story, literature from Project Gutenberg, middle and high school English exams, news articles from CNN, Wikipedia, AI2 Science Questions, Reddit articles) and the overall performance.

For QuAC, we use its original evaluation metrics: F 1 and Human Equivalence Score (HEQ).

HEQ-Q is the accuracy of each question, where the answer is considered correct when the model's F 1 score is higher than the average human F 1 score.

Similarly, HEQ-D is the accuracy of each dialog -it is considered correct if all the questions in the dialog satisfy HEQ.Comparison Systems We compare FLOWQA with baseline models previously tested on CoQA and QuAC.

BID20 presented PGNet (Seq2Seq with copy mechanism), DrQA BID1 and DrQA+PGNet (PGNet on predictions from DrQA) to address abstractive answers.

To incorporate dialog history, CoQA baselines append the most recent previous question and answer to the current question.6 BID3 applied BiDAF++, a strong extractive QA model to QuAC dataset.

They append a feature vector encoding the turn number to the question embedding and a feature vector encoding previous N answer locations to the context embeddings (denoted as N -ctx).

Empirically, this performs better than just concatenating previous question/answer pairs.

Yatskar FLOWQA (N -Ans) is our model: similar to BiDAF++ (N -ctx), we append the binary feature vector encoding previous N answer spans to the context embeddings.

Here we briefly describe the ablated systems: "-FLOW" removes the flow component from IF layer (Eq. 2 in Section 3.2), "-QHIER-RNN" removes the hierarchical LSTM layers on final question vectors (Eq. 7 in Section 3.3).Results TAB2 report model performance on CoQA and QuAC, respectively.

FLOWQA yields substantial improvement over existing models on both datasets (+7.2% F 1 on CoQA, +4.0% F 1 on QuAC).

The larger gain on CoQA, which contains longer dialog chains, 7 suggests that our FLOW architecture can capture long-range conversation history more effectively.

TAB4 shows the contributions of three components: (1) QHierRNN, the hierarchical LSTM layers for encoding past questions, (2) FLOW, augmenting the intermediate representation from the machine reasoning process in the conversation history, and (3) N -Ans, marking the gold answers to the previous N questions in the context.

We find that FLOW is a critical component.

Removing QHier-RNN has a minor impact (0.1% on both datasets), while removing FLOW results in a substantial performance drop, with or without using QHierRNN (2-3% on QuAC, 4.1% on CoQA).

Without both components, our model performs comparably to the BiDAF++ model (1.0% gain).8 Our model exploits the entire conversation history while prior models could leverage up to three previous turns.

By comparing 0-Ans and 1-Ans on two datasets, we can see that providing gold answers is more crucial for QuAC.

We hypothesize that QuAC contains more open-ended questions with multiple valid answer spans because the questioner cannot see the text.

The semantics of follow-up questions may change based on the answer span selected by the teacher among many valid answer spans.

Knowing the selected answer span is thus important.

We also measure the speedup of our proposed alternating parallel processing structure (Fig. 3) over the naive implementation of FLOW, where each question is processed in sequence.

Based on the training time each epoch takes (i.e., time needed for passing through the data once), the speedup is 8.1x on CoQA and 4.2x on QuAC.

The higher speedup on CoQA is due to the fact that CoQA has longer dialog sequences, compared to those in QuAC.

In this section, we consider the task of understanding a sequence of natural language instructions.

We reduce this problem to a conversational MC task and apply FLOWQA.

FIG4 gives a simplified example of this task and our reduction.

Task Given a sequence of instructions, where the meaning of each instruction may depend on the entire history and world state, the task is to understand the instructions and modify the world accordingly.

More formally, given the initial world state W 0 and a sequence of natural language instructions {I 1 , . . .

, I K }, the model has to perform the correct sequence of actions on W 0 , to obtain {W 1 , . . .

, W K }, the correct world states after each instruction.

Reduction We reduce sequential instruction understanding to machine comprehension as follows.• Context C i : We encode the current world state W i−1 as a sequence of tokens.• Question Q i : We simply treat each natural language instruction I i as a question.• Answer A i : We encode the world state change from W i−1 to W i as a sequence of tokens.

At each time step i, the current context C i and question Q i are given to the system, which outputs the answer A i .

Given A i , the next world state C i+1 is automatically mapped from the reduction rules.

We encode the history of instructions explicitly by concatenating preceding questions and the current one and by marking previous answers in the current context similar to N -Ans in conversational MC tasks.

Further, we simplify FLOWQA to prevent overfitting.

Appendix C.2 contains the details on model simplification and reduction rules, i.e., mapping from the world state and state change to a sequence of token.

During training, gold answers (i.e., phrases mapped from world state change after each previous instruction) are provided to the model, while at test time, predicted answers are used.

We evaluate our model on the sequential instruction understanding dataset, SCONE BID15 , which contains three domains (SCENE, TANGRAMS, ALCHEMY).

Each domain has a different environment setting (see Appendix C.2).

We compare our approaches with prior works BID15 BID8 , which are semantic parsers that map each instruction into a logical form, and then execute the logical form to update the world state, and BID5 , which maps each instruction into actions similar to our case.

The model performance is evaluated by the correctness of the final world state after five instructions.

Our learning set-up is similar to that of BID5 , where the supervision is the change in world states (i.e., analogous to logical form), while that of BID15 and used world states as a supervision.

The development and test set results are reported in TAB5 .

Even without FLOW, our model (FLOWQA-FLOW) achieves comparable results in two domains (Tangrams and Alchemy) since we still encode the history explicitly.

When augmented with FLOW, our FLOWQA model gains decent improvements and outperforms the state-of-the-art models for all three domains.

Sequential question answering has been studied in the knowledge base setting BID11 BID23 BID28 , often framed as a semantic parsing problem.

Recent datasets BID3 BID20 BID4 BID22 enabled studying it in the textual setting, where the information source used to answer questions is a given article.

Existing approaches attempted on these datasets are often extensions of strong single-turn models, such as BiDAF BID24 and DrQA BID1 , with some manipulation of the input.

In contrast, we propose a new architecture suitable for multi-turn MC tasks by passing the hidden model representations of preceding questions using the FLOW design.

Dialog response generation requires reasoning about the conversation history as in conversational MC.

This has been studied in social chit-chats (e.g., BID21 BID14 BID7 and goal-oriented dialogs (e.g., BID2 BID13 .

Prior work also modeled hierarchical representation of the conversation history BID17 .

While these tasks target reasoning with the knowledge base or exclusively on the conversation history, the main challenge in conversational MC lies in reasoning about context based on the conversation history, which is the main focus in our work.

We presented a novel FLOW component for conversational machine comprehension.

By applying FLOW to a state-of-the-art machine comprehension model, our model encodes the conversation history more comprehensively, and thus yields better performance.

When evaluated on two recently proposed conversational challenge datasets and three domains of a sequential instruction understanding task (through reduction), FLOWQA outperforms existing models.

While our approach provides a substantial performance gain, there is still room for improvement.

In the future, we would like to investigate more efficient and fine-grained ways to model the conversation flow, as well as methods that enable machines to engage more active and natural conversational behaviors, such as asking clarification questions.

Recall that the FLOW operation takes in the hidden representation generated for answering the current question, fuses into its memory, and passes it to the next question.

Because the answer finding (or reasoning) process operates on top of a context/passage, this FLOW operation is a big memory operation on an m×d matrix, where m is the length of the context and d is the hidden size.

We visualize this by computing the cosine similarity of the FLOW memory vector on the same context words for consecutive questions, and then highlight the words that have small cosine similarity scores, i.e., the memory that changes more significantly.

The highlighted part of the context indicates the QA model's guess on the current conversation topic and relevant information.

Notice that this is not attention; it is instead a visualization on how the hidden memory is changing over time.

The example is from CoQA BID20 .Q1: Where did Sally go in the summer?

→ Q2:

Did she make any friends there?Sally had a very exciting summer vacation .

She went to summer camp for the first time .

She made friends with a girl named Tina .

They shared a bunk bed in their cabin .

Sally 's favorite activity was walking in the woods because she enjoyed nature .

Tina liked arts and crafts .

Together , they made some art using leaves they found in the woods .

Even after she fell in the water , Sally still enjoyed canoeing .

She was sad when the camp was over , but promised to keep in touch with her new friend .

Sally went to the beach with her family in the summer as well .

She loves the beach .

Sally collected shells and mailed some to her friend , Tina , so she could make some arts and crafts with them .

Sally liked fishing with her brothers , cooking on the grill with her dad , and swimming in the ocean with her mother .

The summer was fun , but Sally was very excited to go back to school .

She missed her friends and teachers .

She was excited to tell them about her summer vacation .

Sally had a very exciting summer vacation .

She went to summer camp for the first time .

She made friends with a girl named Tina .

They shared a bunk bed in their cabin .

Sally 's favorite activity was walking in the woods because she enjoyed nature .

Tina liked arts and crafts .

Together , they made some art using leaves they found in the woods .

Even after she fell in the water , Sally still enjoyed canoeing .

She was sad when the camp was over , but promised to keep in touch with her new friend .

Sally went to the beach with her family in the summer as well .

She loves the beach .

Sally collected shells and mailed some to her friend , Tina , so she could make some arts and crafts with them .

Sally liked fishing with her brothers , cooking on the grill with her dad , and swimming in the ocean with her mother .

The summer was fun , but Sally was very excited to go back to school .

She missed her friends and teachers .

She was excited to tell them about her summer vacation .

Sally had a very exciting summer vacation .

She went to summer camp for the first time .

She made friends with a girl named Tina .

They shared a bunk bed in their cabin .

Sally 's favorite activity was walking in the woods because she enjoyed nature .

Tina liked arts and crafts .

Together , they made some art using leaves they found in the woods .

Even after she fell in the water , Sally still enjoyed canoeing .

She was sad when the camp was over , but promised to keep in touch with her new friend .

Sally went to the beach with her family in the summer as well .

She loves the beach .

Sally collected shells and mailed some to her friend , Tina , so she could make some arts and crafts with them .

Sally liked fishing with her brothers , cooking on the grill with her dad , and swimming in the ocean with her mother .

The summer was fun , but Sally was very excited to go back to school .

She missed her friends and teachers .

She was excited to tell them about her summer vacation .Q4: What was Tina's favorite activity?

→ Q5:

What was Sally's?Sally had a very exciting summer vacation .

She went to summer camp for the first time .

She made friends with a girl named Tina .

They shared a bunk bed in their cabin .

Sally 's favorite activity was walking in the woods because she enjoyed nature .

Tina liked arts and crafts .

Together , they made some art using leaves they found in the woods .

Even after she fell in the water , Sally still enjoyed canoeing .

She was sad when the camp was over , but promised to keep in touch with her new friend .

Sally went to the beach with her family in the summer as well .

She loves the beach .

Sally collected shells and mailed some to her friend , Tina , so she could make some arts and crafts with them .

Sally liked fishing with her brothers , cooking on the grill with her dad , and swimming in the ocean with her mother .

The summer was fun , but Sally was very excited to go back to school .

She missed her friends and teachers .

She was excited to tell them about her summer vacation .Q9: Had Sally been to camp before?

→ Q10:

How did she feel when it was time to leave?Sally had a very exciting summer vacation .

She went to summer camp for the first time .

She made friends with a girl named Tina .

They shared a bunk bed in their cabin .

Sally 's favorite activity was walking in the woods because she enjoyed nature .

Tina liked arts and crafts .

Together , they made some art using leaves they found in the woods .

Even after she fell in the water , Sally still enjoyed canoeing .

She was sad when the camp was over , but promised to keep in touch with her new friend .

Sally went to the beach with her family in the summer as well .

She loves the beach .

Sally collected shells and mailed some to her friend , Tina , so she could make some arts and crafts with them .

Sally liked fishing with her brothers , cooking on the grill with her dad , and swimming in the ocean with her mother .

The summer was fun , but Sally was very excited to go back to school .

She missed her friends and teachers .

She was excited to tell them about her summer vacation .Q16:

Does she like it?

→ Q17:

Did she do anything interesting there? (The conversation is now talking about Sally's trip to the beach with her family)Sally had a very exciting summer vacation .

She went to summer camp for the first time .

She made friends with a girl named Tina .

They shared a bunk bed in their cabin .

Sally 's favorite activity was walking in the woods because she enjoyed nature .

Tina liked arts and crafts .

Together , they made some art using leaves they found in the woods .

Even after she fell in the water , Sally still enjoyed canoeing .

She was sad when the camp was over , but promised to keep in touch with her new friend .

Sally went to the beach with her family in the summer as well .

She loves the beach .

Sally collected shells and mailed some to her friend , Tina , so she could make some arts and crafts with them .

Sally liked fishing with her brothers , cooking on the grill with her dad , and swimming in the ocean with her mother .

The summer was fun , but Sally was very excited to go back to school .

She missed her friends and teachers .

She was excited to tell them about her summer vacation .Q18: Did she fish and cook alone?

→ Q19:

Who did she fish and cook with?Sally had a very exciting summer vacation .

She went to summer camp for the first time .

She made friends with a girl named Tina .

They shared a bunk bed in their cabin .

Sally 's favorite activity was walking in the woods because she enjoyed nature .

Tina liked arts and crafts .

Together , they made some art using leaves they found in the woods .

Even after she fell in the water , Sally still enjoyed canoeing .

She was sad when the camp was over , but promised to keep in touch with her new friend .

Sally went to the beach with her family in the summer as well .

She loves the beach .

Sally collected shells and mailed some to her friend , Tina , so she could make some arts and crafts with them .

Sally liked fishing with her brothers , cooking on the grill with her dad , and swimming in the ocean with her mother .

The summer was fun , but Sally was very excited to go back to school .

She missed her friends and teachers .

She was excited to tell them about her summer vacation .

We found that in the first transition (i.e., from Q1 to Q2), many memory regions change significantly.

This is possibly due to the fact that the FLOW operation is taking in the entire context at the start.

Later on, the FLOW memory changes more dramatically at places where the current conversation is focusing on.

For example, from Q4 to Q5, several places that talk about Sally's favorite activity have higher memory change, such as was walking in the woods, she enjoyed nature, and enjoyed canoeing.

From Q16 to Q17, we can see that several memory regions on the interesting things Sally did during the trip to the beach are altered more significantly.

And from Q18 to Q19, we can see that all the activities Sally had done with her family are being activated, including went to the beach with her family, fishing with her brothers, cooking on the grill with her dad, and swimming in the ocean with her mother.

Together, we can clearly see that more active memory regions correspond to what the current conversation is about, as well as to the related facts revolving around the current topic.

As the topic shifts through the conversation, regions with higher memory activity move along.

BID20 .

Context:

When my father was dying, I traveled a thousand miles from home to be with him in his last days.

It was far more heartbreaking than I'd expected, one of the most difficult and painful times in my life.

After he passed away I stayed alone in his apartment.

There were so many things to deal with.

It all seemed endless.

I was lonely.

I hated the silence of the apartment.

But one evening the silence was broken: I heard crying outside.

I opened the door to find a little cat on the steps.

He was thin and poor.

He looked the way I felt.

I brought him inside and gave him a can of fish.

He ate it and then almost immediately fell sound asleep.

The next morning I checked with neighbors and learned that the cat had been abandoned by his owner who's moved out.

So the little cat was there all alone, just like I was.

As I walked back to the apartment, I tried to figure out what to do with him.

But as soon as I opened the apartment door he came running and jumped into my arms.

Analysis: In this example, there is also a jump in the dialog at the question Where is the pope flying to?

The dialog jumps from discussing the events itself to the ending of the event (where the pope is leaving for London).

BiDAF++ fails to grasp this topic shift.

Although "How Great Thou Art" is a song that Boyle will sing during the event, it is not the song Boyle will sing when pope is leaving for London.

On the other hand, FLOWQA is able to capture this topic shift because the intermediate representation for answering the previous question "

Where is the pope flying to?" will indicate that the dialog is revolving at around the ending of the event (i.e., the last sentence).

Question-specific context input representation: We restate how the question-specific context input representation C 0 i is generated, following DrQA BID1 .

DISPLAYFORM0 where g Q i,k is the GloVe embedding for the k-th question word in the i-th question, and g C j is the GloVe embedding for the j-th context word.

The final question-specific context input representation C 0 i contains: (1) word embeddings, (2) a binary indicator em i,j , whether the j-th context word occurs in the i-th question, and (3) output from the attention.

DISPLAYFORM1 Answer Span Selection Method: We restate how the answer span selection method is performed (following BID29 BID10 ) to estimate the start and end probabilities P S i,j , P E i,j of the j-th context token for the i-th question.

DISPLAYFORM2 To address unanswerable questions, we compute the probability of having no answer: DISPLAYFORM3 For each question Q i , we first use P ∅ i to predict whether it has no answer.

9 If it is answerable, we predict the span to be j s , j e with the maximum P S i,j s P E i,j e subject to the constraint 0 ≤ j e −j s ≤ 15.Hyper-parameter setting and additional details: We use spaCy for tokenization.

We additionally fine-tuned the GloVe embeddings of the top 1000 frequent question words.

All RNN output size is set to 125, and thus the BiRNN output would be of size 250.

The attention hidden size used in fully-aware attention is set to 250.

During training, we use a dropout rate of 0.4 BID25 after the embedding layer (GloVe, CoVe and ELMo) and before applying any linear transformation.

In particular, we share the dropout mask when the model parameter is shared, which is also known as variational dropout BID6 .

We batch the dialogs rather than individual questions.

The batch size is set to one dialog for CoQA (since there can be as much as 20+ questions in each dialog), and three dialog for QuAC (since the question number is smaller).

The optimizer is Adamax BID12 with a learning rate α = 0.002, β = (0.9, 0.999) and = 10 −8 .

A fixed random seed is used across all experiments.

All models are implemented in PyTorch (http://pytorch.org/).

We use a maximum of 20 epochs, with each epoch passing through the data once.

It roughly takes 10 to 20 epochs to converge.

We begin by elaborating the simplification for FLOWQA for the sequential instruction understanding task.

First, we use the 100-dim GloVe embedding instead of the 300-dim GloVe and we do not use any contextualized word embedding.

The GloVe embedding is fixed throughout training.

Secondly, the embeddings for tokens in the context C are trained from scratch since C consists of synthetic tokens.

Also, we remove word-level attention because the tokens in contexts and questions are very different (one is synthetic, while the other is natural language).

Additionally, we remove self-attention since we do not find it helpful in this reduced QA setting (possibly due to the very short context here).

We use the same hidden size for both integration LSTMs and FLOW GRUs.

However, we tune the hidden size for the three domains independently, h = 100, 75, 50 for SCENE, TANGRAMS and ALCHEMY, respectively.

We also batch by dialog and use a batch size of 8.

A dropout rate of 0.3 is used and is applied before every linear transformation.

Environment for the Three Domains In SCENE, each environment has ten positions with at most one person at each position.

The domain covers four actions (enter, leave, move, and trade-hats) and two properties (hat color, shirt color).

In TANGRAMS, the environment is a list containing at most five shapes.

This domain contains three actions (add, move, swap) and one property (shape).

Lastly, in ALCHEMY, each environment is seven numbered beakers and covers three actions (pour, drain, mix) dealing with two properties (color, amount).Reducing World State to Context Now, we give details on the encoding of context from the world state.

In SCENE, there are ten positions.

For each position, there could be a person with shirt and hat, a person with a shirt, or no person.

We encode each position as two integers, one for shirt and one for hat (so the context length is ten).

Both integers take the value that corresponds to being a color or being empty.

In TANGRAMS, originally there are five images, but some commands could reduce the number of images or bring back removed images.

Since the number of images present is no greater than five, we always have five positions available (so the context length is five).

Each position consists of an integer, representing the ID of the image, and a binary feature.

Every time an image is removed, we append it at the back.

The binary feature is used to indicate if the image is still present or not.

In ALCHEMY, there are always seven beakers.

So the context length is seven.

Each position consists of two numbers, the color of the liquid at the top unit and the number of units in the beaker.

An embedding layer is used to turn each integer into a 10-dim vector.

Reducing the Logical Form to Answer Next, we encode the change of world states (i.e., the answer) into four integers.

The first integer is the type of action that is performed.

The second and third integers represent the position of the context, which the action is acted upon.

Finally, the fourth integer represents the additional property for the action performed.

For example, in the ALCHEMY domain, (0, i, j, 2) means "pour 2 units of liquid from beaker i to beaker j", and (1, i, i, 3) means "throw out 3 units of liquid in beaker i".

The prediction of each field is viewed as a multi-class classification problem, determined by a linear layer.

<|TLDR|>

@highlight

We propose the Flow mechanism and an end-to-end architecture, FlowQA, that achieves SotA on two conversational QA datasets and a sequential instruction understanding task.