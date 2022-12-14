Conversational question answering (CQA) is a novel QA task that requires the understanding of dialogue context.

Different from traditional single-turn machine reading comprehension (MRC), CQA is a comprehensive task comprised of passage reading, coreference resolution, and contextual understanding.

In this paper, we propose an innovative contextualized attention-based deep neural network, SDNet, to fuse context into traditional MRC models.

Our model leverages both inter-attention and self-attention to comprehend the conversation and passage.

Furthermore, we demonstrate a novel method to integrate the BERT contextual model as a sub-module in our network.

Empirical results show the effectiveness of SDNet.

On the CoQA leaderboard, it outperforms the previous best model's F1 score by 1.6%.

Our ensemble model further improves the F1 score by 2.7%.

Machine reading comprehension (MRC) is a core NLP task in which a machine reads a passage and then answers related questions.

It requires a deep understanding of both the article and the question, as well as the ability to reason about the passage and make inferences.

These capabilities are essential in applications like search engines and conversational agents.

In recent years, there have been numerous studies in this field (Huang et al., 2017; Seo et al., 2016; Liu et al., 2017) , with various innovations in text encoding, attention mechanisms and answer verification.

However, traditional MRC tasks often take the form of single-turn question answering.

In other words, there is no connection between different questions and answers to the same passage.

This oversimplifies the conversational manner humans naturally take when probing a passage, where question turns are assumed to be remembered as context to subsequent queries.

Figure 1 demonstrates an example of conversational question answering in which one needs to correctly refer "she" in the last two rounds of questions to its antecedent in the first question, "Cotton." To accomplish this kind of task, the machine must comprehend both the current round's question and previous rounds of utterances in order to perform coreference resolution, pragmatic reasoning and semantic implication.

To facilitate research in conversation question answering (CQA), several public datasets have been published that evaluate a model's efficacy in this field, such as CoQA (Reddy et al., 2018) , QuAC and QBLink (Elgohary et al., 2018) .

In these datasets, to generate correct responses, models need to fully understand the given passage as well as the dialogue context.

Thus, traditional MRC models are not suitable to be directly applied to this scenario.

Therefore, a number of models have been proposed to tackle the conversational QA task.

DrQA+PGNet (Reddy et al., 2018) combines evidence finding and answer generation to produce answers.

BiDAF++ (Yatskar, 2018) achieves better results by employing answer marking and contextualized word embeddings on the MRC model BiDAF (Seo et al., 2016) .

FlowQA (Huang et al., 2018 ) leverages a recurrent neural network over previous rounds of questions and answers to absorb information from its history context.

Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton.

Cotton lived high up in a nice warm place above the barn where all of the farmer's horses slept.

But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters...

What color was Cotton?

A1: white Q2:

Where did she live?

A2: in a barn Q3: Did she live alone?

A3: no Figure 1 : Example passage and first three rounds of question and answers from CoQA dataset (Reddy et al., 2018) .

Pronouns requiring coreference resolution is marked in bold.

In this paper, we propose SDNet, a contextual attention-based deep neural network for the conversational question answering task.

Our network stems from machine reading comprehension models, but it has several unique characteristics to tackle context understanding.

First, we apply both interattention and self-attention on the passage and question to obtain a more effective understanding of the passage and dialogue history.

Second, we prepend previous rounds of questions and answers to the current question to incorporate contextual information.

Third, SDNet leverages the latest breakthrough in NLP: BERT contextual embeddings (Devlin et al., 2018) .

Different from the canonical way of employing BERT as a monolithic structure with a thin linear task-specific layer, we utilize BERT as a contextualized embedder and absorb its structure into our network.

To accomplish this, we align the traditional tokenizer with the Byte Pair Encoding (BPE) tokenizer in BERT.

Furthermore, instead of using only the last layer's output from BERT (Devlin et al., 2018) , we employ a weighted sum of BERT layer outputs to take advantage of all levels of semantic abstraction.

Finally, we lock the internal parameters of BERT during training, which saves considerable computational cost.

These techniques are also applicable to other NLP tasks.

We evaluate SDNet on the CoQA dataset, and it improves on the previous state-of-the-art F 1 score by 1.6% (from 75.0% to 76.6%).

The ensemble model further increases the F 1 score to 79.3%.

In this section, we propose our neural model, SDNet, for the conversational question answering task.

We first formulate the problem and then present an overview of the model before delving into the details of the model structure.

Given a passage/context C, and question-answer pairs from previous rounds of conversation Q 1 , A 1 , Q 2 , A 2 , ..., Q k???1 , A k???1 , the task is to generate response A k given the latest question Q k .

The response is dependent on both the passage and historic questions and answers.

To incorporate conversation history into response generation, we employ the idea from DrQA+PGNet (Reddy et al., 2018) to prepend the latest N rounds of QAs to the current question Q k .

The problem is then converted into a single-turn machine reading comprehension task, where the reformulated question is

Encoding layer encodes each token in passage and question into a fixed-length vector, which includes both word embeddings and contextualized embeddings.

For contextualized embedding, we utilize the pretrained language understanding model BERT (Devlin et al., 2018) .

Different from previous work, we fix the parameters in BERT model and use the linear combination of embeddings from different layers in BERT.

Integration layer uses multi-layer recurrent neural networks (RNN) to capture contextual information within passage and question.

To characterize the relationship between passage and question, we conduct word-level attention from question to passage both before and after the RNNs.

We employ the idea of history-of-word from FusionNet (Huang et al., 2017) to reduce the dimension of output hidden vectors.

Furthermore, we conduct self-attention to extract relationship between words at different positions of context and question.

Output layer computes the final answer span.

It uses attention to condense the question into a fixedlength vector, which is then used in a bilinear projection to obtain the probability that the answer should start and end at each position.

An illustration of our model SDNet is in Figure 2 .

We first use GloVe (Pennington et al., 2014) embedding for each word in the context and question.

Additionally, we compute a feature vector f w for each context word, following the approach in DrQA .

This feature vector contains a 12-dim POS embedding, an 8-dim NER embedding, a 3-dim exact matching vector em i indicating whether this word, its lower-case form or its stem appears in the question, and a 1-dim normalized term frequency.

BERT as Contextual Embedder.

We design a number of methods to leverage BERT (Devlin et al., 2018) as a contextualized embedder in our model.

First, because BERT uses Byte Pair Encoding (BPE) (Sennrich et al., 2015) as the tokenizer, the generated tokens are sub-words and may not align with traditional tokenizer results.

To incorporate BERT into our network, we first use a conventional tokenizer (e.g. spaCy) to get word sequences, and then apply the BPE tokenizer from BERT to partition each word w in the sequence into subwords w = (b 1 , ..., b s ).

This alignment makes it possible to concurrently use BERT embeddings and other word-level features.

The contextual embedding of w is defined to be the averaged BERT embedding of all sub-words b j , 1 ??? j ??? s.

Second, Devlin et al. (2018) proposes the method to append thin task-specific linear layers to BERT, which takes the result from the last transformer layer as input.

However, as BERT contains multiple layers, we employ a weighted sum of these layer outputs to take advantage of information from all levels of semantic abstraction.

This can help boost the performance compared with using only the last transformer's output.

Third, as BERT contains hundreds of millions of parameters, it takes a lot of time and space to compute and store their gradients during optimization.

To tackle this problem, we lock the internal weights of BERT during training, only updating the linear combination weights.

This can significantly increase the efficiency during training, which can be especially useful when computing resource is limited.

To summarize, suppose a word w is tokenized to s BPE tokens w = (b 1 , b 2 , ..., b s ), and BERT has L layers that generate L embedding outputs for each BPE token, h

The contextual embedding BERT w for word w is computed as:

where ?? 1 , ..., ?? L are trainable parameters.

Word-level Inter-Attention.

We conduct attention from question to context (passage) based on GloVe word embeddings.

Suppose the context word embeddings are {h

where D ??? R k??k is a diagonal matrix and U ??? R d??k , k is the attention hidden size.

To simplify notation, we denote the above attention function as Attn(A, B, C), which linearly combines the vector set C using attention scores computed from vector sets A and B. This resembles the definition of attention in transformer (Vaswani et al., 2017) .

It follows that the word-level interattention can be rewritten as Attn({h

.

Therefore, the input vector for each context word and question word is:

RNN.

In this component, we use two separate bidirectional LSTMs (Hochreiter & Schmidhuber, 1997) to form the contextualized understanding for C and Q:

where h

and K is the number of RNN layers.

We use variational dropout (Kingma et al., 2015) for the input vector to each layer of RNN, i.e. the dropout mask is shared over different timesteps.

Question Understanding.

For each question word in Q, we employ one more RNN layer to generate a higher level of understanding of the question.

Self-Attention on Question.

As the question has integrated previous utterances, the model needs to directly relate the previously mentioned concept with the current question for context understanding.

Therefore we employ self-attention on question:

is the final representation of question words.

Multilevel Inter-Attention.

After multiple RNN layers extract different levels of semantic abstraction, we conduct inter-attention from question to context based on these representations.

However, the cumulative output dimensions from all previous layers can be very large and computationally inefficient.

Here we leverage the history-of-word idea from FusionNet (Huang et al., 2017) : the attention uses all previous layers to compute scores, but only linearly combines one RNN layer output.

In detail, we conduct K + 1 times of multilevel inter-attention from each RNN layer output of question to context {m

where HoW is the history-of-word vector:

An additional RNN layer is added to context C:

Self Attention on the Context.

Similar to questions, SDNet applies self-attention to the context.

Again, it uses the history-of-word concept to reduce the output dimensionality:

The self-attention is followed by an additional RNN layer to generate the final representation of context words: {u

2.5 OUTPUT LAYER Question Condensation.

The question is condensed into a single representation vector:

where w is a trainable vector.

Generating answer span.

As SDNet outputs answers of interval forms, the output layer generates the probability that the answer starts and ends at the i-th context word, 1 ??? i ??? m:

where W S , W E are parameters.

The use of GRU is to transfer information from start position to end position computation.

Special answer types.

SDNet can also output special types of answer, such as affirmation "yes", negation "no" or no answer "unknown".

We separately generate the probabilities of these three answers: P Y , P N , P U .

For instance, the probability that the answer is "yes", P Y , is computed as:

where W Y and w Y are parametrized matrix and vector, respectively.

During training, all rounds of questions and answers for the same passage form a batch.

The goal is to maximize the probability of the ground-truth answer, including span start/end position, affirmation, negation and no-answer situations.

Therefore, we minimize the cross-entropy loss function L: indicate whether the k-th ground-truth answer is a passage span, "yes", "no" and "unknown", respectively.

During inference, we pick the largest span/yes/no/unknown probability.

The span is constrained to have a maximum length of 15.

We evaluated our model on CoQA (Reddy et al., 2018) , a large-scale conversational question answering dataset.

In CoQA, many questions require understanding of both the passage and previous rounds of questions and answers, which poses challenge to conventional machine reading models.

Table 1 summarizes the domain distribution in CoQA.

As shown, CoQA contains passages from multiple domains, and the average number of question answering turns is more than 15 per passage.

For each in-domain dataset, 100 passages are in the development set, and 100 passages are in the test set.

The rest in-domain dataset are in the training set.

The test set also includes all of the out-of-domain passages.

We use spaCy for word tokenization and employ the uncased BERT-large model to generate contextual embedding.

During training, we use a dropout rate of 0.4 for BERT layer outputs and 0.3 for other layers.

We use Adamax (Kingma & Ba, 2014) as the optimizer, with a learning rate of ?? = 0.002, ?? = (0.9, 0.999) and = 10 ???8 .

We train the model for 30 epochs.

The gradient is clipped at 10.

The word-level attention has a hidden size of 300.

The self attention layer for question words has a hidden size of 300.

The RNNs for question and context have K = 2 layers and each layer has a hidden size of 125.

The multilevel attention from question to context has a hidden size of 250.

The self attention layer for context has a hidden size of 250.

The final RNN layer for context words has a hidden size of 125.

We compare SDNet 2 with the following baseline models: DrQA+PGNet (Reddy et al., 2018 ), BiDAF++ (Yatskar, 2018 and FlowQA (Huang et al., 2018) .

Aligned with the official leaderboard, we use F 1 as the evaluation metric, which is the harmonic mean of precision and recall at word level between the predicted answer and ground truth.

Table 2 shows the performance of SDNet and baseline models.

4 As shown, SDNet achieves significantly better results than baseline models.

In detail, the single SDNet model improves overall F 1 by 1.6%, compared with previous state-of-art model on CoQA, FlowQA.

We also trained an ensemble model consisting of 12 SDNet models with the same structure but different random seeds for initialization.

The ensemble model uses the answer from the most number of models as its predicted answer.

Ensemble SDNet model further improves overall F 1 score by 2.7%.

Figure 3 shows the F 1 score of SDNet on development set during training.

As seen, SDNet overpasses all but one baseline models after the second epoch, and achieves state-of-the-art results after 8 epochs.

Ablation Studies.

We conduct ablation studies on SDNet to verify the effectiveness of different parts of the model.

As Table 3 shows, our proposed weighted sum of per-layer output from BERT is crucial, boosting the performance by 1.75% compared with the canonical method of using only the last layer's output.

This shows that the output from each layer in BERT is useful in downstream tasks.

Using BERT-base instead of the BERT-large pretrained model hurts the F 1 score by 2.61%.Variational dropout and self attention can each improve the performance by 0.24% and 0.75% respectively.

Contextual history.

In SDNet, we utilize conversation history via prepending the current question with previous N rounds of questions and ground-truth answers.

We experimented with the effect of N and present the result in Table 4 .

As shown, excluding dialogue history (N = 0) can reduce the F 1 score by as much as 8.56%, manifesting the importance of contextual information in conversational QA task.

The performance of our model peaks when N = 2, which was used in the final SDNet model.

In this paper, we propose a novel contextual attention-based deep neural network, SDNet, to tackle the conversational question answering task.

By leveraging inter-attention and self-attention on passage and conversation history, the model is able to comprehend dialogue flow and the passage.

Furthermore, we leverage the latest breakthrough in NLP, BERT, as a contextual embedder.

We design the alignment of tokenizers, linear combination and weight-locking techniques to adapt BERT into our model in a computation-efficient way.

SDNet achieves superior results over previous approaches.

On the public dataset CoQA, SDNet outperforms previous state-of-the-art model by 1.6% in overall F 1 score and the ensemble model further improves the F 1 by 2.7%.

Our future work is to apply this model to open-domain multiturn QA problem with large corpus or knowledge base, where the target passage may not be directly available.

This will be a more realistic setting to human question answering.

<|TLDR|>

@highlight

A neural method for conversational question answering with attention mechanism and a novel usage of BERT as contextual embedder