Multi-hop text-based question-answering is a current challenge in machine comprehension.

This task requires to sequentially integrate facts from multiple passages to answer complex natural language questions.

In this paper, we propose a novel architecture, called the Latent Question Reformulation Network (LQR-net), a multi-hop and parallel attentive network designed for question-answering tasks that require reasoning capabilities.

LQR-net is composed of an association of \textbf{reading modules} and \textbf{reformulation modules}. The purpose of the reading module is to produce a question-aware representation of the document.

From this document representation, the reformulation module extracts essential elements to calculate an updated representation of the question.

This updated question is then passed to the following hop.

We evaluate our architecture on the \hotpotqa question-answering dataset designed to assess multi-hop reasoning capabilities.

Our model achieves competitive results on the public leaderboard and outperforms the best current \textit{published} models in terms of Exact Match (EM) and $F_1$ score.

Finally, we show that an analysis of the sequential reformulations can provide interpretable reasoning paths.

The ability to automatically extract relevant information from large text corpora remains a major challenge.

Recently, the task of question-answering has been largely used as a proxy to evaluate the reading capabilities of neural architectures.

Most of the current datasets for question-answering focus on the ability to read and extract information from a single piece of text, often composed of few sentences (Rajpurkar et al., 2016; Nguyen et al., 2016) .

This has strengthened the emergence of easy questions in the sense of Sugawara et al. (2018) and influenced the recent state-of-the-art models to be good at detecting patterns and named entities (Devlin et al., 2018; Yu et al., 2018; Wang et al., 2017) .

However they still lack actual reasoning capabilities.

The problem of reasoning requires machine comprehension models to gather and compose over different pieces of evidence spread across multiple paragraphs.

In this work, we propose an original neural architecture that repeatedly reads from a set of paragraphs to aggregate and reformulate information.

In addition to the sequential reading, our model is designed to collect pieces of information in parallel and to aggregate them in its last layer.

Throughout the model, the important pieces of the document are highlighted by what we call a reading module and integrated into a representation of the question via our reformulation module.

Our contributions can be summarised as follows:

• We propose a machine reading architecture, composed of multiple token-level attention modules, that collect information sequentially and in parallel across a document to answer a question, • We propose to use an input-length invariant question representation updated via a dynamic max-pooling layer that compacts information form a variable-length text sequence into a fixed size matrix, • We introduce an extractive reading-based attention mechanism that computes the attention vector from the output layer of a generic extractive machine reading model,

• We illustrate the advantages of our model on the HOTPOTQA dataset.

The remainder of the paper is organized as follows: Section 2 presents the multi-hop machine reading task, and analyses the required reasoning competencies.

In Section 3, we detail our novel reading architecture and present its different building blocks.

Section 4 presents the conducted experiments, several ablation studies, and qualitative analysis of the results.

Finally, Section 5 discusses related work.

Our code to reproduce the results is publicly available at (removed for review).

2 TEXT-BASED QUESTION-ANSWERING AND MACHINE REASONING Figure 1 : Examples of reasoning paths to answer two questions of the HOTPOTQA dataset.

In this picture, we do not display the full paragraphs, but only the supporting facts.

The task of extractive machine reading can be summarized as follows: given a document D and a question Q, the goal is to extract the span of the document that answers the question.

In this work, we consider the explainable multi-hop reasoning task described in Yang et al. (2018) and its associated dataset: HOTPOTQA .

We focus our experiments on the "distractor" configuration of the dataset.

In this task, the input document D is not a single paragraph but a set of ten paragraphs coming from different English Wikipedia articles.

Answering each question requires gathering and integrating information from exactly two paragraphs; the eight others are distractors selected among the results of a tf-idf retriever (Chen et al., 2017) .

These required paragraphs are called the gold paragraphs.

There are two types of questions proposed in this dataset: extractive ones where the answer is a span of text extracted from the document and binary yes/no questions.

In addition to the answer, it is required to predict the sentences, also called supporting facts, that are necessary to produce the correct answer.

This task can be decomposed in three subtasks: (1) categorize the answer among the three following classes:

yes, no, text span, (2) if it is a span, predict the start and end positions of this span in the document, and (3) predict the supporting sentences required to answer the question.

In addition to the "distractor" experiments, we show how our proposed approach can be used for opendomain question answering and evaluate the entire reading pipeline on the "fullwiki" configuration of the HotpotQA dataset.

In this configuration, no supporting documents are provided, and it is required to answer the question from the entire Wikipedia corpus.

Among the competencies that multi-hop machine reading requires, we identify two major reasoning capabilities that human readers naturally exploit to answer these questions: sequential reasoning and parallel reasoning.

Sequential reasoning requires reading a document, seeking a piece of information, then reformulating the question and finally extracting the correct answer.

This is called multi-hop question-answering and refers to the bridge questions in HOTPOTQA .

Another reasoning pattern is parallel reasoning, required to collect pieces of evidence for comparisons or question that required checking multiple properties in the documents.

Figure 1 presents two examples from HOTPOTQA that illustrate such required competencies.

We hypothesize that these two major reasoning patterns should condition the design of the proposed neural architectures to avoid restricting the model to one or the other reasoning skill.

In this section, we describe the Latent Question Reformulation Network (LQR-net), shown in Figure  2 .

This multi-hop model is designed as an association of four modules: (1) an encoding module, (2) a reading module, (3) a question reformulation module, and (4) an answering module.

(1) and (4) are input and output modules, whereas (2) and (3) constitute a hop, and are repeated respectively T and T − 1 times: the answering module does not require a last reformulation step.

Figure 2: Overview of LQR-net with K parallel heads and T sequential reading modules.

In this architecture, a latent representation of the question is sequentially updated to perform multi-hop reasoning.

K independent reading heads collect pieces of information before feeding them to the answering module.

Sections 3 present the different building blocks of this end-to-end trainable model.

Given a document and a question, the reading module is in charge of computing a question-aware representation of the document.

Then, the reformulation module extracts essential elements from this document representation and uses them to update a representation of the question in a latent space.

This reformulated question is then passed to the following hop.

The model can have multiple heads, as in the Transformer architecture (Vaswani et al., 2017) .

In this case, the iterative mechanism is performed several times in parallel in order to compute a set of independent reformulations.

The final representations of the document produced by the different heads are eventually aggregated before being fed to the answering module.

This module predicts the answer and the supporting facts from the document.

The following parts of this section describe each module that composes this model.

Note: The model is composed of K independent reading heads that process the document and question in parallel.

To not overload the notations of the next parts, we do not subscript all the matrices by the index of the head and focus on the description of one.

The aggregation process of the multi-head outputs is explained in Section 3.5.

We adopt a standard representation of each token by using the pre-trained parametric language model BERT (Devlin et al., 2018) .

Let a document D = {p 1 , p 2 , . . .

, p 10 } be the set of input paragraphs, of respective lengths {n 1 , . . .

, n 10 }, associated to a question Q of length L. These paragraphs are independently encoded through the pre-trained BERT model.

Each token is represented by its associated BERT hidden state from the last layer of the model.

The tokens representations are then concatenated to produce a global representation of the set of 10 paragraphs of total length N = 10 i=1 n i .

The representations are further passed through a Bidirectional Gated Recurrent Unit (BiGRU) (Cho et al., 2014) to produce the final representation of the document E D ∈ R N ×2h and question E Q ∈ R L×2h , where h is the hidden state dimension of the BiGRUs.

where [; ] is the concatenation operation.

To compute the first representation of the question U (0) , we use an interpolation layer to map

where M is an hyperparameter of the model.

Intuitively, R M ×2h

corresponds to the space allocated to store the representation of the question and its further reformulations.

It does not depend on the length of the original question L.

Our model is composed of T hops of reading that sequentially extract relevant information from a document regarding the current reformulation of the question.

At step t, given a representation of the reformulated question U (t) ∈ R M ×2h and a representation of the document E D ∈ R N ×2h , this module computes a question-aware representation of the document.

This module is a combination of two layers: a document-question attention followed by a document self-attention.

We first construct the interaction matrix between the document and the current reformulation of the question S ∈ R N ×M as:

where w 1 , w 2 , w 3 are trainable vectors of R 2h and the element-wise multiplication.

Then, we compute the document-to-question attention C q ∈ R N ×2h :

And the question-to-document attention q c ∈ R 2h :

Finally, we compute the question-aware representation of the document X (t) ∈ R N ×8h :

where [;] concatenation operation.

Finally, we use a last BiGRU that reduces the dimension of X (t) to N × 2h.

This specific attention mechanism was first introduced in the Bidirectional Attention Flow model of Seo et al. (2017) .

We hypothesize that such token-level attention will produce a finer-grained representation of the document compared to sentence-level attention used in state-of-the-art Memory Network architectures.

Document Self-Attention: So far, the contextualization between the ten paragraphs has only be done by the BiGRUs of equation 1.

One limitation of the current representation of the document is that each token has very limited knowledge of the other elements of the context.

To deal with long-range dependencies, we apply this same attention mechanism between the question-aware representation of the document, X (t) , and itself to produce the reading module output V ∈ R N ×2h .

This self-contextualization of the document has been found useful in our experiments as presented in the ablation analysis of Section 4.3.

A reformulation module t takes as input the output of the previous attention module V (t) , the previous representation of the reformulated question U (t) , and an encoding of the document E D .

It produces an updated reformulation of the question U (t+1) .

Reading-based Attention: Given V (t) we compute p (t)s ∈ R N and p (t)e ∈ R N using two BiGRUs followed by a linear layer and a softmax operator.

They are computed from:

where w e and w s are trainable vectors of R h .

The two probability vectors p (t)s and p (t)e are not used to predict an answer but to compute a reading-based attention vector a (t) over the document.

Intuitively, these probabilities represent the belief of the model at step t of the probability for each word to be the beginning and the end of the answer span.

We define the reading-based attention of a token as the probability that the predicted span has started before this token and will end after.

It can be computed as follows:

Finally, we use these attention values to re-weight each token of the document representation.

We

Dynamic Max-Pooling: This layer aims at collecting the relevant elements ofẼ (t)D to add to the current representation of dimension M × 2h.

We partition the row of the initial sequence into M approximately equal parts.

It produces a grid of M × 2h in which we apply a max-pooling operator in each individual window.

As a result, a matrix of fixed dimension adequately represents the input, preserving the global structure of the document, and focusing on the important elements of each region.

This can be seen as an adaptation of the dynamic pooling layer proposed by Socher et al. (2011) .

(t)D be the input matrix representation, we dynamically compute the kernel size, w, of the max-pooling according to the length of the input sequence and the required output shape: w = N M , · being the ceiling function.

Then the output representation of this pooling layer will be

Finally, to compute the updated representation of the question U (t+1) ∈ R M ×2h , we sum U (t) and O (t) .

The answering module is a sequence of four BiGRUs, each of them followed by a fully connected layer.

Their respective goal is to supervise (1) the supporting facts p sf , (2) the answer starting and (3) ending probabilities, p e , p s , of each word of the document.

(4) The last layer is used as a three-way classifier to predict p c the probability of the answer be classified as yes, no or a span of text.

where w s ∈ R h , w e ∈ R h , W c ∈ R h×3 are trainable parameters.

To predict the supporting facts, we construct a sentence based representation of the document.

Each sentence is represented by the concatenation of its starting and ending supporting fact tokens from Y sf .

We compute p sf i,j the probability of sentence j of example i of being a supporting fact with a linear layer followed by a sigmoid function.

We define a multi-head version of the model.

In this configuration, we use a set of independent parallel heads.

All heads are composed of the same number of reading and reformulation modules.

Each head produces a representation V (T ) k of the document.

We finally sum these K matrices to compute the input of the answering block.

We jointly optimize the model on the three subtasks (supporting facts, span position, classifier yes/no/span) by minimising a linear combination of the supporting facts loss L sf , the span loss L span and the class loss L class .

Let N d be the number of examples in the training dataset.

L sf (θ) is defined by:

where nbs i corresponds to the number of sentences in the document i. y

(1)

i,j being 1 if the sentence j of the document i is a supporting fact otherwise 0.

Selecting the answer in multi-hop reading datasets is a weakly supervised task.

Indeed, similarly to the observations of Min et al. (2019a) for open-domain question-answering and discrete reasoning tasks, it is frequent for a given answer of HOTPOTQA to appear multiple times in its associated document.

In our case, we assume that all the mentions of the answer in the supporting facts are related to the question.

We tag as a valid solution, the start and end positions of all occurrences of the answer in the given supporting facts.

where y

∈ R N are vectors containing the value 1/n i at the start, end positions of all the occurrences of the answer, 0 otherwise; n i being the number of occurrences of the answer in the context.

where y (4) i corresponds to the index of the label of the question type {yes, no, span}. We finally define the training loss as follows:

where α and β are hyperparameters tuned by cross-validation.

In the original HOTPOTQA dataset, the two gold paragraphs required to answer a given question come with eight distractor paragraphs.

These eight distractor paragraphs, collected from Wikipedia, are selected among the results of a bigram tf-idf retriever (Chen et al., 2017) using the question as the query.

As an augmentation strategy, we created additional "easier" examples by combining the two gold paragraphs with eight other paragraphs randomly selected in the dataset.

For each example of the original training set, we generate an additional "easier" example.

These examples are shuffled in the dataset.

Our model is composed of 3 parallel heads (K = 3) each of them composed of two reading modules and one reformulation module (T = 2).

We set the hidden dimension of all the GRUs to d = 80.

We use M = 100 to allocate a space of R 100×160 to store the question and its reformulations.

We use (Min et al., 2019b) 55 pre-trained BERT-base-cased model (Devlin et al., 2018) and adapt the implementation of Hugging Face 1 to compute embedding representations of documents and questions.

We optimize the network using the Adam optimizer (Kingma & Ba, 2015) with an initial learning rate of 1e −4 .

We set α to 1 and β to 10.

All these parameters have been defined through cross-validation.

Table 1 presents the performance of our LQR-net on the distractor setting of the HOTPOTQA dataset.

We compare our model against the published approaches evaluated on the HOTPOTQA dataset.

We can see from this table that our model achieves strong performance on the answer prediction task.

It outperforms the current best model by 3.9 points of EM and 4.1 points of F 1 score.

Our model also achieves competitive performance for the evidence extraction task.

The LQR-net achieves state-ofthe-art performance on the joint task improving the best published approaches by 2.9 points on EM and 3.9 points of F 1 .

To evaluate the impact of the different components of our model, we perform an ablation analysis.

Table 2 presents the results of this analysis.

Impact of sequential and parallel reading: We study the contributions of the sequentiality in the model and of the multiple parallel heads.

We compare our model to a similar architecture without the sequential reformulation (T = 1).

We find that this sequential association of reading modules and reformulation modules is a critical component.

F 1 score decreases by 6.9 points for the answer prediction task and 5.7 points for the evidence extraction task when the model does not have the capability to reformulate the question.

The impact of the parallel heads is more limited than the sequentiality but still remains significant.

Indeed, the configuration that uses only a single head (K = 1) stands 1 F 1 points below the best model on the joint metric.

Weak supervision of the answer: In this work, we propose to label as positive all occurrences of the answer in the supporting facts.

We compare this configuration to the standard approach, where only the first occurrence of the answer is labeled as positive and the others as negative.

In this last configuration, the span loss corresponds to a cross-entropy loss (CE loss) between the predicted start and end probabilities and the target positions.

This decreases the joint F 1 score by 0.8 points.

Impact of the self-attention layer: We study the impact of the self-attention layer in the reading module.

We found that this self-attention layer is an essential component in the reading process.

Indeed, when we omit this layer, the F 1 score decreases by 8.3 points on the joint metric.

This outlines the necessity to be able to propagate long-range information between the different paragraphs and not only in the local neighborhood of a token.

Compared to previously proposed approaches, this layer does not rely on any handcrafted relationship across words.

Question as a single vector: Finally, we study the case where the question representation is reduced to a vector of R 2h (M = 1).

This configuration achieves the worst results of our analysis, dropping the joint F 1 score by 13.3 points and highlights the importance of preserving a representation of the question as a matrix to maintain its meaning.

In this part, we describe how we integrated our model into an entire reading pipeline for opendomain question answering.

In this setting, no supporting documents are associated to each question, and it is required to retrieve relevant context from large text corpora such as Wikipedia.

We adopt a two-stage process, similar to Chen et al. (2017) ; Clark & Gardner (2018) , to answer multihop complex questions based on the 5 million documents of Wikipedia.

First, we use a paragraph retriever to select a limited amount of relevant paragraphs from a Wikipedia dump, regarding a natural language question.

Second, we fed our LQR model with the retrieved paragraphs to extract the predicted answer.

We evaluate this approach on the open-domain configuration of the HotpotQA dataset called fullwiki.

We use a standard TF-IDF based paragraph retriever to retrieve the paragraphs the most related to the question.

In addition to these paragraphs, we consider as relevant their neighbors in the Wikipedia graph, i.e. the documents linked to them by hyperlinks.

In our experiments, we considered as relevant, the top 10 paragraphs and their associated neighbors.

Table 3 shows the results of our approach compared to other published models.

Although we are using a very simple retriever, only based on TF-IDF, we report strong results on the open-domain question answering task of HotpotQA.

The only published approach (Nie et al., 2019 ) that outperforms us being a combination of sentence/paragraph retrieval based on BERT encodings.

Question Reformulation and Reasoning Chains: Because our model reformulates the question in a latent space, we cannot directly visualize the text of the reformulated question.

However, one way to assess the effectiveness of this reformulation is to analyze the evolution of p s and p e across the two hops of the model.

We present in Figure 3 an analysis of the evolution of these probabilities on two bridge samples of the development dataset.

We display the reading-based attention, that corresponds to the probabilities for each word to be part of the predicted span, computed from p s and p e in Equation 7.

These examples show this attention before the first reformulation of the question and in the answering module.

From these observations, we can see that the model tends to follow a natural reasoning path to answer bridge questions.

Indeed, before the first reformulation module, the attentions tend to focus on the first step of reasoning.

For the question "What award did the writer of Never Let Me Go novel win in 1989?", the model tends to focus on the name of the writer at the first step, before jumping the award description in the second step.

Similarly, for the question "What is the population according to the 2007 population census of the city in which the National Archives and Library of Ethiopia is

SemanticRetrievalMRS (Nie et al., 2019) (Ding et al., 2019) 37.60 49.40 23.10 58.5 12.2 35.3 MUPPET (Feldman & El-Yaniv, 2019) 31.07 40.42 17.00 47.71 11.76 27.62 QFE † (Nishida et al., 2019) 28.70 38.10 14.20 44.40 8.69 23.1 Baseline Model (Yang et al., 2018) 24.68 34.36 5.28 40.98 2.54 17.73 DecompRC (Min et al., 2019b) N/A 43.26 N/A N/A N/A N/A Table 3 : Performance comparison on the development set of HOTPOTQA in the fullwiki setting.

We compare our model in terms of Exact Match and F 1 scores against the published models at the time of submission (November 15th).

† indicates that the paper does not report the results on the development set of the dataset; we display their results on the test set.

Figure 3: Distribution of the probabilities for each word to be part of the predicted span, before the first reformulation module and in the answering module.

We display the reading-based attention computed in Equation 7 and the reading-based attention computed from p s and p e from Equation 10.

In these examples, we show only the supporting facts.

located?

" we can see the model focusing on Addis Ababa at the first step, i.e the name of the city where the National Archives and Library of Ethiopia are located and then jumping to the population of this city in the next hop.

We display more visualizations of the sequential evolution of the answer probabilities in Appendix A.

Limitations:

We manually examine one hundred errors produced by our multi-step reading architecture on the development set of HOTPOTQA .

We identify three recurrent cases of model failure:

(1) the model stops at the first hop of required reasoning, (2) the model fails at comparing two properties, and (3) the answer does not match all the requirements of the question.

We illustrate these three recurrent types of error with examples from the dataset in Appendix B.

During this analysis of errors, we found that in only 3% of the cases, the answer is selected among one of the distractor paragraphs instead of a gold one.

Our architecture successfully detects the relevant paragraphs regarding a question even among similar documents coming from a tf-idf retriever.

Moreover, there are no errors where the model produces a binary yes/no answer instead of extracting a text span and vice versa.

Identifying the type of question is not challenging for the model.

This might be explained by the question's "patterns" that are generally different between binary yes/no and extractive questions.

Multi-hop Machine Comprehension: The question-answering task has recently increased its popularity as a way to assess machine reading comprehension capabilities.

The emergence of large scale datasets such as CNN/Daily Mail, (Hermann et al., 2015) , SQuAD (Rajpurkar et al., 2016) or MSMARCO (Nguyen et al., 2016) have encouraged the development of multiple machine reading models (Devlin et al., 2018; Tan et al., 2017) .

These models are mainly composed of multiple attention layers that update the representation of the document conditioned by a representation of the question.

However, most of this work focuses on the ability to answer questions from a single paragraph, often limited to a few sentences.

Weston et al. (2015a) ; Joshi et al. (2017) were the first attempts to introduce the task of multi-documents question-answering.

QAngaroo (Welbl et al., 2018 ) is another dataset designed to evaluate multi-hop reading architectures.

However, state-of-the-art architectures on this task Cao et al., 2019) tend to exploit the structure of the dataset by using the proposed candidate spans as an input of the model.

Recently, different approaches have been developed for HOTPOTQA focusing on the multiple challenges of the dataset.

Nishida et al. (2019) focuses on the evidence extraction task and highlight its similarity with the extractive summarization task.

Related works also focus on the interpretation of the reasoning chain with an explicit decomposition of the question (Min et al., 2019b) or a decomposition of the reasoning steps (Jiang & Bansal, 2019) .

Other models like Qiu et al. (2019) aim at integrating a graph reasoning type of attention where the nodes are recognized by a BERT NER model over the document.

Moreover, this model leverages on handcrafted relationships between tokens.

Related to our approach, different papers have investigated the idea of question reformulation to build multi-hop open-domain question answering models.

Das et al. (2019) proposes a framework composed of iterative interaction between a document retriever and a reading model.

The question reformulation is performed by a multi-step-reasoner module trained via reinforcement learning.

Similarly, Feldman & El-Yaniv (2019) introduces a multi-hop paragraph retriever.

They propose a reformulation component integrated into a retrieving pipeline to iteratively retrieve relevant documents.

These works are complementary to ours by focusing mostly on the document retrieving part of the problem while we focus on the answer extraction task, and could be combined together.

Memory Networks: Memory networks are a generic type of architecture Weston et al. (2015b) ; Sukhbaatar et al. (2015) ; Miller et al. (2016) designed to iteratively collect information from memory cells using attention mechanism.

They have been used to read from sentences, paragraphs, and knowledge bases.

In these models, the answer layer uses the last value of the controller to predict the answer.

Two main differences with our architecture are the representation of the controller and the associated attention mechanism.

Indeed, in these models, the controller is reduced to a single vector, and the attention mechanism is based on a simple dot-product between each token of the document and the representation of the controller.

We utilize a token-level attention mechanism compared to the sentence-level one, classically used in Memory Networks.

Transformer Networks: The transformer architecture has been introduced by Vaswani et al. (2017) in the context of machine translation.

It is mainly composed of attention layers in both the encoder and the decoder module.

The transformer networks introduced the so-called multi-head attention, consisting of several attention layers running in parallel.

This multi-head attention allows the model to concurrently access information from different representations of the input vector.

Inspired by this work, we designed our multi-head module to read in parallel into different representations of the document while solely accumulate information into the representation of the question.

In this paper, we propose a novel multi-hop reading model designed for question-answering tasks that explicitly require reasoning capabilities.

We have designed our model to gather information sequentially and in parallel from a given set of paragraphs to answer a natural language question.

Our neural architecture, uses a sequence of token-level attention mechanisms to extract relevant information from the paragraphs and update a latent representation of the question.

Our proposed model achieves competitive results on the HOTPOTQA reasoning task and performs better than the current best published approach in terms of both Exact Match and F 1 score.

In addition, we show that an analysis of the sequential attentions can possibly provide human-interpretable reasoning chains.

This section includes examples from the HOTPOTQA development set that illustrate the evolution of the probabilities for each word to be part of the predicted span, before the first reformulation module and in the answering module presented in Section 4.5.

For each example, we show only the text of the two gold paragraphs.

identifies the supporting facts in these visualizations.

This section includes examples from the HOTPOTQA development set that illustrate the categories of errors presented in Section 4.5.

For each example, we show only the text of the two gold paragraphs.

identifies the supporting facts in these visualizations.

The model stops at the first hop of required reasoning:

The model fails at comparing two properties:

<|TLDR|>

@highlight

In this paper, we propose the Latent Question Reformulation Network (LQR-net), a multi-hop and parallel attentive network designed for question-answering tasks that require reasoning capabilities.