Bidirectional Encoder Representations from Transformers (BERT) reach state-of-the-art results in a variety of Natural Language Processing tasks.

However, understanding of their internal functioning is still insufficient and unsatisfactory.

In order to better understand BERT and other Transformer-based models, we present a layer-wise analysis of BERT's hidden states.

Unlike previous research, which mainly focuses on explaining Transformer models by their \hbox{attention} weights, we argue that hidden states contain equally valuable information.

Specifically, our analysis focuses on models fine-tuned on the task of Question Answering (QA) as an example of a complex downstream task.

We inspect how QA models transform token vectors in order to find the correct answer.

To this end, we apply a set of general and QA-specific probing tasks that reveal the information stored in each representation layer.

Our qualitative analysis of hidden state visualizations provides additional insights into BERT's reasoning process.

Our results show that the transformations within BERT go through phases that are related to traditional pipeline tasks.

The system can therefore implicitly incorporate task-specific information into its token representations.

Furthermore, our analysis reveals that fine-tuning has little impact on the models' semantic abilities and that prediction errors can be recognized in the vector representations of even early layers.

In recent months, Transformer models have become more and more prevalent in the field of Natural Language Processing.

Originally they became popular for their improvements over RNNs in Machine Translation BID36 .

Now however, with the advent of large models and an equally large amount of pre-training being done, they have Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page.

Copyrights for components of this work owned by others than the author(s) must be honored.

Abstracting with credit is permitted.

To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee.

Request permissions from permissions@acm.org.

CIKM '19, November 3rd-7th, 2019, Beijing, China.

© 2019 Copyright held by the owner/author(s).

Publication rights licensed to Association for Computing Machinery.

ACM ISBN 978-x-xxxx-xxxx-x/YY/MM. . .

$15.00 https://doi.org/10.1145/nnnnnnn.nnnnnnn proven adept at solving many of the standard Natural Language Processing tasks.

Main subject of this paper is BERT BID8 , arguably the most popular of the recent Transformer models and the first to display significant improvements over previous state-of-the-art models in a number of different benchmarks and tasks.

Problem of black box models.

Deep Learning models achieve increasingly impressive results across a number of different domains, whereas their application to real-world tasks has been moving somewhat more slowly.

One major impediment lies in the lack of transparency, reliability and prediction guarantees in these largely black box models.

While Transformers are commonly believed to be moderately interpretable through the inspection of their attention values, current research suggests that this may not always be the case BID15 .

This paper takes a different approach to the interpretation of said Transformer Networks.

Instead of evaluating attention values, our approach examines the hidden states between encoder layers directly.

There are multiple questions this paper will address:(1) Do Transformers answer questions decompositionally, in a similar manner to humans?

(2) Do specific layers in a multi-layer Transformer network solve different tasks?

(3) What influence does fine-tuning have on a network's inner state?

(4) Can an evaluation of network layers help come to a conclusion on why and how a network failed to predict a correct answer?We discuss these questions on the basis of fine-tuned models on standard QA datasets.

We choose the task of Question Answering as an example of a complex downstream task that, as this paper will show, requires solving a multitude of other Natural Language Processing tasks.

Additionally, it has been shown that other NLP tasks can be successfully framed as QA tasks BID22 , therefore our analysis should translate to these tasks as well.

While this work focuses on the BERT architecture, we perform preliminary tests on the small GPT-2 model BID28 as well, which yield similar results.

First, we propose a layer-wise visualisation of token representations that reveals information about the internal state of Transformer networks.

This visualisation can be used to expose wrong predictions even in earlier layers or to show which parts of the context the model considered as Supporting Facts.

Second, we apply a set of general NLP Probing Tasks and extend them by the QA-specific tasks of Question Type Classification and Supporting Fact Extraction.

This way we can analyse the abilities within BERT's layers and how they are impacted by fine-tuning.

Third, we show that BERT's transformations go through similar phases, even if fine-tuned on different tasks.

Information about general language properties is encoded in earlier layers of BERT and implicitly used to solve the downstream task at hand in later layers.

Transformer Models.

Our analyses focus on BERT, which belongs to the group of Transformer networks, named after how representations are transformed throughout the network layers.

We also partly include the more recent Transformer model GPT-2 BID28 .

This model represents OpenAI's improved version of GPT BID27 and while GPT-2 has not yet climbed leaderboards like BERT has, its larger versions have proven adept enough at the language modeling task, that Open-AI has decided not to release their pre-trained models.

There are also other Transformer models of note, where a similar analysis might prove interesting in future work.

Chief among them are the Universal Transformer BID7 and TransformerXL BID6 , both of which aim to improve some of the flaws of the Transformer architecture by adding a recurrent inductive bias.

Interpretability and Probing.

Explainability and Interpretability of neural models have become an increasingly large field of research.

While there are a multitude of ways to approach these topics BID9 BID12 BID19 , we especially highlight relevant work in the area of research that builds and applies probing tasks and methodologies, post-hoc, to trained models.

There have been a number of recent advances on this topic.

While the majority of the current works aim to create or apply more general purpose probing tasks BID2 BID4 BID33 , BERT specifically has also been probed in previous papers.

Tenney et al. BID34 proposes a novel "edge-probing" framework consisting of nine different probing tasks and applies it to the contextualized word embeddings of ELMo, BERT and GPT-1.

Both semantic and syntactic information is probed, but only pre-trained models are studied, and not specifically fine-tuned ones.

A similar analysis BID11 adds more probing tasks and addresses only the BERT architecture.

Qiao et al. BID26 focus specifically on analysing BERT in the context of a Ranking task.

The authors probe attention values in different layers and measure performance for representations build from different BERT layers.

Like BID34 , they only discuss pre-trained models.

There has also been work which studies models not through probing tasks but through qualitative visual analysis.

Zhang and Zhu BID41 offer a survey of different approaches, though limited to CNNs.

Nagamine et al. BID24 explore phoneme recognition in DNNs by studying single node activations in the task of speech recognition.

Hupkes et al. BID14 go one step further, by not only doing a qualitative analysis, but also training diagnostic classifiers to support their hypotheses.

Finally, Li et al. BID17 take a look at word vectors and the importance of some of their specific dimensions on both sequence tagging and classification tasks.

The most closely related previous work is proposed by Liu et al. BID20 .

Here, the authors also perform a layer-wise analysis of BERT's token representations.

However, their work solely focuses on probing pre-trained models and disregards models fine-tuned on downstream tasks.

Furthermore, it limits the analysis to the general transferability of the network and does not analyze the specific phases that BERT goes through.

Additionally, our work is motivated by Jain and Wallace BID15 .

In their paper, the authors argue that attention, at least in some cases, is not well suited to solve the issues of explainability and interpretability.

They do so both by constructing adversarial examples and by a comparison with more traditional explainability methods.

In supporting this claim, we propose revisiting evaluating hidden states and token representations instead.

We focus our analysis on fine-tuned BERT models.

In order to understand which transformations the models apply to input tokens, we take two approaches: First, we analyse the transforming token vectors qualitatively by examining their positions in vector space.

Second, we probe their language abilities on QA-related tasks to examine our results quantitatively.

The architecture of BERT and Transformer networks in general allows us to follow the transformations of each token throughout the network.

We use this characteristic for an analysis of the changes that are being made to the tokens' representations in every layer.

We use the following approach for a qualitative analysis of these transformations: We randomly select both correctly and falsely predicted samples from the test set of the respective dataset.

For these samples we collect the hidden states from each layer while removing any padding.

This results in the representation of each token throughout the model's layers.

The model can transform the vector space freely throughout its layers and we do not have references for semantic meanings of positions within these vector spaces.

Therefore we consider distances between token vectors as indication for semantic relations.

Dimensionality reduction.

BERT's pre-trained models use vector dimensions of 1024 (large model) and 512 (base model).

In order to visualize relations between tokens, we apply dimensionality reduction and fit the vectors into two-dimensional space.

To that end we apply T-distributed Stochastic Neighbor Embedding (t-SNE) BID35 , Principal Component Analysis (PCA) BID10 and Independent Component Analysis (ICA) BID3 to vectors in each layer.

As the results of PCA reveal the most distinct clusters for our data, we use it to present our findings.

K-means clustering.

In order to verify that clusters in 2D space represent the actual distribution in high-dimensional vector space, we additionally apply a k-means clustering BID21 .

We choose the number of clusters k in regard to the number of observed clusters in PCA, which vary over layers.

The resulting clusters correspond with our observations in 2D space.

Our goal is to further understand the abilities of the model after each transformation.

We therefore apply a set of semantic probing tasks to analyze which information is stored within the transformed tokens after each layer.

We want to know whether specific layers How Does BERT Answer Questions?CIKM '19, November 3rd-7th, 2019, Beijing, China. are reserved for specific tasks and how language information is maintained or forgotten by the model.

We use the principle of Edge Probing introduced by Tenney et al. BID34 .

Edge Probing translates core NLP tasks into classification tasks by focusing solely on their labeling part.

This enables a standardized probing mechanism over a wide range of tasks.

We adopt the tasks Named Entity Labeling, Coreference Resolution and Relation Classification from the original paper as they are prerequisites for language understanding and reasoning BID39 .

We add tasks of Question Type Classification and Supporting Fact Identification due to their importance for Question Answering in particular.

BID0 Named Entity Labeling.

Given a span of tokens the model has to predict the correct entity category.

This is based on Named Entity Recognition but formulated as a Classification problem.

The task was modeled by BID34 , annotations are based on the OntoNotes 5.0 corpus BID38 and contain 18 entity categories.

Coreference Resolution.

The Coreference task requires the model to predict whether two mentions within a text refer to the same entity.

The task was built from the OntoNotes corpus and enhanced with negative samples by BID34 .Relation Classification.

In Relation Classification the model has to predict which relation type connects two known entities.

The task was constructed by BID34 with samples taken from the SemEval 2010 Task 8 dataset consisting of English web text and nine directional relation types.

BID0 We will make the source code to all experiments publicly available.

Question Type Classification.

A fundamental part of answering a question is to correctly identify its question type.

For this Edge Probing task we use the Question Classification dataset constructed by Li and Roth BID18 based on the TREC-10 QA dataset BID37 .

It includes 500 fine-grained types of questions within the larger groups of abbreviation, entity, description, human, location and numeric value.

We use the whole question as input to the model with its question type as label.

Supporting Facts.

The extraction of Supporting Facts is a main prerequisite for Question Answering tasks, especially in the multihop case.

We examine what BERT's token transformations can tell us about the mechanism behind distinguishing distracting from important context parts.

To understand at which stage this distinction is done, we construct a probing task for identifying Supporting Facts.

The model has to predict whether a sentence contains supporting facts regarding a specific question or whether it is irrelevant.

Through this task we test the hypothesis that token representations contain information about their significance to the question.

Both HotpotQA and bAbI contain information about sentencewise Supporting Facts for each question.

SQuAD does not require multi-hop reasoning, we therefore consider the sentence containing the answer phrase the Supporting Fact.

We also exclude all QA-pairs that only contain one context sentence.

We construct a different probing task for each dataset in order to check their task-specific ability to recognize relevant parts.

All samples are labeled sentencewise with true if they are a supporting fact or false otherwise.

Probing Setup.

Analogue to the authors of BID34 , we embed input tokens for each probing task sample with our fine-tuned BERT model.

Contrary to previous work, we do this for all layers (N = 12 for BERT-base and N = 24 for BERT-large), using only the output embedding from n-th layer at step n. The concept of Edge Probing defines that only tokens of "labeled edges" (e.g. tokens of two related entities for Relation Classification) within a sample are considered for classification.

These tokens are first pooled for a fixed-length representation and afterwards fed into a two-layer Multi-layer Perceptron (MLP) classifier, that predicts label-wise probability scores (e.g. for each type of relation).

A schematic overview of this setting is shown in FIG0 .

We perform the same steps on pretrained BERT-base and BERT-large models without any fine-tuning.

This enables us to identify which abilities the model learns during pre-training or fine-tuning.

Our aim is to understand how BERT works on complex downstream tasks.

Question Answering (QA) is one of such tasks that require a combination of multiple simpler tasks such as Coreference Resolution and Relation Modeling to arrive at the correct answer.

We take three current Question Answering datasets into account, namely SQUAD BID30 , bAbI BID39 and HotpotQA BID40 .

We intentionally choose three very different datasets to diversify the results of our analysis.

What is a common punishment in the UK and Ireland?

What is Emily afraid of?

Answer detention cats

Currently detention is one of the most common punishments in schools in the United States, the UK, Ireland, Singapore and other countries.

It requires the pupil to remain in school at a given time in the school day (such as lunch, recess or after school); or even to attend school on a non-school day, e.g. "Saturday detention" held at some schools.

During detention, students normally have to sit in a classroom and do work, write lines or a punishment essay, or sit quietly.

HotpotQA.

This Multihop QA task contains 112.000 natural questionanswer pairs.

The questions are especially designed to combine information from multiple parts of a context.

For our analysis we focus on the distractor-task of HotpotQA, in which the context is composed of both supporting and distracting facts with an average size of 900 words.

As the pre-trained BERT model is restricted to an input size of 512 tokens, we reduce the amount of distracting facts by a factor of 2.7.

We also leave out yes/no-questions (7% of questions) as they require additional specific architecture, diluting our analysis.bAbI. The QA bAbI tasks are a set of artificial toy tasks developed to further understand the abilities of neural models.

The 20 tasks require reasoning over multiple sentences (Multihop QA) and are modeled to include Positional Reasoning, Argument Relation Extraction and Coreference Resolution.

The tasks strongly differ from the other QA tasks in their simplicity (e.g. vocabulary size of 230 and short contexts) and the artificial nature of sentences.

In this section we briefly discuss the models our analysis is based on, BERT BID8 and GPT-2 BID28 .

Both of these models are Transformers that extend and improve on a number of different recent ideas.

These include previous Transformer models BID36 [29], SemiSupervised Sequence Learning BID5 , ELMo BID25 and ULMFit BID13 .

Both have a similar architecture, and they each represent one half of the original encoder-decoder Transformer BID36 .

While GPT-2, like its predecessor, consists of only the decoder half, BERT uses a FIG0 depicts how these models integrate into our probing setup.

We base our training code on the Pytorch implementation of BERT available at [15] .

We use the publicly available pre-trained BERT models for our experiments.

In particular, we study the monolingual models bert-base-uncased and bert-large.

For GPT-2 the small model (117M Parameters) is used, as a larger model has not yet been released.

However, we do not apply these models directly, and instead fine-tune them on each of our datasets.

Training Modalities.

Regarding hyperparameters, we tune the learning rate, batch size and learning rate scheduling according to a grid search and train each model for 5 Epochs with evaluations on the development set every 1000 iterations.

We then select the model of the best evaluation for further analysis.

The input length chosen is 384 tokens for the bAbI and SQuAD tasks and the maximum of 512 tokens permitted by the pre-trained models' positional embedding for the HotpotQA tasks.

For BAbI we evaluate both models that are trained on a single bAbI task and also a multitask model,

How Does BERT Answer Questions?

CIKM '19, November 3rd-7th, 2019, Beijing, China.

that was trained on the data of all 20 tasks.

We further distinguish between two settings: Span prediction, which we include for better comparison with the other datasets, and Sequence Classification, which is the more common approach to bAbI. In order to make span prediction work, we append all possible answers to the end of the base context, since not all answers can be found in the context by default.

For HotpotQA, we also distinguish between two tasks.

In the HotpotQA Support Only (SP) task, we use only the sentences labeled as Supporting Facts as the question context.

This simplifies the task, but more importantly it reduces context length and increases our ability to distinguish token vectors.

Our HotpotQA Distractor task is closer to the original HotpotQA task.

It includes distracting sentences in the context, but only enough to not exceed the 512 token limit.

Training Results.

TAB3 shows the evaluation results of our best models.

Accuracy on the SQuAD task is close to human performance, indicating that the model can fulfill all sub-tasks required to answer SQuAD's questions.

As expected the tasks derived from HotpotQA prove much more challenging, with the distractor setting being the most difficult to solve.

Unsurprisingly too, bAbI was easily solved by both BERT and GPT-2.

While GPT-2 performs significantly worse in the more difficult tasks of SQuAD and HotpotQA, it does considerably better on bAbi reducing the validation error to nearly 0.

Most of BERT's error in the bAbI multi-task setting comes from tasks 17 and 19.

Both of these tasks require positional or geometric reasoning, thus it is reasonable to assume that this is a skill where GPT-2 improves on BERT's reasoning capabilities.

Presentation of Analysis Results.

The qualitative analysis of vector transformations reveals a range of recurring patterns.

In the following, we present these patterns by two representative samples from the SQuAD and bAbI task dataset described in TAB1 .

Examples from HotpotQA can be found in the supplementary material as they require more space due to the larger context.

Results from probing tasks are displayed in FIG1 .

We compare results in macro-averaged F1 over all network layers.

FIG1 shows results from three models of BERT-base with twelve layers: Fine-tuned on SQuAD,on bAbI tasks and without fine-tuning.

FIG2 reports results of two models based on BERT-large with 24 layers: Fine-tuned on HotpotQA and without fine-tuning.

The PCA representations of tokens in different layers suggest that the model is going through multiple phases while answering a question.

We observe these phases in all three selected QA tasks despite their diversity.

These findings are supported by results of the applied probing tasks.

We present the four phases in the following paragraphs and describe how our experimental results are linked.(1) Semantic Clustering.

Early layers within the BERT-based models group tokens into topical clusters.

Figures 4a and 5a reveal this behaviour and show the second layer of each model.

Resulting vector spaces are similar in nature to embedding spaces from e.g. Word2Vec BID23 and hold little task-specific information.

Therefore, these initial layers reach low accuracy on semantic probing tasks, as shown in FIG1 .

BERT's early layers can be seen as an implicit replacement of embedding layers common in neural network architectures.(2) Connecting Entities with Mentions and Attributes.

In the middle layers of the observed neural networks, we see clusters of entities that are less connected by their topical similarity.

Rather, they are connected by their relation within a certain input context.

These task-specific clusters appear to already include a filtering of question-relevant entities.

FIG4 shows a cluster with words like countries, schools, detention and country names, in which 'detention' is a common practice in schools.

This cluster helps to solve the question "

What is a common punishment in the UK and Ireland?".

Another question-related cluster is shown in FIG6 .

The main challenge within this sample is to identify the two facts that Emily is a wolf and Wolves are afraid of cats.

The highlighted cluster implies that Emily has been recognized as a relevant entity that holds a relation to the entity Wolf.

The cluster also contains other mentions of these entities including the plural form Wolves.

We observe similar clusters in the HotpotQA model, which includes more cases of coreferences.

The probing results support these observations.

The model's ability to recognize entities (Named Entity Labeling), to identify their mentions (Coreference Resolution) and to find relations (Relation Recognition) improves until higher network layers.

FIG7 visualizes these abilities.

Information about Named Entities is learned first, whereas recognizing coreferences or relations are more difficult tasks and require input from additional layers until the model's performance peaks.

These patterns are equally observed in the results from BERT-base models and BERT-large models.(3) Matching Questions with Supporting Facts.

Identifying relevant parts of the context is crucial for Question Answering and Information Retrieval in general.

In traditional pipeline models this step is often achieved by filtering context parts based on their similarity to the question BID16 .

We observe that BERT models perform a corresponding step by transforming the tokens so that question tokens are matched onto relevant context tokens.

FIG4 show two examples in which the model transforms the token representation of question and Supporting Facts into the same area of the vector space.

Some samples show this behaviour in lower layers.

However, results from our probing tasks reveal that the models hold the strongest ability to distinguish relevant from irrelevant information wrt.

the question in their higher layers.

FIG1 demonstrates how the performance for this task increases over successive layers for SQuAD and bAbI. Performance of the fine-tuned HotpotQA model in FIG2 is less distinct from the model without fine-tuning and does not reach high accuracy.

BID1 This inability indicates why the BERT model does not perform well on this dataset as it is not able to identify the correct Supporting Facts.

The vector representations enable us to tell which facts a model considered important (and therefore matched with the question).

This helps retracing decisions and makes the model more transparent.(4) Answer Extraction.

In the last network layers we see that the model dissolves most of the previous clusters.

Here, the model separates the correct answer tokens, and sometimes other possible candidates, from the rest of the tokens.

The remaining tokens form one or multiple homogeneous clusters.

The vector representation at this point is largely task-specific and learned during fine-tuning.

This becomes visible through the performance drop in general NLP probing tasks, visualized in FIG7 .

We especially observe this loss of information in last-layer representations in the large BERTmodel fine-tuned on HotpotQA, as shown in FIG2 .

While the model without fine-tuning still performs well on tasks like NEL or COREF, the fine-tuned model loses this ability.

Analogies to Human Reasoning.

The phases of answering questions can be compared to the human reasoning process, including decomposition of input into parts BID0 .

The first phase of semantic clustering represents our basic knowledge of language and the second phase how a human reader builds relations between parts of the context to connect information needed for answering a question.

Separation of important from irrelevant information (phase 3) and grouping of potential answer candidates (phase 4) are also known from human reasoning.

However, the order of these steps might differ from the human abstraction.

One major difference is that while humans read sequentially, BERT can see all parts of the input at once.

Thereby it is able to run multiple processes and phases concurrently depending on the task at hand.

FIG7 shows how the tasks overlap during the answering process.

In this section we compare our insights from the BERT models to the GPT-2 model.

We focus on the qualitative analysis of token representations and leave the application of probing tasks for future work.

One major difference between GPT-2's and BERT's hidden states is that GPT-2 seems to give particular attention to the first token of a sequence.

While in our QA setup this is often the question word, this also happens in cases where it is not.

During dimensionality reduction this results in a separation of two clusters, namely the first token and all the rest.

This problem holds true for all layers of GPT-2 except for the Embedding Layer, the first Transformer block and the last one.

For this reason we mask the first token during dimensionality reduction in further analysis.

FIG8 shows an example of the last layer's hidden state for our bAbI example.

Like BERT, GPT-2 also separates the relevant Supporting Facts and the question in the vector space.

Additionally, GPT-2 extracts another sentence, which is not a Supporting Fact, but is similar in meaning and semantics.

In contrast to BERT, the correct answer "cats" is not particularly separated and instead simply left as part of its sentence.

These findings in GPT-2 suggest that our analysis extends beyond the BERT architecture and hold true for other Transformer networks as well.

Our future work will include more probing tasks to confirm this initial observation.

Observation of Failure States.

One important aspect of explainable Neural Networks is to answer the questions of when, why, and how the network fails.

Our visualizations are not only able to show such failure states, but even the rough difficulty of a specific task can be discerned by a glance at the hidden state representations.

While for correct predictions the transformations run through the phases discussed in previous sections, for wrong predictions there exist two possibilities: If a candidate answer was found that the network has a reasonable amount of confidence in, the phases will look very similar to a correct prediction, but now centering on the wrong answer.

Inspecting early layers in this case can give insights towards the reason why the wrong candidate was chosen, e.g. wrong Supporting Fact selected, misresolution of coreferences etc.

An example of this is shown in FIG9 , where a wrong answer is based on the fact that the wrong Supporting Fact was matched with the question in early layers.

If network confidence is low however, which is often the case when the predicted answer is far from the actual answer, the transformations do not go through the phases discussed earlier.

The vector space is still transformed in each layer, but tokens are mostly kept in a single homogenous cluster.

If something is extracted it usually has little to do with the prediction.

In some cases, especially when the confidence of the network is low in general, the network maintains Phase (1), 'Semantic Clustering' analogue to Word2Vec, even in later layers.

An example is depicted in the supplementary material.

Impact of Fine-tuning.

FIG1 show how little impact fine-tuning has on the core NLP abilities of the model.

The pretrained model already holds sufficient information about words and their relations, which is the reason it works well in multiple downstream tasks.

Fine-tuning only applies small weight changes and forces the model to forget some information in order to fit specific tasks.

However, the model does not forget much of the previously learned encoding when fitting the QA task, which indicates why the Transfer Learning approach proves successful.

Maintained Positional Embedding.

It is well known that the positional embedding is a very important factor in the performance of Transformer networks.

It solves one major problem that Transformers have in comparison with RNNs, that they lack sequential information BID36 .

Our visualizations support this importance and show that even though the positional embedding is only added once before the first layer, its effects are maintained even into very late layers depending on the task.

FIG10 demonstrates this behavior on the SQuAD dataset.

Abilities to resolve Question Type.

The performance curves regarding the Question Type probing task illustrate another interesting result.

FIG1 demonstrates that the model fine-tuned on SQuAD outperforms the base model from layer 5 onwards.

This indicates the relevancy of resolving the question type for the SQuAD task, which leads to an improved ability after fine-tuning.

The opposite is the case for the model fine-tuned on the bAbI tasks, which loses part of its ability to distinguish question types during fine-tuning.

This is likely caused by the static structure of bAbI samples, in which the answer candidates can be recognized by sentence structure and occurring word patterns rather than by the question type.

Surprisingly, we see that the model fine-tuned on HotpotQA does not outperform the model without fine-tuning in FIG2 .

Both models can solve the task in earlier layers, which suggests that the ability to recognize question types is pre-trained in BERT-large.

Our work reveals important findings about the inner functioning of Transformer networks.

The impact of these findings and how future work can build upon them is described in the following: CIKM '19, November 3rd-7th, 2019, Beijing, China.

Anon.

Interpretability.

The qualitative analysis of token vectors reveals that there is indeed interpretable information stored within the hidden states of Transformer models.

This information can be used to identify misclassified examples and model weaknesses.

It also provides clues about which parts of the context the model considered important for answering a question -a crucial part of decision legitimisation.

We leave the development of methods to further process this information for future work.

Transferability.

We further show that lower layers might be more applicable to certain problems than later ones.

For a Transfer Learning task, this means layer depth should be chosen individually depending on the task at hand.

We also suggest further work regarding skip connections in Transformer layers to examine whether direct information transfer between non-adjacent layers (that solve different tasks) can be of advantage.

Modularity.

Our findings support the hypothesis that not only do different phases exist in Transformer networks, but that specific layers seem to solve different problems.

This hints at a kind of modularity that can potentially be exploited in the training process.

For example, it could be beneficial to fit parts of the network to specific tasks in pre-training, instead of using an end-to-end language model task.

Our work aims towards revealing some of the internal processes within Transformer-based models.

We suggest to direct further research at thoroughly understanding state-of-the-art models and the way they solve downstream tasks, in order to improve on them.

<|TLDR|>

@highlight

We investigate hidden state activations of Transformer Models in Question Answering Tasks.