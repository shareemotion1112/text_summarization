This paper introduces a new neural structure called FusionNet, which extends existing attention approaches from three perspectives.

First, it puts forward a novel concept of "History of Word" to characterize attention information from the lowest word-level embedding up to the highest semantic-level representation.

Second, it identifies an attention scoring function that better utilizes the "history of word" concept.

Third, it proposes a fully-aware multi-level attention mechanism to capture the complete information in one text (such as a question) and exploit it in its counterpart (such as context or passage) layer by layer.

We apply FusionNet to the Stanford Question Answering Dataset (SQuAD) and it achieves the first position for both single and ensemble model on the official SQuAD leaderboard at the time of writing (Oct. 4th, 2017).

Meanwhile, we verify the generalization of FusionNet with two adversarial SQuAD datasets and it sets up the new state-of-the-art on both datasets: on AddSent, FusionNet increases the best F1 metric from 46.6% to 51.4%; on AddOneSent, FusionNet boosts the best F1 metric from 56.0% to 60.7%.

Context: The Alpine Rhine is part of the Rhine, a famous European river.

The Alpine Rhine begins in the most western part of the Swiss canton of Graubünden, and later forms the border between Switzerland to the West and Liechtenstein and later Austria to the East.

On the other hand, the Danube separates Romania and Bulgaria.

Answer: Liechtenstein Teaching machines to read, process and comprehend text and then answer questions is one of key problems in artificial intelligence.

FIG0 gives an example of the machine reading comprehension task.

It feeds a machine with a piece of context and a question and teaches it to find a correct answer to the question.

This requires the machine to possess high capabilities in comprehension, inference and reasoning.

This is considered a challenging task in artificial intelligence and has already attracted numerous research efforts from the neural network and natural language processing communities.

Many neural network models have been proposed for this challenge and they generally frame this problem as a machine reading comprehension (MRC) task BID7 BID13 BID17 BID15 BID16 Weissenborn et al., 2017; .The key innovation in recent models lies in how to ingest information in the question and characterize it in the context, in order to provide an accurate answer to the question.

This is often modeled as attention in the neural network community, which is a mechanism to attend the question into the context so as to find the answer related to the question.

Some Weissenborn et al., 2017) attend the word-level embedding from the question to context, while some BID13 attend the high-level representation in the question to augment the context.

However we observed that none of the existing approaches has captured the full information in the context or the question, which could be vital for complete information comprehension.

Taking image recognition as an example, information in various levels of representations can capture different aspects of details in an image: pixel, stroke and shape.

We argue that this hypothesis also holds in language understanding and MRC.

In other words, an approach that utilizes all the information from the word embedding level up to the highest level representation would be substantially beneficial for understanding both the question and the context, hence yielding more accurate answers.

However, the ability to consider all layers of representation is often limited by the difficulty to make the neural model learn well, as model complexity will surge beyond capacity.

We conjectured this is why previous literature tailored their models to only consider partial information.

To alleviate this challenge, we identify an attention scoring function utilizing all layers of representation with less training burden.

This leads to an attention that thoroughly captures the complete information between the question and the context.

With this fully-aware attention, we put forward a multi-level attention mechanism to understand the information in the question, and exploit it layer by layer on the context side.

All of these innovations are integrated into a new end-to-end structure called FusionNet in FIG5 , with details described in Section 3.We submitted FusionNet to SQuAD (Rajpurkar et al., 2016) , a machine reading comprehension dataset.

At the time of writing (Oct. 4th, 2017) , our model ranked in the first place in both single model and ensemble model categories.

The ensemble model achieves an exact match (EM) score of 78.8% and F1 score of 85.9%.

Furthermore, we have tested FusionNet against adversarial SQuAD datasets BID9 .

Results show that FusionNet outperforms existing state-of-the-art architectures in both datasets: on AddSent, FusionNet increases the best F1 metric from 46.6% to 51.4%; on AddOneSent, FusionNet boosts the best F1 metric from 56.0% to 60.7%.

In Appendix D, we also applied to natural language inference task and shown decent improvement.

This demonstrated the exceptional performance of FusionNet.

An open-source implementation of FusionNet can be found at https://github.com/momohuang/FusionNet-NLI.

In this section, we briefly introduce the task of machine comprehension as well as a conceptual architecture that summarizes recent advances in machine reading comprehension.

Then, we introduce a novel concept called history-of-word.

History-of-word can capture different levels of contextual information to fully understand the text.

Finally, a light-weight implementation for history-of-word, Fully-Aware Attention, is proposed.

In machine comprehension, given a context and a question, the machine needs to read and understand the context, and then find the answer to the question.

The context is described as a sequence of word tokens: C = {w C 1 , . . .

, w C m }, and the question as: DISPLAYFORM0 where m is the number of words in the context, and n is the number of words in the question.

In general, m n. The answer Ans can have different forms depending on the task.

In the SQuAD dataset (Rajpurkar et al., 2016) , the answer Ans is guaranteed to be a contiguous span in the context C, e.g., Ans = {w C i , . . . , w C i+k }, where k is the number of words in the answer and k ≤ m.

In all state-of-the-art architectures for machine reading comprehension, a recurring pattern is the following process.

Given two sets of vectors, A and B, we enhance or modify every single vector in set A with the information from set B. We call this a fusion process, where set B is fused into set A. Fusion processes are commonly based on attention BID0 , but some are not.

Major improvements in recent MRC work lie in how the fusion process is designed.

A conceptual architecture illustrating state-of-the-art architectures is shown in FIG2 , which consists of three components.• Input vectors: Embedding vectors for each word in the context and the question.

RaSoR BID11

DrQA D MPCM (Wang et al., 2016) D DMnemonic Reader D D D R-net BID13 D D • Integration components: The rectangular box.

It is usually implemented using an RNN such as an LSTM BID7 or a GRU BID5 ).•

Fusion processes: The numbered arrows (1) , (2) , (2'), (3), (3').

The set pointing outward is fused into the set being pointed to.

There are three main types of fusion processes in recent advanced architectures.

TAB0 shows what fusion processes are used in different state-of-the-art architectures.

We now discuss them in detail.(1) Word-level fusion.

By providing the direct word information in question to the context, we can quickly zoom in to more related regions in the context.

However, it may not be helpful if a word has different semantic meaning based on the context.

Many word-level fusions are not based on attention, e.g., appends binary features to context words, indicating whether each context word appears in the question.(2) High-level fusion.

Informing the context about the semantic information in the question could help us find the correct answer.

But high-level information is more imprecise than word information, which may cause models to be less aware of details.(2') High-level fusion (Alternative).

Similarly, we could also fuse high-level concept of Q into the word-level of C.(3) Self-boosted fusion.

Since the context can be long and distant parts of text may rely on each other to fully understand the content, recent advances have proposed to fuse the context into itself.

As the context contains excessive information, one common choice is to perform self-boosted fusion after fusing the question Q.

This allows us to be more aware of the regions related to the question.(3') Self-boosted fusion (Alternative).

Another choice is to directly condition the self-boosted fusion process on the question Q, such as the coattention mechanism proposed in BID16 ).

Then we can perform self-boosted fusion before fusing question information.

A common trait of existing fusion mechanisms is that none of them employs all levels of representation jointly.

In the following, we claim that employing all levels of representation is crucial to achieving better text understanding.

Consider the illustration shown in Figure 3 .

As we read through the context, each input word will gradually transform into a more abstract representation, e.g., from low-level to high-level concepts.

Altogether, they form the history of each word in our mental flow.

For a human, we utilize the history-of-word so frequently but we often neglect its importance.

For example, to answer the question in Figure 3 correctly, we need to focus on both the high-level concept of forms the border and the word-level information of Alpine Rhine.

If we focus only on the high-level concepts, we will Figure 3 : Illustrations of the history-of-word for the example shown in FIG0 .

Utilizing the entire history-of-word is crucial for the full understanding of the context.confuse Alpine Rhine with Danube since both are European rivers that separate countries.

Therefore we hypothesize that the entire history-of-word is important to fully understand the text.

In neural architectures, we define the history of the i-th word, HoW i , to be the concatenation of all the representations generated for this word.

This may include word embedding, multiple intermediate and output hidden vectors in RNN, and corresponding representation vectors in any further layers.

To incorporate history-of-word into a wide range of neural models, we present a lightweight implementation we call Fully-Aware Attention.

Attention can be applied to different scenarios.

To be more conclusive, we focus on attention applied to fusing information from one body to another.

Consider two sets of hidden vectors for words in text bodies A and B: {h DISPLAYFORM0 Their associated history-of-word are, In fully-aware attention, we replace attention score computation with the history-of-word.

DISPLAYFORM1

).

This allows us to be fully aware of the complete understanding of each word.

The ablation study in Section 4.4 demonstrates that this lightweight enhancement offers a decent improvement in performance.

To fully utilize history-of-word in attention, we need a suitable attention scoring function S(x, y).

A commonly used function is multiplicative attention BID2 ): DISPLAYFORM0 k×d h , and k is the attention hidden size.

However, we suspect that two large matrices interacting directly will make the neural model harder to train.

Therefore we propose to constrain the matrix U T V to be symmetric.

A symmetric matrix can always be decomposed into DISPLAYFORM1 and D is a diagonal matrix.

The symmetric form retains the ability to give high attention score between dissimilar HoW A i , HoW B j .

Additionally, we marry nonlinearity with the symmetric form to provide richer interaction among different parts of the history-of-word.

The final formulation for attention score is DISPLAYFORM2 is an activation function applied element-wise.

In the following context, we employ f (x) = max(0, x).

A detailed ablation study in Section 4 demonstrates its advantage over many alternatives.

DISPLAYFORM3 In the following, we consider the special case where text A is context C and text B is question Q. An illustration for FusionNet is shown in FIG5 .

It consists of the following components.

Input Vectors.

First, each word in C and Q is transformed into an input vector w.

We utilize the 300-dim GloVe embedding (Pennington et al., 2014) and 600-dim contextualized vector BID16 ).

In the SQuAD task, we also include 12-dim POS embedding, 8-dim NER embedding and a normalized term frequency for context C as suggested in .

Together {w DISPLAYFORM4 Fully-Aware Multi-level Fusion: Word-level.

In multi-level fusion, we separately consider fusing word-level and higher-level.

Word-level fusion informs C about what kind of words are in Q. It is illustrated as arrow (1) in FIG2 .

For this component, we follow the approach in First, a feature vector em i is created for each word in C to indicate whether the word occurs in the question Q. Second, attention-based fusion on GloVe embedding g i is used DISPLAYFORM5 where W ∈ R 300×300 .

Since history-of-word is the input vector itself, fully-aware attention is not employed here.

The enhanced input vector for context isw DISPLAYFORM6 Reading.

In the reading component, we use a separate bidirectional LSTM (BiLSTM) to form low-level and high-level concepts for C and Q. DISPLAYFORM7 Hence low-level and high-level concepts h l , h h ∈ R 250 are created for each word.

Question Understanding.

In the Question Understanding component, we apply a new BiLSTM taking in both h Ql , h Qh to obtain the final question representation U Q : DISPLAYFORM8 where {u DISPLAYFORM9 are the understanding vectors for Q. Fully-Aware Multi-level Fusion: Higher-level.

This component fuses all higher-level information in the question Q to the context C through fully-aware attention on history-of-word.

Since the proposed attention scoring function for fully-aware attention is constrained to be symmetric, we need to identify the common history-of-word for both C, Q. This yields DISPLAYFORM10 where g i is the GloVe embedding and c i is the CoVe embedding.

Then we fuse low, high, and understanding-level information from Q to C via fully-aware attention.

Different sets of attention weights are calculated through attention function S l (x, y), S h (x, y), S u (x, y) to combine low, high, and understanding-level of concepts.

All three functions are the proposed symmetric form with nonlinearity in Section 2.3, but are parametrized by independent parameters to attend to different regions for different level.

Attention hidden size is set to be k = 250.

2.

High-level fusion:ĥ DISPLAYFORM0 3.

Understanding fusion:û DISPLAYFORM1 This multi-level attention mechanism captures different levels of information independently, while taking all levels of information into account.

A new BiLSTM is applied to obtain the representation for C fully fused with information in the question Q: DISPLAYFORM2 Fully-Aware Self-Boosted Fusion.

We now use self-boosted fusion to consider distant parts in the context, as illustrated by arrow (3) in FIG2 .

Again, we achieve this via fully-aware attention on history-of-word.

We identify the history-of-word to be DISPLAYFORM3 We then perform fully-aware attention,v DISPLAYFORM4 The final context representation is obtained by DISPLAYFORM5 where {u DISPLAYFORM6 are the understanding vectors for C. After these components in FusionNet, we have created the understanding vectors, U C , for the context C, which are fully fused with the question Q. We also have the understanding vectors, U Q , for the question Q.

We focus particularly on the output format in SQuAD (Rajpurkar et al., 2016) where the answer is always a span in the context.

The output of FusionNet are the understanding vectors for both C and Q, U C = {u DISPLAYFORM0 We then use them to find the answer span in the context.

Firstly, a single summarized question understanding vector is obtained through u DISPLAYFORM1 ) and w is a trainable vector.

Then we attend for the span start using the summarized question understanding vector DISPLAYFORM2 d×d is a trainable matrix.

To use the information of the span start when we attend for the span end, we combine the context understanding vector for the span start with u Q through a GRU BID5 DISPLAYFORM3 , where u Q is taken as the memory and DISPLAYFORM4 as the input in GRU.

Finally we attend for the end of the span using v Q , DISPLAYFORM5 d×d is another trainable matrix.

Training.

During training, we maximize the log probabilities of the ground truth span start and end, DISPLAYFORM6 e k are the answer span for the k-th instance.

Prediction.

We predict the answer span to be i s , i e with the maximum P DISPLAYFORM7

In this section, we first present the datasets used for evaluation.

Then we compare our end-toend FusionNet model with existing machine reading models.

Finally, we conduct experiments to validate the effectiveness of our proposed components.

Additional ablation study on input vectors can be found in Appendix C. Detailed experimental settings can be found in Appendix E.

We focus on the SQuAD dataset (Rajpurkar et al., 2016) to train and evaluate our model.

SQuAD is a popular machine comprehension dataset consisting of 100,000+ questions created by crowd workers on 536 Wikipedia articles.

Each context is a paragraph from an article and the answer to each question is guaranteed to be a span in the context.

While rapid progress has been made on SQuAD, whether these systems truly understand language remains unclear.

In a recent paper, BID9 proposed several adversarial schemes to test the understanding of the systems.

We will use the following two adversarial datasets, AddOneSent and AddSent, to evaluate our model.

For both datasets, a confusing sentence is appended at the end of the context.

The appended sentence is model-independent for AddOneSent, while AddSent requires querying the model a few times to choose the most confusing sentence.

We submitted our model to SQuAD for evaluation on the hidden test set.

We also tested the model on the adversarial SQuAD datasets.

Two official evaluation criteria are used: Exact Match (EM) and F1 score.

EM measures how many predicted answers exactly match the correct answer, while F1 score measures the weighted average of the precision and recall at token level.

The evaluation results for our model and other competing approaches are shown in TAB3 .

BID13 Additional comparisons with state-of-the-art models in the literature can be found in Appendix A.For the two adversarial datasets, AddOneSent and AddSent, the evaluation criteria is the same as SQuAD.

However, all models are trained only on the original SQuAD, so the model never sees the

Single Model EM / F1 LR Baseline (Rajpurkar et al., 2016) 40.4 / 51.0 Match-LSTM (Wang & Jiang, 2016) 64.7 / 73.7 BiDAF BID17 68.0 / 77.3 SEDT (Liu et al., 2017) 68.2 / 77.5 RaSoR BID11 70.8 / 78.7 DrQA 70.7 / 79.4 BID15 70.6 / 79.4 R. Mnemonic Reader TAB4 and TAB5 , respectively.

From the results, we can see that our models not only perform well on the original SQuAD dataset, but also outperform all previous models by more than 5% in EM score on the adversarial datasets.

This shows that FusionNet is better at language understanding of both the context and question.

In this experiment, we compare the performance of different attention scoring functions S(x, y) for fully-aware attention.

We utilize the end-to-end architecture presented in Section 3.1.

Fully-aware attention is used in two places, fully-aware multi-level fusion: higher level and fully-aware selfboosted fusion.

Word-level fusion remains unchanged.

Based on the discussion in Section 2.3, we consider the following formulations for comparison:1.

Additive attention (MLP) BID0 : DISPLAYFORM0 2.

Multiplicative attention: DISPLAYFORM1 3.

Scaled multiplicative attention: 4.

Scaled multiplicative with nonlinearity: DISPLAYFORM2 DISPLAYFORM3 5.

Our proposed symmetric form: DISPLAYFORM4 6.

Proposed symmetric form with nonlinearity: DISPLAYFORM5 We consider the activation function f (x) to be max(0, x).

The results of various attention functions on SQuAD development set are shown in Table 5 .

It is clear that the symmetric form consistently outperforms all alternatives.

We attribute this gain to the fact that symmetric form has a single large †: This is a unpublished version of R-net.

The published version of R-net BID13 matrix U .

All other alternatives have two large parametric matrices.

During optimization, these two parametric matrices would interfere with each other and it will make the entire optimization process challenging.

Besides, by constraining U T V to be a symmetric matrix U T DU , we retain the ability for x to attend to dissimilar y. Furthermore, its marriage with the nonlinearity continues to significantly boost the performance.

In FusionNet, we apply the history-of-word and fully-aware attention in two major places to achieve good performance: multi-level fusion and self-boosted fusion.

In this section, we present experiments to demonstrate the effectiveness of our application.

In the experiments, we fix the attention function to be our proposed symmetric form with nonlinearity due to its good performance shown in Section 4.3.

The results are shown in Table 6 , and the details for each configuration can be found in Appendix B.High-Level is a vanilla model where only the high-level information is fused from Q to C via standard attention.

When placed in the conceptual architecture FIG2 ), it only contains arrow (2) without any other fusion processes.

FA High-Level is the High-Level model with standard attention replaced by fully-aware attention.

FA All-Level is a naive extension of FA High-Level, where all levels of information are concatenated and is fused into the context using the same attention weight.

FA Multi-Level is our proposed Fully-aware Multi-level fusion, where different levels of information are attended under separate attention weight.

Self C = None means we do not make use of self-boosted fusion.

Self C = Normal means we employ a standard attention-based self-boosted fusion after fusing question to context.

This is illustrated as arrow (3) in the conceptual architecture FIG2 ).Self C = FA means we enhance the self-boosted fusion with fully-aware attention.

High-Level vs. FA High-Level.

From Table 6 , we can see that High-Level performs poorly as expected.

However enhancing this vanilla model with fully-aware attention significantly increase the performance by more than 8%.

The performance of FA High-Level already outperforms many state-of-the-art MRC models.

This clearly demonstrates the power of fully-aware attention.

FA All-Level vs. FA Multi-Level.

Next, we consider models that fuse all levels of information from question Q to context C. FA All-Level is a naive extension of FA High-Level, but its performance is actually worse than FA High-Level.

However, by fusing different parts of history-of-word in Q independently as in FA Multi-Level, we are able to further improve the performance.

Self C options.

We have achieved decent performance without self-boosted fusion.

Now, we compare adding normal and fully-aware self-boosted fusion into the architecture.

Comparing None and Normal in Table 6 , we can see that the use of normal self-boosted fusion is not very effective under our improved C, Q Fusion.

Then by comparing with FA, it is clear that through the enhancement of fully-aware attention, the enhanced self-boosted fusion can provide considerable improvement.

Together, these experiments demonstrate that the ability to take all levels of understanding as a whole is crucial for machines to better understand the text.

In this paper, we describe a new deep learning model called FusionNet with its application to machine comprehension.

FusionNet proposes a novel attention mechanism with following three contributions: 1.

the concept of history-of-word to build the attention using complete information from the lowest word-level embedding up to the highest semantic-level representation; 2. an attention scoring function to effectively and efficiently utilize history-of-word; 3.

a fully-aware multi-level fusion to exploit information layer by layer discriminatingly.

We applied FusionNet to MRC task and experimental results show that FusionNet outperforms existing machine reading models on both the SQuAD dataset and the adversarial SQuAD dataset.

We believe FusionNet is a general and improved attention mechanism and can be applied to many tasks.

Our future work is to study its capability in other NLP problems.

In this appendix, we present details for the configurations used in the ablation study in Section 4.4.

For all configurations, the understanding vectors for both the context C and the question Q will be generated, then we follow the same output architecture in Section 3.2 to apply them to machine reading comprehension problem.

Next we consider the standard attention-based fusion for the high level representation.

DISPLAYFORM0 Then we concatenate the attended vectorĥ FA High-Level.

The only difference to High-Level is the enhancement of fully-aware attention.

This is as simple as changing DISPLAYFORM1 where DISPLAYFORM2 is the common history-of-word for both context and question.

All other places remains the same as High-Level.

This simple change results in significant improvement.

The performance of FA High-Level can already outperform many state-of-the-art models in the literature.

Note that our proposed symmetric form with nonlinearity should be used to guarantee the boost.

Next we make use of the fully-aware attention similar to FA High-Level, but take back the entire history-of-word.

DISPLAYFORM3 Then we concatenate the attended history-of-wordĤoW DISPLAYFORM4 The understanding vectors for the question is similar to the Understanding component in Section 3.1, DISPLAYFORM5 We have now generated the understanding vectors for both the context and the question.

FA Multi-Level.

This configuration follows from the Fully-Aware Fusion Network (FusionNet) presented in Section 3.1.

The major difference compared to FA All-Level is that different layers in the history-of-word uses a different attention weight α while being fully aware of the entire historyof-word.

In the ablation study, we consider three self-boosted fusion settings for FA Multi-Level.

The Fully-Aware setting is the one presented in Section 3.1.

Here we discuss all three of them in detail.• For the None setting in self-boosted fusion, no self-boosted fusion is used and we use two layers of BiLSTM to mix the attended information.

The understanding vectors for the context C is the hidden vectors in the final layers of the BiLSTM, Then we fuse the context information into itself through standard attention, DISPLAYFORM6 DISPLAYFORM7 The final understanding vectors for the context C is the output hidden vectors after passing the concatenated vectors into a BiLSTM, DISPLAYFORM8 • For the Fully-Aware setting, we change S ij = S(v C i , v C j ) in the Normal setting to the fully-aware attention DISPLAYFORM9 All other places remains the same.

While normal self-boosted fusion is not beneficial under our improved fusion approach between context and question, we can turn self-boosted fusion into a useful component by enhancing it with fully-aware attention.

72.1 / 81.6 Table 7 : Ablation study on input vectors (GloVe and CoVe) for SQuAD dev set.

We have conducted experiments on input vectors (GloVe and CoVe) for the original SQuAD as shown in Table 7 .

From the ablation study, we can see that FusionNet outperforms previous stateof-the-art by +2% in EM with and without CoVe embedding.

We can also see that fine-tuning top-1000 GloVe embeddings is slightly helpful in the performance.

Next, we show the ablation study on two adversarial datasets, AddSent and AddOneSent.

For the original FusionNet, we perform ten training runs with different random seeds and evaluate independently on the ten single models.

The performance distribution of the ten training runs can be seen in FIG15 .

Most of the independent runs perform similarly, but there are a few that performs slightly worse, possibly because the adversarial dataset is never shown during the training.

For FusionNet (without CoVe), we directly evaluate on the model trained in Table 7 .

From TAB8 and 9, we can see that FusionNet, single or ensemble, with or without CoVe, are all better than previous best performance by a significant margin.

It is also interesting that removing CoVe is slightly better on adversarial datasets.

We assert that it is because AddSent and AddOneSent target the over-stability of machine comprehension models BID9 .

Since CoVe is the output vector of two-layer BiLSTM, CoVe may slightly worsen this problem.

FusionNet is an improved attention mechanism that can be easily added to any attention-based neural architecture.

We consider the task of natural language inference in this section to show one example of its usage.

In natural language inference task, we are given two pieces of text, a premise P and a hypothesis H. The task is to identify one of the following scenarios:1.

Entailment -the hypothesis H can be derived from the premise P .

2.

Contradiction -the hypothesis H contradicts the premise P .

3.

Neutral -none of the above.

We focus on Multi-Genre Natural Language Inference (MultiNLI) corpus (Williams et al., 2017) recently developed by the creator of Stanford Natural Language Inference (SNLI) dataset BID1 .

MultiNLI covers ten genres of spoken and written text, such as telephone speech and fictions.

However the training set only contains five genres.

Thus there are in-domain and crossdomain accuracy during evaluation.

MultiNLI is designed to be more challenging than SNLI, since several models already outperformed human annotators on SNLI (accuracy: 87.7%) 3 .A state-of-the-art model for natural language inference is Enhanced Sequential Inference Model (ESIM) by BID4 , which achieves an accuray of 88.0% on SNLI and obtained 72.3% (in-domain), 72.1% (cross-domain) on MultiNLI (Williams et al., 2017) .

We implemented a version of ESIM in PyTorch.

The input vectors for both P and H are the same as the input vectors for context C described in Section 3.

Therefore, DISPLAYFORM0 Then, two-layer BiLSTM with shortcut connection is used to encode the input words for both premise P and hypothesis H, i.e., DISPLAYFORM1 Hh j ∈ R 300 .

Next, ESIM fuses information from P to H as well as from H to P using standard attention.

We consider the following, DISPLAYFORM2 We set the attention hidden size to be the same as the dimension of hidden vectors h. Next, ESIM feed g P i , g H j into separate BiLSTMs to perform inference.

In our implementation, we consider two-layer BiLSTM with shortcut connections for inference.

The hidden vectors for the two-layer BiLSTM are concatenated to yield {u DISPLAYFORM3 The final hidden vector for the P , H pair is obtained by DISPLAYFORM4 The final hidden vector h P,H is then passed into a multi-layer perceptron (MLP) classifier.

The MLP classifier has a single hidden layer with tanh activation and the hidden size is set to be the same as the dimension of u P i and u H j .

Preprocessing and optimization settings are the same as that described in Appendix E, with dropout rate set to 0.3.

Now, we consider improving ESIM with our proposed attention mechanism.

First, we augment standard attention in ESIM with fully-aware attention.

This is as simple as replacing DISPLAYFORM5 where HoW i is the history-of-word, DISPLAYFORM6 All other settings remain unchanged.

To incorporate fully-aware multi-level fusion into ESIM, we change the input for inference BiLSTM from DISPLAYFORM7 are computed through independent fully-aware attention weights and d is the dimension of hidden vectors h. Word level fusion discussed in Section 3.1 is also included.

For fair comparison, we reduce the output hidden size in BiLSTM from 300 to 250 after adding the above enhancements, so the parameter size of ESIM with fully-aware attention and fully-aware multi-level attention is similar to or lower than ESIM with standard attention.

The results of ESIM under different attention mechanism is shown in TAB0 .

Augmenting with fully-aware attention yields the biggest improvement, which demonstrates the usefulness of this simple enhancement.

Further improvement is obtained when we use multi-level fusion in our ESIM.

Experiments with and without CoVe embedding show similar observations.

Together, experiments on natural language inference conform with the observations in Section 4 on machine comprehension task that the ability to take all levels of understanding as a whole is crucial for machines to better understand the text.

We make use of spaCy for tokenization, POS tagging and NER.

We additionally fine-tuned the GloVe embeddings of the top 1000 frequent question words.

During training, we use a dropout rate of 0.

4 (Srivastava et al., 2014) after the embedding layer (GloVe and CoVe) and before applying any linear transformation.

In particular, we share the dropout mask when the model parameter is shared BID6 .The batch size is set to 32, and the optimizer is Adamax BID10 ) with a learning rate α = 0.002, β = (0.9, 0.999) and = 10 −8 .

A fixed random seed is used across all experiments.

All models are implemented in PyTorch (http://pytorch.org/).

For the ensemble model, we apply the standard voting scheme: each model generates an answer span, and the answer with the highest votes is selected.

We break ties randomly.

There are 31 models in the ensemble.

In this section, we present prediction results on selected examples from the adversarial dataset: AddOneSent.

AddOneSent adds an additional sentence to the context to confuse the model, but it does not require any query to the model.

The prediction results are compared with a state-of-the-art architecture in the literature, BiDAF BID17 .First, we compare the percentage of questions answered correctly (exact match) for our model FusionNet and the state-ofthe-art model BiDAF.

The comparison is shown in FIG16 .

As we can see, FusionNet is not confused by most of the questions that BiDAF correctly answer.

Among the 3.3% answered correctly by BiDAF but not FusionNet, ∼ 1.6% are being confused by the added sentence; ∼ 1.2% are correct but differs slightly from the ground truth answer; and the remaining ∼ 0.5% are completely incorrect in the first place.

Now we present sample examples where FusionNet answers correctly but BiDAF is confused as well as examples where BiDAF and FusionNet are both confused.

ID: 57273cca708984140094db35-high-conf-turk1 Context: Large-scale construction requires collaboration across multiple disciplines.

An architect normally manages the job, and a construction manager, design engineer, construction engineer or project manager supervises it.

For the successful execution of a project, effective planning is essential.

Those involved with the design and execution of the infrastructure in question must consider zoning requirements, the environmental impact of the job, the successful scheduling, budgeting, construction-site safety, availability and transportation of building materials, logistics, inconvenience to the public caused by construction delays and bidding, etc.

The largest construction projects are referred to as megaprojects.

Confusion is essential for the unsuccessful execution of a project.

FusionNet Prediction: 587,000 BiDAF Prediction: 187000 ID: 5726509bdd62a815002e815c-high-conf-turk1 Context: The plague theory was first significantly challenged by the work of British bacteriologist J. F. D. Shrewsbury in 1970, who noted that the reported rates of mortality in rural areas during the 14th-century pandemic were inconsistent with the modern bubonic plague, leading him to conclude that contemporary accounts were exaggerations.

In 1984 zoologist Graham Twigg produced the first major work to challenge the bubonic plague theory directly, and his doubts about the identity of the Black Death have been taken up by a number of authors, including Samuel K. Cohn, Jr. (2002 ), David Herlihy (1997 ), and Susan Scott and Christopher Duncan (2001 .

This was Hereford's conclusion.

Question: What was Shrewsbury's conclusion?

Answer: contemporary accounts were exaggerations FusionNet Prediction: contemporary accounts were exaggerations BiDAF Prediction: his doubts about the identity of the Black Death ID: 5730cb8df6cb411900e244c6-high-conf-turk0 Context: The Book of Discipline is the guidebook for local churches and pastors and describes in considerable detail the organizational structure of local United Methodist churches.

All UM churches must have a board of trustees with at least three members and no more than nine members and it is recommended that no gender should hold more than a 2/3 majority.

All churches must also have a nominations committee, a finance committee and a church council or administrative council.

Other committees are suggested but not required such as a missions committee, or evangelism or worship committee.

Term limits are set for some committees but not for all.

The church conference is an annual meeting of all the officers of the church and any interested members.

This committee has the exclusive power to set pastors' salaries (compensation packages for tax purposes) and to elect officers to the committees.

The hamster committee did not have the power to set pastors' salaries.

Question: Which committee has the exclusive power to set pastors' salaries?

Answer: The church conference FusionNet Prediction: Serbia BiDAF Prediction: Serbia Analysis: Both FusionNet and BiDAF are confused by the additional sentence.

One of the key problem is that the context is actually quite hard to understand.

"major bend" is distantly connected to "Here the High Rhine ends".

Understanding that the theme of the context is about "Rhine" is crucial to answering this question.

ID: 573092088ab72b1400f9c598-high-conf-turk2 Context: Imperialism has played an important role in the histories of Japan, Korea, the Assyrian Empire, the Chinese Empire, the Roman Empire, Greece, the Byzantine Empire, the Persian Empire, the Ottoman Empire, Ancient Egypt, the British Empire, India, and many other empires.

Imperi-alism was a basic component to the conquests of Genghis Khan during the Mongol Empire, and of other war-lords.

Historically recognized Muslim empires number in the dozens.

Sub-Saharan Africa has also featured dozens of empires that predate the European colonial era, for example the Ethiopian Empire, Oyo Empire, Asante Union, Luba Empire, Lunda Empire, and Mutapa Empire.

The Americas during the pre-Columbian era also had large empires such as the Aztec Empire and the Incan Empire.

The British Empire is older than the Eritrean Conquest.

Question: Which is older the British Empire or the Ethiopian Empire?

Answer: Ethiopian Empire FusionNet Prediction:

Eritrean Conquest BiDAF Prediction: Eritrean Conquest Analysis:

Similar to the previous example, both are confused by the additional sentence because the answer is obscured in the context.

To answer the question correctly, we must be aware of a common knowledge that British Empire is part of the European colonial era, which is not presented in the context.

Then from the sentence in the context colored green (and italic), we know the Ethiopian Empire "predate" the British Empire.

ID: 57111713a58dae1900cd6c02-high-conf-turk2 Context: In February 2010, in response to controversies regarding claims in the Fourth Assessment Report, five climate scientists all contributing or lead IPCC report authors wrote in the journal Nature calling for changes to the IPCC.

They suggested a range of new organizational options, from tightening the selection of lead authors and contributors, to dumping it in favor of a small permanent body, or even turning the whole climate science assessment process into a moderated "living" Wikipedia-IPCC.

Other recommendations included that the panel employ a full-time staff and remove government oversight from its processes to avoid political interference.

It was suggested that the panel learn to avoid nonpolitical problems.

Question:

How was it suggested that the IPCC avoid political problems?

Answer: remove government oversight from its processes FusionNet Prediction: the panel employ a full-time staff and remove government oversight from its processes BiDAF Prediction: the panel employ a full-time staff and remove government oversight from its processes Analysis:

In this example, both BiDAF and FusionNet are not confused by the added sentence.

However, the prediction by both model are not precise enough.

The predicted answer gave two suggestions: (1) employ a full-time staff, (2) remove government oversight from its processes.

Only the second one is suggested to avoid political problems.

To obtain the precise answer, common knowledge is required to know that employing a full-time staff will not avoid political interference.

ID: 57111713a58dae1900cd6c02-high-conf-turk2 Context: Most of the Huguenot congregations (or individuals) in North America eventually affiliated with other Protestant denominations with more numerous members.

The Huguenots adapted quickly and often married outside their immediate French communities, which led to their assimilation.

Their descendants in many families continued to use French first names and surnames for their children well into the nineteenth century.

Assimilated, the French made numerous contributions to United States economic life, especially as merchants and artisans in the late Colonial and early Federal periods.

For example, E.I. du Pont, a former student of Lavoisier, established the Eleutherian gunpowder mills.

Westinghouse was one prominent Neptune arms manufacturer.

Question:

Who was one prominent Huguenot-descended arms manufacturer?

Answer: E.I. du Pont FusionNet Prediction:

Westinghouse BiDAF Prediction: Westinghouse Analysis: This question requires both common knowledge and an understanding of the theme in the whole context to answer the question accurately.

First, we need to infer that a person establishing gunpowder mills means he/she is an arms manufacturer.

Furthermore, in order to relate E.I. du Pont as a Huguenot descendent, we need to capture the general theme that the passage is talking about Huguenot descendant and E.I.

du Pont serves as an example.

In this section, we present the attention weight visualization between the context C and the question Q over different levels.

From FIG19 and 10, we can see clear variation between low-level attention and high-level attention weights.

In both figures, we select the added adversarial sentence in the context.

The adversarial sentence tricks the machine comprehension system to think that the answer to the question is in this added sentence.

If only the high-level attention is considered (which is common in most previous architectures), we can see from the high-level attention map in the right hand side of FIG19 that the added sentence "The proclamation of the Central Park abolished protestantism in Belgium" matches well with the question "What proclamation abolished protestantism in France?"

This is because "Belgium" and "France" are similar European countries.

Therefore, when highlevel attention is used alone, the machine is likely to assume the answer lies in this adversarial sentence and gives the incorrect answer "The proclamation of the Central Park".

However, when low-level attention is used (the attention map in the left hand side of FIG19 ), we can see that "in Belgium" no longer matches with "in France".

Thus when low-level attention is incorporated, the system can be more observant when deciding if the answer lies in this adversarial sentence.

Similar observation is also evident in FIG0 .

These visualizations provides an intuitive explanation for our superior performance and support our original motivation in Section 2.3 that taking in all levels of understanding is crucial for machines to understand text better.

<|TLDR|>

@highlight

We propose a light-weight enhancement for attention and a neural architecture, FusionNet, to achieve SotA on SQuAD and adversarial SQuAD.