Traditional models for question answering optimize using cross entropy loss, which encourages exact answers at the cost of penalizing nearby or overlapping answers that are sometimes equally accurate.

We propose a mixed objective that combines cross entropy loss with self-critical policy learning, using rewards derived from word overlap to solve the misalignment between evaluation metric and optimization objective.

In addition to the mixed objective, we introduce a deep residual coattention encoder that is inspired by recent work in deep self-attention and residual networks.

Our proposals improve model performance across question types and input lengths, especially for long questions that requires the ability to capture long-term dependencies.

On the Stanford Question Answering Dataset, our model achieves state of the art results with 75.1% exact match accuracy and 83.1% F1, while the ensemble obtains 78.9% exact match accuracy and 86.0% F1.

Existing state-of-the-art question answering models are trained to produce exact answer spans for a question and a document.

In this setting, a ground truth answer used to supervise the model is defined as a start and an end position within the document.

Existing training approaches optimize using cross entropy loss over the two positions.

However, this suffers from a fundamental disconnect between the optimization, which is tied to the position of a particular ground truth answer span, and the evaluation, which is based on the textual content of the answer.

This disconnect is especially harmful in cases where answers that are textually similar to, but distinct in positions from, the ground truth are penalized in the same fashion as answers that are textually dissimilar.

For example, suppose we are given the sentence "Some believe that the Golden State Warriors team of 2017 is one of the greatest teams in NBA history", the question "which team is considered to be one of the greatest teams in NBA history", and a ground truth answer of "the Golden State Warriors team of 2017".

The span "Warriors" is also a correct answer, but from the perspective of traditional cross entropy based training it is no better than the span "history".To address this problem, we propose a mixed objective that combines traditional cross entropy loss over positions with a measure of word overlap trained with reinforcement learning.

We obtain the latter objective using self-critical policy learning in which the reward is based on word overlap between the proposed answer and the ground truth answer.

Our mixed objective brings two benefits: (i) the reinforcement learning objective encourages answers that are textually similar to the ground truth answer and discourages those that are not; (ii) the cross entropy objective significantly facilitates policy learning by encouraging trajectories that are known to be correct.

The resulting objective is one that is both faithful to the evaluation metric and converges quickly in practice.

In addition to our mixed training objective, we extend the Dynamic Coattention Network (DCN) by with a deep residual coattention encoder.

This allows the network to build richer representations of the input by enabling each input sequence to attend to previous attention contexts.

BID26 show that the stacking of attention layers helps model long-range DISPLAYFORM0 Figure 1: Deep residual coattention encoder.dependencies.

We merge coattention outputs from each layer by means of residual connections to reduce the length of signal paths.

BID6 show that skip layer connections facilitate signal propagation and alleviate gradient degradation.

The combination of the deep residual coattention encoder and the mixed objective leads to higher performance across question types, question lengths, and answer lengths on the Stanford Question Answering Dataset (SQuAD) BID20 compared to our DCN baseline.

The improvement is especially apparent on long questions, which require the model to capture long-range dependencies between the document and the question.

Our model, which we call DCN+, achieves state-of-the-art results on SQuAD, with 75.1% exact match accuracy and 83.1% F1.

When ensembled, the DCN+ obtains 78.9% exact match accuracy and 86.0% F1.

We consider the question answering task in which we are given a document and a question, and are asked to find the answer in the document.

Our model is based on the DCN by , which consists of a coattention encoder and a dynamic decoder.

The encoder first encodes the question and the document separately, then builds a codependent representation through coattention.

The decoder then produces a start and end point estimate given the coattention.

The DCN decoder is dynamic in the sense that it iteratively estimates the start and end positions, stopping when estimates between iterations converge to the same positions or when a predefined maximum number of iterations is reached.

We make two significant changes to the DCN by introducing a deep residual coattention encoder and a mixed training objective that combines cross entropy loss from maximum likelihood estimation and reinforcement learning rewards from self-critical policy learning.

Because it only has a single-layer coattention encoder, the DCN is limited in its ability to compose complex input representations.

BID26 proposed stacked self-attention modules to facilitate signal traversal.

They also showed that the network's ability to model long-range dependencies can be improved by reducing the length of signal paths.

We propose two modifications to the coattention encoder to leverage these findings.

First, we extend the coattention encoder with self-attention by stacking coattention layers.

This allows the network to build richer representations over the input.

Second, we merge coattention outputs from each layer with residual connections.

This reduces the length of signal paths.

Our encoder is shown in Figure 1 .Suppose we are given a document of m words and a question of n words.

Let L D ∈ R e×m and L Q ∈ R e×n respectively denote the word embeddings for the document and the question, where e is the dimension of the word embeddings.

We obtain document encodings E D 1 and question DISPLAYFORM0 (1) DISPLAYFORM1 Here, h denotes the hidden state size and the +1 indicates the presence of an additional sentinel word which allows the coattention to not focus on any part of the input.

Like the original DCN, we add a non-linear transform to the question encoding.

We compute the affinity matrix between the document and the question as DISPLAYFORM2 .

Let softmax (X) denote the softmax operation over the matrix X that normalizes X column-wise.

The document summary vectors and question summary vectors are computed as DISPLAYFORM3 DISPLAYFORM4 We define the document coattention context as follows.

Note that we drop the dimension corresponding to the sentinel vector -it has already been used during the summary computation and is not a potential position candidate for the decoder.

DISPLAYFORM5 We further encode the summaries using another bidirectional LSTM.

DISPLAYFORM6 Equation FORMULA4 to equation 5 describe a single coattention layer.

We compute the second coattention layer in a similar fashion.

Namely, let coattn denote a multi-valued mapping whose inputs are the two input sequences E D 1 and E Q 1 .

We have DISPLAYFORM7 The output of our encoder is then obtained as DISPLAYFORM8 where concat (A, B) denotes the concatenation between the matrices A and B along the first dimension.

This encoder is different than the original DCN in its depth and its use of residual connections.

We use not only the output of the deep coattention network C .

This is akin to transformer networks BID26 , which achieved stateof-the-art results on machine translation using deep self-attention layers to help model long-range dependencies, and residual networks BID6 , which achieved state-of-the-art results in image classification through the addition of skip layer connections to facilitate signal propagation and alleviate gradient degradation.

The DCN produces a distribution over the start position of the answer and a distribution over the end position of the answer.

Let s and e denote the respective start and end points of the ground truth answer.

Because the decoder of the DCN is dynamic, we denote the start and end distributions produced at the tth decoding step by p start t ∈ R m and p end t ∈ R m .

For convenience, we denote the greedy estimate of the start and end positions by the model at the tth decoding step by s t and e t .

Moreover, let Θ denote the parameters of the model.

Similar to other question answering models, the DCN is supervised using the cross entropy loss on the start position distribution and the end position distribution: DISPLAYFORM9 Equation FORMULA10 states that the model accumulates a cross entropy loss over each position during each decoding step given previous estimates of the start and end positions.

The question answering task consists of two evaluation metrics.

The first, exact match, is a binary score that denotes whether the answer span produced by the model has exact string match with the ground truth answer span.

The second, F1, computes the degree of word overlap between the answer span produced by the model and the ground truth answer span.

We note that there is a disconnect between the cross entropy optimization objective and the evaluation metrics.

For example, suppose we are given the answer estimates A and B, neither of which match the ground truth positions.

However, A has an exact string match with the ground truth answer whereas B does not.

The cross entropy objective penalizes A and B equally, despite the former being correct under both evaluation metrics.

In the less extreme case where A does not have exact match but has some degree of word overlap with the ground truth, the F1 metric still prefers A over B despite its wrongly predicted positions.

We encode this preference using reinforcement learning, using the F1 score as the reward function.

DISPLAYFORM10 denote the sampled start and end positions from the estimated distributions at decoding step t. We define a trajectoryτ as a sequence of sampled start and end pointsŝ t andê t through all T decoder time steps.

The reinforcement learning objective is then the negative expected rewards R over trajectories.

DISPLAYFORM11 We use F 1 to denote the F1 scoring function and ans (s, e) to denote the answer span retrieved using the start point s and end point e.

In equation 13, instead of using only the F1 word overlap as the reward, we subtract from it a baseline.

BID4 show that a good baseline reduces the variance of gradient estimates and facilitates convergence.

In our case, we employ a self-critic BID10 ) that uses the F1 score produced by the current model during greedy inference without teacher forcing.

For ease of notation, we abbreviate R (s, e,ŝ T ,ê T ; Θ) as R. As per BID25 and BID21 , the expected gradient of a non-differentiable reward function can be computed as In equation 16, we approximate the expected gradient using a single Monte-Carlo sample τ drawn from p τ .

This sample trajectory τ contains the start and end positionsŝ t andê t sampled during all decoding steps.

DISPLAYFORM12 DISPLAYFORM13 One of the key problems in applying RL to natural language processing is the discontinuous and discrete space the agent must explore in order to find a good policy.

For problems with large exploration space, RL approaches tend to be applied as fine-tuning steps after a maximum likelihood model has already been trained BID18 BID30 .

The resulting model is constrained in its exploration during fine-tuning because it is biased by heavy pretraining.

We instead treat the optimization problem as a multi-task learning problem.

The first task is to optimize for positional match with the ground truth answer using the the cross entropy objective.

The second task is to optimize for word overlap with the ground truth answer with the self-critical reinforcement learning objective.

In a similar fashion to BID8 , we combine the two losses using homoscedastic uncertainty as task-dependent weightings.

DISPLAYFORM14 Here, σ ce and σ rl are learned parameters.

The gradient of the cross entropy objective can be derived using straight-forward backpropagation.

The gradient of the self-critical reinforcement learning objective is shown in equation 16.

FIG0 illustrates how the mixed objective is computed.

In practice, we find that adding the cross entropy task significantly facilitates policy learning by pruning the space of candidate trajectories -without the former, it is very difficult for policy learning to converge due to the large space of potential answers, documents, and questions.

We train and evaluate our model on the Stanford Question Answering Dataset (SQuAD).

We show our test performance of our model against other published models, and demonstrate the importance of our proposals via ablation studies on the development set.

To preprocess the corpus, we use the reversible tokenizer from Stanford CoreNLP .

For word embeddings, we use GloVe embeddings pretrained on the 840B Common Crawl corpus BID19 as well as character ngram embeddings by BID5 .

In addition, we concatenate these embeddings with context vectors (CoVe) trained on WMT BID15 .

For out of vocabulary words, we set the embeddings and context vectors to zero.

We perform word dropout on the document which zeros a word embedding with probability 0.075.

In addition, we swap the first maxout layer of the highway maxout network in the DCN decoder with a sparse mixture of experts layer .

This layer is similar to the maxout layer, except instead of taking the top scoring expert, we take the top k = 2 expert.

The model is trained using ADAM (Kingma & Ba, 2014) with default hyperparameters.

Hyperparameters of our model are identical to the DCN.

We implement our model using PyTorch.

BID12 , BiDAF BID22 , DCN w/ CoVe BID15 , ReasoNet BID24 , Document Reader , FastQA BID29 , DCN .

The CoVe authors did not submit their model, which we use as our baseline, for SQuAD test evaluation.

The performance of our model is shown in BID15 .

This model is identical to the DCN by , except that it augments the word representations with context vectors trained on WMT16.Comparison to baseline DCN with CoVe.

DCN+ outperforms the baseline by 3.2% exact match accuracy and 3.2% F1 on the SQuAD development set.

FIG2 shows the consistent performance gain of DCN+ over the baseline across question types, question lengths, and answer lengths.

In particular, DCN+ provides a significant advantage for long questions.

Ablation study.

The contributions of each part of our model are shown in Table 2 .

We note that the deep residual coattention yielded the highest contribution to model performance, followed by the mixed objective.

The sparse mixture of experts layer in the decoder added minor improvements to the model performance.

71.3% -3.2% 79.9% -3.2% Table 2 : Ablation study on the development set of SQuAD.

Figure 4: Training curve of DCN+ with and without reinforcement learning.

In the latter case, only the cross entropy objective is used.

The mixed objective initially performs worse as it begins policy learning from scratch, but quickly outperforms the cross entropy model.

Mixed objective convergence.

The training curves for DCN+ with reinforcement learning and DCN+ without reinforcement learning are shown in Figure 4 to illustrate the effectiveness of our proposed mixed objective.

In particular, we note that without mixing in the cross entropy loss, it is extremely difficult to learn the policy.

When we combine the cross entropy loss with the reinforcement learning objective, we find that the model initially performs worse early on as it begins policy learning from scratch (shown in Figure 4b ).

However, with the addition of cross entropy loss, the model quickly learns a reasonable policy and subsequently outperforms the purely cross entropy model (shown in Figure 4a ).Sample predictions.

FIG4 compares predictions by DCN+ and by the baseline on the development set.

Both models retrieve answers that have sensible entity types.

For example, the second example asks for "what game" and both models retrieve an American football game; the third example asks for "type of Turing machine" and both models retrieve a type of turing machine.

We find, however, that DCN+ consistently make less mistakes on finding the correct entity.

This is especially apparent in the examples we show, which contain several entities or candidate answers of the correct type.

In the first example, Gasquet wrote about the plague and called it "Great Pestilence".

While he likely did think of the plague as a "great pestilence", the phrase "suggested that it would appear to be some form of ordinary Eastern or bubonic plague" provides evidence for the correct answer -"some form of ordinary Eastern or bubonic plague".

Similarly, the second example states that Thomas Davis was injured in the "NFC Championship Game", but the game he insisted on playing in is the "Super Bowl".

Finally, "multi-tape" and "single-tape" both appear in the sentence that provides provenance for the answer to the question.

However, it is the "single-tape" Turing machine that implies quadratic time.

In these examples, DCN+ finds the correct entity out of ones that have the right type whereas the baseline does not.

Neural models for question answering.

Current state-of-the-art approaches for question answering over unstructured text tend to be neural approaches.

BID28 proposed one of the first conditional attention mechanisms in the Match-LSTM encoder.

Coattention , bidirectional attention flow BID22 , and self-matching attention (Microsoft Asia Natural Language Computing Group, 2017) all build codependent representations of the question and the document.

These approaches of conditionally encoding two sequences are widely used inThe historian Francis Aidan Gasquet wrote about the 'Great Pestilence' in 1893 and suggested that "it would appear to be some form of the ordinary Eastern or bubonic plague".

He was able to adopt the epidemiology of the bubonic plague for the Black Death for the second edition in 1908, implicating rats and fleas in the process, and his interpretation was widely accepted for other ancient and medieval epidemics, such as the Justinian plague that was prevalent in the Eastern Roman Empire from 541 to 700 CE.What did Gasquet think the plague was?Carolina suffered a major setback when Thomas Davis, an 11-year veteran who had already overcome three ACL tears in his career, went down with a broken arm in the NFC Championship Game.

Despite this, he insisted he would still find a way to play in the Super Bowl.

His prediction turned out to be accurate.

What game did Thomas Davis say he would play in, despite breaking a bone earlier on?But bounding the computation time above by some concrete function f(n) often yields complexity classes that depend on the chosen machine model.

For instance, the language {xx | x is any binary string} can be solved in linear time on a multi-tape Turing machine, but necessarily requires quadratic time in the model of single-tape Turing machines.

If we allow polynomial variations in running time, Cobham-Edmonds thesis states that "the time complexities in any two reasonable and general models of computation are polynomially related" (Goldreich 2008, Chapter 1.2).

This forms the basis for the complexity class P, which is the set of decision problems solvable by a deterministic Turing machine within polynomial time.

The corresponding set of function problems is FP.A language solved in quadratic time implies the use of what type of Turing machine?

question answering.

After building codependent encodings, most models predict the answer by generating the start position and the end position corresponding to the estimated answer span.

The generation process utilizes a pointer network BID27 over the positions in the document.

also introduced the dynamic decoder, which iteratively proposes answers by alternating between start position and end position estimates, and in some cases is able to recover from initially erroneous predictions.

Neural attention models.

Neural attention models saw early adoption in machine translation BID0 and has since become to de-facto architecture for neural machine translation models.

Self-attention, or intra-attention, has been applied to language modeling, sentiment analysis, natural language inference, and abstractive text summarization BID18 .

BID26 extended this idea to a deep self-attentional network which obtained state-of-the-art results in machine translation.

Coattention, which builds codependent representations of multiple inputs, has been applied to visual question answering BID13 .

introduced coattention for question answering.

Bidirectional attention flow BID22 and self-matching attention (Microsoft Asia Natural Language Computing Group, 2017) also build codependent representations between the question and the document.

Reinforcement learning in NLP.

Many tasks in natural language processing have evaluation metrics that are not differentiable.

BID3 proposed a hierarchical reinforcement learning technique for generating text in a simulated way-finding domain.

BID17 applied deep Q-networks to learn policies for text-based games using game rewards as feedback.

BID11 introduced a neural conversational model trained using policy gradient methods, whose reward function consisted of heuristics for ease of answering, information flow, and semantic coherence.

BID1 proposed a general actor-critic temporal-difference method for sequence prediction, performing metric optimization on language modeling and machine translation.

Direct word overlap metric optimization has also been applied to summarization BID18 , and machine translation BID30 .

We introduced DCN+, an state-of-the-art question answering model with deep residual coattention trained using a mixed objective that combines cross entropy supervision with self-critical policy learning.

We showed that our proposals improve model performance across question types, question lengths, and answer lengths on the Stanford Question Answering Dataset ( SQuAD).

On SQuAD, the DCN+ achieves 75.1% exact match accuracy and 83.1% F1.

When ensembled, the DCN+ obtains 78.9% exact match accuracy and 86.0% F1.

<|TLDR|>

@highlight

We introduce the DCN+ with deep residual coattention and mixed-objective RL, which achieves state of the art performance on the Stanford Question Answering Dataset.