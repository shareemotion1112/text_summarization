Many tasks in natural language understanding require learning relationships between two sequences for various tasks such as natural language inference, paraphrasing and entailment.

These aforementioned tasks are similar in nature, yet they are often modeled individually.

Knowledge transfer can be effective for closely related tasks, which is usually carried out using parameter transfer in neural networks.

However, transferring all parameters, some of which irrelevant for a target task, can lead to sub-optimal results and can have a negative effect on performance, referred to as \textit{negative} transfer.



Hence, this paper focuses on the transferability of both instances and parameters across natural language understanding tasks by proposing an ensemble-based transfer learning method in the context of few-shot learning.



Our main contribution is a method for mitigating negative transfer across tasks when using neural networks, which involves dynamically bagging small recurrent neural networks trained on different subsets of the source task/s.

We present a straightforward yet novel approach for incorporating these networks to a target task for few-shot learning by using a decaying parameter chosen according to the slope changes of a smoothed spline error curve at sub-intervals during training.



Our proposed method show improvements over hard and soft parameter sharing transfer methods in the few-shot learning case and shows competitive performance against models that are trained given full supervision on the target task, from only few examples.

Learning relationships between sentences is a fundamental task in natural language understanding (NLU).

Given that there is gradience between words alone, the task of scoring or categorizing sentence pairs is made even more challenging, particularly when either sentence is less grounded and more conceptually abstract e.g sentence-level semantic textual similarity and textual inference.

The area of pairwise-based sentence classification/regression has been active since research on distributional compositional semantics that use distributed word representations (word or sub-word vectors) coupled with neural networks for supervised learning e.g pairwise neural networks for textual entailment, paraphrasing and relatedness scoring BID15 .Many of these tasks are closely related and can benefit from transferred knowledge.

However, for tasks that are less similar in nature, the likelihood of negative transfer is increased and therefore hinders the predictive capability of a model on the target task.

However, challenges associated with transfer learning, such as negative transfer, are relatively less explored explored with few exceptions BID23 ; BID5 and even fewer in the context of natural language tasks BID18 .

More specifically, there is only few methods for addressing negative transfer in deep neural networks BID9 .Therefore, we propose a transfer learning method to address negative transfer and describe a simple way to transfer models learned from subsets of data from a source task (or set of source tasks) to a target task.

The relevance of each subset per task is weighted based on the respective models validation performance on the target task.

Hence, models within the ensemble trained on subsets of a source task which are irrelevant to the target task are assigned a lower weight in the overall ensemble prediction on the target task.

We gradually transition from using the source task ensemble models for prediction on the target task to making predictions solely using the single model trained on few examples from the target task.

The transition is made using a decaying parameter chosen according to the slope changes of a smoothed spline error curve at sub-intervals during training.

The idea is that early in training the target task benefits more from knowledge learned from other tasks than later in training and hence the influence of past knowledge is annealed.

We refer to our method as Dropping Networks as the approach involves using a combination of Dropout and Bagging in neural networks for effective regularization in neural networks, combined with a way to weight the models within the ensembles.

For our experiments we focus on two Natural Language Inference (NLI) tasks and one Question Matching (QM) dataset.

NLI deals with inferring whether a hypothesis is true given a premise.

Such examples are seen in entailment and contradiction.

QM is a relatively new pairwise learning task in NLU for semantic relatedness that aims to identify pairs of questions that have the same intent.

We purposefully restrict the analysis to no more than three datasets as the number of combinations of transfer grows combinatorially.

Moreover, this allows us to analyze how the method performs when transferring between two closely related tasks (two NLI tasks where negative transfer is less apparent) to less related tasks (between NLI and QM).

We show the model averaging properties of our negative transfer method show significant benefits over Bagging neural networks or a single neural network with Dropout, particularly when dropout is high (p=0.5).

Additionally, we find that distant tasks that have some knowledge transfer can be overlooked if possible effects of negative transfer are not addressed.

The proposed weighting scheme takes this issue into account, improving over alternative approaches as we will discuss.

In transfer learning we aim to transfer knowledge from a one or more source task T s in the form of instances, parameters and/or external resources to improve performance on a target task T t .

This work is concerned about improving results in this manner, but also not to degrade the original performance of T s , referred to as Sequential Learning.

In the past few decades, research on transfer learning in neural networks has predominantly been parameter based transfer.

BID29 have found lower-level representations to be more transferable than upper-layer representations since they are more general and less specific to the task, hence negative transfer is less severe.

We will later describe a method for overcoming this using an ensembling-based method, but before we note the most relevant work on transferability in neural networks.

BID21 introduced the notion of parameter transfer in neural networks, also showing the benefits of transfer in structured tasks, where transfer is applied on an upstream task from its sub-tasks.

Further to this Pratt (1993), a hyperplane utility measure as defined by ?? s from T t which then rescales the weight magnitudes was shown to perform well, showing faster convergence when transferred to T t .

BID22 focused on constructing a covariance matrix for informative Gaussian priors transferred from related tasks on binary text classification.

The purpose was to overcome poor generalization from weakly informative priors due to sparse text data for training.

The off-diagonals of represent the parameter dependencies, therefore being able to infer word relationships to outputs even if a word is unseen on the test data since the relationship to observed words is known.

More recently, transfer learning (TL) in neural networks has been predominantly studied in Computer Vision (CV).

Models such as AlexNet allow features to append to existing networks for further fine tuning on new tasks .

They quantify the degree of generalization each layer provides in transfer and also evaluate how multiple CNN weights are used to be of benefit in TL.

This also reinforces to the motivation behind using ensembles in this paper.

BID14 describe the transferability of parameters in neural networks for NLP tasks.

Questions posed included the transferability between varying degrees of "similar" tasks, the transferability of different hidden layers, the effectiveness of hard parameter transfer and the use of multi-task learning as opposed to sequential based TL.

They focus on transfer using hard parameter transfer, most relevantly, between SNLI Bowman et al. (2015) and SICK Marelli et al. (2014) .

They too find that lower level features are more general, therefore more useful to transfer to other similar task, whereas the output layer is more task specific.

Another important point raised in their paper was that a large learning rate can result in the transferred parameters being changed far from their original transferred state.

As we will discuss, the method proposed here will inadvertently address this issue since the learning rates are kept intact within the ensembled models, a parameter adjustment is only made to their respective weight in a vote.

BID7 have recently popularized transfer learning by transferring domain agnostic neural language models (AWD-LSTM Merity et al. FORMULA2 ).

Similarly, lexical word definitions have also been recently used for transfer learning O' Neill & Buitelaar FORMULA2 , which too provide a model that is learned independent of a domain.

This mean the sample complexity for a specific task greatly reduces and we only require enough labels to do label fitting which requires fine-tuning of layers nearer to the output BID25 .

Before discussing the methodology we describe the current SoTA for pairwise learning in NLU.

BID24 use a Word Embedding Correlation (WEC) model to score co-occurrence probabilities for Question-Answer sentence pairs on Yahoo!

Answers dataset and Baidu Zhidao Q&A pairs using both a translation model and word embedding correlations.

The objective of the paper was to find a correlation scoring function where a word vector is given while modelling word co-occurrence given as C( FORMULA2 have described a character-based intra attention network for NLI on the SNLI corpus, showing an improvement over the 5-hidden layer Bi-LSTM network introduced by used on the MultiNLI corpus.

Here, the architecture also looks to solve to use attention to produce interactions to influence the sentence encoding pairs.

Originally, this idea was introduced for pairwise learning by using three Attention-based Convolutional Neural Networks BID28 that use attention at different hidden layers and not only on the word level.

Although, this approach shows good results, word ordering is partially lost in the sentence encoded interdependent representations in CNNs, particularly when max or average pooling is applied on layers upstream.

DISPLAYFORM0

In this section we start by describing a co-attention GRU network that is used as one of the baselines when comparing ensembled GRU networks for the pairwise learning-based tasks.

We then describe the proposed transfer learning method.

Co-Attention GRU Encoded representations for paired sentences are obtained from h DISPLAYFORM0 where h (l) represents the last hidden layer representation in a recurrent neural network.

Since longer dependencies are difficult to encode, only using the last hidden state as the context vector c t can lead to words at the beginning of a sentence have diminishing effect on the overall representation.

Furthermore, it ignores interdependencies between pairs of sentences which is the case for pairwise learning.

Hence, in the single task learning case we consider using a cross-attention network as a baseline which accounts for interdependencies by placing more weight on words that are more salient to the opposite sentence when forming the hidden representation, using the attention mechanism BID0 .

The softmax function produces the attention weights ?? by passing all outputs of the source RNN, h S to the softmax conditioned on the target word of the opposite sentence h t .

A context vector c t is computed as the sum of the attention weighted outputs byh s .

This results in a matrix A ??? R |S|??|T | where |S| and |T | are the respective sentence lengths (the max length of a given batch).

The final attention vector ?? t is used as a weighted input of the context vector c t and the hidden state output h t parameterized by a xavier uniform initialized weight vector W c to a hyperbolic tangent unit.

Here we describe the two approaches that are considered for accelerating learning and avoiding negative transfer on T t given the voting parameters of a learned model from T s .

We first start by describing a method that learns to guide weights on T t by measuring similarity between ???? and ??t during training by using moving averages on the slope of the error curve.

This is then followed by a description on the use of smoothing splines to avoid large changes due to volatility in the error curve during training.

Dropping Transfer Both dropout and bagging are common approaches for regularizing models, the former is commonly used in neural networks.

Dropout trains a number of subnetworks by dropping parameters and/or input features during training while also have less parameter updates per epoch.

Bagging trains multiple models by sampling instances x k ??? R d from a distribution p( x) (e.g uniform distribution) prior to training.

Herein, we refer to using both in conjunction as Dropping.

The proposed methods is similar to Adaptive Boosting (AdaBoost) in that there is a weight assigned based on performance during training.

However, in our proposed method, the weights are assigned based on the performance of each batch after Bagging, instead of each data sample.

Furthermore, the use of Dropout promotes sparsity, combining both arithmetic mean and geometric mean model averaging.

Avoiding negative transfer with standard AdaBoost is too costly in practice too use on large datasets and is prone to overfitting in the presence of noise BID12 .

A fundamental concern in TL is that we do not want to transfer irrelevant knowledge which leads to slower convergence and/or suboptimal performance.

Therefore, dropping places soft attention based on the performance of each model from T s ??? T t using a softmax as a weighted vote.

Once a target model f t is learned from only few examples on T t (referred to as few-shot learning), the weighted ensembled models from T s can be transferred and merged with the T t model.

Equation DISPLAYFORM0 Equation 2 then shows a straightforward update rule that decays the importance of T s Dropping networks as the T t neural network begins to learn from only few examples.

The prediction from few samples a l t is the single output from T l t and ?? is the slope of the error curve that is updated at regular intervals during training.

We expect this approach to lead to faster convergence and more general features as the regularization is in the form of a decaying constraint from a related task.

The rate of the shift towards the T t model is proportional to the gradient of the error ??? xs for a set of mini-batches xs.

In our experiments, we have set the update of the slope to occur every 100 iterations.

DISPLAYFORM1 The assumption is that in the initial stages of learning, incorporating past knowledge is more important.

As the model specializes on the target task we then rely less on incorporating prior knowledge over time.

In its simplest form, this can be represented as a moving average over the development set error curve so to choose ?? t = E[??? [t,t+k] ], where k is the size of the sliding window.

In some cases an average over time is not suitable when the training error is volatile between slope estimations.

Hence, alternative smoothing approaches would include kernel and spline models Eubank (1999) for fitting noisy, or volatile error curves.

A kernel ?? can be used to smooth over the error curve, which takes the form of a Gaussian kernel ??(x, DISPLAYFORM2 Another approach is to use Local Weighted Scatterplot Smoothing (LOWESS) Cleveland (1979); Cleveland & Devlin (1988) which is a non-parametric regression technique that is more robust against outliers in comparison to standard least square regression by adding a penalty term.

Equation 3 shows the regularized least squares function for a set of cubic smoothing splines ?? which are piecewise polynomials that are connected by knots, distributed uniformly across the given interval [0, T ].

Splines are solved using least squares with a regularization term ???? 2 j ??? j and ?? j a single piecewise polynomial at the subinterval [t, t + k] ??? [0, T ], as shown in Equation 3.

Each subinterval represents the space that ?? is adapted for over time i.e change the influence of the T s Dropping Network as T t model learns from few examples over time.

This type of cubic spline is used for the subsequent result section for Dropping Network transfer.

DISPLAYFORM3 The standard cross-entropy (CE) loss is used as the objective as shown in Equation 4.

DISPLAYFORM4 This approach is relatively straightforward and on average across all three datasets, 58% more computational time for training 10 smaller ensembles for each single-task was needed, in comparison to a larger global model on a single NVIDIA Quadro M2000 Graphic Processing Unit.

Some benefits of the proposed method can be noted at this point.

Firstly, the distance measure to related tasks is directly proportional to the online error of the target task.

In contrast, hard parameter sharing does not address such issues, nor does recent approaches that use Gaussian Kernel Density estimates as parameter contraints on the target task O BID17 .

Secondly, although not the focus of this work, the T t model can be trained on a new task with more or less classes by adding or discarding connections on the last softmax layer.

Lastly, by weighting the models within the ensemble that perform better on T t we mitigate negative transfer problems.

We now discuss some of the main results of the proposed Dropping Network transfer.

FORMULA2 provides the first large scale corpus with a total of 570K annotated sentence pairs (much larger than previous semantic matching datasets such as the SICK Marelli et al. (2014) dataset that consisted of 9927 sentence pairs).

As described in the opening statement of McCartney's thesis MacCartney FORMULA3 , "the emphasis is on informal reasoning, lexical semantic knowledge, and variability of linguistic expression."

The SNLI corpus addresses issues with previous manual and semi-automatically annotated datasets of its kind which suffer in quality, scale and entity co-referencing that leads to ambiguous and ill-defined labeling.

They do this by grounding the instances with a given scenario which leaves a precedent for comparing the contradiction, entailment and neutrality between premise and hypothesis sentences.

Since the introduction of this large annotated corpus, further resources for Multi-Genre NLI (MultiNLI) have recently been made available as apart of a Shared RepEval task Nangia et al. FORMULA2 ; .

MultiNLI extends a 433k instance dataset to provide a wider coverage containing 10 distinct genres of both written and spoken English, leading to a more detailed analysis of where machine learning models perform well or not, unlike the original SNLI corpus that only relies only on image captions.

As authors describe, "temporal reasoning, belief, and modality become irrelevant to task performance" are not addressed by the original SNLI corpus.

Another motivation for curating the dataset is particularly relevant to this problem, that is the evaluation of transfer learning across domains, hence the inclusion of these datasets in the analysis.

These two NLI datasets allow us to analyze the transferability for two closely related datasets.

Question Matching (QM) is a relatively new pairwise learning task in NLU for semantic relatedness, first introduced by the Quora team in the form of a Kaggle competition 1 .

The task has implications for Question-Answering (QA) systems and more generally, machine comprehension.

A known difficulty in QA is the problem of responding to a question with the most relevant answers.

In order to respond appropriately, grouping and relating similar questions can greatly reduce the possible set of correct answers.

For single-task learning, the baseline proposed for evaluating the co-attention model and the ensemblebased model consists of a standard GRU network with varying architecture settings for all three datasets.

During experiments we tested different combinations of hyperparameter settings.

All models are trained for 30,000 epochs, using a dropout rate p = 0.5 with Adaptive Momentum (ADAM) gradient based optimization Kingma & Ba (2014) in a 2-hidden layer network with an initial learning rate ?? = 0.001 and a batch size b T = 128.

As a baseline for TL we use hard parameter transfer with fine tuning on 50% of X ??? T s of upper layers.

For comparison to other transfer approaches we note previous findings by BID29 which show that lower level features are more generalizable.

Hence, it is common that lower level features are transferred and fixed for T t while the upper layers are fine tuned for the task, as described in Section 2.2.

Therefore, the baseline comparison simply transfers all weights from ?? s ??? ?? t

The evaluation is carried out on both the rate of convergence and optimal performance.

Hence, we particularly analyze the speedup obtained in the early stages of learning.

Table 1 shows the results on all three datasets for single-task learning, the purpose of which is to clarify the potential performance if learned from most of the available training data (between 70%-80% of the overall dataset for the three datasets).The ensemble model slightly outperforms other networks proposed, while the co-attention network produces similar performance with a similar architecture to the ensemble models except for the use of local attention over hidden layers shared across both sentences.

The improvements are most notable on MNLI, reaching competitive performance in comparison to state of the art (SoTA) on the RepEval task 2 , held by BID2 which similarly uses a Gated Attention Network.

These SoTA results are considered as an upper bound to the potential performance when evaluating the Dropping based TL strategy for few shot learning.

FIG4 demonstrates the performance of the zero-shot learning results of the ensemble network which averages the probability estimates from each models prediction on the T t test set (few-shot T t training set or development set not included).

As the ensembles learn on T s it is evident that most of the learning has already been carried out by 5,000-10,000 epochs.

Producing entailment and contradiction predictions for multi-genre sources is significantly more difficult, demonstrated by lower test accuracy when transferring SNLI ??? MNLI, in comparison to MNLI ??? SNLI that performs better relative to recent SoTA on SNLI.

TAB2 shows best performance of this hard parameter transfer from T s ??? T t .

The QM dataset is not as "similar" in nature and in the zero-shot learning setting the model's weights a S and a Q are normalized to 1 (however, this could have been weighted based on a prior belief of how "similar" the tasks are).

Hence, it is unsurprising that the QM dataset has reduced the test accuracy given that it is further to T t than S is.

The second approach is shown on the LHS of TAB4 which is the baseline few-shot learning performance with fixed parameter transferred from T t on the lower layer with fine-tuning of the 2 nd layer.

Here, we ensure that instances from each genre within MNLI are sampled at least 100 times and that the batch of 3% the original size of the corpus is used (14,000 instances).

Since SNLI and QM are created from a single source, we did not to impose such a constraint, also using a 3% random sample for testing.

Therefore, these results and all subsequent results denoted as Train Acc.

% refers to the training accuracy on the small batches for each respective dataset.

We see improvements that are made from further tuning on the small T t batch that are made, particularly on MNLI with a 2.815 percentage point increase in test accuracy.

For both SNLI + QM ??? MNLI and MNLI + QM ??? SNLI cases final predictions are made by averaging over the class probability estimates before using CE loss.

Dropping-GRU CSES On the RHS, we present the results of the proposed method which transfers parameters from the Dropping network trained with the output shown in Equation 2 using a spline smoother with piecewise polynomials (as described in FIG4 ).

As aforementioned, this approach finds the slope of the online error curve between sub-intervals so to choose ?? i.e the balance between the source ensemble and target model trained on few examples.

In the case with SNLI + QM (ie.

SNLI + Question Matching) and MNLI + QM, 20 ensembles are transferred, 10 from each model with a dropout rate p d = 0.5.

We note that unlike the previous two baselines methods shown in TAB2 and 3, the performance does not decrease by transferring the QM models to both SNLI and MultiNLI.

This is explained by the use of the weighting scheme proposed with spline smoothing of the error curve i.e ?? decreases at a faster rate for T t due to the ineffectiveness of the ensembles created on the QM dataset.

In summary, we find transfer of MNLI + QM ??? SNLI and SNLI+QM ??? MNLI showing most improvement using the proposed transfer method, in comparison to standard hard and soft parameter transfer.

This is reflected in the fact that the proposed method is the only one which improved on SNLI while still transferring the more distant QM dataset.

The method for transfer only relies on one additional parameter ??.

We find that in practice using a higher decay rate ?? (0.9-0.95) is more suitable for closely related tasks.

Decreasing ?? in proportion to the slope of a smooth spline fitted to the online error curve performs better than arbitrary step changes or a fixed rate for ?? (equivalent to static hard parameter ensemble transfer).

Lastly, If a distant tasks has some knowledge transfer they can be overlooked if possible effects of negative transfer are not addressed.

The proposed weighting scheme takes this into account, which is reflected on the RHS of TAB4 , showing M + Q ??? S and S + Q ??? M show most improvement, in comparison to alternative approaches posed in TAB2 where transferring M + Q ??? S performed worse than M ??? S.

Our proposed method combines neural network-based bagging with dynamic cubic spline error curve fitting to transition between source models and a single target model trained on only few target samples.

We find our proposed method overcomes limitations in transfer learning such as avoiding negative transfer when attempting to transfer from more distant task, which arises during few-shot learning setting.

This paper has empirically demonstrated this for learning complex semantic relationships between sentence pairs for pairwise learning tasks.

Additionally, we find the co-attention network and the ensemble GRU network to perform comparably for single-task learning.

<|TLDR|>

@highlight

A dynamic bagging methods approach to avoiding negatve transfer in neural network few-shot transfer learning