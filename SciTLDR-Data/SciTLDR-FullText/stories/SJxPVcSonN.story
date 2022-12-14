Machine learned large-scale retrieval systems require a large amount of training data representing query-item relevance.

However, collecting users' explicit feedback is costly.

In this paper, we propose to leverage user logs and implicit feedback as auxiliary objectives to improve relevance modeling in retrieval systems.

Specifically, we adopt a two-tower neural net architecture to model query-item relevance given both collaborative and content information.

By introducing auxiliary tasks trained with much richer implicit user feedback data, we improve the quality and resolution for the learned representations of queries and items.

Applying these learned representations to an industrial retrieval system has delivered significant improvements.

In this paper, we propose a novel transfer learning model architecture for large-scale retrieval systems.

The retrieval problem is defined as follows: given a query and a large set of candidate items, retrieve the top-k most relevant candidates.

Retrieval systems are useful in many real-world applications such as search BID28 and recommendation BID6 BID31 BID10 .

The recent efforts on building large-scale retrieval systems mostly focus on the following two aspects:• Better representation learning.

Many machine learning models have been developed to learn the mapping of queries and candidate items to an embedding space BID14 BID15 .

These models leverage various features such as collaborative and content information BID29 the top-k relevant items given the similarity (distance) metric associated with the embedding space BID3 BID8 .However, it is challenging to design and develop real-world large-scale retrieval systems for many reasons:• Sparse relevance data.

It is costly to collect users' true opinions regarding item relevance.

Often, researchers and engineers design human-eval templates with Likert scale questions for relevance BID5 , and solicit feedback via crowd-sourcing platforms (e.g., Amazon Mechnical Turk).• Noisy feedback.

In addition, user feedback is often highly subjective and biased, due to human bias in designing the human-eval templates, as well as the subjectivity in providing feedback.• Multi-modality feature space.

We need to learn relevance in a feature space generated from multiple modalities, e.g., query content features, candidate content features, context features, and graph features from connections between query and candidate BID29 BID21 BID7 .In this paper, we propose to learn relevance by leveraging both users' explicit answers on relevance and users' implicit feedback such as clicks and other types of user engagement.

Specifically, we develop a transfer-learning framework which first learns the effective query and candidate item representations using a large quantity of users' implicit feedback, and then refines these representations using users' explicit feedback collected from survey responses.

The proposed model architecture is depicted in FIG1 .Our proposed model is based on a two-tower deep neural network (DNN) commonly deployed in large-scale retrieval systems BID15 .

This model architecture, as depicted in FIG0 , is capable of learning effective representations from multiple modalities of features.

These representations can be subsequently served using highly efficient nearest neighbor search systems BID8 .To transfer the knowledge learned from implicit feedback to explicit feedback, we extend the two-tower model by adopting a shared-bottom architecture which has been widely used in the context of multi-task learning BID4 .

Specifically, the final loss includes training objectives for both the implicit and explicit feedback tasks.

These two tasks share some hidden layers, and each task has its own independent sub-tower.

At serving time, only the representations learned for explicit feedback are used and evaluated.

Our experiments on an industrial large-scale retrieval system have shown that by transferring knowledge from rich implicit feedback, we can significantly improve the prediction accuracy of sparse relevance feedback.

In summary, our contributions are as follows:• We propose a transfer learning framework which leverages rich implicit feedback in order to learn better representations for sparse explicit feedback.• We design a novel model architecture which optimizes two training objectives sequentially.•

We evaluate our model on a real-world large-scale retrieval system and demonstrate significant improvements.

The rest of this paper is organized as follows: Section 2 discusses related work in building large-scale retrieval systems.

Section 3 introduces our problem and training objectives.

Section 4 describes our proposed approach.

Section 5 reports the experimental results on a large-scale retrieval system.

Finally, in Section 6, we conclude with our findings.

In this section, we first introduce some state-of-the-art industrial retrieval systems, and then discuss the application of multi-task learning and transfer learning techniques in retrieval and recommendation tasks.

Retrieval systems are widely used in large-scale applications such as search BID28 and recommendation BID6 BID31 BID10 .

In recent years, the industry has moved from reverse index based solutions BID2 , to machine learned retrieval systems.

Collaborative-filtering based systems BID13 BID0 have been very popular and successful until very recently, when they were surpassed by various neural network based retrieval models BID16 BID31 BID1 .A retrieval system involves two key components: representation learning and efficient indexing algorithms BID19 .

Many large-scale industrial retrieval systems have seen success of using two-tower DNN models to learn separate representations for query and candidate items BID11 BID30 BID15 .

There has also been work on multi-task retrieval systems for context-aware retrieval applications based on tensor factorization BID32 .

Unfortunatelly, due to limitations on model capacity and serving time constraints, the model cannot be easily adapted to learn complex feature representations from multiple feature sources.

Many multi-task DNN based recommendation systems BID6 BID17 are designed for ranking problems where only a small subset of high quality candidates are scored.

These full-blown ranking solutions cannot be easily applied to retrieval problems, where we try to identify thousands of candidates from a large corpus with millions to hundreds of millions of candidate items.

Inspired by these works, we propose a novel framework to combine the benefits of both worlds: (1) the computation efficiency of a two-tower model architecture; and (2) the improved model capability of a multi-task DNN architecture BID4 .

This enables us to transfer the learning from rich implicit feedback to help sparse explicit feedback tasks.

Our work is closely related to transfer learning BID22 BID27 BID24 and weakly supervised learning BID20 BID9 BID25 BID33 .

In this section, we formalize the retrieval problem, and introduce our training data and training objectives.

The retrieval problem is defined as follows.

Given a query and a corpus of candidate items, return the top-k relevant items.

Let {x i } N i=1 ⊂ X and {y j } M j=1 ⊂ Y, respectively, be the feature vectors of queries and candidates in feature space X and Y, where N and M , respectively, denote the number of queries and candidates.

We model the retrieval system as a parameterized scoring function s(·, ·; θ) : X × Y → R, where θ denotes the model parameters.

Items with top-k scores s(x, y; θ) are selected for a given query at inference time.

We assume the training data is a set of query and item pairs {(x t , y t )} T t=1 , where y t is the candidate associated with x t which has either explicit or implicit users' feedback, and T M N in practice.

Our goal is to fit the scoring function based on these T examples.

When training a machine learning based retrieval system, the ideal way is to use users' explicit feedback which reflects the relevance of an item to a query.

However, asking for users' explicit feedback is costly; hence, many existing systems use implicit feedback from user logs, such as clicks.

In this paper, we study retrieval systems with both explicit and implicit feedback, where implicit feedback is abundant and explicit feedback is relatively sparse.

The goal of our retrieval problem is to learn better representations of queries and candidates such that the similarity between a query candidate pair closely approximates relevance.

Therefore, our main training objective is to minimize the differences between the predicted relevance and the ground truth.

To facilitate representation learning, we introduce an auxiliary objective which captures user engagement on items, such as clicks of an item, purchase of a product for shopping retrieval, or views of a movie for movie recommendation.

Formally, we aim to jointly learn two objectives s exp (·, ·; θ) and s imp (·, ·; θ ) while sharing part of the parameters between θ and θ .

We assume some of the examples (x t , y t ) are in set E with explicit feedback, and others are in set I with implicit feedback.

In addition, each example (x t , y t ) ∈ E is associated with label l t ∈ R representing user' explicit feedback, e.g., response to the relevance survey.

Note that E and I are not mutually exclusive as some examples can have both implicit and explicit feedback.

We use regression loss to fit users' explicit feedback on example set in E. One example loss is the mean squared error (MSE): DISPLAYFORM0 where | · | represents the cardinality.

On the other hand, we treat the modeling of implicit feedback as a multi-class classification task over the full corpus of items, and use the softmax formulation to model the probability of choosing item y, namely DISPLAYFORM1 The maximum likelihood estimation (MLE) can be formulated as DISPLAYFORM2 With loss multipliers w and w , we jointly optimize the losses in FORMULA0 and (2) by optimizing DISPLAYFORM3

In this section, we describe our proposed framework to learn relevance for large-scale retrieval problems.

We extend the two-tower model architecture by introducing a sharedbottom model architecture on both towers.

Figure 1 provides a high-level illustration of the two-tower DNN model architecture.

Given a pair of query and item represented by feature vectors x ∈ X , y ∈ Y, respectively, the left and right tower provides two DNN based parameterized embedding functions u : DISPLAYFORM0 which encode features of query and item to a k-dimensional embedding space.

The scoring function is then computed as the dot product between the query and item embeddings at the top layer, i.e., DISPLAYFORM1

To enable multi-task learning, we extend the two-tower model by adopting the shared-bottom architecture.

Specifically, we introduce two sub-towers on top of the bottom hidden layers, one for the explicit-feedback task and the other for the implicit-feedback task.

The outputs of bottom hidden layers are fed in parallel to the two sub-towers.

The bottom hidden layers are shared between the two sub-towers BID4 , and are referred to as shared-bottom layers.

The final model architecture is depicted in FIG1 .

During training, we first train the model for the auxiliary user engagement objective, which uses the cross entropy loss.

Having learned the shared representations, we finetune the model for the main relevance objective, which uses the squared loss.

To prevent potential over-fitting caused by the sparse relevance data, we apply stop gradients for the relevance objective on the shared-bottom layers.

For serving, we only need to store and serve the top layer of the two relevance sub-towers to predict the relevance.

In this section, we describe the experiments of our proposed framework on one of Google's large-scale retrieval systems for relevant item recommendations, e.g., apps.

Our system contains several millions of candidates.

Our training data contains hundreds of thousands of explicit feedback from relevance survey, and billions of implicit feedback from user logs.

We randomly split the data into 90% for training and 10% for evaluation.

Model performance was measured on the eval set by the Root Mean Square Error (RMSE) for relevance prediction.

The model was implemented in Tensorflow, of which the output relevance embeddings for queries and candidates were served for retrieval.

The hyper-parameters including model size, learning rate, and training steps were carefully tuned for the best model performance.

We study the effects of applying transfer learning to relevance prediction.

The following experiment results suggest that transfer learning significantly improves the prediction quality of sparse relevance task and helps avoid over-fitting.

Table 1 reports relevance RMSE (the lower the better) for different combinations of training objectives and feature types.

We see that using implicit feedback leads to a significant improvement as compared to using explicit feedback only.

Also, using collaborative information together with content information performs better than the model which uses collaborative information alone.

TAB1 .

Eval RMSE on relevance with varying model sizes.

The success of transfer learning hinges on a proper parameterization of both the auxiliary and main tasks.

On one hand, we need sufficient capacity to learn a high-quality representation from a large amount of auxiliary data.

On the other hand, we want to limit the capacity for the main task to avoid over-fitting to its sparse labels.

As a result, our proposed model architecture is slightly different from the traditional pre-trained and fine-tuning model BID12 .

Besides shared layers, each task has its own hidden layers with different capacities.

In addition, we apply a two-stage training with stop gradients to avoid potential issues caused by the extreme data skew between the main task and auxiliary task.

Our experiences have motivated us to continue our work in the following directions:• We will consider multiple types of user implicit feedback using different multi-task learning frameworks, such as Multi-gate Mixture-of-Expert BID17 and Sub-Network Routing BID18 .

We will continue to explore new model architectures to combine transfer learning with multi-task learning.• The auxiliary task requires hyper-parameter tuning to learn the optimal representation for the main task.

We will explore AutoML BID26 techniques to automate the learning of proper parameterizations across tasks for both the query and the candidate towers.

In this paper, we propose a novel model architecture to learn better query and candidate representations via transfer learning.

We extend the two-tower neural network approach to enhance sparse task learning by leveraging auxiliary tasks with rich implicit feedback.

By introducing auxiliary objectives and jointly learning this model using implicit feedback, we observe a significant improvement for relevance prediction on one of Google's large-scale retrieval systems.

@highlight

We propose a novel two-tower shared-bottom model architecture for transferring knowledge from rich implicit feedbacks to predict relevance for large-scale retrieval systems.