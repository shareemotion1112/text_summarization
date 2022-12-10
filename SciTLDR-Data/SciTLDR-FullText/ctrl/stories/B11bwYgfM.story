We investigate task clustering for deep learning-based multi-task and few-shot learning in the settings with large numbers of diverse tasks.

Our method measures task similarities using cross-task transfer performance matrix.

Although this matrix provides us critical information regarding similarities between tasks, the uncertain task-pairs, i.e., the ones with extremely asymmetric transfer scores, may collectively mislead clustering algorithms to output an inaccurate task-partition.

Moreover, when the number of tasks is large, generating the full transfer performance matrix can be very time consuming.

To overcome these limitations, we propose a novel task clustering algorithm to estimate the similarity matrix based on the theory of matrix completion.

The proposed algorithm can work on partially-observed similarity matrices based on only sampled task-pairs with reliable scores, ensuring its efficiency and robustness.

Our theoretical analysis shows that under mild assumptions, the reconstructed matrix perfectly matches the underlying “true” similarity matrix with an overwhelming probability.

The final task partition is computed by applying an efficient spectral clustering algorithm to the recovered matrix.

Our results show that the new task clustering method can discover task clusters that benefit both multi-task learning and few-shot learning setups for sentiment classification and dialog intent classification tasks.

This paper leverages knowledge distilled from a large number of learning tasks BID0 BID19 , or MAny Task Learning (MATL), to achieve the goal of (i) improving the overall performance of all tasks, as in multi-task learning (MTL); and (ii) rapid-adaptation to a new task by using previously learned knowledge, similar to few-shot learning (FSL) and transfer learning.

Previous work on multi-task learning and transfer learning used small numbers of related tasks (usually ∼10) picked by human experts.

By contrast, MATL tackles hundreds or thousands of tasks BID0 BID19 , with unknown relatedness between pairs of tasks, introducing new challenges such as task diversity and model inefficiency.

MATL scenarios are increasingly common in a wide range of machine learning applications with potentially huge impact.

Examples include reinforcement learning for game playing -where many numbers of sub-goals are treated as tasks by the agents for joint-learning, e.g. BID19 achieved the state-of-the-art on the Ms. Pac-Man game by using a multi-task learning architecture to approximate rewards of over 1,000 sub-goals (reward functions).

Another important example is enterprise AI cloud services -where many clients submit various tasks/datasets to train machine learning models for business-specific purposes.

The clients could be companies who want to know opinion from their customers on products and services, agencies that monitor public reactions to policy changes, and financial analysts who analyze news as it can potentially influence the stock-market.

Such MATL-based services thus need to handle the diverse nature of clients' tasks.

Challenges on Handling Diverse (Heterogeneous) Tasks Previous multi-task learning and fewshot learning research usually work on homogeneous tasks, e.g. all tasks are binary classification problems, or tasks are close to each other (picked by human experts) so the positive transfer between tasks is guaranteed.

However, with a large number of tasks in a MATL setting, the above assumption may not hold, i.e. we need to be able to deal with tasks with larger diversity.

Such diversity can be reflected as (i) tasks with varying numbers of labels: when tasks are diverse, different tasks could have different numbers of labels; and the labels might be defined in different label spaces without relatedness.

Most of the existing multi-task and few-shot learning methods will fail in this setting; and more importantly (ii) tasks with positive and negative transfers: since tasks are not guaranteed to be similar to each other in the MATL setting, they are not always able to help each other when trained together, i.e. negative transfer BID22 between tasks.

For example, in dialog services, the sentences "What fast food do you have nearby" and "Could I find any Indian food" may belong to two different classes "fast_food" and "indian_food" for a restaurant recommendation service in a city; while for a travel-guide service for a park, those two sentences could belong to the same class "food_options".

In this case the two tasks may hurt each other when trained jointly with a single representation function, since the first task turns to give similar representations to both sentences while the second one turns to distinguish them in the representation space.

A Task Clustering Based Solution To deal with the second challenge above, we propose to partition the tasks to clusters, making the tasks in each cluster more likely to be related.

Common knowledge is only shared across tasks within a cluster, thus the negative transfer problem is alleviated.

There are a few task clustering algorithm proposed mainly for convex models BID12 BID9 BID5 BID0 , but they assume that the tasks have the same number of labels (usually binary classification).

In order to handle tasks with varying numbers of labels, we adopt a similarity-based task clustering algorithm.

The task similarity is measured by cross-task transfer performance, which is a matrix S whose (i, j)-entry S ij is the estimated accuracy by adapting the learned representations on the i-th (source) task to the j-th (target) task.

The above task similarity computation does not require the source task and target task to have the same set of labels, as a result, our clustering algorithm could naturally handle tasks with varying numbers of labels.

Although cross-task transfer performance can provide critical information of task similarities, directly using it for task clustering may suffer from both efficiency and accuracy issues.

First and most importantly, evaluation of all entries in the matrix S involves conducting the source-target transfer learning O(n 2 ) times, where n is the number of tasks.

For a large number of diverse tasks where the n can be larger than 1,000, evaluation of the full matrix is unacceptable (over 1M entries to evaluate).

Second, the estimated cross-task performance (i.e. some S ij or S ji scores) is often unreliable due to small data size or label noises.

When the number of the uncertain values is large, they can collectively mislead the clustering algorithm to output an incorrect task-partition.

To address the aforementioned challenges, we propose a novel task clustering algorithm based on the theory of matrix completion BID2 .

Specifically, we deal with the huge number of entries by randomly sample task pairs to evaluate the S ij and S ji scores; and deal with the unreliable entries by keeping only task pairs (i, j) with consistent S ij and S ji scores.

Given a set of n tasks, we first construct an n × n partially-observed matrix Y, where its observed entries correspond to the sampled and reliable task pairs (i, j) with consistent S ij and S ji scores.

Otherwise, if the task pairs (i, j) are not sampled to compute the transfer scores or the scores are inconsistent, we mark both Y ij and Y ji as unobserved.

Given the constructed partially-observed matrix Y, our next step is to recover an n × n full similarity matrix using a robust matrix completion approach, and then generate the final task partition by applying spectral clustering to the completed similarity matrix.

The proposed approach has a 2-fold advantage.

First, our method carries a strong theoretical guarantee, showing that the full similarity matrix can be perfectly recovered if the number of observed correct entries in the partially observed similarity matrix is at least O(n log 2 n).

This theoretical result allows us to only compute the similarities of O(n log 2 n) instead of O(n 2 ) pairs, thus greatly reduces the computation when the number of tasks is large.

Second, by filtering out uncertain task pairs, the proposed algorithm will be less sensitive to noise, leading to a more robust clustering performance.

The task clusters allow us to handle (i) diverse MTL problems, by model sharing only within clusters such that the negative transfer from irrelevant tasks can be alleviated; and (ii) diverse FSL problems, where a new task can be assigned a task-specific metric, which is a linear combination of the metrics defined by different clusters, such that the diverse few-shot tasks could derive different metrics from the previous learning experience.

Our results show that the proposed task clustering algorithm, combined with the above MTL and FSL strategies, could give us significantly better deep MTL and FSL algorithms on sentiment classification and intent classification tasks.

Task/Dataset Clustering on Model Parameters This class of task clustering methods measure the task relationships in terms of model parameter similarities on individual tasks.

Given the parameters of convex models, task clusters and cluster assignments could be derived via matrix decomposition BID12 or k-means based approach BID9 .

The parameter similarity based task clustering method for deep neural networks BID21 applied low-rank tensor decomposition of the model layers from multiple tasks.

This method is infeasible for our MATL setting because of its high computation complexity with respect to the number of tasks and its inherent requirement on closely related tasks because of its parametersimilarity based approach.

Task/Dataset Clustering with Clustering-Specific Training Objectives Another class of task clustering methods joint assign task clusters and train model parameters for each cluster that minimize training loss within each cluster by K-means based approach BID5 or minimize overall training loss combined with sparse or low-ranker regularizers with convex optimization BID0 BID16 .

Deep neural networks have flexible representation power and they may overfit to arbitrary cluster assignment if we consider training loss alone.

Also, these methods require identical class label sets across different tasks, which does not hold in most of the real-world MATL settings.

Few Shot Learning FSL BID14 BID15 aims to learn classifiers for new classes with only a few training examples per class.

Bayesian Program Induction BID13 represents concepts as simple programs that best explain observed examples under a Bayesian criterion.

Siamese neural networks rank similarity between inputs BID11 .

Matching Networks BID20 ) maps a small labeled support set and an unlabeled example to its label, obviating the need for fine-tuning to adapt to new class types.

These approaches essentially learn one metric for all tasks, which is sub-optimal when the tasks are diverse.

An LSTM-based meta-learner BID18 learns the exact optimization algorithm used to train another learner neural-network classifier for the few-shot setting.

However, it requires uniform classes across tasks.

Our FSL approach can handle the challenges of diversity and varying sets of class labels.

Let T = {T 1 , T 2 , · · · , T n } be the set of n tasks to be clustered, and each task T i consists of a train/validation/test data split DISPLAYFORM0 .

We consider text classification tasks, comprising labeled examples {x, y}, where the input x is a sentence or document (a sequence of words) and y is the label.

We first train each classification model M i on its training set D train i , which yields a set of models M = {M 1 , M 2 , · · · , M n }.

We use convolutional neural network (CNN), which has reported results near state-of-the-art on text classification BID10 BID7 .CNNs also train faster than recurrent neural networks BID6 , making large-n MATL scenarios more feasible.

FIG0 shows the CNN architecture.

Following BID4 BID10 , the model consists of a convolution layer and a max-pooling operation over the entire sentence.

The model has two parts: an encoder part and a classifier part.

Hence each model DISPLAYFORM1 The above broad definitions encompasses other classification tasks (e.g. image classification) and other classification models (e.g. LSTMs BID6 ).We propose a task-clustering framework for both multi-task learning (MTL) and few-shot learning (FSL) settings.

In this framework, we have the MTL and FSL algorithms summarized in Section 3.3 & 3.4, where our task-clustering framework serves as the initial step in both algorithms.

FIG1 gives an overview of our idea and an example on how our task-clustering algorithm helps MTL.

Using single-task models, we can compute performance scores s ij by adapting each M i to each task T j (j = i).

This forms an n × n pair-wise classification performance matrix S, called the transfer-performance matrix.

Note that S is asymmetric since usually S ij = S ji .When all tasks have identical label sets, we can directly evaluate the model M i on the training set of task j, D train j, and use the accuracy as the cross-task transfer score S ij .When tasks have different label sets, we freeze the encoder M to get the accuracy as the transfer-performance S ij .

The score shows how the representations learned on task i can be adapted to task j, thus indicating the similarity between tasks.

Task Pair Sampling: When the number of tasks n is very large, the evaluation of O(n 2 ) entries is time-consuming.

Thus we sample n pairs of tasks {i, j} (i = j), with n n.

Then we set S ij and S ji as the transfer performance defined above when {i, j} is in the n samples, otherwise the entry is marked as "unobserved" 1 .

As discussed in the introduction, directly generating the full matrix S and partitioning tasks based on it has the following disadvantages: (i) there are too many entries to evaluate when the number of tasks is large; (ii) some task pairs are uncertain, thus can mislead the clustering algorithm to output an incorrect task-partition; and (iii) S is asymmetric, thus cannot be directly analyzed by many conventional clustering methods.

We address the first issue by randomly sample some task pairs to evaluate, as described in Section 3.1.

Besides, we address the other issues by constructing a symmetric similarity matrix and only consider the reliable task relationships, as will be introduced in Eq.(1).

Below, we describe our method (summarized in Algorithm 1) in detail.

First, we use only reliable task pairs to generate a partially-observed similarity matrix Y. Specifically, if S ij and S ji are high enough, then it is likely that tasks {i, j} belong to a same cluster and share significant information.

Conversely, if S ij and S ji are low enough, then they tend to belong to different clusters.

To this end, we need to design a mechanism to determine if a performance is high or low enough.

Since different tasks may vary in difficulty, a fixed threshold is not suitable.

Hence, we define a dynamic threshold using the mean and standard deviation of the target task performance, i.e., µ j = mean(S :j ) and σ j = std(S :j ), where S :j is the j-th column of S. We then introduce two positive parameters p 1 and p 2 , and define high and low performance as S ij greater than µ j + p 1 σ j or lower than µ j − p 2 σ j , respectively.

When both S ij and S ji are high and low enough, we set their pairwise similarity as 1 and 0, respectively.

Other task pairs are treated as uncertain task pairs and are marked as unobserved, and will have no influence to our clustering method.

This leads to a partially-observed symmetric matrix Y, i.e., DISPLAYFORM0 Given the partially observed matrix Y, we then reconstruct the full similarity matrix X ∈ R n×n .

We first note that the similarity matrix X should be of low-rank (proof deferred to appendix).

Additionally, since the observed entries of Y are generated based on high and low enough performance, it is safe to assume that most observed entries are correct and only a few may be incorrect.

Therefore, we introduce a sparse matrix E to capture the observed incorrect entries in Y. Combining the two observations, Y can be decomposed into the sum of two matrices X and E, where X is a low rank matrix storing similarities between task pairs, and E is a sparse matrix that captures the errors in Y. The matrix completion problem can be cast as the following convex optimization problem: DISPLAYFORM1 where • * denotes the matrix nuclear norm, the convex surrogate of rank function.

Ω is the set of observed entries in Y, and P Ω : R n×n → R n×n is a matrix projection operator defined as DISPLAYFORM2 The following theorem shows the perfect recovery guarantee for the problem (2).

The proof is deferred to Appendix.

Theorem 3.1.

Let X * ∈ R n×n be a rank k matrix with a singular value decomposition X * = UΣV , where U = (u 1 , . . . , u k ) ∈ R n×k and V = (v 1 , . . .

, v k ) ∈ R n×k are the left and right singular vectors of X * , respectively.

Similar to many related works of matrix completion, we assume that the following two assumptions are satisfied:1.

The row and column spaces of X have coherence bounded above by a positive number µ 0 .2.

Max absolute value in matrix UV is bounded above by µ 1 √ r/n for a positive number µ 1 .Suppose that m 1 entries of X * are observed with their locations sampled uniformly at random, and among the m 1 observed entries, m 2 randomly sampled entries are corrupted.

Using the resulting partially observed matrix as the input to the problem (2), then with a probability at least 1 − n −3 , the underlying matrix X * can be perfectly recovered, given DISPLAYFORM3 where C is a positive constant; ξ(•) and µ(•) denotes the low-rank and sparsity incoherence BID3 .Theorem 3.1 implies that even if some of the observed entries computed by (1) are incorrect, problem (2) can still perfectly recover the underlying similarity matrix X * if the number of observed correct entries is at least O(n log 2 n).

For MATL with large n, this implies that only a tiny fraction of all task pairs is needed to reliably infer similarities over all task pairs.

Moreover, the completed similarity matrix X is symmetric, due to symmetry of the input matrix Y. This enables analysis by similarity-based clustering algorithms, such as spectral clustering.

For each cluster C k , we train a model Λ k with all tasks in that cluster to encourage parameter sharing.

We call Λ k the cluster-model.

When evaluated on the MTL setting, with sufficient data to train a task-specific classifier, we only share the encoder part and have distinct task-specific classifiers FIG0 ).

These task-specific classifiers provide flexibility to handle varying number of labels.

We only have access to a limited number of training samples in few-shot learning setting, so it is impractical to train well-performing task-specific classifiers as in the multi-task learning setting.

Instead, we make the prediction of a new task by linearly combining prediction from learned clusters.

where Λ k is the learned (and frozen) model of the k-th cluster, {α k } K k=1 are adaptable parameters.

We use some alternatives to train cluster-models Λ k , which could better suit (and is more consistent to) the above FSL method.2 When all tasks have identical label sets, we train a single classification model on all the tasks like in previous work BID0 , the predictor P (y|x; Λ k ) is directly derived from this cluster-model.

When tasks have different label sets , we train a metriclearning model like BID20 among all the tasks in C k , which consist a shared encoding function Λ enc k aiming to make each example closer to examples with the same label compared to the ones with different labels.

Then we use the encoding function to derive the predictor by DISPLAYFORM0 where x l is the corresponding training sample for label y l .

Data Sets We test our methods by conducting experiments on three text classification data sets.

In the data-preprocessing step we used NLTK toolkit 3 for tokenization.

For MTL setting, all tasks are used for clustering and model training.

For FSL setting, the task are divided into training tasks and testing tasks (target tasks), where the training tasks are used for clustering and model training, the testing tasks are few-shot learning ones used to for evaluating the method in Eq. (3).1.

Amazon Review Sentiment Classification First, following BID0 , we construct a multi-task learning setting with the multi-domain sentiment classification BID1 data set.

The dataset consists of Amazon product reviews for 23 types of products (see Appendix 3 for the details).

For each domain, we construct three binary classification tasks with different thresholds on the ratings: the tasks consider a review as positive if it belongs to one of the following buckets =5 stars, >=4 stars or >=2 stars 4These review-buckets then form the basis of the task-setup for MATL, giving us 23 × 3 = 69 tasks in total.

For each domain we distribute the reviews uniformly to the three tasks.

For evaluation, we select tasks from 4 domains (Books, DVD, Electronics, Kitchen) as the target tasks (12 tasks) out of all 23 domains.

For FSL evaluation, we create five-shot learning tasks on the selected target tasks.

The cluster-models for this evaluation are standard CNNs shown in FIG0 (a), and we share the same output layer to evaluate the probability in Eq.(3) as all tasks have the same number of labels.2.

Diverse Real-World Tasks: User Intent Classification for Dialog System The second dataset is from an on-line service which trains and serves intent classification models to various clients.

The dataset comprises recorded conversations between human users and dialog systems in various domains, ranging from personal assistant to complex serviceordering or a customer-service request scenarios.

During classification, intent-labels 5 are assigned to user utterances (usually sentences).

We use a total of 175 tasks from different clients, and randomly sample 10 tasks from them as our target tasks.

For each task, we randomly sample 64% data into a training set, 16% into a validation set, and use the rest as the test set (see Appendix 3 for details).

The number of labels for these tasks vary from 2 to 100.

Hence, to adapt this to a FSL scenario, we keep one example for each label (one-shot), plus 20 randomly picked labeled examples to create our training data.

We believe this is a fairly realistic estimate of labeled examples one client could provide easily.

Since we deal with various number of labels in the FSL setting, we chose matching networks BID20 as the cluster-models.3.

Extra-Large Number of Real-World Tasks Similar to the second dataset, we further collect 1,491 intent classification tasks from the on-line service.

This setting is mainly used to verify the robustness of our task clustering method, since it is difficult to estimate the full transfer-performance matrix S in this setting (1,491 2 =2.2M entries).

Therefore, in order to extract task clusters, we randomly sample task pairs from the data set to obtain 100,000 entries in S, which means that only about 100K/2.2M ≈ 4.5% of the entries in S are observed.

The number of 100,000 is chosen to be close to n log 2 n in our theoretical bound in Theorem 3.1, so that we could also verify the tightness of the bound empirically.

To make the best use of the sampled pairs, in this setting we modified the Eq. 1, so that each entry Y ij = Y ji = 1 if S ij ≥ µ j or S ji ≥ µ i and Y ij = 0 otherwise.

In this way we could have determined number of entries in Y as well, since all the sampled pairs will correspond to observed (but noisy) entries in Y. We only run MTL setting on this data set.

Baselines For MTL setting, we compare our method to the following baselines: (1) single-task CNN: training a CNN model for each task individually; (2) holistic MTL-CNN: training one MTL-CNN model FIG0 ) on all tasks; (3) holistic MTL-CNN (target only): training one MTL-CNN model on all the target tasks.

For FSL setting, the baselines consist of: (1) single-task CNN: training a CNN model for each task individually; (2) single-task FastText: training one FastText model BID8 with fixed embeddings for each individual task; (3) Fine-tuned the holistic MTL-CNN: fine-tuning the classifier layer on each target task after training initial MTL-CNN model on all training tasks; (4) Matching Network: a metric-learning based few-shot learning model trained on all training tasks.

We initialize all models with pre-trained 100-dim Glove embeddings (trained on 6B corpus) BID17 .As the intent classification tasks usually have various numbers of labels, to our best knowledge the proposed method is the only one supporting task clustering in this setting; hence we only compare with the above baselines.

Since sentiment classification involves binary labels, we compare our method with the state-of-the-art logistic regression based task clustering method (ASAP-MT-LR) BID0 .

We also try another approach where we run our MTL/FSL methods on top of the (ASAP-Clus-MTL/FSL) clusters (as their entire formulation is only applicable to convex models).

In all experiments, we set both p 1 and p 2 parameters in (1) to 0.5.

This strikes a balance between obtaining enough observed entries in Y, and ensuring that most of the retained similarities are consistent with the cluster membership.

For MTL settings, we tune parameters like the window size and hidden layer size of CNN, learning rate and the initialization of embeddings (random or pre-trained) based on average accuracy on the union of all tasks' dev sets, in order to find the best identical setting for all tasks.

Finally we have the CNN with window size of 5 and 200 hidden units.

The learning rate is selected as 0.001; and all MTL models use random initialized word embeddings on sentiment classification and use Glove embeddings as initialization on intent classification, which is likely because the training sets of the intent tasks are usually small.

We also used the early stopping criterion based on the previous condition.

For the FSL setting, hyper-parameter selection is difficult since there is no validation data (which is a necessary condition to qualify as a k-shot learning).

So, in this case we preselect a subset of training tasks as validation tasks and tune the learning rate and training epochs (for the rest we follow the best setting from the MTL experiments) on the validation tasks.

During the testing phase (i.e. model training on the target FSL tasks), we fix the selected hyper-parameter values for all the algorithms.

Out-of-Vocabulary in Transfer-Performance Evaluation In text classification tasks, transferring an encoder with fine-tuned word embeddings from one task to another may not work as there can be a significant difference between the vocabularies.

Hence, while learning the single-task models (line 1 of Algorithm 1) we always use the CNNs with fixed set of pre-trained embeddings.

Improving Observed Tasks (MTL Setting) TAB0 shows the results of the 12 target tasks when all 69 tasks are used for training.

Since most of the tasks have a significant amount of training data, the single-task baselines achieve good results.

Because the conflicts among some tasks (e.g. the 2-star bucket tasks and 5-star bucket tasks require opposite labels on 4-star examples), the holistic MTL-CNN does not show accuracy improvements compared to the single-task methods.

It also lags behind the holistic MTL-CNN model trained only on 12 target domains, which indicates that the holistic MTL-CNN cannot leverage large number of background tasks.

Our ROBUSTTC-MTL method based on task clustering achieves a significant improvement over all the baselines.

BID0 85 The ASAP-MTLR (best score achieved with five clusters) could improve single-task linear models with similar merit of our method.

However, it is restricted by the representative strength of linear models so the overall result is lower than the deep learning baselines.

Adaptation to New Tasks (FSL Setting) Table 1(b) shows the results on the 12 five-shot tasks by leveraging the learned knowledge from the 57 previously observed tasks.

Due to the limited training resources, all the baselines perform poorly.

Our ROBUSTTC-FSL gives far better results compared to all baselines (>6%).

It is also significantly better than applying Eq. (3) without clustering (78.85%), i.e. using single-task model from each task instead of cluster-models for P (y|x; ·).Comparison to the ASAP Clusters Our clustering-based MTL and FSL approaches also work for the ASAP clusters, in which we replace our task clusters with the task clusters generated by ASAP-MTLR.

In this setting we get a slightly lower performance compared to the ROBUSTTC-based ones on both MTL and FSL settings, but overall it performs better than the baseline models.

This result shows that, apart from the ability to handle varying number of class labels, our ROBUSTTC model can also generate better clusters for MTL/FSL of deep networks, even under the setting where all tasks have the same number of labels.

It is worth to note that from Table 1(a), training CNNs on the ASAP clusters gives better results compared to training logistic regression models on the same 5 clusters (86.07 vs. 85.17), despite that the clusters are not optimized for CNNs.

Such result further emphasizes the importance of task clustering for deep models, when better performance could be achieved with such models.

TAB2 (a) & (b) show the MTL & FSL results on dialog intent classification, which demonstrates trends similar to the sentiment classification tasks.

Note that the holistic MTL methods achieve much better results compared to single-task CNNs.

This is because the tasks usually have smaller training and development sets, and both the model parameters learned on training set and the hyperparameters selected on development set can easily lead to over-fitting.

ROBUSTTC-MTL achieves large improvement (5.5%) over the best MTL baseline, because the tasks here are more diverse than the sentiment classification tasks and task-clustering greatly reduces conflicts from irrelevant tasks.

Although our ROBUSTTC-FSL improves over baselines under the FSL setting, the margin is smaller.

This is because of the huge diversity among tasks -by looking at the training accuracy, we found several tasks failed because none of the clusters could provide a metric that suits the training examples.

To deal with this problem, we hope that the algorithm can automatically decide whether the new task belongs to any of the task-clusters.

If the task doesn't belong to any of the clusters, it would not benefit from any previous knowledge, so it should fall back to single-task CNN.

The new task is treated as "out-of-cluster" when none of the clusters could achieve higher than 20% accuracy (selected on dev tasks) on its training data.

We call this method Adaptive ROBUSTTC-FSL, and it gives more than 5% performance boost over the best ROBUSTTC-FSL result.

Discussion on Clustering-Based FSL The single metric based FSL method (Matching Network) achieved success on homogeneous few-shot tasks like Omniglot and miniImageNet BID20 but performs poorly in both of our experiments.

This indicates that it is important to maintain multiple metrics for few-shot learning problems with more diverse tasks, similar to the few-shot NLP problems investigated in this paper.

Our clustering-based FSL approach maintains diverse metrics while keeping the model simple with only K parameters to estimate.

It is worthwhile to study how and why the NLP problems make few-shot learning more difficult/heterogeneous; and how well our method can generalize to non-NLP problems like miniImageNet.

We will leave these topics for future work.

TAB3 shows the MTL results on the extra-large dialog intent classification dataset.

Compared to the results on the 175 tasks, the holistic MTL-CNN achieves larger improvement (6%) over the single-task CNNs, which is a stronger baseline.

Similar as the observation on the 175 tasks, here the main reason for its improvement is the consistent development and test performance due to holistic multi-task training approach: both the single-task and holistic multi-task model achieve around 66% average accuracy on development sets.

Unlike the experiments in Section 4.3, we did not evaluate the full transfer-performance matrix S due to time considerations.

Instead, we only use the information of ∼ 4.5% of all the task-pairs, and our algorithm still achieves a significant improvement over the baselines.

Note that this result is obtained by only sampling about n log 2 n task pairs, it not only confirms the empirical advantage of our multi-task learning algorithm, but also verifies the correctness of our theoretical bound in Theorem 3.1.

In this paper, we propose a robust task-clustering method that not only has strong theoretical guarantees but also demonstrates significantly empirical improvements when equipped by our MTL and FSL algorithms.

Our empirical studies verify that (i) the proposed task clustering approach is very effective in the many-task learning setting especially when tasks are diverse; (ii) our approach could efficiently handle large number of tasks as suggested by our theory; and (iii) cross-task transfer performance can serve as a powerful task similarity measure.

Our work opens up many future research directions, such as supporting online many-task learning with incremental computation on task similarities, and combining our clustering approach with the recent learning-to-learn methods (e.g. BID18 ), to enhance our MTL and FSL methods.

We first prove that the full similarity matrix X ∈ R n×n is of low-rank.

To see this, let A = (a 1 , . . .

, a k ) be the underlying perfect clustering result, where k is the number of clusters and a i ∈ {0, 1} n is the membership vector for the i-th cluster.

Given A, the similarity matrix X is computed as DISPLAYFORM0 where B i = a i a i is a rank one matrix.

Using the fact that rank(X) ≤ k i=1 rank(B i ) and rank(B i ) = 1, we have rank(X) ≤ k, i.e., the rank of the similarity matrix X is upper bounded by the number of clusters.

Since the number of clusters is usually small, the similarity matrix X should be of low rank.

APPENDIX B: PROOF OF THEOREM 4.1We then prove our main theorem.

First, we define several notations that are used throughout the proof.

Let X = UΣV be the singular value decomposition of matrix X, where U = (u 1 , . . . , u k ) ∈ R n×k and V = (v 1 , . . .

, v k ) ∈ R n×k are the left and right singular vectors of matrix X, respectively.

Similar to many related works of matrix completion, we assume that the following two assumptions are satisfied:1.

A1: the row and column spaces of X have coherence bounded above by a positive number µ 0 , i.e., n/r max i P U (e i ) ≤ µ 0 and n/r max i P V (e i ) ≤ µ 0 , where P U = UU , P V = VV , and e i is the standard basis vector, and 2.

A2: the matrix UV has a maximum entry bounded by µ 1 √ r/n in absolute value for a positive number µ 1 .Let T be the space spanned by the elements of the form u i y and xv i , for 1 ≤ i ≤ k, where x and y are arbitrary n-dimensional vectors.

Let T ⊥ be the orthogonal complement to the space T , and let P T be the orthogonal projection onto the subspace T given by DISPLAYFORM1 The following proposition shows that for any matrix Z ∈ T , it is a zero matrix if enough amount of its entries are zero.

Proposition 1.

Let Ω be a set of m entries sampled uniformly at random from [1, . . .

, n] × [1, . . .

, n], and P Ω (Z) projects matrix Z onto the subset Ω. If m > m 0 , where m 0 = C 2 R µ 0 rnβ log n with β > 1 and C R being a positive constant, then for any Z ∈ T with P Ω (Z) = 0, we have Z = 0 with probability 1 − 3n −β .Proof.

According to the Theorem 3.2 in Candès & Tao (2010), for any Z ∈ T , with a probability at least 1 − 2n 2−2β , we have DISPLAYFORM2 where δ = m 0 /m < 1.

Since Z ∈ T , we have P T (Z) = Z. Then from (5), we have Z F ≤ 0 and thus Z = 0.In the following, we will develop a theorem for the dual certificate that guarantees the unique optimal solution to the following optimization problem DISPLAYFORM3 Theorem 1.

Suppose we observe m 1 entries of X with locations sampled uniformly at random, denoted by Ω. We further assume that m 2 entries randomly sampled from m 1 observed entries are corrupted, denoted by ∆. Suppose that P Ω (Y) = P Ω (X + E) and the number of observed correct entries m 1 − m 2 > m 0 = C 2 R µ 0 rnβ log n.

Then, for any β > 1, with a probability at least 1 − 3n −β , the underlying true matrices (X, E) is the unique optimizer of (6) if both assumptions A1 and A2 are satisfied and there exists a dual Q ∈ R n×n such that (a) DISPLAYFORM4 , and (e) P ∆ c (Q) ∞ < λ.

Proof.

First, the existence of Q satisfying the conditions (a) to (e) ensures that (X, E) is an optimal solution.

We only need to show its uniqueness and we prove it by contradiction.

Assume there exists another optimal solution (X + N X , E + N E ), where P Ω (N X + N E ) = 0.

Then we have DISPLAYFORM5 where Q E and Q X satisfying DISPLAYFORM6 As a result, we have DISPLAYFORM7 We then choose P ∆ c (Q E ) and P T ⊥ (Q X ) to be such that DISPLAYFORM8 is also an optimal solution, we have P Ω c (N E ) 1 = P T ⊥ (N X ) * , leading to P Ω c (N E ) = P T ⊥ (N X ) = 0, or N X ∈ T .

Since P Ω (N X + N E ) = 0, we have N X = N E + Z, where P Ω (Z) = 0 and P Ω c (N E ) = 0.

Hence, P Ω c ∩Ω (N X ) = 0, where |Ω c ∩ Ω| = m 1 − m 2 .

Since m 1 − m 2 > m 0 , according to Proposition 1, we have, with a probability 1 − 3n −β , N X = 0.

Besides, since P Ω (N X + N E ) = P Ω (N E ) = 0 and ∆ ⊂ Ω, we have P ∆ (N E ) = 0.

Since N E = P ∆ (N E ) + P ∆ c (N E ), we have N E = 0, which leads to the contradiction.

Given Theorem 1, we are now ready to prove Theorem 3.1.Proof.

The key to the proof is to construct the matrix Q that satisfies the conditions (a)-(e) specified in Theorem 1.

First, according to Theorem 1, when m 1 − m 2 > m 0 = C 2 R µ 0 rnβ log n, with a probability at least 1 − 3n −β , mapping P T P Ω P T (Z) : T → T is an one to one mapping and therefore its inverse mapping, denoted by (P T P Ω P T ) −1 is well defined.

Similar to the proof of Theorem 2 in BID3 , we construct the dual certificate Q as follows Q = λ sgn(E) + ∆ + P ∆ P T (P T P Ω P T ) −1 (UV + T )where T ∈ T and ∆ = P ∆ ( ∆ ).

We further define H = P Ω P T (P T P Ω P T ) −1 (UV ) DISPLAYFORM9 Evidently, we have P Ω (Q) = Q since ∆ ⊂ Ω, and therefore the condition (a) is satisfied.

To satisfy the conditions (b)-(e), we need P T (Q) = UV → T = −P T (λ sgn(E) + ∆ ) (7) P T ⊥ (Q) < 1 → µ(E) (λ + ∆ ∞ ) + P T ⊥ (H) + P T ⊥ (F) < 1 (8) P ∆ (Q) = λ sgn(E) → ∆ = −P ∆ (H + F) (9) |P ∆ c (Q)| ∞ < λ → ξ(X)(1 + T ) < λBelow, we will first show that there exist solutions T ∈ T and ∆ that satisfy conditions (7) and (9).

We will then bound Ω ∞ , T , P T ⊥ (H) , and P T ⊥ (F) to show that with sufficiently small µ(E) and ξ(X), and appropriately chosen λ, conditions (8) and FORMULA18 can be satisfied as well.

First, we show the existence of ∆ and T that obey the relationships in (7) and (9).

It is equivalent to show that there exists T that satisfies the following relation T = −P T (λ sgn(E)) + P T P ∆ (H) + P T P ∆ P T (P T P Ω P T ) −1 ( T ) or P T P Ω\∆ P T (P T P Ω P T ) −1 ( T ) = −P T (λ sgn(E)) + P T P ∆ (H),where Ω \ ∆ indicates the complement set of set ∆ in Ω and |Ω \ ∆| denotes its cardinality.

Similar to the previous argument, when |Ω \ ∆| = m 1 − m 2 > m 0 , with a probability 1 − 3n −β , P T P Ω\∆ P T (Z) : T → T is an one to one mapping, and therefore (P T P Ω\∆ P T (Z)) −1 is well defined.

Using this result, we have the following solution to the above equation T = P T P Ω P T (P T P Ω\∆ P T )−1 (−P T (λ sgn(E)) + P T P ∆ (H))We now bound T and ∆ ∞ .

Since T ≤ T F , we bound T F instead.

First, according to Corollary 3.5 in BID2 , when β = 4, with a probability 1 − n −3 , for any Z ∈ T , we have P T ⊥ P Ω P T (P T P Ω P T ) −1 (Z) F ≤ Z F .Using this result, we have DISPLAYFORM10 In the last step, we use the fact that rank( T ) ≤ 2k if T ∈ T .

We then proceed to bound T as follows DISPLAYFORM11 Combining the above two inequalities together, we have 1 − 2(k + 1)ξ(X)µ(E) To ensure that there exists λ ≥ 0 satisfies the above two conditions, we have 1 − 5(k + 1)ξ(X)µ(E) + (10k 2 + 21k + 8)[ξ(X)µ(E)] 2 > 0 and 1 − ξ(X)µ(E)(4k + 5) ≥ 0 Since the first condition is guaranteed to be satisfied for k ≥ 1, we have ξ(X)µ(E) ≤ 1 4k + 5 .Thus we finish the proof.

<|TLDR|>

@highlight

We propose a matrix-completion based task clustering algorithm for deep multi-task and few-shot learning in the settings with large numbers of diverse tasks.