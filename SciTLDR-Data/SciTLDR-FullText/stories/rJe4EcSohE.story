The cost of annotating training data has traditionally been a bottleneck for supervised learning approaches.

The problem is further exacerbated when supervised learning is applied to a number of correlated tasks simultaneously since the amount of labels required scales with the number of tasks.

To mitigate this concern, we propose an active multitask learning algorithm that achieves knowledge transfer between tasks.

The approach forms a so-called committee for each task that jointly makes decisions and directly shares data across similar tasks.

Our approach reduces the number of queries needed during training while maintaining high accuracy on test data.

Empirical results on benchmark datasets show significant improvements on both accuracy and number of query requests.

A triumph of machine learning is the ability to predict with high accuracy.

However, for the dominant paradigm, which is supervised learning, the main bottleneck is the need to annotate data, namely, to obtain labeled training examples.

The problem becomes more pronounced in applications and systems which require a high level of personalization, such as music recommenders, spam filters, etc.

Several thousand labeled emails are usually sufficient for training a good spam filter for a particular user.

However, in real world email systems, the number of registered users is potentially in the millions, and it might not be feasible to learn a highly personalized spam filter for each of them by getting several thousand labeled data points for each user.

One method to relieve the need of the prohibitively large amount of labeled data is to leverage the relationship between the tasks, especially by transferring relevant knowledge from information-rich tasks to information-poor ones, which is called multitask learning in the literature.

We consider multitask learning in an online setting where the learner sees the data sequentially, which is more practical in real world applications.

In this setting, the learner receives an example at each time round, along with its task identifier, and then predicts its true label.

Afterwards, the learner queries the true label and updates the model(s) accordingly.

The online multitask setting has received increasing attention in the machine learning community in recent years BID6 BID0 BID7 BID9 BID4 BID13 BID11 .

However, they make the assumption that the true label is readily available to be queried, which is impractical in many applications.

Also, querying blindly can be inefficient when annotation is costly.

Active learning further reduces the work of the annotator by selectively requesting true labels from the oracles.

Most approaches in active learning for sequential and streambased problems adopt a measure of uncertainty / confidence of the learner in the current example BID5 BID3 BID12 BID8 BID1 .The recent work by BID10 combines active learning with online multitask learning using peers or related tasks.

When the classifier of the current task is not confident, it first queries its similar tasks before requesting a true label from the oracle, incurring a lower cost.

Their learner gives priority to the current task by always checking its confidence first.

In the case when the current task is confident, the opinions of its peers are ignored.

This paper proposes an active multitask learning framework which is more humble, in a sense that both the current task and its peers' predictions are considered simultaneously using a weighted sum.

We have a committee which makes joint decisions for each task.

In addition, after the true label of a training sample is obtained, this sample is shared directly to similar tasks, which makes training more efficient.

The problem formulation and setup are similar to BID11 BID10 .

Suppose we are given K tasks and the k-th task is associated with N k training samples.

We consider each task to be a linear binary classification problem, but the extensions to multiclass or non-linear cases are straightforward.

We use the good-old perceptron-based update rule in which the model for a given task is only updated when the prediction for that training example is in error.

The data for task k is {x DISPLAYFORM0 D is the i-th instance from the k-th task, yk ??? {???1, +1} is the corresponding label and D is the dimension of features.

When the notation is clear from the context, we drop task index k and simply write ((x (i) , k), y (i) ).

We consider the online setting where the training example ((x (t) , k), y (t) ) comes at round t.

Denote {w DISPLAYFORM1 the set of weights learned for the K binary classifiers at round t. Also denote w ??? R K??D the weight matrix whose k-th row is w k .

The label?? (t) is predicted based on the sign of the output value from the model.

Then the hinge loss of task k on the sample (( DISPLAYFORM2 at round t is given by DISPLAYFORM3 In addition, we also consider the losses of its peer tasks m (m = k) as DISPLAYFORM4 km indicates the loss incurred by using task m's knowledge / classifier to predict the label of task k's training sample.

DISPLAYFORM5 km plays an important role in learning the similarities among tasks and hence the committee weights.

Intuitively, two tasks should be more similar if one task's training samples can be correctly predicted using the other task's classifier.

The goal of this paper is to achieve a high accuracy on the test data, and at the same time to issue as small a number of queries to the oracle as possible during training, by efficiently sharing and transferring knowledge among similar tasks.

In this section we introduce our algorithm Active Multitask Learning with Committees (AMLC) as shown in Algorithm 1.

This algorithm provides an efficient way for online multitask learning.

Each task uses not only its own knowledge but also knowledge from other tasks, and shares training examples across similar tasks when necessary.

The two main components of Algorithm 1 are described in Section 3.1 and 3.2.

In Section 3.3, we compare AMLC with the state-of-the-art online multitask learning algorithm.

We maintain and update a relationship matrix ?? ??? R K??K through the learning process.

The k-th row of ?? , denoted ?? k , is the committee weight vector for task k, also referred to as committee for brevity.

Element ?? ij of the relationship matrix indicates the closeness or similarity between task i and task

if P (t) = 1 then 10: Query true label y (t) and set DISPLAYFORM0 12: Update ?? : DISPLAYFORM1 13: 20: end function j, and also the importance of task j in task i's committee in predicting.

Given a sample ((x (t) , k), y (t) ) at round t, the confidence of task k is jointly decided by its committee; namely, a weighted sum of confidences of all tasks, DISPLAYFORM2 DISPLAYFORM3 Each confidence is just the common confidence measure for perceptron, using distance from the decision boundary BID5 .

The prediction is done by taking the sign of the confidence value.

The learner then makes use of this confidence value by drawing a sample P (t) from a Bernoulli distribution, to decide whether to query the true label of this sample.

The larger p is, the more likely for P (t) to be 0, signifying greater confidence.

The hyperparameter b controls the level of confidence that the current task has to have to not request the true label.

The learner only queries the true label when the current task's committee turns out to be unconfident.

Another binary variable M (t) is set to be 1 if task k makes a mistake.

Subsequently, its weight vector is updated following the conventional perceptron scheme.

The learner then updates the relationship matrix following a similar policy as in BID11 BID10 Table 1 .

Accuracy on test set and total number of queries during training over 10 random shuffles of the training examples.

The 95% confidence level is provided after the average accuracy.

The best performance is highlighted in bold.

On Spam Detection, both PEER+Share and AMLC are highlighted because AMLC has a lower mean but also smaller variance.

DISPLAYFORM4 km /??).

The hyperparameter C decides how much decrease happens on the weight given non-zero loss, and ?? = K m=1 l (t) km .

These new weights are then normalized to sum to 1.

To further encourage data sharing and information transfer between similar tasks, after the true label is obtained, the learner also shares the data with similar tasks of task k, so that peer tasks can learn from this sample as well.

Similar tasks are identified by having a larger weight than the current task in the committee.

We set S (t) m = 1 to indicate task m is a similar task to k and thus the data is shared with it.

The most related work to ours is active learning from peers (PEER) BID10 .

In this section we discuss the main difference between our method and theirs with some intuition.

Firstly, we do not treat the task itself and its peer tasks separately.

Instead, the final confidence of the current task is jointly decided using the confidences of all tasks, weighted by the committee weight vector.

It is humble in a sense that it always considers its peer tasks' advice when making a decision.

There are two main advantages of our approach.

1) For PEER, no updates happen and no knowledge is transferred when the current task itself is confident.

This can result in difficulties for the learner to recover from being blindly confident.

Blind confidence happens when the classifier makes mistakes on training examples but with high confidence, especially in early stage of training when data are not enough.

2) Our method updates the committee weight vector while keeping m??? [K] ?? km = 1 instead of m??? [K] ,m =k ?? km = 1.

It then becomes possible that the current task itself has an equal or lower influence than other tasks on the final prediction.

This is more desirable because identical tasks should have equal weights, and informationpoor tasks should rely more on their information-rich peers when making predictions.

Secondly, our algorithm enables the sharing of training data across similar tasks directly, after acquiring the true label of this data.

Querying can be costly, and the best way to make use of the expensive label information is to share it.

Assuming that all tasks are identical, the most productive algorithm would merge all data to learn a single classifier.

PEER is not able to achieve this because each task is still trained independently, since tasks only have access to their own data.

Though PEER indirectly accesses others' data through querying peer tasks, this sharing mechanism can be insufficient when tasks are highly similar.

In the case that all tasks are identical, our algorithm converges to a relationship matrix with identical elements and eventually all tasks are trained on every example that has been queried.

In this section, we evaluate our proposed algorithm on three benchmark datasets for multitask learning, and compare our performance with many baseline models.

We set b = 1 for all the experiments and tune the value of C from 20 values using 10-fold cross validation.

Unless otherwise specified, all other model parameters are chosen via 10-fold cross validation.

Landmine Detection 1 consists of 19 tasks collected from different landmine fields.

Each task is a binary classification problem: landmines (+) or clutter (-), and each example consists of 9 features.

Spam Detection 2 consists of labeled training data: spam (+) or non-spam (-) from the inboxes of 15 users, and each user is considered as a single task.

Sentiment Analysis 3 (Blitzer et al.) consists of product re- views from Amazon containing reviews from 22 domains.

We consider each domain as a binary classification task: positive review (+) and negative review (-).

Details about our training and test sets are shown in Appendix A.

We compare the performance of 5 different models.

Random does not use any measure of confidence.

Namely, the probability of querying or not querying true label are equal.

Independent uses the confidence which is purely computed form the weight vector of the current task.

Obviously both Random and Independent have no knowledge transfer among tasks.

PEER is the algorithm from BID10 .

AMLC (Active Multitask Learning with Committees) is our proposed method as shown in Algorithm 1.

In addition, we also show the performance of PEER+Share, in which we simply add to PEER the data sharing mechanism as illustrated in section 3.2.

Table 1 shows the accuracy on test set and the total number of queries (label requests) to oracles during training of five models.

Each value is the average of 10 random shuffles of the training set.

The 95% confidence level is also shown.

Notice that our re-implementation of PEER achieves similar performance on the Landmine and Spam datasets but seems to perform worse on Sentiment.

The reason is that we are using a different representation of the training examples.

We use the default bag-of-words representation coming with the dataset and there are approximately 2.9M features.

The highlighted values illustrate the best performance across all models.

On Spam Detection, AMLC is also highlighted because it is more confident about its accuracy even though the actual value is slightly lower than PEER+Share.

It can be seen that our proposed methods (PEER+Share and AMLC) significantly outperform the the others.

PEER has better performance compared to Random and Independent but still behaves worse than PEER+Share and AMLC.

It can be shown that simply adding data sharing can improve both accuracy and number of queries used during training.

The only exception is on Landmine Detection, where PEER+Share requests more queries than PEER.

Though simply adding data sharing results in improvement, after learning with joint decisions in AMLC, we observe further drastic decrease on the number of queries, while maintaining a high accuracy.

Another goal of active multitask learning is to efficiently make use of the labels.

In order to evaluate this, we give each model a fixed number of query budget and the training process is ended after the budget is exhausted.

We show three plots (one for each dataset) in FIG1 .

Based on the difficulty of learning from each dataset, we choose different budgets to evaluate (up to 10%, 30% and 30% of the total training examples for Landmine, Spam and Sentiment respectively).

We can see that given a limited number of query budgets, AMLC outperforms all models on all three datasets, as a result of it encouraging more knowledge transfer among tasks.

It is worth noting that the Landmine dataset is quite unbalanced (high proportion of negative labels), and PEER+Share and AMLC can achieve high accuracy with extremely limited number of queries.

However, the classifier learned by PEER+Share is unconfident and thus it keeps requesting true labels in the following training process.

We propose a new active multitask learning algorithm that encourages more knowledge transfer among tasks compared to the state-of-the-art models, by using joint decision / prediction and directly sharing training examples with true labels among similar tasks.

Our proposed methods achieve both higher accuracy and lower number of queries on three benchmark datasets for multitask learning problems.

Future work includes theoretical analysis of the error bound and comparison with those of the baseline models.

Another interesting direction is to handle unbalanced task data.

In other words, one task has much more / less training data than the others.

@highlight

We propose an active multitask learning algorithm that achieves knowledge transfer between tasks.