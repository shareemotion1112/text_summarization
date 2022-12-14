Lifelong learning poses considerable challenges in terms of effectiveness (minimizing prediction errors for all tasks) and overall computational tractability for real-time performance.

This paper addresses continuous lifelong multitask learning by jointly re-estimating the inter-task relations (\textit{output} kernel) and the per-task model parameters at each round, assuming data arrives in a streaming fashion.

We propose a novel algorithm called  \textit{Online Output Kernel Learning Algorithm} (OOKLA) for lifelong learning setting.

To avoid the memory explosion, we propose a robust budget-limited versions of the proposed algorithm that efficiently utilize the relationship between the tasks to bound the total number of representative examples in the support set.

In addition, we propose a two-stage budgeted scheme for efficiently tackling the task-specific budget constraints in lifelong learning.

Our empirical results over three datasets indicate superior AUC performance for OOKLA and its budget-limited cousins over strong baselines.

Instead of learning individual models, learning from multiple tasks leverages the relationships among tasks to jointly build better models for each task and thereby improve the transfer of relevant knowledge between the tasks, especially from information-rich tasks to information-poor ones.

Unlike traditional multitask learning, where the tasks are presented simultaneously and an entire training set is available to the learner (Caruana (1998)), in lifelong learning the tasks arrives sequentially BID27 ).

This paper considers a continuous lifelong learning setting in which both the tasks and the examples of the tasks arrive in an online fashion, without any predetermined order.

Following the online setting, particularly from BID24 BID7 , at each round t, the learner receives an example from a task, along with the task identifier and predicts the output label for the example.

Subsequently, the learner receives the true label and updates the model(s) as necessary.

This process is repeated as we receive additional data from the same or different tasks.

Our approach follows an error-driven update rule in which the model for a given task is updated only when the prediction for that task is in error.

Lifelong learning poses considerable challenges in terms of effectiveness (minimizing prediction errors for all tasks) and overall computational tractability for real-time performance.

A lifelong learning agent must provide an efficient way to learn new tasks faster by utilizing the knowledge learned from the previous tasks and also not forgetting or significantly degrading performance on the old tasks.

The goal of a lifelong learner is to minimize errors as compared to the full ideal hindsight learner, which has access to all the training data and no bounds on memory or computation.

This paper addresses lifelong multitask learning by jointly re-estimating the inter-task relations from the data and the per-task model parameters at each round, assuming data arrives in a streaming fashion.

We define the task relationship matrix as output kernels in Reproducing Kernel Hilbert Space (RKHS) on multitask examples.

We propose a novel algorithm called Online Output Kernel Learning Algorithm (OOKLA) for lifelong learning setting.

For a successful lifelong learning with kernels, we need to address two key challenges: (1) learn the relationships between the tasks (output kernel) efficiently from the data stream and (2) bound the size of the knowledge to avoid memory explosion.

The key challenge in learning with a large number of tasks is to adaptively learn the model parameters and the task relationships, which potentially change over time.

Without manageability-efficient updates at each round, learning the task relationship matrix automatically may impose a severe computational burden.

In other words, we need to make predictions and update the models in an efficient real time manner.

We propose simple and quite intuitive update rules for learning the task relationship matrix.

When we receive a new example, the algorithm updates the output kernel when the learner made a mistake by computing the similarity between the new example and the set of representative examples (stored in the memory) that belongs to a specific task.

If the two examples have similar (different) labels and high similarity, then the relationship between the tasks is increased (decreased) to reflect the positive (negative) correlation and vice versa.

To avoid the memory explosion associated with the lifelong learning setting, we propose a robust budget-limited version of the proposed algorithm that efficiently utilizes the relationship between the tasks to bound the total number of representative examples in the support set.

In addition, we propose a two-stage budgeted scheme for efficiently tackling the task-specific budget constraints in lifelong learning.

It is worth noting that the problem of lifelong multitask learning is closely related to online multitask learning.

Although the objectives of both online multitask learning and lifelong learning are similar, one key difference is that the online multitask learning, unlike in the lifelong learning, may require that the number of tasks be specified beforehand.

In recent years, online multitask learning has attracted extensive research attention BID0 ; BID10 ; BID16 BID7 ; BID24 BID17 .

We evaluate our proposed methods with several state-of-the-art online learning algorithms for multiple tasks.

Throughout this paper, we refer to our proposed method as online multitask learning or lifelong learning.

There are many useful application areas for lifelong learning, including optimizing financial trading as market conditions evolve, email prioritization with new tasks or preferences emerging, personalized news, and spam filtering, with evolving nature of spam.

Consider the latter, where some spam is universal to all users (e.g. financial scams), some messages might be useful to certain affinity groups, but spam to most others (e.g. announcements of meditation classes or other special interest activities), and some may depend on evolving user interests.

In spam filtering each user is a "task," and shared interests and dis-interests formulate the inter-task relationship matrix.

If we can learn the matrix as well as improving models from specific spam/not-spam decisions, we can perform mass customization of spam filtering, borrowing from spam/not-spam feedback from users with similar preferences.

The primary contribution of this paper is precisely the joint learning of inter-task relationships and its use in estimating per-task model parameters in a lifelong learning setting.

Most existing work in online learning of multiple task focuses on how to take advantage of task relationships.

To achieve this, BID16 imposed a hard constraint on the K simultaneous actions taken by the learner in the expert setting, BID1 used matrix regularization, and BID10 proposed a global loss function, as an absolute norm, to tie together the loss values of the individual tasks.

Different from existing online multitask learning models, our paper proposes an intuitive and efficient way to learn the task relationship matrix automatically from the data, and to explicitly take into account the learned relationships during model updates.

BID7 assumes that task relationships are available a priori.

However often such taskrelation prior knowledge is either unavailable or infeasible to obtain for many applications especially when the number of tasks K is large BID28 ) and/or when the manual annotation of task relationships is expensive BID15 ).

BID24 formulated the learning of task relationship matrix as a Bregman-divergence minimization problem w.r.t.

positive definite matrices.

The model suffers from high computational complexity as semi-definite programming is required when updating the task relationship matrix at each online round.

We show that with a different formulation, we can obtain a similar but much cheaper updating rule for learning the inter-task weights.

BID17 proposed an efficient method for learning the task relationship matrix using the cross-task performance measure, but their approach learns only the positive correlation between the tasks.

Our proposed approach learns positive and negative correlations between the tasks for robust transfer of knowledge from the previously learned tasks.

Recent work in output kernel learning estimate the task covariance matrix in RKHS space, inferred it directly from the data BID12 BID25 BID14 ).

The task covariance matrix is called the output kernel defined on the tasks, similar to the scalar kernel on the inputs.

Most recently, BID14 showed that for a class of regularization functions, we can efficiently learn this output kernel.

Unfortunately most of the proposed methods for learning output kernels require access to the entire data for the learning algorithm, a luxury unavailable in online learning and especially in the lifelong learning setting.

Unlike in online multitask learning, most lifelong learning approaches use a single model for all the tasks or reuse the models from the previous tasks to build a model for the new task BID2 FORMULA0 ).

These approaches either increase the computation time on iterations where we encounter a novel task or reduce the prediction power of the model learned from the previous tasks due to catastrophic forgetting.

To the best of our knowledge, relationships among the tasks has not been successfully exploited in the lifelong learning setting due to the difficulty in learning a positive semi-definite task relationship matrix in large-scale applications.

This paper provides an efficient way to learn the task relationship matrix in the lifelong learning setting.

Let ((x t , i t ), y t ) be the example received by the learner from the task i t (at the time step t) where we assume that the x t ??? X and y t is its corresponding true label.

The task i t can be a new task or one seen by the learner in the previous iterations.

We denote by [N ] the consecutive integers ranging from 1 to N .

In this paper, we do not assume that the number of tasks is known to the learner ahead of time, an important constraint in lifelong learning problems.

Let K be the number of tasks seen so far until the current iteration t.

For brevity, we consider a binary classification problem for each task y t ??? {???1, +1}, but the methods generalize to multi-class cases and are also applicable to regression tasks.

We assume that the learner made a mistake if y t =?? t where?? t is the predicted label.

Our approach follows a mistake-driven update rule in which the model for a given task is updated only on rounds where the learner predictions differ from the true label.

Let K : X ?? X ??? R (kernel on input space) and ??? : N ?? N ??? R (output kernel) be symmetric, positive semi-definite (p.s.d) multitask kernel functions and denote H as their corresponding RKHS of functions with the norm ?? H on multitask examples BID26 ; BID12 BID14 ).

Using the above notation, we can define a kernel representation of an example based on a set of representative examples collected on the previous iterations (prototypes).

Formally, given an example x ??? X , its kernel representation can be written using this set:x ?????? {K(x, x s ) : s ??? S} S is the set of stored examples for which the learner made a mistake in the past.

The set S is called the support set.

The online classification function is then defined as the weighted sum of the kernel combination of the examples in the support set.

To account for the examples from the different tasks, we consider both the kernel on the input space K and the output kernel ??? in our classification function.

DISPLAYFORM0 We set ?? s = y s .

The predicted label for a new example is computed from the linear combination of the labels of the examples from the support set S weighted by their input similarity K and the task similarity ??? to the new example.

Using the kernel trick, one can write: DISPLAYFORM1 Note that, in the above representation, we need to learn both the support set S and the output kernel ??? from the data.

As explained in the previous section, for a successful lifelong learning with kernels, we need to address two key challenges: (1) learn the relationships between the tasks (output kernel) efficiently from the data arriving in an online fashion and (2) bound the size of the support set S to avoid memory explosion.

We address these two challenges in the following sections.

DISPLAYFORM2 (or) DISPLAYFORM3 end end

Our objective function for the lifelong learning problem is given as follows: DISPLAYFORM0 where (??) is some loss function such as hinge loss or logistic loss, R(??) is the regularization on the task relationship matrix ??? and ?? is the regularization parameter.

Note that f in the above equation depends on ???. In order to reduce the time taken for each time-step, we require an efficient update to the task relationship matrix ???. Following the work of BID14 in the batch setting, we consider a subset of regularization functions R for which we can efficiently learn the task covariance matrix.

Consider the dual function of the above equation, at time-step t (see BID4 ; BID14 ): DISPLAYFORM1 When we consider the entry-wise l p norm between ??? and ??? (t???1) from the previous iteration as our regularization i.e., R(???, DISPLAYFORM2 we get the update function in Equation 2.

Similarly, if we consider the generalized KL-divergence between ??? and ??? (t???1) i.e., R(???, FIG1 ) itk , we get the update function in Equation 3.

Unlike in the previous work, we update only the row (and the corresponding column) of the task relationship matrix ??? specific to the task i t , which significantly reduces the time taken per example.

DISPLAYFORM3 We can see that the update equations are simple and quite intuitive.

For a given new example (x t , i t ) at round t, the algorithm updates ??? itk (for some k ??? [K]) by computing the similarity between the new example and the examples in the support set S that belongs to the task k. If the two examples have similar (different) labels and high similarity K(x t , x s ), then the ??? itk is increased to reflect the positive (negative) correlation and vice versa.

A value close to 0 implies no significant relationship between the tasks.

The update to the ??? itk is normalized by the regularization parameter ?? for scaling.

It is worth noting that our update equations do not violate the p.s.d constraints on ??? in Equation 5.If ??? from the previous iteration is a p.s.d matrix and the update is a p.s.d matrix (as it is computed using the Gram matrix of the example from the previous iteration), the sum and Hadamard product of two p.s.d matrices satisfy the p.s.d constraint (using the Schur Product Theorem).Algorithm 1 outlines the key steps in our proposed method.

We write f ((x t , i t )) as f it (x t ) for notational convenience.

At each time-step t, the learner receives an example x t ??? X and predicts the output label y t using?? t = sign(f it (x t )).

We update both the support set S and the output kernel ??? it?? when the learner makes a mistake.

DISPLAYFORM4 Find an example to remove arg max DISPLAYFORM5 as in Algorithm 1. end end Algorithm 3: Two-Stage Budgeted Learning Initialize: DISPLAYFORM6

In Algorithm 1, we can see that both the classification function f and the update equations for ??? use the support set S. When the target function changes over time, the support set S grows unboundedly.

This leads to serious computational and runtime issues especially in the lifelong learning setting.

The most common solution to this problem is to impose a bound on the number of examples in the support set S. There are several budget maintenance strategies proposed recently BID6 ; BID11 ; BID18 ).

Unfortunately these schemes cannot be directly used in our setting due to the output kernels in our learning formulation.

BID5 proposed multitask variants of these schemes but they are impractical for the lifelong learning setting.

We follow a simple support set removal schemes based on BID8 .

In single-task setting, when the number of examples in the support set S exceeds the limit (say B), a simple removal scheme chooses an example x r with the highest confidence from S. The confidence of an example x r is measured using y r f(x r ) after removing x r from the support set S. DISPLAYFORM0 We extend the above approach to the multitask and lifelong learning settings.

Since the support set S is shared by all the tasks, we choose an example x r with high confidence to remove from each task function f k , weighted by the relationship among the tasks.

The objective function to choose the example is shown in Equation 6.

We show in the experiment section that this simple approach is efficient and performs significantly better than the state-of-the-art budget maintenance strategies.

Algorithm 2 shows pseudocode of the proposed budgeted learning algorithm.

In lifelong learning setting, the number of tasks is typically large.

The support set S may have hundreds or thousands of examples from all the tasks.

Each task does not use all the examples from the support set S. For example, in movie recommendations task, recommendation for each user (task) can be characterized by just a few movies (subset of examples) in the support set S. Motivated by this observation, we propose a two-stage budgeted learning algorithm for the lifelong learning setting.

Algorithm 3 shows pseudocode of the proposed two-stage budgeted learning algorithm.

In addition to the support set S, we maintain task-specific support set T k .

We choose the budget for each task (say L) where L <<< B. Similar to the removal strategies for S, we remove an example from T k when |T k | > L and replace with an example from the set S ??? T k .

The proposed two-stage approach provides better runtime complexity compared to the budgeted algorithm in Algorithm 2.

Since only a subset of tasks may hold an example from S, the removal step in Equation 6 requires only a subset of tasks for choosing an example.

This improves the runtime per iteration significantly when the number of tasks is large.

One may consider a different budget size for each task L k based on the complexity of the task.

In addition, the proposed two-stage budgeted learning algorithm provides an alternative approach to using state-of-the-art budget maintenance strategies.

For example, it is easier to use the Projectron algorithm BID18 ) on T k , rather than on S. We will further explore this line of research in our future work.

In this section, we evaluate the performance of our algorithms.

All reported results are averaged over 10 random runs on permutations of the training data.

Unless otherwise specified, all model parameters are chosen via 5-fold cross validation.

We use three benchmark datasets, commonly used for evaluating online multitask learning.

Details are given below:Newsgroups Dataset 2 consists of 20 tasks generated from two subject groups: comp and talk.politics.

We paired two newsgroups, one from each subject (e.g.,comp.graphics vs talk.politics.guns), for each task.

In order to account for positive/negative correlation between the tasks, we randomly choose one of the newsgroups as positive (+) or negative (???) class.

Each post in a newsgroup is represented by a vocabulary of approximately 60K unique features.

3 We use the dataset obtained from ECML PAKDD 2006 Discovery challenge for the spam detection task.

We used the task B challenge dataset, which consists of labeled training data from the inboxes of 15 users.

We consider each user as a single task and the goal is to build a personalized spam filter for each user.

Each task is a binary classification problem: spam (+) or non-spam (???) and each example consists of approximately 150K features representing term frequency of the word occurrences.

Some spam is universal to all users (e.g. financial scams), but some messages might be useful to certain affinity groups and spam to most others.

Such adaptive behavior of each user's interests and dis-interests can be modeled efficiently by utilizing the data from other users to learn per-user model parameters.

4 We also evaluated our algorithm on product reviews from amazon.

The dataset contains product reviews from 25 domains.

We consider each domain as a binary classification task.

Reviews with rating > 3 were labeled positive (+), those with rating < 3 were labeled negative (???), reviews with rating = 3 are discarded as the sentiments were ambiguous and hard to predict.

Similar to the previous datasets, each example consists of approximately 350K features representing term frequency of the word occurrences.

We choose 2000 examples (100 posts per task) for 20 Newsgroups, 1500 emails for spam (100 emails per user inbox) and 2500 reviews for sentiment (100 reviews per domain) as training set for our experiments.

Note that we intentionally kept the size of the training data small to simulate the lifelong learning setting and drive the need for learning from previous tasks, which diminishes as the training sets per task become large.

Since these datasets have a class-imbalance issue (with few (+) examples as compared to (???) examples), we use average Area Under the ROC Curve (AU C) as the performance measure on the test set.

To evaluate the performance of our proposed algorithm (OOKLA), we use the three datasets (Newsgroups, Spam and Sentiment) for evaluation and compare our proposed methods to 5 baselines.

We implemented Perceptron and Passive-Aggressive algorithm (PA) BID9 for online multitask learning.

Both Perceptron and PA learn independent model for each task.

These two baselines do not exploit the task-relationship or the data from other tasks during model update.

Next, we implemented two online multitask learning related to our approach: FOML -initializes ??? with fixed weights BID7 , Online Multitask Relationship Learning (OMTRL) BID24 -learns a task covariance matrix along with task parameters.

Since OMTRL requires expensive calls to SVD routines, we update the task-relationship matrix every 10 iterations.

In addition, we compare our proposed methods against the performance of Online Smooth Multitask Learning (OSMTL) which learns a probabilistic distribution over all tasks, and adaptively refines the distribution over time BID17 .

We implement two versions of our proposed algorithm with different update rules for the task-relationship matrix: OOKLA-sum (Equation 2 OOKLA with sum update) OOKLA-exp (Equation 3 OOKLA with exponential update) as shown in Algorithm 1.

TAB0 summarizes the performance of all the above algorithms on the three datasets.

In addition to the AU C scores, we report the average total number of support vectors (nSV) and the CPU time taken for learning from one instance (Time).From the table, it is evident that both OOKLA-sum and OOKLA-exp outperform all the baselines in terms of both AU C and nSV.

This is expected for the two default baselines (Perceptron and PA).

The update rule for FOML is similar to ours but using fixed weights.

The results justify our claim that learning the task-relationship matrix adaptively leads to improved performance.

As expected, both OOKLA and OSMTL consume less or comparable CPU time than the subset of baselines which take into account learning inter-task relationships.

Unlike in the OMTRL algorithm that recomputes the task covariance matrix every (10) iteration using expensive SVD routines, the task-relationship matrix in our proposed methods (and OSMTL) are updated independently for each task.

We implement the OSMTL with exponential update for our experiments as it has shown to perform better than the other baselines.

One of the major drawbacks of OSMTL is that it learn only the positive correlations between the tasks.

The performance of OSMTL worsens when the tasks are negatively correlated.

As we can see from the table, our proposed methods outperform OSMTL significantly in the Newsgroup dataset.

TAB1 compares the proposed methods with different budget schemes and budget sizes in terms of test set AU C scores and the runtime.

We use OOKLA-sum for this experiment.

We set the value of B to {50, 100, 150} for all the datasets.

We compare our proposed budgeted learning algorithm (Algorithm 2) with the following state-of-the-art algorithms for online budgeted learning: TAB1 shows both the test set AUC scores (first line) and time taken for learning from one instance (including the removal step).

It is evident from the table, our proposed budgeted learning algorithm for online multitask learning significantly outperforms the other state-of-the-art budget schemes on most settings.

Our proposed algorithm uses the relationship between the tasks efficiently to choose the next example for removal.

Finally, we evaluate the performance of the proposed two-stage budgeted scheme compared to the Algorithm 2.

To study the effect of different budget sizes L, we compute the cumulative mistake rate which uses all the examples from the support set S. We observe similar trend in the test set AUC scores.

On average, we achieved over 16% improvement in running time compared to the budget maintenance scheme in Algorithm 2.

We believe that the time consumption and the performance improvement will be even better for applications with larger numbers of tasks.

We proposed a novel lifelong learning algorithm using output kernels.

The proposed method efficiently learns both the model and the inter-task relationships at each iteration.

Our update rules for learning the task relationship matrix, at each iteration, were motivated by the recent work in output kernel learning.

In order to handle the memory explosion from an unbounded support set in the lifelong learning setting, we proposed a new budget maintenance scheme that utilizes the task relationship matrix to remove the least-useful (high confidence) example from the support set.

In addition, we proposed a two-stage budget learning scheme based on the intuition that each task only requires a subset of the representative examples in the support set for efficient learning.

It provides a competitive and efficient approach to handle large number of tasks in many real-life applications.

The effectiveness of our algorithm is empirically verified over several benchmark datasets, outperforming several competitive baselines both in the unconstrained case and the budget-limited case, where selective forgetting was required.

<|TLDR|>

@highlight

a novel approach for online lifelong learning using output kernels.