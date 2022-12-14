In order to mimic the human ability of continual acquisition and transfer of knowledge across various tasks, a learning system needs the capability for life-long learning, effectively utilizing the previously acquired skills.

As such, the key challenge is to transfer and generalize the knowledge learned from one task to other tasks, avoiding interference from previous knowledge and improving the overall performance.

In this paper, within the continual learning paradigm, we introduce a method that effectively forgets the less useful data samples continuously across different tasks.

The method uses statistical leverage score information to measure the importance of the data samples in every task and adopts frequent directions approach to enable a life-long learning property.

This effectively maintains a constant training size across all tasks.

We first provide some mathematical intuition for the method and then demonstrate its effectiveness with experiments on variants of MNIST and CIFAR100 datasets.

It is a typical practice to design and optimize machine learning (ML) models to solve a single task.

On the other hand, humans, instead of learning over isolated complex tasks, are capable of generalizing and transferring knowledge and skills learned from one task to another.

This ability to remember, learn and transfer information across tasks is referred to as lifelong learning or continual learning BID16 BID3 BID11 .

The major challenge for creating ML models with lifelong learning ability is that they are prone to catastrophic forgetting BID9 BID10 .

ML models tend to forget the knowledge learned from previous tasks when re-trained on new observations corresponding to a different (but related) task.

Specifically when a deep neural network (DNN) is fed with a sequence of tasks, the ability to solve the first task will decline significantly after training on the following tasks.

The typical structure of DNNs by design does not possess the capability of preserving previously learned knowledge without interference between tasks or catastrophic forgetting.

There have been different approaches proposed to address this issue and they can be broadly categorized in three types: I) Regularization: It constrains or regularizes the model parameters by adding some terms in the loss function that prevent the model from deviating significantly from the parameters important to earlier tasks.

Typical algorithms include elastic weight consolidation (EWC) BID4 and continual learning through synaptic intelligence (SynInt) BID19 .

II) Architectural modification: It revises the model structure successively after each task in order to provide more memory and additional free parameters in the model for new task input.

Recent examples in this direction are progressive neural networks BID14 and dynamically expanding networks BID18 .

III) Memory replay: It stores data samples from previous tasks in a separate memory buffer and retrains the new model based on both the new task input and the memory buffer.

Popular algorithms here are gradient episodic memory (GEM) BID8 , incremental classifier and representation learning (iCaRL) BID12 .Among these approaches, regularization is particularly prone to saturation of learning when the number of tasks is large.

The additional / regularization term in the loss function will soon lose its competency when important parameters from different tasks are overlapped too many times.

Modifications on network architectures like progressive networks resolve the saturation issue, but do not scale as number and complexity of tasks increase.

The scalability problem is also present when using memory replay and often suffer from high computational and memory costs.

In this paper, we propose a novel approach to lifelong learning with DNNs that addresses both the learning saturation and high computational complexity issues.

In this method, we progressively compresses the input information learned thus far along with the input from current task and form more efficiently condensed data samples.

The compression technique is based on the statistical leverage scores measure, and it uses frequent directions idea in order to connect the series of compression steps for a sequence of tasks.

Our approach resembles the use of memory replay since it preserves the original input data samples from earlier tasks for further training.

However, our method does not require extra memory for training and is cost efficient compared to most memory replay methods.

Furthermore, unlike the importance assigned to model specific parameters when using regularization methods like EWC or SynInt, we assign importance to the training data that is relevant in effectively learning new tasks, while forgetting less important information.

Before presenting the idea, let's first setup the problem: DISPLAYFORM0 ..} represent a sequence of tasks, each task consists of n i data samples and each sample has a feature dimension d and an output dimension m, i.e., input A i ??? R ni??d and true output B i ??? R ni??m .

Here, we assume the feature and output dimensions are fixed for all tasks 1 .

The goal is to train a DNN over the sequence of tasks and ensure it performs well on all of them.

Here, we consider that the network's architecture stays the same and the tasks are received in a sequential manner.

Formally, with f representing a DNN, our objective is to minimize the loss 2 : DISPLAYFORM1 (1) Under this setup, let's look at some existing models: Online EWC trains f on task (A i , B i ) with a loss function containing additional penalty terms min f f ( DISPLAYFORM2 j=1 ?? j and each ?? j is defined as the change of important parameters (using Fisher information matrix) in f with respect to the jth task.

GEM keeps an extra memory buffer containing data samples from each of the previous tasks M k with k < i, it trains on the current task (A i , B i ) with a regular loss func- DISPLAYFORM3 , but subject to inequalities on each update of f , DISPLAYFORM4 The new approach OLSS is to find an approximation of A in a streaming manner, i.e., to form an?? i to approximate DISPLAYFORM5 T such that the resultin?? DISPLAYFORM6 is likely to perform on all tasks as good as DISPLAYFORM7 (2) To avoid extra memory and computation cost during the training process, we restrict the approximate?? i to have the same number of rows as the current task A i .Equation (1) and (2) represent nonlinear least squares problems.

It is to be noted that a nonlinear least squares problem can be solved with an approximation deduced from an iteration of linear least squares problems with J T J????? = J T ???B where J is the Jacobian of f at each update (Gauss-Newton Method).

Besides this technique, there are various approaches in addressing this problem.

Here we adopt a cost effective simple randomization technique -leverage score sampling, which has been used extensively in solving large scale linear least squares and low rank approximation problems BID17 BID0 .

Definition 1 BID1 ) Given a matrix A ??? R n??d with n > d, let U denote the n ?? d matrix consisting of the d left singular vectors of A, and let U (i,:) denote the i-th row of U , then the statistical leverage score of the i-th row of A is defined as U (i,:) 2 2 for i ??? {1, ..., n}.Statistical leverage scores define the relevant non-uniformity structure of a matrix and a higher score indicates a heavier weight of the row contributing to the non-uniformity of the matrix; it has been widely used for constructing a randomized sketch of a matrix BID1 BID17 .

In our case, given an input matrix A, we will compute the leverage score of each row, then sample the rows with probability proportional to the scores.

Using leverage score sampling, we are able to select the important samples given a dataset.

The remaining problem is to embed it in a sequence of tasks.

In order to achieve this, we make use of the concept of frequent directions.

Frequent directions extends the idea of frequent items in item frequency approximation problem to a matrix BID7 BID2 BID15 .

Given a matrix A ??? R n??d whose rows are received one by one and a space parameter , the algorithm considers the first 2 rows in A and shrinks its top orthogonal vectors by the same amount to obtain an ?? d matrix; then combines them with the next rows in A for the next iteration, repeat the procedure until reaching the final sketch of dimension ?? d. Frequent directions algorithm is targeted at finding a low rank approximation on a continuously expanding matrix.

This is well suited for a continuous stream of data (tasks) within the lifelong learning setting.

We present the step by step procedure of performing leverage score sampling together with compression using frequent directions idea in Algorithm 1.

In our setting, we append the new task data samples to the existing buffer set and perform leverage score sampling to form a new buffer set and then train on it, this process is repeated for the entire sequence of tasks.

Input: A sequence of tasks { FIG1 , ..., (A i , B i ) , ...} with A i ??? R ni??d and B i ??? R ni??m ; initialization of the model parameters; a space parameter i.e., number of samples to pass in the model for training.

It can be set as n i or even smaller after receiving the i-th task, which avoids extra memory and computations during training.

Output: A trained neural network on a sequence of tasks.

Step 1 Initialize a buffer set S = {??,B} where both?? andB are empty.

Step 2 While the ith task is presented:Step 3 If?? andB are empty: Step 4 set?? = A i andB = B i , Step 5 else:Step 6 set?? = ?? A i andB = B B i .Step 7 Perform SVD: DISPLAYFORM0 Step 8 Randomly select rows of?? andB without replacement based on probability U j,:2 2 / U 2 F for j ??? {1, ..., n i + } (or j ??? {1, ..., n i } when i = 1) and set them as?? andB respectively.

Step 9Train the model with?? ??? R ??d andB ??? R ??m .

When n i is large, the SVD (singular value decomposition) of matrix?? ??? R (ni+ )??d in Step 6 is computationally expensive, we could use a streaming SVD method to speed up the process if is chosen much smaller than n i .

In that case the computational cost for SVD could be reduced from O((n i + )d2 ) to O(log 2 (n i + ) d 2 ) (assuming d < < n i ).

In addition, there exists various efficient ways to approximate the leverage scores BID1 BID13 which would further reduce the computational cost.

Remark: A major concern with this algorithm is that leverage scores is a linear measure, i.e., the selected samples capture the important information embedded linearly in the data matrix which may not fully represent the importance of the data samples.

Another related issue is that the nonlinear information probably depend on the structure of f , the DNN.

As such, there may be some underlying dependency of a data sample's importance on the DNN architecture.

We leave this open as a future research direction.

We evaluate the performance of the proposed algorithm OLSS on three classification tasks used as benchmarks in related prior work.??? Rotated MNIST (Lopez-Paz & Ranzato, 2017): a variant of the MNIST dataset of handwriten digits BID6 , the digits in each task are rotated by a fixed angle between 0 ??? to 180??? .

The experiment is on 20 tasks and each task consists of 60, 000 training and 10, 000 testing samples.??? Permutated MNIST BID4 : a variant of the MNIST dataset BID6 , the digits in each task are transformed by a fixed permutation of pixels.

The experiment is on 20 tasks and each task consists of 60, 000 training and 10, 000 testing samples.??? Incremental CIFAR100 BID12 BID19 : a variant of the CIFAR object recognition dataset with 100 classes BID5 ).

The experiment is on 20 tasks and each task consists of 5 classes; each task consists of 2, 500 training and 500 testing samples.

Where, each task introduces a new set of classes; for a total number of 20 tasks, each new task concerns examples from a disjoint subset of 5 classes.

In the original setting of (Lopez-Paz & Ranzato, 2017), a softmax layer is added to the output vector which only allows entries representing the 5 classes in the current task to output values larger than 0.

In our setting, we allow the entries representing all the past occurring classes to output values larger than 0.

We believe this is a more natural setup for lifelong learning.

The DNN used for rotated and permuted MNIST is an MLP with 2 hidden layers and each with 400 units; whereas a ResNet18 is used for the incremental CIFAR100 experiment.

We train 5 epochs with batch size 200 on rotated and permuted MNIST datasets and 10 epochs with batch size 100 on incremental CIFAR100.

In all experiments we compare the following algorithms: I) A simple SGD predictor, II) EWC BID4 , III) GEM (Lopez-Paz & Ranzato, 2017) and IV) OLSS (ours).In all the algorithms, we use a plain SGD optimizer.

All algorithms were implemented based on the publicly available code from the original authors of the GEM paper BID8 .

The regularization and memory hyper-parameters in EWC and GEM were set as described in BID8 .

The space parameter for our OLSS algorithm was set to be equal to the number of samples in each task; the learning rate for each algorithm was determined through a grid search on {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}.

Comparing across all the algorithms, we summarize the average test accuracy on the learned tasks in FIG1 (see Appendix Figure 2 for the change in the test accuracy at the first task, as more tasks are learned.) and the computational costs for each algorithm in TAB0 .

As observed from the figures, across the three benchmarks, OLSS and GEM achieve similar accuracy and significantly outperform both EWC and simple SGD training.

Nevertheless, GEM demands much higher computational resources (see TAB0 ) as the algorithm requires a constraint validation step and a potential gradient projection step to correct for constraint violations across all previously learned tasks during training (see Section 3 in BID8 ).

In detail, for GEM, the time complexity is proportional to the product of the number of samples kept in the memory buffer, the number of parameters in the model and the number of iterations required to converge.

In contrast, OLSS requires a SVD (or QR factorization) to compute the leverage scores for each task which can be achieved in a time complexity proportional to the product of the square of the number of features and the number of data samples, and is much less compared to GEM.

As observed in Appendix Figure 2 , OLSS shows robustness to catastrophic forgetting of the first task with positive backward transfer across all three datasets while learning the remaining sequence of tasks.

In the case of rotated and permuted MNIST, OLSS is the most robust method.

As presented in Appendix Figure 3 , after training on the whole sequence of tasks, both GEM and OLSS are able to preserve the accuracy for most tasks on rotated and permuted MNIST.

In contrast, it is hard to preserve the accuracy of the previously trained tasks on CIFAR100 for all algorithms.

As we noted earlier, EWC exhibits a saturation issue when the number of tasks increases.

This may hold for most regularization methods in order to achieve continual learning, as they target constraining the model parameters successively, thereby limiting the model capacity.

We presented a new approach in addressing the lifelong learning problem with deep neural networks.

It is inspired by the randomization and compression techniques typically used in statistical analysis.

We combined a simple importance sampling technique -leverage score sampling with the frequent directions concept and developed an online effective forgetting or compression mechanism that enables lifelong learning across a sequence of tasks.

Despite its simple structure, the results on MNIST and CIFAR100 experiments show its effectiveness as compared to recent state of the art.

@highlight

A new method uses statistical leverage score information to measure the importance of the data samples in every task and adopts frequent directions approach to enable a life-long learning property.