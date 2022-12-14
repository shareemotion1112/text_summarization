We formulate a new problem at the intersection of semi-supervised learning and contextual bandits, motivated by several applications including clinical trials and dialog systems.

We demonstrate how contextual bandit and graph convolutional networks can be adjusted to the new problem formulation.

We then take the best of both approaches to develop multi-GCN embedded contextual bandit.

Our algorithms are verified on several real world datasets.

We formulate the problem of Online Partially Rewarded (OPR) learning.

Our problem is a synthesis of the challenges often considered in the semi-supervised and contextual bandit literature.

Despite a broad range of practical cases, we are not aware of any prior work addressing each of the corresponding components.

Online: data incrementally collected and systems are required to take an action before they are allowed to observe any feedback from the environment.

Partially: oftentimes there is no environment feedback available, e.g. a missing label Rewarded: instead of the true label, we can only hope to observe feedback indicating whether our prediction is good or bad (1 or 0 reward), the latter case obscuring the true label for learning.

Practical scenarios that fall under the umbrella of OPR range from clinical trials to dialog orchestration.

In clinical trials, reward is partial, as patients may not return for followup evaluation.

When patients do return, if feedback on their treatment is negative, the best treatment, or true label, remains unknown.

In dialog systems, a user's query is often directed to a number of domain specific agents and the best response is returned.

If the user provides negative feedback to the returned response, the best available response is uncertain and moreover, users can also choose to not provide feedback.

In many applications, obtaining labeled data requires a human expert or expensive experimentation, while unlabeled data may be cheaply collected in abundance.

Learning from unlabeled observations is the key challenge of semi-supervised learning BID2 .

We note that the problem of online semi-supervised leaning is rarely considered, with few exceptions BID14 BID13 .

In our setting, the problem is further complicated by the bandit-like feedback in place of labels, rendering existing semi-supervised approaches inapplicable.

We will however demonstrate how one of the recent approaches, Graph Convolutional Networks (GCN) BID9 , can be extended to our setting.

The multi-armed bandit problem provides a solution to the exploration versus exploitation tradeoff while maximizing cumulative reward in an online learning setting.

In Linear Upper Confidence Bound (LINUCB) BID10 BID4 and in Contextual Thompson Sampling (CTS) BID0 , the authors assume a linear dependency between the expected reward of an action and its context.

However, these algorithms assume that the bandit can observe the reward at each iteration.

Several authors have considered variations of partial/corrupted rewards BID1 BID6 , but the case of entirely missing rewards has not been studied to the best of our knowledge.

The rest of the paper is structured as follows.

In section 2, we formally define the Online Partially Rewarded learning setup and present two extensions to GCN to suit our problem setup.

Section 3 presents quantitative evidence of these methods applied to four datasets and analyses the learned latent space of these methods.

We first formally define each of the OPR keywords:Online: at each step t = 1, . . .

, T we observe observation x t and seek to predict its label?? t using x t and possibly any information we had obtained prior to step t.

Partially: after we make a prediction?? t , the environment may not provide feedback (we will use -1 to encode its absence) and we must proceed to step t + 1 without knowledge of the true y t .Rewarded: suppose there are K possible labels y t ??? {1, . . .

, K}. The environment at step t will not provide true y t , but instead a response h t ??? {???1, 0, 1}, where h t = 0 indicates?? t = y t and h t = 1 indicates?? t = y t (-1 indicates missing response).

Update?? with new edges using x t

Update GCN weights W DISPLAYFORM0 Retrieve GCN embeddings g(X) (k) 8: DISPLAYFORM1 (k) t 10: DISPLAYFORM2 (k) t

Predict?? t = argmax k (?? k + ?? k ) and observe h t 14: DISPLAYFORM0 ???k, y DISPLAYFORM1 Append t to each C k and 1 to r ??,k if?? t = k and 0 otherwise DISPLAYFORM2 Append t to C?? t , output of?? t -th GCN to r ??,??tRewarded Online GCN (ROGCN) is a natural extension of GCN, adapted to the online, partially rewarded setting along with a potential absence of true graph information.

We assume availability of a small portion of data and labels (size T 0 ) available at the start, X 0 ??? R T0??D and y 0 ??? {???1, 1, . . .

, K} T0 .

When there is no graph available we can construct a k-NN graph (k is a parameter chosen a priori) based on similarities between observations -this approach is common in convolutional neural networks on feature graphs BID8 BID5 and we adopt it here for graph construction between observations X 0 to obtain graph adjacency A 0 .

Using X 0 , y 0 , A 0 , we can train GCN with L hidden units (a parameter chosen a priori) to obtain initial estimates of hidden layer weights W 1 ??? R D??L and softmax weights W 2 ??? R L??K .

Next we start to observe the stream of data -as new observation x t arrives, we add it to the graph and data matrix, and append -1 (missing label) to y.

Then we run additional training steps of GCN and output a prediction to obtain environment response h t ??? {???1, 0, 1}. Here 1 indicates correct prediction, hence we include it to the set of available labels for future predictions; 0 indicates wrong prediction and -1 an absence of a response, in the later two cases we continue to treat the label of x t as missing.

ROGCN is unable to learn from missclassified observations and has to treat them as missing labels.

The bandit perspective allows one to learn from missclassfied observations, i.e. when the environment response h t = 0, and the neural network perspective facilitates learning better features such that linear classifier is sufficient.

This observation motivates us to develop a more sophisticated synthesis of GCN and LINUCB approaches, where we can combine advantages of both perspectives.

Notice that if K = 2, a h t = 0 environment response identifies the correct class, hence the OPR reduces to online semi-supervised learning for which GCN can be trivially adjusted using ideas from ROGCN.

To take advantage of this for K > 2, we propose to use a suite of class specific GCNs, where the hidden layer representation from the k-th class GCN, i.e. g(X)(k) =?? ReLU(??XW (k) 1 ) and g(X) (k) t denotes the embedding of observation x t , is used as context by the contextutal bandit for the predictions of the k-th arm.

Based on the environment response to the prediction, we update the labels and the reward information to reflect a correct, incorrect, or a missing environment response.

The reward is imputed from the corresponding GCN when the response is missing.

As we add new observation x t+1 to the graph and update weights of the GCNs, the embedding of the previous observations x 1 , . . .

, x t evolves.

Therefore instead of dynamically updating bandit parameters, we maintain a set of indices for each of the arms C k = {t :?? t = k or h t = 1} and use observations and responses from only these indices to update the corresponding bandit parameters.

Similar to ROGCN, we can use a small amount of data X 0 and labels y 0 converted to binary labels y DISPLAYFORM0 T0 (as before -1 encodes missing label) for each class k to initialize GCNs weights DISPLAYFORM1 for k = 1, . . .

, K. We present the GCNUCB in Algorithm 1, where r t,k ??? [0, 1] denotes the reward observed or imputed at step t for arm k as described in the algorithm.

In this section we compare baseline method LINUCB which ignores the data with missing rewards to ROGCN and GCNUCB.

We consider four different datasets: CNAE-9 and Internet Advertisements from the the UCI Machine Learning Repository 1 , Cora 2 , and Warfarin BID12 .

Cora is naturally a graph structured data which can be utilized by ROGCN and GCNUCB.

For other datasets we use a 5-NN graph built online from the available data as follows.

Suppose at step t we have observed data points x i ??? R D for i = 1, . . .

, t. Weights of the similarity graph computed as follows: DISPLAYFORM0 .

As it was done by Defferrard et al. (2016) we set ?? = A ij is the diagonal matrix of node degrees.

For pre-processing we discarded features with large magnitudes (3 features in Internet Advertisements and 2 features in Warfarin) and row normalized all observations to have unit l 1 norm.

For all the methods that use GCN, we use 16 hidden units for GCN, and use Adam optimizer with a learning rate of 0.01, and regularization strength of 5e-4, along with a dropout of 0.5.

To simulate the OPR setting, we randomly permute the order of the observations in a dataset and remove labels for 25% and 75% of the observations chosen at random.

For all methods we consider initial data X 0 and y 0 to represent a single observation per class chosen randomly (T 0 = K).

At a step t = T 0 + 1, . . .

, T each algorithm is given a feature vector x t and is ought to make a prediction?? t .

The environment response h t ??? {???1, 0, 1} is then observed and algorithms moves onto step t + 1.

To compare performance of different algorithms at each step t we compare?? t to true label y t available from the dataset (but concealed from the algorithms themselves) to evaluate running accuracy.

DISPLAYFORM1 For GCNUCB we use baseline LINUCB for first 300 steps, and for both we use explorationexploitation trade-off parameter ?? = 0.25.

Results are summarized in TAB0 .

Since ordering of the data can affect the problem difficulty, we performed 10 data resampling for each setting to obtain error margins.

GCNUCB outperforms the LINUCB baseline and ROGCN in all of the ex-periments, validating our intuition that a method synthesizing the exploration capabilities of bandits coupled with the effective feature representation power of neural networks is the best solution to the OPR problem.

We see the greatest increase in accuracy between GCNUCB and the alternative approaches on the Cora dataset which has a natural adjacency matrix.

This suggests that GCNUCB has a particular edge in OPR applications with graph structure.

Such problems are ubiquitous.

Consider our motivating example of dialog systems -for dialog systems deployed in social network or workplace environments, there exists graph structure between users, and user information can be considered alongside queries for personalization of responses.

Visualizing GCNUCB context space.

Recall that the context for each arm of GC-NUCB is provided by the corresponding binary GCN hidden layer.

The motivation for using binary GCNs to provide the context to LINUCB is the ability of GCN to construct more powerful features using graph convolution and neural networks expressiveness.

To see how this procedure improves upon the baseline LINUCB utilizing input features as context, we project the context and the corresponding bandit weight vectors, ?? 1 , . . .

, ?? K , for both LINUCB and GCNUCB to a 2-dimensional space using t-SNE BID11 .

In this experiment we analyzed CNAE-9 dataset with 25% missing labels.

Recall that the bandit makes prediction based on the upper confidence bound of the regret: argmax k (?? k x k,t + ?? k ) and that x k,t = x t ???k = 1, . . .

, K for LINUCB and x k,t = g(X) (k) t for GCNUCB.

To better visualize the quality of the learned weight vectors, for this experiment we set ?? = 0 and hence ?? k = 0 resulting in a greedy bandit, always selecting an arm maximizing expected reward ?? k x t,k .

In this case, a good combination of contexts and weight vectors is the one where observations belonging to the same class are well clustered and corresponding bandit weight vector is directed at this cluster.

For LINUCB FIG2 , 68% accuracy) the bandit weight vectors mostly point in the direction of their respective context clusters, however the clusters themselves are scattered, thereby inhibiting the capability of LINUCB to effectively distinguish between different arms given the context.

In the case of GCNUCB (Figure 2 , 77% accuracy) the context learned by each GCN is tightly clustered into two distinguished regions -one with context for corresponding label and binary GCN when it is the correct label (points with bolded colors), and the other region with context for the label and GCN when a different label is correct (points with faded colors).

The tighter clustered contexts allow GCNUCB to effectively distinguish between different arms by assigning higher expected reward to contexts from the correct binary GCN than others, thereby resulting in better performance of GCNUCB than other methods.

@highlight

Synthesis of GCN and LINUCB algorithms for online learning with missing feedbacks