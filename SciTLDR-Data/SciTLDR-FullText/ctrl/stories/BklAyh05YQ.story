We propose a new method for training neural networks online in a bandit setting.

Similar to prior work, we model the uncertainty only in the last layer of the network, treating the rest of the network as a feature extractor.

This allows us to successfully balance between exploration and exploitation due to the efficient, closed-form uncertainty estimates available for linear models.

To train the rest of the network, we take advantage of the posterior we have over the last layer, optimizing over all values in the last layer distribution weighted by probability.

We derive a closed form, differential approximation to this objective and show empirically over various models and datasets that training the rest of the network in this fashion leads to both better online and offline performance when compared to other methods.

Applying machine learning models to real world applications almost always involves deploying systems in dynamic, non-stationary environments.

This dilemma requires models to be constantly re-updated with new data in order to maintain a similar model performance across time.

Of course, doing this usually requires the new data to be relabeled, which can be expensive or in some cases, impossible.

In many situations, this new labeled data can be cheaply acquired by utilizing feedback from the user, where the feedback/reward indicates the quality of the action taken by the model for the given input.

Since the inputs are assumed independent, this task can be framed in the contextual bandit setting.

Learning in this setting requires a balance between exploring uncertain actions (where we risk performing sub optimal actions) and exploiting actions the model is confident will lead to high reward (where we risk missing out on discovering better actions).Methods based on Thompson sampling (TS) or Upper Confidence Bounds (UCB) provide theoretically BID1 BID0 and empirically established ways BID12 BID3 for balancing exploration/exploitation in this setting.

Unfortunately, both methods require estimation of model uncertainty.

While this can be done easily for most linear models, it is a difficult and open problem for large neural network models underlying many modern machine learning systems.

An empirical study by BID17 shows that having good uncertainty estimates is vital for neural networks learning in a bandit setting.

Closed formed uncertainty estimations (and online update formulas) are available for many linear models.

Since the last layer of many neural networks are usually (generalized) linear models, a straightforward way for learning neural networks in a bandit setting is to estimate the uncertainty (as a distribution over weights) on the last layer only, holding the previous layers fixed as feature functions which provide inputs to the linear model.

This method (and variants thereof) has been proposed in bandit settings BID17 BID13 as well as other related settings (Snoek et al., 2015; BID16 BID11 and has been shown to work surprisingly well considering its simplicity.

This style of methods, which we refer to as Bayesian last layer or BLL methods, also has the advantage of being both relatively model-agnostic and scalable to large models.

Of course, BLL methods come with the tacit assumption that the feature functions defined by the rest of the network output good (linearly separable) representations of our inputs.

This means that, unless the input data distribution is relatively static, the rest of the network will need to be updated in regular intervals to maintain low regret.

In order to maintain low regret, the retraining objective must: 1) allow new data to be incorporated quickly into the learned model, and 2) prevent previously learned information from being quickly forgotten.

Previous papers retrain BLL methods simply by sampling minibatches from the entire pool of previously seen data and maximizing log-likelihood over these minibatches, which fails to meet the first criteria above.

In this paper we present a new retraining objective for BLL methods meeting both requirements.

We avoid retraining the last layer with the entire network (throwing out the uncertainty information we learned about the last layer) or retraining with the last layer fixed (fixing the last layer to the mean of its distribution).

Instead, we utilize the uncertainty information gathered about the last layer, and optimize the expected log-likelihood of both new and old data, marginalizing 1 over the entire distribution we have on the last layer.

This gives a more robust model that performs relatively well over all likely values of the last layer.

While this objective cannot be exactly computed, we derive a closed form, differentiable, approximation.

We show that this approximation meets both criteria above, with a likelihood term to maximize that depends only on the new data (meeting the first point), and a quadratic regularization term that is computed only with previously seen data (meeting the second point).

We show empirically that this method improves regret on the most difficult bandit tasks studied in BID17 .

We additionally test the method on a large state-of-the-art recurrent model, creating a bandit task out of a paraphrasing dataset.

Finally, we test the method on convolutional models, constructing a bandit task from a benchmark image classification dataset.

We show that our method is fast to adapt to new data without quickly forgetting previous information.

Contextual bandits are a well researched class of sequential decision problems (Wang et al., 2005; BID10 , of which, many variants exist.

In this paper, we mainly consider the multiclass contextual bandit problem studied in BID7 .

The problem takes place over T online rounds.

On round t, the learner receives a context x t , predicts a class label y t , and receives binary reward r t indicating whether the chosen label is correct.

No other information is received about the other classes not picked.

In our setting, we assume each class c (the arms of the bandit) has associated with it a vector z c and that the probability of a class label is modeled by: DISPLAYFORM0 , where ?? is the logistic function and f ?? is a feature function parameterized by ??.

In our case, f ?? defines a neural network, while z c can be seen as the last layer of the network 2 .Our goal is to get low regret, R = T i r * i ??? T i r i , where r * i is the optimal reward at step i. The key to getting low regret is employing a policy for balancing exploration and exploitation.

If we capture the uncertainty in each z c at time t by modeling its posterior distribution over previous data D t???1 as a multivariate Gaussian, z c ??? p(z c |D t???1 ), then we can easily deploy sound exploration strategies such as Thompson sampling or UCB.

If we hold f ?? fixed, then we can easily model this distribution by doing an online Bayesian regression on the outputs of f ?? , which gives us closed form formulas for updating the posterior over the last layer (specifically, its mean and covariance) given a single datapoint.3 When f ?? is a neural network, then this is an instance of a BLL method.

BLL methods have been shown to be an effective, model agnostic, and scalable way to deal with exploration problems involving neural networks.

Previous work has found them to be a pragmatic method for obtaining approximate uncertainty estimates for exploration BID13 BID17 BID16 BID2 and as proxies for Gaussian processes in both Bayesian Optimization problems (Snoek et al., 2015) and general regression tasks BID11 .If f ?? is fixed, then z c can be updated efficiently in an online manner.

An unanswered question still remains however: how does one actually update and learn f ?? ?

If you don't care about achieving low regret (ie you only care about offline performance), then the answer is easy; just gather your data, train f ?? offline, possibly with off-policy learning methods (Strehl et al., 2010; BID6 , and learn the Bayesian regression post-hoc.

Of course, if you are concerned about online performance (regret) then this is not a viable option.

A training method for f ?? must take care of two things: when do we update the feature functions and what do we update them with?

A reasonable answer to the former question is to update on a fixed schedule (every T rounds).

In this paper, we focus on answering the latter questions of which there are two obvious solutions, each with a corresponding problem:(1) Sample minibatches only from recent data.

Problem: We may overfit on this data and forget old information.(2) Sample minibatches uniformly from the set of all collected data (both recent and old).

Problem:We have lost the ability to adapt quickly to new data.

If the input distribution suddenly shifts, we will likely have to wait many iterations before newer data becomes common enough in our minibatch samples, all while our regret increases.

One thing to consider is that when it comes time to train our feature functions, we have access to a distribution over our last layer.

If, for example, our distribution has suddenly shifted, then the last layer distribution should have more variance, ideally placing density on last layers that do well on older data, as well as those that fit well to the new data.

If the distribution remains the same, then the variance should be low and density should be placed on relatively few values.

Intuitively, we can get the best of both worlds (ability to adapt or retain information when needed) by optimizing over all values of the last layer weighted by their probability.

In the next section, we derive a local approximation to this objective that shows this indeed is the case.

Let D t and D t???1 be the most recent set of data collected online, and the set of all previously collected data, respectively.

Additionally, assume a zero mean Gaussian prior over the last layer, p(z|??) that is constant with respect to ??.

Recall that during online training we fix the parameters ?? = ?? t???1 , and model the distribution Q = p(z c |D t , D t???1 , ?? t???1 ).

We want a value of ?? such that both our data and the last layer values drawn from Q are likely.

Thus our objective is to maximize: DISPLAYFORM0 We can write the marginal likelihood as a convolution between a logistic function and a Gaussian (based on our assumption of zero mean Gaussian prior p(z|??)), which can be approximated in closed form as per MacKay FORMULA1 : DISPLAYFORM1 Where ?? is the mean of p(z|??).

Since we have a zero mean prior, the above term evaluates to ??(0) whose value is a constant 1 2 .

Using this result, we can rewrite equation FORMULA1 as: DISPLAYFORM2 Where c is a constant entropy term which can be ignored.

For the first expectation term, we can use a second order Taylor expansion around E z???Q [log p(D t |z, ??)] to get a closed form approximation 4 : DISPLAYFORM3 This approximation was used in Teh et al. (2007) and shown to work well empirically.

The expectations in equation FORMULA4 The KL term in equation FORMULA3 can also be approximated locally with a second Taylor expansion around the current value of ?? = ?? t???1 .

Let K(??) = KL(Q||p(z|D t???1 , ??)).

Then, the second order Taylor expansion around ?? = ?? t???1 is: DISPLAYFORM4 Utilizing properties of KL divergence, as well as equation FORMULA2 , it can be derived 5 that K (?? t???1 ) will evaluate to 0, and K (?? t???1 ) will evaluate to ??F P , where ?? = DISPLAYFORM5 and F P is the Fisher Information Matrix of P = p(z|D t???1 , ?? t???1 ).

Getting rid of constants, we can write the local KL approximation (when ?? is close to ?? t???1 ) as: DISPLAYFORM6 The term ?? defines the ratio between the expected data likelihood given the last layer z distributed as z ??? p(z|D t???1 , ?? t???1 ) and the expected data likelihood given the last layer is distributed under the prior p(z|??).

This indicates that the better our previous values of ?? t???1 and z explain the data, the more we should regularize when incorporating new data (i.e raising the value of ??).

In practice, these values may be computed or approximated, however for efficiency, we treat ?? as a hyperparameter, and linearly increase it throughout the learning process.

Our final objective to optimize is thus: DISPLAYFORM7 Notice that the old data is now only used to calculate the Fisher information matrix F P and is not actually involved in the optimization.

Thus, the optimization (at least temporarily) over all our data can be done by simply drawing minibatches from the new data only, while the old data is only used to calculate the regularization term.

The practical benefit of this is that the regularization term can be easily computed in parallel while doing the online Bayesian regression and collecting new data.

The quadratic regularization term shares similarities to objective functions in continual learning which aim to prevent catastrophic forgetting BID8 Ritter et al., 2018) .

Combining the online learning and the network retraining stages described in the previous section gives us the general form of the iterative algorithm we study in this paper.

The algorithm alternates between two stages:Online Phase: As input, take in a set of data D t???1 , the posteriors (one for each arm) of the last layer conditioned on previous data p(z|D t???1 ) as well as a fixed value of f ?? .

This phase takes place over a series of T online rounds.

In every round, the learner receives a context x i , and uses the posteriors over the last layer to decide which arm to pull (via Thompson sampling or UCB).

The learner receives feedback y i upon pulling the arm c, and updates the posterior over z c with it.

After T rounds, the learner outputs the updated posteriors over z, and the data collected this phase, D t .Offline/Retraining Phase: As input, take in D t , D t???1 , and the posteriors over z. Retrain f ?? using method described in Section 3.

Set D t???1 = D t ???D t???1 .

Recompute the posteriors over z conditioned on D t???1 using the new value of f ?? .

Output D t???1 , f ?? , and the updated posteriors p(z|D t???1 ).The marginalization method described in Section 3 is one type of retraining method.

We compare it against two other methods in the next section and present results for various experiments.

We evaluate our technique across a diverse set of datasets and underlying models.

As an initial sanity check, we first evaluate our method on the three most difficult (but low dimensional) bandit tasks analyzed in BID17 .

We next look at two higher dimensional problems and models; one being a Natural Language Processing (NLP) task using a state-of-the-art recurrent model and the other being a vision task using a convolutional model.

In particular, we look at the degree at which each method can achieve both good online performance (regret) and good offline performance (offline test set accuracy), even in the face of large shifts in the data distribution.

We provide additional details on hyperparameters, experimental setup, and dataset information in Appendix 7.4.

All experiments are run 5 times, and reported with the mean and standard error.

We evaluate all the datasets against the baseline presented in BID17 , as well as a variant of our proposed method.

Bandit feedback.

In this setting the models are trained using bandit feedback as the label:Marginalize.

This is our method of marginalizing over all values of the last layer for the neural network training.

Minibatches are sampled from the new data only and the regularization term is computed from the old data.

Sample New.

This baseline creates minibatches using only the newly collected data, optimizing the likelihood of the new data only.

It is equivalent to our method with a regularization constant of zero.

As mentioned in Section 2.1, this method is good at adapting to new data but has a drawback of forgetting the old information.

Sample All.

This is the retraining method presented in BID17 .

In this method, a set number minibatches are created by uniformly sampling from all collected data (both old and new).

SGD gradient updates are then performed using these batches.

This method is slow to adapt but retains older information (refer Section 2.1).Full feedback.

In this setting models are trained using all the labels for the datasets:Batch Train.

When evaluating the offline accuracy, we also give the results for a model that has been trained on the shuffled data in batch mode, with all the labels for the dataset (i.e. full feedback).

This measures how well we could do given we had access to all the labels (instead of just the bandit feedback), and trained in a normal offline setting.

Surprisingly, as we see in some cases, training online with marginalization sometimes performs comparable to training offline.

We first confirm that our method gives good online performance on simpler, but previously studied, problems in BID17 .We present results on the three hardest bandit tasks analyzed in BID17 , the Census, Jester, and Adult dataset.

The bandit problems for these datasets are defined as in previous work:Census and Adult.

Both these datasets are used for multiclass classification problem.

Census dataset has 9 classes whereas Adult dataset consists of 14 classes.

For both datasets the bandit problem is created as follows: for each class we assign an arm, and each arm is associated with a logistic regression (parametrized by a vector) that takes a context as input and returns the expected reward (0 or 1) for selecting the arm.

In the online round we receive a context (feature vector) and pick an arm according to some policy (like UCB), and receive a reward.

Only the picked arm is updated in each round.

Jester BID5 This dataset consists of jokes with their user rating.

For the bandit problem, the model receives a context representing a user along with 8 jokes out of which it is required to make recommendation of 1 joke.

In this setting each joke is defined as a bandit arm.

The problem here is similar to above with the exception that each arm is associated with a linear regression and outputs the predicted user rating for the selected joke.

The reward returned is the actual user rating for the selected joke.

As previously done in BID17 , we use a two layer MLP as the underlying model, using the same configuration across all methods.

For Marginalize and Sample New, we perform the retraining after 1000 rounds.

For Sample All we update after 100 rounds just like BID17 .

In Table 1 we report the average cumulative regret as well as the cumulative regret relative to a policy that selects arms uniformly at random.

We report the results for both Thompson Sampling (TS) and for UCB policies.

Results are similar for either UCB and TS which shows that policies does not influence performance of the training mechanisms.

On most of the tasks both Marginalize (our method) and Sample New outperforms Sample All (method used in BID17 ) in terms of cumulative regret.

Both Marginalize and Sample New techniques are very similar in performance for the three datasets.

All the three datasets used in this experiment are low dimensional, static, and relatively easy to learn, hence there is not much history to retain for Sample New technique.

In the next section we will present results on larger datasets and also evaluate where we will show that our method performs better than Sample New.

Next we evaluate our method with a bigger and more complex underlying model on the NLP domain.

We selected Bilateral Multi-Perspective Matching (BiMPM) (Wang et al., 2017 ), a recurrent model that performs well on several sentence matching tasks, including the paraphrase identification task, to evaluate our method.

The goal of the paraphrase identification task is to determine whether a sentence is a paraphrase of another sentence, i.e., whether they convey the same meaning.

To evaluate whether our algorithm is robust to shifts in data distribution we combined two different paraphrase identification datasets: i) The Quora Question Pairs dataset (Quora), 6 which contains 400,000 question pairs from the QA website Quora.com, and ii) The MSR Paraphrase Corpus (MSR) BID4 , which contains 5,800 pairs of sentences extracted from news articles.

To create an online training dataset we concatenate the MSR training set to a sample of 10,000 examples from the Quora training dataset 7 .We run the online algorithms on this dataset to report the regret values, while we report the offline performance on the MSR and Quora test sets.

We use UCB as our search strategy, as it performs similarly to Thompson sampling and runs much faster in our implementation.

We analyze the following two bandit tasks:Multiclass.

Like the previous datasets, we create a bandit problem by treating each class as an arm parameterized by a vector and the contexts as the individual data instances.

A reward 1 is awarded for identifying correctly if the two sentences in the pair are paraphrase.

For each method, we perform an offline retraining after 1,000 online rounds.

Pool.

Like the multiclass task, the pool based task occurs over a series of rounds.

On each round, the model receives a pool of k(=3) instances, and must select one of them for the user.

After that the model receives a reward based on its selection.

The goal of the model is to learn a scoring function that predicts the expected reward for selecting a certain instance, while at the same time trying to keep regret low.

This setting can be seen as an instance of the bandit problem formulation described in (Russo & Van Roy, 2014) .

In our case, our instances are candidate paraphrase pairs, where the model gets a reward of 1 for returning a valid paraphrase pair, and 0 otherwise.

We use the same implementation and hyperparameters for BiMPM as in (Wang et al., 2017) .

For Marginalize and Sample All, we perform the retraining every 500 rounds.

Sample New performed poorly offline on this setting and is thus updated every 1,000 rounds.

In TAB2 we show that our method Marginalize outperforms both Sample All and Sample New techniques for both multiclass and pool based tasks.

Sample All and Sample New have comparable cumulative regret.

Sample New has worse offline accuracy on Quora dataset (because it forgets old information), while it has better offline accuracy on MSR (because it is able to adapt quicker).

For Batch train, both multiclass and pool based tasks are same-a binary classification problem.

Batch train performs only slightly better than our method in terms of offline accuracy, where Batch train gets full feedback, while our method only gets partial (bandit) feedback.

FIG1 further shows that when the data distribution changes (switching form Quora to MSR in the pool based task) Marginalize and Sample New are able to adapt much faster than Sample All.

Overall Marginalize achieved a lower regret as well as higher offline accuracy for both the bandit settings.

We additionally evaluate our method using a convolutional neural network, which is a common network architecture for computer vision applications.

Table 3 : Image classification Bandit Task results on CIFAR-10 dataset.

We use CIFAR-10 dataset BID9 ) for this experiment.

It is a commonly used dataset for image classification task consisting of 60,000 images from 10 different classes.

Similar to the Quora/MSR tasks, we simulate a domain shift by concatenating together two datasets.

In this case we create two data sets from CIFAR-10 by partitioning the dataset into images depicting animals (6 classes) and images depicting transportation (4 labels).

As above, we analyze two bandit tasks:Multiclass.

We define the multiclass bandit task similarly as above, for each of the 10 classes in CIFAR-10, we assign one arm.

At each round, the model receives an image, guesses a class, and receives feedback (0 or 1) for this class only.

The task is considerably more difficult than the multiclass paraphrase bandit due to the number of classes.

We use 1,000 rounds for the retraining frequency for all methods.

Pool.

We also define a pool based bandit, similar to the pool based paraphrase bandit, with pool size k = 5.

In this case we turn CIFAR-10 info a binary classification task.

We select the two most difficult classes to classify (airplanes and birds, according to the confusion matrix of our base CNN model) in CIFAR-10, and denote these as the positive class.

Like the previous pool task, the learner receives a pool of images and must select one.

A reward of 1 is given for selecting an image belonging to the positive class, 0 otherwise.

As done in the previous pool task, the data is sorted as to simulate a change in domain.

We use a standard convolutional neural network architecture for this task, detailed in Appendix 7.5.

We use 500 rounds for the retraining frequency for all methods.

In Table 3 we present results for the image classification bandit task, using average cumulative regret and offline accuracy as evaluation metrics.

Again, Sample New performs better than Sample All for cumulative regret but under-performs in the offline setting.

As expected, our method performs well for both cumulative regret and offline setting.

For the multiclass task, our method performs significantly lower than batch train.

This is not too surprising, for two reasons: i) Training a CNN architecture takes many more epochs over the data to converge (??? 20 in our case) which is not achieved in a bandit setting; ii) CIFAR-10 has 10 classes, each defining an arm and in our setting; the bandit algorithms only gets feedback for one class in each round, compared to the full feedback received in batch train.

Effectively, the number of labels per class in cut by a factor of 10.

This is not as much an issue in the pool task, where we can see the results between batch train and the bandit algorithms are comparable.

In this paper we proposed a new method for training neural networks in a bandit setting.

We tackle the problem of exploration-exploitation by estimating uncertainty only in the last layer, allowing the method to scale to large state-of-the-art models.

We take advantage of having a posterior over the last layer weights by optimizing the rest of the network over all values of the last layer.

We show that method outperforms other methods across a diverse set of underlying models, especially in online tasks where the distribution shifts rapidly.

We leave it as future work to investigate more sophisticated methods for determining when to retrain the network, how to set the weight (??) of the regularization term in a more automatic way, and its possible connections to methods used for continual learning.

We utilize the same notation here as in Section 3.

First, we show how to arrive at equation (3) from the main objective, equation FORMULA1 .

DISPLAYFORM0 By marginalizing over our zero mean Gaussian prior and using equation FORMULA2 , it follows that the last expectation is approximately a constant, and can be removed.

The second expectation is equal to the negative cross entropy between Q = p(z|D t , D t???1 , ?? t???1 ) and P ?? = p(z|D t???1 , ??).

Using the fact that KL(Q||P ?? ) = CE(Q||P ?? ) ??? H(Q), we can replace the negative cross entropy with ???KL(Q||P ?? ) ??? H(Q), where H(Q) is the entropy of Q which is constant and can be ignored, yielding equation (3).We next derive the KL term.

As mentioned, we approximate the KL term locally around ?? t???1 with a second order Taylor expansion.

Again, let K(??) = KL(p(z|D t , D t???1 , ?? t???1 )||p(z|D t???1 , ??)).

The second order Taylor expansion around ?? = ?? t???1 is: DISPLAYFORM1 We can rewrite K (??) with respect to ?? as: DISPLAYFORM2 This can also be done for K .

Let ??? i indicate either the gradient (i = 1) or the Hessian (i = 2).

Then we can rewrite the above expectation with respect to the distribution P = p(z|D t???1 , ?? t???1 ) instead: DISPLAYFORM3 Using the fact that p(z|D t , D t???1 , ?? t???1 ) = p(D t |?? t???1 ) ???1 p(D t |z, ?? t???1 )p(z|D t???1 , ?? t???1 ), we can rewrite the above as: DISPLAYFORM4 the UCB is straightforward; if ?? is the covariance of our posterior over z, and x is the context, then with probability at least 1 ??? ??, the term (1 + ln(2/??)/2) f ?? (x) T ??f ?? (x) is an upper confidence bound.

The term ?? = (1 + ln(2/??)/2) is treated as a hyperparameter.

The arm c chosen is the one whose parameter vector z c maximizes the following: DISPLAYFORM5

The hyperparameter values that are used for all methods across all tasks (the global hyperparameters) are presented in TAB6 .

The hyper parameters for the low dimensional bandit tasks (we uses the same values for each low dimensional dataset), the paraphrase bandit (multiclass and pool), and the image classification bandit (multiclass and pool) are presented in Table 5 .

The meanings of the non obvious hyperparameter values are described below:Retrain Epochs:

For Sample New and Marginalize; how many times to pass over the new data D t .Update Frequency: How many online rounds to do before updating the rest of the network offline.

Num Updates:

For Sample All; the number of batches to uniformly sample (and update with) from the entire distribution.

Table 5 : Method specific experiment details for all tasks.

In this section we detail the architectures used for each task.

We use the same underlying model for each method.

Low dimensional task models.

For the Low dimensional tasks, we utilize the same exact Multi Layer Perceptron (MLP) models as used in BID17 .

The model is a 2 layer MLP with a hidden layer dimension of 100, with ReLu activation functions.

Paraphrase task.

The paraphrase task uses the same BiMPM architecture and parameters used in Wang et al. (2017) .

We refer readers to the corresponding paper for details.

Image classification task.

The convolutions architecture we use is defined as: (i) Two 3 ?? 3 convolutional layers with 64 filters and ReLu activations; (ii) A 2 ?? 2 max pooling layer with a stride of 2; (iii) Dropout layer with drop probability 0.25; (iv) A 3 ?? 3 and 2 ?? 2 convolutional layer both with 128 filters and ReLu activations; (v) A 2 ?? 2 max pooling layer with a stride of 2; (vi) Dropout layer with drop probability 0.25; (vii) A fully connected layer with 1024 units, followed by a tanh activation, followed by another fully connected layer with 100 units and a tanh activation.

We utilize tanh activations at the end as per Snoek et al. (2015) , who note that ReLu activations lead to difficulties in estimating the uncertainty.

<|TLDR|>

@highlight

This paper proposes a new method for neural network learning in online bandit settings by marginalizing over the last layer