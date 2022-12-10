The goal of survival clustering is to map subjects (e.g., users in a social network, patients in a medical study) to $K$ clusters ranging from low-risk to high-risk.

Existing survival methods assume the presence of clear \textit{end-of-life} signals or introduce them artificially using a pre-defined timeout.

In this paper, we forego this assumption and introduce a loss function that differentiates between the empirical lifetime distributions of the clusters using a modified Kuiper statistic.

We learn a deep neural network by optimizing this loss, that performs a soft clustering of users into survival groups.

We apply our method to a social network dataset with over 1M subjects, and show significant improvement in C-index compared to alternatives.

Free online subscription services (e.g., Facebook, Pandora) use survival models to predict the relationship between observed subscriber covariates (e.g. usage patterns, session duration, gender, location, etc.) and how long a subscriber remains with an active account BID26 BID11 .

Using the same tools, healthcare providers make extensive use of survival models to predict the relationship between patient covariates (e.g. smoking, administering drug A or B) and the duration of a disease (e.g., herpes, cancer, etc.).

In these scenarios, rarely there is an end-of-life signal: non-paying subscribers do not cancel their accounts, tests rarely declare a patient cancer-free.

We want to assign subjects into K clusters, ranging from short-lived to long-lived subscribers (diseases).Despite the recent community interest in survival models BID1 BID33 , existing survival analysis approaches require an unmistakable end-of-life signal (e.g., the subscriber deletes his or her account, the patient is declared disease-free), or a pre-defined endof-life "timeout" (e.g., the patient is declared disease-free after 5 years, the subscriber is declared permanently inactive after 100 days of inactivity).

Methods that require end-of-life signals also include BID23 BID8 BID3 BID14 BID24 BID29 BID31 BID47 BID9 BID19 BID41 BID40 BID17 BID48 BID26 BID0 BID4 BID5 BID35 BID46 BID30 .In this work, we propose to address the lifetime clustering problem without end-of-life signals for the first time, to the best of our knowledge.

We begin by describing two possible datasets where such a clustering approach could be applied.• Social Network Dataset : Users join the social network at different times and participate in activities defined by the social network (login, send/receive comments).

The covariates are the various attributes of a user like age, gender, number of friends, etc., and the inter-event time is the time between user's two consecutive activities.

In this case, censoring is due to a fixed point of data collection that we denote t m , the time of measurement.

Thus, time till censoring for a particular user is the time from her last activity to t m .

Lifetime of a user is defined as the time from her joining till she permanently deletes her account.• Medical Dataset : Subjects join the medical study at the same time and are checked for the presence of a particular disease.

The covariates are the attributes of the disease-causing cell in subject, inter-event time is the time between two consecutive observations of the presence of disease.

The time to censoring is the difference between the time of last observation when the disease was present and the time of final observation.

If the final observation for a subject indicates presence of the disease, then time to censoring is zero.

Lifetime of the disease is defined as the time between the first observation of the disease and the time until it is permanently cured.

We use a deep neural network and a new loss function, with a corresponding backpropagation modification, for clustering subjects without end-of-life signals.

We are able to overcome the technical challenges of this problem, in part, thanks to the ability of deep neural networks to generalize while overfitting the training data BID49 .

The task is challenging for the following reasons:• The problem is fully unsupervised, as there is no pre-defined end-of-life timeout.

While semisupervised clustering approaches exist BID0 BID4 BID5 BID35 BID46 , they assume that end-of-life signals appearing before the observation time are observed; to the best of our knowledge, there are no fully unsupervised approach that can take complex input variables.• There is no hazard function that can be used to define the "cure" rate, as we cannot determine whether the disease is cured, or whether the subscriber will never return to the website, without observing for an infinitely long time.• Cluster assignments may depend on highly complex interactions between the observed covariates and the observed events.

The unobserved lifetime distributions may not be smooth functions.

Contributions.

Using the ability of deep neural networks to model complex nonlinear relationships in the input data, our contribution is a loss function (using the p-value from a modified Kuiper nonparametric two-sample test BID28 ) and a backpropagation algorithm that can perform model-free (nonparametric) unsupervised clustering of subjects based on their latent lifetime distributions, even in the absence of end-of-life signals.

The output of our algorithm is a trained deep neural network classifier that can (soft) assign test and training data subjects into K categories, from high-risk and to low-risk individuals.

We apply our method to a large social network dataset and show that our approach is more robust than competing methods and obtains better clusters (higher C-index scores).Why deep neural networks.

As with any optimization method that returns a point estimate (a set of neural network weights W in our case), our approach is subject to overfitting the training data.

And because our loss function uses p-values, the optimization and overfitting have a rather negative name: p-hacking BID36 .

That is, the optimization is looking for a W (hypothesis) that decreases the p-value.

Deep neural networks, however, are known to both overfit the training data and generalize well BID49 .

That is, the hypothesis (W ) tends to also have small p-values in the (unseen) test data, despite overfitting in the training data (p-hacking).Outline: In section 3, we describe the traditional survival analysis concepts that assume the presence of end-of-life signals.

In section 4, we define a loss function that quantifies the divergence between empirical lifetime distributions of two clusters without assuming end-of-life signals.

We also provide a neural network approach to optimize said loss function.

We describe the dataset used in our experiments followed by results in section 5.

In section 6, we describe a few methods in literature that are related to our work.

Finally, we present our conclusions in section 7.

In this section, we formally define the statistical framework underlying the clustering approach introduced later in this paper.

We begin by defining the datasets relevant to the survival clustering task.

Definition 1 (Dataset).

Dataset D consists of a set of n subjects with each subject u having the following observable quantities DISPLAYFORM0 , S u }, where X u are the covariates of subject DISPLAYFORM1 are the observed inter-event times (disease outbreaks, website usage), q u is the number of observed events of u, and S u is the time till censoring.

Note that the two example datasets described in section 1 fit into this definition.

For instance, in the social network dataset, for a particular user u, X u is a vector of her covariates (such as age, gender, etc.)

, Y u,i is the time between her ith and (i − 1)st activity (login, send/receive comments), and her time till censoring is given by, S u = t m − i Y u,i , where t m is the time of measurement.

Next, we define the lifetime clustering problem applicable to the aforementioned datasets.

Definition 2 (Clustering problem).

Consider a dataset of n subjects, D, constructed as in definition 1.

LetP (U k ) be the latent lifetime distribution of all subjects U k = {u} that belong to cluster k ∈ {1, . . .

, K}. Our goal is to find a mapping κ : X u → {1, . . .

, K}, of covariates into clusters, in the set of all possible such mappings K, that maximizes the divergence d between the latent lifetime distributions of all subjects: DISPLAYFORM2 where U k (κ) is the set of users in U mapped to cluster k through κ, and d is a distribution divergence metric.

κ * optimized in this fashion clusters the subjects into low-risk/high-risk groups.

However, becauseP (U k ) are latent distributions, we cannot directly optimize Eq.(1).

Rather, our loss function must provide an indirect way to optimize Eq.(1) without end-of-life signals.

In what follows, we define the activity process of subjects in cluster k as a Random Marked Point Processes (RMPP).Definition 3 (Observable RMPP cluster process).

Consider the k-th cluster.

The RMPP is DISPLAYFORM3 is the inter-event time between the (i − 1)-st and the i-th activities, X k are the random variable representing the covariates of subjects in cluster k, S k is the time from last event until censoring at cluster k, and A k,i = 0 indicates an activity with an end-oflife signal, otherwise A k,i = 1.

All these variables may be arbitrarily dependent.

This definition is model-free, i.e., we will not prescribe a model for Φ k .Note that, at least theoretically, Φ k continues to evolve beyond the end-of-life signal, but this evolution will be ignored as it is irrelevant to us.

The relative time of the i-th activity of a subject of cluster k, since the subject's first activity, is i ≤i Y k,i , as long as we haven't seen an end-of-life signal, i.e., i <i A k,i = 1.Definition 4 (RMPP Lifetime).

The random variable that defines the lifetime of a subject of cluster k is DISPLAYFORM4 We now define censored lifetimes using Φ k .Definition 5 (RMPP Censored Lifetimes).

The random variable that defines the last observed action time of a subject u of cluster k is DISPLAYFORM5 Let i (S k ) be a random variable that denotes the number of events until the censoring time S k .

The main challenge is not knowing when H k = T k , because we are unable to observed the end-of-life signal A k,i (S k ) = 0.

Clearly, this affects the decision of which subjects have been censored and which have not.

Later, we introduce probability of end-of-life, p : DISPLAYFORM6 , that provides a way around this challenge.

In this section, we review the major concepts in survival analysis that are used in this paper.

Let T u denote the lifetime of a subject u. For now, our description assumes an Oracle that provides end-of-life signals.

Thus, in addition to Ψ u , we assume for each subject u and, another observable quantity, A u,i that denotes whether end-of-life has been reached at the user's ith activity.

In survival applications, A u is typically used to specify if the required event did not occur until the end of study, known as right-censoring.

We shall forego this assumption in subsequent sections and provide a way around the lack of these signals.

Lifetime distribution & Hazard function (Oracle).

Lifetime (or survival) distribution is defined as the probability that a subject u survives at least until time t, DISPLAYFORM0 where F u (t) is the cumulative distribution function of T u .In survival applications, it is typically convenient to define the hazard function, that represents the instantaneous rate of death of a subject given that she has survived until time t.

The hazard function of a subject u is λ u (t) = dFu(t) Su(t) , where dF u is the probability density of F u .

Due to rightcensoring, we do not observe the true lifetimes of the subjects even in the presence of end-of-life signals, A u .

We define the observed lifetime of subject u, H u , as the difference between the time of first event and time of last observed event, i.e., DISPLAYFORM0 Kaplan and Meier (1958) provide a way to estimate the lifetime distribution for a set of subjects while incorporating the right censoring effect.

The Kaplan-Meier estimates of lifetime distribution are given by, DISPLAYFORM1 where DISPLAYFORM2 ) denotes the number of subjects with end-of-life at time j, and r j = u∈D I[H u ≥ j] denotes the number of subjects at risk just prior to time j.

Cox regression model BID12 ) is a widely used method in survival analysis to estimate the hazard function λ u (t) using the covariates, X u , of a subject u. The hazard function has the form, λ(t|X u ) = λ 0 (t) · e {β T Xu} , where λ 0 (t) is a base hazard function common for all subjects, and β are the regression coefficients.

The model assumes that the ratio of hazard functions of any two subjects is constant over time.

This assumption is violated frequently in real-world datasets BID32 .

A near-extreme case when this assumption does not hold is shown in FIG0 , where the survival curves of two groups of subjects cross each other.

Survival Based Clustering Methods (Oracle).

There have been relatively fewer works that perform survival based clustering.

BID3 proposed a semi-supervised method for clustering survival data in which they assign Cox scores BID12 for each feature in their dataset and considered only the features with scores above a predetermined threshold.

Then, an unsupervised clustering algorithm, like k-means, is used to group the individuals using only the selected features.

BID17 proposed supervised sparse clustering as a modification to the sparse clustering algorithm of BID46 .

The sparse clustering algorithm has a modified k-means score that uses distinct weights in the feature set.

Supervised sparse clustering initializes these feature weights using Cox scores BID12 and optimizes the same objective function.

Both these methods assume the presence of end-of-life signals.

In this paper, we consider the case when end-of-life signals are not available.

We provide a loss function that quantifies the divergence between survival distributions of the clusters, and we minimize said loss function using a neural network in order to obtain the optimal clusters.

Our goal is to cluster the subjects into K clusters ranging from low-risk to high-risk by keeping the empirical lifetime distributions of these K groups as different as they can be, while ensuring that the observed difference is statistically significant.

In this section, we assume there are no end-of-life signals.

We introduce a loss function that is based on a divergence measure between empirical lifetime distributions of two groups, and at the same time takes into account the uncertainty regarding the end-of-life of the subjects.

Instead of a clear end-of-life signal, we specify a probability for each subject u that represents how likely her last observed activity coincides with her end-of-life.

Definition 6 (Probability of end-of-life).

Given a dataset D (Definition 1), we define a function, by an abuse of notation, p(X u , S u ) → [0, 1] that gives a probability of end-of-life of each subject u.

Divergence measures like Kullback-Leibler divergence and Earth-Mover's distance that do not incorporate the empirical nature of the given probability distributions are not appropriate for our task as they do not discourage highly imbalanced groups FIG0 .

This motivates the use of twosample tests that allow for the probability distributions to be empirical.

Logrank test BID34 BID37 BID7 ) is commonly used to compare groups of subjects based on their hazard rates.

However, the test assumes proportional hazards (section 3) and will not be able to find groups whose hazard rates are not proportional to each other FIG0 .

BID16 introduced Modified Kolmogorov-Smirnov (MKS) statistic that works for arbitrarily right-censored data and does not assume hazards proportionality.

But MKS suffers from the same drawback as the standard Kolmogorov-Smirnov statistic, namely that it is not sensitive to the differences in the tails of the distributions.

In this paper, we use p-value from the Kuiper statistic BID28 which extends the Kolmogorov-Smirnov statistic to increase the statistical power of distinguishing distribution tails BID45 .Definition 7 (Optimization of Kuiper loss).

Given a dataset D (Definition 1), we define a loss L(κ, p) where, by an abuse of notation, κ(X u ) → [0, 1] is a mapping that performs soft clustering of subjects into two clusters 0 & 1 by outputting a probability of a subject belonging in cluster 0, and p(X u , S u ) → [0, 1] is a function that gives a probability of end-of-life of a subject in D. Our goal is to obtainκ where the loss function DISPLAYFORM0 DISPLAYFORM1 returns the logarithm of a p-value from the Kuiper statistic BID38 , with DISPLAYFORM2 and DISPLAYFORM3 , and for k = 0, 1, DISPLAYFORM4 where DISPLAYFORM5 The following theorem states a few properties of our loss function.

Theorem 1 (Kuiper loss properties).

From Definition 7, consider two clusters with true lifetime distributionsP (U 1 ) andP (U 2 ).

Assume an infinite number of samples/subjects.

Then, the loss function defined in equation FORMULA12 has the following properties:(a) If the two clusters have distinct lifetime distributions, i.e.

P ( DISPLAYFORM6 (b) If the two clusters have the same stochastic process Ψ u (Definition 1), Ψ u = Ψ v , for any two subjects u and v, regardless of cluster assignments, then ∀κ, p, L(κ, p) → 0.We prove Theorem 1 in Appendix 8.1 by defining the activity process of the subjects using shifted Random Marked Point Processes.

The loss defined above solves all the aforementioned issues; a) does not need clear end-of-life signals, b) use of a p-value forces sufficient number of examples in both groups, c) does not assume proportionality of hazards and works even for crossing survival curves, and d) accounts for differences at the tails.

In this section, we describe the functions κ(·) and p(·) in definition 7 κ(·) gives the probability of a subject u being in cluster 0, and we define it using a neural network as follows, DISPLAYFORM0 where DISPLAYFORM1 are the weights and the biases of a neural network with L − 1 hidden layers, X u are the covariates of subject u, φ is an activation function (tanh or relU in our experiments), and σ is the softmax function.

An example of a feedforward neural network that optimizes Kuiper loss is shown in FIG1 .Next, we describe the form of end-of-life probability function, p(·).

We make the reasonable assumption that p(·) is an increasing function of S u .

For example, consider two subjects a and b, with last activities one year and one week before their respective time of censoring.

Clearly, it is more likely that subject a's activity is her last one than that b's activity is her last one.

In our experiments, we also assume that p(·) only depends on S u , and not on the covariates X u .

Survival tasks commonly use the following naive technique to identify the end-of-life signal.

They define p(·) using a step function, p(X u , S u ) = 1[S u > W ], where W is the size of an arbitrary window from the time of censoring (see FIG1 .

However, this approach does not allow learning of the window size parameter W , and hence, the analysis can be highly coupled with the choice of W .We remedy this by choosing p(·) to be a smooth non-decreasing function of S u , parameters of which can be learnt by minimizing the loss function L(κ, p).

We use the cumulative distribution function of an exponential distribution in our experiments, i.e, p(X u , S u ) = 1 − e −β·Su FIG1 .

The rate parameter, β, is learnt using gradient descent along with the weights of the neural network.

Extension to K Clusters Until now, we dealt with clustering the subjects into two groups.

However, it is not hard to extend the framework for K clusters.

We increase the number of units in the output layer from 2 to K. As before, a softmax function applied at the output layer gives probabilities that define a soft clustering of the subjects into K groups.

These probabilities can be used to obtain the loss, L A,B , between any two groups, D A and D B .We define the total loss for K groups as the average of all the pairwise losses between individual groups and get the geometric mean of the pairwise p-values, i.e., DISPLAYFORM2 In other words, the loss L 1...K is minimized only if each of the individual p-values are low indicating that each group's lifetime distribution is different (in divergence) from every other group's lifetime distribution.

Implementation We implement a feedforward neural network in Theano (Theano Development Team, 2016) and use ADAM BID27 to optimize the loss L 1...

K defined in equation 10.

Each iteration of the optimization takes as input a batch of subjects (full batch or a minibatch), generates a value for the loss, calculates the gradients, and updates the parameters (β, DISPLAYFORM3 ).

This is done repeatedly until there is no improvement in the validation loss.

We use L2 regularization over the weights and experiment with different values for the regularization parameter.

We also experiment with different neural network sizes (number of hidden layers, number of hidden units), activation functions for the hidden layers, and weight initialization techniques.

We applied different deep learning techniques like batch normalization BID22 and dropout to better learn the neural network.

In this paper, we analyze a large-scale social network dataset collected from Friendster.

After processing 30TB of data, originally collected by the Internet Archive in June 2011, the resulting network has around 15 million users with 335 million friendship links.

Each user has profile information such as age, gender, marital status, occupation, and interests.

Additionally, there are user comments on each other's profile pages with timestamps that indicate activity in the site.

In our experiments, we only use the data up to March 2008 as Friendster's monthly active users have been significantly affected with the introduction of "new Facebook wall" BID39 .

From this, we only consider a subset of 1.1 million users who had participated in atleast one comment, and had specified their basic profile information like age and gender.

We make our processed data available to the public at location (anonymized).

We build the dataset D : DISPLAYFORM0 , S u } (Definition 1) for our clustering task as follows.

X u : We use each user's profile information (like age, gender, relationship status, occupation and location) as features.

For the high-dimensional discrete attributes like location and occupation, we use 20 most frequently occurring values.

To help improve the performance of competing methods, we also construct additional features using the user's activity over the initial 5 months (like number of comments sent and received, number of individuals interacted with, etc.).

In total, we construct 60 features that are used for each of the models in our experiments.

Y u,i : We define Y u,i as the time between u's ith comment (sent or received) and (i − 1)st comment (sent or received).

q(u) is then defined as the total number of comments the user participated in.

S u : We calculate S u as the time between u's last activity and the chosen time of measurement (March 2008).

We experimented with different neural network architectures as shown in TAB1 .

In Table 1 , we show the results for a simple neural network configuration with one fully-connected hidden layer with 128 hidden units and tanh activation function.

We use a batch size of 8192 and a learning rate of 10 −4 .

We also use batch normalization BID22 to facilitate convergence, and regularize the weights of the neural network using an L2 penalty of 0.01.

Appendix 8.2 shows a more detailed evaluation of different architecture choices.

We evaluate the models using 10-fold cross validation as follows.

We split the dataset D randomly into 10 folds, sample 100,000 users without replacement from ith fold for testing and sample 100,000 users similarly from the remaining 9 folds for training.

We use 20% of the training data as validation data for early stopping in our neural network training.

We repeat this for i ranging from 1 to 10.We compare our clustering approach with the only two survival-based clustering methods in literature; a) Semi-supervised clustering BID3 ) and b) Supervised sparse clustering BID17 .

Since both these methods require clear end-of-life signals, we use an arbitrary window of 10 months (i.e., a pre-defined "timeout") prior to the time of measurement in order to obtain these signals FIG1 .

We also try window sizes of 0 months (only the users with activity at t m are censored) and 5 months, and obtain similar results (not reported here).

We test our approach in two cases -in the presence and lack of end-of-life signals.

In the former case, we optimize the loss function in equation FORMULA2 keeping p(·) fixed to the end-of-life signals obtained from using a window of 10 months.

In the latter case, our approach learns the latent end-of-life signals.

We also experiment with a loss function based on the Kolmogorov-Smirnov statistic (denoted NN-KS) and report the performance for the same.

We evaluate the clusters obtained from each of these methods using concordance index.

Concordance Index Concordance index or C-index BID18 ) is a commonly used metric in survival applications BID1 BID33 to quantify a model's ability to discriminate between subjects with different lifetimes.

It calculates the fraction of pairs of subjects for which the model predicts the correct order of survival while also incorporating censoring.

We use the end-of-life signals calculated using a pre-defined "timeout" of 10 months.

Rather than populating all possible pairs of users, we sample 10000 random pairs to calculate the C-index.

Table 1 shows the C-index values for the baselines and the proposed method.

Table 1 : C-index (%) for clusters from different methods with K = 2, 3, 4 where K is the number of clusters.

The proposed approach is more robust with lower values of standard deviations than the competing methods.

DISPLAYFORM0 Discussion The proposed neural network approach performs better on average than the two competing methods.

Even without end-of-life signals, the proposed approach achieves comparable scores for K = 3, 4 and the best C-index score for K = 2.

Although NN-Kuiper is theoretically more robust than NN-KS because of its increased statistical power in distinguishing distribution tails BID45 , we do not observe a performance difference in the Friendster dataset.

Further, we use the endof-life signals obtained using a window of 10 months to calculate the empirical lifetime distributions of the clusters identified by the neural network ( FIG2 ).

The empirical lifetime distributions of clusters seem distinct from each other at K = 2 but not at K = 3, 4.

In addition, there is not significant gain in the C-index values as we increase the number of clusters from K = 2 to K = 4.

Hence, we can conclude that there are only two types of users in the Friendster dataset -long-lived and short-lived.

Majority of the work in survival analysis has dealt with the task of predicting the survival outcome especially when the number of features is much higher than the number of subjects BID47 BID9 BID19 BID41 .

A number of approaches have also been proposed to perform feature selection in survival data BID24 BID29 .

In the social network scenario, BID43 tried to predict the relationship building time, that is, the time until a particular link is formed in the network.

Many unsupervised approaches have been proposed to identify cancer subtypes in gene expression data without considering the survival outcome BID13 BID2 BID6 .

Traditional semi-supervised clustering methods BID0 BID4 BID5 BID35 do not perform well in this scenario since they do not provide a way to handle the issues with right censoring.

Semi-supervised clustering BID3 and supervised sparse clustering Witten and Tibshirani (2010a) use Cox scores BID12 to identify features associated with survival.

They treat these features differently in order to perform clustering based on the survival outcome.

Although there are quite a few works on using neural networks to predict the hazard rates of individuals BID14 , to the best of our knowledge, there hasn't been a work on using neural networks for a survival-based clustering task.

Recently, Alaa and van der Schaar (2017) proposed a nonparametric Bayesian approach for survival analysis in the case of more than one competing events (multiple diseases).

They not only assume the presence of end-of-life signals but also the type of event that caused the end-of-life.

BID33 optimize a loss based on Cox's partial likelihood along with a penalty using a deep neural network to predict the probability of survival at a point in time.

Here we considered the task of clustering subjects into low-risk/high-risk groups without observing any end-of-life signals.

Extensive research has been done on what is known as frailty analysis, for predicting survival outcomes in the presence of clustered observations BID20 BID10 BID21 .

Although frailty models provide more flexibility in the presence of clustered observations, they do not provide a mechanism for obtaining the clusters themselves, which is our primary goal.

In addition, our approach does not assume proportional hazards unlike most frailty models.

In this work we introduced a Kuiper-based nonparametric loss function, and a corresponding backpropagation procedure (which backpropagates the loss over clusters rather than the loss per training example).

These procedures are then used to train a feedforward neural network to inductively assign observed subject covariates into K survival-based clusters, from high-risk to low-risk subjects, without requiring an end-of-life signal.

We showed that the resultant neural network produces clusters with better C-index values than other competing methods.

We also presented the survival distributions of the clusters obtained from our procedure and concluded that there were only two groups of users in the Friendster dataset.

Both parts (a) and (b) of our proof need definition 3 that translates the observed data D u for subject u into a stochastic process.

Proof of (a): If the two clusters have distinct lifetime distributions, it means that the distributions of T 0 and T 1 in eq. (2) are different.

Then, either the right-censoring δ in eq. (3) does not allow us to see the difference between T 0 and T 1 , and then there is no mappingsp andκ that can get the distribution of S 0 (t;κ,p) and S 1 (t;κ,p) to be distinct, implying an L(κ, p) → 0, as n → ∞ as the observations come from the same distribution, making the Kuiper score asymptotically equal to one; or δ does allow us to see the difference and then, clearlyp ≡ 0 with a mappingκ that assigns more than half of the subjects to their correct clusters, which would allow us to see the difference in H 0 and H 1 , would give Kuiper score asymptotically equal to zero.

Thus, L(κ, p) → −∞, as n →

∞.Proof of (b): Because κ only take the subject covariates as input, and there are no dependencies between the subject covariates and the subject lifetime in eq. (2), any clustering based on the covariates will be a random assignment of users into clusters.

Moreover, from eq. (3), the censoring time of subject u, S u , has the same distribution for both clusters because the RMPPs are the same.

Thus, H 0 d = H 1 , i.e., H 0 and H 1 have the same distributions, and the Kuiper p-value test returns zero, L(κ, p) → 0, as n → ∞. Table 4 : C-index (%) over different learning rates and batch sizes for the proposed NN approach with Kuiper loss (with learnt exponential) and K = 2.

<|TLDR|>

@highlight

The goal of survival clustering is to map subjects into clusters. Without end-of-life signals, this is a challenging task. To address this task we propose a new loss function by modifying the Kuiper statistics.