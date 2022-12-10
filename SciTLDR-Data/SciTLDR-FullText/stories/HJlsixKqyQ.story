Supervised learning problems---particularly those involving social data---are often subjective.

That is, human readers, looking at the same data, might come to legitimate but completely different conclusions based on their personal experiences.

Yet in machine learning settings feedback from multiple human annotators is often reduced to a single ``ground truth'' label, thus hiding the true, potentially rich and diverse interpretations of the data found across the social spectrum.

We explore the rewards and challenges of discovering and learning representative distributions of the labeling opinions of a large human population.

A major, critical cost to this approach is the number of humans needed to provide enough labels not only to obtain representative samples but also to train a machine to predict representative distributions on unlabeled data.

We propose aggregating label distributions over, not just individuals, but also data items, in order to maximize the costs of humans in the loop.

We test different aggregation approaches on state-of-the-art deep learning models.

Our results suggest that careful label aggregation methods can greatly reduce the number of samples needed to obtain representative distributions.

This paper explores the problem of label aggregation in domains that are highly subjective, i.e., where different annotators may disagree for perfectly legitimate reasons.

Such settings are common, if underacknowledged.

Though increasingly, mass media provides stories about the unintended consequences of ignoring this diversity in machine learning.

For example, Beauty.ai sponsored a worldwide beauty contest, judged by a machine learning algorithm.

Though light-skinned entrants made up the majority of entrants, they nonetheless won a disproportionate number of contests.

BID0 Tay, a Twitter-based learning agent, developed by Microsoft, was taught to tweet that the Holocaust was made up 2 (though the Holocaust factually existed, the same cybersocial dynamics of training bias found in subjective domains led to this outcome).

ProPublica discovered that Northpointe risk assessment software-used to help judges determine sentence length for convicts-recommended longer sentences for African-American men than other groups, even when controlled for confounding factors.

BID2 X:2 Fig. 1 .

In this example, data items (black dots) are labeled by five human annotators each (left), where color indicates label choice, yielding an empirical label distribution y i for each data item i. By clustering similarly labeled objects, we pool together (right) the labels of all data items assigned to the same cluster k into a single, much larger sample θ k for all items in the cluster.

Our research suggests that, in some cases, this larger sample (or a mixture of cluster samples) is a better representation of the true population distribution of beliefs about each data item in the cluster and can lead to better predictive supervised learning.

Learning a distribution of beliefs about a data item, rather than a single "ground truth" label, poses unique challenges.

It increases the dimensionality of the learning problem so that more data items may be needed.

It also may require more labels per item to get a representative sample of the human populations' beliefs.

And for most problems, labels are relatively expensive to obtain.

Though crowdsourcing platforms have made this task convenient, they are frequently a resource bottleneck in supervised learning loops.

Our main contribution is a method for minimizing the number of labels needed to learn to predict socially representative label distributions.

It is based on the hypothesis that the sources are subjectivity are limited, and so the number of distinct distributions of beliefs over all data items is likewise limited.

In other words, the label distributions are samples from a relatively small number of true, but hidden, distributions.

See Figure 1 .

These hidden distributions can be seen as latent classes representing population-level beliefs about the labels.

According to this hypothesis, we can use unsupervised clustering algorithms to pool together the labels of data items with similar distributions into higher resolution distributions of beliefs shared commonly among all data items in the same cluster.

In particular, we: (1) explore subjectivity as the problem of learning representative distributions from a target population of responses to target questions, BID1 propose clustering as a sensible means for pooling together labels from similar data items, to reduce the number of labels needed (3) test what we call our clustering hypothesis, that the label distributions of subjective data are clustered around a small number of underlying, true distributions (4) study how different label aggregation strategies and representations affect the performance of state-of-the art deep learning predictors.

It would seem that bias is an inherent part of any information reduction process, such as those found in statistical learning BID29 .

So it seems naive to expect that machines can learn unbiased models through unsupervised learning alone, or even for any supervised learning that assumes a singular, correct answer to most problems.

We hope that this research sparks a broader debate about the best practices for machine learning with humans in the loop.

The rest of this paper is organized as follows.

Section 2 describes our experimental workflow, Section 3 presents our results, Section 4 discusses our study, Section 5 presents related work, and Section 6 is the conclusion.

Figure 2 describes the basic experimental workflow in this study.

We discuss each phase below.

Note that there are two testing phases, one for determining how well each aggregation method fits the data and another for how well supervised learning algorithms trained by each aggregation strategy perform.

Since these test phases share some methods, we discuss them together at the end of the section.

Fig. 2 .

The basic experimental workflow involves obtaining crowdsourced labels for raw data (yielding empirical label distributions for each data item), trying various strategies for aggregating and pooling those labels (including no aggregation), and finally testing how each method affects the accuracy of machine learning prediction.

Note there are two testing phases: one for how well each aggregation strategy fits the data and one for machine learning performance.

We also list important terms, keywords, and abbreviations associated with each phase of the workflow.

We performed extensive experiments, through the aggregation phase, on a number of publicly available, human-labeled datasets BID0 BID1 BID22 BID30 BID38 , including the data described below.

However, due to space and time constraints-and because our preliminary studies suggested that these sets covered most of the features of the other sets and exhibited representative performance, we decided to report and focus this study on two of them.

Table 1 summarizes the basic properties of these sets, which we now describe in detail.

Before conducting this research, we consulted with our institutional review board, who determined that it did not fall under federal or institutional guidelines as human subjects research.

Nonetheless, we took extra precautions to ensure the privacy of all human-generated data.

For the twitter data, we replaced all mentions with "@SOMEONE" and URLs with "URL, " paraphrased all examples, and adhered to Twitter's developer policy.

BID3 2.1.1 Job-themed data.

We obtained directly from Liu et al. BID23 a corpus of machine-filtered job-related tweets (i.e., Twitter posts).

From this corpus, we randomly selected 2,000 tweets to acquire human annotations from two popular crowdsourcing platforms-Amazon Mechanical Turk 5 (abbreviated as MT ) and CrowdFlower 6 (CF).

For each tweet and each platform, we asked five crowdworkers to answer three questions (see FIG0 .

We provided contextual information in the form of the three tweets proceeding and succeeding the target tweet made by the target user.

Suicide 2000 4 124 13175 6.59 0.27 0.17 Table 1 .

Basic properties of the label sets we use.

Density indicates the average number of labels per data item.

"MVTD" (majority-voted-true-class deviation) and "RMSD" (root-mean-square deviation) are two divergence measures for estimating the uncertainty of different label sets, motivated by the literature on scale and outliers BID17 BID34 BID45 .

MVTD is the average (over all data items) weight of the weight of the most frequent label: DISPLAYFORM0 RMSD is the L2 deviation from the average label: This resulted six distinct label sets, one for each choice of platform and question, where each question and each platform, each data item has labels from five crowdworkers.

We additionally constructed three additional label sets by combining the labels from both crowdsourcing platforms (denoted BOTH), so that each tweet has ten labels.

DISPLAYFORM1

We obtained another data set of 2000 tweets, filtered for suiciderelated discourse BID22 .

Labels come from 122 CrowdFlower workers and 2 suicide prevention domain experts.

For each tweet, five crowdworkers chose the label that described its content from four possible choices: A. Suicidal thoughts, B. Supportive messages or helpful information, C. Reaction to suicide news/movie/music and D. Others.

Experts were invited to the second stage to annotate the tweets without unanimous labels from five crowdworkers.

Thus tweet can have up to 7 labels, from crowdworkers and experts.2.1.3 Data splits.

Due to the expense of obtaining detailed samples from the populations of crowdworkers, we used two different train/dev/test splits.

Broad split.

We randomly split each 2,000-tweet dataset into 1000/500/500 train/dev/test sets.

Deep split.

We randomly split the job-related dataset only into 1500/540/50 train/dev/test sets.

For each item in the 50-item held-out test set, we obtain 50 additional labels from new AMT crowdworkers.

For a given data set with items i ∈ {1, . . .

, n} and label choice j ∈ {1, . . .

, d }, let y i j denote the number of crowdworkers who select label j for data item i.

Thus y i is a distribution over labels for data item i.

We will sometimes abuse notation and also use y i to denote the probability distribution obtained by normalizing the label distribution.

Aggregation composes two substages: clustering (including no clustering) and reduction, which depends on whether or not no clustering is the strategy used.

We discuss this case first.

Majority.

Typically, when annotators disagree on which label is best for a data item x i , majority voting is used to determine a single gold-standard label:ŷ i = arg max DISPLAYFORM0 Repeated.

This strategy assumes each (data item, label) is a separate data item, e.g., if three annotators choose to label 'A.' then we make three identical copies of the data in each training epoch.

The model effectively weighs each choice by the number of times it is selected, with the goal of learning a single label, and treats each empirical label distribution as a Bayesian model of the degree of belief in each label choice.

Probability.

This is a baseline method for predicting population distributions over label choices.

Instead of training on a single label choice for each data item, it uses a d-dimensional vector representing the distribution y i of labels for data item i as a probability distribution (which by abuse of notation we also call y i ).

It effectively treats each empirical label distribution as a frequentist sample of the true distribution of beliefs (though it crucially does not capture the degree of belief labels, either individually or collective).

These associate with each data item a probability distribution z i over a finite number p of clusters, i.e., a mixture of models, over the space of empirical label distributions y i .

According to our main hypothesis, pooling labels by cluster reveals the true label distributions underlying our empirical distributions, thus amplifying the labeling power of each crowdworker.

We can thus associate with each cluster k ∈ {1, . . .

, p} a distribution θ k over the label choices.

This is simply the cluster centroid if the strategy has one (like MMM and GMM below), or the weighted average (θ k = i z ik y i /n) of the labels (as in DS and LDA below; we call them "centroids" in either case).

Our goal is to improve prediction accuracy by replacing each empirical label y i with one based on its cluster likelihoods z i and cluster-wide label distributions θ k .We consider five different clustering strategies: the multinomial mixture model (MMM), the Gaussian mixture model 7 (GMM), Dawid and Skene's model BID7 for selecting labels conditioned on annotator accuracy (DS) BID8 and latent Dirichlet allocation 9 (LDA) BID4 .

We wrote our own MMM from scratch.

We get two distinct strategies from LDA by, in addition to clustering over empirical labels, also clustering on bag-of-word representations of each data item's text, i.e., as LDA is most commonly used.

Though rather elementary, these models collectively provide an informative experimental basis for testing our central hypothesis, i.e., that label distributions in subjective domains are clustered around a finite number of true label distribution.

According to this hypothesis, the model that best describes subjective domains should be MMM since it is a generative model where each centroid is defined as a distribution of which each cluster item is a sample. (By contrast, the centroid of GMM has a very different generative interpretation-i.e., as a parameter of a multivariate Gaussian distribution-even though in both models they are the (weighted) means of their respective cluster items.)Although DS and LDA are not, strictly speaking, clustering models, we can easily obtain clusterlike latent classes-along with likelihood estimates-by integrating over the users (for David and Skene) or the data items (for LDA).

Moreover, both models provide useful comparisons to our true clustering models.

In particular, DS is widely-used in collaborative filtering settings, of which this can be seen as an example.

This model incorporates labeler accuracy and is effective in settings where a labeler provides many examples.

In our setting, which uses microtask crowdworkers, anyone labeler only provides ten or so examples (see Table 1 ), and so we would not expect this model to fit our data especially well.

LDA is very similar to MMM, though it is more commonly used, in part because it tends to be a better fit, both hypothetically and empirically, for more problems, but also because estimating prior distributions is computationally more efficient (in our case, we sidestep and use a maximum likelihood estimator for MMM, but not for LDA).

The main difference between these models is in how data is generated.

In MMM an empirical label distribution is assumed to come from choosing a cluster, then choosing all samples from that one chosen cluster.

In LDA we choose a new cluster for each sample.

If each cluster represented the beliefs of an individual, this might make sense, especially if we had a lot of data about the labeling preferences of individual labelers.

However, since that is not the case in our setting, and since we are assuming that each cluster represents the distribution of beliefs across society, MMM makes more sense as the best model to fit our hypothesis.

Except for DS, each model requires the number of clusters p as a hyperparameter.

We considered all values for p between roughly half and twice the number of label choices for each question.

We investigated several model selection strategies, including some of the methods described below, and discovered that numbers they provided were roughly correlated.

Furthermore, many of these strategies were designed for specific models or are based on strong prior assumptions.

We ultimately chose the native likelihood function of each model, because we felt it provided the most externally consistent strategy for choosing the best p within each clustering strategy, even though it cannot really be used to compare models from different families.

As the estimators for these models are stochastic and/or sensitive to initial conditions, for every model and every choice of hyperparameters, we ran 100 trials on the training data and chose the model with the highest estimated likelihood.

Cluster-based reduction strategies.

We use these strategies to replace each empirical label distribution y i with one based on the cluster centroids {θ k } and the likelihood of i belonging to each cluster z i .

Maximum a posteriori (Max) selection replaces y i with the most likely cluster centroid θ k : k = arg max k z ik and expected distribution (Avg) replaces y i by integrating out the clusters k z ik θ k .

Note that the integration step we use to produce aggregate distributions from LDA or DS essentially applies the Avg reduction to each model and that the Max reduction does not have a reasonable interpretation for these models (other than selecting the most likely label, which we can do more directly by simply not clustering).

Yet understanding the performance differences between these two reduction strategies stands to yield important insights into the clustering hypothesis.

If the clusters can discover the true representative label distribution underlying each empirical distribution, then we would expect predictive models to perform better using the Max strategy for training data, as it commits to a single distribution.

Since LDA and DS cannot support both reductions, and thus deny us this important observation, we used only MMM-and GMM-based aggregations (to which either reduction can apply) as inputs to the supervised learning phase.

We built various text-based supervised classifiers based on a single convolutional neural network (CNN) architecture as illustrated in FIG1 , using Keras with a Tensorflow back end.

The differences among the model inputs are rooted in the aggregation strategies used.

CNNs have been used for various sentiment analysis and topic categorization tasks BID19 and proved effective across a wide range of corpora.

Each takes the text of a tweet as input and outputs a predicted label distribution.

The CNN architecture we use consists of an input layer composing concatenated pre-trained word embeddings, a convolutional layer with numerous filters, a max-pooling layer which captures the most significant feature, and a softmax classifier which outputs the probability distribution over labels/classes.

We tested this supervised approach with various label aggregation strategies to obtain the ground truth labels, including clustering approaches, in our text classification experiments.

The hyper-parameter settings of the CNN architecture depend on the splits of datasets.

We use the GloVe pre-trained word embeddings trained particularly on a Twitter corpus with 2B tweets BID33 .

We set the vector size of the word embeddings as 100 through our experiments.

In our text pre-processing step, we keep the most common 20,000 words and pad the sentence up to 1,000 tokens.

We use the Adam optimizer to minimize the loss function BID20 .

We set the batch size as 32 and the number of epochs to train the model as 25.

For each data item i, we now have three associated probability distributions: the empirical distribution of labels y i , the likelihood distribution over the clusters z i , and a new label distribution from the aggregation phase w i .For the sake of using the clustering strategies to test our main hypothesis, we argue that the set of distributions θ 1 , . . .

, θ p is a good fit for the hypothesis if, in addition to maximizing likelihood, the entropy over the cluster likelihoods H (z i ) is less than that of the empirical distributions H (y i ).

However, these entropies cannot be directly compared because the number d of alternatives in the label set may be different from the number p of clusters.

So we normalize by dividing by the logarithm of the number of items in each distribution.

We call this the entropy gap (EG): DISPLAYFORM0 This score applies to any label aggregation model or clustering approach that has likelihoods associated with each (data point, cluster) pair and where each point can be interpreted as a probability distribution.

The danger with this score is that it is easy to "cheat" to get a good score, say, by assigning all data items to the same cluster.

Since, however, we select our models based on maximum likelihood, we use this metric honestly here.

Another useful, and standard test is the Kullback-Leibler divergence, which measures how one probability distribution diverges from a second one BID21 .

For discrete probability distributions P (say, y i ) and Q (say, w i , or, later, the label predicted by the CNN) it is: DISPLAYFORM1 We also use KL divergence to evaluate the performance of the CNN model (entropy gap does not make sense here).

In addition, KL1 measures the divergence from the CNN predicted probability to the empirical distribution of labels y i , and, when clustering is used.

KL2 measures the divergence from the CNN predicted probability to the label distribution from the aggregation phase w i .

Additionally, Score is the loss (cost) function-categorical cross entropy-used to train the CNN.

Accuracy measures how often the prediction have the maximum probability in the same class as the true value does.

Note that KL divergence and cross entropy are standard tools for comparing probability distributions, while accuracy requires us to convert each distribution into a single scalar label.

Table 2 , 3, 4 show the performance results for each X T est of datasets in Table 1 using the best model selected by the likelihood criterion.

Since we only had 50 data items with 50 extra labels, we tried clustering them visually using histograms.

FIG3 shows that the labels do appear to group clearly into seven clusters.

We describe the tweets that fall into each cluster.

Group 1 (Red) distributions have most of their mass on label choices Getting hired/job seeking and None of the above, but job-related.

Here, all the tweets in this group were talking about plans to get a job (e.g., really want a job, dont put that on ur resume for a minimum wage job), or the process of getting a job.

In contrast, Group 2 (cyan) has almost all the mass exclusively on Getting hired/job seeking (e.g., got the job).

The third group (brown) clusters around Complaining about work and Going to work, suggesting a topic about complaining about having to go to work.

Group four (green) are a set of tweets complaining about work while at work.

Groups five and six (blue and orange) have most of their labels on None of the above, but job-related and Not job-related.

Group six (where Not job-related was more frequent than None of the above) were mostly about road work.

Group five (where None of the above was more frequent and complicated.

It seemed to contain cases where work was mentioned, but was central to the other topics (e.g., TODAY AT WORK I Broad split CL MMM LDA l ab el GMM LDA t ex t FMM DPMM jobQ1CF TAB11 Table 2 .

Numbers of clusters for the optimal label aggregation model we achieved on each dataset using two splits.

"CL": Number of clusters in the best model.

LEARNED ABOUT...) or used "work" or "job" metaphorically, though there exist some clear None of the above, but job-related tweets, like Perks of working overnight: donuts fresh out of the fryer.

In Table 5 -10, we show the score, accuracy, and KL divergence metrics for a series of CNN-based text classifiers for the job (Broad split: 5-7, Deep split: 8-10) and suicide datasets built with different label aggregation approaches.

Among the aggregation methods tested, MMM and LDA had the best KL scores (Table 4) .

Since this metric is the best honest score for testing fitness across models, and since the MMM and LDA are better fits for the clustering hypothesis, this seems to partially support the hypothesis.

This would seem to suggest that LDA is a better model for the underlying space of label distributions, and one reason for this could be because the labels indeed depend on independent classes of labelers (or possibly even individual labelers, as witnessed by the somewhat unexpectedly good performance of DS).

Note that another explanation could be that even in the MMM model the clusters have a substantial enough amount of uncertainty (captured by the w i distributions) and that averaging over this uncertainty leads to better predictions.

This is a common phenomenon, even in situations where the data is known to be generated from a single model; that is, maximum a posteriori estimates often underperform fully Bayesian ones.

We were surprised by how much worse GMM performed compared to the other methods (except LDA t ex t , which uses a different feature set than the others and so is a priori an outlier), as for Table 3 .

Entropy gap obtained using the optimal label aggregation model on each dataset using two splits.

"EG": Normalized entropy gap (i.e., the average entropy gap per data item).

The highest EG for each dataset is highlighted in bold.large samples GMM and MMM rather similar.

However, the sample sizes (number of labels) we use here are normal for many supervised learning tasks, and at this scale, the differences appear to be significant.

Regarding EG (Table 3) : that GMM and DS tend to outperform the other models is not too surprising, given that EG is not honest (see discussion in the testing subsection).

And we expected LDA to perform poorly on this metric, due to the fact that, under LDA, most empirical distributions are drawn from multiple clusters.

Thus we would expect the cluster likelihood distribution to have higher entropy than in the MMM model (which assumes all labels are drawn from a single cluster).

Starting again with KL divergence TAB11 ), CNNs trained and tested on MMM AV G outperform all other models most of the time, with no-clustering, probability-based CNNs a close second.

GMM Avд has some very good and very bad results, and the relative dominance of MMM AV G recedes when the deep label distributions are used for evaluation.

Together, these results show that learning over the entire distribution of labels is feasible and that using clustering to aggregate labels sometimes results in better performance.

What was not expected (though consistent with our clustering hypothesis tests, where LDA outperformed MMM) is that MMM Avд outperforms MMM Max .

Better MMM Max performance would seem to be more consistent with the hypothesis that the clustering algorithm can discover the true underlying label distributions.

Instead, MMM Avд draws from each of the predicted ground truth label distributions, yielding distributions that very similar in construction to those produced by LDA.

Table 4 .

KL divergence obtained using the optimal label aggregation model on each dataset using two splits.

"KL": Kullback-Leibler divergence.

The lowest KL divergence for each dataset is highlighted in bold.

Broad split Score majority repeated probability Table 5 .

Scores of CNN-based text classification experiments with different aggregation models, using the Broad split.

The lowest score for each dataset is highlighted in bold.

DISPLAYFORM0 Among the clustering models, as expected, KL2 outperforms KL1, and this supports our hypothesis by showing that principled aggregation processes are effective for training and prediction.

Also of interest are the accuracy tests (Tables 6 and 9 ).

Since this test requires the model to produce a single "best" label, and since clustering is used here for preserving diversity in the label distributions, we expected the clustering methods to underperform the no-clustering methods.

Among the no-clustering methods, majority can be seen as a the standard approach of learning a single label for each data item, while probability attempts to learn the frequentist, empirical labels, even though (for the purpose of the accuracy test) it only reveals one label.

Repeated is implicitly a Bayesian approach.

Except for jobQ3BOTH-new, majority and probability give nearly the same performance, which suggests that modeling the underlying distribution, even without clustering, 8 10 12 2 4 6 8 10 12 2 4 6 8 10 12 2 4 6 8 10 12 2 4 6 8 10 12 2 4 6 8 10 12 2 4 6 8 10 12 2 4 6 8 10 12 2 4 6 8 10 12 Choices for Question 3Number of Labels Broad split ACC majority repeated probability Table 6 .

Accuracy of CNN-based text classification experiments with different aggregation models, using the Broad split.

The highest accuracy for each dataset is highlighted in bold.

DISPLAYFORM1 generally does not degrade the accuracy of single-label models.

That repeated underperforms the other perhaps reflects the reality that each empirical distribution represents a sample of population beliefs, rather than degree of belief.

One important and obvious limitation of this work is that uncertainty in human labeling is caused by many things other than subjectivity, including data encoding errors and communication ambiguities BID2 BID7 BID47 , lack of sufficient information BID5 BID7 BID14 , and unreliable annotators and their bias BID14 .

We do not attempt to quantity whether the uncertainty we observe is due to these other causes or to subjectivity (i.e., varying user perspectives).

We hope to explore this avenue in future work.

In order to truly understand the social impact of representative learning, we need to know the underlying demographics of the sampling frames in question, in this case AMT and CrowdFlower.

Broad split KL1/2 majority repeated probability Table 7 .

Kullback-Leibler divergence of CNN-based text classification experiments with different aggregation models, using the Broad split.

The lowest KL divergence for each dataset is highlighted in bold.

DISPLAYFORM2 Deep split Score majority repeated probability Table 8 .

Scores of CNN-based text classification experiments with different aggregation models, using the Deep split.

The lowest score for each dataset is highlighted in bold.

DISPLAYFORM3 Deep split ACC majority repeated probability Table 9 .

Accuracy of CNN-based text classification experiments with different aggregation models, using the Deep split.

The highest accuracy for each dataset is highlighted in bold.

DISPLAYFORM4 Several studies have investigated these demographics BID9 BID10 BID18 BID39 .

Among the findings: Mechanical Turk pulls most of its workforce from the United States, whereas CrowdFlower's workforce has proportionally higher levels of participation from smaller countries, like Venezuela.

The male to female ratio is similar on both platforms with more female workers than male.

A majority of contributors have some college education, of which most have a bachelor's degree.

The worker population on both platforms is dynamic and changes frequently, but the number of workers available is steady, so every year some new workers join and balance the workers who quit contributing.

The majority of workers fall in the legal working age in the US, most of which are young workers of age group 20-35.

These workers earn below the median salary range in the US.

The American

Deep split KL1/2 majority repeated probability racial composition is mostly white.

According to Ellie et al. BID32 workers speak a diverse set of languages.

According to Huff and Tingley BID15 those working as office and administrative support are major contributors to AMT.

DISPLAYFORM0

It is common in supervised learning settings to model data labels as probability distributions, as we do here, though the similarities are somewhat superficial.

In most machine learning problems these probabilities are Bayesian, meaning that the distributions represent uncertainty or degree of belief.

In sharp contrast, our label probabilities are frequentist (though the some of the model probabilities used for clustering are Bayesian), i.e., they literally represent an estimate of the frequency of events (i.e., labels chosen) in a population sample.

As mentioned in the discussion, there are many sources for uncertainty when humans in the loop are concerned BID16 BID26 BID27 BID40 BID41 .

However, most such studies into this matter assume that there is an underlying, if unknown, true label for each data item and do not account for the subjective nature of human comprehensions and beliefs, i.e., more than one answer is reasonably correct and acceptable.

Two broad research areas overlapping with our subjective domain research question include recommender systems and multi-label learning problems BID12 .Recommender systems BID3 study the tastes and preferences of individuals, typically in online commercial settings.

The goal of such systems is to personalize the shopping, viewing, or playing experience of the users of such system, and they rely on copius amounts of data on the users and in grouping users into groups with similar tastes.

Here we are interested in how populations beliefs, not tastes, vary, and although modeling users and group of users is of interest to us (particularly to distinguish between different sorts of expertise on the annotation domain), in many annotation setting, such as in crowdsourcing, little information on the annotators may be available.

Multilabel classification BID12 BID13 BID25 BID28 BID31 BID35 BID36 BID36 BID37 BID42 BID43 BID44 BID46 allows for each data item to simultaneously belong to multiple classes BID6 BID25 .

However, it is possible for there to be multiple valid labels, even when there is no disagreement among labelers.

It is often important to know when multiplicity is due to disagreement, especially when such disagreements fall along key demographic boundaries, and indicate important but opposing perspectives that should be equally preserved in the predictive model.

Multilabel models are not designed to detect such disagreement.

Rather, they are designed to detect a rich collection of labels, individualized to each data item, and with no frequentist representation of the diversity of underlying population beliefs.

By contrast, we seek to throw disagreement into high relief by assuming that label sets fall into a small number of stereotypical classes, which can be discovered through clustering in the space of label distributions.

We study the problem of learning to predict the underlying diversity of beliefs present in supervised learning domains.

We compare the performance of predictive models that are trained on the empirical distribution of labels produced by crowdworkers to those that collapse those labels to a single ground truth value.

Our results show that it is feasible to predict such distributions over labels.

Doing so is an important first step in producing intelligent agents that understand the diversity of beliefs in society.

We also studied the use of clustering to pool and aggregate labels in order to reduce the costs of labeling in this richer domain.

Our results suggest that such methods are effective, and though the reason may have to do with the underlying sources of subjectivity being limited, more research is needed to understand why.

This paper provides a substantial framework of models and tests to further explore this question and others and advance though rigorous testing and evaluation socially-aware intelligent systems.

Indeed, our results suggest a number of next steps.

For one, we regret not using LDA-based distributions in the supervised learning phase, since they seemed to perform so well in the aggregation phase.

We also need to explore more powerful variants of MMM, including the standard fully Bayesian variant, Dirichlet-multinomial mixtures, and the standard nonparametric variant, Dirichlet process multinomial models.

This project was motivated by the need for active learning methods that are socially aware, and recognizing that the there was very little research in this area to build on.

We hope to incorporate the lessons learned here into new active learning query strategies that make learning socially representative labels even more efficient.

@highlight

We study the problem of learning to predict the underlying diversity of beliefs present in supervised learning domains.