Knowledge Graph Embedding (KGE) has attracted more attention in recent years.

Most of KGE models learn from time-unaware triples.

However, the inclusion of temporal information beside triples would further improve the performance of a KGE model.

In this regard, we propose LiTSE, a temporal KGE model which incorporates time information into entity/relation representations by using linear time series decomposition.

Moreover, considering the temporal uncertainty during the evolution of entity/relation representations over time, we map the representations of temporal KGs into the space of multi-dimensional Gaussian distributions.

The mean of each entity/ relation embedding at a time step shows the current expected position, whereas its covariance (which is stationary over time) represents its temporal uncertainty.

Experiments show that LiTSE not only achieves the state-of- the-art on link prediction in temporal KGs, but also has the ability to predict the occurrence time of facts with missing time annotations, as well as the existence of future events.

To the best of our knowledge, no other model is capable to perform all these tasks.

Knowledge Graphs (KGs) are being used for gathering and organizing scattered human knowledge into structured knowledge systems.

YAGO (Suchanek et al., 2007) , NELL BID3 , DBpedia BID0 and Freebase BID1 are among existing KGs that have been successfully used in various applications including question answering, assistant systems, information retrieval, etc.

In these KGs, knowledge can be represented as RDF triples (s, p ,o) in which s (subject) and o (object) are entities (nodes), and p (predicate) is the relation (edge) between them.

KG embedding attempts to learn the representations of entities and relations in high-dimensional latent feature spaces while preserving certain properties of the original graph.

Recently, KGE has become a very active research topic due to the wide ranges of downstream applications.

Different KGE models have been proposed so far to efficiently learn the representations of KGs and perform KG completion as well as inferencing BID2 BID22 BID23 BID20 BID5 .Most of existing KGE models solely learn from time-unknown facts and ignore the useful temporal information in KGs.

In fact, there are many time-aware facts (or events) in some temporal KGs.

For instance, (Obama, wasBornIn, Hawaii) happened at August 4, 1961, and (Obama, presidentOf, USA) was true from 2009 to 2017.

These temporal KGs, e.g. ICEWS BID9 , YAGO3 BID11 , store such temporal information either explicitly or implicitly.

Traditional KGE models such as TransE learn only from time-unknown facts and consequently cannot distinguish relations with similar semantic meaning.

For instance, they often confuse relations such as wasBornIn and diedIn when predicting (person,?,location) .To tackle this problem, Temporal KGE models BID4 BID6 BID18 encode time information in their embeddings.

Temporal KGE models outperform traditional KGE models on link prediction over temporal KGs.

It justifies that incorporation of time information can further improve the performance of a KGE model.

Some existing temporal KGE models encode time information in a latent space e.g. representing time as a vector BID4 BID10 .

These models cannot capture some prop-erties of time information such as the length of time interval as well as order of two time points.

Moreover, some exiting temporal graph embedding models BID18 BID19 consider the changes of entity representations over time as a kind of temporal evolution process, while they ignore the uncertainty during the temporal evolution.

We argue that the evolution of entity representations has randomness, because the features of an entity at a certain time are not completely determined by the past information.

For example, (Steve Jobs, diedIn, California) happened on 2011-10-05.

The semantic characteristics of this entity should have a sudden change at this time point.

However, due to the incompleteness of knowledge in KGs, this change can not be predicted only according to its past evolutionary trend.

Therefore, the representation of Steve Jobs is supposed to include some random components to handle this uncertainty, e.g. a Gaussian noise component.

To address the above problems, we propose a new temporal KGE model based on linear time series decomposition (LiTSE) that captures the evolution process of KG representations.

LiTSE fits the evolution process of an entity or relation as a linear function of time with a Gaussian random noise.

Inspired by , our approach represents each entity and relation as a multi-dimensional Gaussian distribution at each time step to introduce a random component.

The mean of an entity/relation representation at a certain time step indicates its current expected position, which is obtained from its initial representation, its evolutionary direction vector which represents the long-term trend of its evolution and the current time.

The covariance which describes the temporal uncertainty during its evolution, is denoted as a constant diagonal matrix for computing efficiency.

Our contributions are as follows.• Learning the representations for temporal KGs is a relatively unexplored problem because most of existing KGE models only learn from time-unknown facts.

We propose LiTSE, a new KGE model to incorporate the time information into the KG representations.• Different from the previous temporal KGE models which use time encoding to incorporate time information, LiTSE fits the evolution process of KG representations as a linear function of time.

This enables us to observe and predict the time information directly from entity/relation representations.

In particular, we can predict the occurrence of a fact in a future time, according to the known evolution trends of KG representations learned from the past information.• We specially consider the temporal uncertainty during the evolution process of KG representation.

Thus, we model each entity as a Gaussian distribution at each time step and use KL-divergence between two Gaussian distributions to compute the scores of facts for optimization.• Beside performing link prediction in temporal KGs, our models are proved to be capable of estimating the occurrence time of a fact with missing time annotation, and predicting future events.

The rest of the paper is organized as follows: Section 2 reviews related works.

Our model is introduced in the section 3.

The proposed model is evaluated and compared with state-of-the-art models in the section 4.

Finally, the paper is concluded in the last section.

A large amount of research has been done in KGE.

These approaches can generally be categorized into two groups, namely semantic matching models and transnational distance models .

RESCAL BID13 and its extensions, e.g. DistMult BID23 , ComplEx BID20 , ConvE BID5 , are the semantic matching models.

These models measure plausibility of facts by matching latent semantics of entities and relations embodied in their vector space representations.

A few examples of translational distance models include TransE BID2 , TransH BID22 , TransD .

These models measure the plausibility of a fact as the distance between the two entities, usually after a translation carried out by the relation.

Particularly, KG2E takes into account the uncertainties of KG representations and represents entities and relations as random vectors drawn from multivariate Gaussian distributions.

KG2E scores a fact by measuring the distance between the distributions of the entities and the relation.

The above methods achieve good results on link prediction in KGs.

Moreover, some re-cent researches illustrate that the performances of KGE models can be further improved by incorporating time information in temporal KGs.

TAE BID17 imposes temporal order constraints on time-sensitive relation pairs, e.g. BornIn and wasDiedIn, where the prior relation is supposed to lie close to the subsequent relation after a temporal transition.

TAE only uses temporal order information between relations, but not the exact time information in facts.

TTransE BID10 propose scoring functions which incorporate time representations into a TransE-type score function in different ways.

BID6 utilizes recurrent neural networks to learn time-aware representations of relations and uses standard scoring functions from the existing KGE model, e.g. TransE BID2 and DistMult BID23 .

HyTE BID4 encodes time in the entity-relation space by associating a corresponding hyperplane to each timestamp.

The above three methods represent each time step as a latent feature vector or a hyperplane matrix and update the entity/relation representations at different time steps with the corresponding time representations.

That means the all entity/relation representations have the same evolution trend.

In contrast, Know-Evolve BID18 ) models the temporal evolution of each entity representation as an individual temporal point process, which is a non-linear function of time.

They exploit recurrent neural network to capture the dynamic characteristics of entity representations.

Know-Evolve is also proved to perform well on time prediction.

In this paper, we fit the temporal evolution of entity/relation representations by deploying linear time series decomposition.

This enables us to directly observe and predict time information from entity/relation representations.

We can also predict the future events according to the evolution trends of entity/relation representations.

Moreover, inspired by KG2E , we map the entity and relation representations in a space of multi-dimensional Gaussian distributions to model the uncertainty of temporal KGs.

Different from KG2E, we focus on the temporal embedding and give a specific definition of the uncertainty in temporal KGs, i.e. the randomness during the temporal evolution of KG representations.

In this section, we present a detailed description of our proposed method, LiTSE, which not only uses relational properties between entities in triples but also incorporate the associated temporal meta-data by using linear time series decomposition.

A time series is a series of time-oriented data.

Time series analysis is widely used in many fields, ranging from economics and finance to managing production operations, to the analysis of political and social policy sessions BID12 .

An important technique for time series analysis is time series decomposition.

This technique decomposes a time series into four components, including a trend component, a cyclical component, a seasonal component and an irregular component (i.e. "noise").In our method, we regard the evolution of an entity/relation representation as a linear time series and assume that it only consists of two components, i.e. a linear trend component and a Gaussian noise component.

The motivation of this assumption is based on the following three points.• The simplicity of the embedding model architecture.

Considering a temporal KG consisting of thousands of entities and relations, we can avoid introducing too many parameters by only using a trend component and a noise component to fit the temporal evolution of each relation/entity representation.• The capability of time prediction.

Based on our proposed assumption, our model is capable to estimate the occurring time of a triple with the missing time annotation.• The efficiency of model training.

Commonly, a moving-average model (MA model) is used when modeling the irregular term of a time series BID12 .

But we have to deploy a global optimization algorithm while training a MA model.

Instead, we take a Gaussian noise as the irregular component in time series decomposition.

This method enables us to employ mini-batch training for efficiency purposes.

To incorporate temporal information into traditional KGs, a new temporal dimension is added to fact triples, denoted as a quadruple (s, p, o, t).

It represents the creation of relationship edge p between subject entity s, and object entity o at time step t. The score term x spot = f t (e s , r p , e o ) can represent the conditional probability or the confidence value of this event x spot , where e s , e o ∈ R Le , r p ∈ R Lr are representations of s, o and p. In the case of a fact (s, p, o, [t s , t e ]), we consider it to be a positive triple for each time step between t s and t e .

t s and t e denote the start and end time during which the triple (s, p, o) is valid.

At each time step, the time-specific representations of an entity e i or a relation r i should be updated as e i,t or r i,t .

In order to avoid information redundancy, we only incorporate time information into entity representations or relation representations, but not both.

The model where time information is incorporated into relation representations is denoted as LiTSER.

Another model with evolving entity representations is called as LiT-SEE.

Thus, the score of a quadruple (s, p, o, t) can be represented as x spot = f e (e s,t , r p , e o,t ) or x spot = f r (e s , r p,t , e o ).

Due to the similarity between LiTSEE and LiTSER, we take LiTSEE as an example to describe our method in this section.

In our proposed model LiTSEE, we first utilize a linear function to fit the evolution processes of entity representations as: DISPLAYFORM0 where the e i is the time-independent latent representation of the ith entity which is subjected to ||e i || 2 = 1, the coefficient α i denotes its evolutionary rate, and the vector w i represents the direction of its evolution which is restricted to ||w i || 2 = 1.

For LiTSEE, we use the following translationbased scoring function to measure the plausibility of a fact (s, p, o, t) BID2 .

DISPLAYFORM1 where ||r p || 2 = 1.

Furthermore, to model the temporal uncertainty of the latent representations of the subject and the object in this fact, we assume e s,t and e o,t have randomness and obey Gaussian probability distributions: P s,t ∼ N (e s,t , Σ s ) and P o,t ∼ N (e o,t , Σ o ).

Similarly, the predicate is represented as P r ∼ N (r p , Σ r ).

The mean vectors e and r, and covariance matrix Σ indicate the corresponding embedding representations for the Gaussian distribution.

This advanced model based on LiTSEE is denoted as LiTSEE G .Similarly to LiTSEE, we consider the transformation result of LiTSEE G from the subject to the object to be akin to the predicate in a positive fact.

We use the following formula to express this transformation: P s,t − P o,t , which corresponds to the probability distribution P e,t ∼ N (µ e,t , Σ e ).

Here, µ e,t = e s,t − e o,t and Σ e = Σ s + Σ o .

As a result, combined with the probability of relation P r ∼ N (r p , Σ r ), we measure the similarity between P e,t and P r to score the fact.

KL divergence is a straightforward method of measuring the similarity of two probability distributions.

We optimize the following score function based on the KL divergence between the entity-transformed distribution and relation distribution (Yu et al., 2013 ).

DISPLAYFORM2 where, tr(Σ) and Σ −1 indicate the trace and inverse of the covariance matrix, respectively.

Considering the simplified diagonal covariance, we can compute the trace and inverse of the matrix simply and effectively for LiTSEE G .

The gradient of log determinant is et al., 2008) .

We can compute the gradients of Equation 3 with respect to the time-independent latent feature vectors, evolutionary direction vectors and covariance matrix (here acting as a vector) as follows: DISPLAYFORM3 DISPLAYFORM4 where DISPLAYFORM5 spot = Σ −1 r (r p + e o − e s + t(α o wIn the same way, we can extend LiTSER to LiTSER G by adding a Gaussian noise component into the linear evolution function of each relation/entity representation.

The architectures of LiTSER and LiTSER G are similar to LiTSEE and LiTSEE G .

The processes of computing gradients of the score functions in LiTSEE G and LiTSER G are also alike.

Therefore, it is unnecessary to go into details about LiTSER and LiTSER G here.

As mentioned in section 3.1, our proposed models are translation-based.

Thus, we train our models by minimizing the margin-based ranking loss.

DISPLAYFORM0 where, [T ] is the set of time steps in the temporal KG, D + t is the set of positive triples with time stamp t and D − t is the set of negative examples.

In this paper, we not only generate negative samples by randomly corrupting subjects or objects of the positives such as (s , p, o, t) and (s, p, o , t), but also add extra negative samples (s, p, o, t ) which are present in the KG but do not exist in the subgraph for a particular time BID4 .

We use this time-dependent negative sampling approach for time prediction.

In the other hand, to compare our model with baseline models fairly, we use uniform negative sampling method BID2 for link prediction and future event prediction.

To avoid overfitting, we add some regularizations while learning the Gaussian embedding.

As described in Section 3.1, the norms of the original representations of entities and relations, as well as the norms of all evolutionary direction vectors, are restricted by 1.

Besides, the following constraint is considered for covariance when we minimize the loss L : DISPLAYFORM1 where, E and R are the set of entities and relations respectively, c min and c max are two positive constants.

During training process, we use Σ l ← max(c min , min(c max , Σ l )) to achieve this regularization for diagonal covariance matrices.

These constraints for the mean and covariance are also considered during initialization.

To show the capability of LiTSE, we compare it and its extensions with other state-of-the-art baselines on link prediction.

Particularly, we also evaluate our method for two other tasks: time prediction and future event prediction, which baseline models are not capable to handle.

To compare our model with baselines, we used the following three datasets, namely ICEWS14, ICEWS05-15 and YAGO11K, released by Dasgupta et al. FORMULA0 and BID6 .

ICEWS14 and ICEWS05-15 are subsets of Integrated Crisis Early Warning System (ICEWS) BID9 .

ICEWS is a repository that contains political events with specific time annotations, e.g. The statistics of the datasets are listed in Table 1.

We compare our method and other baselines by performing link prediction on ICEWS14, ICEWS05-15 and YAGO11k D .

Specially, we also evaluate the performance of our proposed models on time prediction and future event prediction with two event-based datasets, ICEWS14 and ICEWS05-15.

In this paper, we report the experimental results on three tasks: Link Prediction, Time Prediction and Future Event Prediction.

We split anew the facts into training, validation and test in a proportion of 80%/10%/10%.

All facts in the test set occur after facts in the training/validation set.

We train the model with the training set and judge whether quadruples in the test set are positive or not.

The decision process is similar to triple classification BID15 : for a fact (s, p, o, t), if x spot is below a relation-specific threshold δ r , then positive; otherwise negative.

The thresholds δ r are determined on the validation set.

For link prediction task, we compare our method with several state-of-the-art KGE models and existing time-wise KGE models, including TransE BID2 , DistMult BID23 , KG2E , ComplEx BID20 , ConvE BID5 , TTransE BID10 ,TA-TransE and TA-DistMult (García-Durán et al., 2018) as well as HyTE BID4 .

All these baselines are not applicable to estimating time information by computation (HyTE did it by ranking, which is much more time-consuming).

Therefore, we compare the results among our proposed models for time prediction.

Considering that the above time-wise KGE models are not capable to represent a future time step, we compare our models with the above static KGE models for future event prediction.

We implemented our models and baseline models in PyTorch, except TA-TransE and TA-DistMult.

Since some implementation details of these two models were unclear, we report their results from the original paper BID6 .

We used Adagrad optimizer to train all the implemented models and selected the optimal hyperparameters by early validation stopping according to MRR on the validation set.

We restricted the iterations to 5000.

For all the models, the batch size b = 512 was kept on both the datasets.

We tuned the embedding dimensionalities d in {50, 100}, the learning rates lr in {0.001, 0.01, 0.1} and the ratio of negatives over positive training samples η in {1, 3, 5, 10}. For translation-based models, the margins γ were varied in the range {1, 2, 3, 5, 10}. For semantic matching models, the regularizer weights λ were chosen from the set {0.001, 0.01, 0.1}. For ConvE, we select dropout parameters from the set {0, 0.2}. Similar to the setting in KG2E , we selected the pair of restriction values c min and c max for covariance among {(0.005, 0.5), (0.01, 1), (0.03, 3), (0.05, 5)} for Gaussian embedding models.

The default configuration for our proposed models is as follows: lr = 0.1, η = 10, γ = 1.

Below, we only list the non-default parameters.

For LiTSEE, the optimal configuration is as follows: lr = 0.01, γ = 10 on YAGO11k D .

For Table 2 : Link prediction results (filtered setting).

Rows 1-7: basic models with no time information.

Rows 8-11: models which encode information.

* indicates results in this row were taken from (García-Durán et al., 2018).

Dashes: results could not be obtained.

The best results among all models are written bold.

The red numbers are the best results obtained from our implemention.

LiTSEE G , the optimal configuration is as follows: γ = 2, (c min , c max ) = (0.01, 1) on ICEWS14; (c min , c max ) = (0.01, 1) on ICEWS05-15; lr = 0.01, γ = 10, (c min , c max ) = (0.005, 0.5)] on YAGO11k D .

For LiTSER, the optimal configuration is as follows: γ = 2 on ICEWS14; lr = 0.01, γ = 10 on YAGO11k D .

For LiTSER G , the optimal configuration is as follows: (c min , c max ) = (0.005, 0.5) both on ICEWS14 and ICEWS05-15;lr = 0.01, γ = 10, (c min , c max ) = (0.005, 0.5)] on YAGO11k D .

The above configurations were used for all three tasks.

The obtained results for different tasks are based on the above mentioned experimental setup.

Table 2 shows the results for link prediction task.

In ICEWS14 and ICEWS05-15, LiTSEE G outperformed all embedding models considering MRR, Hits@10 and Hits@1.

TransE implemented by BID6 got the best MR in these two datasets.

It is noteworthy that the ratio of negatives over positive samples η used in (García-Durán et al., 2018) was 500, much higher than our setting.

BID20 investigated the influence of η on KGE models and discovered that increasing η could lead to better results.

Thus, the results obtained from BID6 would become worse if the same η as ours was used.

Except the results obtained from BID6 BID5 and BID20 showed that the performances of ConvE and ComplEx were remarkably better than KG2E on static KGs, e.g. FB15k and WN18 BID2 .

These results prove that modeling temporal uncertainty in temporal KGs by mapping KG representations into the space of multi-dimensional Gaussian distribution substantially improve the performances of KGE models on temporal KGs.

As mentioned in Section 3.2, we corrupted time information in positive facts to generate negative samples (s, p, o, t ) for time prediction.

TAB5 shows the results of our proposed models for time prediction on ICEWS14 and ICEWS05-15.

Beside Mean Absolute Errors (MAEs), we also report the proportions of testing examples which prediction errors are under 10 days, denoted as Error@10.

Although MAEs of our models were high due to a small part of bad predictions, 57.5% of prediction errors of LiTSEE G on ICEWS14 were under 10 days, which proves the ability of our method for time prediction.

We introduce LiTSE, a temporal KGE model that incorporates time information into KG representations by using linear time series decomposition.

LiTSE fits the temporal evolution of KG representations over time as linear time series, which enables itself to estimate time information of a triple with the missing time annotation and predict the occurrence of a future event.

Considering the uncertainty during the temporal evolution of KG representations, LiTSE maps the representations of temporal KGs into the space of multi-dimensional Gaussian distributions.

The covariance of an entity/relation representation represents its randomness component.

Experimental results demonstrate that our method significantly outperforms the state-of-the-art methods on link prediction and future event prediction.

Besides, our method can effectively predict the occurrence time of a fact.

Our work establishes a previously unexplored connection between relational processes and time series analysis with a potential to open a new direction of research on reasoning over time.

In the future, we will explore to use other time series analysis techniques to model the temporal evolution of KG representations.

Along with considering the temporal uncertainty, another benefit of using time series analysis is to enable the embedding model to encode temporal rules.

For instance, given two quadruple (s, p, o, t p ) and (s, q, o, t q ), there exists a temporal constraint t p < t q .

Since the time information is represented as a numerical variable in a time series model, it is feasible to incorporate such temporal rules into our models.

We will investigate the possibility of encoding temporal rules into our proposed models.

DISPLAYFORM0 regularize the covariances for each entity and relation with constraint 6.

18. end loop TAB10 shows the statistics of datasets which are anew split for future event prediction, denoted as ICEWS14-F and ICEWS05-15F.

As mentioned in Section 4.2, all of the facts in test set occur after the facts in training set and validation set, and the facts of validation set occur after the facts in training set.

The time spans of training sets, validation sets and test sets of ICEWS14 and ICEWS05-15 are reported in TAB10 .

t e represents the end time of the dataset.

For instance, t e of the training set of ICEWS14 is 2014/10/20 and t e of the validation set of ICEWS14 is 2014/11/22, which means the time stamps of quadruples in the validation set of ICEWS14 are between 2014/10/21 and 2014/11/22.

In TAB1 , we summarize the scoring function of baselines and our models and compare their space complexities.

x, y, z = i x i y i z i denotes the tri-linear dot product; * denotes the convolution operator; Seq denotes a LSTM network; P t denotes the temporal projection for embeddings; w t denotes the embedding for the time step t. As shown in TAB1 , our models have the same space complexities as traditional KGE models.

On the other hand, the space complexities of TTransE and HyTE will be much higher than our models if n t is larger than n e and n r .

Comparison of our models with baseline models for space complexity.

n e , n r and n t are numbers of entities, relations and time steps.

We borrow some notations from BID5 for simplicity.

@highlight

Submitted in EMNLP