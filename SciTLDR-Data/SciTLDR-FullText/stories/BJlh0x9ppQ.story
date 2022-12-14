Knowledge bases (KB) are often represented as a collection of facts in the form (HEAD, PREDICATE, TAIL), where HEAD and TAIL are entities while PREDICATE is a binary relationship that links the two.

It is a well-known fact that knowledge bases are far from complete, and hence the plethora of research on KB completion methods, specifically on link prediction.

However, though frequently ignored, these repositories also contain numerical facts.

Numerical facts link entities to numerical values via numerical predicates; e.g., (PARIS, LATITUDE, 48.8).

Likewise, numerical facts also suffer from the incompleteness problem.

To address this issue, we introduce the numerical attribute prediction problem.

This problem involves a new type of query where the relationship is a numerical predicate.

Consequently, and contrary to link prediction, the answer to this query is a numerical value.

We argue that the numerical values associated with entities explain, to some extent, the relational structure of the knowledge base.

Therefore, we leverage knowledge base embedding methods to learn representations that are useful predictors for the numerical attributes.

An extensive set of experiments on benchmark versions of FREEBASE and YAGO show that our approaches largely outperform sensible baselines.

We make the datasets available under a permissive BSD-3 license.

Knowledge Bases (KBs) are playing an increasingly important role in a number of AI applications.

KBs can be seen as a collection of facts or triples of the form (head, predicate, tail) , denoted as (h, p, t) , where head and tail correspond to entities and predicate corresponds to a relationship that holds between these two entities.

This structured information is easily accessible by AI systems to enhance their performance.

A variety of AI applications such as recommender systems, natural language chatbots or question answering models, have benefited from the rich structural information archived in these repositories.

This is because much of human knowledge can be expressed with one or more conjunctions of knowledge facts.

However, KBs' capabilities are limited due to their incompleteness 1 .

Consequently there has been a flurry of research on knowledge base completion methods in recent years.

Relationship extraction BID27 (i.e., classification of semantic relationship mentions), knowledge graph matching BID32 BID12 (i.e., alignment and integration of entities and predicates across KBs), or search-based question-answering BID36 (i.e., queries issued to a web search engine) are a few different ways to address the incompleteness problem.

However, the literature on the so-called link prediction methods BID22 has received more attention in the last few years in comparison to the aforementioned approaches.

Contrary to other solutions, link prediction methods aim to find missing links between entities exclusively based on the existing information contained in the KB.

This is achieved by ranking entities that are answer candidates for the query.

The queries these methods typically address are of the form (USA, /location/contains, ?), or (Madrid, /location/capitalOf, ?), whereas the missing element -represented by a question mark-is an entity contained in the KB.Many link prediction methods only harness feature types learned from the rich relational information contained in the KB to infer new links, and only very recently Niepert, 2018, Pezeshkpour et al., 2018] numerical attributes have been integrated along with other feature types to improve link prediction performance.

Similarly, numerical information is also represented as facts such as (Berlin, /location/latitude, 52.31) or (Albert Einstein, /person/birth year, 1879).

However, as shown in BID5 ] the application of numerical attributes is limited because of the same incompleteness problem: Many entities are missing numerical attribute values they are expected to possess.

For example, entities that represent locations should have numerical information regarding latitude, longitude or area, among others; whereas for entities representing people, numerical predicates such as the birth year, weight or height would be more appropriate.

In this work we focus on the problem of completing queries where the relationship is a numerical predicate.

Consequently, the answer to this new type of query is a numerical value.

This is contrary to the link prediction problem, wherein the answer to a query is always an element of a closed vocabulary.

Examples of queries addressed in this paper are (Apple Inc., revenue, ?) or (California, average salary, ?).

While one can interpret link prediction as a classification/ranking problem, this is rather a regression problem.

The main contributions of this paper are:??? We introduce the problem of predicting the value of entities' numerical attributes in KBs.

For the sake of simplicity we term this as 'numerical attribute prediction problem'.

To our knowledge, this is the first time this problem is addressed in the literature.??? We create benchmark datasets for this problem.

We use well-known subsets of Freebase and Yago as the blueprints for creating these benchmarks.

We also create versions of these datasets for different percentages of sparsity by artificially removing facts that involve numerical predicates.

All these benchmark datasets will be made publicly available.??? We propose two meaningful baselines for this problem.

These baselines are inspired by previous work done in the node classification and the imputation literature.??? We propose supervised and semi-supervised approaches to this problem.

The semisupervised approaches significantly outperform the baselines in all datasets and conditions.

The paper is organized as follows: We discuss the related work in Section 2.

Afterwards we formalize the problem of predicting numerical values for entities' numerical attributes in KBs in Section 3.

We describe our approaches to this problem, as well as the two baselines.

Section 4 reports the experimental setting followed by an extensive set of experiments on different datasets with different degrees of sparsity in Section 5.

Finally, we summarize the conclusions of our study in Section 6.

There is an extensive body of work on link prediction BID2 BID37 BID22 BID34 .

Logical approaches [Russell and Norvig, 2016] operate on a set of logical rules that are usually handcrafted and/or mined.

These logical formulas are evaluated between entity pairs to generate feature representations which are then used for a downstream machine learning model.

On the other hand, KB embedding methods BID22 learn feature representations -embeddings-for all elements in a KG by optimizing a designed scoring function.

Given a fact, these scoring functions output a score that relates to the likelihood of that fact being true.

A popular and successful instance of KB embedding method is TransE BID2 , where predicates are modeled as translations in the entity embedding space.

Much less work has been done on entity-type classification BID19 , Yogatama et al., 2015 .

This problem is inherently related to link prediction, since it amounts to complete queries of the form (head, typeOf, ?) , where the question mark corresponds to a certain entity type (e.g. location, artist, ...).

Therefore, link prediction and entity-type classification share certain similarities with the numerical attribute prediction problem.

Most importantly, they all make use of the relational information for KB completion, one way or another.

However, there is a crucial difference between link prediction and numerical attribute prediction.

In the former, a query can be completed with one or several elements contained in a relatively small vocabulary, whereas in the later the answer may (potentially) take an infinite number of real values.

There is another line of research related to our work, namely value imputation BID28 .

In statistics, imputation is the process of replacing missing data with substituted values.

In the simplest case, one can replace the missing values of a variable by the mean of all existing values of that variable.

This technique is called mean imputation.

It preserves the mean of the variable, but alters the underlying variable distribution to be more peaked at the mean BID0 .

However, it is the most commonly practiced approach for value imputation BID29 , and it has been shown to be competitive for a number of downstream tasks BID1 Moore, 2017, Malone et al., 2018] .

Another popular approach is called regression imputation, where the missing values of a variable are estimated by a regression model from the observed values of other variables.

There is some work on using text for predicting numerical attributes of entities such as BID3 Rappoport, 2010, Gupta et al., 2015] .

BID9 uses Word2Vec embeddings of named entities as inputs to a number of regression models.

Similar to us, they aim to predict numerical attributes of knowledge base entities.

Different to us, they leverage text information to do so.

This difference is important, because we do not assume the existence of information other than the graph structure.

Our problem is general enough to address knowledge bases where entities names are unknown or anonymized (e.g. medical knowledge bases).To our knowledge there is no existing work in the value imputation literature that attempts to fill missing values in KBs while taking advantage of the structural information provided by the KB.

A knowledge base, KB, is denoted as G = (E, P), where E is a set of entities and, P is a set of relation types or predicates.

This standard definition can be found in many papers in the link prediction literature [Nickel et al., 2016, Garcia-Duran and .

A KB is a collection of facts (or standard facts) (h, p, t) where p ??? P and h, t ??? E. We now define a knowledge base enriched with numerical attributes as G N A = (G, A, N ).

Entities in G are associated with numerical values N via numerical predicates A. This information can be expressed as a collection of numerical facts (h, a, t) where a ??? A, h ??? E and t ??? N .

In the paper we interchangeably use the term 'numerical predicate' with 'numerical attribute'.

The numerical attribute prediction problem seeks the most probable completion of a fact (h, a, ?), where h ??? E, a ??? A and ?

??? N .We refer to the set of entities for which the value of the numerical attribute a is known as E a ??? E. Let e be an entity with numerical attribute a, then we denote the known numerical value for attribute a as n a e .

The goal is to learn a function f : E ??? , denotes the set of reals.

One can omit the relational information given by the graph G and apply a value imputation method to fill missing values.

However, it is intuitive to assume the existence of an underlying generative model that (partially) determines the relational structure of the KB based on the values of the entities' numerical attributes.

For instance, two entities are likely linked via the relationship /location/contains if they have similar latitude and longitude; or two highly connected entities that correspond to people are likely to have similar birth years.

If this assumption is true, then a model that exploits the graph structure information is likely to outperform simple value imputation methods.

Nevertheless, while this may be true for a number of numerical attributes, for others the graph structure may introduce noise or, in the best case, be irrelevant.

Inspired by previous work in the value imputation and the node classification literature, we propose the following baselines.

A simple and natural baseline is simply using the sample mean of the attribute specific training data as a predictor for missing values.

This is known as mean imputation BID29 .

At test time, given an entity e for which we aim to predict the value of numerical predicate a, denoted asn a e , this baseline simply assigns the sample mean of all known entities possessing the same numerical attribute (E a ).

This is formally described below.n DISPLAYFORM0 where f is the sample mean.

We term this model as Global because it harnesses global information from the entire attribute specific training set.

In this work we use the root mean square error (RMSE) and the mean absolute error (MAE) as evaluation metrics.

While the sample mean is the best estimator for the former, the sample median is the optimum for the latter BID20 .

Consequently, in the experimental section we use median imputation when reporting the MAE and mean imputation when reporting on RMSE metrics.

Median imputation is obtained by simply replacing the sample average by the median in Eq. (1).

Our second baseline takes into account that entities are interconnected through a relational graph structure.

Thus it is natural to define a baseline that exploits the neighborhood or local graph structure.

The weighted-vote relational neighbor BID14 ] is a relational classifier often used as a benchmark in the node classification literature.

It estimates the class label of a node as a weighted average of its neighbors' class labels.

Despite its simplicity, it is shown to be competitive BID24 and is advocated as a sensible relational classification baseline BID15 .Inspired by such work, we propose an adaptation for our setting and problem.

For a numerical attribute a, this baseline estimates a value for the entity e as the average of its neighbors' attribute values for that numerical attribute.

Here, the neighborhood of a node e, denoted N e , is defined as the set of nodes that are connected to e through any relation type.

The baseline is formalized as follow?? DISPLAYFORM0 where, as before, f is either the sample mean or the sample median depending on the evaluation metric reported.

We term this model as Local because it uses the local neighborhood information for prediction.

In the case where E a ??? N e = ???, we make use of the so-called Global baseline to make a prediction.

We leverage KB embedding methods to learn feature representations -embeddings-of entities that (ideally) are predictive for the different numerical attributes.

As we argued before, this is only true if the entities' numerical attributes determine, to a certain extent, the existence of a certain relation type between two entities.

We first learn knowledge base embeddings, and in a second step we use these embeddings, along with the numerical facts, to train downstream machine learning models for predicting numerical attributes.

This pipeline is reminiscent of recent work BID25 in the node classification literature.

While there is an extent literature on KG embedding methods BID21 BID2 BID22 , recent work BID10 shows that well-tuned "simple" scoring functions BID37 are very hard to beat.

Likewise BID6 shows that TransE BID2 performs similarly or even better than many of its variants, such as TransH BID35 or TransR BID13 .Due to its simplicity and good performance in related problems, we choose TransE to illustrate the generic principles behind our models.

Note, however, that the methodology described is agnostic to the chosen KG embedding method.

The probability for a fact DISPLAYFORM0 c g(c|??)) , where c indexes all possible triples, and ?? all learnable parameters of TransE, whose scoring function g is g(d | ??) = ||h + p ??? t|| 2 .

We use bold letters h, p, t ??? d to denote the corresponding d-dimensional feature representations of h, p, t, respectively.

Note that this formulation is impractical because the cost of computing all possible triples is unfeasible.

Instead, for each triple d = (h, p, t) ??? G we generate a set of N triples (h, p, t ) by sampling N entities t uniformly at random from the set of all entities.

This process, which is termed as negative sampling, is repeated for the head of the triple.

For a given set of facts D that are part of the KB G, the logarithmic loss is defined as DISPLAYFORM1 All parameters ?? are learned for minimizing L G with stochastic gradient descent.

Once the representation learning phase is finished we evaluate two different approaches that utilize these embeddings for addressing the numerical attribute prediction problem.

In the simplest case, for each numerical attribute we use the learned feature representations as input to a regression model to predict the corresponding numerical attribute.

For numerical attribute a the loss function is given by DISPLAYFORM0 where f ?? a refers to the regression function for numerical attribute a, ?? a refers to the learnable parameters of f ?? a , and ?? a is the regularization hyper-parameter.

In this work we use a linear regression model: f ?? a (e) = e T w a + b a , where w a ??? d is the weight vector and b a is the corresponding bias term.

At test time, given a query related to a certain numerical attribute a and a certain entity e, the prediction is computed by applying the corresponding linear regression model:n a e = f ?? a (e).

We refer to this approach as Lr.

Previously we defined E a as the set of entities with known numerical attribute a. Similarly, we define Q a as the set of entities with missing values for numerical attribute a.

We consider numerical attribute values as labels, and, consequently, we can think of E a and Q a as the set of labeled and unlabeled nodes, respectively.

Therefore, semi-supervised learning is a natural choice because it also uses unlabeled data to infer values of numerical attributes.

Label propagation (LP) BID13 Ghahramani, 2002, Fujiwara and BID4 has been proven to be an effective semi-supervised learning algorithm for classification problems.

The key assumption of label propagation, and in general most semi-supervised learning algorithms, is similar to ours: Data points close to each other are likely to have the same label -numerical attribute values in our case.

We aim to propagate numerical attribute information across the graph using LP.

For numerical attribute a, we use the learned representations {e} e???E a ???Q a to induce a k-nearest neighbor graph (kNN) using euclidean distance.

This graph is characterized by an adjacency matrix A ??? N ??N , where N = |E a | + |Q a |.

The edge weights of the adjacency matrix represent similarities between the connected entities, which are computed according to a similarity metric ?? -in this work we use a radial basis function kernel 2 .We then compute the transition matrix T by row-wise normalizing the matrix A. Without loss of generality, we arrange labeled and unlabeled data so that T can be decomposed as DISPLAYFORM0 2.

??(x, y) = exp(??? ||x ???

y|| The transition matrix T (illustrated in FIG2 ) can be iteratively used to propagate numerical information across the graph until a stopping criterion is reached.

Alternatively, this problem can be solved in a closed form: DISPLAYFORM1 where n a E a ??? |E a | is a vector that contains all values of numerical attribute a for labeled nodes.

Similarly,n a Q a ??? |Q a | is a vector that contains all predicted values of numerical attribute a for unlabeled nodes.

We refer to the matrix M a in Section 5.1.

We term this as Numerical Attribute Propagation (or Nap).Related work BID18 uses label propagation to perform link prediction in web ontologies by casting it as a binary classification problem, where the similarity graph is built based on homophilic relationships.

In the two aforementioned solutions, we fully rely on the feature representations learned by, in this case, TransE to be meaningful with respect to the numerical attributes we aim to predict.

This relates to our initial assumption that the relational structure of a KB can be explained, to some extent, by the numerical attributes of the entities.

However, there might be cases where the values taken by entities for a certain numerical attribute do not fully relate to the relational structure of the KB.Motivated by this consideration, we set out to answer the question: Can these models benefit from learning feature representations incorporating, beside the graph structure, numerical attribute information?To answer this question we incorporate the learning objective of Eq. (4) into the learning objective of TransE (Eq. (3)): DISPLAYFORM0 where ?? weights the importance of the linear regression objectives.

All parameters are learned using stochastic gradient descent.

We term these embeddings as TransE++, which, contrary to TransE, are also learned with numerical facts.

While Nap and Lr use TransE feature representations, their counterparts Nap++ and Lr++ leverage TransE++ embeddings.

Different numerical attributes exhibit different scales of attribute values.

For example 'person.height' ranges from 1 to 2 meters while 'location.area' scales from several hundred to even millions of kilometers.

Bigger scales lead to larger prediction errors during training which in turn affect the back-propagation gradients used for learning the TransE++ embeddings.

To alleviate this problem, we normalize each numerical attribute to zero-mean and unit-variance.

We also experimented with min-max scaling, however it gave worse performance compared to standard scaling.

Scaling numerical attribute values remains an interesting challenge.

For Lr++ we do not directly use the regression models learned during opimizing the TransE++'s learning objective (Eq. FORMULA7 ).

Instead we use the learned TransE++ embeddings to train a new regression model L a R for each numerical attribute a ??? A. This is because of the computational difficulty in tuning hyper parameter ?? a for each numerical attribute while learning TransE++, which we found important to obtain good performance.

Note that the hyper parameter space grows exponentially with the number of attributes |A|.

For this reason, we set ?? a = ?? = 0 in the regression objectives in Eq. (7) for learning TransE++ embeddings (first step).

For the final regression models (second step) we do tune ?? a s (independently), which, though suboptimal, facilitates their tuning.

The proposed methods are evaluated by their ability to answer completion queries of the form (h, a, ?), where h ??? E, a ??? A and ?

??? N .

We evaluate the baselines and our models on two benchmark datasets: FB15K-237 BID33 and YAGO15K .

While for the former, numerical attributes were introduced in BID5 , for the later we obtained this information from dumps found online on YAGO's website 3 .The FB15K-237 dataset contains a total of 29,395 numerical facts divided in 116 different numerical predicates.

We evaluate our models on the top 10 numerical attributes ranked by the number of data samples.

This reduces the dataset to 22,929 samples.

We split these numerical facts into training, validation and test in the proportion of 80/10/10%, respectively.

All other facts from FB15K-237 whose predicate belongs to P are used as training data, which amounts to 310,116 facts.

Thus we only evaluate our approaches on their ability to answer queries whose answer is a numerical value.

The YAGO15K dataset contains 23,520 numerical facts divided in 7 different attributes.

Similarly, we split these numerical facts into training, validation and test in the same proportion.

We use all other 122,886 facts from this dataset for learning knowledge base embeddings.

A summary of the datasets can be found in Table 1 .

All the splits of both datasets used in this work will be made publicly available to facilitate future comparisons.

We compare performance across methods using two evaluation metrics -MAE and RMSE.

These are standard metrics used in regression problems and were also used in a related study of predicting numerics for language models BID31 .

DISPLAYFORM0 For TransE and TransE++ we fix the embedding dimension d to 100.

After some preliminary experiments, the weight ?? of TransE++ was fixed to 1.

We used Adam BID11 ] to learn the parameters in a mini-batch setting with a learning rate of 0.001.

We fixed the number of epochs to 100 and the mini-batch size to 256.

The parameter N of the negative sampling was set to 50.

Within a batch, the number of data points for each of the TransE++'s regression objectives is proportional to the frequency of each of the numerical predicates in the training set.

In all cases, the parameters were initialized following BID8 .We used the Scikit-learn BID23 implementation of ridge regression for the approaches Lr and Lr++.

The regularization term ?? a is tuned using the values [0, 0.1, 1, 10, 100].For Nap and Nap++ the number of neighbors (k) of the kNN graph is validated among [3, 5, 10, 20] ; and the ?? of the RBF kernel is validated among [0.25, 0.5, 1, 10] .All of the above is validated for each numerical predicate and evaluation metric.

The objectives of this section is twofold: First we investigate the performance of our approaches (Lp and Nap, and their variants) with respect to the baselines.

And second we experimentally check how robust these methods are for different degrees of sparsity in the training data.

TAB2 detail the performance of the baselines and our approaches on FB15K-237.

For each numerical attribute we always indicate in bold font the best performing method, which happens to be either Nap++ or Nap most of the time.

Interestingly enough, from TAB2 we observe that for the numerical attributes 'location.area' and 'population.number' Global largely outperforms Local.

This seems to indicate that the relational structure of this data set does not relate to these two numerical predicates.

Overall, predictions for all other numerical attributes tend to benefit from the local information given by the entities' neighborhood.

In comparing Tables 2 and 3, we note that Local is very competitive in Table 3 : Performance of Lr-and Nap-based models on FB15K-237.regard to the numerical attributes 'latitude' and 'longitude'.

This can be explained by the presence of predicates such as 'location.adjoins' or 'location.contains' in the relational structure of the graph.

Similarly, entities' neighborhoods are useful for predicting 'date of birth' or 'date of death' because (some of the) surrounding entities correspond to people who have similar birth or death dates.

Interestingly, all our approaches beat both baselines in the numerical attribute 'person.height mt', for which a priori one would not expect performance gains in learning from the graph structure.

Overall, Lr++ and Nap++ outperform their counterparts Lr and Nap, respectively, for most numerical predicates.

As we argued in Section 3.2.3 it is not feasible to validate the regularization term ?? a for every numerical attribute while learning TransE++.

We speculate that setting ?? a = 0 while training TransE++ may explain why Lr++ and Nap++ do not always beat their counterparts.

Table 4 : Performance of Local and Nap++ on FB15K-237 for different degrees of sparsity, P r , on the numerical facts.

Results are reported in terms of Mean Absolute Error (MAE).Another observation from Table 3 is that, in general, Nap-based models perform much better compared to Lr-based models.

One can find a number of explanations to this.

The obvious explanation is that the numerical attribute propagation approaches learn from labeled and unlabeled data, whereas the regression models only learn from labeled data.

A second explanation is that whereas Nap's predictions are computed as a weighted average of observed numerical values, Lr's predictions are not bounded.

This prevents Nap-based approaches from making large mistakes.

On the other hand, for example, we observed non-plausible values (e.g. > 2020) predicted by the Lr-based models for the numerical attribute 'date of birth'.

We also experimented with non-linear regression models, but did not observe any performance improvement.

Knowledge graphs are known to suffer from data sparsity due to missing facts.

The same incompleteness is also true for numerical facts.

Therefore it is crucial to study model performance under a sparse data regime.

We generate data sparsity by artificially removing numerical facts from the training set while keeping the validation and test sets unchanged.

We keep the underlying knowledge graph G unchanged because we aim to isolate the effect of numerical fact sparsity.

In other words, only a number of numerical facts are removed from the training set.

We retained a percentage P r of training numerical facts and ran Local and Nap++ with the same experimental set-up.

We experimented with the following values of P r : [100 4 , 80, 50, 20]%.

We detail the results of these experiments in Table 4 .

Note that the performance of Local degrades more rapidly compared to Nap++ as the sparsity increases.

Even in high regimes of sparsity, Nap++'s performance is remarkably robust.

TAB5 lists results for Global and Local in YAGO15K.

As for FB15K-237, Local outperforms Global for most of the numerical attributes.

This reinforces our assumption that the numerical attributes explain, to some extent, relation structure between entities.

Table 6 depicts the performance of Local, Nap and Nap++ under different degrees of sparsity in YAGO15K.

In the light of these numbers, we can conclude that the Nap-based Table 6 : Performance of Local, Nap and Nap++ on YAGO15K for different degrees of sparsity, P r , on the numerical facts.

Results are reported in terms of Mean Absolute Error.

models are more robust than Local to data sparsity.

Nap++ achieves the best performance for most of the numerical attributes and degrees of sparsity.

It performs remarkably well for the numerical attribute 'happenedOnDate' in comparison to Nap.

Across all values of P r , on average, NAP++ improves Nap's performance by 20 points (in mean absolute value) for 'happenedOnDate'.

We recognize that reporting model performance in absolute values complicates comparison since numerical attributes lie on different ranges of values.

To have a better picture of performance gains we report percentage error reduction between Nap++ and the best performing baseline.

For numerical attribute a, the percentage error reduction in MAE is computed as follows DISPLAYFORM0 One can compute the percentage error reduction in terms of RMSE in a similar manner.

This is shown in TAB7 for P r = 100.

We do not include 'location.area' and 'population.number', as previous experiments indicate that they do not relate to the graph structure of FB15K-237.

Overall, Nap++ significantly outperforms baselines for almost all numerical attributes in both FB15K-237 and YAGO15K data sets.

These results demonstrate that the embeddings learned from the graph structure are useful predictors of entity numerical attributes.

Table 8 : Qualitative comparison between Nap and Nap++.

The three first rows correspond to queries where the numerical attribute is 'date of birth', whereas for the two last queries it is 'date of death'.

The actual value of labeled entities for the corresponding numerical attribute is shown in parenthesis.

This last experimental section aims to provide some insights on the benefit of adding numerical information during the representation learning stage.

Table 3 shows a noteworthy behavior of these methods with respect to the numerical attributes 'date of birth' and 'date of death'.

While the performance of both approaches is comparable in terms of MAE, their RMSE largely differ.

It is known that the mean absolute error is an evaluation metric more robust to outliers than the root mean squared error.

We set out to inspect these outliers to shed light on the usefulness of incorporating numerical information in the embeddings.

Nap-based models leverage these embeddings to build a similarity graph on which numerical information is propagated via Eq. (6).

The resulting predictions are the result of multiplying the matrix M a by the observed numerical values.

This matrix 5 determines which observed entities' numerical values to pay attention to.

These attention values are different for Nap and Nap++ as the graph similarity is constructed with different embeddings.

We qualitatively compare Nap and Nap++ based on a number of predictions computed in the test set.

For each method we compare the two labeled entities they pay the most attention to.

For the sake of simplicity we refer to these two entities as nearest neighbors.

This is shown in Table 8 .

An interesting observation is that for NAP the two nearest neighbors are always entities topically similar to the entity in the query.

On the other hand the nearest entities retrieved by NAP++ are more meaningful with respect to the queried numerical attribute.

This is seen in the first query: (Alexander the Great 6 , date of birth, ?).

While Nap pays the most attention to topically similar entities, Nap++ puts a high attention on Julius Caesar, 7 which is more meaningful in regard to the date of birth.

Nap++ uses Euclidean distance between vectors to build the k-nearest neighbor graph.

Table 8 that subsets of entities latent factors could be encoding different relational and numerical information.

For instance, a few dimensions of the entity embeddings encode location information, while others encode population information and so on.

To exploit this we learned Mahalanobis metrics for capturing different entity similarities.

We did this while learning knowledge base embeddings by using an additional nearest neighbor loss.

It did slightly improve the performance for few attributes, but overall it did not make significant distance.

We suggest that future work should address this research direction in greater depth.

We introduce a novel problem, namely numerical attribute prediction in knowledge bases.

Contrary to link prediction, the answer to this new query type is a numerical value, and not an element from a (small) closed vocabulary.

Our premise to this problem is that the relational structure of a KB can be partially explained by the numerical attribute values associated with entities.

This allows for leveraging KB embedding methods to learn representations that are useful predictors of numerical attributes.

An extensive set of experiments validates our premise.

Furthermore, we also show that learning KB representations enriched with numerical attribute information are helpful for addressing this task.

Finally, we believe that this new problem introduced in the paper will spur interest and deeper investigation from the research community.5.

Note that it is non-negative and is row normalized.

6.

For all practical purposes he is deemed a philosopher in FB15K-237.

7.

Julius Caesar belongs to profession Politician in FB15K-237

@highlight

Prediction of numerical attribute values associated with entities in knowledge bases.