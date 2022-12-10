Knowledge graph embedding research has overlooked the problem of probability calibration.

We show popular embedding models are indeed uncalibrated.

That means probability estimates associated to predicted triples are unreliable.

We present a novel method to calibrate a model when ground truth negatives are not available, which is the usual case in knowledge graphs.

We propose to use Platt scaling and isotonic regression alongside our method.

Experiments on three datasets with ground truth negatives show our contribution leads to well calibrated models when compared to the gold standard of using negatives.

We get significantly better results than the uncalibrated models from all calibration methods.

We show isotonic regression offers the best the performance overall, not without trade-offs.

We also show that calibrated models reach state-of-the-art accuracy without the need to define relation-specific decision thresholds.

Knowledge graph embedding models are neural architectures that learn vector representations (i.e. embeddings) of nodes and edges of a knowledge graph.

Such knowledge graph embeddings have applications in knowledge graph completion, knowledge discovery, entity resolution, and link-based clustering, just to cite a few (Nickel et al., 2016a) .

Despite burgeoning research, the problem of calibrating such models has been overlooked, and existing knowledge graph embedding models do not offer any guarantee on the probability estimates they assign to predicted facts.

Probability calibration is important whenever you need the predictions to make probabilistic sense, i.e., if the model predicts a fact is true with 80% confidence, it should to be correct 80% of the times.

Prior art suggests to use a sigmoid layer to turn logits returned by models into probabilities (Nickel et al., 2016a ) (also called the expit transform), but we show that this provides poor calibration.

Figure 1 shows reliability diagrams for off-the-shelf TransE and ComplEx.

The identity function represents perfect calibration.

Both models are miscalibrated: all TransE combinations in Figure 1a under-forecast the probabilities (i.e. probabilities are too small), whereas ComplEx under-forecasts or over-forecasts according to which loss is used (Figure1b).

Calibration is crucial in high-stakes scenarios such as drug-target discovery from biological networks, where end-users need trustworthy and interpretable decisions.

Moreover, since probabilities are not calibrated, when classifying triples (i.e. facts) as true or false, users must define relationspecific thresholds, which can be awkward for graphs with a great number of relation types.

To the best of our knowledge, this is the first work to focus on calibration for knowledge embeddings.

Our contribution is two-fold: First, we use Platt Scaling and isotonic regression to calibrate knowledge graph embedding models on datasets that include ground truth negatives.

One peculiar feature of knowledge graphs is that they usually rely on the open world assumption (facts not present are not necessarily false, they are simply unknown).

This makes calibration troublesome because of the lack of ground truth negatives.

For this reason, our second and main contribution is a calibration heuristics that combines Platt-scaling or isotonic regression with synthetically generated negatives.

Experimental results show that we obtain better-calibrated models and that it is possible to calibrate knowledge graph embedding models even when ground truth negatives are not present.

We also experiment with triple classification, and we show that calibrated models reach state-of-the-art accuracy without the need to define relation-specific decision thresholds.

A comprehensive survey of knowledge graph embedding models is out of the scope of this paper.

Recent surveys such as (Nickel et al., 2016a) and (Cai et al., 2017) present an overview or recent literature.

TransE (Bordes et al., 2013) is the forerunner of distance-based methods, and spun a number of models commonly referred to as TransX. The intuition behind the symmetric bilinear-diagonal model DistMult (Yang et al., 2015) paved the way for its asymmetric evolutions in the complex space, RotatE (Sun et al., 2019) and ComplEx (Trouillon et al., 2016 ) (a generalization of which uses hypercomplex representations (Zhang et al., 2019) ).

HolE relies instead on circular correlation (Nickel et al., 2016b) .

The recent TorusE (Ebisu & Ichise, 2018) operates on a lie group and not in the Euclidean space.

While the above models can be interpreted as multilayer perceptrons, others such as ConvE (Dettmers et al., 2018) or ConvKB (Nguyen et al., 2018) include convolutional layers.

More recent works adopt capsule networks architectures (Nguyen et al., 2019) .

Adversarial learning is used by KBGAN (Cai & Wang, 2018) , whereas attention mechanisms are instead used by (Nathani et al., 2019) .

Some models such as RESCAL (Nickel et al., 2011) , TuckER (Balažević et al., 2019), and SimplE (Kazemi & Poole, 2018) rely on tensor decomposition techniques.

More recently, ANALOGY adopts a differentiable version of analogical reasoning (Liu et al., 2017) .

In this paper we limit our analysis to four popular models: TransE, DistMult, ComplEx and HolE. They do not address the problem of assessing the reliability of predictions, leave aside calibrating probabilities.

Besides well-established techniques such as Platt scaling (Platt et al., 1999) and isotonic regression (Zadrozny & Elkan, 2002) , recent interest in neural architectures calibration show that modern neural architectures are poorly calibrated and that calibration can be improved with novel methods.

For example, (Guo et al., 2017) successfully proposes to use temperature scaling for calibrating modern neural networks in classification problems.

On the same line, (Kuleshov et al., 2018) proposes a procedure based on Platt scaling to calibrate deep neural networks in regression problems.

The Knowledge Vault pipeline in (Dong et al., 2014) extracts triples from unstructured knowledge and is equipped with Platt scaling calibration, but this is not applied to knowledge graph embedding models.

KG2E proposes to use normally-distributed embeddings to account for the uncertainty, but their model does not provide the probability of a triple being true, so KG2E would also benefit from the output calibration we propose here.

To the best of our knowledge, the only work that adopts probability calibration to knowledge graph embedding models is Krompaß & Tresp (2015) .

The authors propose to use ensembles in order to improve the results of knowledge graph embedding tasks.

For that, they propose to calibrate the models with Platt scaling, so they operate on the same scale.

No further details on the calibration procedure are provided.

Besides, there is no explanation on how to handle the lack of negatives.

Knowledge Graph.

Formally, a knowledge graph G = {(s, p, o)} ⊆ E × R × E is a set of triples t = (s, p, o) , each including a subject s ∈ E, a predicate p ∈ R, and an object o ∈ E. E and R are the sets of all entities and relation types of G.

Triple Classification.

Binary classification task where G (which includes only positive triples) is used as training set, and T = {(s, p, o)} ⊆

E × R × E is a disjoint test set of labeled triples to classify.

Note T includes positives and negatives.

Since the learned models are not calibrated, multiple decision thresholds τ i must be picked, where 0 < i < |R|, i.e. one for each relation type.

This is done using a validation set (Bordes et al., 2013) .

Classification metrics apply (e.g. accuracy).

Link Prediction.

Given a training set G that includes only positive triples, the goal is assigning a score f (t) ∈ R proportional to the likelihood that each unlabeled triple t included in a held-out set S is true.

Note S does not have ground truth positives or negatives.

This task is cast as a learning to rank problem, and uses metrics such as mean rank (MR), mean reciprocal rank (MRR) or Hits@N.

Knowledge Graph Embeddings.

Knowledge graph embedding models are neural architectures that encode concepts from a knowledge graph G (i.e. entities E and relation types R) into lowdimensional, continuous vectors ∈ R k (i.e, the embeddings).

Embeddings are learned by training a neural architecture over G. Although such architectures vary, the training phase always consists in minimizing a loss function L that includes a scoring function f m (t), i.e. a model-specific function that assigns a score to a triple t = (s, p, o) (more precisely, the input of f m are the embeddings of the subject e s , the predicate r p , and the object e o ).

The goal of the optimization procedure is learning optimal embeddings, such that the scoring function f m assigns high scores to positive triples t + and low scores to triples unlikely to be true t − .

Existing models propose scoring functions that combine the embeddings e s , r p , e o ∈ R k using different intuitions.

Table 1b lists the scoring functions of the most common models.

For example, the scoring function of TransE computes a similarity between the embedding of the subject e s translated by the embedding of the predicate e p and the embedding of the object e o , using the L 1 or L 2 norm || · ||.

Such scoring function is then used on positive and negative triples t + ∈ G, t − ∈ N in the loss function.

This is usually a pairwise margin-based loss (Bordes et al., 2013) , negative log-likelihood, or multi-class log-likelihood (Lacroix et al., 2018) .

Since the training set usually includes positive statements, we generate synthetic negatives t − ∈ N required for training.

We do so by corrupting one side of the triple at a time (i.e. either the subject or the object), following the protocol proposed by (Bordes et al., 2013) .

Calibration.

Given a knowledge graph embedding model identified by its scoring function f m , with f m (t) =p, wherep is the estimated confidence level that a triple t = (s, p, o) is true, we define f m to be calibrated ifp represents a true probability.

For example, if f m (·) predicts 100 triples all with confidencep = 0.7, we expect exactly 70 to be actually true.

Calibrating a model requires reliable metrics to detect miscalibration, and effective techniques to fix such distortion.

Appendix A.1 includes definitions and background on the calibration metrics adopted in the paper.

We propose two scenario-dependent calibration techniques: we first address the case with ground truth negatives t − ∈ N .

The second deals with the absence of ground truth negatives.

Calibration with Ground Truth Negatives.

We propose to use off-the-shelf Platt scaling and isotonic regression, techniques proved to be effective in literature.

It is worth reiterating that to calibrate a model negative triples N are required from a held-out dataset (which could be the validation set).

Such negatives are usually available in triple classification datasets (FB13, WN11, YAGO39K)

Calibration with Synthetic Negatives.

Our main contribution is for the case where no ground truth negatives are provided at all, which is in fact the usual scenario for link prediction tasks.

We propose to adopt Platt scaling or isotonic regression and to synthetically generate corrupted triples as negatives, while using sample weights to guarantee that the frequencies adhere to the base rate of the population (which is problem-dependent and must be user-specified).

It is worth noting that it is not possible to calibrate a model without implicit or explicit base rate.

If it is not implicit on the dataset (the ratio of positives to totals), it must be explicitly provided.

We generate synthetic negatives N following the standard protocol proposed by (Bordes et al., 2013) 1 : for every positive triple t = (s, p, o), we corrupt one side of the triple at a time (i.e. either the subject s or the object o) by replacing it with other entities in E. The number of corruptions generated per positive is defined by the user-defined corruption rate η ∈ N. Since the number of negatives N = |N | can be much greater than the number of positive triples P = |G|, when dealing with calibration with synthetically generated corruptions, we weigh the positive and negative triples to make the calibrated model match the population base rate α = P/(P + N ) ∈ [0, 1], otherwise the base rate would depend on the arbitrary choice of η.

Given a positive base rate α, we propose the following weighting scheme:

where ω + ∈ R is the weight associated to the positive triples and ω − ∈ R to the negatives.

The ω + weight removes the imbalance determined by having a higher number of corruptions than positive triples in each batch.

The ω − weight guarantees that the given positive base rate α is respected.

The above can be verified as follows.

For the unweighted problem, the positive base rate is simply the ratio of positive examples to the total number of examples:

If we add uniform weights to each class, we have:

By defining ω + = η, i.e. adopting the ratio of negatives to positives (corruption rate), we then have:

Thus, the negative weights is:

We compute the calibration quality of our heuristics, showing that we achieve calibrated predictions even when ground truth negative triples are not available.

We then show the impact of calibrated predictions on the task of triple classification.

Datasets.

We run experiments on triple classification datasets that include ground truth negatives (Table 1) .

We train on the training set, calibrate on the validation set, and evaluate on the test set.

• WN11 (Socher et al., 2013) .

A subset of Wordnet (Miller, 1995) , it includes a large number of hyponym and hypernym relations thus including hierarchical structures.

• FB13 (Socher et al., 2013) .

A subset of Freebase (Bollacker et al., 2008) , it includes facts on famous people (place of birth and/or death, profession, nationality, etc).

• YAGO39K (Lv et al., 2018) .

This recently released dataset has been carved out of YAGO3 (Mahdisoltani et al., 2013) , and includes a mixture of facts about famous people, events, places, and sports teams.

We also use two standard link prediction benchmark datasets, WN18RR (Dettmers et al., 2018 ) (a subset of Wordnet) and FB15K-237 (Toutanova et al., 2015) (a subset of Freebase).

Their test sets do not include ground truth negatives.

Implementation Details.

The knowledge graph embedding models are implemented with the AmpliGraph library (Costabello et al., 2019 ) version 1.1, using TensorFlow 1.13 (Abadi et al., 2016) and Python 3.6 on the backend 2 .

All experiments were run under Ubuntu 16.04 on an Intel Xeon Gold 6142, 64 GB, equipped with a Tesla V100 16GB.

Hyperparameter Tuning.

For each dataset in Table 1a , we train a TransE, DistMult, and a ComplEx knowledge graph embedding model.

We rely on typical hyperparameter values: we train the embeddings with dimensionality k = 100, Adam optimizer, initial learning rate α 0 = 1e-4, negatives per positive ratio η = 20, epochs = 1000.

We train all models on four different loss functions: Selfadversarial (Sun et al., 2019) , pairwise (Bordes et al., 2013) , NLL, and Multiclass-NLL (Lacroix et al., 2018) .

Different losses are used in different experiments.

Calibration Success.

Table 2 reports Brier scores and log losses for all our calibration methods, grouped by the type of negative triples they deal with (ground truth or synthetic).

All calibration methods show better-calibrated results than the uncalibrated case, by a considerable margin and for all datasets.

In particular, to put the results of the synthetic strategy in perspective, if we suppose to predict the positive base rate as a baseline, for each of the cases in Table 2 (the three datasets share the same positive base rate α = 0.5), we would get Brier score B = 0.25 and log loss L log = 0.69, results that are always worse than our methods.

There is considerable variance of results between models given a dataset, which also happens when varying losses given a particular combination of model and dataset (Table 3) .

TransE provides the best results for WN11 and FB13, while DistMult works best for YAGO39K.

We later propose that this variance comes from the quality of the embeddings themselves, that is, better embeddings allow for better calibration.

In Figure 2 , we also evaluate just the frequencies themselves, ignoring sharpness (i.e. whether probabilities are close to 0 or 1), using reliability diagrams for a single model-loss combination, for all datasets (ComplEx+NLL).

Calibration plots show a remarkable difference between the uncalibrated baseline (s-shaped blue line on the left-hand side) and all calibrated models (curves closer to the identity function are better).

A visual comparison of uncalibrated curves in Figure 1 with those in Figure 2 also gives a sense of the effectiveness of calibration.

Ground Truth vs Synthetic.

As expected, the ground truth method generally performs better than the synthetic calibration, since it has more data in both quantity (twice as much) and quality (two classes instead of one).

Even so, the synthetic method is much closer to the ground truth than to the uncalibrated scores, as highlighted by the calibration plots in Figure 2 .

For WN11, it is actually as good as the calibration with the ground truth.

This shows that our proposed method works as intended and could be used in situations where we do not have access to the ground truth, as is the case for most knowledge graph datasets.

Isotonic vs Platt.

Isotonic regression performs better than Platt scaling in general, but in practice Isotonic regression has the disadvantage of not being a convex or differentiable algorithm Zadrozny & Elkan (2002) .

This is particularly problematic for the synthetic calibration, as it requires the generation of the synthetic corruptions, which can only be made to scale via a mini-batch based optimization procedure.

Platt scaling, given that it is a convex and differentiable loss, can be made Table 2 : Calibration test results (self-adversarial loss (Sun et al., 2019) ).

Low score = better.

Best results in bold for each combination of dataset and metric.

part of a computational graph and optimized with mini-batches, thus it can rely on the modern computational infrastructure designed to train deep neural networks.

Influence of Loss Function.

We experiment with different losses, to assess how calibration affects each of them (Table 3) .

We choose to work with TransE, which is reported as a strong baseline in (Hamaguchi et al., 2017 tion methods, across all datasets.

Experiments also show the choice of the loss has a big impact, greater than the choice of calibration method or embedding model.

We assess whether such variability is determined by the quality of the embeddings.

To verify whether better embeddings lead to sharper calibration, we report the mean reciprocal rank (MRR), which, for each true test triple, computes the (inverse) rank of the triple against synthetic corruptions, then averages the inverse rank (Table 3 ).

In fact, we notice no correlation between calibration results and MRR.

In other words, embeddings that lead to the best predictive power are not necessary the best calibrated.

Positive Base Rate.

We apply our synthetic calibration method to two link prediction benchmark datasets, FB15K-237 and WN18RR.

As they only provide positive examples, we apply our method with varying base rates α i , linearly spaced from 0.05 to 0.95.

We evaluate results relying on the closed-world assumption, i.e. triples not present in training, validation or test sets are considered negative.

For each α i we calibrate the model using the synthetic method with both isotonic regression and Platt scaling.

We sample negatives from the negative set under the implied negative rate, and calculate a baseline which is simply having all probability predictions equal to α i .

Figure 3 shows that isotonic regression and Platt scaling perform similarly and always considerably below the baseline.

As expected from the previous results, the uncalibrated scores perform poorly, only reaching acceptable levels around some particular base rates.

Triple Classification and Decision Threshold.

To overcome the need to learn |R| decision thresholds τ i from the validation set, we propose to rely on calibrated probabilities, and use the natural threshold of τ = 0.5.

Table 4 shows how calibration affects the triple classification task, comparing with the literature standard of per-relation thresholds (last column).

For simplicity, note we use the same self-adversarial loss in Table 2 and Table 4 .

We learn thresholds τ i on validation sets, resulting in 11, 7, and 33 thresholds for WN11, FB13 and YAGO39K respectively.

Using a single τ = 0.5 and calibration provides competitive results compared to multiple learned thresholds (note uncalibrated results with τ = 0.5 are poor, as expected).

It is worth mentioning that Figure 3 : Synthetic calibration on FB15K-237 and WN18RR, with varying positive base rates.

The baseline stands for using the positive base rate as the probability prediction.

Results are evaluated under the closed-world assumption, using the same positive base rate used to calibrate the models.

Ground Truth (τ = .5) Table 4 : Effect of calibration on triple classification accuracy.

Best results in bold.

For all calibration methods there is one single threshold, τ = 0.5.

For the per-relation τ , we learned multiple thresholds from validation sets (Appendix A.5).

We did not carry out additional model selection, and used Table 2 hyperparameters instead.

Isotonic regression reaches state-of-the-art results for WN11.

Results of * from (Zhang et al., 2018) ; from (Ji et al., 2016) ; † from (Lv et al., 2018) .

we are at par with state-of-the-art results for WN11.

Isotonic regression is again the best method, but there is more variance in the model choice.

Our proposed calibration method with synthetic negatives performs well overall, even though calibration is performed only using half of the validation set (negatives examples are replaced by synthetic negatives).

We propose a method to calibrate knowledge graph embedding models.

We target datasets with and without ground truth negatives.

We experiment on triple classification datasets and apply Platt scaling and isotonic regression with and without synthetic negatives controlled by our heuristics.

All calibration methods perform significantly better than uncalibrated scores.

We show that isotonic regression brings better calibration performance, but it is computationally more expensive.

Additional experiments on triple classification shows that calibration allows to use a single decision threshold, reaching state-of-the-art results without the need to learn per-relation thresholds.

Future work will evaluate additional calibration algorithms, such as beta calibration (Kull et al., 2017) or Bayesian binning (Naeini et al., 2015) .

We will also experiment on ensembling of knowledge graph embedding models, inspired by (Krompaß & Tresp, 2015) .

The rationale is that different models operate on different scales, but calibrating brings them all to the same probability scale, so their output can be easily combined.

Reliability Diagram (DeGroot & Fienberg, 1983; Niculescu-Mizil & Caruana, 2005) .

Also known as calibration plot, this diagram is a visual depiction of the calibration of a model (see Figure 1 for an example).

It shows the expected sample accuracy as a function of the estimated confidence.

A hypothetical perfectly calibrated model is represented by the diagonal line (i.e. the identity function).

Divergence from such diagonal indicates calibration issues (Guo et al., 2017) .

Brier Score (Brier, 1950) .

It is a popular metric used to measure how well a binary classifier is calibrated.

It is defined as the mean squared error between n probability estimatesp and the corresponding actual outcomes y ∈ 0, 1.

The smaller the Brier score, the better calibrated is the model.

Note that the Brier score B ∈ [0, 1].

Log Loss is another effective and popular metric to measure the reliability of the probabilities returned by a classifier.

The logarithmic loss measures the relative uncertainty between the probability estimates produced by the model and the corresponding true labels.

Platt Scaling.

Proposed by (Platt et al., 1999) for support vector machines, Platt scaling is a popular parametric calibration techniques for binary classifiers.

The method consists in fitting a logistic regression model to the scores returned by a binary classifier, such thatq = σ(ap + b), wherê p ∈ R is the uncalibrated score of the classifier, a, b ∈ R are trained scalar weights.

andq is the calibrated probability returned as output.

Such model can be trained be trained by optimizing the NLL loss with non-binary targets derived by the Bayes rule under an uninformative prior, resulting in an Maximum a Posteriori estimate.

Isotonic Regression (Zadrozny & Elkan, 2002) .

This popular non-parametric calibration techniques consists in fitting a non-decreasing piecewise constant function to the output of an uncalibrated classifier.

As for Platt scaling, the goal is learning a functionq = g(p), such thatq is a calibrated probability.

Isotonic regression learns g by minimizing the square loss n i=1 (q i −

y i ) 2 under the constraint that g must be piecewise constant (Guo et al., 2017) .

We present in Figure 4 the total count of instances for each bin used in the calibration plots included in Figure 2 .

As expected, calibration considerably helps spreading out instances across bins, whereas in uncalibrated scenarios instances are squeezed in the first or last bins.

A.3 IMPACT OF MODEL HYPERPARAMETERS: η AND EMBEDDING DIMENSIONALITY In Figure 5 we report the impact of negative/positive ratio η and the embedding dimensionality k. Results show that the embedding size k has higher impact than the negative/positive ratio η.

We observe that calibrated and uncalibrated low-dimensional embeddings have worse Brier score.

Results also show that any k > 50 does not improve calibration anymore.

The negative/positive ratio η follows a similar pattern: choosing η > 10 does not have any effect on the calibration score.

In Table 5 , we present the traditional knowledge graph embedding rank metrics: MRR (mean reciprocal rank), MR (mean rank) and Hits@10 (precision at the top-10 results).

We report the results for all datasets and models used in the main text, which appear in Table 2, Table 4 Table 5 : Standard filtered metrics for knowledge graph embeddings models.

The models are implemented in the same codebase and share the same evaluation protocol.

Note that we do not include results from reciprocal evaluation protocols.

We report in Table 6 the per-relation decision thresholds τ used in Table 4 , under the 'Reproduced' column.

Note that the thresholds reported here are not probabilities, as they have been applied to the raw scores returned by the model-dependent scoring function f m (t).

Table 6 : Relation-specific decision thresholds learned on uncalibrated raw scores (See also Table 4 for a report on triple classification results.)

<|TLDR|>

@highlight

We propose a novel method to calibrate knowledge graph embedding models without the need of negative examples.