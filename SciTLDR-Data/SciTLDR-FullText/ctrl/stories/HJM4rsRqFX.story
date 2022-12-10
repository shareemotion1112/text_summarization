Recent advances in Neural Variational Inference allowed for a renaissance in latent variable models in a variety of domains involving high-dimensional data.

In this paper, we introduce two generic Variational Inference frameworks for generative models of Knowledge Graphs; Latent Fact Model and Latent Information Model.

While traditional variational methods derive an analytical approximation for the intractable distribution over the latent variables, here we construct an inference network conditioned on the symbolic representation of entities and relation types in the Knowledge Graph, to provide the variational distributions.

The new framework can create models able to discover underlying probabilistic semantics for the symbolic representation by utilising parameterisable distributions which permit training by back-propagation in the context of neural variational inference, resulting in a highly-scalable method.

Under a Bernoulli sampling framework, we provide an alternative justification for commonly used techniques in large-scale stochastic variational inference, which drastically reduces training time at a cost of an additional approximation to the variational lower bound.

The generative frameworks are flexible enough to allow training under any prior distribution that permits a re-parametrisation trick, as well as under any scoring function that permits maximum likelihood estimation of the parameters.

Experiment results display the potential and efficiency of this framework by improving upon multiple benchmarks with Gaussian prior representations.

Code publicly available on Github.

In many fields, including physics and biology, being able to represent uncertainty is of crucial importance BID18 .

For instance, when link prediction in Knowledge Graphs is used for driving expensive pharmaceutical experiments (Bean et al., 2017) , it would be beneficial to know what is the confidence of a model in its predictions.

However, a significant shortcoming of current neural link prediction models BID13 BID38 -and for the vast majority of neural representation learning approaches -is their inability to express a notion of uncertainty.

Furthermore, Knowledge Graphs can be very large and web-scale BID14 and often suffer from incompleteness and sparsity BID14 .

In a generative probabilistic model, we could leverage the variance in model parameters and predictions for finding which facts to sample during training, in an Active Learning setting BID22 BID17 .

BID16 use dropout for modelling uncertainty, however, this is only applied at test time.

However, current neural link prediction models typically only return point estimates of parameters and predictions BID32 , and are trained discriminatively rather than generatively: they aim at predicting one variable of interest conditioned on all the others, rather than accurately representing the relationships between different variables BID31 , however, BID16 could still be applied to get uncertainty estimates for these models.

The main argument of this article is that there is a lack of methods for quantifying predictive uncertainty in a knowledge graph embedding representation, which can only be utilised using probabilistic modelling, as well as a lack of expressiveness under fixed-point representations.

This constitutes a significant contribution to the existing literature because we introduce a framework for creating a family of highly scalable probabilistic models for knowledge graph representation, in a field where there has been a lack of this.

We do this in the context of recent advances in variational inference, allowing the use of any prior distribution that permits a re-parametrisation trick, as well as any scoring function which permits maximum likelihood estimation of the parameters.

In this work, we focus on models for predicting missing links in large, multi-relational networks such as FREEBASE.

In the literature, this problem is referred to as link prediction.

We specifically focus on knowledge graphs, i.e., graph-structured knowledge bases where factual information is stored in the form of relationships between entities.

Link prediction in knowledge graphs is also known as knowledge base population.

We refer to BID32 for a recent survey on approaches to this problem.

A knowledge graph G {(r, a 1 , a 2 )} ⊆

R × E × E can be formalised as a set of triples (facts) consisting of a relation type r ∈ R and two entities a 1 , a 2 ∈ E, respectively referred to as the subject and the object of the triple.

Each triple (r, a 1 , a 2 ) encodes a relationship of type r between a 1 and a 2 , represented by the fact r(a 1 , a 2 ).Link prediction in knowledge graphs is often simplified to a learning to rank problem, where the objective is to find a score or ranking function φ Θ r : E × E → R for a relation r that can be used for ranking triples according to the likelihood that the corresponding facts hold true.

Recently, a specific class of link predictors received a growing interest BID32 .

These predictors can be understood as multi-layer neural networks.

Given a triple x = (s, r, o), the associated score φ Θ r (s, o) is given by a neural network architecture encompassing an encoding layer and a scoring layer.

In the encoding layer, the subject and object entities s and o are mapped to low-dimensional vector representations (embeddings) h s h(s) ∈ R k and h o h(o) ∈ R k , produced by an encoder h Γ : E → R k with parameters Γ. Similarly, relations r are mapped to h r h(r) ∈ R k .

This layer can be pre-trained BID41 or, more commonly, learnt from data by back-propagating the link prediction error to the encoding layer BID4 BID32 BID38 .The scoring layer captures the interaction between the entity and relation representations h s , h o and h r are scored by a function φ Θ (h s , h o , h r ), parametrised by Θ. Other work encodes the entity-pair in one vector BID34 .

Summarising, the high-level architecture is defined as: DISPLAYFORM0 Ideally, more likely triples should be associated with higher scores, while less likely triples should be associated with lower scores.

While the literature has produced a multitude of encoding and scoring strategies, for brevity we overview only a small subset of these.

However, we point out that our method makes no further assumptions about the network architecture other than the existence of an argument encoding layer.

Given an entity e ∈ E, the entity encoder h Γ is usually implemented as a simple embedding layer h Γ (e) [Γ] e , where Γ is an embedding matrix BID32 .

For pre-trained embeddings, the embedding matrix is fixed.

Note that other encoding mechanisms are conceivable, such as; recurrent, graph convolution BID26 b) or convolutional neural networks BID13 .

DistMult DISTMULT BID43 represents each relation r using a parameter vector Θ ∈ R k , and scores a link of type r between (h s , h o , h r ) using the following scoring function: DISPLAYFORM0 where ·, ·, · denotes the tri-linear dot product.

ComplEx COMPLEX BID38 ) is an extension of DISTMULT using complex-valued embeddings while retaining the mathematical definition of the dot product.

In this model, the scoring function is defined as follows: DISPLAYFORM1 k are complex vectors, x denotes the complex conjugate of x, Re (x) ∈ R k denotes the real part of x and Im (x) ∈ C k denotes the imaginary part of x.

Let D {(τ 1 , y 1 ), . . .

, (τ n , y n )} denote a set of labelled triples, where τ i s i , p i , o i , and y i ∈ {0, 1} denotes the corresponding label, denoting that the fact encoded by the triple is either true or false.

We can assume D is generated by a corresponding generative model.

In the following, we propose two alternative generative models.

In Figure 1 's graphical model, we assume that the Knowledge Graph was generated according to the following generative model.

Let V E × R × E the space of possible triples.

where τ s, p, o , and h τ [h s , h p , h o ] denotes the sampled embedding representations of s, o ∈ E and p ∈ R.Note that, in this model, the embeddings are sampled for each triple.

As a consequence, the set of latent variables in this model is H {h τ | τ ∈ E × R × E}.The joint probability of the variables p θ (H, D) is defined as follows: DISPLAYFORM0 The marginal distribution over D is then bounded as follows, with respect to our variational distribution q: DISPLAYFORM1 Proposition 1 As a consequence, the log-marginal likelihood of the data, under the Latent Fact Model, is bounded by: DISPLAYFORM2 Proof.

We refer the reader to the Appendix 6 for a detailed proof LFM's ELBO.Assumptions: LFM model assumes each fact of is a randomly generated variable, as well as a mean field variational distribution and that each training example is independently distributed.

Note that this is an enormous sum over |D| elements.

However, this can be approximated via Importance Sampling, or Bernoulli Sampling BID6 .

DISPLAYFORM0 By using Bernoulli Sampling, ELBO can be approximated by: DISPLAYFORM1 where p θ (s τ = 1) = b τ can be defined as the probability that for the coefficient s τ each positive or negative fact τ is equal to one (i.e is included in the ELBO summation).

The exact ELBO can be recovered from setting b τ = 1.0 for all τ .

We can define a probability distribution of sampling from D + and D − -similarly to Bayesian Personalised Ranking BID33 , we sample one negative triple for each positive one -we use a constant probability for each element depending on whether it is in the positive or negative set.

Proposition 2 The Latent Fact models ELBO can be estimated similarly using a constant probability for positive or negative samples, we end up with the following estimate: DISPLAYFORM2 where DISPLAYFORM3

In Figure 2 's graphical model, we assume that the Knowledge Graph was generated according to the following generative model.

Let V E × R × E the space of possible triples.

DISPLAYFORM0 The marginal distribution over D is then defined as follows: DISPLAYFORM1 Proposition 3 The log-marginal likelihood of the data, under the Latent Information Model, is the following: DISPLAYFORM2 Proof.

We refer the reader to the Appendix 6 for a detailed proof LIM's ELBO.Assumptions: LIM assumes each variable of information is randomly generated, as well as a mean field variational distribution and that each training example is independently distributed.

This leads to a factorisation of the ELBO that seperates the KL term from the observed triples, making the approximation to the ELBO through Bernoulli sampling simpler, as the KL term is no longer approximated and instead fully computed.

Similarly to Section 3.1.1, by using Bernoulli Sampling the ELBO can be approximated by: DISPLAYFORM0 Which can be estimated similarly using a constant probability for positive or negative samples, we end up with the following estimate:Proposition 4 The Latent Information Models ELBO can be estimated similarly using a constant probability for positive or negative samples, we end up with the following estimate: DISPLAYFORM1 where DISPLAYFORM2

Variational Deep Learning has seen great success in areas such as parametric/non-parametric document modelling BID29 BID30 and image generation BID25 .

Stochastic variational inference has been used to learn probability distributions over model weights BID3 , which the authors named "Bayes By Backprop".

These models have proven powerful enough to train deep belief networks BID40 , by improving upon the stochastic variational bayes estimator BID25 , using general variance reduction techniques.

Previous work has also researched word embeddings within a Bayesian framework BID40 , as well as researched graph embeddings in a Bayesian framework .

However, these methods are expensive to train due to the evaluation of complex tensor inversions.

Recent work by BID0 BID8 show that it is possible to train word embeddings through a variational Bayes BID2 framework.

KG2E proposed a probabilistic embedding method for modelling the uncertainties in KGs.

However, this was not a generative model.

BID42 argued theirs was the first generative model for knowledge graph embeddings.

However, their work is empirically worse than a few of the generative models built under our proposed framework, and their method is restricted to a Gaussian distribution prior.

In contrast, we can use any prior that permits a re-parameterisation trick -such as a Normal (Kingma & Welling, 2013a) or von-Mises distribution .Later, BID27 ) proposed a generative model for graph embeddings.

However, their method lacks scalability as it requires the use of the full adjacency tensor of the graph as input.

Moreover, our work differs in that we create a framework for many variational generative models over multi-relational data, rather than just a single generative model over uni-relational data BID27 BID20 .

In a different task of graph generation, similar models have been used on graph inputs, such as variational auto-encoders, to generate full graph structures, such as molecules BID35 BID28 .Recent work by BID9 ) constructed a variational path ranking algorithm, a graph feature model.

This work differs from ours for two reasons.

Firstly, it does not produce a generative model for knowledge graph embeddings.

Secondly, their work is a graph feature model, with the constraint of at most one relation per entity pair, whereas our model is a latent feature model with a theoretical unconstrained limit on the number of existing relationships between a given pair of entities.

We run each experiment over 500 epochs and validate every 50 epochs.

Each KB dataset is separated into 80 % training facts, 10% development facts, and 10% test facts.

During the evaluation, for each fact, we include every possible corrupted version of the fact under the local closed world assumption, such that the corrupted facts do not exist in the KB.

Subsequently, we make a ranking prediction of every fact and its corruptions, summarised by mean rank and filtered hits@m.

During training Bernoulli sampling to estimate the ELBO was used, with linear warm-up BID7 , compression cost BID3 , ADAM (Kingma & Ba, 2014) Glorot's initialiser for mean vectors BID19 and variance values initialised uniformly to embedding size −1 .

We experimented both with a N (0, 1) and a N (0, embedding size −1 ) prior on the latent variables.

Table 1 shows definite improvements on WN18 for Variational ComplEx compared with the initially published ComplEX.

We believe this is due to the well-balanced model regularisation induced by the zero mean unit variance Gaussian prior.

Table 1 also shows that the variational framework is outperformed by existing non-generative models, highlighting that the generative model may be better suited at identifying and predicting symmetric relationships.

WordNet18 (Bordes et al., 2013b ) (WN18) is a large lexical database of English.

WN18RR is a subset with only asymmetric relations.

FB15K is a large collaboratively made dataset which covers a vast range of relationships and entities, with FB15K-257 BID37 , with 257 relations -a significantly reduced number from FB15K due to being a similarly refined asymmetric dataset.

We now compare our model to the previous state-of-the-art multi-relational generative model TransG BID42 , as well as to a previously published probabilistic embedding method KG2E (similarly represents each embedding with a multivariate Gaussian distribution) on the WN18 dataset.

Table 2 makes clear the improvements in the performance of the previous state-of-the-art generative multi-relational knowledge graph model.

LFM has marginally worse performance than the state-of-the-art model on raw Hits@10.

We conjecture two reasons may cause this discrepancy.

Firstly, the fact the authors of TransG use negative samples provided only (True negative examples), whereas we generated our negative samples using the local closed world assumption (LCWA) .

Secondly, we only use one negative sample per positive to estimate the Evidence Lower Bound using Bernoulli sampling, whereas it is likely they used significantly more negative samples.

Scoring Table 1 : Filtered and Mean Rank (MR) for the models tested on the WN18, WN18RR, and FB15K datasets.

Hits@m metrics are filtered.

Variational written with a "V".

*Results reported from BID39 ) and **Results reported from BID13 for ComplEx model.

"-" in a table cell equates to that statistic being un-reported in the models referenced paper.

Dataset Scoring Function MR Raw Hits@ Filtered Hits @ Raw Filter 10 1 3 10 WN18 KG2E 362 345 0.805 --0.932 TransG (Generative) BID42 345 FORMULA7 We split the analysis into the predictions of subject ((?, r, o)) or object ((s, r, ?)) for each test fact.

Note all results are filtered predictions, i.e., ignoring the predictions made on negative examples generated under LCWA -using a randomly corrupted fact (subject or object corruption) as a negative example.

TAB3 shows that the relation "_derivationally_related_form", comprising 34% of test subject predictions, was the most accurate relation to predict for Hits@1 when removing the subject from the tested fact.

Contrarily, "_member_of_domain_region" with zero Hits@1 subject prediction, making up less than 1% of subject test predictions.

However, "_member_meronym " was the least accurate and prominent (8% of the test subject predictions) for subject Hits@1.

We learn from this that even for a near state-of-the-art model there is a great deal of improvement to be gained among asymmetric modelling.

TAB3 , as before the relation "_derivationally_related_form" was the most accurate relation to predict Hits@1.

TAB3 as it highlights the Latent Information Model's inability to achieve a high Hits@1 performance predicting objects for the "_hypernym" relation, which is significantly hindering model performance as it is the most seen relation in the test set-its involvement in 40% of object test predictions.

These results hint at the possibility that the slightly stronger results of WN18 are due to covariances in our variational framework able to capture information about symbol frequencies.

We verify this by plotting the mean value of covariance matrices, as a function of the entity or predicate frequencies FIG2 ).

The plots confirm our hypothesis: covariances for the variational Latent Information Model grows with the frequency, and hence the LIM would put a preference on predicting relationships between less frequent symbols in the knowledge graph.

This also suggests that covariances from the generative framework can capture genuine information about the generality of symbolic representations.

We project the high dimensional mean embedding vectors to two dimensions using Probabilistic Principal Component Analysis (PPCA) BID36 to project the variance embedding vectors down to two dimensions using Non-negative Matrix Factorisation (NNMF) BID15 .

Once we have the parameters for a bivariate normal distribution, we then sample from the bivariate normal distribution 1,000 times and then plot a bi-variate kernel density estimate of these samples.

By visualising these two-dimensional samples, we can conceive the space in which the entity or relation occupies.

We complete this process for the subject, object, relation, and a randomly sampled corrupted entity (under LCWA) to produce a visualisation of a fact, as shown in FIG3 .

Figure 4 displays a clustering of the subject, object, and predicate that create a positive (true) fact.

We also observe a separation between the items which generate a fact and a randomly sampled (corrupted) entity which is likely to create a negative (false) fact.

The first test fact "(USA, Commonbloc0, Netherlands)" shows clear irrationality similarity between all objects in the tested fact, i.e. the vectors are pointing towards a south-east direction.

We can also see that the corrupted entity Jordan is quite a distance away from the items in the tested fact, which is good as Jordan does not share a common bloc either USA or Netherlands.

We used scoring functions which measure the similarity between two vectors, however, for more sophisticated scoring functions which distance/ similarity is not important to the end result we would unlikely see such interpretable images.

This analysis of the learnt distributions is evidence to support the notion of learnt probabilistic semantics through using this framework.

We have successfully created a framework allowing a model to learn embeddings of any prior distribution that permits a re-parametrisation trick via any score function that permits maximum likelihood estimation of the scoring parameters.

The framework reduces the parameter by one hyperparameter -as we typically would need to tune a regularisation term for an l1/ l2 loss term, however as the Gaussian distribution is self-regularising this is deemed unnecessary for matching state-ofthe-art performance.

We have shown, from preliminary experiments, that these display competitive results with current models.

Overall, we believe this work will enable knowledge graph researchers to work towards the goal of creating models better able to express their predictive uncertainty.

The score we acquire at test time even through forward sampling does not seem to differ much compared with the mean embeddings, thus using the learnt uncertainty to impact the results positively is a fruitful path.

We would also like to see additional exploration into various encoding functions, as we used only the most basic for these experiments.

We would also like to see more research into measuring how good the uncertainty estimate is.

We would like to thank all members of the Machine Reading lab for useful discussions.

<|TLDR|>

@highlight

Working toward generative knowledge graph models to better estimate predictive uncertainty in knowledge inference. 