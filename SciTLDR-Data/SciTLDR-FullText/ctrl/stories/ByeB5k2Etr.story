Many approaches to causal discovery are limited by their inability to discriminate between Markov equivalent graphs given only observational data.

We formulate causal discovery as a marginal likelihood based Bayesian model selection problem.

We adopt a parameterization based on the notion of the independence of causal mechanisms which renders Markov equivalent graphs distinguishable.

We complement this with an empirical Bayesian approach to setting priors so that the actual underlying causal graph is assigned a higher marginal likelihood than its alternatives.

Adopting a Bayesian approach also allows for straightforward modeling of unobserved confounding variables, for which we provide a variational algorithm to approximate the marginal likelihood, since this desirable feat renders the computation of the marginal likelihood intractable.

We believe that the Bayesian approach to causal discovery both allows the rich methodology of Bayesian inference to be used in various difficult aspects of this problem and provides a unifying framework to causal discovery research.

We demonstrate promising results in experiments conducted on real data, supporting our modeling approach and our inference methodology.

Causal networks (CNs) are special Bayesian networks where all edges reflect causal relations (Pearl, 2009 ).

The aim of causal structure learning is identifying the CN underlying the observed data.

In this paper, we focus on the problem of scoring causal graphs using marginal likelihood in a way that identifies the unique causal generative graph.

Succeeding to do so is very valuable, since once the correct CN is selected, various causal inference tasks such as estimating causal effects or examining confounder distributions becomes straightforward in a Bayesian framework.

A central challenge in such an attempt, however, is adopting a prior selection policy that not only allows discriminating between Markov equivalent graphs but also assigns higher marginal likelihood score to the actual underlying CN.

The key notion underlying our solution to first part of this challenge is the widely accepted principle of independence of the cause-effect mechanisms (Janzing et al., 2012) , that is, the natural mechanisms that generate the cause and the effect (based on cause) must be independent of each other.

We embody this assumption by assuming the mutual independence of the parameters pertaining to cause and effect distributions in a Bayesian model, a line of reasoning that is natural to this modeling perspective, where parameters are modeled as random variables (Spiegelhalter et al., 1993; Heckerman et al., 1995; Geiger et al., 1997; Blei et al., 2003) .

By assigning independent priors to the cause and effect variables, we render them statistically independent.

Critically, this assignment of independent priors also breaks the likelihood equivalence between Markov equivalent graphs.

This is contrast to other ways of selecting independent priors such as the BDeu prior, which leads to assigning equal marginal likelihood to Markov equivalent graphs (Heckerman et al., 1995) .

As mentioned above, though breaking likelihood equivalence does not necessarily lead to assigning a higher marginal likelihood to the actual underlying CN, it is a prerequisite for doing so 1 .

The second part of the problem is adapting a prior selection policy that leads to assigning a higher marginal likelihood to the actual CN compared to its alternatives.

In this work, we use an empirical Bayesian approach in selecting the hyperparameters of the independent priors described above, as we learn the priors that lead to assigning higher marginal likelihood to the actual CN from labeled data.

The current approach is in the intersection of various other approaches in the literature, thereby combining many of their respective advantages (Spirtes and Zhang, 2016; Glymour et al., 2019) .

It is based on the notion of mechanism independence similar to Janzing et al. (2012) ; Zhang et al. (2015) , does not assume causal sufficiency similar to Silva et al. (2006) ; Shimizu et al. (2009) ; Janzing et al. ( , 2012 ; Zhang et al. (2015) ; Sch??lkopf et al. (2016) , can theoretically work on arbitrary graph structures that possibly include latent variables similar to Spirtes et al. (1993) , and can discriminate between Markov equivalent structures similar to Shimizu et al. (2006) ; Zhang and Hyv??rinen (2008); Hoyer et al. (2009); Janzing et al. (2012); Zhang et al. (2015) .

Our approach diverges from other Bayesian methods (Stegle et al., 2010; Shimizu and Bollen, 2014; Zhang et al., 2016) in various dimensions such as by being able to distinguish between Markov equivalent causal graphs, using marginal likelihood (or approximations thereof) instead of surrogate scores such as BIC, or being able to model non-linear relationships.

In Section 2, we introduce an example model for continuous observations and latent categorical confounders.

To approximate the marginal likelihood in graphs which include latent confounders, we present a variational inference algorithm in Section 3.

After testing our approach on various real data sets in Section 4, we present our conclusions and further avenues of research in Section 5.

A general causal graph G(V G , E G ) is a combination of a vertex set V G , which is the set of observed and latent random variables, and a set of directed edges E G ??? V G ?? V G where directed edges imply immediate cause-effect relationships between these variables.

Let {x 1 , . . . , x n , . . . , x N } ??? V G denote the set of continuous random variables, and similarly {r 1 . . .

, r k , . . . , r K } ??? V G denote the discrete latent variables of the network where each x n and each r k are defined in the domains X n and R k , respectively.

The set of parent vertices of a vertex v ??? V G is denoted by ??(v), while we denote its continuous parents by x ??(v) , and discrete parents by r ?? (v) .

For the scope of this text, we specify conditional distributions for the graphs as follows: we assume categorical distributions on the discrete variables r 1:K and linear basis functions models with Gaussian noise on the continuous variables x 1:N .

Though these choices are by no means mandatory for our framework, we define latent variables as categorical.

Furthermore, we restrict our attention to the graphical structures that do not include a continuous variable as a parent of a categorical variable for inferential convenience (Heckerman et al., 1995) , and construct the following generative model for T independent and identically distributed observations from the network G:

where 1 ??? t ??? T , ?? is an arbitrary basis function with the convention ??({}) = 1, and

's are the parameters of the conditional distributions.

Namely, ?? k is the conditional distribution table of r k , w n is the weights of the basis functions, and ?? n is the precision parameter of the conditional distribution of x n .

Notice that declaring parameters as random variables simplifies the notion of independent cause-effect mechanisms as follows: Since the conditional distributions are the functions of the parameters, independence of the conditional distributions boils down to the independence of the parameters.

Therefore, we complete our generative model by defining independent conjugate prior distributions on the parameters

???n, r ??(xn) :

where ?? k|r ??(r k ) , m n|r ??(xn) , ?? n|r ??(xn) , a n|r ??(xn) , b n|r ??(xn) are the prior parameters, i.e. hyperparameters, of our generative model.

Variational Bayesian inference (VB) (Beal et al., 2006 ) is a technique where an intractable posterior distribution P is approximated by a variational distribution Q via minimizing Kullback-Leibler divergence KL(Q||P).

In the context of Bayesian model selection, minimization of the KL(Q||P) corresponds to establishing a tight lower bound for the marginal log-likelihood, which we refer to as evidence lower bound (ELBO).

This correspondence is due to the following decomposition of marginal log-likelihood log p(x

where P = p(r 1:T 1:K , ?? 1:K , ?? 1:N , w 1:N | x 1:T 1:N ) is the full posterior distribution, and ELBO is denoted by B P [Q] .

In a typical scenario of VB, Q is assumed to be a member of a restricted family of distributions.

In its most common form, also known as mean-field approximation, Q is assumed to factorize over some partition of the latent variables, in a way that is reminiscent to a rank-one approximation in the space of distributions Q(r 1:T 1:K , ?? 1:K , ?? 1:N , w 1:N ) = q(r 1:T 1:K ) q(?? 1:K , ?? 1:N , w 1:N ) ELBO is then maximized with respect to Q which is restricted to the class of factorized distributions.

Due to conjugacy, maximization of Q results in further factorized variational distributions which also belong to the same family as the prior

To calculate variational parameter updates, we need to calculate the expected sufficient statistics.

In its final form, our variational algorithm becomes equivalent to iteratively calculating the expected sufficient statistics and updating the parameters.

The explicit forms for the variational parameters and ELBO can be found in Appendix C.

In Section 4.1 we test the performance of our approach in bivariate causal discovery.

Then in Section 4.2 we identify the cardinality and distribution of a latent confounder in a multivariate data set, exemplifying the versatility of a Bayesian approach to causality.

In the first part we measured the accuracy of VB for the causal direction determination problem.

The data set in this part is CEP (Mooij et al., 2016) , frequently used in causal discovery research, which includes 100 data sets, vast majority of which is bivariate.

For the hyperparameters of the model, we created 36 different settings by varying the critical hyperparameters systematically.

We detail this hyperparameter creation process in the Appendix D.1.

In making a decision between two causal directions in a given hyperparameter setting, we choose the model which obtains a higher ELBO 2 .

We tested our algorithm on the data set by using 10 ?? 3 cross-validation.

That is, for each test, we separated the data set into three, detected the hyperparameter setting (of 36) that obtained the best accuracy score on the first two thirds, and tested our model on the last third of the data set, which corresponds to an empirical Bayesian approach to prior selection.

We conducted the same process two more times, each fold becoming the test set once.

We conducted this split and tested randomly 10 times.

We report the accuracy and AUC values according to these 10 runs.

the CEP data set, we obtained a mean accuracy of .78??.09 and AUC score of .84??.13 (the values following the mean values correspond to 68% CI) where the accuracy and AUC calculations are performed by using the weights mentioned by Mooij et al. (2016) .

Mooij et al. (2016) also compared most recent methods on their performance on the data set; our results correspond to a state-of-the-art performance in bivariate causality detection.

Using a different data set, we next examine the ability of our approach to identify a latent confounder.

For this purpose, we use the smallest database in the Thyroid data set from the UCI repository (Dheeru and Karra Taniskidou, 2017) .

This data involves five different diagnostic measurements from patients with low, normal, and high thyroid activity.

This being a diagnostic data set, the causal structure is known, where the thyroid activity is the cause of the rest of the variables (Figure 1(a) ).

In our experiments we ignore the thyroid activity variable, thus it becomes a latent confounder.

This way we can test how well our approach identifies the latent confounder.

To assess our method's performance, we first examine whether the latent variable cardinality our method favors corresponds to the cardinality of the actual variable that we held out.

Figure 1 (b) shows that the ELBO of the model is maximized at the latent cardinality which corresponds to the actual cardinality of thyroid activity variable (which is 3).

Then, to ascertain that the inferred latent variable indeed corresponds to thyroid activity variable, we compare the assignments of our model to actual patient thyroid activity levels.

The results demonstrate an accuracy of .93, thus we conclude that our method accurately identified the latent causal variable.

Overall, we show that Bayesian model selection is a promising framework that can facilitate causal research significantly both through conceptual unification and increased performance.

Given that Bayesian modeling is agnostic to specific variable types, conditional distributions, and to approximate inference methodology, the value of a successful Bayesian modeling approach for causal research is immense.

Though our empirical Bayesian approach to setting priors can be useful in various contexts (e.g. in data sets where only some of the bivariate causal directions are known), finding other principled ways of assigning (or integrating out) priors that do not require labeled data is an important direction for future research.

Conducting causal discovery with different variable types, and/or different distributions would also be beneficial for demonstrating current approach's viability in various contexts.

When constructing a generative model for causal inference, our aim is making Markov equivalent graph structures identifiable.

However, the model that is described only by Equations (1) and (2) is not necessarily identifiable (Shimizu et al., 2006; Hoyer et al., 2009) .

To be more precise, consider the case where we have two continuous variables and no latent categorical variable, which is equivalent to the following structural equation model:

One can also construct the following equivalent structural equation model in which the dependence structure is reversed:

These two models are not identifiable with the descriptions above, since they both correspond to linear models with Gaussian noise.

However, by assuming priors on the parameters we can break the symmetry and make these Markov equivalent models identifiable.

For instance, assuming Gaussian priors on the weights of the first model implies non-Gaussian priors on the second model, which makes these two models distribution inequivalent (Spirtes and Zhang, 2016).

Moreover, even when two Markov equivalent models are also distribution equivalent, choosing appropriate prior parameters that violate likelihood equivalence still makes them identifiable (Heckerman et al., 1995) .

Indeed, for a model with a parameterization as described, only a very specific choice of priors leads to likelihood equivalence between the Markov equivalent models (Geiger et al., 1997; Dawid et al., 1993) , and we will avoid following such a constraint.

Choosing arbitrary priors almost always leads to likelihood inequivalent, hence identifiable models.

In this section, we define the appropriate graphical structures for causal structure learning in the bivariate case.

As we stated in Section 1, we do not assume causal sufficiency and allow the existence of possibly many exogenous variables.

Luckily, we can combine the effects of exogenous variables into a single latent variable with an arbitrary cardinality.

As a result, the relationship between two observable dependent variables x 1 and x 2 boils down to one of three cases due to causal Markov condition (Hausman and Woodward, 1999):

2. x 2 causes x 1 , 3.

they do not cause each other, but a latent variable r 1 causes both of them.

Associated causal networks corresponding to each of these hypotheses are depicted in Figure 2 , where latent variable r 1 represents the overall effect of the all unobserved variables.

For the spurious relationship (Figure 2(a) ), marginally correlated variables x 1 and

(a) Spurious correlation.

Figure 2: Graphical models for bivariate causality.

x 2 become independent once the latent common cause variable r 1 is known.

However in direct causal relationships (Figures 2(b) and 2(c)), even when the latent common cause is known, two variables are still dependent and the direction of cause-effect relationship is implicit in the parameterization of the models.

The identifiability of these models resides in the fact that modelling parameters explicitly as random variables makes these graphs Markov inequivalent.

If we were considering only the marginal models of the observed variables, then we would end up with three Markov equivalent graphs.

However, including latent variables and independent parameters renders distinctive conditional independence properties for each graph.

For instance, when x 2 and r 1 are known, x 1 and the parameters of x 2 are dependent only in the case of x 1 ??? x 2 , or knowing r 1 makes x 1 and x 2 independent only if they have a spurious relationship.

These distinctive conditional independence properties are the underlying reasons making all of these graphs identifiable.

In this section, we supply the brief descriptions of the basic distributions that we mentioned in the main part of the manuscript.

which is equal to (z ??? 1)! for nonnegative integer z.

Gamma(??; a, b) = exp((a ??? 1) log ?? ??? b?? ??? log ??(a) + a log b)

where a is the shape and b is the rate parameter.

3.

Expected sufficient statistics:

4.

Cross entropy:

Here, ??(x) is the digamma function which is defined as ??(x) = d log ??(x) dx .

1.

Multivariate Beta function:

2.

Dirichlet density:

4.

Cross entropy:

where ?? is the mean parameter and ?? is the precision parameter, i.e. ?? ???1 is the variance.

B.1.5.

Multivariate Normal Distribution 1.

Multivariate Normal density:

where ?? is the mean vector and ?? is the precision matrix, i.e. ?? ???1 is the covariance matrix.

for any symmetric matrix A.

B.1.6.

Normal-Gamma Distribution 1.

Normal-Gamma density:

which can be equivalently decomposed into a marginal Gamma distribution and a conditional Normal distribution:

2.

Expected sufficient statistics:

3.

Cross entropy:

B.1.7.

Multivariate Normal-Gamma Distribution 1.

Multivariate Normal-Gamma density:

which can be equivalently decomposed into a marginal Gamma distribution and a conditional Multivariate Normal distribution:

2.

Expected sufficient statistics:

for any symmetric matrix A.

E N G(m,??,??,b) {??? log N G(w, ??; m, ??, a, b)} = ???a log b + log ??(a) ??? 1 2 log det(??) + 1 2 tr(?? ???1 ??) + M 2 log 2?? ??? a + M 2 ??? 1 (??(??) ??? logb) +?? b b +?? 2b (m ??? m) T ??(m ??? m)

In this section we summarize the basic conjugate models that are closely related to our example model.

2.

Posterior of ??:

where ?? * r = ?? r +

2.

Posterior of ?? and ??:

where

1.

Generative model:

An equivalent description with Normal-Gamma priors is

2.

Posterior of w and ??:

where

In this section, we will explicitly evaluate these equations to derive closed form expressions for the variational posteriors:

1.

We first simplify the (6) In order to keep the notation uncluttered, from now on we will omit the implicit subscripts in expectation operators.

So each individual factor q(r t 1:K ) above is equal to

2.

We now pursue the same strategy for the expression in (7) q(?? 1:K , ?? 1:N , w 1:

where each individual factor turns out to be

Finally, we match the coefficients of the sufficient statistics in above equations with the natural parameters and find the following variational parameters in terms of the expected sufficient statistics:

Update log?? t Update expected sufficient statistics

end for A simplified sketch of our variational inference algorithm VB-CN is also presented in Algorithm 1.

ELBO can be expressed as a sum of expectation terms most of which are in the form of negative cross entropy or negative entropy:

In this section we will evaluate each of those expectations explicitly.

We start our derivation with the trickier Gaussian log-likelihood term, then the rest of the expectations will correspond to negative cross entropy values of standard exponential family distributions:

Variational distribution Q treats r t 1:K and ?? 1:K as independent variables.

So, the expectations of the categorical log-likelihood terms admit the following form

The rest of the terms are related to cross entropy or entropy of the well-known exponential family distributions, and closed form expressions for them are supplied in Appendix B.

So here, we only modify these expressions by changing their parameters with the appropriate variational parameters.

1.

By using the negative cross entropy formulation in Appendix B.1.3 for categorical distributions:

r 1 ??? R 1 , m n|r 1 's were set to 0, and ?? n|r 1 's were set to 1 10 I each; while for all values of r 1 ??? R 1 , ?? 1 (r 1 )'s were set to 10.

We next describe the remaining hyperparameters with respect to the causal graph in Figure 2 (b) in which x 1 causes x 2 .

Their adaptation to other two graphs is straightforward due to symmetry.

The hyperparameters of the Gamma distributions, (a 1 , b 1 , a 2 , b 2 ) , from which the precision of the observed variables were drawn, were allowed to take different values with the condition that a n|r 1 ??? b n|r 1 at all times, but again every element of these vectors corresponding to different values of r 1 assumed to be constant within the vector.

This is because the mean of a Gamma distribution Gamma(a, b) is a/b and its variance is a/b 2 , therefore when b is allowed to take a greater value than a, this results in a close to zero precision value for the relevant distribution for the observed variable.

Obeying the constraint, the a and b's were allowed to take values among 1, 10, and 100 each.

The a parameter was not allowed to be larger than 100 since this leads to an equivalent sample size much larger than the sample size of certain data sets used in experiments, effectively rendering the observations unimportant.

The b parameter was not allowed to be smaller than 1 since this again implies extremely imprecise Gaussian distributions for the observed variables to which the Gamma distribution provided the precision variable.

The combinations with these constraints lead to a total of 36 sets of hyperparameters.

While doing model comparison in a hyperparameter setting, we expect several criteria to be satisfied for maintaining consistency.

For instance, in the spurious model (Figure 2(a) ) there is no reason to assign different priors on variables x 1 and x 2 .

Otherwise, just by permuting the labels of the pairs, we would obtain inconsistent marginal likelihoods.

Likewise, when the labels of a pair are permuted, e.g. 1 , x 1:T 2 ) given the relation x 1 ??? x 2 to be equal to the marginal likelihood of the permuted pair (x 1:T 1 ,x 1:T 2 ) given the relationx 2 ???x 1 .

The rule we used to solve inconsistency issues in such situations is the following: the prior parameters of two variables must be identical whenever the parental graphs of them are homomorphic.

So, if we are calculating the marginal likelihood of the relation x 1 ??? x 2 with a particular hyperparameter setting, say (a 1 = 100, b 1 = 10, a 2 = 10, b 2 = 1), then the corresponding consistent hyperparameter setting for x 2 ??? x 1 should be (a 1 = 10, b 1 = 1, a 2 = 100, b 2 = 10), whereas the corresponding consistent hyperparameters for the spurious relationship should be (a 1 = 100, b 1 = 10, a 2 = 100, b 2 = 10).

For this experiment, for each of 36 hyperparameter combinations, and for each rank values of |R 1 | = 1 to 5 for the linear model, a total of 3 different data pairs (one for each different graphical model) with 2000 observations were generated.

This amounted to a total of 540 data pairs.

For each synthetic data pair, the corresponding hyperparameters were used to compare the three hypotheses demonstrated in Figure 2 using the marginal likelihood estimate of the variational Bayes algorithm.

The resulting ROC curves can be seen in the Figure 3 .

With an overall accuracy of .961 and AUC of .998, the results demonstrate that our method can identify the data generating graph comfortably, given the correct hyperparameter settings.

The CEP data set is not labeled as to the spurious relationships, therefore it is not possible to conduct hyperparameter selection with cross-validation.

However, we ran the experiments again, this time including the spurious relationship hypothesis in the experiments, for all 36 parameter settings, and recorded the pairs for which the marginal likelihood of the spurious hypothesis was the highest.

We observed that, using the hyperparameter setting that achieved the highest accuracy in the previous experiment, these four data sets were found to be spurious: 19, 91, 92, and 98.

The scatter plots of these data sets are presented in Figure 4 .

Visual examination of the first three pairs reveals that, although each of these pairs are correlated, they can be separated into two clusters in which X and Y axes become independent.

In other words, once the confounding variables governing the cluster affiliations are decided, then the variables X and Y generated independently, so their correlation is indeed spurious.

As we lack the expertise, we do not know what these confounding variables correspond in reality, but the existence of such variables is evident from the scatter plots.

The case of the fourth spurious pair is slightly different than other correlated pairs.

The fourth pair consists of the measurements of initial and final speeds of a ball on a ball track where initial speed is thought as the cause of final speed.

However, our variational algorithm selected the spurious model with a latent variable having cardinality |R 1 | = 1, which actually corresponds to the marginal independence of X and Y .

Such an explanation makes sense considering the plot in Figure 4 , as the initial speed of the ball does not seem related to its final speed.

<|TLDR|>

@highlight

We cast causal structure discovery as a Bayesian model selection in a way that allows us to discriminate between Markov equivalent graphs to identify the unique causal graph.