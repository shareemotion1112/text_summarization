We propose to use a meta-learning objective that maximizes the speed of transfer on a modified distribution to learn how to modularize acquired knowledge.

In particular, we focus on how to factor a joint distribution into appropriate conditionals, consistent with the causal directions.

We explain when this can work, using the assumption that the changes in distributions are localized (e.g. to one of the marginals, for example due to an intervention on one of the variables).

We prove that under this assumption of localized changes in causal mechanisms, the correct causal graph will tend to have only a few of its parameters with non-zero gradient, i.e. that need to be adapted (those of the modified variables).

We argue and observe experimentally that this leads to faster adaptation, and use this property to define a meta-learning surrogate score which, in addition to a continuous parametrization of graphs, would favour correct causal graphs.

Finally, motivated by the AI agent point of view (e.g. of a robot discovering its environment autonomously), we consider how the same objective can discover the causal variables themselves, as a transformation of observed low-level variables with no causal meaning.

Experiments in the two-variable case validate the proposed ideas and theoretical results.

The data used to train our models is often assumed to be independent and identically distributed (iid.), according to some unknown distribution.

Likewise, the performance of a model is typically evaluated using test samples from the same distribution, assumed to be representative of the learned system's usage.

While these assumptions are well analyzed from a statistical point of view, they are rarely satisfied in many real-world applications.

For example, a medical diagnosis system trained on historical data from one hospital might perform poorly on patients from another institution, due to shifts in distribution.

Ideally, we would like our models to generalize well and adapt quickly to out-of-distribution data.

However, this comes at a price -in order to successfully transfer to a novel distribution, one might need additional information about data at hand.

In this paper, we are not considering assumptions on the data distribution but rather on how it changes (e.g., when going from a training distribution to a transfer distribution, possibly resulting from some agent's actions).

We focus on the assumption that the changes are sparse when the knowledge is represented in an appropriately modularized way, with only one or a few of the modules having changed.

This is especially relevant when the distributional change is due to actions by one or more agents, because agents intervene at a particular place and time, and this is reflected in the form of the interventions discussed in the causality literature (Pearl, 2009; Peters et al., 2016) , where a single causal variable is clamped to a particular value or a random variable.

In general, it is difficult for agents to influence many underlying causal variables at a time, and although this paper is not about agent learning as such, this is a property of the world that we propose to exploit here, to help discovering these variables and how they are causally related to each other.

In this context, the causal graph is a powerful tool because it tells us how perturbations in the distribution of intervened variables will propagate to all other variables and affect their distributions.

As expected, it is often the case that the causal structure is not known in advance.

The problem of causal discovery then entails obtaining the causal graph, a feat which is in general achievable only with strong assumptions.

One such assumption is that a learner that has learned to capture the correct structure of the true underlying data-generating process should still generalize to the case where the structure has been perturbed in a certain, restrictive way.

This can be illustrated by considering the example of temperature and altitude (Peters et al., 2017) : a learner that has learned to capture the mechanisms of atmospheric physics by learning that it makes more sense to predict temperature from the altitude (rather than vice versa) given training data from (say) Switzerland will still remain valid when tested on out-of-distribution data from a less mountainous country like (say) the Netherlands.

It has therefore been suggested that the out-of-distribution robustness of predictive models be used to guide the inference of the true causal structure (Peters et al., 2016; 2017) .

How can we exploit the assumption of localized change?

As we explain theoretically and verify experimentally here, if we have the right knowledge representation, then we should get fast adaptation to the transfer distribution when starting from a model that is well trained on the training distribution.

This arises because of our assumption that the ground truth data generative process is obtained as the composition of independent mechanisms, and that very few ground truth mechanisms and parameters need to change when going from the training distribution to the transfer distribution.

A model capturing a corresponding factorization of knowledge would thus require just a few updates, a few examples, for this adaptation to the transfer distribution.

As shown below, the expected gradient on the unchanged parameters would be near 0 (if the model was already well trained on the training distribution), so the effective search space during adaptation to the transfer distribution would be greatly reduced, which tends to produce fast adaptation, as found experimentally.

Thus, based on the assumption of small change in the right knowledge representation space, we can define a meta-learning objective that measures the speed of adaptation, i.e., a form of regret, in order to optimize the way in which knowledge should be represented, factorized and structured.

This is the core idea presented in this paper.

Returning to the example of temperature and altitude: when presented with out-of-distribution data from the Netherlands, we expect the correct model to adapt faster given a few transfer samples of actual weather data collected in the Netherlands.

Analogous to the case of robustness, the adaptation speed can then be used to guide the inference of the true causal structure of the problem at hand, possibly along with other sources of signal about causal structure.

Contributions.

We first verify on synthetic data that the model that correctly captures the underlying causal structure adapts faster when presented with data sampled after a performing certain interventions on the true two-variable causal graph (which is unknown to the learner).

This suggests that the adaptation speed can indeed function as score to assess how well the learner fits the underlying causal graph.

We then use a smooth parameterization of the considered causal graph to directly optimize this score in an end-to-end manner.

Finally, we show in a simple setting that the score can be exploited to disentangle the correct causal variables given an unknown mixture of the said variables.

As an illustrative example of the proposed ideas, let us consider two discrete random variables A and B, each taking N possible values.

We assume that A and B are correlated, without any hidden confounder.

Our goal is to determine whether the underlying causal graph is A ??? B (A causes B), or B ??? A. Note that this underlying causal graph cannot be identified from observational data from a single (training) distribution p only, since both graphs are Markov equivalent for p (Verma & Pearl, 1991) ; see Appendix B. In order to disambiguate between these two hypotheses, we will use samples from some transfer distributionp in addition to our original samples from the training distribution p.

Without loss of generality, we can fix the true causal graph to be A ??? B, which is unknown to the learner.

Moreover, to make the case stronger, we will consider a setting called covariate shift (Rojas-Carulla et al., 2018; Quionero-Candela et al., 2009 ), where we assume that the change (again, unknown to the learner) between the training and transfer distributions occurs after an intervention on the cause A. In other words, the marginal of A changes, while the conditional p(B | A) does not, i.e. p(B | A) =p(B | A).

Changes on the cause will be most informative, since they will have direct effects on B. This is sufficient to fully identify the causal graph (Hauser & B??hlmann, 2012) .

In order to demonstrate the advantage of choosing the causal model A ??? B over the anti-causal B ??? A, we can compare how fast the two models can adapt to samples from the transfer distributio?? p.

We quantify the speed of adaptation as the log-likelihood after multiple steps of fine-tuning via (stochastic) gradient ascent, starting with both models trained on a large amount of data from the training distribution.

In Figure 1 (see Section 3.3 for the experimental setup), we can see that the model corresponding to the underlying causal model adapts faster.

Moreover, the difference is more significant when adapting on a small amount of data, of the order of 10 to 30 samples from the transfer distribution.

We will make use of this property as a noisy signal to infer the direction of causality, which here is equivalent to choosing how to modularize the joint distribution.

Since we are using gradient ascent for the adaptation, let's first inspect how the gradients of the log-likelihood wrt.

each module behave under the transfer distribution.

Proposition 1.

Let G be a causal graph, and p a (training) distribution that factorizes according to G, with parameters ??.

Letp be a second (transfer) distribution that also factorizes according to G. If the training and transfer distributions have the same conditional probability distributions for all V i but a subset C (e.g. the transfer distribution is the result of an intervention on the nodes in C):

then the expected gradient w.r.t.

the parameters ?? i such that V i / ??? C of the log-likelihood under the transfer distribution will be zero

Proposition 1 (see proof in Appendix C.1) suggests that if both distributions factorize according to the correct causal graph, then only the parameters of the mechanisms that changed between the training and transfer distributions need to be updated.

This effectively reduces the number of parameters that need to be adapted compared to any other factorization over a different graph.

It also affects the number of examples necessary for the adaptation, since the sample complexity of a model grows approximately linearly with the VC-dimension (Ehrenfeucht et al., 1989; Vapnik & Chervonenkis, 1971) , which itself also grows approximately linearly with the number of parameters (for linear models and neural networks; Shalev-Shwartz & Ben-David, 2014 ).

Therefore we argue that the performance on the transfer distribution (in terms of log-likelihood) will tend to improve faster if it factorizes according to the correct causal graph, an assertion which may not be true for every graph but that we can test by simulations.

Recall that in our example on two discrete random variables (each taking say N values), we assumed that the underlying causal model is A ??? B, and the transfer distribution is the result of an intervention on the cause A. If the model we learn on the training distribution factorizes according to the correct graph, then only N ??? 1 free parameters should be updated to adapt to the shifted distribution, accounting for the change in the marginal distributionp(A), since the conditionalp(B | A) = p(B | A) stays invariant.

On the other hand, if the model factorizes according to the anti-causal graph B ??? A, then the parameters for both the marginalp(B) and the conditionalp(A | B) must be adapted.

Assuming there is a linear relationship between sample complexity and the number of free parameters, the sample complexity would be O(N 2 ) for the anti-causal graph, compared to only O(N ) for the true underlying causal graph A ??? B.

Since the speed of adaptation to some transfer distribution is closely related to the right modularization of knowledge, we propose to use it as a noisy signal to iteratively improve inference of the causal structure from data.

Moreover, we saw in Figure 1 that the gap between correct and incorrect models is largest with a small amount of transfer data.

In order to compare how fast some models adapt to a change in distribution, we can quantify the speed of adaptation based on their accumulated online performance after fine-tuning with gradient ascent on few examples from the transfer distribution.

More precisely, given a small "intervention" dataset D int = {x t } T t=1 fromp, we can define the online likelihood as

where ??

G aggregates all the modules' parameters in G after t steps of fine-tuning with gradient ascent, with learning rate ??, starting from the maximum-likelihood estimate?? M L G (D obs ) on a large amount of data D obs from the training distribution p. Note that, in addition to its contribution to the update of the parameters, each data point x t is also used to evaluate the performance of our model so far; this is called a prequential analysis (Dawid, 1984) , also corresponding to sequential cross-validation (Gingras et al., 1999) .

From a structure learning perspective, the online likelihood (or, equivalently, its logarithm) can be interpreted as a score we would like to maximize, in order to recover the correct causal graph.

We can draw an interesting connection between the online log-likelihood, and a widely used score in structure learning called the Bayesian score (Heckerman et al., 1995; Geiger & Heckerman, 1994) .

The idea behind this score is to treat the problem of learning the structure from a fully Bayesian perspective.

If we define a prior over graphs p(G) and a prior p(?? G | G) over the parameters of each graph G, the Bayesian score is defined as score B (G ;

(4) In the online likelihood, the adapted parameters ?? (t) G act as summary of past data x 1:t???1 .

Eq. (3) can be seen as an approximation of the marginal likelihood in Equation (4), where the posteriors over the parameters p(?? G | x 1:t???1 , G) is approximated by the point estimate ?? (t) G .

Therefore, the online log-likelihood provides a simple way to approximate the Bayesian score, which is often intractable.

Due to the super-exponential number of possible Directed Acyclic Graphs (DAGs) over n nodes, the problem of searching for a causal structure that maximizes some score is, in general, NP-hard (Chickering, 2002a) .

However, we can parametrize our belief about causal graphs by keeping track of the probability for each directed edge to be present.

This provides a smooth parametrization of graphs, which hinges on gradually changing our belief in individual binary decisions associated with each edge of the causal graph.

This allows us to define a fully differentiable meta-learning objective, with all the beliefs being updated at the same time by gradient descent.

In this section, we study the simplest version of this idea, applied to our example on two random variables from Section 2.

Recall that here, we only have two hypotheses to choose from: either A ??? B or B ??? A. We represent our belief of having an edge connecting A to B with a structural parameter ?? such that p(A ??? B) = ??(??), where ??(??) = 1/(1 + exp(?????)) is the sigmoid function.

We propose, as a meta-transfer objective, the negative log-likelihood R (a form of regret) over the mixture of these two models, where the mixture parameter is given by ??(??):

This meta-learning mixture combines the online adaptation likelihoods of each model over one meta-example or episode (specified by a D int ???p), rather than considering and linearly mixing the per-example likelihoods as in ordinary mixtures.

In the experiments below, after each episode involving T examples D int from the transfer distributio?? p, we update ?? by doing one step of gradient descent, to reduce the regret R. Therefore, in order to update our belief about the edge A ??? B, the quantity of interest is the gradient of the objective R with respect to the structural parameter, ???R/?????.

This gradient is pushing ??(??) towards the posterior probability that the correct model is A ??? B, given the evidence from the transfer data: Proposition 2.

The gradient of the negative log-likelihood of the transfer data D int in Equation (5) wrt.

the structural parameter ?? is given by

where p(A ??? B | D int ) is the posterior probability of the hypothesis A ??? B (when the alternative is B ??? A).

Furthermore, this can be equivalently written as

where

is the difference between the online log-likelihoods of the two hypotheses on the transfer data D int .

The proof is given in Appendix C.2.

Note how the posterior probability is basically measuring which hypothesis is better explaining the transfer data D int overall, along the adaptation trajectory.

This posterior depends on the difference in online log-likelihoods ???, showing the close relation between minimizing the regret R and maximizing the online log-likelihood score.

The sign and magnitude of ??? have a direct effect on the convergence of the meta-objective.

We can show that the meta-objective is guaranteed to converge to one of the two hypotheses.

Proposition 3.

With stochastic gradient descent (and an appropriately decreasing learning rate)

, where the gradient steps are given by Proposition 2, the structural parameter converges towards

This proposition (proved in Appendix C.3) shows that optimizing ?? is equivalent to picking the hypothesis that has the smallest regret (or fastest convergence), measured as the accumulated loglikelihood of the transfer dataset D int during adaptation.

The distribution over datasets D int is similar to a distribution over tasks in meta-learning.

This analogy with meta-learning also appears in our gradient-based adaptation procedure, which is linked to existing methods like the first-order approximation of MAML (Finn et al., 2017) , and its related algorithms (Grant et al., 2018; Kim et al., 2018; Finn et al., 2018) .

The pseudo-code for the proposed algorithm is given in Algorithm 1.

This smooth parametrization of the causal graph, along with the definition of the meta-transfer objective in Equation (5), can be extended to graphs with more than 2 variables.

This general formulation builds on the bivariate case, where decisions are binary for each individual edge of the graph.

See Appendix F for details and a generalization of Proposition 2; the structure of Algorithm 1 remains unchanged.

Experimentally, this generalization of the meta-transfer objective proved to be effective on larger graphs (Ke et al., 2019) , in work following the initial release of this paper.

To illustrate the convergence result from Proposition 3, we experiment with learning the structural parameter ?? in a bivariate model.

Following the setting presented in Section 2.1, we assume in all our experiments that A and B are two correlated random variables, and the underlying causal model (unknown to the algorithm) is fixed to A ??? B. Recall that both variables are observed, and there is no hidden confounding factor.

Since the correct causal model is A ??? B, the structural parameter should converge correctly, with ??(??) ??? 1.

The details of the experimental setups, as well as details about the models, can be found in Appendix D.

Algorithm 1 Meta-learning algorithm for learning the structural parameter Require: Two graph candidates G = A ??? B and G = B ??? A Require: A training distribution p that factorizes over the correct causal graph 1: Set the initial structural parameter ?? = 0 equal belief for both hypotheses 2: Sample a large dataset D obs from the training distribution p 3: Pretrain the parameters of both models with maximum likelihood on D obs 4: for each episode do for t = 1, . . .

, T do 8:

Accumulate the online log-likelihood for both models L A???B and L B???A as they adapt 9: Do one step of gradient ascent for both models: ??

Compute the regret R(D int )

Compute the gradient of the regret wrt.

?? (see Proposition 2) 12: Do one step of gradient descent on the regret w.r.t.

??

Reset the models' parameters to the maximum likelihood estimate on D obs

We first experiment with the case where both A and B are discrete random variables, taking N possible values.

In this setting, we explored how two different parametrizations of the conditional probability distributions (CPDs) might influence the convergence of the structural parameter.

In the first experiment, we parametrized the CPDs as multinomial logistic CPDs (Koller & Friedman, 2009) , maintaining a tabular representation of the conditional probabilities.

For example, the conditional distribution p(B | A) is represented as

where the parameter ?? is an N ?? N matrix.

We used a similar representation for the other marginal and conditional distributions p(A), p(B) and p(A | B).

In a second experiment, we used structured CPDs, parametrized with multi-layer perceptrons (MLPs) with a softmax nonlinearity at the output layer.

The advantage over a tabular representation is the ability to share parameters for similar contexts, and reduces the overall number of parameters required for each module.

This would be crucial if either the number of categories N , or the number of variables, increased significantly.

In Figure 2 , we show the evolution of ??(??), which is the model's belief of A ??? B being the correct causal model, as the number of episodes increases, for different values of N .

As expected, the structural parameter converges correctly to ??(??) ??? 1, within a few hundreds episodes.

This observation is consistent on both experiments, regardless of the parametrization of the CPDs.

Interestingly, the structural parameter tends to converge faster with a larger value of N and a tabular representation, illustrating the effect of the parameter counting argument described in Section 2.2, which is stronger as N increases.

Precisely when generalization is more difficult (too many parameters and too few examples), we get a stronger signal about the better modularization.

We also experimented with A and B being continuous random variables, where they follow either multimodal distributions, or they are linear and Gaussian.

Similar to Figure 2 , we found that the structural parameter ??(??) consistently converges to the correct causal model as well.

See Appendix D.3 and Appendix D.4 for details about these experiments.

So far, we have assumed that all the variables in the causal graph are fully observed.

However, in many realistic scenarios for learning agents, the learner might only have access to low-level observations (e.g. sensory-level data, like pixels or acoustic samples), which are very unlikely to be individually meaningful as causal variables.

In that case, our assumption that the changes in distributions are localized might not hold at this level of observed data.

To tackle this, we propose to follow the deep learning objective of disentangling the underlying causal variables (Bengio et al., 2013) , and learn a representation in which the variables can be meaningfully be cause or effect for each other.

Our approach is to jointly learn this representation, as well as the causal graph over the latent variables.

We consider the simplest setting where the learner maps raw observations to a hidden representation space with two causal variables, via an encoder E. The encoder is trained such that this latent space helps to optimize the meta-transfer objective described in Section 3.

We consider the parameters of the encoder, as well as ?? (see Section 3.2), as part of the set of structural meta-parameters to be optimized.

We assume that we have two raw observed variables (X, Y ), generated from the true causal variables (A, B) via the action of a ground truth decoder D (or generator network), that the learner is not aware of.

This allows us to still have the ability to intervene on the underlying causal variables (e.g. to shift from training to transfer distributions) for the purpose of conducting experiments, while the learner only sees data from (X, Y ).

The encoder E must be learned to undo this action of the decoder, and thereby recover the true causal variables up to symmetries.

The faded components to the left are hidden to the model.

In this experiment, we only want to validate the proposed meta-objective as a way to recover a good encoder, and we assume that both the decoder D and the encoder E are rotations, whose angles are ?? D and ?? E respectively.

The encoder maps the raw observed variables (X, Y ) to the latent variables (U, V ), over which we want to infer the causal graph.

Similar to our experiments in Section 3.3, we assume that the underlying causal graph is A ??? B, and the transfer distributionp (now over (X, Y )) is the result of an intervention over A. Therefore, the encoder should ideally recover the structure U ??? V in the learned latent space, along with the angle of the encoder ?? E = ????? D .

However, since the encoder is not uniquely defined, V ??? U might also be a valid solution, if the encoder is

Details about the experimental setup are provided in Appendix E. In Figure 4 , we consider that the learner succeeds, since both structural parameters converge to one of the two options.

This shows how minimizing the meta-transfer objective can disentangle (here in a very simple setting) the ground-truth variables.

Although this paper focuses on the causal graph, the proposed objective is motivated by the more general question of discovering the underlying causal variables and their dependencies to explain the environment of the learner, and make it possible for that learner to plan appropriately under changes due to interventions, either from self or from another agent.

The discovery of underlying explanatory variables has come under different names, in particular the notion of disentangling underlying variables (Bengio et al., 2013; Locatello et al., 2019) , and studied in the causal setting (Chalupka et al., 2015; 2017) and domain adaptation (Magliacane et al., 2018) .

This paper is also related to meta-learning (Finn et al., 2017; Finn, 2018; Alet et al., 2018; Dasgupta et al., 2019) , to Bayesian structure learning (Koller & Friedman, 2009; Heckerman et al., 1995; Daly et al., 2011; Chickering, 2002b) , causal discovery (Pearl, 1995; Tian & Pearl, 2001; Pearl, 2009; Bareinboim & Pearl, 2016; Peters et al., 2017) and how non-stationarity makes causal discovery easier (Zhang et al., 2017) .

Please see Appendix A for a longer discussion of related work.

We have established, in very simple bivariate settings, that the rate at which a learner adapts to sparse changes in the distribution of observed data can be exploited to infer the causal structure, and disentangle the causal variables.

This relies on the assumption that with the correct causal structure, those distributional changes are localized.

We have demonstrated these ideas through theoretical results, as well as experimental validation.

The source code for the experiments is available here:

This work is only a first step in the direction of causal structure learning based on the speed of adaptation to modified distributions.

On the experimental side, many settings other than those studied here should be considered, with different kinds of parametrizations, richer and larger causal graphs (see already Ke et al. (2019)

based on a first version of this paper), or different kinds of optimization procedures.

Also, more work needs to be done in exploring how the proposed ideas can be used to learn good representations in which the causal variables are disentangled.

Scaling up these ideas would permit their application towards improving the way learning agents deal with non-stationarities, and thus improving sample complexity and robustness of these agents.

An extreme view of disentangling is that the explanatory variables should be marginally independent, and many deep generative models (Goodfellow et al., 2016) , and Independent Component Analysis models (Hyv??rinen et al., 2001; Hyv??rinen et al., 2018) , are built on this assumption.

However, the kinds of high-level variables that we manipulate with natural language are not marginally independent: they are related to each other through statements that are usually expressed in sentences (e.g. a classical symbolic AI fact or rule), involving only a few concepts at a time.

This kind of assumption has been proposed to help discover relevant high-level representations from raw observations, such as the consciousness prior (Bengio, 2017) , with the idea that humans focus at any particular time on just a few concepts that are present to our consciousness.

The work presented here could provide an interesting meta-learning approach to help learn such encoders outputting causal variables, as well as figure out how the resulting variables are related to each other.

In that case, one should distinguish two important assumptions: the first one is that the causal graph is sparse, which a common assumption in structure learning (Schmidt et al., 2007) ; the second is that the changes in distributions are sparse, which is the focus of this work.

As stated already by Bengio et al. (2013) , and clearly demonstrated by Locatello et al. (2019) , assumptions, priors, or biases are necessary to identify the underlying explanatory variables.

The latter paper (Locatello et al., 2019 ) also reviews and evaluates recent work on disentangling, and discusses different metrics that have been proposed.

Chalupka et al. (2015; 2017) recognize the potential and the challenges underlying causal representation learning.

Closely related to our efforts is (Chalupka et al., 2017) , which places a strong focus on the coalescence of low (e.g. sensory) level observations (microvariables) to higher level causal variables (macrovariables), albeit in a more observational setting.

There also exists an extensive literature on learning the structure of Bayesian networks from (observational) data, via score-based methods ( , 2002b) , whereas we propose a continuous and fully-differentiable alternative.

While most of these approaches only rely on observational data, it is sometimes possible to extend the definition of these scores to interventional data (Hauser & B??hlmann, 2012) .

The online-likelihood score presented here supports interventional data as its main feature.

Some identifiability results exist for causal models with purely observational data though (Peters et al., 2017) , based on specific assumptions on the underlying causal graph.

However, causal discovery is more natural under local changes in distributions (Tian & Pearl, 2001 ), similar to the setting used in this paper.

Pearl's seminal work on do-calculus (Pearl, 1995; 2009; Bareinboim & Pearl, 2016) lays the foundation for expressing the impact of interventions on causal graphical models.

Here we are proposing a meta-learning objective function for learning the causal structure (without hidden variables), requiring mild assumptions such as localized changes in distributions and faithfulness of the causal graph, in contrast to the stronger assumptions necessary for these identifiability results.

Our work is also related to other recent advances in causation, domain adaptation, and transfer learning.

Magliacane et al. (2018) have sought to identify a subset of features that leads to the best predictions for a variable of interest in a source domain, such that the conditional distribution of that variable given these features is the same in the target domain.

Zhang et al. (2017) also examine non-stationarity and find that it makes causal discovery easier.

Our adaptation procedure, using gradient ascent, is also closely related to gradient-based methods in meta-learning (Finn et al., 2017; Finn, 2018) .

Alet et al. (2018) proposed a meta-learning algorithm to recover a set of specialized modules, but did not establish any connections to causal mechanisms.

More recently, Dasgupta et al.

(2019) adopted a meta-learning approach to perform causal inference on purely observational data.

Suppose that A and B are two discrete random variables, each taking N possible values.

We show here that the maximum likelihood estimation of both models A ??? B and B ??? A yields the same estimated distribution over A and B. The joint likelihood on the training distribution is not sufficient to distinguish the causal model between the two hypotheses.

If p is the training distribution, let

Let D obs be a training dataset.

If N

The estimated distributions for each model A ??? B and B ??? A, under the maximum likelihood estimator, will be equal:p

To illustrate this result, we also experiment with maximizing the likelihood for each modules for both models A ??? B and B ??? A with SGD.

In Figure B .1, we show the difference in log-likelihoods between these two models, evaluated on training and test data sampled from the same distribution, during training.

We can see that while the model A ??? B fits the data faster than the other model (corresponding to a positive difference in the figure), both models achieve the same log-likelihoods at convergence.

This shows that the two models are indistinguishable, in the limit, based on data sampled from the same distribution, even on test data.

Let us restate Proposition 1 here for convenience: Proposition 1.

Let G be a causal graph, and p a (training) distribution that factorizes according to G, with parameters ??.

Letp be a second (transfer) distribution that also factorizes according to G. If the training and transfer distributions have the same conditional probability distributions for all V i but a subset C (e.g. the transfer distribution is the result of an intervention on the nodes in C):

) then the expected gradient w.r.t.

the parameters ?? i such that V i / ??? C of the log-likelihood under the transfer distribution will be zero

Proof.

For V i / ??? C, we can simplify the expected gradient as follows:

where Equation (20) arises from our assumption that the conditional distribution of V i given its parents in G does not change between the training distribution p and the transfer distributionp.

Moreover, the last equality arises from the marginalization

C.2 GRADIENT OF THE STRUCTURAL PARAMETER Let us restate Proposition 2 here for convenience: Proposition 2.

The gradient of the negative log-likelihood of the transfer data D int in Equation (5) wrt.

the structural parameter ?? is given by

where p(A ??? B | D int ) is the posterior probability of the hypothesis A ??? B (when the alternative is B ??? A).

Furthermore, this can be equivalently written as

where

is the difference between the online log-likelihoods of the two hypotheses on the transfer data D int .

Proof.

First note that, using Bayes rule,

where

is the online likelihood of the transfer data under the mixture, so that the regret is R(D int ) = ??? log M .

For Equation (27) , note that if

where ??

A???B encapsulates the information about the previous datapoints {a s , b s } t???1 s=1 in the graph A ??? B, through some adaptation procedure.

Since we only consider the two hypotheses A ??? B and B ??? A, we also have

Therefore, the gradient of the regret wrt.

the structural parameter ?? is

which concludes the first part of the proof.

Moreover, given Equation (35), it is sufficient to show that p(A ??? B | D int ) = ??(?? + ???) to prove the equivalent formulation in Equation (25).

Using the logit function ?? ???1 (z) = log z 1???z , and the expression in Equation (28), we have

Let us restate Proposition 3 here for convenience: Proposition 3.

With stochastic gradient descent (and an appropriately decreasing learning rate)

, where the gradient steps are given by Proposition 2, the structural parameter converges towards

Proof.

We are going to consider the fixed point of gradient descent (a point where the gradient is zero), since we already know that SGD converges with an appropriately decreasing learning rate.

Let us introduce some notations to simplify the algebra:

, so that the regret is R(D int ) = ??? log M .

We define P 1 and P 2 as (see also the proof in Appendix C.2)

Framing the stationary point in terms of p rather than ?? gives us a constrained optimization problem, with inequality constraints ???p ??? 0 and p ??? 1 ??? 0, and no equality constraint.

Applying the KKT conditions to this problem, with constraint functions ???p and p ??? 1, gives us

We already see from equations (48) & (49) that if p ??? (0, 1) (i.e. excluding 0 and 1), we must have

Let us study that case first, and show that it leads to an inconsistent set of equations (thus, forcing the solution to be either p = 0 or p = 1).

Let us rewrite the gradient to highlight p in it (using Proposition 2):

This derivation is valid since we assume that p ??? (0, 1).

Suppose that p = 0; multiplying both sides of Equation (50) by p gives

For this equation to be satisfied, we need

This would, however, correspond to p = 0, which contradicts our assumption.

Similarly, assuming that p = 1, we can also multiply both sides of Equation (50) by 1 ??? p and get

Again, this can only be true if L A???B = M almost surely, meaning that p = 1, contradicting our assumption.

We conclude that the solutions p ??? (0, 1) are not possible because they would lead to inconsistent conclusions, which leaves only p = 0 or p = 1.

In order to assess the performance of our meta-learning algorithm, we applied it on generated data from three different domains: discrete random variables, multimodal continuous random variables and multivariate Gaussian-distributed variables.

In this section, we describe the setups for all three experiments, along with additional results to complement the results descrbed in Section 3.3.

Note that in all these experiments, we fix the ground-truth structure as A ??? B, and only perform interventions on the cause A.

We consider a bivariate model, where both random variables are sampled from a categorical distribution.

The underlying ground-truth model can be described as

with ?? A a probability vector of size N , and ?? B|a a probability vector of size N , which depends on the value of the variable A. In our experiment, each random variable can take one of N = 10 or N = 100 values.

Since we are working with only two variables, the only two possible models are:

We build 4 different modules, corresponding to every possible marginal and conditional distributions.

Here, we use multinomial logistic Conditional Probability Distributions (Koller & Friedman, 2009 ).

The modules' definition, and their corresponding parameters, are shown in Table D .1.

In order to get a set of initial parameters, we first train all 4 modules on a training distribution (p in the main text).

This distribution corresponds to a fixed choice of ?? A emphasizes the fact that this defines the distribution prior to an intervention, with the mechanism p(B | A) being unchanged by the intervention.

These probability vectors are sampled randomly from a uniform Dirichlet distribution:

Given this training distribution, we can sample a large dataset of samples

for the ground truth model, using ancestral sampling.

Using D obs , we can train all 4 modules using gradient ascent on the log-likelihood (or any other advanced first-order optimizer, like RMSprop).

The parameters ?? A , ?? B|A , ?? B & ?? A|B of the maximum likelihood estimate will be used as the initial parameters for the adaptation on the new transfer distribution.

Similar to the way we defined the training distribution, we can define a transfer distribution (p in the main text) as an intervention on the random variable A. In this experiment, this accounts for changing the distribution of A, that is with a new probability vector ?? (2) A , also sampled from a uniform Dirichlet distribution ??

To perform adaptation on the transfer distribution, we also sample a smaller transfer dataset

, with T m. In our experiment, we used T = 20 datapoints, following the observation from Section 2.1.

We consider a bivariate model, similar to the one defined in Appendix D.1, where each random variable is sampled from a categorical distribution.

Instead of expressing the CPDs in tabular form, we use structured CPDs, parametrized with multi-layer perceptrons (MLPs).

In our experiment, all the MLPs have only one hidden layer with H = 8 hidden units, with a ReLU non-linearity, and the output layer has a softmax non-linearity.

To avoid any modeling bias, we assume that the ground-truth model is also parametrized by MLPs, such that

where 0 is a vector of size N will all zeros, and

Again, to define the training distribution, we first fix the parameters W

(1)

A and W B .

We use randomly initialized networks for the training distribution, with the parameters sampled using the He initialization.

We train all the modules using maximum likelihood on a large dataset of training samples D obs , to get the initial set of parameters for the adaptation on the transfer distribution.

We also define a transfer distribution as the result of an intervention on A. In this experiment, this means sampling a new set of parameters W (2) A , still as a randomly initialized network.

We sample a transfer dataset

, with T = 20 datapoints.

Consider a family of joint distributions p ?? (A, B) over the causal variables A and B, defined by the following structural causal model (SCM):

where f is a randomly generated spline, and the noise N B is sampled iid.

from the unit Gaussian distribution.

To obtain the spline, we sample K points {x k } K k=1 uniformly spaced from the interval [???8, 8] , and another K points {y k } K k=1 uniformly randomly from the interval [???8, 8] .

This yields K pairs {x k , y k } K k=1 , which make the knots of a second-order spline.

We choose K = 8 points in our experiments.

A

Module Parameters Dimension

We select p 0 (A, B) as the training distribution, from which we sample a large dataset D obs using ancestral sampling.

Similar to the earlier experiments, this dataset is used to get the initial set of parameters for the adaptation on the transfer distribution.

The MDNs are fitted with gradient descent, while the GMMs are learned via Expectation Maximization.

The transfer distribution is the result of an intervention on A, where we shift the distribution p ?? (A) with ?? sampled uniformly in [???1, 1].

In Figure D .1, we plot samples from the training distribution (?? = 0), as well as two transfer distributions (?? = ??4).

Figure D.1: Samples from the training (blue) and transfer (red and green) distributions, from an SCM generated with the procedure described above.

The red datapoints are sampled from p ???4 (A, B), the green datapoints from p 4 (A, B), and the blue datapoints from p 0 (A, B).

The structural regret R(??) is now minimized with respect to ?? for 500 iterations (updates of ??).

In the notation of Algorithm 1, these are the iterations over the number of episodes.

Figure D. 2 shows the evolution of ??(??) as training progresses.

This is expected, given that we expect the causal model to perform better on the transfer distributions, i.e. we expect L A???B > L B???A in expectation.

Consequently, assigning a larger weight to L A???B optimizes the objective (see Proposition 3).

Finally as a sanity check, we test the experimental set-up described above on a linear SCM with additive Gaussian noise.

In this setting, it is well known that the causal structure cannot be discovered from observations alone Peters et al. (2017) and one must rely on the transfer distribution tell cause from effect.

To that end, we repeat the experiment in Figure D .2 with the following amendments: (a) we replace the non-linear spline with a linear curve ( Figure D. 3), and (b) in addition to training the structural parameter by adapting the A ??? B and B ??? A models to multiple interventional distributions, we train it by "adapting" the said model to the train distribution, where the latter serves as a baseline.

Figure D .4 shows that using multiple transfer (i.e. interventional) distributions ("With Interventions") enables causal discovery, as opposed to the model trained with a single observational distribution.

This confirms that our method indeed relies on the interventional distributions to discover the causal structure.

The blue curve corresponds to the setting where we make use of interventions, whereas the orange curve corresponds to one where we do not (i.e. use a single distribution).

The shaded bands show the standard deviation over 40 runs (of both pre-and meta-training).

We find (as expected) that causal discovery fails without interventions but succeeds when transfer distributions are available.

In this experiment, the two variables A and B are vector-valued, taking values in R d .

The ground-truth causal model is given by

where ?? ??? R d , ?? 0 ??? R d and ?? 1 ??? R d??d .

?? and ?? are two d ?? d covariance matrices.

In our experiment, d = 100.

Once again, we want to identify the correct causal direction between A and B.

To do so, we consider two models A ??? B and B ??? A parametrized with Gaussian distributions.

The details of the modules' definitions, as well as their parameters, is given in Table D .4.

Note that each covariance matrix is parametrized using the Cholesky decomposition.

Table D .4: Description of the 2 models, with the parametrization of each module, for a bivariate model with linear Gaussian variables.

Model A ??? B and Model B ??? A both have the same number of parameters 2d 2 + 3d.

To build the training distribution, we draw ?? (1) , ?? 0 and ?? 1 from a Gaussian distribution, and ??

( 1) and ?? from an inverse Wishart distribution.

The transfer distribution is the result of an intervention on A, meaning that the marginalp(A) changes.

To do so, we sample new parameters ?? (2) from a Gaussian distribution, and ?? (2) from an inverse Wishart distribution as well.

Unlike the previous experiments, we are not conduction any pre-training on actual data from the training distribution.

Instead, we fix the parameters of both models to their exact values, according to the ground truth distribution.

For Model A ??? B, this can be done easily.

For the Model B ??? A, we compute the exact parameters analytically using Bayes rule.

This can be seen as the maximum likelihood estimate in the limit of infinite data.

In Figure D .5, we show that, after 200 episodes, ??(??) converges to 1, indicating the success of the method on this particular task.

In this section, we describe an experimental setting where the conditional p(B | A) is perturbed while the distribution of the cause, p(A), is left unchanged.

To that end, consider a set-up similar to that in Section D.3:

where f 0 is a randomly generated spline and N B is sampled iid.

from the unit Gaussian distribution and the cause variable A is sampled from the uniform distribution supported on [???8, 8] .

To induce soft-interventions, we modify the SCM as follows.

Consider the knots {a i , b i } 5 i=1 of the order 3 spline f 0 ; we obtain a new spline f int by randomly perturbing the b-coordinate of the knots, where the perturbations are sampled from another uniform distribution 1 .

Using the perturbed spline f int instead of f 0 in Equation (74) results in a new SCM, from which we generate a single transfer distribution (i.e. for a single episode).

In Figure D .6 we plot samples from three such transfer distributions.

The models used are identical to those detailed in Appendix D.3 and are trained on the training SCM (corresponding base-spline f 0 ) with a large amount of samples (??? 3,000k).

The meta-training procedure differs in that (a) in every transfer episode, we create a new spline f int and sample a transfer distribution D int from the corresponding SCM, and (b) we use the following measure of adaptation:

where G is one of A ??? B or B ??? A. The meta-transfer objective in Equation (5) Failure case In addition to the result above, we also observed that using soft interventions on the effect B instead on changes on the marginal p(A) was sometimes failing to recover the correct causal graph.

Instead, the anti-causal graph (here B ??? A) was found, with high confidence.

We describe here one such experiment where using the meta-transfer objective failed at recovering the correct causal graph.

Our experimental setting is similar to the one described in Appendix D.1.

However instead of changing the marginal p(A), the conditional distribution p(B | A) changes and p(A) remains unchanged.

Following the notations in Appendix D.1, we have

where ??

B|a are the parameters of the conditional distribution before intervention, and ??

B|a its parameters after intervention.

We again sample data from both the training and transfer distributions to get datasets D obs and D int .

The different modules and their corresponding parameters are defined in Table D In Figure D .8, we show the evolution of the structural parameter ??(??), the model's belief that A ??? B is the correct causal model, as the number of episodes increases.

Unlike our previous experiments in Section 3.3, the structural parameter now converges to ??(??) ??? 0, corresponding to a strong belief that the model is B ??? A. We are therefore unable to recover the correct causal graph here under the assumption that p(B | A) changes.

Note that here the parameter counting argument from Section 2.2 clearly does not hold anymore, since the modules all use a tabular representation, and both models require the same order O(N 2 ) of updates to adapt to a transfer distribution.

The true latent causal variables (A, B) are sampled from the distribution described in Appendix D.3 (Equations (74) & (75)).

These variables are then mapped to observations (X, Y ) ??? p ?? (X, Y ) via a hidden (and unknown to the learner) decoder D = R ?? D , where R ?? is a rotation of angle ??.

The observations are then mapped to the hidden state (U, V ) ??? p ?? (U, V ) via the encoder E = R ?? E ; in this experiment, the angle ?? E is the only additional meta-parameter, besides the structural parameter ??.

The computational graph is depicted in Figure 3 .

In our experiment, ?? D = ?????/4 is fixed for all our observation and intervention datasets.

Interventional data is acquired by intervening on the latent variables (A, B), following the process described in Appendix D.3, and then mapping the data through the decoder D.

Since the underlying latent causal variables (A, B) are unobserved, we need to define the online likelihood over the recovered variables (U, V ) instead.

Analogous to how we defined the online likelihood in the fully observable case in Section 3, this is defined as

where

Note that here the online likelihood depends on the parameters of the encoder E (here, ?? E ).

Using this definition of the online likelihood that takes into account the encoder, the meta-transfer objective is also similar to the one defined in Equation (5):

(80) On the one hand, the gradient of R(D int ; ??, ?? E ) with respect to the structural parameter ?? can be computed using Proposition 2, similar to the fully observable case.

On the other hand, the gradient of the meta-transfer objective with respect to the meta-parameter ?? E is computed using backpropagation through the T updates of the parameters ?? G of the modules in Equation (79); this process is similar to backpropagation through time.

In our experiment, we did not observe any degenerate behaviour like vanishing gradients, due to the limited amount of interventional data (T = 5).

In Section 3.2, we defined the meta-transfer objective only in the context of bivariate models.

The challenge with learning the structure of graphs on n variables is that there is a super-exponential number of DAGs on n variables, making the problem of structure learning NP-hard (Chickering, 2002a) .

If we were to naively extend the meta-transfer objective to graphs on n > 2 variables, this would require adaptation of 2 O(n 2 ) different models (hypotheses), which is intractable.

Instead, we can decouple the optimization of the graph from the acyclicity constraint, since causal graphs can have cycles (Peters et al., 2017) .

This constraint can be enforced as an extra penalty to the meta-transfer objective (Zheng et al., 2018) .

We consider the problem of optimization on the graph as O(n 2 ) independent binary decisions on whether V j is a parent (or direct cause) of V i .

Motivated by the mechanism independence assumption (Parascandolo et al., 2017) , we propose a heuristic to learn the causal graph, in which we independently parametrize the binary probability p ij that V j is a parent of V i .

We can then define a distribution over graphs (or more precisely, their adjacency matrix B) as:

where p ij = ??(?? ij ).

We denote Pa B (V i ) as the parent set of V i in the graph defined by the adjacency matrix B (that is the nodes V j such that B ij = 1).

We can slightly rewrite the definition of the online-likelihood as in Section 3 to show the dependence on B:

where the second equality uses the factorization of p in the graph defined by B. Note that since the graph defined by B can contain cycles, the definition in Equation (83) involves the pseudolikelihood instead of the joint likelihood (which is defined as the product of individual conditional distributions only if the graph is a DAG).

The pseudolikelihood was shown to be a reasonable approximation of the true joint likelihood when maximizing the joint likelihood (which is performed here for adaptation; Koller & Friedman (2009) ).

Similar to the bivariate case, we want to consider a mixture over all possible graph structures, but where each component must explain the whole adaptation sequence.

We can generalize our definition of the regret as

Note, however, that this expectation is over the O(2 n 2 ) possible values of B, which is intractable.

We can rewrite the regret in a more convenient form: Proposition 4.

The regret R(D int ) defined in Equation (84) can be decomposed as

where B i is a row of the matrix B, and

:

, so that we can rewrite the regret as follows:

The structural parameters, here, are the O(n 2 ) scalars ?? ij .

Regardless of the intractability of the regret, we can still derive its gradient wrt.

each ?? ij .

The following proposition provides a direct extension of Proposition 2 to the case of multiple variables: Proposition 5.

The gradient of the regret R(D int ) wrt.

the structural parameter ?? ij is given by

where ??? ij is the difference in log-likelihoods of two mixture candidates, conditioning on the variable

Proof.

To simplify the notation, we remove the explicit dependence on the transfer distribution D int in this proof.

Recall from Proposition 4 that the regret can be written as

Using a conditional expectation, it follows that for any i, j

To simplify the notation, let us define E

(1) ij and E (0) ij the two conditional expectations of L Bi , conditioned on whether or not V j is a parent of V i in B

so that Equation (98) can be written as

Note that neither E (0) ij nor E

(1) ij depend on the structural parameter ?? ij .

Therefore we can now easily compute the gradient of R wrt.

?? ij only ???R ????? ij = ??? ??? ????? ij log ??(?? ij )E

(1) ij + (1 ??? ??(?? ij ))E (0) ij

If we substract ??(?? ij ) from this expression gives us

(1) ij + ??(?? ij )(1 ??? ??(?? ij ))E (0) ij

Denoting the previous expression as x, we can also easily compute 1 ??? x:

Using the logit function ?? ???1 (x) = log x 1???x , we can conclude that ?? ???1 ??(?? ij ) ??? ???R ????? ij = log ??(?? ij )E

(1) ij

(1 ??? ??(?? ij ))E (0) ij

= log ??(?? ij ) 1 ??? ??(?? ij )

+ log E

(1) ij ??? log E (0) ij (108)

While Proposition 5 gives an analytic form for the gradient of the regret wrt.

the structural parameters, computing it is still intractable, due to ??? ij .

However, we can still get an effecient stochastic gradient estimator from Proposition 4, which can be computed separately for each node of the graph (with samples arising only out of B i , the incoming edges of V i ): Proposition 6.

If we consider multiple samples of B in parallel, a biased but asymptotically unbiased (as the number K of these samples B (k) increases to infinity) estimator of the gradient of the overall regret with respect to the meta parameters can be defined as:

where the index (k) indicates the values obtained for the k-th draw of B.

Proof.

The gradient of the regret with respect to the meta-parameters ?? i of node i is

Note that with the sigmoidal parametrization of p(B i ), log p(B i ) = B ij log ??(?? ij ) + (1 ??? B ij ) log(1 ??? ??(?? ij ))

as in the cross-entropy loss.

Its gradient can similarly be simplified to

A biased, but asymptotically unbiased, estimator of ???R/????? ij is thus obtained by sampling K graphs (over which the means below are run):

where index (k) indicates the k-th draw of B, and we obtain a weighted sum of the individual binomial gradients weighted by the relative regret of each draw B

(k) i of B i , leading to Equation (110).

We can therefore adapt Algorithm 1 using the gradient estimate in Proposition 6 to update the structural parameters ?? ij , without having to explicitly compute the full regret R(D int ).

In addition to the gradient estimate provided by Proposition 6, we can also derive a Rao-Blackwellized (Rao, 1992; Blackwell, 1947) estimate of the gradient of the regret, based on the formulation derived in Proposition 5.

Proposition 7.

Let {B (k) } K k=1 be K binary matrices (corresponding to sample graphs), sampled from independent Bernoulli distributions depending on the structural parameters ?? ij

and their corresponding likelihoods L (k)

Bi .

A Monte-Carlo estimate of the log-likelihood difference ??? ij in Equation (94) is given by ??? (K) ij = log 1

where K (0) ij = {k ; B (k) ij = 0} and K

(1) ij = {k ; B (k) ij = 1} are (disjoint) sets of indices k, depending on the value of B (k) ij .

Based on this Monte-Carlo estimate of ??? ij , we can define an estimate of the gradient of the regret R wrt.

the structural parameter ?? ij by

Since we are using the online likelihood defined in Equation (3) as a measure of adaptation in our meta-transfer objective, it is reasonable to know if this measure is sound.

To validate this assumption, we are running an experiment similar to the one described in Section 2.1 and Figure 1 , using the same experimental setup on discrete variables described in Section 3.3.

However, instead of measuring the raw log likelihood on a validation set, we report the online likelihood L G (D int ) for both models in Figure G .1.

The online likelihoods are scaled by the number of transfer examples seen for visualization.

Similar to Figure 1 , we can see that the difference in online likelihoods for both models is most significant on a small amount of data.

@highlight

This paper proposes a meta-learning objective based on speed of adaptation to transfer distributions to discover a modular decomposition and causal variables.

@highlight

The paper shows that a model with the correct underlying structure will adapt faster to a causal intervention than a model with the incorrect structure.

@highlight

In this work, the authors proposed a general and systematic framework of meta-transfer objective incorporating the causal structure learning under unknown interventions.