Models of user behavior are critical inputs in many prescriptive settings and can be viewed as decision rules that transform state information available to the user into actions.

Gaussian processes (GPs), as well as nonlinear extensions thereof, provide a flexible framework to learn user models in conjunction with approximate Bayesian inference.

However, the resulting models may not be interpretable in general.

We propose decision-rule GPs (DRGPs) that apply GPs in a transformed space defined by decision rules that have immediate interpretability to practitioners.

We illustrate this modeling tool on a real application and show that structural variational inference techniques can be used with DRGPs.

We find that DRGPs outperform the direct use of GPs in terms of out-of-sample performance.

Models of user behavior are critical in many decision making problems and can be viewed as decision rules that transform state information (in set S) available to the user to actions (in set A).

Formally, a user model is a function f : S → A. Gaussian processes (GPs) employed to learn functions on the action/target space (henceforth target GPs or TGPs for short) can thus be used to place a prior on user models and identify a posterior distribution over them supported by data in conjunction with approximate Bayesian inference techniques (Blei et al., 2017; Beaumont, 2019) .

TGPs for user modeling would assume that user actions at a given set of finite states follow a multivariate Gaussian.

To capture non-Gaussian action distributions, one could apply GPs to learn functions in a transformed space that is not the target.

Examples include warped and chained GPs proposed in Snelson et al. (2004) and Saul et al. (2016) , respectively.

Extending this literature, we study the application of GPs in a transformed space defined by decision rules.

Such rules are known in several applications and depend on functions themselves.

Specifically, a user model based on a decision rule takes the form g : Π k P k × S → A, where the arguments are obtained using functions h k : S → P k , k = {1, . . .

, K} that map from S to transformed spaces P k , possibly different from the target space A. Each such function has immediate interpretability to a practitioner, and we model them using GPs.

We refer to such a user model {g, h 1 , ..., h k } as a decision-rule GP (DRGP).

To make the notion of DRGPs concrete in this short article, we focus on the problem faced by a firm providing services to store ethanol -a real application that motivated this work.

Suppose capacity (in gallons) is sold via annual contracts to N users.

The contract of user n specifies the maximum amount of ethanol that can be stored, denoted by C n .

User behavior corresponds to the injection of ethanol and the withdrawal of previously injected ethanol, which can be modeled as a time series.

The inventory I n,t in storage associated with user n at time t is the net of past injections and withdrawals.

A TGP approach would employ a GP to determine the next-period storage inventory level function I n,t+1 directly.

In contrast, we propose a DRGP that leverages a well-known decision rule based on injection and withdrawal threshold functions (Charnes et al., 1966; Secomandi, 2010) .

These threshold functions are learned as GPs instead of the (relatively less interpretable) inventory function.

We focus on the following research questions in the context of the ethanol storage application: (Q1) Can existing exact and approximate Bayesian inference techniques be used for inference with DRGP?

and (Q2) How does DRGP perform relative to TGP?

We answer these questions by executing numerical experiments based on real data of aggregated ethanol storage injection and withdrawals.

For Q1, we show that sparse vari-ational inference (Titsias, 2009; Hensman et al., 2013) , which can be applied to TGP on our data set, can also be used with DRGP, albeit heuristically, which is encouraging from an implementation standpoint.

For Q2, we find that DRGP implemented in this manner leads to lesser out-of-sample error than TGP on most of our datasets, in addition to being more interpretable to practitioners.

This preliminary finding is promising and suggests that applying GPs in the interpretable space of the decision rule threshold functions has potential value, which adds to the growing literature on interpretable machine learning and optimization (Letham et al., 2015; Bertsimas and Dunn, 2017) .

In addition, the improvements we report are based on the heuristic use of sparse variational inference with DRGPs, which bodes well for additional potential improvements from the development of new inference techniques targeting DRGPs.

Finally, several applications in energy, health care, and transportation, among other domains, have known interpratable decision rules, which can be leveraged in the DRGP framework proposed here.

Snelson et al. (2004) show that modeling data using a warped GP, which is a non-linear transformation (aka warping) of a GP, can enhance predictive performance.

Inference using a warped GP can be performed in closed-form provided the warping function satisfies certain properties, such as being invertible.

Lázaro-Gredilla (2012) consider the case where the warping function is not fixed a priori.

DRGPs differ from warped GPs as they are based on a potentially non-invertible transformation of multiple GPs.

Chained GPs by Saul et al. (2016) extend warped GPs by considering a likelihood function that factorizes across the data and is a general nonlinear transformation of multiple latent functions, each modeled as a GP.

Exact inference of chained GPs is not tractable in general and thus approximate inference techniques are used instead.

See Lázaro-Gredilla and Titsias ( Recent work has focused on finding a balance between the modeling generality (restrictiveness) of chained (warped) GPs and its associated challenging (straightforward) inference procedure.

For example, Tobar and Rios, 2019 extend a warped GP using a composition of simple functions and retain closed form inference.

DRGP is similar to a chained GP because its underlying decision rule is a nonlinear transformation g(·) of multiple GPs that model functions h 1 , . . .

, h K .

However, unlike a chained GP, each function h k is interpretable and not necessarily latent, which simplifies inference (see §3 for details).

For instance, in our energy storage application (where K = 2), the functions h 1 and h 2 correspond to injection and withdrawal threshold functions, respectively, and are fully or partially observable.

For each user n ∈ N , the most basic inventory update model capturing temporal dependencies can be written as: I n,t+1 = f n (I n,t , X n,t ) + n,t , where f n is the user specific transition function, X n,t is an exogenous variable with information such as commodity price at time t and other observable user characteristics (e.g., contract size C n ), and n,t is an i.i.d.

zero mean Gaussian noise variable.

We assume that the exogenous state evolves in a Marko-vian manner.

Given sufficient historical inventory usage data for each user, we can infer a posterior on f n for each user n separately (this is TGP).

While TGP can capture rich user behavior patterns, it is relatively less interpretable because the relationship between the previous inventory level (and other inputs) and the next inventory level can turn out be highly nonlinear, and using the corresponding posterior belief in downstream overbooking decisions may become cumbersome.

To alleviate this, we enhance the interpretibility by incorporating findings from prior literature (Charnes et al., 1966) .

In particular, it is known that a user (e.g., a merchant operator) makes injectionwithdrawal decisions using a two threshold decision rule structure (also called a double base-stock policy) under reasonable assumptions on the stochasticity of the exogenous variable X n,t :

, where f a , f b are two threshold functions and G is a known operational parameter.

Because this two-threshold structure for user behavior is interpretable (user injects if below a given threshold, withdraws if above another threshold, and holds still in between), we use this to define DRGP as follows:

n (I n,t , X n,t ) if f 2 n (I n,t , X n,t ) ≤ I n,t , where GP beliefs are placed on the threshold functions f 1 and f 2 (with noise terms associated with each function suppressed to ease notation).

Note that this composition of two functions f 1 and f 2 is non-invertible.

We use aggregate inventory level data (∼ 100 observations over 2 years) provided by a US ethanol storage operator.

The aggregate values are log-transformed and split into separate inventory levels for four users based on three different heuristics to simulate different types of injection-withdrawal behavior (see Appendix A).

As a result, we obtain three datasets with low, medium, and high variability of injection and withdrawal patterns.

We also vary the number of data points across all users, T , between 200 and 400.

These data sets also include information about the exogenous state vector X n,t that includes: (i) the lease capacity of each user; (ii) the spot and prompt-week futures prices for ethanol; and (iii) the prompt-week futures prices for corn and natural gas.

We obtain price data from Bloomberg.

At any time step, a user may inject, withdraw, or do nothing.

When a user injects, the inventory level X n,t+1 reached as a result of this injection is f 1 at X n,t , and as a result, this threshold value is observed but the withdrawal threshold is not.

Similarly, if there is a withdrawal action, f 2 at X n,t is observed while f 1 is not.

In other words, f 1 and f 2 are partially observable over time.

To avoid handling partial observability during inference, we partition the dataset based on when users inject and withdraw and learn the functions f 1 and f 2 , respectively, on the resulting subsets.

When computing posteriors in this manner, the ordering of f 1 and f 2 may not satisfy the condition f 1 ≤ f 2 that is implicitly assumed in the DRGP model.

To overcome this issue, we train a classifier to first predict if a user's decision is either injection or withdrawal and then employ the corresponding threshold to determine the next stage inventory level.

We compute posteriors on f 1 and f 2 using sparse (GP) variational inference (Titsias, 2009) with 10 inducing points and an Automatic Relevance Determination kernel (note that while one can also use exact GP regression here as an alternative, we chose the former for future scalability).

We use a gradient boosting decision tree based classifier.

Both TGP and DRGP can be combined with transfer learning by assuming a common component across users and a user specific latent variable.

We also consider such models and label them TGP-TL and DRGP-TL.

Details of these models and their accompanying inference procedures can be found in Appendix B.

In the first experiment, we answer the question (Q2) laid out in the Introduction, which seeks to relate the empirical performance of DRGP when compared to TGP.

In order to do so, we perform a training-validation partition of each dataset based on a 70% − 30% split.

The training data is then used to obtain the posteriors, for instance on f 1 n and f 2 n in the case of DRGP, for each user n = 1, ..., 4.

Subsequently these posteriors are used to predict the inventory levels in the validation data.

The mean and standard deviation of the root mean squared errors (RMSEs) for TGP and DRGP are displayed in Figure 4 .3 for two values of dataset size T .

When T equals 200, the RMSE of DRGP is smaller than TGP across all datasets.

As the number of data points T is increased to 400, this trend continues to be true for datasets 1 and 3 but is reversed for dataset 2.

Overall, we can conclude that DRGP obtains a lower RMSE than TGP in most cases, while also buying us interpretability.

In the second experiment, we investigate the value of transfer learning (where users share common priors).

Figure 2 (a-c) compares the models with and without transfer learning (T = 400).

We observe that incorporating transfer learning produces mixed results, suggesting that these datasets may lack a common user behavioral pattern that can be exploited.

Further, in figure 3 (a-d) , we illustrate the quality of one-step predictions of the transfer learning models (for all users in Dataset 3, with T = 400) as a function of one of the exogenous variables (spot price) in the validation data.

We observe that DRGP-TL can predict the out-of-sample log-inventory levels with higher accuracy and low uncertainty when compared to TGP-TL.

Our initial study of DRGPs shows that there is promise in leveraging decision rules to define non-linear transformations of GPs for user modeling in the ethanol storage application.

Extending this investigation to other real-world applications and developing inference procedures tailored to DRGP would be valuable.

For instance, we are in the process of developing an inference procedure that directly handles the partial observability of the thresholds and thus benefits from the full dataset, as opposed to partitioning the dataset as we did in this paper.

Other research directions being explored include: (i) the interplay between the structure of decision rules in a class of applications, their interpretability, and how this can be leveraged within inference procedures for DRGPs; and (ii) robust inference techniques for DRGP, where parameters are computed by optimizing a metric other than the (exact/approximate) likelihood function.

The ethanol application dataset contains the daily aggregate inventory level of a storage tank in the US, and the daily price of ethanol over a period of two years.

We consider weekly inventory levels to model the behavior of users, as suggested by practitioners.

There were 39 companies signed up in the system, with various contract sizes.

Assuming that users cluster into groups that have similar injection and withdrawal patterns, we created four users (essentially user types/groups) by assigned these companies to each group based on their contract size.

We break down the aggregate inventory levels to four user levels based on three heuristics.

These three heuristics are designed to test the performance of the four approaches we have for user modeling; and they capture low, medium, and high variance of injection-withdrawal patterns of users.

The first dataset is created by assigning fractions of the aggregate inventory to each user proportional to their contracted capacity, and simulates a system where the users have low variability.

In the second dataset, we simulate a setting where users have medium variability when interacting with the system.

This is captured by ensuring that the users do not change their inventory levels with probability 0.5, and change their inventory levels randomly between 0 and their rented capacity, again with probability 0.5.

Finally, to simulate a system where users interact with the system with high volatility, we make the users change their inventory level randomly from 0 to their rented capacity in every period, such that the aggregate of these individual inventory levels is equal to the aggregate inventory level.

Sparse GP for Scalability:

For TGP and DRGP, we rely on variational sparse Gaussian process based inference procedure Titsias (2009); Hensman et al. (2013) .

Inducing point methods involve introducing M T inducing points at locations Z = {z i } M i=1 with corresponding function values given by u i = f (z i ) such that:

where f is the vector of function evaluations at the T observation points.

Using this approach, we are able to approximate the posterior GP with a variational distribution that only depends on the inducing points by obtaining a lower bound on the marginal likelihood.

Transfer Learning: TGP-TL modifies TGP by assuming that user specific latent variables and a common target function together drive the inventory updates of all users (Wang and Neal, 2012; Damianou and Lawrence, 2015; Dai et al., 2017) .

That is, I n,t+1 = f (I n,t , X n,t , γ n ) + n,t , where f is a common target function across users that maps the triple (I n,t , X n,t , γ n ) to the next inventory level I n,t+1 , and γ n is a user specific latent variable.

We can jointly infer a posterior belief on f (we fix this to be a GP) and γ n (which we take to be Gaussian distributed) using LVMOGP Dai et al. (2017) .

Common temporal patterns of all users can now be captured by f , while idiosyncratic aspects of each user can be captured using γ n .

Similarly, DRGP can be extended to DRGP-TL, where we have common threshold functions for all users and user-specific latent variables to capture user heterogeneity.

The graphical models for TGP-TL and DRGP-TL are illustrated in The inference procedure for transfer learning extensions of TGP-TL and DRGP-TL involves handling the joint distribution with respect to the latent variables Γ = {γ 1 , .., γ N } and the common function (two functions in DRGP-TL).

The following independence assumption is made in the variational approximation for tractability:

where q(f ) is a GP and q(Γ) = Π N n=1 N (γ n |µ n , Σ n ).

Below, we show the evidence lower bound (ELBO) in TGP-TL:

E q(f ) [log p(I n,t |f )] − KL(q(u)||p(u)) − KL(q(Γ)||p(Γ)), where we use Figure 4 in the second inequality.

Following Hoffman et al. (2013) and other prior works, we maximize the evidence lower bound (ELBO) which provides a lower bound for the log-marginal likelihood of observed data, and jointly optimize with respect to the model hyper-parameters and the variational parameters as suggested in Saemundsson et al. (2018) .

<|TLDR|>

@highlight

We propose a class of user models based on using Gaussian processes applied to a transformed space defined by decision rules