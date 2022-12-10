Bayesian optimization (BO) is a popular methodology to tune the hyperparameters of expensive black-box functions.

Despite its success, standard BO focuses on a single task at a time and is not designed to leverage information from related functions, such as tuning performance metrics of the same algorithm across multiple datasets.

In this work, we introduce a novel approach to achieve transfer learning across different datasets as well as different metrics.

The main idea is to regress the mapping from hyperparameter to metric quantiles with a semi-parametric Gaussian Copula distribution, which provides robustness against different scales or outliers that can occur in different tasks.

We introduce two methods to leverage this estimation: a Thompson sampling strategy as well as a Gaussian Copula process using such quantile estimate as a prior.

We show that these strategies can combine the estimation of multiple metrics such as runtime and accuracy, steering the optimization toward cheaper hyperparameters for the same level of accuracy.

Experiments on an extensive set of hyperparameter tuning tasks demonstrate significant improvements over state-of-the-art methods.

Tuning complex machine learning models such as deep neural networks can be a daunting task.

Object detection or language understanding models often rely on deep neural networks with many tunable hyperparameters, and automatic hyperparameter optimization (HPO) techniques such as Bayesian optimization (BO) are critical to find the good hyperparameters in short time.

BO addresses the black-box optimization problem by placing a probabilistic model on the function to minimize (e.g., the mapping of neural network hyperparameters to a validation loss), and determine which hyperparameters to evaluate next by trading off exploration and exploitation through an acquisition function.

While traditional BO focuses on each problem in isolation, recent years have seen a surge of interest in transfer learning for HPO.

The key idea is to exploit evaluations from previous, related tasks (e.g., the same neural network tuned on multiple datasets) to further speed up the hyperparameter search.

A central challenge of hyperparameter transfer learning is that different tasks typically have different scales, varying noise levels, and possibly contain outliers, making it hard to learn a joint model.

In this work, we show how a semi-parametric Gaussian Copula can be leveraged to learn a joint prior across datasets in such a way that scale issues vanish.

We then demonstrate how such prior estimate can be used to transfer information across tasks and objectives.

We propose two HPO strategies: a Copula Thompson Sampling and a Gaussian Copula Process.

We show that these approaches can jointly model several objectives with potentially different scales, such as validation error and compute time, without requiring processing.

We demonstrate significant speed-ups over a number of baselines in extensive experiments.

The paper is organized as follows.

Section 2 reviews related work on transfer learning for HPO.

Section 3 introduces Copula regression, the building block for the HPO strategies we propose in Section 4.

Specifically, we show how Copula regression can be applied to design two HPO strategies, one based on Thompson sampling and an alternative GP-based approach.

Experimental results are given in Section 5 where we evaluate both approaches against state-of-the-art methods on three algorithms.

Finally, Section 6 outlines conclusions and further developments.

A variety of methods have been developed to induce transfer learning in HPO.

The most common approach is to model tasks jointly or via a conditional independence structure, which has been been explored through multi-output GPs (Swersky et al., 2013) , weighted combination of GPs (Schilling et al., 2016; Wistuba et al., 2018; Feurer et al., 2018) , and neural networks, either fully Bayesian (Springenberg et al., 2016) or hybrid (Snoek et al., 2015; Perrone et al., 2018; Law et al., 2018) .

A different line of research has focused on the setting where tasks come over time as a sequence and models need to be updated online as new problems accrue.

A way to achieve this is to fit a sequence of surrogate models to the residuals relative to predictions of the previously fitted model (Golovin et al., 2017; Poloczek et al., 2016) .

Specifically, the GP over the new task is centered on the predictive mean of the previously learned GP.

Finally, rather than fitting a surrogate model to all past data, some transfer can be achieved by warm-starting BO with the solutions to the previous BO problems (Feurer et al., 2015; Wistuba et al., 2015b) .

A key challenge for joint models is that different black-boxes can exhibit heterogeneous scale and noise levels (Bardenet et al., 2013; Feurer et al., 2018) .

To address this, some methods have instead focused on search-space level, aiming to prune it to focus on regions of the hyperparameter space where good configurations are likely to lie.

An example is Wistuba et al. (2015a) , where related tasks are used to learn a promising search space during HPO, defining task similarity in terms of the distance of the respective data set meta-features.

A more recent alternative that does not require meta-features was introduced in Perrone et al. (2019) , where a restricted search space in the form of a low-volume hyper-rectangle or hyper-ellipsoid is learned from the optimal hyperparameters of related tasks.

Rank estimation can be used to alleviate scales issues however the difficulty of feeding back rank information to GP leads to restricting assumptions, for instance (Bardenet et al., 2013) does not model the rank estimation uncertainty while (Feurer et al., 2018) uses independent GPs removing the adaptivity of the GP to the current task.

Gaussian Copula Process (GCP) (Wilson & Ghahramani, 2010) can also be used to alleviate scale issues on a single task at the extra cost of estimating the CDF of the data.

Using GCP for HPO was proposed in Anderson et al. (2017) to handle potentially non-Gaussian data, albeit only considering non-parametric homoskedastic priors for the single-task and single objective case.

For each task denote with f j : R p → R the error function one wishes to minimize, and with

the evaluations available for an arbitrary task.

Given the evaluations on M tasks

, we are interested in speeding up the optimization of an arbitrary new task f , namely in finding arg min x∈R p f (x) in the least number of evaluations.

One possible approach to speed-up the optimization of f is to build a surrogate modelf (x).

While using a Gaussian process is possible, scaling such an approach to the large number of evaluations available in a transfer learning setting is challenging.

Instead, we propose fitting a parametric estimate of f θ (x) distribution which can be later used in HPO strategies as a prior of a Gaussian Copula Process.

A key requirement here is to learn a joint model, e.g., we would like to find θ which fits well on all observed tasks f j .

We show how this can be achieved with a semi-parametric Gaussian Copula in two steps: first we map all evaluations to quantiles with the empirical CDF, and then we fit a parametric Gaussian distribution on quantiles mapped through the Gaussian inverse CDF.

First, observe that as every y i comes from the same distribution for a given task, the probability integral transform results in u i = F (y i ), where F is the cumulative distribution function of y. We then model the CDF of (u 1 , . . .

, u N ) with a Gaussian Copula:

where Φ is the standard normal CDF and φ µ,Σ is the CDF of a normal distribution parametrized by µ and Σ. Assuming F to be invertible, we define the change of variable z = Φ −1 • F (y) = ψ(y) and g = ψ • f .

1 We regress the marginal distribution of P (z|x) with a Gaussian distribution whose mean and variance are two deterministic parametric functions given by

where h w h (x) ∈ R d is the output of a multi-layer perceptron (MLP) where

are projection parameters and Ψ(t) = log(1 + exp t) is an activation mapping to positive numbers.

The parameters θ = {w h , w µ , b µ , w σ , b σ } together with the parameters in MLP are learned by minimizing the Gaussian negative log-likelihood on the available evaluations

, e.g., by minimizing

with SGD.

Here, the term ψ (ψ −1 (z)) accounts for the change of variable z = ψ(y).

Due to the term ψ (ψ −1 (z)), errors committed where the quantile function changes rapidly have larger gradient than when the quantile function is flat.

Note that while we weight evaluations of each tasks equally, one may alternatively normalize gradient contributions per number of task evaluations.

The transformation ψ requires F , which needs to be estimated.

Rather than using a parametric or density estimation approach, we use the empirical CDFF (t) = 1 N N i=1 1 yi≤t .

While this estimator has the advantage of being non-parametric, it leads to infinite value when evaluating ψ at the minimum of maximum of y. To avoid this issue, we use the Winsorized cut-off estimator

where N is the number of observations of y and choosing δ N = 1 4N 1/4 √ π log N strikes a bias-variance trade-off (Liu et al., 2009 ).

This approach is semi-parametric in that the CDF is estimated with a non-parametric estimator and the Gaussian Copula is estimated with a parametric approach.

The benefit of using a non-parametric estimator for the CDF is that it allows us to map the observations of each task to comparable distributions as z j ∼ N (0, 1) for all tasks j.

This is critical to allow the joint learning of the parametric estimates µ θ and σ θ , which share their parameter θ across all tasks.

As our experiments will show, one can regress a parametric estimate that has a standard error lower than 1.

This means that information can be leveraged from the evaluations obtained on related tasks, whereas a trivial predictor for z would predict 0 and yield a standard error of 1.

In the next section we show how this estimator can be leveraged to design two novel HPO strategies.

4 COPULA BASED HPO 4.1 COPULA THOMPSON SAMPLING Given the predictive distribution P (z|x) ∼ N (µ θ (x), σ θ (x)), it is straightforward to derive a Thompson sampling strategy (Thompson, 1933) exploiting knowledge from previous tasks.

Given N candidate hyperparameter configurations x 1 , . . .

, x N , we sample from each predictive distributionz i ∼ N (µ θ (x i ), σ θ (x i )) and then evaluate f (x i ) where i = arg min izi .

Pseudo-code is given in the appendix.

While this approach can re-use information from previous tasks, it does not exploit the evaluations from the current task as each draw is independent of the observed evaluations.

This can become an issue if the new black-box significantly differs from previous tasks.

We now show that Gaussian Copula regression can be combined with a GP to both learn from previous tasks while adapting to the current task.

Instead of modeling observations with a GP, we model them as a Gaussian Copula Process (GCP) (Wilson & Ghahramani, 2010) .

Observations are mapped through the bijection ψ = Φ −1 • F , where we recall that Φ is the standard normal CDF and that F is the CDF of y. As ψ is monotonically increasing and mapping into the line, we can alternatively view this modeling as a warped GP (Snelson et al., 2004 ) with a non-parametric warping.

One advantage of this transformation is that z = ψ(y) follows a normal distribution, which may not be the case for y = f (x).

In the specific case of HPO, y may represent accuracy scores in [0, 1] of a classifier where a Gaussian cannot be used.

Furthermore, we can use the information gained on other tasks with µ θ and σ θ by using them as prior mean and variance.

To do so, the following residual is modeled with a GP:

where g = ψ • f .

We use a Matérn-5/2 covariance kernel and automatic relevance determination hyperparameters, optimized by type II maximum likelihood to determine GP hyperparameters (Rasmussen & Williams, 2006) .

Fitting the GP gives the predictive distribution of the residual surrogatê

Because µ θ and σ θ are deterministic functions, the predictive distribution of the surrogateĝ is then given byĝ

Using this predictive distribution, we can select the hyperparameter configuration maximizing the Expected Improvement (EI) (Mockus et al., 1978) of g(x).

The EI can then be defined in closed form as

, with Φ and φ being the CDF and PDF of the standard normal, respectively.

When no observations are available, the empirical CDFF is not defined.

Therefore, we warm-start the optimization on the new task by sampling a set of N 0 = 5 hyperparameter configurations via Thompson sampling, as described above.

Pseudo-code is given in Algorithm 1.

Learn the parameters θ of µ θ (x) and σ θ (x) on hold-out evaluations D M by minimizing equation 1.

Sample an initial set of evaluations

while Has budget do Fit the GP surrogater to the observations {(x,

) | (x, y) ∈ D} Sample N candidate hyperparameters x 1 , . . . , x N from the search space Compute the hyperparameter x i where i = arg max i EI(

In addition to optimizing the accuracy of a black-box function, it is often desirable to optimize its runtime or memory consumption.

For instance, given two hyperparameters with the same expected error, the one requiring fewer resources is preferable.

For tasks where runtime is available, we use both runtime and error objectives by averaging in the transformed space, e.g., we set

, where z error (x) = ψ(f error (x)) and z time (x) = ψ(f time (x)) denote the transformed error and time observations, respectively.

This allows us to seamlessly optimize for time and error when running HPO, so that the cheaper hyperparameter is favored when two hyperparameters lead to a similar expected error.

Notice many existing multi-objective methods can potentially be combined with our Copula transformation as an extension, which we believe is an interesting venue for future work.

We considered the problem of tuning three algorithms on multiple datasets: XGBoost (Chen & Guestrin, 2016) , a 2-layer feed-forward neural network (FCNET) (Klein & Hutter, 2019) , and the RNN-based time series prediction model proposed in Salinas et al. (2017) (DeepAR) .

We tuned XGBoost on 9 libsvm datasets (Chang & Lin, 2011) to minimize 1−AUC, and FCNET on 4 datasets from Klein & Hutter (2019) to minimize the test mean squared error.

As for DeepAR, the evaluations were collected on the data provided by GluonTS (Alexandrov et al., 2019) , consisting of 6 datasets from the M4-competition (Makridakis et al., 2018) and 5 datasets used in Lai et al. (2017) , and the goal is to minimize the quantile loss.

Additionally, for DeepAR and FCNET the runtime to evaluate each hyperparameter configuration was available, and we ran additional experiments exploiting this objective.

More details on the HPO setup are in Table 1 , and the search spaces of the three problems is in Table 4 of the appendix.

Lookup tables are used as advocated in Eggensperger et al. (2012) , more details and statistics can be found in the appendix.

We compare against a number of baselines.

We consider random search and GP-based BO as two of the most popular HPO methods.

As a transfer learning baseline, we consider warm-start GP (Feurer et al., 2015) , using the best-performing evaluations from all the tasks to warm start the GP on the target task (WS GP best).

As an extension of WS GP best, we apply standardization on the objectives of the evaluations for every task and then use all of them to warm start the GP on the target task (WS GP all).

We also compare against two recently published transfer learning methods for HPO: ABLR (Perrone et al., 2018) and a search space-based transfer learning method (Perrone et al., 2019) .

ABLR is a transfer learning approach consisting of a shared neural network across tasks on top of which lies a Bayesian linear regression layer per task.

Finally, Perrone et al. (2019) transfers information by fitting a bounding box to contain the best hyperparameters from each previous task, and applies random search (Box RS) or GP-based BO (Box GP) in the learned search space.

We assess the transfer learning capabilities of these methods in a leave-one-task-out setting: we sequentially leave out one dataset and then aggregate the results for each algorithm.

The performance of each method is first averaged over 30 replicates for one dataset in a task, and the relative improvements over random search are computed on every iteration for that dataset.

The relative improvement for an optimizer (opt) is defined by (y random − y opt )/y random , which is upper bounded by 100%.

Notice that all the objectives y are in R + .

By computing the relative improvements, we can aggregate results across all datasets for each algorithm.

Finally, for all copula-based methods, we learn the mapping to copulas via a 3-layer MLP with 50 units per layer, optimized by ADAM with early-stopping.

To give more insight into the components of our method, we perform a detailed ablation study to investigate the choice of the MLP and compare the copula estimation to simple standardization.

Choice of copula estimators For copula-based methods, we use an MLP to estimate the output.

We first compare to other possible options, including a linear model and a k-nearest neighbor estimator in a leave-one-out setting: we sequentially take the hyperparameter evaluations of one dataset as test set and use all evaluations from the other datasets as a training set.

We report the RMSE in Table 5 of the appendix when predicting the error of the blackbox.

From this table, it is clear that MLP tends to be the best performing estimator among the three.

In addition, a low RMSE indicates that the task is close to the prior that we learned on all the other tasks, and we should thus expect transfer learning methods to perform well.

As shown later by the BO experiments, FCNET has the lowest RMSE among the three algorithms, and all transfer learning methods indeed perform much better than single-task approaches.

Homoskedastic and Heteroskedastic noise The proposed Copula estimator (MLP) uses heteroskedastic noise for the prior.

We now compare it to a homoskedastic version where we only estimate the mean.

The results are summarized in Table 2 where average relative improvements over random search across all the iterations and replicates are shown.

It is clear that heteroskedasticity tends to help on most datasets.

Copula transformation and standardization In our method, we map objectives to be normally distributed in two steps: first we apply the probability integral transform, followed by a Copula transform using the inverse CDF of a Gaussian.

To demonstrate the usefulness of such transformation, we compare it to a simple standardization of the objectives where mean and std are computed on each datasets separately.

Results are reported in Table 2 .

It is clear that standardization performs significantly worse than the Copula transformation, indicating that it is not able to address the problem of varying scale and noise levels across tasks.

Note that the relative improvement objective is not lower bounded, so that when random search finds very small values the scale of relative improvement can be arbitrary large (such as for the Protein dataset in FCNET).

Table 2 : Relative improvements over random search.

TS std and GP std respectively using a simple standardization instead of the copula transformation.

Ho and He stand for Homoskedastic and Heteroskedastic noise.

We now compare the proposed methods to other HPO baselines.

The results on using only the error information are shown first followed by the results using both time and error information.

Results using only error information We start by studying the setting where only error objectives are used to learn the copula transformation.

Within each task, we first aggregate 30 replicates for each method to compute the relative improvement over random search at every iteration, and then average the results across all iterations.

The results are reported in Table 3 , showing that CGP is the best method for almost every task except XGBoost.

In XGBoost, there are several tasks on which methods without transfer learning perform quite well.

This is not surprising as we observe in an ablation study on copula estimators (see Table 5 in the appendix) that some tasks in XGBoost have relatively high test errors, implying that the transferred prior will not help.

In those tasks, CGP is usually the runner-up method after standard GP.

We also report the results at iteration 10, 50 and 100 in the Tables 7, 8 and 9 in the appendix where we observe CGP and Box RS are the most competitive methods at 10th iteration but at 100 iteration, CGP is clearly the best transfer learning method.

This highlights the advantage of being adaptive to the target task of our method while making effective transfer in the beginning.

We also show results on two example datasets from each algorithm in Figure 1 , reporting confidence intervals obtained via bootstrap.

Note that the performance of the methods in the examples for DeepAR and XGBoost exhibit quite high variations, especially in the beginning of the BO.

We conjecture this is due to an insufficient number of evaluations in the lookup tables.

Nevertheless, the general trend is that CTS and CGP outperform all baselines, especially in the beginning of the BO.

It can also be observed that CGP performs at least on par with the best method at the end of the BO.

Box RS is also competitive at the beginning, but as expected fails to keep its advantage toward the end of the BO.

Results using both error and time information We then studied the ability of the copula-based approaches to transfer information from multiple objectives.

Notice it is possible to combine Copula transformation with other multi-objective BO methods and we will leave this direction as future work.

We show two example tasks in DeepAR and FCNET in Figure 2 , where we fix the total number of iterations and plot performance against time with 2 standard error.

To obtain distributions over seeds for one method, we only consider the time range where 20 seeds are available ,which explains why methods can start and end at different times.

With the ability to leverage training time information, the copula-based approaches have a clear advantage over all baselines, especially at the beginning of the optimization.

We also report aggregate performance over all the tasks in Table 6 in the appendix.

Due to the different methods finishing the optimization at different times, we only compare them up to the time taken by the fastest method.

For each method we first compute an average over 30 replicates, then compute the relative improvement over random search, and finally average the results across all time points.

The copula based methods converge to a good hyperparameter configuration significantly faster than all the considered baselines.

Note that we obtain similar results as for Hyperband-style methods (Li et al., 2016) , where the optimization can start much earlier than standard HPO, with the key difference that we only require a single machine.

We introduced a new class of methods to accelerate hyperparameter optimization by exploiting evaluations from previous tasks.

The key idea was to leverage a semi-parametric Gaussian Copula prior, using it to account for the different scale and noise levels across tasks.

Experiments showed that we considerably outperform standard approaches to BO, and deal with heterogeneous tasks more robustly compared to a number of transfer learning approaches recently proposed in the literature.

Finally, we showed that our approach can seamlessly combine multiple objectives, such as accuracy and runtime, further speeding up the search of good hyperparameter configurations.

A number of directions for future work are open.

First, we could combine our Copula-based HPO strategies with Hyperband-style optimizers (Li et al., 2016) .

In addition, we could generalize our approach to deal with settings in which related problems are not limited to the same algorithm run over different datasets.

This would allow for different hyperparameter dimensions across tasks, or perform transfer learning across different black-boxes.

Learn the parameters θ of µ θ (x) and σ θ (x) on hold-out evaluations D M by minimizing equation 1.

while Has budget do Sample N candidate hyperparameters x 1 , . . .

, x N from the search space

where i = arg min izi end while

To speed up experiments we used a lookup table approach advocated in Eggensperger et al. (2012) which proposed to use an extrapolation model built on pre-generated evaluations to limit the number of blackbox evaluations, thus saving a significant amount of computational time.

However, the extrapolation model can introduce noise and lead to inconsistencies compared to using real blackbox evaluations.

As a result, in this work we reduced BO to the problem of selecting the next hyperparameter configurations from a fixed set that has been evaluated in advance, so that no extrapolation error is introduced.

All evaluations were obtained by querying each algorithm at hyperparameters sampled (log) uniformly at random from their search space as described in Table 4 .

The CDF on the error objectives is given in Figure 3 .

Results on different iterations.

We plot the improvement over random research for all the methods at iteration 10, 50 and 100 at Table 7 , 8 and 9, respectively.

In short, at 10th iteration, transfer learning methods, especially our CGP and Box RS, performed much better than GP.

But, when looking at results at 50 and 100 iterations, CGP outperforms clearly all other transfer methods because of its improved adaptivity.

More details on prior MLP architecture.

The MLP used to regress µ θ and σ θ consists of 3 layers with 50 nodes, each with a dropout layer set to 0.5.

The learning rate is set to 0.01, batch size to 64 and we optimize over 100 gradient updates 3 times, lowering the learning rate by 10 each time.

Table 9 : Relative improvements over random search at iteration 100.

<|TLDR|>

@highlight

We show how using semi-parametric prior estimations can speed up HPO significantly across datasets and metrics.