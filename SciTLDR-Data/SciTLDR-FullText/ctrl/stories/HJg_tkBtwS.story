Answering questions about data can require understanding what parts of an input X influence the response Y. Finding such an understanding can be built by testing relationships between variables through a machine learning model.

For example, conditional randomization tests help determine whether a variable relates to the response given the rest of the variables.

However, randomization tests require users to specify test statistics.

We formalize a class of proper test statistics that are guaranteed to select a feature when it provides information about the response even when the rest of the features are known.

We show that f-divergences provide a broad class of proper test statistics.

In the class of f-divergences, the KL-divergence yields an easy-to-compute proper test statistic that relates to the AMI.

Questions of feature importance can be asked at the level of an individual sample.

We show that estimators from the same AMI test can also be used to find important features in a particular instance.

We provide an example to show that perfect predictive models are insufficient for instance-wise feature selection.

We evaluate our method on several simulation experiments, on a genomic dataset, a clinical dataset for hospital readmission, and on a subset of classes in ImageNet.

Our method outperforms several baselines in various simulated datasets, is able to identify biologically significant genes, can select the most important predictors of a hospital readmission event, and is able to identify distinguishing features in an image-classification task.

Model interpretation techniques aim to select features important for a response by reducing models (sometimes locally) to be human interpretable.

However, the phrase model interpretation can be a bit of a misnomer.

Any interpretation of a model must be imbued to the model by the population distribution that provides the data to train the model.

In this sense, interpreting a model should be viewed as understanding the population distribution of data through the lens of a model.

Existing methods for understanding the population distributions only work with particular models fit to the population, particular choices of test statistic, or particular auxiliary models for interpretation (Ribeiro et al., 2016; Lundberg and Lee, 2017) .

Such structural restrictions limit the applicability of these methods to a smaller class of population distributions.

To be able to work in a black-box manner, feature selection methods can use models but must not require a particular structure in models used in selection processes.

Understanding the population distribution can be phrased as assessing whether a response is independent of a feature given the rest of the features; this test is called a conditional randomization test (Candes et al., 2018) .

Conditional randomization tests require test statistics.

Test statistics like linear model coefficients (Barber et al., 2015) or correlation may miss dependence between the response and outcome.

To avoid missing relationships between variables, we develop the notion of a proper test statistic.

Proper test statistics are those whose power increases to one as the amount of data increases.

Conditional independence implies the conditional-joint factorizes into conditionalmarginals.

Measuring the divergence between these distributions yields a proper test statistic.

Of the class of integral probability metrics (M??ller, 1997) and f -divergences (Csisz??r, 1964) , the KLdivergence simplifies estimation and allows for reuse of the model structures and code from the standard task of predicting the response from the features.

Using the KL-divergence in this context has a natural interpretation; it is a measure of the additional information each feature provides about the outcome over the rest.

This measure of information is known as the additional mutual information (AMI) .

Our proposed procedure is called the additional mutual information conditional randomization test (AMI-CRT).

AMI-CRT uses regressions to simulate data from the null for each feature and compares the additional mutual information (AMI) of the original data to the AMI of the simulations from Beyond understanding the population distribution, some tasks require interpreting a population distribution on the level of an individual datapoint.

Methods that test for conditional independence work under distributional notions of feature selection, but are not designed to identify the relevant features for a particular sample.

To address this issue of "instance-wise feature selection," several methods have been proposed, including local perturbations (Simonyan et al., 2013; Sundararajan et al., 2017; Ribeiro et al., 2016) and fitting simpler auxiliary models to explain the predictions of a large model (Chen et al., 2018; Lundberg and Lee, 2017; Yoon et al., 2019; Turner, 2016; ??trumbelj and Kononenko, 2014; Shrikumar et al., 2017) .

Our instance-wise work is most similar to that of Burns et al. (2019) , who repurpose the HRT framework to perform instance-wise feature selection, or Gimenez and Zou (2019) , who define a conditional randomization test (CRT) procedure for subsets of the feature space.

In general, however, the conditions under which instance-wise feature selection with predictive models may be possible are not well developed.

We address this issue by first identifying a set of sufficient conditions under which instance-wise feature selection is always possible.

We then show how estimators used in AMI-CRT can be repurposed for use in an instance-wise setting, yielding a procedure called the AMI-IW.

Practitioners of machine learning use feature selection to identify important features for their predictive task.

One way to filter out important features is to find those that improve predictions given the rest of the features.

This can be formalized through conditional independence.

Let x j be the jth feature of x and let x ???j be all features but the jth one.

The goal is to discover a set S such that ???x j ??? S, x j ??? y | x ???j , where independence is with respect to the true population distribution q. The only knowledge about q comes from a finite set of samples

sampled from the population.

This means that it is impossible to assess exact conditional independence.

Therefore, in the finite sample setting, we must formulate a statistical hypothesis test.

A conditional randomization test (CRT) (Candes et al., 2018 ) defines a hypothesis test for conditional independence.

For the jth feature, CRTs first compute a test statistic t using the N samples of data D N .

CRTs place this statistic in a null distribution where samples of the jth feature x j are replaced by samples of x j | x ???j which by construction satisfy x j ??? y | x ???j .

Letting D j,N be a dataset where {x

, the p-value for a CRT is

Under smoothness constraints, the p-value is uniform under the null because it computes the cumulative distribution function of the test statistic under the null.

While CRTs provide a general method for conditional independence testing, they leave several components including the choice of test statistic unspecified.

) that uses only a feature x j and the outcome y. Any p-values computed using this test statistic would be meaningless when testing for conditional independence, as t never considers the remaining features x ???j .

Therefore, particular choices for test statistics limit what can be tested.

To address this, we introduce the concept of a proper test statistic.

A test statistic t(D N ) is proper if p-values produced by the statistic converge to 0 when the null must be rejected, and are uniformly distributed otherwise.

Using t in Equation (1), this is:

where d ??? ??? indicates a convergence in distribution.

Under the alternate hypothesis, which in the case of feature selection is x j ??? y | x ???j , the power to reject the null hypothesis must be 1, implying p j ??? 0.

A proper test statistic requires that Equation (2) must hold for all distributions of y, x.

Proper tests statistics in a CRT select the features in S as the data grows.

Definition 1 mirrors the concept of a scoring rule (Gneiting and Raftery, 2007) , which measures the calibration of a probabilistic prediction by a model.

A proper scoring rule is one such that the highest expected score is obtained by a model that uses the true probability distribution to make predictions.

Divergences are proper test statistics.

Conditional independence means the conditional distribution r factorizes:

Divergences measure the closeness between two distributions.

A divergence is zero when the two distributions are the same and positive otherwise.

Computing any divergence K, like an integral probability metric (M??ller, 1997) or an f -divergence (Csisz??r, 1964) , between the left hand side and right hand side of Equation (3) would be a proper test statistic.

Let K(a, b) ??? 0 with equality holding only when a is equal in distribution to b, then a proper test statistic Define the resampling distributionq j =q j ( x j | x ???j )q(y, x ???j ).

Using a divergence in a CRT requires estimates of the following conditional distributions: q(x j | x ???j ), q(y | x),q j (y | x j , x ???j ), and q(y | x ???j ).

The first distribution q(x j | x ???j ) is required for any CRT.

The next distribution q(y | x) corresponds to the standard task of building a good regression model.

The third distributio?? q j (y | x j , x ???j ) requires a regression model with corrupted inputs.

This regression can reuse the model structure and code from the standard regression task q(y | x).

However, the last distribution q(y | x ???j ) could require development of new model structures.

For example, if x is an image, a good model for q(y | x) could be a convolutional neural network.

If the conditioning set x ???j is a subregion of that image, the convolutional neural network used for q(y | x) would need to be modified for different padding and filter sizes.

This means new models could be needed for each x ???j .

In the next section, we show that the KL-divergence removes the need for estimating this distribution, and therefore only requires the piece needed for all CRTs, q(x j | x ???j ), and model code to fit the response from the features.

Using the KL-divergence as a test statistic in Equation (1) requires the difference ?? j :

The second equality above follows from q(x ???j ) =q j (x ???j ), and the fourth from q(y | x ???j ) = q j (y | x ???j ).

This simplification of ?? j means that for the computation of the average KL-divergence as a test statistic, the distribution q(y | x ???j ) is unecessary, thereby reducing computation and allowing the reuse of training infrastructure for predicting the response y. The KL-divergence provides this reduction since log splits products into sums.

This expected KL-divergence is called the additional mutual information (AMI) .

Computing the expected value of ?? j to get the p-value in Equation (1) requires estimation from a finite sample.

Recall that D N is a collection of N datapoints sampled iid from q(x, y).

These samples from the population come from the provided data.

Similarly, D j,N is sampled iid from q(x ???j , y)q( x j | x ???j ).

The samples x ???j , y come from the population distribution, however sampling q( x j | x ???j ) requires learning a model for this distribution also known as the complete conditional.

The other two pieces of ?? j require building a model from x to y and ( x j , x ???j ) to y. To learn these models, the data D N is split into training and test sets.

Let ??, ?? be parameters, q ?? ( x j | x ???j ) and q ?? (y | x) are fit to the training set.

The distribution q ?? (y | x j , x ???j ) is fit to the training part of D j,N .

To compute the expectations in ?? j , these models are evaluated on their corresponding test sets.

Finally, a Monte Carlo estimate of the expectation for the p-value in Equation (1) requires replicating this procedure K times.

We call this procedure AMI-CRT (Algorithm 1).

The models q ?? ( x j | x ???j ) and q ?? (y | x) can be shared across the replications, however q ?? (y | x j , x ???j ) must be recomputed for each replication.

While this is an embarrassingly parallel problem, it involves the computation of a new model for each draw from the null distribution.

To speed up the computation of AMI-CRT, we use the average of the original model q ?? (y | x) and a single model trained for q ?? (y | x j , x ???j ).

This averaged model is used to estimate the two terms in the AMI-difference ?? j .

Under the null, the estimates are identically distributed because each estimate evaluates the same function on identically distributed samples.

Thus, averaging produces p-values that are uniform under the null, but the average may not result in a proper test-statistic.

However, the averaged model performs almost as well as AMI-CRT empirically.

We call this procedure the FAST-AMI-CRT, which is summarized in Algorithm 2.

Averaging is needed because models trained on data drawn from the same distribution have variance (Friedman et al., 2001 ).

The averaged model provides FAST-AMI-CRT with several advantages.

First, it is more conservative than using just the original model as in HRTs (Tansey et al., 2018 ) since the averaged model both predicts better on the null data and worse on the real data.

We show empirically that this guards against errors in the estimation of the complete conditional distribution.

Second, it requires only a single null model per feature instead of one per replication.

To estimate each of q ?? and q ?? , standard regression models like logistic regression, neural networks, and random forests can be used at no more computational cost than training.

Nonparametric regression can be used as well.

The choice between these estimators should be made by using the best fitting regression on validation data.

The estimation procedure is straightforward, yet effective as we demonstrate in Section 4.

In the next section, we show how the building blocks for FAST-AMI-CRT can be used to provide feature importances on an instance-wise level.

So far, we show how to recover features important across the whole population.

We have not yet addressed the issue that different samples could have different important features.

We call this problem of recovering important features for each sample, instance-wise feature selection (IWFS).

To identify important features instance-wise, we can use the probability of observing a particular label y (i) given a set of features x (i) .

This suggests a candidate definition for important features:

The jth feature for the ith sample,

Definition 2 says that a feature x (i) j is important if observing it increases the probability of y (i) .

This formulation is exploited in (Yoon et al., 2019; Chen et al., 2018 ) to obtain instance-wise important features.

However, Definition 2 can sometimes fail to identify relevant features, even with access to the true conditional distribution q(y | x).

While important features may satisfy this condition, so will a few unimportant features.

As a demonstrative example, consider the data generating process where

, and ??? N (0, ?? 2 ).

Assume we have the true q(y | x 1 , x 2 ), and let z be unobserved.

Pick any sample (x

1 is important for this instance.

We can expand:

For all i such that 3, 7] will violate this inequality.

In all of those cases, the wrong feature will be selected as important as per the candidate Definition 2.

We show the full derivation of this example in Appendix C.1.

The fundamental issue with the formulation in Definition 2 is that noise can act as a "selection" mechanism, but cannot be estimated because it is unobserved.

While predictive models q ?? for q(y | x) suffice for understanding population distributions, they might not be sufficient to perform IWFS.

We develop the following condition under which q(y | x) is sufficient to perform IWFS: Proposition 1.

Sufficient conditions for instance-wise feature selection:

For each sample (x (i) , y (i) ), let S (i) be a set of features that contribute to the prediction of y (i) defined as:

If y is discrete, and q(y = y

we have perfect predictions on our dataset, then it is possible to recover such a set S (i) for all i. (Benjamini and Hochberg, 1995) , allow instance-wise feature selection, and make no distributional assumptions about the data-generating process.

This table compares these methods to widely-used feature selection methods.

The set in Equation (4)

j is not important to y (i) .

Assuming the sufficient conditions in Proposition 1, we can now construct an IWFS procedure using the same estimators from AMI-CRT or FAST-AMI-CRT.

Instance-wise feature selection can be performed using the building blocks of the FAST-AMI-CRT.

Starting from Definition 2, we begin by manipulating q(y = y (i) | x = x (i) ) and marginalizing out

We then use Jensen's inequality to upper-bound the log of this expectation as follows:

This suggests the following instance-wise test.

If the inequality in Equation (5) is strict, the feature is considered important.

If equality holds in Equation (5), the feature is considered unimportant.

Notice that Jensen's inequality could introduce slack in this bound that could make a feature seem relevant when it is not.

We use Proposition 1 to show that this is not an issue.

Recall that given a model q ?? for q(y | x) that satisfies the instance-wise sufficient conditions in Proposition 1,

In the case where

j .

Then the left-hand side of Equation (5)

j , implying an equality.

Therefore, checking for equality in Equation (5) is a valid test to see if a feature is either important or unimportant.

In Appendix C.2, we detail an example that shows how scores computed using Equation (5) can help rank features from most to least helpful for prediction.

We term this procedure the additional mutual information instance-wise feature selection (AMI-IW).

If we computed an expectation over x (i) , y (i) of Equation (5), this procedure resembles FAST-AMI-CRT.

We can reuse the estimators from FAST-AMI-CRT to compute these instance-wise logprobability differences for AMI-IW.

Therefore, we only use one null estimator forq j (y | x j , x ???j ).

Like FAST-AMI-CRT, we can potentially use a mixture of estimators in AMI-IW, but at the cost of power to select important features.

We compare our methods, the AMI-CRT [ami-crt] and fast additional mutual information conditional randomization test (FAST-AMI-CRT) [fast-ami-crt] to widely-used approaches on various performance metrics.

The baselines are:

Under review as a conference paper at ICLR 2020 (Tansey et al., 2018) HRTs construct a test by comparing a loss function evaluated on x and x j , x ???j .

The choice of loss, however, is left to the practitioner.

We study the 0-1 loss, which is a proper scoring rule, in all of our experiments.

In specific settings, we equip an HRT with our AMI test-statistic.

This method is called the ami-hrt.

We show that ami-hrt is better calibrated than 0-1 HRT.

Table 1 presents a summary comparison of the properties of each selection method.

We use the regression approach using conditional categorical distribution parameterized with neural networks highlighted in (Miscouridou et al., 2018) to model q(x j |x ???j ) for all experiments unless specified otherwise.

Simulations: We simulate data for evaluating each selection method.

These tests are designed to highlight the differences between each method.

[xor]: To test the case where features on their own are not informative, but together provide information, we use the xor dataset.

We first sample x ??? N (0, ?? D ) N times, where ?? D is a Ddimensional covariance matrix.

We translate the first two dimensions of each sample x (i) away from the origin in 4 different directions : {(s, s), (???s, s), (s, ???s), (???s, ???s)} with uniform probability.

If the resulting translation has first two coordinates with the same sign, the label is one.

Otherwise, it is zero.

All but the first two features are independent of y. We set N = 2000, D = 20.

[orange] (Chen et al., 2018) : To test the case where y is some nonlinear function of x, we use the orange dataset.

In this dataset [selector, noisy-selector]: These experiments test instance-wise feature selection methods.

We first sample x ??? N (0, ?? D ) N times, where ?? D is a D-dimensional covariance matrix, and D ??? 11.

The first feature x 1 , called the "selector" feature, determines the feature selection mechanism.

We generate y ??? {0, 1} as Equation (6).

We also investigate the effectiveness of each feature selection method in the presence of noise, and generate y ??? {0, 1} as Equation (7).

We set the parameter N = 2000, D = 20.

Results.

For methods based on CRTs or HRTs, we select features using p-values.

For the baselines that do not produce p-values, we select features using the importance scores provided by each method.

We threshold p-values or importance scores respectively, and compute an ROC curve for each method.

We present the mean area under each curve over 100 simulations for the xor and orange datasets in Table 2 .

We notice that this task is easily solved by most methods apart from corr-crt.

This test fails to account for dependencies between features.

The ami-crt achieves a higher AUROC than baselines, while fast-ami-crt achieves similar performance.

To identify important features in practice, a threshold for importance scores must be chosen.

If a method produces p-values, we can control the false discovery rate (FDR) (Benjamini and Hochberg, 1995) .

This is the expected proportion of falsely identified features.

An assumption for standard FDR-controlling procedures is independent p-values (Benjamini and Hochberg, 1995) .

Therefore, we investigate the calibration of p-values across ami-crt, fast-ami-crt, corr-crt, and loss-hrt.

We omit other baselines in this comparison as they do not produce p-values and therefore have no direct FDR-control.

To evaluate each method, we use the generating process for the orange dataset, and set N = 3000, D = 104, = 4.

If the p-values are independent, null p-values should resemble iid draws from a Uniform(0,1) distribution.

Figure 3 (Appendix D) shows a quantile-quantile plot of null p-values.

We also perform a Kolmogorov-Smirnov (KS) (Massey Jr, 1951) test where the null distribution is Uniform(0,1).

All CRT-based methods produce independent p-values, while loss-hrt produces deflated and significantly non-uniform p-values (p = 0.0006), implying dependence.

As a result loss-hrt incorrectly identifies many null features as important.

Models for each replication of null features ami-crt potentially decrease the correlation of the p-values.

The fast-ami-crt achieves a middle ground between the hrt and ami-crt by yielding well-calibrated p-values while requiring only one null model q ?? (y | x j , x ???j ) per feature.

We also investigate the use of ami-hrt, which uses the AMI as a test statistic in an HRT in Figure 3 .

To better understand the difference between refitting (CRTs) and not refitting (HRTs), we inspect the robustness of each method to poor simulations from the null (poor estimation of q(x j | x ???j )).

We use the orange dataset and set all off-diagonal values of ?? D to 0.5.

We sample x j from N (0, 1).

We see in Figure 4 (Appendix D) that refitting protects against poor approximations of q(x j | x ???j ); the p-value distribution for fast-ami-crt is uniform, while the p-value distribution for both HRT methods is significantly non-uniform (p = 5 ?? 10 ???6 for ami-hrt and p = 10 ???7 for loss-hrt).

The robustness of fast-ami-crt comes from averaging which makes predictions for the original data worse and predictions on the null better regardless of the quality of the null simulations.

For instance-wise feature selection, we perform two tests for each selection method.

To test HRTs in this setting, we use the procedure prescribed by Burns et al. (2019) but with different test statistics.

For precision, we identify the 7 (the true # of relevant features per instance) most important features as dictated by each selection method and report average precision scores across a held-out test set in Table 3 .

For selector identification, we count the number of instances where the selector variable x 1 was identified.

For the selector task, we notice that the ami-iw achieves the highest precision, followed by the loss-hrt.

In the noisy-selector task, we notice a decrease in scores across all methods with the largest decrease for ami-iw and loss-hrt.

The noisy-selector case violates the sufficient conditions for instance-wise feature selection (Equation (4)) meaning that the noise in sampling the response can obscure which features are important.

This explains the reduction in performance.

Even with the decrease in performance, ami-iw performs best.

The ami-iw and loss-hrt identify the selector variable x 1 in nearly every sample in our test set.

Linear methods like lime fail because the selection mechanism is highly non-linear.

Further, rf and corr-crt are not designed to assign importance at the level of an individual sample and therefore do not provide meaningful scores per instance.

Wellcome Trust Celiac disease: We study data from a genomic analysis on Celiac disease (Dubois et al., 2010) .

For each individual in this dataset, we have a set of single nucleotide polymorphisms (SNPs).

SNPs represents genetic variance in the individual with respect to some reference population.

This dataset consists of two classes of individuals: cases (n = 3796) and controls (n = 8154), where the cases are those with Celiac disease.

After standard preprocessing steps as prescribed by Bush and Moore (2012) , 1759 SNPs remain.

To model q(x j |x ???j ), we use the same procedures as (Candes et al., 2018) where we estimate q(x j |x Sj ) where S j is only the set of SNPs (not including x j ) known to be correlated with x j .

To model q(y|x), we use an L 1 -logistic regression model.

Results.

In Table 4 , we show the results for all methods with FDR-control.

We identify the SNPs that most likely contribute to distinguishing between those with Celiac disease and those without it.

Since these methods produce pvalues, we can select features at a theoretical FDR of 20% using the Benjamini and Hochberg (1995) procedure.

We report the percentage of selected SNPs that have been previously shown to be associated with Celiac disease in a biological context as reported by one of (Dubois et al., 2010; Sollid, 2002; Adamovic et al., 2008; Hunt et al., 2008) .

There are 40 SNPs in total that are both in our dataset and in these papers.

Ami-crt outperforms all other methods tested; fast-ami-crt performs similarly.

We also list the SNPs returned by ami-crt in Appendix E. As expected, corr-crt selects a large set of features, but achieves fairly low precision.

This is because many SNPs are correlated with each other, and all of these seem relevant marginally.

The AMI-based methods have better precision and recall compared to loss-hrt potentially both due to aforementioned deviation from uniform and that the zero-one loss may not change when only one out of more than a thousand features gets perturbed.

Hospital readmission: We use a dataset consisting of ten years of medical logs from over 130 hospitals (Strack et al., 2014) .

Features in the dataset include time spent in the hospital, medical specialty of attending doctor, age, and various other diagnostic information.

Labels for each sample represent one of three events: readmitted within the next 30 days (n = 35, 545), readmitted after 30 days (n = 11, 357), or not readmitted (n = 54, 864).

Due to class imbalance, we grouped all readmitted patients into one category (n = 46, 902).

We detail further preprocessing steps in Appendix F. To model q(y|x), we use a random forest classifier with 100 estimators. (Strack et al., 2014) as a clinically validated ground truth, we observe that AMI-CRT is able to achieve the highest area under the receiver operating characteristic curve (AUROC) when compared to state-of-the-art benchmarks.

Results.

The ground truth features come from clinical validation done by (Strack et al., 2014) .

We use importance scores (or pvalues) estimated by each selection method with these ground truth features to compute an ROC curve.

Figure 1 shows these curves for each method.

We observe that ami-crt and fast-ami-crt achieve a higher area under the ROC curve than state-of-the-art approaches.

The loss-hrt performs well, but achieves low power at false positive rates less than 0.5.

These methods, unlike locally-linear methods such as lime and shap, do not assume that relevant features are marginally independent of irrelevant features (Lundberg and Lee, 2017) .

We consider the task of differentiating between ambulances and policevans.

This task is interesting as both objects are physically very similar and there are only a few features that can be used to differentiate the two.

For example, both objects have windows, wheels, and doors, so other features must be used to distinguish between the two classes.

Rather than consider each pixel as an individual feature x j , we consider a patch of pixels x S as a single feature, such that no two patches contain overlapping pixels.

To model the distribution q(x S |x ???S ), we make use of a generative inpainting model ?? g (Yu et al., 2018) .

We split the image into an 8 ?? 8 grid so that there are 64 non-overlapping x S patches.

To model q(y|x), we use a VGG-16 network (Simonyan and Zisserman, 2015) .

To perform our instance-wise test, we compute log-probability differences using fifty generated samples from q(x S |x ???S ) for each patch.

Results.

In Figure 2 , we show a subset of results of AMI-IW.

The first and third columns show the original images for each class: ambulance and policevan respectively.

The second and fourth columns mask out the original image in patches where the patch is not found to be relevant to the prediction.

The model used to estimate q(y|x) is able to achieve roughly 90% accuracy on a heldout test set.

We see that our predictive model uses relevant details like the words "ambulance" or "police" printed on the vehicle to distinguish between each class.

The model also tends to ignore objects like windscreens and other features shared across classes, as is expected.

These results indicate that the difference in log probabilities between a model using the true data, and one using x S sampled from q(x S | x ???S ) works well in determining a relevant set of features even on an instancewise level.

We show several additional images in Figure 5 , in Appendix G. We also compare our method to local interpretable model-agnostic explanations (LIME) and shapley additive explanations (SHAP) (Figures 6 and 7) .

Both methods perform reasonably well on this task, but identify objects that are known to be common to both classes like wheels and headlamps.

Neither method identifies writing on the vehicles in the images.

This is likely because of the simplifying assumptions made by these locally-linear methods.

They assume that the set of relevant features is independent of the set of irrelevant features, which may not be the case in images.

For example, the location of the word "ambulance" may depend on the window position.

We develop AMI-CRT for testing for conditional independence of each feature x j ??? y | x ???j from a finite sample from the population distribution.

AMI-CRT uses the KL-divergence to cast independence testing as regression and allows for the reuse of code from building the original model from the features to the response.

We develop FAST-AMI-CRT which requires less computation than AMI-CRT and is robust to poor estimation of the null conditional.

We define sufficient conditions under which to perform instance-wise feature selection and develop the AMI-IW, an instance-wise feature selection method built from the pieces of FAST-AMI-CRT.

AMI-CRT, FAST-AMI-CRT, and AMI-IW all outperform several popular methods.

in various simulated tasks, in identifying biologically significant genes, selecting the most indicative features to predict hospital readmissions, and in identifying distinguishing features in an image classification task.

where L?? is an log-likelihood estimate using q (k,m) ?? end end

Let x be a dataset such that x ???j = x ???j , and x j is randomly sampled from q ?? (

Let x (k) be a dataset such that x ???j = x ???j , and x j is randomly sampled from

and

.

We first list the assumptions here:

3.

The cumulative distribution functions of t(D N ) and t( D j,N ) are both continuous everywhere.

4.

We have access to complete conditionals q(

Proof.

We prove that t is a proper test statistic if and only if t(E n ) is a consistent estimator of

We do this by showing t yields p-values that are zero under the alternate hypothesis and uniformly distributed under the null.

Recall that the p-value for our test is:

Under the alternate hypothesis Consider the case where

whereq j ??? ??? indicates a convergence in probability.

Since x j ??? y | x ???j , notice also that

Therefore, the term inside the expectation in the p j (D N ) above is always 0, yielding a p-value of 0 in the limit of N .

Since these p-values converge in probability to a single point, the p-values converge in distribution to a delta mass at 0.

Under the null hypothesis In the case where x j ??? y | x ???j , the samples in q N (y, x) and q j,N (y, x j , x ???j ) are both sampled from the same distribution q =q j .

Therefore, the distribution of t(D N ) as a function of q N , is the same as that of t( D j,N ) as a function ofq j,N .

Let F N be the cumulative distribution function of t( D j,N ) which in this case is the same as that of t(D N ).

We rewrite the p-value expression as p

N (??) be the generalized inverse cumulative distribution function which exists because F N is a continuous everywhere function.

With this, we derive the distribution of the p-value: Discontinuities could occur when the event t(D N ) = t( D j,N ) occurs with some non-zero probability c. This means that the p-value does not take all the values in [0, 1].

To see this, note that

To remedy this, we can replace the indicator function in our test-statistic with the following function:

where Uniform([0, 1]) is a continuous uniform random variable. ,N ) , the distribution of the p-value is the same as the uniform random variable : N ) ) is continuous everywhere in its support because t(D N ) = t( D j,N ) occurs with zero probability.

Thus, this modification ensures that the p-values are distributed uniformly.

Recall our sufficiency condition for instance-wise feature selection as mentioned in Proposition 1.

In this example, we see what happens when this condition is not met.

We notice that this definition does not suffice to reject an unimportant feature.

Consider a simple data generating process:

where z is not observed.

We can now write out the probability distributions we care about.

Note that taking an expectation like E x1???q(x1|x2) q(y| x 1 x 2 ) yields q(y|x 2 ).

For simplicity, we leave out the use of complete conditions and work directly with the latter probability distributions:

Now consider an instance (x

2 and z (i) are important.

This offers a significant speedup as the HRT framework avoids having to refit estimators using x j ??? q(x j | x ???j ).

Figure 3 shows a quantile-quantile plot of the null p-values for each FDR-controlling feature selection method.

We notice that both HRT-based E CELIAC DISEASE GENOMIC FEATURE SELECTION Table 6 shows the set of SNPs deemed significant by AMI-CRT.

We annotate each SNP with its position in the human genome, and whether it was previously reported as significant in a biological study.

For the hospital readmission dataset, we applied several standard pre-processing techniques.

We filtered each sample the data in a manner similar to (Strack et al., 2014 ):

??? It is an inpatient encounter (a hospital admission).

??? It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.

??? The length of stay was at least 1 day and at most 14 days.

??? Laboratory tests were performed during the encounter.

??? Medications were administered during the encounter.

Further, we binarized the labels so that a label of 1 indicates a readmission event, and a label of 0 indicates no readmission event.

We then encoded each categorical feature as a one-hot encoding.

We then imputed missing values using the median across the dataset, and dropped the "weight" feature as it was found to be 97% missing.

To sample from the complete conditional distributions q(x j | x ???j ), we used a neural network to fit the complete conditional regression detailed in (Miscouridou et al., 2018) .

For continuous values of x j , we first discretized the data into bins, then used our neural network to predict the bins.

To map the bins back to values in the domain of x j , we used the mean of the range of values in each bin.

Figure 5 shows some of the results of instance-wise feature selection on ImageNet data using AMI-CRT.

Figures 6 and 7 show results on the same task, using LIME and SHAP respectively.

We notice that AMI-CRT identifies patches that seem more likely to help differentiate between ambulances and policevans.

AMI-CRT identifies relevant text like the words "ambulance" or "police" that are very likely to help distinguish between the two classes.

LIME identifies some relevant features of the image like wheels and lights, but fails to identify relevant words.

SHAP does a good job at identifying distinguishing symbols like the caduceus and the FBI logo, but occasionally misses out on relevant text.

Ambulance (original) Ambulance (masked) Policevan (original) Policevan (masked) Figure 5 : Instance-wise feature selection using AMI-CRT.

The first and third columns show the original image of ambulances or policevans respectively.

The second and fourth columns show only the patches which were found to have non-zero AMI with the label, given the rest of the patches.

<|TLDR|>

@highlight

We develop a simple regression-based model-agnostic feature selection method to interpret data generating processes with FDR control, and outperform several popular baselines on several simulated, medical, and image datasets.

@highlight

This paper proposes a practical improvement of the conditional randomization test and a new test statistic, proves f-divergence is one possible choice, and shows that KL-divergence cancels out some conditional distributions.

@highlight

This paper addresses the problem of finding useful features in an input that are dependent on a response variable even when conditioning on all other input variables.

@highlight

A model agnostic method to provide interpretation on the influence of input features on the response of a machine level model down to instance level, and proper test statistics for model agnostic feature selection.