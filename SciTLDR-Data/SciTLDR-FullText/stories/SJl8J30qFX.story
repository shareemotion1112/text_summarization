Interpretability has largely focused on local explanations, i.e. explaining why a model made a particular prediction for a sample.

These explanations are appealing due to their simplicity and local fidelity.

However, they do not provide information about the general behavior of the model.

We propose to leverage model distillation to learn global additive explanations that describe the relationship between input features and model predictions.

These global explanations take the form of feature shapes, which are more expressive than feature attributions.

Through careful experimentation, we show qualitatively and quantitatively that global additive explanations are able to describe model behavior and yield insights about models such as neural nets.

A visualization of our approach applied to a neural net as it is trained is available at https://youtu.be/ErQYwNqzEdc

Recent research in interpretability has focused on developing local explanations: given an existing model and a sample, explain why the model made a particular prediction for that sample BID40 .

The accuracy and quality of these explanations have rapidly improved, and they are becoming important tools to understand model decisions for individual samples.

However, the human cost of examining multiple local explanations can be prohibitive with today's large data sets, and it is unclear whether multiple local explanations can be aggregated without contradicting each other BID41 BID0 .In this paper, we are interested in global explanations that describe the overall behavior of a model.

While usually not as accurate as local explanations on individual samples, global explanations provide a different, complementary view of the model.

They allow us to clearly visualize trends in feature space, which is useful for key tasks such as understanding which features are important, detecting unexpected patterns in the training data and debugging errors learned by the model.

We propose to use model distillation techniques BID7 BID24 to learn global additive explanations of the form DISPLAYFORM0 to approximate the prediction function of the model F (x).

Figure 1 illustrates our approach.

The output of our approach is a set of p feature shapes {h i } p 1 that can be composed to form an explanation model that can be quantitatively evaluated.

Through controlled experiments, we empirically validate that feature shapes provide accurate and interesting insights into the behavior of complex models.

In this paper, we focus on interpreting F from fully-connected neural nets trained on tabular data.

Our goal is not to replace local explanations nor to explain how the model functions internally.

What we claim is that we can complement local explanations with global additive explanations that clearly illustrate the relationship between input features and model predictions.

Our contributions are:??? We propose to learn global additive explanations for complex, non-linear models such as neural nets.??? We leverage powerful generalized additive models in a model distillation setting to learn feature shapes that are more expressive than feature attributions Figure 1 : Given a black box model and unlabeled samples (new unlabeled data or training data with labels discarded), our approach leverages model distillation to learn feature shapes that describe the relationship between input features and model predictions.??? We perform a quantitative comparison of feature shapes to other global explanation methods in terms of fidelity to the model being explained, accuracy on independent test data, and interpretability through a user study.

Although our approach of using model distillation with powerful additive models of the form in equation 1 is new, our work is based on two previous research threads: (1) decomposing F into additiveF to understand how F is affected by its inputs (e.g. BID26 ), and (2) learning an interpretable model (often some form of decision tree) to mimic F (e.g. BID13 ).

Global additive explanations have been used to analyze inputs to complex, nonlinear mathematical models and computer simulations (Sobol, 2001) , analyze how hyperparameters affect the performance of machine learning algorithms BID27 , and decompose prediction functions into lower-dimensional components BID25 .

They are determined by the choice of metric L between F and its approximationF , degree d of highest order components (d = 3 in equation 1, and type of base learner h. One common theme of these methods is that they decompose F intoF using numerical or computational methods (e.g. matrix inversion, quasi Monte Carlo).

Rather than approximately decomposing F (which can be prohibitively expensive with large n or p) 1 , we propose to learnF using model distillation.

This is equivalent to choosing L that minimizes the empirical risk between the prediction function F and our global additive explanationF on the training data.

To minimize ||F ???F || L , we select two flexible, nonparametric base learners for h: splines (Wood, 2006) and bagged trees.

This gives us two global additive explanation models: Student Bagged Additive Boosted Trees (SAT) and Student Additive Splines (SAS).

Other choices of h are possible.

We describe our distillation setup to learn these models in Section 2.2.

In most of this paper,F consists of main components h i (d = 1 in equation 1).

Higher order components h ij and h ijk can increase the accuracy ofF , but make interpretation more difficult.

WhenF consists of only main components h i , any pairwise or higher order interactions in F are expressed as a best-fit additive approximation added to main components h i , plus a pure-interaction residual.

We show examples of this expression in Section 4.1, and show the utility of adding higher order components h ij and h ijk , when present in F , in Section D.2.

Neural nets and other black-box models have been approximated by interpretable models such as trees, rule lists, etc., either via model distillation/compression BID13 BID10 BID19 or model extraction BID20 Sanchez et al., 2015; BID32 BID4 BID41 .

However, all of them approximated classifier models; there has been less work approximating regression models.

Another gap in the literature is rule lists for regression: state-of-the-art rule lists BID34 BID2 or rule sets BID31 do not have regression implementations.

Model distillation requires only that the teacher model label a training set, not repeated probing or access to the teacher's internal structure or derivatives.

This, combined with the applicability of generalized additive models to both classification and regression, means that our approach can approximate a broad class of classification and regression models.

We also show in Sections 4.2.2 and 4.3, with a user study, that additive explanations have advantages over decision trees when it comes to interpretability.

Training teacher neural nets.

Our teacher models are fully-connected neural nets (FNNs) with ReLU nonlinearities.

We use the Adam optimizer BID30 with Xavier initialization BID21 and early stopping based on validation loss.

At each depth, we search for optimal hyperparameters (number of hidden units, learning rate, weight decay, dropout probability, batch size, enabling batch norm) based on average validation performance on multiple train-val splits and random initializations.

The most accurate nets we trained are FNNs with 2-hidden layers and 512 hidden units per layer (2H-512,512); nets with three or more hidden layers had lower training loss, but did not generalize as well.

In some experiments we also use a restricted-capacity model with 1 hidden layer of 8 units (1H-8) to compare explanations.

Training student additive explanation models.

To train SAT and SAS, we find optimal feature shapes {h i } p 1 that minimize the mean square error between the teacher F and the studentF , i.e. DISPLAYFORM0 where F (x) is the output of the teacher model (scores for regression tasks and logits for classification tasks), T is the number of training samples, x t is the t-th training sample, and x t i is its i-th feature.

The exact optimization details depend on the choice of h. For trees we use cyclic gradient boosting BID8 BID35 which learns the feature shapes in a cyclic manner.

As trees are high-variance, low-bias learners BID23 , when used as base learners in additive models, it is standard to bag multiple trees BID35 BID36 BID9 .

We follow that approach here.

For splines, we use cubic regression splines trained using penalized maximum likelihood in R's mgcv library (Wood, 2011) and cross-validate the splines' smoothing parameters.

Our global additive explanation models, SAT and SAS, can be visualized as feature shapes ( Figure 1 ).

These are plots with the x-axis being the domain of input feature x i and the y-axis being the feature's contribution to the prediction h i (x i ).

This way of representing the relationship between input features and model predictions has precedence in interpretability, from work that learned monotonic BID22 or concave/convex BID39 feature shapes from original data (i.e. without distillation), to post-hoc explanations such as partial dependence BID17 , and Shapley additive explanations dependence plots BID37 .

The latter two are hence natural baselines for SAT and SAS, and we describe the results from our comparison in Section 4.2.1.

In Section 4.3, we also describe the results of a user study to evaluate the interpretability of feature shapes, showing that humans are able to understand and use feature shapes.

How are feature shapes different from feature attribution?

A classic way to interpret black-box models is feature attribution/importance measures.

Examples include permutation-based measures BID6 , gradients/saliency (see BID38 or BID1 for a review), and measures based on variance decomposition BID28 , game theory BID14 BID37 , etc.

We highlight that feature shapes are different from and more expressive than feature attributions.

Feature attribution is a single number describing the feature's contribution to either the prediction of one sample (local) or the model (global), whereas our feature shapes describe the contribution of a feature, across the entire domain of the feature, to the model.

Nonetheless, feature attribution, both global and local, can be automatically derived from feature shapes: global feature attribution by averaging feature shape values at each unique feature value; local feature attribution by simply taking one point on the feature shape.

In Section 4.3 we show that humans are able to derive feature attribution from feature shapes.

BID37 suggested the perspective of viewing an explanation of a model's prediction as a model itself.

With this perspective, we propose to quantitatively evaluate explanation models as if they were models.

Specifically, we evaluate not just fidelity (how well the explanation matches the teacher's predictions) but also accuracy (how well the explanation predicts the original label).

Note that BID37 and BID40 evaluated local fidelity (called local accuracy by BID37 ), but not accuracy.

A similar evaluation of global accuracy was performed by BID29 who used their explanations (prototypes) to classify test data.

In our case, we use the feature shapes generated by our approach to predict on independent test data.

Baselines.

We compare to two types of baselines: (1) additive explanations obtained by querying the neural net (i.e. without distillation): partial dependence, Shapley additive explanations BID37 and linearization through gradients; (2) interpretable models learned by distilling the neural net: trees, rules, and sparse linear models.

Partial dependence (PD) is a classic global explanation method that estimates how predictions change as feature x j varies over its domain: DISPLAYFORM0 where the neural net is queried with new data samples generated by setting the value of their x j feature to z, a value in the domain of x j .

Plotting P D(x j = z) by z returns a feature shape.

Linearization through gradient approximation (GRAD).

We construct the additive function G through the Taylor decomposition of F , defining DISPLAYFORM1 ???xi x i , and defining the attribution of feature i of value x i as ???F (x) ???xi x i .

This formulation is related to the "gradient*input" method (e.g. Shrikumar et al. (2017) ) used to generate saliency maps for images.

.

SHAP is a stateof-the-art local explanation method that satisfies several desirable local explanation properties BID37 .

Given a sample and its prediction, SHAP decomposes the prediction additively between features using a game-theoretic approach.

We use the python package by the authors of SHAP.Both GRAD and SHAP provide local explanations that we adapt to a global setting by averaging the generated local attributions at each unique feature value.

For example, the global attribution for feature "Temperature" at value 10 is the average of local attribution "Temperature" for all training samples with "Temperature=10".

This is the red line passing through the points in FIG0 .

Applying this procedure to GRAD and SHAP's local attributions, we obtain global attributions gGRAD and gSHAP that we can now plot as feature shapes.

First, we validate our approach on synthetic data with known ground-truth feature shapes (Section 4.1).

Next, we quantitatively evaluate our approach on real data against other non-distilled additive explanations (Section 4.2.1) and distilled, not-necessarily additive, interpretable models (Section 4.2.2).

Third, we design a user study to evaluate the interpretability of feature shapes (Section 4.3).

Finally, we further validate our approach with controlled experiments on real data (Section 4.4).

For this experiment, we simulate data from synthetic functions with known ground-truth feature shapes to see if our approach can recover these feature shapes.

We are particularly interested in observing how predicted feature shapes differ for neural nets of different capacity trained on the same data.

Our expectation is that for neural nets that are accurate, our predicted shapes would match the ground-truth feature shapes, independent of how the features are used internally by the net.

On the other hand, predicted shapes of less accurate neural nets should less accurately match ground-truth shapes.

Experimental setup.

We designed an additive, highly nonlinear function combining components from synthetic functions proposed by BID25 , BID18 and Tsang et al. (2018) : Friedman & Popescu FORMULA1 , we add noise features to our samples that have no effect on F 1 (x) via two noise features x 9 and x 10 .

We trained two teacher neural nets, 2H-512,512 and 1H-8, as described in Section 2.2 to predict F 1 using all ten features.

DISPLAYFORM0 Performance of teachers and students.

The high-capacity 2H neural net obtained test RMSE of 0.14, while the low-capacity neural net obtained test RMSE of 0.48, more than 3x larger.

For each neural net, we used our approach to generate two global additive explanation models, SAT and SAS.

These explanation models are faithful: the reconstruction RMSE of SAT is 0.14 for the 1H model and 0.08 for the 2H model, while the reconstruction RMSE of SAS is 0.14 for the 1H model and 0.07 for the 2H model.

This suggests that both student methods should accurately represent the teacher, and that they will probably be very similar to each other.

Table 1 : RMSE error of the teacher models on all samples, compared to the error on samples sampled from regions where the predicted feature shapes "agree" or "disagree" with the ground truth shape.

Do SAT and SAS explain the teacher model, or just the original data?

The top row of Figure 3 compares the feature shapes of our global explanation models SAT and SAS to function F 1 's analytic ground-truth feature shapes.

SAT and SAS' feature shapes are almost identical.

More importantly, it is clear that the feature shapes for the 2H model are different from shapes for the 1H model, and that the shapes for the 2H model better match ground-truth shapes.

In general, the shapes of the 2H model are very faithful to the ground-truth shapes, but sometimes fall short when there are sharp changes in the ground-truth, highlighting the limitations of a 2-hidden-layer neural net (which achieves 0.14 test RMSE, as noted before).

On the other hand, both SAT and SAS' feature shapes for the 1H neural net show a less accurate teacher model that captures the gist of the ground-truth function but not its details, which is consistent with the original teacher RMSE of 0.48.

This shows that our methods fit what the teacher model has learned, and not the original data, and that when the teacher model is accurate the learned shapes match the ground-truth shapes.

Do SAT and SAS' feature shapes match the real behavior of the model?

To further validate this we use the feature shapes to predict which samples will be inaccurately predicted by the teacher model.

Specifically, we sample testing points from the space regions where the predicted feature shapes agree (or disagree) with the the feature shape ground truth (for example, for the 2H model, x 4 ??? 0, x 7 ??? 0, and |x 6 | ??? 0.3 define a region where the predicted feature shapes and the ground truth feature shapes disagree) and evaluate them using the teacher model.

If the learned feature shapes correctly represent the teacher model, we would expect a lower teacher error on the samples drawn from areas of agreement, and a higher teacher error on the samples drawn from areas of disagreement, compared to the RMSE of all samples.

Indeed, as shown in Table 1 , points sampled on the agreement regions have lower error than points sampled from the disagreement regions.

We performed a two-sample t-test to test if the errors of the samples in the (disjoint) agree and disagree groups are significantly different (p-values 4.3e-21 for 1H, 2.3e-4 for 2H).

Additionally, to be robust against potential violation of the t-tests normal distribution assumption, we also performed a nonparametric Mann-Whitney-Wilcoxon rank sum test (p-values 5.4e-14 for 1H, 1.8e-6 for 2H).

Hence, the difference between the errors is statistically significant, supporting our conclusion that teacher error is higher for samples where feature shapes do not match ground truth, and vice versa, i.e., feature shapes correctly represent the behavior of the models.

How do interactions between features affect feature shapes?

We design an augmented version of F 1 to investigate how interactions in the teacher's predictions are expressed by feature shapes: DISPLAYFORM1 .

We again simulate 50,000 samples.

Note that this function is much harder to learn (the 2H model obtained an RMSE of 0.21) and also harder for students that do not model interactions to mimic (SAT and SAS obtain fidelity RMSEs of 0.35).

The bottom row of Figure 3 displays features with interactions (x 4 , x 2 ) and a feature without interactions (x 8 ), and compares them with the shapes from F 1 .

For x 4 the part of the interactions that can be approximated additively by h i 's has been absorbed into the h i feature shapes, changing their shapes as expected.

On the other hand, we were still able to recover perfectly the feature shapes of features without interactions (e.g. x 8 ).

An interesting case is x 2 , where, despite interacting with x 1 , its feature shape has not changed.

This is less surprising if we recall that feature shapes describe the expected importance of the feature, learned in a data-driven fashion.

The interaction term is x 1 x 2 , which, for x 1 ??? U(???1, 1), has an expected value of zero, and therefore does not affect the feature shape.

Similarly, the expected value of |x 3 | 2|x4| when x 3 ??? U(???1, 1) is 1/(2|x 4 | + 1), an upward pointing cusp, which modifies the feature shape as shown in Figure 3 (bottom left figure).

We selected five data sets: two UCI data sets (Bikeshare and Magic), a Loan risk scoring data set from an online lending company (LendingClub, 2011), the 2018 FICO Explainable ML Challenge's credit data set BID16 , and the pneumonia data set analyzed by BID9 .

TAB2 provides details about the data sets and performance of the 1H and 2H neural nets.

2H neural nets exhibited the most gain in accuracy over 1H neural nets on Bikeshare, Loan, and Magic.

For the rest of this section we focus on 2H neural nets; results for 1H neural nets are in the Appendix.

TAB4 presents the fidelity and accuracy results for SAT and SAS compared to other additive explanations.

SAT and SAS yield similar results in all cases, both in terms of accuracy and fidelity.

In some cases, such as Magic, SAT (which uses tree base learners) is more accurate, while in some others such as FICO, SAS (which uses spline base learners) has the edge.

Trees are locally adaptive smoothers BID5 better able to adapt to sudden changes in input-output relationships than splines, but that also gives them more capacity to overfit.

We also see this in the feature shapes, where trees tend to be more jagged than splines, particularly in regions with fewer points.

TAB2 .

Hence, methods such as gSHAP excel at local explanations and should be used for those, but, to produce global explanations, global model distillation methods optimized to learn the teacher's predictions perform better.

Figure 5 : Fidelity (RMSE) of SAT compared to other interpretable models on Bikeshare (left) and Pneumonia (right), as a function of model-specific parameter K. Figure 5 presents the fidelity of SAT measured with RMSE (accuracy has similar pattern) compared to two other distilled interpretable models: decision trees (DT) and sparse L1-regularized linear model (SPARSE), both trained using scikit-learn.

We present results as a function of a model-specific parameter K that controls the complexity of the model.

For DT, K represents depth, while for SPARSE it represents the number of features with non-zero coefficients.

For trees, true model complexity falls between K and 2 K because a binary tree of depth K has 2 K leaves (2 K rules), but the complexity is somewhat less than 2 K because there is overlap in the rules resulting from the tree structure.

SPARSE obtained by far the worst results in terms of accuracy and fidelity: even if it is interpretable, linear models do not have the fidelity necessary to accurately represent most teacher models.

Note that two explanation methods that use sparse linear models BID40 and rules (Ribeiro et al., 2018) use them as local (not global) explanations, and only for classification (not regression).

Trees start to match the accuracy of SAT on Bikeshare at depth K = 6 (64 leaves) ( Figure A7 ).

However, the largest tree that is readable on letter-size paper has depth K = 4 (16 leaves).

As seen in the user study in Section 4.3, depth hinders the interpretability of trees.

Furthermore, they do not always perform as well as powerful additive models such as SAT.

For example, on Pneumonia, a depth K = 12 tree (4, 096 leaves) achieved an accuracy of 80.9 AUC and a fidelity of 0.57 RMSE, significantly worse than SAT's 82.24 AUC and 0.35 RMSE.

Table 4 : Quantitative results from user study.

Since SAT-2, DT-2, and SPARSE only had two features, the task to rank five features does not apply.

Since the data error only appeared in the output of SAT-5, DT-4, and S-RULES, the other subjects could not have caught the error.

We tried to compare to rule lists, however, as noted in Section 2.2, state-of-the-art rule lists BID34 BID2 do not support regression which is needed for distillation.

Hence we used a subgroup discovery algorithm BID3 ) that supports regression but does not generate disjoint rules.

For the rest of this paper we call them S-RULES, short for subgroup rules.

Although S-RULES generated semantically meaningful rules, they were no more faithful than SPARSE on Bikeshare (1.42 RMSE), and less faithful than SPARSE on Pneumonia (perhaps because Pneumonia is highly imbalanced).

We now describe the results from a user study to see if feature shapes can be understood and used by humans, comparing them to other interpretable models (DT, SPARSE, S-RULES).

Table 4 presents quantitative results from the user study.

Study design.

50 subjects were recruited to participate in the study.

These subjects -STEM PhD students, or college-educated individuals who had taken a machine learning course -were familiar with concepts such as if-then-else structures (for trees and rule lists), reading scatterplots (for SAT), and interpreting equations (for sparse linear models).

Each subject only used one explanation model (between-subject design) to answer a set of questions (see Section C) covering common inferential and comprehension tasks on machine learning models: (1) Rank features by importance; (2) Describe relationship between a feature and the prediction; (3) Determine how the prediction changes when a feature changes value; (4) Detect an error in the data.

The study proceeded in three stages.

First, we compared the two most accurate and faithful students of the Bikeshare 2H neural net: trees and SAT.

We used the depth 4 tree (16 leaves), the largest tree that is readable on letter-size paper, and which does not lag too far behind the depth 6 tree in accuracy (RMSE: SAT 0.98, DT-6 1, DT-4 1.16).

DT-4 used five features: Hour, Year, Temperature, Working Day, Season ( FIG2 ), hence we select the corresponding five feature shapes to display for SAT ( FIG5 ).

In the first stage, 24 of 50 subjects were randomly assigned to see output from DT-4 or SAT-5.

In the second stage, we experimented with smaller versions of trees and SAT using only the two most important features, Hour and Temperature.

14 of 50 subjects were randomly assigned to see output from SAT-2 or DT-2.

In the last stage, the remaining 12 subjects were randomly assigned to see output from one of the two worst performing models (in terms of accuracy and fidelity): sparse linear models and subgroup-rules.

Can humans understand and use feature shapes?

From the absolute magnitude of the SAT feature shapes as well as Gini feature importance metrics for the tree, we determined the ground truth feature importance ranking (in decreasing order): Hour, Temperature, Year, Season, Working Day.

More SAT-5 than DT-4 subjects were able to rank the top 2 and all features correctly (75% vs. 58%, see Table 4 ).

When ranking all 5 features, 0% of the DT-4 subjects were able to predict the right order, while 45% of the SAT-5 subjects correctly predicted the order of the 5 features, showing that ranking feature importance for trees is actually a very hard task.

The most common mistake made by DT-4 subjects (42% of subjects) was to invert the ranking of the last two features, Season and Working Day, perhaps because Working Day's first appearance in the tree (in terms of depth) was before Season's first appearance ( FIG2 ).When asked to describe, in free text, the relationship between the variable Hour and the label, one SAT-5 subject wrote:There are increases in demand during two periods of commuting hours: morning commute (e.g. 7-9 am) and evening commute (e.g. 4-7 pm).

Demand is flat during working hours and predicted to be especially low overnight, whereas DT-4 subjects' answers were not as expressive, e.g.:Demand is less for early hours, then goes up until afternoon/evening, then goes down again.75% of SAT-5 subjects detected and described the peak patterns in the mornings and late afternoons, and 42% of them explicitly mentioned commuting or rush hour in their description.

On the other hand, none of the DT-4 subjects discovered this pattern on the tree: most (58%) described a concave pattern (low and increasing during the night/morning, high in the afternoon, decreasing in the evening) or a positively correlated relation (42%).

Similarly, more SAT-5 subjects were able to precisely compute the change in prediction when temperature changed in value, and detect the error in the data -that spring had lower bike demand whereas winter had high bike demand (bottom right feature shape in FIG5 ).How do tree depth and number of feature shapes affect human performance?

We also experimented with smaller models, SAT-2 and DT-2, that used only the two most important features, Hour and Temperature.

As the models are simpler, some of the tasks become easier.

For example, SAT-2 subjects predict the order of the top 2 features 100% of the time (vs 75% for SAT-5), and DT-2 subjects, 85% of the time (vs 58% for DT-4).

The most interesting change is in the percentage of subjects able to compute the change in prediction after changing a feature: only 25% for DT-4, compared to 100% for DT-2.

Reducing the complexity of the explanation made using it easier, at the price of reducing the fidelity and accuracy of the explanation.

Another important aspect is the time needed to perform the tasks: increasing the number of features from 2 to 5 increases the time needed by the subjects to finish the study by 60% for the SAT model, but increases it by 166% for the DT model, that is, interpreting a tree becomes much more costly as the tree becomes deeper (and more accurate), and, in general, subjects make more mistakes.

SAT appears to scale up more gracefully.

Remaining interpretable models: subgroup-rules and sparse linear models.

These explanations were the least accurate and faithful.

We found that human subjects can easily read the (few) weights of SPARSE, establish feature importance, and compute prediction changes, and do so quickly -at 5.1 minutes on average, this was the fastest explanation to interpret.

However, the model is highly constrained and hid interesting patterns.

For example, 100% of the subjects described the relation between demand and hour as increasing, and 83% predicted the exact amount of increase, but none were able to provide insights like the ones provided by SAT-5 and DT-4 subjects.

S-RULES was the second hardest explanation to interpret based on mean time required to answer the questions: 14.9 minutes.

Understanding non-disjoint rules appears to be hard: none of the subjects correctly predicted the feature importance order, even for just two features; none were able to compute exactly the change in prediction when feature value changes, and none were able to find the data error.

The rules in S-RULES are not disjoint because we could not find a regression implementation of disjoint rules.

However, 66% of the subjects discovered the peak during rush hour, as that appeared explicitly in some rules, e.g. "If hour=17 and workingday=yes then bike demand is 5".To summarize, feature shapes, the interpretable representation we focus on in this paper: (1) allowed humans to perform better (than decision trees, sparse linear models, and rules) at ranking feature importance, pointing out patterns between certain feature values and predictions, and catching a data error; (2) Feature shapes were also faster to understand than big decision trees; (3) Very small decision trees and sparse linear models had the edge in calculating how predictions change when feature values change, but were much less faithful and accurate.

In this section we further validate global additive explanations on real data.

Although here we do not have an analytic solution for the ground-truth feature shapes, we can still design experiments where we modify data in ways that will lead to expected known changes to the ground-truth feature shapes and then verify that these changes are captured in the learned feature shapes.

Label modification.

On Bikeshare, we added 1.0 to the label (the number of rented bikes) for samples where one of the features (humidity) is between 55 and 65.

We then retrained a 2H neural net on the modified data, and applied our approach to learn feature shapes from the 2H net.

Ideally, the feature shapes of that new neural net should be almost identical to those of the original net except in that particular range of the humidity feature, where we should see an abrupt "bump" that increases its feature shape value by one.

FIG3 (left) displays the feature shapes.

Our method was able to recover the change to the label for the neural net in the new feature shape.

Data modification: expert discretization.

Sometimes features are transformed before training.

For example, in medical data, continuous variables such as body temperature may be discretized by domain experts into bins such as normal, mild fever, moderate fever, high fever, etc.

In this experiment we test if our additive explanation models can recover these discretizations from the neural net without access to the discretized features.

We train our student additive models using as input features the original un-discretized features, but using as labels the outputs of a neural net that was trained on discretized features.

Our expectation is that if the student models are an accurate representation of what the neural net learned from the discretized features, they will detect the discretizations, even if they never have access to the discretized features or to the internal structure of the neural-net teacher.

We study the feature shapes of two features in the Pneumonia data (Blood pO 2 and Respiration Rate) in FIG3 , where we compare the feature shapes learned from teachers trained on the original continuous data (dotted lines) with those from teachers trained on discretized features (solid lines).

Recall that in both cases the student models only saw non-discretized features to generate feature shapes.

Our approach captures the expected discretization intervals (in yellow) as described in Cooper et al. (1997) .We discuss extensions & applications of our approach in Section D in the Appendix, including visualizing a neural net as it is trained (https://youtu.be/ATNcgurNHhc).

We presented a method for "opening up" complex models such as neural nets trained on tabular data.

The method, based on distillation with high-accuracy additive models, has clear advantages over other approaches that learn additive explanations but not using distillation, and non-additive explanations using distillation.

Our global additive explanations do not aim to compete with local explanations or non-additive explanations such as decision trees.

Instead, we show that different interpretable representations work well for different tasks, and global additive explanations are valuable for important tasks that require quick understanding of feature-prediction relationships.

Although in this paper we focus on explaining FNNs, the method will work with any classification or regression model including random forests and CNNs, but is not designed to work with raw image inputs such as pixels where providing a global explanation in terms of input pixels is not meaningful.

One way to address this is to define more meaningful "features", e. hi (xi) hi (xi) hi(xi) Figure A1 : Feature shapes for features x 1 to x 9 of F 1 from Section 4.1.

Notice how x 9 , which is a noise feature that does not affect F 1 , has been assigned an importance of approximately 0 throughout its range.

The feature shape of x 10 , another noise feature, is very similar to x 9 and hence not included here.hi (xi) hi (xi) hi(xi) FIG0 : Feature shapes for features x 1 to x 9 of F 2 from Section 4.1.

Notice how x 9 , which is a noise feature that does not affect F 2 , has been assigned an importance of approximately 0 throughout its range.

The feature shape of x 10 , another noise feature, is very similar to x 9 and hence not included here.

Table A1 : Accuracy and fidelity of global explanation models across 1H and 2H teacher neural nets and datasets.

TAB4 is a subset of this table with only 2H neural nets.

In general, the lower-capacity 1H neural nets are easier to approximate (i.e. better student-teacher fidelity), but their explanations are less accurate on independent test data.

Students of simpler teachers tend to be less accurate even if they are faithful to their (simple) teachers.

One exception is the FICO data, where the fidelity of the 2H explanations is better.

Our interpretation is that many features in the FICO data have almost linear feature shapes (see Figure A5 for a sample of features), and the 2H model may be able to better capture fine details while being simple enough that it can still be faithfully approximated.

The accuracy of the SAT and SAS for 1H and 2H neural nets are comparable, taking into account the confidence intervals.

On the Magic data, the fidelity of the gGRAD explanation to the 1H neural net (see * in Table A1 ) is markedly worse than other explanation methods.

We investigate the individual gradients of the 1H neural net with respect to each feature ( DISPLAYFORM0 ???xi in GRAD equation in Section 3).

99% of them have reasonable values (between -5.6 and 6).

However, 3 are larger than 1,000 (with none between 6 and 1,000) and 13 are lower than -1,000 (with none between -1,000 and -5.6), resulting in the ensuing gGRAD explanation generating extreme predictions for several samples that are not faithful to the teacher's predictions.

Because AUC is a ranking loss, accuracy (AUC) is less affected than fidelity (RMSE) by the presence of these extreme values.

This shows that gGRAD explanations may be problematic when individual gradients are arbitrarily large, e.g. in overfitted neural nets.

Figure A7 , removing the color and number of samples in each node, to improve readability for the user study.

In this section we discuss applications of our approach and extensions to include higher-order interactions.

Checking for monotonicity.

Domains such as credit scoring have regulatory requirements that prescribe monotonic relationships between predictions and some features (Federal Reserve Governors, 2007) .

For example, the 2018 FICO Explainable ML Challenge encouraged participants to impose monotonicity on 16 features BID16 .

We use feature shapes to see if the function learned by the neural net is monotone for these features.

15 of 16 features are monotonically increasing/decreasing as required.

One feature, however, "Months Since Most Recent Trade Open" was expected to decrease monotonically, but actually increased monotonically.

This is true not just in our explanations, but also in PD, gGRAD, and gSHAP ( Figure A5 ).

Note that testing for monotonicity requires global explanations or checking and aggregating many local explanations.

With the insight from the global explanations that the neural net may not be exhibiting the expected pattern for "Months Since Most Recent Trade Open", we perform a quick experiment to verify this in the neural net.

We sample values of this feature across its domain, set all data samples to this value (for this feature), and obtain the neural net's predictions for these modified samples.

The majority of samples (70%) had predictions that increased as this feature increased across its domain, confirming that on average, the neural net exhibits a monotonically increasing instead of decreasing pattern for this feature.

Note that we could not have checked for a monotonicity pattern (which is by definition a global behavior) without checking and aggregating multiple local explanations.hi(xi) Figure A5 : 3 of 16 features with expected monotonically increasing/decreasing patterns in the FICO data.

"

Months Since Most Recent Trade Open", the leftmost figure, was expected to decrease monotonically, but actually increased monotonically according to all explanations.

The two figures on the right are two related features, "Months Since Oldest Trade Open" and "Number of Trades Open in Last 12 Months", both of which exhibit the expected monotonically decreasing/increasing patterns.

Visualizing neural net training: from underfit to overfit.

Using additive models to peek inside a neural net creates many opportunities.

For example, we can see what happens in the neural net when it is underfit or overfit; when it is trained with different losses such as squared, log, or rank loss or with different activation functions such as sigmoid or ReLUs; when regularization is performed with dropout or weight decay; when features are coded in different ways; etc.

The video at https: //youtu.be/ATNcgurNHhc shows what is learned by a neural net as it trains on a medical dataset.

The movie shows feature shapes for five features before, at, and after the early-stopping point as the neural net progresses from underfit to optimally fit to overfit.

We had expected that the main cause of overfitting would be increased non-linearity (bumpiness) in the fitting function, but a significant factor in overfitting appears to be unwarranted growth in the confidence of the model as the logits grow more positive or negative than the early-stopping shape suggests is optimal.

FIG3 : An important pairwise interaction in Bikeshare.

Functions learned by neural nets cannot always be represented with adequate fidelity by the additive functionF in equation 1.

We can improveF 's expressive power by adding pairwise and higher-order components h ij , h ijk , and so on to account for interactions between two or more input features.

In Bikeshare, RMSE decreases from 0.98 to 0.60 when we add pairwise interactions to the student model.

FIG3 shows an interesting interaction between two features: "Time of Day", and "Working Day".

On working days, the highest bike rental demand occurs at 7-9am and 5-7pm, but on weekends there is very low demand at 7-9am (presumably because people are still sleeping) and at 5-7pm, and demand peaks during midday from 10am-4pm.

These two features also form a three-way interaction with temperature.

Whenever the teacher neural net learned these (and other) interactions, a global explanation method must also incorporate interactions if it is to provide high-fidelity explanations of the teacher model.

Our approach is able to do so by adding higher-order components h ij , h ijk , and so on to the global additive explanationF .

E TREE THAT MATCHED SAT FIDELITY ON BIKESHARE DATASET (FROM SECTION 4.2.2) Figure A7 : Tree of depth 6 (64 leaves), the least deep tree that matched SAT's fidelity.

This uses the default tree visualizer in scikit-learn.

For the tree of depth 4 (DT-4) presented in the user study ( FIG2 ), we removed the color and number of samples in each node to increase readability.

@highlight

We propose to leverage model distillation to learn global additive explanations in the form of feature shapes (that are more expressive than feature attributions) for models such as neural nets trained on tabular data.

@highlight

This paper incorporates Generalized Additive Models (GAMs) with model distillation to provide global explanations of neural nets.