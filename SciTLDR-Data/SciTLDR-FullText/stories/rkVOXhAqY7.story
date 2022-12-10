We present a new family of objective functions, which we term the Conditional Entropy Bottleneck (CEB).

These objectives are motivated by the Minimum Necessary Information (MNI) criterion.

We demonstrate the application of CEB to classification tasks.

We show that CEB gives: well-calibrated predictions; strong detection of challenging out-of-distribution examples and powerful whitebox adversarial examples; and substantial robustness to those adversaries.

Finally, we report that CEB fails to learn from information-free datasets, providing a possible resolution to the problem of generalization observed in Zhang et al. (2016).

The field of Machine Learning has suffered from the following well-known problems in recent years 1 :• Vulnerability to adversarial examples.

Essentially all machine-learned systems are currently believed by default to be highly vulnerable to adversarial examples.

Many defenses have been proposed, but very few have demonstrated robustness against a powerful, general-purpose adversary.

Lacking a clear theoretical framework for adversarial attacks, most proposed defenses are ad-hoc and fail in the presence of a concerted attacker BID8 BID5 ).•

Poor out-of-distribution detection.

Classifiers do a poor job of signaling that they have received data that is substantially different from the data they were trained on.

Ideally, a trained classifier would give less confident predictions for data that was far from the training distribution (as well as for adversarial examples).

Barring that, there would be a clear, principled statistic that could be extracted from the model to tell whether the model should have made a low-confidence prediction.

Many different approaches to providing such a statistic have been proposed BID18 BID28 BID19 BID32 BID30 BID13 , but most seem to do poorly on what humans intuitively view as obviously different data.• Miscalibrated predictions.

Related to the issues above, classifiers tend to be very overconfident in their predictions BID18 .

This may be a symptom, rather than a cause, but miscalibration does not give practitioners confidence in their models.• Overfitting to the training data.

BID48 demonstrated that classifiers can memorize fixed random labelings of training data, which means that it is possible to learn a classifier with perfect inability to generalize.

This critical observation makes it clear that a fundamental test of generalization is that the model should fail to learn when given what we call information-free datasets.

I

Consider a joint distribution, p(x, y), represented by the graphical model: DISPLAYFORM0 This joint distribution is our data, and may take any form.

We don't presume to know how the data factors.

It may factor as p(x, y) = p(x)p(y|x), p(x, y) = p(y)p(x|y), or even p(x, y) = p(x)p(y).The first two factorings are depicted in FIG0 in a standard information diagram showing the various entropies and the mutual information.

We can ask: given this generic setting, what is the optimal representation?

It seems there are only two options: capture all of the information in both X and Y (measured by the joint entropy, H(X, Y)), or capture only the information shared between X and Y (measured by the mutual information, I(X; Y)).The field of lossless compression is concerned with representations that perfectly maintain all of the information in both X and Y, as are the closely related studies of Kolmogorov Complexity BID25 and Minimum Description Length (MDL) BID17 , all three of which are concerned with perfect reconstruction of inputs or messages.

In contrast, we think that the field of machine learning is primarily concerned with making optimal predictions on unseen data.

The requirements of perfect reconstruction from a compressed representation may result in the retention of much more information in the model than may be needed for prediction or stochastic generation tasks.

For most such machine learning tasks, this points towards learning representations that capture only the information shared between X and Y, which is measured by the mutual information, I(X; Y).The mutual information is defined in a variety of ways; we will use two (Cover & BID12 : DISPLAYFORM1 I(X; Y) measures the amount of information necessary to define the relationship between X and Y. For some fixed dataset X, Y, any information less than I(X; Y) must be insufficient to predict Y from X or vice-versa with minimal error.

Equivalently, any information more than I(X; Y) must contain some superfluous information for those two tasks.

For example, consider a labeled dataset, where X is high-dimensional and information-rich, and Y is a single integer.

All of the information in X that is not needed to correctly predict the single value Y = y is useless for the prediction task defined by the dataset, and may be harmful to the performance of a machine learning system if retained in the learned representation, as we will show empirically below.

Next, we formalize this intuition about the information required for an optimal representation.

We propose the Minimum Necessary Information (MNI) criterion for a learned representation.

We can define MNI in three parts.

First is Information: we would like a representation that captures semantically meaningful information.

In order to measure how successfully we capture meaningful information, we must first know how to measure information.

Thus, the criterion prefers informationtheoretic approaches, given the uniqueness of entropy as a measure of information BID36 .

The semantic value of information is given by a task, which is specified by the set of variables in the dataset.

I.e., the dataset X, Y defines two tasks: predict Y given X, or predict X given Y.

This brings us to Necessity: the information we capture in our representations must be necessary to solve the task.

2 Finally, Minimality: this simply refers to the amount of information -given that we learn a representation that can solve the task, we require that the representation we learn retain the smallest amount of information about the task out of the set of all representations that solve the task.

This part of the criterion restricts us from incorporating "non-semantic" information into our representation, such as noise or spurious correlation.

More formally, in the case of two observed variables, X and Y, a necessary set of conditions for a representation Z to satisfy the MNI criterion is the following: DISPLAYFORM0 This fully constrains the amount of information.

To constrain the necessity of the information in the representation Z, the following conditions must be satisfied: DISPLAYFORM1 These four distributions of z correspond to the two tasks: predict Y given X and predict X given Y.

One way to satisfy Equation FORMULA2 is to learn a representation Z X of X only, indicated by the Markov chain Z X ← X ↔ Y.

We show this Markov chain as an information diagram in FIG0 (Right).

The placement of H(Z X ) in that diagram carefully maintains the conditional independence between Y and Z X given X, but is otherwise fully general.

Some of the entropy of Z X is unassociated with any other variable; some is only associated with X, and some is associated with X and Y together.

FIG0 (Right), then, shows diagrammatically the state of the learned representation early in training.

At the end of training, we would like Z X to satisfy the equalities in Equation (2), which corresponds to FIG0 (Left), where the gray region labeled I(X; Y) also corresponds to I(X; Z X ) and I(Y; Z X ).Given the conditional independence Z X Y|X in our Markov chain, I(Y; Z X ) is maximal at I(X; Y), by the data processing inequality.

However, I(X; Z X ) does not clearly have a constraint that targets I(X; Y).

We cannot maximize I(X; Z X ) in general while being compatible with the MNI criterion, as that is only constrained from above by H(X) ≥ I(X; Y).

Instead, we could use the Information Bottleneck objective BID40 which starts from the same Markov chain and minimizes βI(X; Z X ) − I(Y; Z X ), but it is not immediately clear what value of β will achieve the MNI.Thus, we need a different approach to hit the MNI.

Considering the information diagram in FIG0 (Left), we can notice the following identities when when we have achieved the MNI: DISPLAYFORM0 With our Markov chain and the chain rule of mutual information (Cover & BID12 , we have: DISPLAYFORM1 This conditional information is guaranteed to be non-negative, as both terms are mutual informations, and the Markov chain guarantees that I(Y; Z X ) is no larger than I(X; Z X ), by the data processing inequality.

From an optimization perspective, this is ideal -we have a term that we can minimize, and we can directly know how far we are from the optimal value of 0 (measured in nats, so it is interpretable), when we are done (when it's close enough to 0 that we are satisfied), and when our model is insufficient for the task (i.e., when this term isn't close enough to 0).

This leads us to the general Conditional Entropy Bottleneck objective: DISPLAYFORM2 Typically we would add a Lagrange multiplier on one of the two terms.

In Appendix A, we present some geometric arguments to prefer leaving the two terms balanced.

It is straightforward to turn this into a variational objective function that we can minimize.

Taking the terms in turn: DISPLAYFORM3 e(z X |x) is our encoder.

It is not a variational approximation, even though it has learned parameters.

b(z X |y) is the backward encoder, a variational approximation of p(z X |y).In the second term, H(Y) can be dropped because it is constant with respect to the model: DISPLAYFORM4 c(y|z x ) is the classifier (although that name is arbitrary, given that Y may not be labels), which variationally approximates p(y|z X ).The variational bounds derived above give us a fully tractable objective function that works on large-scale problems and supports amortized inference, Variational Conditional Entropy Bottleneck (VCEB): DISPLAYFORM5 The distributions with letters other than p are assumed to have learned parameters, which we otherwise omit in the notation.

In other words, all three of e(·), b(·), and c(·) have learned parameters, just as in the encoder and decoder of a normal VAE BID24 , or the encoder, classifier, and marginal in a VIB model.

We will name the I(X; Z X |Y) term the Residual Information -this is the excess information in our representation beyond the information shared between X and Y: DISPLAYFORM6 There are a number of natural variations on this objective.

We describe a few of them in Appendix E.

The Information Bottleneck (IB) BID40 learns a representation of X and Y subject to a soft information constraint: DISPLAYFORM0 where β controls the size of the constraint.

In Figure 2 we show the optimal surfaces for CEB and IB, labeling the MNI point on both.

In Figure 4 we show the same surfaces for finite models and that adjusting β determines a unique point in these information planes relative to I(X; Y).As described in BID40 , IB is a tabular method, so it is not usable for amortized inference.5 Two recent works have extended IB for amortized inference.

Both of these approaches 4 We write expectations log e(z X |x) .

They are always with respect to the joint distribution; here, that is p(x, y, z X ) = p(x, y)e(z X |x).5 The tabular optimization procedure used for IB trivially applies to CEB, just by setting β = 1 2.

A recent work on IB using tabular methods is the Deterministic Information Bottleneck BID38 , which learns hard clusterings, rather than the soft clusterings of earlier IB approaches.

Figure 2: Geometry of the optimal surfaces for IB and CEB, with all points labeled.

CEB rectifies IB's parallelogram by subtracting I(Y; Z) at every point.rely on sweeping β, and do not propose a way to set β directly to train models where I(X; Z) = I(Y; Z) = I(X; Y).

BID0 presents InfoDropout, which uses IB to motivate a variation on Dropout BID37 .

A varational version of IB is presented in BID2 .

That objective is the Variational Information Bottleneck (VIB): DISPLAYFORM0 Instead of the backward encoder, VIB has a marginal posterior, m(z X ), which is a variational approximation to e(z X ) = dx p(x)e(z X |x).

Additionally, it has a hyperparameter, β.

We show in Appendix A that the optimal value for β = 1 2 when attempting to adhere to the MNI criterion.

Following , we define the Rate (R): DISPLAYFORM1 We can compare variational CEB with VIB by taking their difference at β = 1 2 .

Note that both objectives have an elided dependence on log p(y) from the I(Y; Z X ) term that we must track: DISPLAYFORM2 Solving for m(z X ) when that difference is 0: DISPLAYFORM3 Since the optimal m * (z X ) is the marginalization of e(z X |x), at convergence we must have: DISPLAYFORM4 Depending on the distributional families and the parameterizations, this point may be difficult to find, particularly given that m(z X ) only gets information about y indirectly through e(z X |x).

Consequently, for otherwise equivalent models, we may expect V IB 1 2 to converge to a looser approximation of I(X; Z) = I(Y; Z) = I(X; Y) than CEB.

Since VIB optimizes an upper bound on I(X; Z), that means that V IB 1 2 will report R converging to I(X; Y), but will capture less than the MNI.

In contrast, if Re X/Y converges to 0, the variational tightness of b(z X |y) to the optimal p(z X |y) depends only on the tightness of c(y|z X ) to the optimal p(y|z X ).

6 MNI Optimality of CEB In this work we do not attempt to give a formal proof that CEB representations learn the optimal information about the observed data (and certainly the variational form of the objective will prevent that from happening in general cases).

However, CEB's targeting of the MNI is motivated by the following simple observations: If I(X; Z) < I(X; Y), then we have thrown out relevant information in X for predicting Y. If I(X; Z) > I(X; Y), then we are including information in X that is not useful for predicting Y. Thus I(X; Z) = I(X; Y) is the "correct" amount of information, which is one of the equalities required in order to satisfy the MNI criterion.

Only models that successfully learn that amount of information can possibly be MNI-optimal.

The second condition of MNI (Equation FORMULA3 ) is only fully satisfied when optimizing the bidirectional CEB objective, described in Appendix E.2, as log e(z X |x) − log b(z X |y) and log b(z Y |y) − log e(z Y |x) are both 0 only when b(z|y) = p(z|y) and e(z|x) = p(z|x) and the corresponding decoder terms are both maximal.

We leave such models for future work.

Our primary experiments are focused on comparing the performance of otherwise identical models when we change only the objective function.

Consequently, we aren't interested in demonstrating state-of-the-art results for a particular classification task.

Instead, we are interested in relative differences in performance that can be directly attributed to the difference in objective.

With that in mind, we present results for classification of Fashion MNIST BID46 for five different models.

The five models are: a deterministic model (Determ); three VIB models, with β ∈ { 1 2 , 10 −1 , 10 −2 } (VIB 0.5 , VIB 0.1 , VIB 0.01 ); and a CEB model.

These same models are used in the calibration, out-of-distribution, and adversarial experiments (Sections 8 to 10).

Critically, all five models share the same inference architecture mapping X to Y. See Appendices C and D for details on training and the architectures.

Since Fashion MNIST doesn't have a prespecified validation set, it offers an opportunity to test training algorithms that only look at training results, rather than relying on cross validation.

To that end, the five models presented here are the first models with these hyperparameters that we trained on Fashion MNIST.

6 The learning rate for the CEB model was lowered according to the training algorithm described in Appendix C. The other four models followed the same algorithm, but instead of tracking Re X/Y , they simply tracked their training loss.

All five models were required to retain the initial learning rate of 0.001 for 40 epochs before they could begin lowering the learning rate.

At no point during training did any of the models exhibit non-monotonic test accuracy, so we do not believe that this approach harmed any performance -all five models converged essentially smoothly to their final, reported performance.

In spite of the dynamic learning rate schedule, all five models took approximately the same number of epochs to reach the minimum learning rate.

Underconfidence occurs when the points are above the diagonal.

Overconfidence occurs when the points are below the diagonal.

In the case of a simple classification problem with a uniform distribution over classes in the training set, we can directly compute I(X; Y) as log C, where C is the number of classes.

7 See TAB0 for a comparison of the rates between the four variational models, as well as their accuracies.

All but VIB 0.5 achieve the same accuracy.

All four stochastic models get close to the ideal rate of 2.3 nats, but they get there by different paths.

For the VIB models, the lower β is, the higher the rate goes early in training, before converging down to (close to) 2.3 nats.

CEB never goes above 2.3 nats.

In FIG1 , we show calibration plots at various points during training for the four models.

Calibration curves help analyze whether models are underconfident or overconfident.

Each point in the plots corresponds to a 5% confidence range.

Accuracy is averaged for each bin.

A well-calibrated model is correct half of the time it gives a confidence of 50% for its prediction.

All of the networks move from under-to overconfidence during training.

However, CEB and VIB 0.5 are only barely overconfident, while β = 0.1 is sufficent to make it nearly as overconfident as the deterministic model.

This overconfidence is one of the issues that is correlated with exceeding the MNI during training TAB0 .

See Appendix A for a geometric explanation for how this can occur.

We test the ability of the five models to detect three different out-of-distribution (OoD) detection datasets.

U(0, 1) is uniform noise in the image domain.

MNIST uses the MNIST test set.

Vertical Flip is the most challenging, using vertically flipped Fashion MNIST test images, as originally proposed in .We use three different metrics for thresholding.

The first two, H and R, were proposed in .

H is the classifier entropy.

R is the rate, defined in Section 5.

The third metric is specific to CEB: Re X/Ŷ .

This is the predicted residual information -since we don't have access to the true value of Y at test time, we useŷ ∼ c(y|z X ) to calculate H(Z X |Ŷ).

This is no longer a valid bound on Re X/Y , asŷ may not be from the true distribution p(x, y, z X ).

However, the better the classifier, the closer the estimate should be.

These three threshold scores are used with the standard suite of proper scoring rules: False Positive Rate at 95% True Positive Rate (FPR 95% TPR), Area Under the ROC Curve (AUROC), and Area Under the Precision-Recall Curve (AUPR).

See BID31 for definitions.

The core result is that VIB 0.5 performs much less well at the OoD tasks than the other two VIB models and CEB.

We believe that this is another result of VIB 0.5 learning the right amount of information, but not learning all of the right information, thereby demonstrating that it is not a valid MNI objective, as explored in Appendix A. On the other hand, the other two VIB objectives seem to perform extremely well, which is the benefit they get from capturing a bit more information about the training set.

We will see below that there is a price for that information, however.

Adversarial examples were first noted in BID39 .

The first practical attack, Fast Gradient Method (FGM) was introduced shortly after BID15 .

Since then, many new attacks have been proposed.

Most relevant to us is the Carlini-Wagner (CW) attack BID9 , which was the first practical attack to directly use a blackbox optimizer to find minimal perturbations.

8 Many defenses have also been proposed, but almost all of them are broken BID8 BID5 .

This work may be seen as a natural continuation of the adversarial analysis of BID2 , which showed that VIB naturally had robustness to whitebox adversaries, including CW.

In that work, the authors did not train any VIB models with a learned m(z X ), which results in much weaker models, as shown in .

We believe this is the first work that trains a VIB model with a learned marginal and using it in an adversarial setting.

BID9 .

CW, (C = 1) is CW with an additional confidence penalty set to 1.

CW, (C = 1) Det. is a custom CW attack targeting CEB's detection mechanism, Re X/Ŷ .

L 0 , L 1 , L 2 , L ∞ report the corresponding norm (mean ±1 std.) of successful adversarial perturbations.

Higher norms on CW indicate that the attack had a harder time finding adversarial perturbations, since it starts by looking for the smallest possible perturbation.

The remaining columns are as in TAB1 .

Arrows denote whether higher or lower scores are better.

Bold indicates the best score in that column for a particular adversarial attack.

We consider CW in the whitebox setting to be the current gold standard attack, even though it is more expensive than FGM or the various iterative attacks like DeepFool BID33 or iterative variants of FGM BID27 .

Running an optimizer directly on the model to find the perturbation that can fool that model tells us much more about the robustness of the model than approaches that focus on attack efficiency.

CW searches over the space of perturbation magnitudes, which makes the attack hard to defend against, and thus a strong option for testing robustness.

DISPLAYFORM0 Here, we explore three variants of the CW L 2 targeted attack.

The implementation the first two CW attacks are from BID35 .

CW and CW (C = 1) are the baseline CW attack, and CW with a confidence adjustment of 1.

Note that in order for these attacks to succeed at all on CEB, we had to increase the default CW learning rate to 5 × 10 −1 .

Without that increase, CW found almost no adversaries in our early experiments.

All other parameters are left at their defaults for CW, apart from setting the clip ranges to [0, 1] .

The final attack, CW (C = 1) Det. is a modified version of CW (C = 1) that additionally incorporates a detection tensor into the loss that CW minimizes.

For CEB, we had it target minimizing Re X/Ŷ in order to break the network's ability to detect the attack.

All of the attacks are targeting the trouser class of Fashion MNIST, as that is the most distinctive class.

Targeting a less distinctive class, such as one of the shirt classes, would confuse the difficulty of classifying the different shirts and the robustness of the model to adversaries.

We run each of the first three attacks on the entire Fashion MNIST test set (all 10,000 images).

For the stochastic networks, we permit 32 encoder samples and take the mean classification result (the same number of samples is also used for gradient generation in the attacks to be fair to the attacker).

CW is expensive, but we are able to run these on a single GPU in about 30 minutes.

However, CW (C = 1) Det. ends up being about 200 times more expensive -we were only able to run 1000 images and only 8 encoder samples, and it took 2 1 2 hours.

Consequently, we only run CW (C = 1) Det.

on the CEB model.

Our metric for robustness is the following: we count the number of adversarial examples that change a correct prediction to an incorrect prediction of the target class, and divide by the number of correct predictions the model makes on the non-adversarial inputs.

We additionally measure the size of the resulting perturbations using the L 0 , L 1 , L 2 , and L ∞ norms.

For CW, a larger perturbation generally indicates that the attack had to work harder to find an adversarial example, making this a secondary indication of robustness.

Finally, we measure adversarial detection using the same thresholding techniques from TAB1 .The results of these experiments are in TAB2 .

We show all 20,000 images for four of the models in FIG6 .

The most striking pattern in the models is how well VIB 0.01 and VIB 0.1 do at detection, while VIB 0.5 is dramatically more robust.

We think that this is the most compelling indication of the importance of not overshooting I(X; Y) -even minor amounts of overshooting appear to destroy the robustness of the model.

On the other hand, VIB 0.5 has a hard time with detection, which indicates that, while it has learned a highly compressed representation, it has not learned the optimal set of bits.

Thus, as we discuss in Appendix A, VIB trades off between learning the necessary information, which allows it to detect attacks perfectly, and learning the minimum information, which allows it to be robust to attacks.

The CEB model permits both -it maintains the necessary information for detecting powerful whitebox attacks, but also retains the minimum information, providing robustness.

This is again visible in the CW (C = 1) Det.

attack, which directly targets CEB's detection mechanism.

Even though it no longer does well detecting the attack, the model becomes more robust to the attack, as indicated both by the much lower attack success rate and the much larger perturbation magnitudes.

We replicate the basic experiment from BID48 : we use the images from Fashion MNIST, but replace the training labels with fixed random labels.

This dataset is information-free in the sense that I(X; Y) = 0.

We use that dataset to train multiple deterministic models, CEB models, and a range of VIB models.

We find that the CEB model never learns (even after 100 epochs of training), the deterministic model always learns (after about 40 epochs of training it begins to memorize the random labels), and the VIB models only learn with β ≤ 0.001.The fact that CEB and VIB with β near 1 2 manage to resist memorizing random labels is our final empirical demonstration that MNI is a powerful criterion for objective functions.

We have presented the basic form of the Conditional Entropy Bottleneck (CEB), motivated by the Minimum Necessary Information (MNI) criterion for optimal representations.

We have shown through careful experimentation that simply by switching to CEB, you can expect substantial improvements in OoD detection, adversarial example detection and robustness, calibration, and generalization.

Additionally, we have shown that it is possible to get all of these advantages without using any additional form of regularization, and without any new hyperparameters.

We have argued empirically that objective hyperparameters can lead to hard-to-predict suboptimal behavior, such as memorizing random labels, or reducing robustness to adversarial examples.

In Appendix E and in future work, we will show how to generalize CEB beyond the simple case of two observed variables.

It is our perspective that all of the issues explored here -miscalibration, failure at OoD tasks, vulnerability to adversarial examples, and dataset memorization -stem from the same underlying issue, which is retaining too much information about the training data in the learned representation.

We believe that the MNI criterion and CEB show a path forward for many tasks in machine learning, permitting fast, amortized inference while ameliorating major problems.

a b Figure 4 : Geometry of the optimal surfaces for both CEB (purple) and IB (green) for models that can only come within of the optimal surface (a: = 0.1I(X; Y); b: = 0.01I(X; Y)).

The tangent lines have the slope of the corresponding β -the tangent point on the ball corresponds to the point on the pareto-optimal frontier for the corresponding model.

Note that β determines the "exchange rate" between bits of I(X; Z) and I(Y; Z), which is how we determine the coordinate of the center of the ball.

For IB to achieve the MNI point, 2 bits of I(Y; Z) are needed for every bit of I(X; Z).

Consequently, even for an infitely powerful model (corresponding to = 0), the only value of β that hits the MNI point is β = 2.

Thus, knowing the function (β) for a given model and dataset completely determines the model's pareto-optimal frontier.

Here we collect a number of results that are not critical to the core of the paper, but may be of interest to particular audiences.

A Analysis of CEB and IB From Equation FORMULA5 and the definition of CEB in Equation (6), the following equivalence between CEB and IB is obvious: DISPLAYFORM0 where we are parameterizing IB with β on the I(Y; Z) term for convenience.

This equivalence generalizes as follows: DISPLAYFORM1 DISPLAYFORM2 In Figure 4 , we show the combined information planes for CEB and IB given the above parameterization.

The figures show the simple geometry that determines a point on the pareto-optimal frontier for both objectives.

Every such point is fully determined by the function (β) for a given model and dataset, where is the closest the model can approach the true optimal surface.

(β) = 0 corresponds to the "infinite" model family that exactly traces out the boundaries of the feasible region.

The full feasible regions can be seen in Figure 2 .From this geometry we can immediately conclude that if an IB model and a CEB model have the same value of > 0 at equivalent β, the CEB model will always yield a value of I(Y; Z) closer to I(X; Y).

This is because the slope of the tangent lines for CEB are always lower, putting the tangent points higher on the ball.

This gives part of a theoretical justification for the empirical observations above that V IB 0.5 (equivalent to IB 2 in the parameterization we are describing here) fails to capture as much of the necessary information as the CEB model.

Even at the pareto-optimal frontier, V IB 0.5 cannot get I(Y; Z) as close to I(X; Y) as CEB can.

Of course, we do not want to claim that this effect accounts for the fairly substantial difference in performance -that is likely to be due to a combination of other factors, including the fact that it is often easier to train continuous conditional distributions (like b(z|y)) than it is to train continuous marginal distributions (like m(z)).We also think that this analysis of the geometry of IB and CEB supports our preference for targeting the MNI point and treating CEB as an objective without hyperparameters.

First, there are only a maximum of 4 points of interest in both the IB and CEB information planes (all 4 are visibile in Figure 2 ): the origin, where there is no information in the representation; the MNI point; the point at (I(Y; Z) = I(X; Y), I(X; Z) = H(X)) (which is an MDL-compatible representation BID17 ); and the point at (I(Y; Z) = 0, I(X; Z) = H(X|Y)) (which would be the optimal decoder for an MNI representation).

These are the only points naturally identified by the dataset -selecting a point on one of the edges between those four points seems to need additional justification.

Second, if you do agree with the MNI criterion, for a given model it is impossible to get any closer to the MNI point than by setting CEB's β = 1, due to the convexity of the pareto-optimal frontier.

Much more useful is making changes to the model, architecture, dataset, etc in order to make smaller.

One possibility in that direction that IB and CEB models offer is inspecting training examples with high rate or residual information to check for label noise, leading to a natural human-in-the-loop model improvement algorithm.

Another is using CEB's residual information as a measure of the quality of the trained model, as mentioned in Appendix C.

In this case, the feasible region for CEB collapses to the line segment I(X; Z|Y) = 0 with 0 ≤ I(Y; Z) ≤ I(X; Y).

Similarly, the corresponding IB feasible region is the diagonal line I(X; Z) = I(Y; Z).

This case happens if we choose as our task to predict images given labels, for example.

We should expect such label-conditional generative models to be particularly easy to train, since the search space is so simple.

Additionally, it is never possible to learn a representation that exceeds the MNI, I(X; Z) ≤ H(X) = I(X; Y).

As an objective function, CEB is independent of the methods used to optimize it.

Here we focus on variational objectives because they are simple, tractable, and well-understood, but any approach to optimize mutual information terms can work, so long as they respect the side of the bounds required by the objective.

There are many approaches in the literature that attempt to optimize mutual information terms in some form, including BID26 BID10 BID21 BID20 BID34 .

It is worth noting that none of those approaches by themselves are compatible with the MNI criterion.

Some of them explicitly maximize I(X; Z X ), while others maximize I(Y; Z X ), but leave I(X; Z X ) unconstrained.

We expect all of these approaches to capture more than the MNI in general.

Because of the properties of Re X/Y , we can consider training algorithms that don't rely on observing validation set performance in order to decide when to lower the learning rate.

The closer we can get Re X/Y to 0 on the training set, the better we expect to generalize to data drawn from the same distribution.

One simple approach to training is to set a high initial learning rate (possibly with reverse annealing of the learning rate BID16 ), and then lower the learning rate after any epoch of training that doesn't result in a new lowest mean residual information on the training data.

This is equivalent to the logic of dev-decay training algorithm of BID45 , but does not require the use of a validation set.

Additionally, since the training set is typically much larger than a validation set would be, the average loss over the epoch is much more stable, so the learning rate is less likely to be lowered spuriously.

The intuition for this algorithm is that Re X/Y directly measures how far from optimal our learned representation is for a given c(y|z X ).

At the end of training Re X/Y indicates that we could improve performance by increasing the capacity of our architecture or Algorithm 1: Training algorithm that lowers the learning rate when the mean Re X/Y of the previous epoch is not less than the lowest Re * X/Y seen so far.

The same idea can be applied to training VIB and deterministic models by tracking that the training loss is always going down.

For the experiments in Section 7, we set the values specified in the Input section.

−3 , min_learning_rate=10 −6 , lowering_scale=1 −

All of the models in our experiments have the same core architecture: A 7×2 Wide Resnet BID47 for the encoder, with a final layer of D = 4 dimensions for the latent representation, followed by a two layer MLP classifier using ELU BID11 activations with a final categorical distribution over the 10 classes.

The stochastic models parameterize the mean and variance of a D = 4 fully covariate multivariate Normal distribution with the output of the encoder.

Samples from that distribution are passed into the classifier MLP.

Apart from that difference, the stochastic models don't differ from Determ during evaluation.

None of the five models uses any form of regularization (e.g., L 1 , L 2 , DropOut BID37 , BatchNorm BID22 ).The VIB models have an additional learned marginal, m(z X ), which is a mixture of 240 D = 4 fully covariate multivariate Normal distributions.

The CEB model instead has the backward encoder, b(z X |y) which is a D = 4 fully covariate multivariate Normal distribution parameterized by a 1 layer MLP mapping the label, Y = y, to the mean and variance.

In order to simplify comparisons, for CEB we additionally train a marginal m(z X ) identical in form to that used by the VIB models.

However, for CEB, m(z X ) is trained using a separate optimizer so that it doesn't impact training of the CEB objective in any way.

Having m(z X ) for both CEB and VIB allows us to compare the rate, R, of each model except Determ.

Any distributional family may be used for the encoder.

Reparameterizable distributions BID24 BID14 are convenient, but it is also possible to use the score function trick BID44 to get a high-variance estimate of the gradient for distributions that have no explicit or implicit reparameterization.

In general, a good choice for b(z|y) is the same distributional family as e(z|x), or a mixture thereof.

These are modeling choices that need to be made by the practitioner, as they depend on the dataset.

In this work, we chose normal distributions because they are easy to work with and will be the common choice for many problems, particularly when parameterized with neural networks, but that choice is incidental rather than fundamental.

Note that we did not use additional regularization on the deterministic model, but all models have a 4 dimensional bottleneck, which is likely to have acted as a strong regularizer for the deterministic model.

Additionally, standard forms of regularization, including stochastic regularization, did not prevent the CW attack from being successful 100% of the time in the original work BID9 .

Nor did regularization cause the deterministic networks in BID48 to avoid memorizing the training set.

Thus, we don't think that our deterministic baseline is disadvantaged on the tasks we considered in Sections 7 and 11.

It is worth noting that the conditions for infinite mutual information given in BID4 do not apply to either CEB or VIB, as they both use stochastic encoders e(z X |x).

In our experiments using continuous representations, we did not encounter mutual information terms that diverged to infinity, although it is possible to make modeling and data choices that make it more likely that there will be numerical instabilities.

This is not a flaw specific to CEB or VIB, however, and we found numerical instability to be almost non-existent across a wide variety of modeling and architectural choices for both variational objectives.

Here we describe a few of the more obvious variants of the CEB objective.

In the above presentation of CEB, we derived the objective for what may be termed "classification" tasks (although there is nothing in the derivation that restricts the form of either X or Y).

However, CEB is fully symmetric, so it is natural to consider the second task defined by our choice of dataset, conditional generation of X given Y = y.

In this case, we can augment our graphical model with a new variable, Z Y , and derive the same CEB objective for that variable: DISPLAYFORM0 In the same manner as above, we can derive variational bounds on H(Z Y |X) and H(X|Z Y ).

In particular, we can variationally bound p(z Y |x) with e(z Y |x).

Additionally, we can bound p(x|z Y ) with a decoder distribution of our choice, d(x|z Y ).Because the decoder is maximizing a lower bound on the mutual information between Z Y and X, it can never memorize X. It is directly limited during training to use exactly H(Y) nats of information from Z Y to decode X. For a mean field decoder, this means that the decoder will only output a canonical member of each class.

For a powerful decoder, such as an autoregressive decoder, it will learn to select a random member of the class.

For discrete Y, this model can trivially be turned into an unconditional generative model by first sampling Y from the training data or using any other appropriate procedure, such as sampling Y uniformly at random.

DISPLAYFORM1 Figure 5: Information diagram for the basic hierarchical CEB model, DISPLAYFORM2

Given the presentation of conditional generation above, it is natural to consider that both c(y|z) and d(x|z) are conditional generative models of Y and X, respectively, and to learn a Z that can handle both tasks.

This can be done easily with the following bidirectional CEB model: DISPLAYFORM0 This corresponds to the following factorization: p(x, y, z X , z Y ) ≡ p(x, y)e(z X |x)b(z Y |y).

The two objectives from above then become the following single objective: DISPLAYFORM1 A natural question is how to ensure that Z X and Z Y are consistent with each other.

Fortunately, that consistency is trivial to encourage by making the natural variational approximations: p(z Y |x) → e(z Y |x) and p(z X |y) → b(z X |y).

The full bidirection variational CEB objective then becomes:min log e(z X |x) − log b(z X |y) − log c(y|z X ) DISPLAYFORM2 At convergence, we learn a unified Z that is consistent with both Z X and Z Y , permitting generation of either output given either input in the trained model, in the same spirit as BID42 , but without any objective function hyperparameter tuning.

Thus far, we have focused on learning a single latent representation (possibly composed of multiple latent variables at the same level).

Here, we consider how to learn a hierarchical model with CEB.Consider the graphical model Z 2 ← Z 1 ← X ↔ Y.

This is the simplest hierarchical supervised representation learning model.

The general form of its information diagram is given in Figure 5 .The key observation for generalizing CEB to hierarchical models is that the target mutual information doesn't change.

By this, we mean that all of the Z i in the hierarchy should cover I(X; Y) at convergence, which means maximizing I(Y; Z i ).

It is reasonable to ask why we would want to train such a model, given that the final set of representations are presumably all effectively identical in terms of information content.

The answer is simple: doing so allows us to train deep models in a principled manner such that all layers of the network are consistent with each other and with the data.

We need to be more careful when considering the residual information terms, though -it is not the case that we want to minimize I(X; Z i |Y), which is not consistent with the graphical model.

Instead, we want to minimize I(Z i−1 ; Z i |Y), defining Z 0 = X.This gives the following simple Hierarchical CEB objective: DISPLAYFORM0 DISPLAYFORM1 Because all of the Z i are targetting Y, this objective is as stable as regular CEB.

Note that if all of the Z i have the same dimensionality, in principle they may all use the same networks for b(z i |Y) and/or c(y|z i ), which may substantially reduce the number of parameters in the model.

All of the individual loss terms in the objective must still appear, of course.

There is no requirement, however, that the Z i have the same latent dimensionality, although doing so may give a unified hiearchical representation.

Many of the richest problems in machine learning vary over time.

In BID7 , the authors define the Predictive Information:

DISPLAYFORM0 This is of course just the mutual information between the past and the future.

However, under an assumption of temporal invariance (any time of fixed length is expected to have the same entropy), they are able to characterize the predictive information, and show that it is a subextensive quantity: lim T →∞ I(T )/T → 0, where I(T ) is the predictive information over a time window of length 2T (T steps of the past predicting T steps into the future).

This concise statement tells us that past observations contain vanishingly small information about the future as the time window increases.

The application of CEB to extracting the predictive information is straightforward.

Given the Markov chain X <t → X ≥t , we learn a representation Z t that optimally covers I(X <t , X ≥t ) in Predictive CEB: DISPLAYFORM1 Note that the model entailed by this objective function does not rely on Z <t when predicting X ≥t .

A single Z t captures all of the information in X <t and is to be used to predict as far forward as is desired.

"Rolling out" Z t to make predictions is a modeling error according to the predictive information.

Also note that, given a dataset of sequences, CEB pred may be extended to a bidirectional model, as in Appendix E.2.

In this case, two representations are learned, Z <t and Z ≥t .

Both representations are for timestep t, the first representing the observations before t, and the second representing the observations from t onwards.

As in the normal bidirectional model, using the same encoder and backwards encoder for both parts of the bidirectional CEB objective ties the two representations together.

Modeling and architectural choices.

As with all of the variants of CEB, whatever entropy remains in the data after capturing the entropy of the mutual information in the representation must be modeled by the decoder.

In this case, a natural modeling choice would be a probalistic RNN with powerful decoders per time-step to be predicted.

However, it is worth noting that such a decoder would need to sample at each future step to decode the subsequent step.

An alternative, if the prediction horizon is short or the predicted data are small, is to decode the entire sequence from Z t in a single, feed-forward network (possibly as a single autoregression over all outputs in some natural sequence).

Given the subextensivity of the predictive information, that may be a reasonable choice in stochastic environments, as the useful prediction window may be small.

Multi-scale sequence learning.

As in WaveNet BID41 , it is natural to consider sequence learning at multiple different temporal scales.

Combining an architecture like time-dilated WaveNet with CEB is as simple as combining CEB pred with CEB hier (Appendix E.3).

In this case, each of the Z i would represent a wider time dilation conditioned on the aggregate Z i−1 .

The advantage of such an objective over that used in WaveNet is avoiding unnecessary memorization of earlier timesteps.

E.5 Unsupervised CEB Pure unsupervised learning is fundamentally an ill-posed problem.

Without knowing what the task is, it is impossible to define an optimal representation directly.

We think that this core issue is what lead the authors of BID6 to prefer barely compressed representations.

But by that line of reasoning, it seems that unsupervised learning devolves to lossless compression -perhaps the correct representation is the one that allows you to answer the question: "

What is the color of the fourth pixel in the second row?"On the other hand, it also seems challenging to put the decision about what information should be kept into objective function hyperparameters, as in the β VAE and penalty VAE objectives.

That work showed that it is possible to constrain the amount of information in the learned representation, but it is unclear how those objective functions keep only the "correct" bits of information for the downstream tasks you might care about.

This is in contrast to all of the preceeding discussion, where the task clearly defines the both the correct amount of information and which bits are likely to be important.

However, unsupervised representation learning is still an interesting problem, even if it is ill-posed.

Our perspective on the importance of defining a task in order to constrain the information in the representation suggests that we can turn the problem into a data modeling problem in which the practitioner who selects the dataset also "models" the likely form of the useful bits in the dataset for the downstream task of interest.

In particular, given a dataset X, we propose selecting a function f (X) →

X that transforms X into a new random variable X .

This defines a paired dataset, P(X, X ), on which we can use CEB as normal.

Note that choosing the identity function for f results in maximal mutual information between X and X (H(X) nats), which will result in a representation that is far from the MNI for normal downstream tasks.

In other words, representations learned by true autoencoders are unlikely to be any better than simply using the raw X.It may seem that we have not proposed anything useful, as the selection of f (.) is unconstrained, and seems much more daunting than selecting β in a β VAE or σ in a penalty VAE.

However, there is a very powerful class of functions that makes this problem much simpler, and that also make it clear using CEB will only select bits from X that are useful.

That class of functions is the noise functions.

Given a dataset X without labels or other targets, and some set of tasks in mind to be solved by a learned representation, we may select a random noise variable U, and function X = f (X, U) that we believe will destroy the irrelevant information in X. We may then add representation variables Z X , Z X to the model, giving the joint distribution p(x, x , u, z X , z X ) ≡ p(x)p(u)p(x | f (x, u))e(z X |x)b(z X |x ).

This joint distribution is represented in Figure 6 .Denoising Autoencoders were originally proposed in BID43 .

In that work, the authors argue informally that reconstruction of corrupted inputs is a desirable property of learned representations.

In this paper's notation, we could describe their proposed objective as min H(X|Z X ), or equivalently min log d(x|z X = f (x, η)) x,η∼p(x)p(θ) .Here we make this idea somewhat more formal through the MNI criterion and the derivation of CEB as the optimal objective for that criterion.

We also note that, practically speaking, we would like to learn a representation that is consistent with uncorrupted inputs as well.

Consequently, we are going to use a bidirectional model.

CEB denoise ≡ min I(X; Z X |X ) − I(X ; Z X ) + I(X ; Z X |X) − I(X; Z X ) (37) ⇒ min −H(Z X |X) + H(Z X |X ) + H(X |Z X ) − H(Z X |X ) + H(Z X |X) + H(X|Z X ) (38) This requires two encoders and two decoders, which may seem expensive, but it permits a consistent learned representation that can be used cleanly for downstream tasks.

Using a single encoder/decoder pair would result in either an encoder that does not work well with uncorrupted inputs, or a decoder that only generates noisy outputs.

If you are only interested in the learned representation and not in generating good reconstructions, the objective simplifies to the first three terms.

In that case, the objective is properly called a Noising CEB Autoencoder, as the model predicts the noisy X from X: CEB noise ≡ min I(X; Z X |X ) − I(X ; Z X ) (39) ⇒ min −H(Z X |X) + H(Z X |X ) + H(X |Z X ) denoising CEB, introduced above.

We present the assumed graphical model in FIG4 .

We give the corresponding Semi-Supervised CEB directly: DISPLAYFORM0 1 Y∈(X,Y) is the indicator function, equal to 1 when a Y is part of the paired data, and equal to 0 otherwise.

In other words, if we have Y = y paired with a given X = x, we can include those terms in the objective.

If we do not have that, we can simply leave them out.

Note that it is straightforward to generalize this to semisupervised learning with two or more observations that are both being learned unsupervisedly, but also have some amount of paired data.

For example, images and natural language, assuming we have a reasonable noise model for unsupervisedly learning natural language.

Here we provide some visualizations of the Fashion MNIST tasks.

In Figure 8 , we show a trained 2D CEB latent representation of Fashion MNIST.

The model learned to locate closely related concepts together, including the cluster of "shirt" classes near the center, and the cluster of "shoe" classes toward the lower right.

In spite of the restriction to 2 dimensions, this model achieves ∼ 92% on the test set.

In FIG6 , the 10,000 test images and their 10,000 adversaries are shown for four of the models.

It is easy to see at a glance that the CEB model organizes all of the adversaries into the "trousers" class, with a crisp devision between the true examples and the adversaries.

In contrast, the two VIB models have adversaries mixed throughout.

However, all three models are clearly preferable to the deterministic model, which has all of the adversaries mixed into the "trousers" class with no ability to distinguish between adversaries and true examples.

@highlight

The Conditional Entropy Bottleneck is an information-theoretic objective function for learning optimal representations.