We present Generative Adversarial Privacy and Fairness (GAPF), a data-driven framework for learning private and fair representations of the data.

GAPF leverages recent advances in adversarial learning to allow a data holder to learn "universal" representations that decouple a set of sensitive attributes from the rest of the dataset.

Under GAPF, finding the optimal decorrelation scheme is formulated as a constrained minimax game between a generative decorrelator and an adversary.

We show that for appropriately chosen adversarial loss functions, GAPF provides privacy guarantees against strong information-theoretic adversaries and enforces demographic parity.

We also evaluate the performance of GAPF on multi-dimensional Gaussian mixture models and real datasets, and show how a designer can certify that representations learned under an adversary with a fixed architecture perform well against more complex adversaries.

The use of deep learning algorithms for data analytics has recently seen unprecedented success for a variety of problems such as image classification, natural language processing, and prediction of consumer behavior, electricity use, political preferences, to name a few.

The success of these algorithms hinges on the availability of large datasets, that often contain sensitive information, and thus, may facilitate learning models that inherit societal biases leading to unintended algorithmic discrimination on legally protected groups such as race or gender.

This, in turn, has led to privacy and fairness concerns and a growing body of research focused on developing representations of the dataset with fairness and/or privacy guarantees.

These techniques predominantly involve designing randomizing schemes, and in recent years, distinct approaches with provable statistical privacy or fairness guarantees have emerged.

In the context of privacy, preserving the utility of published datasets while simultaneously providing provable privacy guarantees is a well-known challenge.

While context-free privacy solutions, such as differential privacy BID10 a; BID7 BID8 , provide strong worst-case privacy guarantees, they often lead to a significant reduction in utility.

In contrast, context-aware privacy solutions, e.g., mutual information privacy BID31 BID5 BID33 BID32 BID3 ), achieve improved privacy-utility tradeoff, but assume that the data holder has access to dataset statistics.

In the context of fairness, machine learning models seek to maximize predictive accuracy.

Fairness concerns arise when models learned from datasets that include patterns of societal bias and discrimination inherit such biases.

Thus, there is a need for actively decorrelating sensitive and non-sensitive data.

In the context of publishing datasets or meaningful representations that can be "universally" used for a variety of learning tasks, modifying the training data is the most appropriate and is the focus of this work.

Fairness can then be achieved by carefully designing objective functions which approximate a specific fairness definition while simultaneously ensuring maximal utility (Zemel et al., 2013; BID6 BID16 .

This, in turn, requires dataset statistics.

Adversarial learning approaches for context-aware privacy and fairness have been studied extensively BID13 BID0 BID30 BID20 BID36 BID4 BID27 Zhang et al., 2018) .

They allow the data curator to cleverly decorrelate the sensitive attributes from the rest of the dataset.

These approaches overcome the lack of statistical knowledge by taking a data-driven approach that leverages recent advancements in generative adversarial networks (GANs) BID17 BID28 .

However, most existing efforts focus on extensive empirical studies without theoretical verification and focus predominantly on providing guarantees for a specific classification task.

This work introduces a general framework for context-aware privacy and fairness that we call generative adversarial privacy and fairness (GAPF) (see FIG0 .

We provide precise connections to information-theoretic privacy and fairness formulations and derive game-theoretically optimal decorrelation schemes to compare against those learned directly from the data.

While our framework can be generalized to learn an arbitrary representation using an encoder-decoder structure, this paper primarily focuses on learning private/fair representations of the data (of the same dimension).Our Contributions.

We list our main contributions below.1.

We introduce GAPF, a framework for creating private/fair representations of data using an adversarially trained conditional generative model.

Unlike existing works, GAPF can create representations that are useful for a variety of classification tasks, without requiring the designer to model these tasks at training time.

We validate this observation via experiments on the GENKI (Whitehill & Movellan, 2012) and HAR BID2 datasets.2.

We show that via the choice of the adversarial loss function, our framework can capture a rich class of statistical and information-theoretic adversaries.

This allows us to compare data-driven approaches directly against strong inferential adversaries (e.g., a maximum a posteriori probability (MAP) adversary with access to dataset statistics).

We also show that by carefully designing the loss functions in the GAPF framework, we can enforce demographic parity.3.

We make precise comparison between data-driven privacy/fairness methods and the minimax game-theoretic GAPF formulation.

For Gaussian mixture data, we derive game-theoretically optimal decorrelation schemes and compare them with those that are directly learned in a datadriven fashion to show that the gap between theory and practice is negligible.

Furthermore, we propose using mutual information estimators to verify that no adversary (regardless of their computational power) can reliably infer the sensitive attribute from the learned representation.

Related work.

In the context of publishing datasets with privacy and utility guarantees, a number of similar approaches have been recently considered.

We briefly review them here.

A detailed literature review is included in Appendix A. DP-based obfuscators for data publishing have been considered in BID18 BID26 .

These novel approaches leverage non-generative minimax filters and deep auto-encoders to allow non-malicious entities to learn some public features from the filtered data, while preventing malicious entities from learning other sensitive features.

However, DP can still incur a significant utility loss since it assumes worst-case dataset statistics.

Our approach models a rich class of randomization-based schemes via a generative model that allows the generative decorrelator to tailor the noise to the dataset.

Our work is closely related to adversarial neural cryptography BID0 , learning censored representations BID13 , privacy preserving image sharing BID30 , privacy-preserving adversarial networks BID36 , and adversarially learning fair representation BID27 in which adversarial learning is used to learn how to protect communications by encryption or hide/remove sensitive information or generate fair representation of the data.

Similar to these problems, our model includes a minimax formulation and uses adversarial neural networks to learn decorrelation schemes that prevent an adversary from inferring the sensitive variable.

However, most of these papers use non-generative auto-encoders to remove sensitive information.

Instead, we use a GANs-like approach to learn decorrelation schemes.

We also go beyond in formulating a game-theoretic setting subject to a distortion constraint which allows us to learn private/fair representation for a variety of learning tasks.

Enforcing the distortion constraint calls for a new training process that relies on the Penalty method or Augmented Lagrangian method presented in Appendix C. We show that our framework captures a rich class of statistical and information-theoretic adversaries by changing the loss function.

We also compare the performance of data-driven privacy/fairness methods and the minimax game-theoretic GAPF.Fair representations using information-theoretic objective functions and constrained optimization have been proposed in BID6 BID16 .

However, both approaches require the knowledge of dataset statistics, which is very difficult to obtain for real datasets.

We overcome the issue of statistical knowledge by taking a data-driven approach, i.e., learning the representation from the data directly via adversarial models.

In contrast to in-processing approaches that modify learning algorithms to ensure fair predictions (e..g, using linear programs in BID11 BID14 or via adversarial learning approach in (Zhang et al., 2018) ), we focus on a pre-processing approach to ensure fairness for a variety of learning tasks.

Using GANs to generate synthetic non-sensitive attributes and labels which ensure fairness while preserving the utility of the data (predicting the label) has been studied in (Xu et al., 2018; BID34 .

Rather than using a conditional-generative model to generate synthetic data, we focus on creating fair/private representations of the original data while preserving the utility of the representations for a variety of learning tasks by learning nonlinear compression and noise adding schemes via a generative adversarial model.

We consider a dataset D with n entries where each entry is denoted as (S, X, Y ) where S ∈ S is the sensitive variable, X ∈ X is the public variable, and Y ∈ Y is the target (non-sensitive) variable (for learning).

Instances of X, S, and Y are denoted by x, s and y, respectively.

We assume that each entry (X, S, Y ) is independent and identically distributed according to P (X, S, Y ).

Notice that we model (X, S, Y ) jointly in the dataset.

However, GAPF does not require the knowledge of Y .Privacy and fairness.

Context-aware notions of privacy model how well an adversary, with access to the public data X, can infer the sensitive features S from the data.

Research on context-aware privacy focus on privacy that capture a range of adversarial capabilities ranging from a belief refining adversary using mutual information to quantify privacy to a guessing adversary using a hard-decision rule.

On the other hand, recent results on fairness in learning applications guarantees that for a specific target variable Y , the prediction of a machine learning model is accurate with respect to (w.r.t.)

Y but unbiased w.r.t.

the sensitive variable S. The three oft-used fairness measures are demographic parity, equalized odds, and equal opportunity.

Demographic parity imposes the strongest fairness requirement via complete independence ofŶ and S, and thus, least favors (for correlated Y and S) utility BID19 .

Equalized odds ensures this independence conditioned on the label Y thereby ensuring equal rates for true and false positives (binary Y ) for all demographics.

Equal opportunity ensures equalized odds for the true positive case alone BID19 .When publishing a useful representation of the data for multiple users with different learning tasks, it is difficult to identify a set of target variables (labels) a priori.

Thus, our decorrelation scheme does not restrict itself to a specific Y .

Formally, we define the decorrelation schemes as a randomized mapping given byX = g(X).

We note that g(·) can more generally depend on both X and S but for the sake of simplicity, we restrict our attention to schemes that only depend on X.Let h be a decision rule used by the adversary to infer the sensitive variable S asŜ = h(g(X)) from the representation g(X).

We allow for hard decision rules under which h(g(X)) is a direct estimate of S and soft decision rules under which h(g(X)) = P h (·|g(X)) is a distribution over S.To quantify the adversary's performance, we use a loss function (h(g(X = x)), S = s) defined for every public-sensitive pair (x, s).

Thus, the adversary's expected loss w.r.t.

X and S is L(h, g) E[ (h(g(X)), S)], where the expectation is taken over P (X, S) and the randomness in g and h.

Intuitively, the generative decorrelator would like to minimize the adversary's ability to learn S reliably from the published representation.

This can be trivially done by releasing anX independent of X. However, such an approach provides no utility for data analysts who want to learn nonsensitive variables fromX. To overcome this issue, we capture the loss incurred by perturbing the original data via a distortion function d(x, x), which measures how far the original data X = x is from the processed dataX =x. Ensuring statistical utility in turn requires constraining the average distortion E[d(g(X), X)] where the expectation is taken over P (X, S) and the randomness in g.

The data holder would like to find a decorrelation scheme g that is both privacy/fairness preserving (in the sense that it is difficult for the adversary to learn S fromX) and utility preserving (in the sense that it does not distort the original data too much).

In contrast, for a fixed decorrelation scheme g, the adversary would like to find a (potentially randomized) function h that minimizes its expected loss, which is equivalent to maximizing the negative of the expected loss.

This leads to a constrained minimax game between the generative decorrelator and the adversary given by DISPLAYFORM0 where the constant D ≥ 0 determines the allowable distortion for the generative decorrelator and the expectation is taken over P (X, S) and the randomness in g and h.

Our GAPF framework places no restrictions on the adversary.

Indeed, different loss functions and decision rules lead to different adversarial models.

In what follows, we consider a general α-loss BID24 .

We show that α-loss can capture various information-theoretic adversaries ranging from a hard-decision adversary under the 0-1 loss function (h(g(X)), s) = I h(g(X)) =s to a soft-decision adversary under the log-loss function (h(g(X)), s) = − log P h (s|g(X)).

Theorem 1.

Under α-loss, the optimal adversary decision rule is a 'α-tilted' conditional distribution P * h (s|g(X)) = P (s|g(X)) Under the hard-decision rules in which the adversary uses a 0-1 loss function, the optimal adversarial strategy simplifies to using a MAP decision rule that maximizes P (s|g(X)).

For a soft-decision adversary under log-loss, the optimal adversarial strategy h * is P (s|g(X)) and the GAPF minimax problem in equation 1 simplifies to min g(·) I(g(X); S) subject to E[d(g(X), X)]

≤ D, where I(g(X); S) is the mutual information (MI) between g(X) and S. Corollary 1.

Using α-loss, we can obtain a continuous interpolation between a hard-decision adversary under 0-1 loss (α → ∞) and a soft-decision adversary under log-loss function (α → 1).

Proposition 1.

Under log-loss, GAPF enforces fairness subject to the distortion constraint.

As the distortion increases, the ensuing fairness guarantee approaches ideal demographic parity.

DISPLAYFORM1 The proofs of Theorem 1 , Corollary 1 and Proposition 1 are presented in Appendix B. Many notions of fairness rely on computing probabilities to ensure independence of sensitive and target variables that are not easy to optimize in a data-driven fashion.

In Proposition 1, we propose log-loss (modeled in practice via cross-entropy) in GAPF as a proxy for enforcing fairness.

Data-driven GAPF.

Thus far, we have focused on a setting where the data holder has access to P (X, S).

When P (X, S) is known, the data holder can simply solve the constrained minimax optimization problem in equation 1 (game-theoretic version of GAPF) to obtain a decorrelation scheme that would perform best against a chosen type of adversary.

In the absence of P (X, S), we propose a data-driven version of GAPF that allows the data holder to learn decorrelation schemes directly from a dataset DISPLAYFORM2 .

Under the data-driven version of GAPF, we represent the decorrelation scheme via a generative model g(X; θ p ) parameterized by θ p .

This generative model takes X as input and outputsX. In the training phase, the data holder learns the optimal parameters θ p by competing against a computational adversary: a classifier modeled by a neural network h(g(X; θ p ); θ a ) parameterized by θ a .

In the evaluation phase, the performance of the learned decorrelation scheme can be tested under a strong adversary that is computationally unbounded and has access to dataset statistics.

We follow this procedure in the next section.

In theory, the functions h and g can be arbitrary.

However, in practice, we need to restrict them to a rich hypothesis class.

FIG0 shows an example of the GAPF model in which the generative decorrelator and adversary are modeled as deep neural networks.

For a fixed h and g, if S is binary, we can quantify the adversary's empirical loss using cross entropy DISPLAYFORM3 It is easy to generalize cross entropy to the multi-class case using the softmax function.

The optimal model parameters are the solutions to min DISPLAYFORM4 where the expectation is over D and the randomness in g.

The minimax optimization in equation 2 is a two-player non-cooperative game between the generative decorrelator and the adversary with strategies θ p and θ a , respectively.

In practice, we can learn the equilibrium of the game using an iterative algorithm (see Algorithm 1 in Appendix C).

We first maximize the negative of the adversary's loss function in the inner loop to compute the parameters of h for a fixed g. Then, we minimize the decorrelator's loss function, which is modeled as the negative of the adversary's loss function, to compute the parameters of g for a fixed h. Observe that the distortion constraint in equation 2 makes our minimax problem different from what is extensively studied in previous works.

To incorporate the distortion constraint, we use the penalty method BID25 to replace the constrained optimization problem by adding a penalty to the objective function.

The penalty consists of a penalty parameter ρ t multiplied by a measure of violation of the constraint at the t th iteration.

The constrained optimization problem of the generative decorrelator can be approximated by a series of unconstrained optimization problems with the loss function DISPLAYFORM5 2 , where ρ t is a penalty coefficient increases with the number of iterations t. The algorithm and the penalty method are detailed in Appendix C.

To demonstrate the performance of the decorrelation schemes learned in a data-driven fashion against a computationally bounded adversary, we evaluate the performance of the learned schemes against a maximum a posteriori probability (MAP) adversary that has access to distributional information and knows the applied decorrelation schemes.

First, we derive game-theoretically optimal decorrelation schemes by considering a MAP adversary.

Second, we compare the game-theoretically optimal scheme against the data-driven decorrelation scheme learned from a synthetic dataset by competing against a computational adversary (modeled by a multi-layer neural network).

To quantify the performance of the learned decorrelation scheme, we compute the accuracy of inferring S under a MAP adversary that has access to both the joint distribution of (X, S) and the decorrelation scheme.

Furthermore, we use mutual information estimator (detailed in Appendix F) to demonstrate that GAPF effectively decorrelates the sensitive variables from the data.

The details of the game-theoretically optimal and data-driven GAPF are included in Appendix D.Game-Theoretical Approach.

We focus on a setting where S ∈ {0, 1} and X is an m-dimensional Gaussian mixture random vector whose mean is dependent on S. Let P (S = 1) = q, X|S = 0 ∼ N (−µ, Σ) and X|S = 1 ∼ N (µ, Σ), where µ = (µ 1 , ..., µ m ).

We assume that X|S = 0 and X|S = 1 have the same covariance Σ. Both the generative decorrelator and the adversary have access to P (X, S).

In order to have a tractable model for the decorrelator, we mainly focus on linear (precisely affine) GAPF schemesX = g(X) = X + Z + β, where Z is a zero-mean multi-dimensional Gaussian random vector.

This linear GAPF enables controlling both the mean and covariance of the privatized data.

Although other distributions can be considered, we choose additive Gaussian noise for tractability reasons.

To quantify utility of the learned representation, we use the 2 distance between X andX to obtain a distortion constraint E X,X X −X 2 ≤ D.Without loss of generality, we assume that β = (β 1 , ..., β m ) is a constant parameter vector and DISPLAYFORM0 , following similar analysis in (Gallager, 2013), we can show that the adversary's probability of detection is given by DISPLAYFORM1 Theorem 2.

Consider GAPF schemes g(X) = X + Z + β, where X and Z are multidimensional Gaussian random vectors with diagonal covariance matrices Σ = diag(σ DISPLAYFORM2 The parameters of the minimax optimal decorrelation scheme are DISPLAYFORM3 The accuracy of the MAP adversary is given by substituting β i * and σ * pi into equation 3.

Numerical Results.

FIG2 illustrates the performance of the learned GAPF scheme against a strong theoretical MAP adversary for a 32-dimensional Gaussian mixture model with P (S = 1) = 0.75 and 0.5.

We observe that the inference accuracy of the MAP adversary decreases as the distortion increases and asymptotically approaches (as expected) the prior on the sensitive variable.

The decorrelation scheme obtained via the data-driven approach performs very well when pitted against the MAP adversary (maximum accuracy difference around 0.7% compared to the theoretical optimal).

Furthermore, the estimated mutual information decreases as the distortion increases.

In other words, for the data generated by Gaussian mixture model with binary sensitive variable, the datadriven version of GAPF can learn decorrelation schemes that perform as well as the decorrelation schemes computed under the theoretical version of GAPF, given that the generative decorrelator has access to the statistics of the dataset.

We apply our GAPF framework to real-world datasets to demonstrate its effectiveness.

The GENKI dataset consists of 1, 740 training and 200 test samples.

Each data sample is a 16 × 16 greyscale face image with varying facial expressions.

Both training and test datasets contain 50% male and 50% female.

Among each gender, we have 50% smile and 50% non-smile faces.

We consider gender as sensitive variable S and the image pixels as public variable X. The HAR dataset consists of 561 features of motion sensor data collected by a smartphone from 30 subjects performing six activities (walking, walking upstairs, walking downstairs, sitting, standing, laying).

We choose subject identity as sensitive variable S and features of motion sensor data as public variable X. The dataset is randomly partitioned into 8, 000 training and 2, 299 test samples.

We train our model based on the data-driven GAPF presented in Section 2 using TensorFlow .

For the GENKI dataset, we consider two different decorrelator architectures: the feedforward neural network decorrelator (FNND) and the transposed convolution neural network decorrelator (TC-NND).

The FNND architecture uses a feedforward multi-layer neural network to combine the lowdimensional random noise (100×1) and the original image together FIG7 .

The TCNND takes a low-dimensional random noise and generates high-dimensional noise using a multi-layer transposed convolution neural network.

The generated high-dimensional noise is added to each pixel of the original image to produce the processed image (Figure 8 ).

For the HAR dataset, we use the FNND architecture modeled by a four-layer feedforward neural network.

The details of the architectures for both the generative decorrelator and the adversary are presented in Appendix E. Table 2 : Error rates for expression classification using representation learned by TCNND

The GENKI Dataset.

Figure 3a illustrates the gender classification accuracy of the adversary for different values of distortion.

It can be seen that the adversary's accuracy of classifying the sensitive variable (gender) decreases progressively as the distortion increases.

Given the same distortion value, FNND achieves lower gender classification accuracy compared to TCNND.

An intuitive explanation is that the FNND uses both the noise vector and the original image to generate the processed image.

However, the TCNND generates the noise mask that is independent of the original image pixels and adds the noise mask to the original image in the final step.

To demonstrate the effectiveness of the learned GAPF schemes, we compare the gender classification accuracy of the learned GAPF schemes with adding uniform or Laplace noise.

Figure 3a shows that for the same distortion, the learned GAPF schemes achieve much lower gender classification accuracies than using uniform or Laplace noise.

Furthermore, the estimated mutual informationÎ(X; S) normalized byÎ(X; S) also decreases as the distortion increases ( Figure 3b ).

To evaluate the influence of GAPF on other non-sensitive variable (Y ) classification tasks, we train another CNN (see Figure 9 ) to perform facial expression classification on datasets processed by different decorrelation schemes.

The trained model is then tested on the original test data.

In Figure 3a, we observe that the expression classification accuracy decreases gradually as the distortion increases.

Even for a large distortion value (5 per image), the expression classification accuracy only decreases by 10%.

Furthermore, the estimated normalized mutual informationÎ(X; Y )/Î(X; Y ) decreases much slower thanÎ(X; S)/Î(X; S) as the distortion increases (Figure 3b ).

Table 1 and 2 present different error rates for the facial expression classifiers trained using data representations created by different decorrelator architectures.

We observe that as distortion increases, the error rates difference for different sensitive groups decrease.

This implies the classifier's decision is less biased to the sensitive variables when trained using the processed data.

When D = 5, the differences are already very small.

Furthermore, we notice that the FNND architecture performs better in enforcing fairness but suffers from higher error rate.

The images processed by FNND is shown in Figure 4 .

The decorrelator changes mostly eyes, nose, mouth, beard, and hair.

The HAR Dataset.

Figure 5a illustrates the activity and identity classification accuracy for different values of distortion.

The adversary's sensitive variable (identity) classification accuracy decreases progressively as the distortion increases.

When the distortion is small (D = 2), the adversary's classification accuracy is already around 27%.

If we increase the distortion to 8, the classification accuracy further decreases to 3.8%.

Figure 5a depicts that even for a large distortion value (D = 8), the activity classification accuracy only decreases by 18% at most.

Furthermore, Figure 5b shows that the estimated normalized mutual information also decreases as the distortion increases.

We have introduced a novel adversarial learning framework for creating private/fair representations of the data with verifiable guarantees.

GAPF allows the data holder to learn the decorrelation scheme directly from the dataset (to be published) without requiring access to dataset statistics.

Under GAPF, finding the optimal decorrelation scheme is formulated as a game between two players: a generative decorrelator and an adversary.

We have shown that for appropriately chosen loss functions, GAPF can provide guarantees against strong information-theoretic adversaries, such as MAP and MI adversaries.

It can also enforce fairness, quantified via demographic parity by using the log-loss function.

We have also validated the performance of GAPF on Gaussian mixture models and real datasets.

There are several fundamental questions that we seek to address.

An immediate one is to develop techniques to rigorously benchmark data-driven results for large datasets against computable theoretical guarantees.

More broadly, it will be interesting to investigate the robustness and convergence speed of the decorrelation schemes learned in a data-driven fashion.

In this paper, we connect our objective function in GAPF with demographic parity.

Since there is no single metric for fairness, this leaves room for designing objective functions that link to other fairness metrics such as equalized odds and equal opportunity.

In the context of publishing datasets with privacy and utility guarantees, a number of similar approaches have been recently considered.

We briefly review them and clarify how our work is different.

DP-based obfuscators for data publishing have been considered in BID18 BID26 .

The author in BID18 considers a deterministic, compressive mapping of the input data with differentially private noise added either before or after the mapping.

The approach in BID26 ) relies on using deep auto-encoders to determine the relevant feature space to add differentially private noise, thereby eliminating the need to add noise to the original data.

These novel approaches leverage minimax filters and deep auto-encoders to allow non-malicious entities to learn some public features from the filtered data, while preventing malicious entities from learning other sensitive features.

Both approaches incorporate a notion of context-aware privacy and achieve better privacy-utility tradeoffs while using DP to enforce privacy.

However, DP can still incur a significant utility loss since it assumes worst-case dataset statistics.

Our approach models a rich class of randomization-based schemes via a generative model that allows the generative decorrelator to tailor the noise to the dataset.

Our work is closely related to adversarial neural cryptography BID0 , learning censored representations BID13 , privacy preserving image sharing BID30 , privacy-preserving adversarial networks BID36 , and adversarially learning fair representation BID27 in which adversarial learning is used to learn how to protect communications by encryption or hide/remove sensitive information or generate fair representation of the data.

Similar to these problems, our model includes a minimax formulation and uses adversarial neural networks to learn decorrelation schemes.

However, in BID13 BID30 BID27 , the authors use non-generative auto-encoders to remove sensitive information.

Instead, we use a GANs-like approach to learn decorrelation schemes that prevent an adversary from inferring the sensitive variable.

Furthermore, these formulations uses weighted combination of different loss functions to balance privacy with utility.

We also go beyond in formulating a game-theoretic setting subject to a distortion constraint.

These approaches are not equivalent because of the non-convexity (resp.

concavity) of the minimax problem with respect to the decorrelator (resp.

adversary) neural network parameters and requires new methods to enforce the distortion constraint during the training process.

The distortion constraint allows us to directly limit the amount of distortion added to learn the private/fair representation for a variety of learning tasks, which is crucial for preserving the utility of the learned representation.

Moreover, we compare the performance of the decorrelation schemes learned in an adversarial fashion with the game-theoretically optimal ones for canonical synthetic data models thereby providing formal verification of decorrelation schemes that are learned by competing against computational adversaries.

Finally, we propose using mutual information as a criterion to certify that the representations we learned adversarially against an attacker with a fixed architecture generalize against unseen attackers with (possibly) more complex architecture.

Fair representations using information-theoretic objective functions and constrained optimization have been proposed in BID6 BID16 .

However, both approaches require the knowledge of dataset statistics, which are very difficult to obtain for real datasets.

We overcome the issue of statistical knowledge by taking a data-driven approach, i.e., learning the representation from the data directly via adversarial models.

In contrast to in-processing approaches that modify learning algorithms to ensure fair predictions (e..g, using linear programs in BID11 BID14 or via adversarial learning approach in (Zhang et al., 2018)), we focus on a pre-processing approach to ensure fairness for a variety of learning tasks.

Generative adversarial networks (GANs) have recently received a lot of attention in the machine learning community BID17 BID28 .

Ultimately, deep generative models hold the promise of discovering and efficiently internalizing the statistics of the target signal to be generated.

Using GANs to generate synthetic non-sensitive attributes and labels which ensure fairness while preserving the utility of the data (predicting the label) has been studied in (Xu et al., 2018; BID34 .

The goal here is to develop a conditional GAN-based model to ensure fairness in the system by learning to generate a fairer synthetic dataset using an unconstrained minimax game with carefully designed loss functions corresponding with both fairness and utility.

The synthetic data is generated by a conditional generative adversarial network (GAN) which generates the non-sensitive attributes-label pair given the noise variable and the sensitive attribute.

The utility is preserved by generating data that is very similar to the original data.

To ensure fairness, the generator generates data samples such that an auxiliary classifier trained to predict the sensitive attribute from the synthetic data performs as poorly as possible.

The methods presented in these papers are very different from our method since we are focusing on creating a fair/private representations of the original data while preserving the utility of the representation for a variety of learning tasks.

There are different ways for enforcing fairness, and our work presents a framework that aids in achieving this goal.

More work is needed to be done in this area.

Our GAPF framework places no restrictions on the adversary.

Indeed, different loss functions and decision rules lead to different adversarial models.

In what follows, we will discuss a variety of loss functions under hard and soft decision rules, and show how our GAPF framework can recover several popular information theoretic privacy notions.

We will also show that we can obtain a continuous interpolation between a hard-decision adversary under 0-1 loss function and a soft-decision adversary under log-loss function using the α-loss function.

Hard Decision Rules.

When the adversary adopts a hard decision rule, h(g(X)) is an estimate of S. Under this setting, we can choose (h(g(X)), S) in a variety of ways.

For instance, if S is continuous, the adversary can attempt to minimize the difference between the estimated and true sensitive variable values.

This can be achieved by considering a squared loss function DISPLAYFORM0 which is known as the 2 loss.

In this case, one can verify that the adversary's optimal decision rule is h * = E[S|g(X)], which is the conditional mean of S given g(X).

Furthermore, under the adversary's optimal decision rule, the minimax problem in equation 1 simplifies to DISPLAYFORM1 mmse(S|g(X)), subject to the distortion constraint.

Here mmse(S|g(X)) is the resulting minimum mean square error (MMSE) under h * = E[S|g(X)].

Thus, under the 2 loss, GAPF provides privacy guarantees against an MMSE adversary.

On the other hand, when S is discrete (e.g., age, gender, political affiliation, etc), the adversary can attempt to maximize its classification accuracy.

This is achieved by considering a 0-1 loss function BID29 given by DISPLAYFORM2 In this case, one can verify that the adversary's optimal decision rule is the maximum a posteriori probability (MAP) decision rule: h * = arg max s∈S P (s|g(X)), with ties broken uniformly at random.

Moreover, under the MAP decision rule, the minimax problem in equation 1 reduces to DISPLAYFORM3 subject to the distortion constraint.

Thus, under a 0-1 loss function, the GAPF formulation provides privacy guarantees against a MAP adversary.

Soft Decision Rules.

Instead of a hard decision rule, we can also consider a broader class of soft decision rules where h(g(X)) is a distribution over S; i.e., h(g(X)) = P h (s|g(X)) for s ∈ S. In this context, we can analyze the performance under a log-loss DISPLAYFORM4 In this case, the objective of the adversary simplifies to DISPLAYFORM5 and that the maximization is attained at P * h (s|g(X)) = P (s|g(X)).

Therefore, the optimal adversarial decision rule is determined by the true conditional distribution P (s|g(X)), which we assume is known to the data holder in the game-theoretic setting.

Thus, under the log-loss function, the minimax optimization problem in equation 1 reduces to DISPLAYFORM6 I(g(X); S) − H(S), subject to the distortion constraint.

Thus, under the log-loss in equation 7, GAPF is equivalent to using MI as the privacy metric BID5 ).The 0-1 loss captures a strong guessing adversary; in contrast, log-loss or information-loss models a belief refining adversary.

Consider the α-loss function BID24 DISPLAYFORM0 for any α > 1.

Denoting H a α (S|g(X)) as the Arimoto conditional entropy of order α, one can verify that DISPLAYFORM1 which is achieved by a 'α-tilted' conditional distribution DISPLAYFORM2 Under this choice of a decision rule, the objective of the minimax optimization in equation 1 reduces to min DISPLAYFORM3 where I a α is the Arimoto mutual information and H α is the Rényi entropy.

For large α (α → ∞), this loss approaches that of the 0-1 (MAP) adversary in the limit.

As α decreases, the convexity of the loss function encourages the estimatorŜ to be probabilistic, as it increasingly rewards correct inferences of lesser and lesser likely outcomes (in contrast to a hard decision rule by a MAP adversary of the most likely outcome) conditioned on the revealed data.

As α → 1, equation 8 yields the logarithmic loss, and the optimal belief PŜ is simply the posterior belief.

Therefore, using α-loss, we can obtain a continuous interpolation between a hard-decision adversary under 0-1 loss (α → ∞) and a soft-decision adversary under log-loss function (α → 1).

Let's consider an arbitrary target variable Y which a user is interested in learning from the data.

The objective of the learning task is to train a good model that takesX to predict Y .

Thus, we have the Markov chain: S → X →X →Ŷ , whereŶ is an estimate of Y from the trained machine learning model.

According to data processing inequality, we have I(S;X) ≥ I(S;Ŷ ).

As we have shown in the above analysis, for the log-loss function, the objective of GAPF is equivalent to minimizing I(S;X) , which is an upperbound on I(S;Ŷ ).

Notice that demographic parity requires S andŶ to be independent, which is equivalent to I(S;Ŷ ) = 0.

Since mutual information is non-negative, GAPF ensures fairness by minimizing an upperbound of I(S;Ŷ ) subject to the distortion constraint under the log-loss function.

As the distortion increases, the ensuing fairness guarantee approaches ideal demographic parity by enforcing I(S;Ŷ ) ≤ I(S;X) = 0.

In this section, we present the alternate minimax algorithm to learn the GAPF scheme from a dataset.

The alternating minimax privacy preserving algorithm is presented in Algorithm 1.

To incorporate where ρ t is a penalty coefficient which increases with the number of iterations t. For convex optiwhere ρ t is a penalty coefficient which increases with the number of iterations t and λ t is updated according to the rule DISPLAYFORM0 .

For convex optimization problems, the solution to the series of unconstrained problems formulated by the augmented Lagrangian method also converges to the solution of the original constrained problem BID12 .

DISPLAYFORM1 Let us considerX = X + Z + β, where β ∈ R and Σ p is a diagonal covariance whose diagonal entries is given by {σ 2 p1 , ..., σ 2 pm }.

Given the MAP adversary's optimal inference accuracy in equation 3, the objective of the decorrelator is to DISPLAYFORM2 DISPLAYFORM3 Note that DISPLAYFORM4 Therefore, the second term in equation 14 is 0.

Furthermore, the first term in equation 14 is always positive.

Thus, DISPLAYFORM5 is monotonically increasing in α.

As a result, the optimization problem in equation 12 is equivalent to DISPLAYFORM6 The objective function in equation 16 can be written as DISPLAYFORM7 Thus, the optimization problem in equation 16 is equivalent to DISPLAYFORM8 Since a non-zero β does not affect the objective function but result in positive distortion, the optimal scheme satisfies β = (0, ..., 0).

Furthermore, the Lagrangian of the above optimization problem is given by DISPLAYFORM9 where λ = {λ 0 , ..., λ m } denotes the Lagrangian multipliers associated with the constraints.

Taking the derivatives of L(σ 2 p1 , ..., σ 2 pm , λ) with respect to σ 2 pi , ∀i ∈ {1, ..., m}, we have DISPLAYFORM10 Notice that the objective function in equation 16 is decreasing in σ 2 pi , ∀i ∈ {1, ..., m}. Thus, the optimal solution σ * pi DISPLAYFORM11 By the KKT conditions, we have DISPLAYFORM12 Since λ * i , i ∈ {0, 1, ..., m} is dual feasible, we have λ * i ≥ 0, i ∈ {0, 1, ..., m}. Therefore DISPLAYFORM13 2 ) 2 .

This implies λ * i > 0.

Thus, by complementary slackness, DISPLAYFORM14 Therefore, σ * pi , the amount of noise added to this dimension is proportional to |µ i |; this is intuitive since a large |µ i | indicates the two conditionally Gaussian distributions are further away on this dimension, and thus, distinguishable.

Thus, more noise needs to be added in order to reduce the MAP adversary's inference accuracy.

DISPLAYFORM15

For the data-driven linear GAPF scheme, we assume the generative decorrelator only has access to the dataset D with n data samples but not the actual distribution of (X, S).

Computing the optimal decorrelation scheme becomes a learning problem.

In the training phase, the data holder learns the parameters of the GAPF scheme by competing against a computational adversary modeled by a multi-layer neural network.

When convergence is reached, we evaluate the performance of the learned scheme by comparing with the one obtained from the game-theoretic approach.

To quantify the performance of the learned GAPF scheme, we compute the accuracy of inferring S under a strong MAP adversary that has access to both the joint distribution of (X, S) and the decorrelation scheme.

Since the sensitive variable S is binary, we measure the training loss of the adversary network by the empirical log-loss function DISPLAYFORM0 For a fixed decorrelator parameter θ p , the adversary learns the optimal θ * a by maximizing equation 22.

For a fixed θ a , the decorrelator learns the optimal θ * p by minimizing −L n (h(g(X; θ p ); θ a ), S) subject to the distortion constraint E X,X X −X 2 ≤ D.As shown in FIG6 , the decorrelator is modeled by a two-layer neural network with parameters θ p = {β 0 , ..., β m , σ p0 , ..., σ pm }, where β k and σ pk represent the mean and standard deviation for each dimension k ∈ {1, ..., m}, respectively.

The random noise Z is drawn from a m-dimensional independent zero-mean standard Gaussian distribution with covariance Σ 1 .

Thus, we haveX k = X k + β k + σ pk Z k .

The adversary, whose goal is to infer S from privatized dataX, is modeled by a three-layer neural network classifier with leaky ReLU activations.

To incorporate the distortion constraint into the learning process, we add a penalty term to the objective of the decorrelator.

Thus, the training loss function of the decorrelator is given by DISPLAYFORM1 where ρ t is a penalty coefficient which increases with the number of iterations t.

The added penalty consists of a penalty parameter ρ multiplied by a measure of violation of the constraint.

This measure of violation is non-zero when the constraint is violated.

Otherwise, it is zero.

We use synthetic data generated by Gaussian mixture model as our first attempt to evaluate the performance of the learned GAPF schemes.

Each dataset contains 20K training samples and 2K test samples.

Each data entry is sampled from an independent multi-dimensional Gaussian mixture model.

We consider two categories of synthetic datasets with P (S = 1) equal to 0.75 and 0.5, respectively.

Both the decorrelator and the adversary in the GAPF framework are trained on Tensorflow using Adam optimizer with a learning rate of 0.005 and a minibatch size of 1000.

The distortion constraint is enforced by the penalty method as detailed in supplement C (see equation 10).E GAPF ARCHITECTURE FOR GENKI AND HAR DATASETS The FNND is modeled by a four-layer feedforward neural network.

We first reshape each image to a vector (256 × 1), and then concatenate it with a 100 × 1 Gaussian random noise vector.

Each entry in the noise vector is sampled independently from a standard Gaussian distribution.

We feed the entire vector to a four-layer fully connected (FC) neural network.

Each layer has 256 neurons with a leaky ReLU activation function.

Finally, we reshape the output of the last layer to a 16 × 16 image.

To model the TCNND, we first generate a 100 × 1 Gaussian random vector and use a linear projection to map the noise vector to a 4 × 4 × 256 feature tensor.

The feature tensor is then fed to an initial transposed convolution layer (DeCONV) with 128 filters (filter size 3 × 3, stride 2) and a ReLU activation, followed by another DeCONV layer with 1 filter (filter size 3 × 3, stride 2) and a tanh activation.

The output of the DeCONV layer is added to the original image to generate the processed data.

For both decorrelators, we add batch normalization BID21 on each hidden layer to prevent covariance shift and help gradients to flow.

We model the adversary using convolutional neural networks (CNNs).

This architecture outperforms most of other models for image classification BID23 .

Figure 9 illustrates the architecture of the adversary.

The processed images are fed to two convolution layers (CONV) whose sizes are 3 × 3 × 32 and 3 × 3 × 64, respectively.

Each convolution layer is followed by ReLU activation and batch normalization.

The output of each convolution layer is fed to a 2 × 2 maxpool layer (POOL) to extract features for classification.

The second maxpool layer is followed by two fully connected layers, each contains 1024 neurons with a batch normalization and a ReLU activation.

Finally, the output of the last fully connected layer is mapped to the output layer, which contains two neurons capturing the belief of the subject being a male or a female.

For the HAR dataset, We first concatenate the original data with a 100 × 1 Gaussian random noise vector.

We then feed the entire 661 × 1 vector to a Feed Forward neural network with three hidden fully connected (FC) layers.

Each hidden layer has 512 neurons with a leaky ReLU activation.

Finally, we use another FC layer with 561 neurons to generate the processed data.

For the adversary, we use a five-layer feedforward neural network.

The hidden layers have 512, 512, 256, and 128 neurons with leaky ReLU activation, respectively.

The output of the last hidden layer is mapped to the output layer, which contains 30 neurons capturing the belief of the subject's identity.

For both decorrelator and adversary, we add a batch normalization after the output of each hidden layer.

Our GAPF framework offers a scalable way to find a (local) equilibrium in the constrained min-max optimization, under certain attacks (e.g. attacks based on a neural network).

Yet the privatized data, through our approach, should be immune to any general attacks and ultimately achieving the goal of decreasing the correlation between the privatized data and the sensitive labels.

Therefore we use the estimated mutual information to certify that the sensitive data indeed is protected via our framework.

We use the nearest k-th neighbor method BID22 to estimate the entropyĤ given bŷ DISPLAYFORM0 where r i is the distance of the i-th samplex i to its k-th nearest neighbor, ψ is the digamma function, DISPLAYFORM1 Γ(1+d/2) in Euclidean norm, and N is the number of samples.

Notice thatX is learned representation and S is the sensitive variable.

Then, we calculate the mutual information usinĝ I(X; S) =Ĥ(X) −Ĥ(X|S) For a binary sensitive variable, we can simplify the empirical MI tô I(X; S) =Ĥ(X) − P (S = 1)Ĥ(X|S = 1) + P (S = 0)Ĥ(X|S = 0) ,where P (S = 1) and P (S = 0) can be approximated by the empirical probability.

One noteworthy difficulty is thatX usually lives in high dimensions (e.g. each image has 256 dimensions in GENKI dataset) which is almost impossible to calculate the empirical entropy based on raw data due to the sample complexity.

Thus, we train a neural network that classifies the sensitive variable from the learned data representation to reduce the dimension of the data.

We choose the layer before the softmax outputs (denoted byX f ) to be the feature embedding that has a much lower dimension than originalX and also captures the information about the sensitive variable.

We usê X f as a surrogate ofX for estimating the entropy.

The resulting approximate MI iŝ DISPLAYFORM2 =Ĥ(X f ) − P (S = 1)Ĥ(X f |S = 1) + P (S = 0)Ĥ(X f |S = 0) .Following the same manner, the MI between the learned representationX and the label Y is approximated byÎ(X f ; Y ), whereX f is the feature embedding that represents a privatized imagê X.For the GENKI dataset, we construct a CNN initialized by two conv blocks, then followed by two fully connected (FC) layers, and lastly ended with two neurons having the softmax activations.

In each conv block, we have a convolution layer consisting of filters with the size equals 3 × 3 and the stride equals 1, a 2 × 2 max-pooling layer with the stride equals 2, and a ReLU activation function.

Those two conv blocks have 32 and 64 filters respectively.

We flatten out the output of second conv block yielding a 256 dimension vector.

The extracted features from the second conv layers is passed through the first FC layer with batch normalization and ReLU activation to get a 8-dimensional vector, followed with the second FC layer to output a 2 dimensional vector that applied with the softmax function.

The aforementioned 8-dimensional vector is the feature embedding vectorX f in our empirical MI estimation.

Estimating mutual information for HAR dataset has a slightly different challenge, as the alphabetic size of values that the sensitive label (i.e. identity) can take is 30.

Thus, it requires at least 30 neurons prior to the output layer of the corresponding classification task.

In fact we pose 128 neurons before the final softmax output layer in order to get a reasonably good classification accuracy.

Using the 128-dimensional vector as our feature embedding to calculate mutual information is almost impossible due to the curse of dimensionality.

Therefore, we apply Principal Component Analysis (PCA), shown in FIG0 , and pick the first 12 components to circumvent this issue.

The resulting 12-dimensional vector is considered to be an approximate feature embedding that encapsulates the major information of the processed data.

@highlight

We present Generative Adversarial Privacy and Fairness (GAPF), a data-driven framework for learning private and fair representations with certified privacy/fairness guarantees

@highlight

This paper uses a GAN model to provide an overview of the related work to Private/Fair Representation Learning (PRL).

@highlight

This paper presents an adversarial-based approach for private and fair representations by learned distortion of data that minimises the dependency on sensitive variables while the degree of distortion is constrained.

@highlight

The authors describe a framework of how to learn a demographic parity representation that can be used to train certain classifiers.