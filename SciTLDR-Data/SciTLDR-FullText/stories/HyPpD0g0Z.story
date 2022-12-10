When training a deep neural network for supervised image classification, one can broadly distinguish between two types of latent features of images that will drive the classification of class Y. Following the notation of Gong et al. (2016), we can divide features broadly into the classes of (i) “core” or “conditionally invariant” features X^ci whose distribution P(X^ci | Y) does not change substantially across domains and (ii) “style” or “orthogonal” features X^orth whose distribution P(X^orth | Y) can change substantially across domains.

These latter orthogonal features would generally include features such as position, rotation, image quality or brightness but also more complex ones like hair color or posture for images of persons.

We try to guard against future adversarial domain shifts by ideally just using the “conditionally invariant” features for classification.

In contrast to previous work, we assume that the domain itself is not observed and hence a latent variable.

We can hence not directly see the distributional change of features across different domains.



We do assume, however, that we can sometimes observe a so-called identifier or ID variable.

We might know, for example, that two images show the same person, with ID referring to the identity of the person.

In data augmentation, we generate several images from the same original image, with ID referring to the relevant original image.

The method requires only a small fraction of images to have an ID variable.



We provide a causal framework for the problem by adding the ID variable to the model of Gong et al. (2016).

However, we are interested in settings where we cannot observe the domain directly and we treat domain as a latent variable.

If two or more samples share the same class and identifier, (Y, ID)=(y,i), then we treat those samples as counterfactuals under different style interventions on the orthogonal or style features.

Using this grouping-by-ID approach, we regularize the network to provide near constant output across samples that share the same ID by penalizing with an appropriate graph Laplacian.

This is shown to substantially improve performance in settings where domains change in terms of image quality, brightness, color changes, and more complex changes such as changes in movement and posture.

We show links to questions of interpretability, fairness and transfer learning.

Deep neural networks (DNNs) have achieved outstanding performance on prediction tasks like visual object and speech recognition BID24 BID15 BID18 .

Issues can arise when the learned representations rely on dependencies that vanish in test distributions (e.g. see BID11 and references therein).

Such domain shifts can be caused by changing conditions, e.g. color, background or location changes arising when deploying the machine learning (ML) system in production.

Predictive performance is then likely to degrade.

For instance, the "Russian tank legend" is an example where the training data was subject to sampling biases that were not replicated in the real world.

Concretely, the story relates how a machine learning system was trained to distinguish between Russian and American tanks from photos.

The accuracy was very high but only due to the fact that all images of Russian tanks were of bad quality while the photos of American tanks were not.

The system learned to discriminate between images of different qualities but would have failed badly in practice BID12 1 .Hidden confounding factors like in the example above between image quality and the origin of the tank give rise to indirect associations.

These are arguably one reason why deep learning requires large sample sizes as large sample sizes tend to ensure that the effect of the confounding factors averages out (although a large sample size is clearly not per se a guarantee that the confounding effect will become weaker).

A large sample size is also required if one is trying to achieve invariance to known factors like translation, point of view, and rotation by using data augmentation.

Another related example where human and artificial cognition deviate strongly are adversarial examplesimperceptibly but intentionally perturbed inputs that are misclassified by a ML model (Szegedy et al., 2014; .

Adversarial examples do not fool humans and in general we only need to see one rotated example of the same object to achieve invariance to rotations in our perception.

Our starting point is the question whether we can in a simple way mimic the human ability to learn desired invariances from a few instances of the same object and whether we can better align the features DNNs exploit with human cognition.

Considerations of fairness and discrimination might be another reason why we are interested in controlling that certain characteristics of the input data are not included in the learned representations and thus have no impact on the resulting decisions BID3 BID21 .

Unfortunately, existing biases in datasets used for training ML algorithms tend to be replicated in the estimated models BID7 .

For instance, in June 2015 Google's photo app tagged two non-white people as "gorillas"-most likely because the training examples for "people" were mainly photos of white persons, making "color" predictive for the class label BID10 BID12 .

A human would not make the same mistake after only seeing one instance of a non-white person.

Addressing the issues outlined above, we propose counterfactual regularization (CORE) to control what latent features an estimator extracts from the input data.

Conceptually, we take a causal view of the data generating process and categorize the latent data generating factors into 'conditionally invariant' (core) and 'orthogonal' (style) features, as in BID14 .

It is desirable that a classifier uses only the core features as they pertain to the target of interest in a stable and coherent fashion.

CORE yields an estimator which is invariant to factors of variation corresponding to style features.

Consequently, it is robust with respect to adversarial domain shifts, arising through arbitrarily strong interventions on the style features.

CORE relies on the fact that for certain datasets we can observe "counterfactuals" in the sense that we observe the same object under different conditions.

Rather than pooling over all examples, CORE exploits knowledge about this grouping, i.e. that a number of instances relate to the same object.

The remainder of this manuscript is structured as follows: §2 starts with two motivating examples, showing how CORE can reduce the need for data augmentation and help predictive performance in small sample size settings.

In §3 we review related work and in §4 we formally introduce counterfactual regularization, along with the CORE estimator and theoretical insights for the logistic regression setting.

In §5 we further evaluate the performance of CORE in a variety of experiments.

The CelebA dataset BID26 contains face images of celebrities.

We consider the task of classifying whether a person wears glasses.

Several photos of the same person are available.

We use this grouping information and constrain the classification to yield the same prediction for all images belonging to the same person and sharing the same class label.

We call the additional instances of the same person counterfactual (CF) observations.

FIG0 shows examples from the training set.

The standard approach would be to pool all examples.

The only additional information we exploit is that some observations can be grouped.

We include n = 10 identities in the training set, resulting in a total sample size m = 321 as there are approximately 30 images of each person 2 .(a) Grouping-by-ID with ID=identity.(b) Grouping-by-ID with ID=original image.

Figure 1: Examples from a) the subsampled CelebA dataset and b) the augmented MNIST dataset.

Connected images are counterfactual examples as they share the same realization of the ID which is the identity of the person in a) and the original image used for data augmentation in b).

The comparison is a training of exactly the same network architecture that does not make use of the grouping information but using a standard ridge penalty.

In a) exploiting the grouping information reduces the test error by 32% compared to pooling over all samples.

In b) the test error on rotated digits is reduced by 50%.Exploiting the group structure reduces the average test error from 24.76% to 16.89%, i.e. by approx.

32%, compared to the estimator which just pools all images and uses a standard ridge penalty for the cofficients 3 .

A different use case of CORE is to make data augmentation more efficient in terms of the required samples.

In data augmentation, one creates additional samples by modifying the original inputs, e.g. by rotating, translating, or flipping the images (Schölkopf et al., 1996) .

In other words, additional samples are generated by interventions on style features.

Using this augmented data set for training results in invariance of the estimator with respect to the transformations (style features) of interest.

For CORE we can use the grouping information that the original and the augmented samples belong to the same object.

This enforces the invariance with respect to the style features more strongly compared to normal data augmentation which just pools all samples.

We assess this for the style feature "rotation" on MNIST BID25 ) and only include c = 100 augmented training examples for n = 10000 original samples, resulting in a total sample size of m = 10100.

The degree of the rotations is sampled uniformly at random from [35, 70] .

FIG0 shows examples from the training set.

By using CORE the average test error on rotated examples is reduced from 32.86% to 16.33%, around half its original value 4 .

Perhaps most similar to this work in terms of their goals are the work of BID14 and Domain-Adversarial Neural Networks (DANN) proposed in BID13 , an approach motivated by the work of BID5 .

While our approach requires grouped observations, both of these works rely on unlabeled data from the target task being available.

The main idea of BID13 is to learn a representation that contains no discriminative information about the origin of the input (source or target domain).

This is achieved by an adversarial training procedure: the loss on domain classification is maximized while the loss of the target prediction task is minimized simultaneously.

In contrast, we do not assume that we have data from different domains but just different realizations of the same object under different interventions.

The data generating process assumed in BID14 is similar to our model, introduced in §4.2 where we detail the similarities and differences between the models (cf.

FIG0 .

BID14 identify the conditionally independent features by adjusting a transformation of the variables to minimize the squared MMD distance between distributions in different domains 5 .

The fundamental difference to our approach is that we use a different data basis.

The domain identifier is explicitly observable in BID14 , while it is latent in our approach.

In contrast, we exploit presence of an identifier variable ID to penalize the classifier using any latent features outside the set of conditionally independent features.

Causal modeling has related aims to the setting of transfer learning and guarding against adversarial domain shifts.

Specifically, causal models have the defining advantage that the predictions will be valid even under arbitrarily large interventions on all predictor variables BID17 BID1 BID34 Schölkopf et al., 2012; BID35 Zhang et al., 2013; X. Yu, 2017; BID21 Magliacane et al., 2017) .

There are two difficulties in transferring these results to the setting of adversarial domain changes in image classification.

The first hurdle is that the classification task is typically anti-causal since the image we use as a predictor is a descendant of the true class of the object we are interested in rather than the other way around.

The second challenge is that we do not want to guard against arbitrary interventions on any or all variables but only would like to guard against a shift of the style features.

It is hence not immediately obvious how standard causal inference can be used to guard against large domain shifts.

Recently, various approaches have been proposed that leverage causal motivations for deep learning or use deep learning for causal inference.

In all of the following methods, the goals and the settings are different from ours.

Specifically, the setting of anti-causal prediction and non-ancestral interventions on style variables is not considered.

Various approaches focus on cause-effect inference where the goal is to find the causal relation between two random variables, X and Y BID27 BID16 .

propose the Neural Causation Coefficient (NCC) to estimate the probability of X causing Y and apply it to finding the causal relations between image features.

Specifically, the NCC is used to distinguish between features of objects and features of the objects' contexts.

BID27 note the similarity between structural equation modeling and CGANs BID33 .

One CGAN is fitted in the direction X → Y and another one is fitted for Y → X. Based on a two-sample test statistic, the estimated causal direction is returned.

BID16 use generative neural networks for cause-effect inference, to identify v-structures and to orient the edges of a given graph skeleton.

Bahadori et al. (2017) devise a regularizer that combines an 1 penalty with weights corresponding to the estimated probability of the respective feature being causal for the target.

The latter estimates are obtained by causality detection networks or scores such as estimated by the NCC.

Besserve et al. (2017) draw connections between GANs and causal generative models, using a group theoretic framework.

Kocaoglu et al. (2017) propose causal implicit generative models to sample from conditional as well as interventional distributions, using a conditional GAN architecture (CausalGAN).

The generator structure needs to inherit its neural connections from the causal graph, i.e. the causal graph structure must be known.

BID29 propose the use of deep latent variable models and proxy variables to estimate individual treatment effects.

BID21 exploit causal reasoning to characterize fairness considerations in machine learning.

Distinguishing between the protected attribute and its proxies, they derive causal nondiscrimination criteria.

The resulting algorithms avoiding proxy discrimination require classifiers to be constant as a function of the proxy variables in the causal graph, thereby bearing some structural similarity to our style features.

Distinguishing between core and style features can be seen as some form of disentangling factors of variation.

Estimating disentangled factors of variation has gathered a lot of interested in the context of generative modeling BID20 BID9 Bouchacourt et al., 2017) .

For example, Matsuo et al. (2017) propose a "Transform Invariant Autoencoder" where the goal is to reduce the dependence of the latent representation on a specified transform of the object in the original image.

Specifically, Matsuo et al. (2017) predefine location as the orthogonal style feature X ⊥ and the goal is to learn a latent representation that does not include X ⊥ .

Here, we do not predefine which features are in X ⊥ .

It could be location but also image quality, posture, brightness, background and contextual information.

Additionally, the approach in Matsuo et al. (2017) DISPLAYFORM0 Figure 2: Left: data generating process for the considered model as in BID14 , where the effect of the domain on the orthogonal features X ⊥ is mediated via unobserved noise ∆. Right: our setting.

The domain itself is unobserved but we can now observe the ID variable we use for grouping.with a confounding situation where the distribution of the style features differs conditional on the class (this is a natural restriction as the class label is not even observed in the autoencoder setting).

As in CORE, Bouchacourt et al. (2017) exploit grouped observations.

In a variational autoencoder framework, they aim to separate style and content-they assume that samples within a group share a common but unknown value for one of the factors of variation while the style can differ.

Here we try to solve a classification task directly without estimating the latent factors explicitly as in a generative framework.

We first describe the standard notation for classification before developing a causal graph that allows us to compare the setting of adversarial domain shifts to transfer learning, domain adaptation and adversarial examples.

Let Y ∈ Y be a target of interest.

Typically Y = R for regression or Y = {1, . . .

, K} in classification with K classes.

Let X ∈ R p be a predictor, for example the p pixels of an image.

The prediction y for y, given X = x, is of the form f θ (x) for a suitable function f θ with parameters θ ∈ R d , where the parameters θ correspond to the weights in a DNN.

For regression, f θ (x) ∈ R, whereas for classification f θ (x) corresponds to the conditional probability distribution of Y ∈ {1, . . . , K}. Let be a suitable loss that maps y andŷ = f θ (x) to R + .

A standard goal is to minimize the expected loss or risk DISPLAYFORM0 Let (x i , y i ) for i = 1, . . .

, n be the samples that constitute the training data andŷ i = f θ (x i ) the prediction for y i .

A standard approach to parameter estimation is penalized empirical risk minimization, where we choose the weights or parameters asθ = argmin θ L n (θ), with the empirical loss given by DISPLAYFORM1 , where the penalty pen(θ) could be a ridge penalty or penalties that exploit underlying geometries such as the Laplacian regularized least squares BID4 .

The full structural model for all variables is shown in the right panel of FIG0 .

The domain variable D is latent, in contrast to BID14 .

We add the ID variable (identity of a person, for example), whose distribution can change conditional on class Y .

The ID variable is used to group observations, see Section 4.4, and can be assumed to be latent in the setting of BID14 .The rest of the graph is in analogy to BID14 .

The prediction is anti-causal, that is the predictors X that we use forŶ are non-ancestral to Y .

In other words, the class label is causal for the image and not the other way around.

The causal effect from the class label Y on the image X is mediated via two types of latent variables: the so-called core or 'conditionally invariant' features X ci and the orthogonal or style features X ⊥ .

The distinguishing factor between the two is that external interventions ∆ are possible on the style features but not on the core features.

If the interventions ∆ have different distributions in different domains, then the distribution P (X ci |Y ) is constant across domains while P (X ⊥ |Y ) can change across domains.

The style features X ⊥ and Y are confounded, in other words, by the latent domain D. In contrast, the core or 'conditionally invariant' features satisfy X ci ⊥ ⊥ D|Y .

The dimension of X ci is chosen maximally large such that this conditional independence is still true.

The style variable can include point of view, image quality, resolution, rotations, color changes, body posture, movement etc.

and will in general be context-dependent 6 .

The style intervention variable ∆ influences both the latent style X ⊥ , and hence also the image X. In potential outcome notation, we let X ⊥ (∆ = δ) be the style under intervention ∆ = δ, X(Y, ID, ∆ = δ) the image for class Y , identity ID and style intervention ∆ and this sometimes abbreviated as X(∆ = δ) for notational simplicity.

Finally, f θ (X(∆ = δ)) is the prediction under the style intervention ∆ = δ.

For a formal justification of using a causal graph and potential outcome notation simultaneously see BID36 .

In this work, we are interested in guarding against adversarial domain shifts.

We use the causal graph to explain the related but not identical goals of domain adaptation, transfer learning and guarding against adversarial examples.

FORMULA3 and can also be described by the causal graph above by using X ⊥ (∆) = ∆ and identifying X ⊥ with pixel-by-pixel additive effects.

The magnitude of the intervention ∆ is then typically assumed to be within an -ball in q -norm around the origin, with q = ∞ or q = 2 for example.

If the input dimension is large many imperceptible changes in the coordinates of X can cause a large change in the output, leading to a misclassification of the sample.

The goal is to devise a classification in this graph that minimizes the adversarial loss E max DISPLAYFORM0 where X(∆) is the image under the intervention ∆ andŶ = f θ (X(∆)) is the estimated conditional distribution of Y , given the image under the chosen interventions. (iii) Adversarial domain shifts.

Here we are interested in arbitrarily strong interventions ∆ ∈ R q on the style features X ⊥ , which are not known explicitly in general.

Analogously to (1), the adversarial loss under arbitrarily large style interventions is DISPLAYFORM1 In contrast to (1) the interventions can be arbitrarily strong but we assume that the style features X ⊥ can only change certain aspects of the image, while other aspects of the image (mediated by the core features) cannot be changed.

In contrast to BID13 , we use the term "adversarial" to refer to adversarial interventions on the style features, while the notion of "adversarial" in domain adversarial neural networks describes the training procedure.

Nevertheless, the motivation of BID13 is equivalent to ours-that is, to protect against shifts in the distribution(s) of test data which we characterize by distinguishing between core and style features.

The classical problem of causal inference is that we can never observe a counterfactual.

For instance, we can only see the health outcome Z if we take a medicine, T = 1, or not, T = 0, but we can never see both health outcomes simultaneously.

The counterfactual in this context would be an observation where we change the treatment but hold all observed and unobserved confounders constant.

If the treatment T changes while all other variables are kept constant, we could just read off the treatment effect as Z(T = 1) − Z(T = 0) if Z is the health outcome of interest.

Observing such counterfactuals is in general impossible as we can either observe the outcome under treatment or under no treatment but not both.

Here, we use the term counterfactual for a situation where we keep class label Y and ID constant but allow the value of the style intervention ∆ to change.

The new value of ∆ could be a do-intervention (as when explicitly rotating an image in data augmentation) or it could be a noise-intervention by sampling a new realization of ∆. The style intervention ∆ takes the same role as the treatment T in the previous medical example.

In contrast to the medical example, however, counterfactuals are conceivable for image analysis as we can see the same object (Y, ID) under different conditions ('treatments') ∆.As an example, if Y is the binary variable whether a person wears glasses and ID is the identity of a person, then ∆ corresponds to all other variables that determine the different images of the same person (either consistently wearing glasses or not) and includes background, posture, viewing angle, image quality, etc.

In further contrast to the medical setting, we are not interested primarily in the 'treatment effect' of the style intervention ∆ but we merely use it to implicitly rule out parts of the feature space for classification.

We know that any 'treatment effect' of ∆ occurs in the space of the style or orthogonal features X ⊥ and not in the 'conditionally invariant' space X ci and we would thus like to penalize any change in the classification under different style interventions ∆ but constant class and identity (Y, ID).Notationally, we have for sample i ∈ {1, . . .

, n} with class label and identifier DISPLAYFORM0 the total number of samples and c = m − n, the number of counterfactual observations.

Denote the j-th observation of sample i, by x i,j ∈ R p .

Typically m i = 1 for most samples and occasionally m i ≥ 2.

The standard approach is to simply pool over all available observations, ignoring any grouping information that might be available.

The pooled estimator thus treats all examples identically by summing over the loss aŝ DISPLAYFORM0 where pen(θ) could be a ridge penalty.

The pooled estimator in all examples is always the ridge estimator with a cross-validated choice of the penalty parameter.

The adversarial loss of the pooled estimator will in general be infinite; see §4.6 for a concrete example.

Using FIG0 , one can show that the pooled estimator will work well in terms of the adversarial loss DISPLAYFORM1 The first condition (i) implies that if the estimator learns to extract X ci from the image X, there is no further information in X that explains Y and, therefore, the direction corresponding to X ⊥ is not required for predicting Y .

The second condition (ii) is fulfilled if the relations between Y , X ci , and X ⊥ are not deterministic.

Intuitively, it ensures that X ⊥ cannot replace X ci in the first condition.

From (i) and (ii), we see that the pooled estimator will work well in terms of the adversarial loss L adv if (a) the edge from X ⊥ to X is absent or if (b) both the edge from D to X ⊥ and the edge from Y to X ⊥ are absent (cf.

FIG0 ).

In order to minimize the adversarial loss (2) we have to ensure f θ (x(∆)) is as constant as possible as a function of ∆ for all x ∈ R p .

Let I be the invariant parameter space DISPLAYFORM0 is a function of the core features x ci ∈ R p only.}.For all θ ∈ I, the adversarial loss FORMULA4 is identical to the loss under no interventions at all.

More precisely, let X be a shorthand notation for X(∆ = 0), the images in absence of external interventions: DISPLAYFORM1 The optimal predictor in the invariant space I is DISPLAYFORM2 If f θ is only a function of the core features X ci , then θ ∈ I. The challenge is that the core features are not directly observable and we have to infer the invariant space I from data.

To get an approximation to the optimal invariant parameter vector (3), we use empirical risk minimization: DISPLAYFORM3 where the first part is the empirical version of the expectation in (3).

The unknown invariant parameters space I is approximated by an empirically invariant space I n , defined as DISPLAYFORM4 where σ 2 i (θ) is the variance of f θ (x i,j ) when varying j = 1, . . .

, m i for a fixed value of i and τ ≥ 0 is a regularization constant.

Setting τ = 0 is equivalent to demanding that the estimated predictions for the class labels are identical across all m i counterfactuals of image i, while slightly larger values of τ allow for some small degree of variations.

For all values τ ≥ 0 the true invariant space I is a subset of the empirically invariant subspace I n , that is I ⊆

I n .

Under the right assumptions we get I n = I for n → ∞. We return to this question in §4.6.

One can equally use the Lagrangian form of the constrained optimization in (4), with a penalty parameter λ instead of a constraint τ to get DISPLAYFORM5 where DISPLAYFORM6 The matrix L ID is a graph Laplacian BID4 , where the underlying graph has n connectivity components as all samples that have the same ID are connected by an edge and form fully connected connectivity components.

The graph Laplacian regularization is identical to penalizing the sum over the variances σ 2 i (θ).

The graph for the underlying regularization is formed in the sample space and induced by the identifier variable ID, in contrast to graphs formed in feature space as in Sandler et al. (2009) , where prior knowledge is used to form the graph by connecting features that share similar characteristics.

We show in §C.1 that the outcome does not depend strongly on the chosen value of the penalty λ and the experiments show that it is crucial to define the graph in terms of the identifier variable ID.

Other regularizations do not perform nearly as well when trying to guard against adversarial domain shifts.

In §A we analyze the adversarial loss, defined in Eq. (2), for the pooled and the CORE estimator in a one-layer network for binary classification (logistic regression).

Here, we briefly sketch the result while all details are given in §A. Assume the structural equation for the image X ∈ R p is linear in the style features X ⊥ ∈ R q (with generally p q), the interventions are additive and we use logistic regression to predict a class label Y ∈ {−1, 1}. Under suitable assumptions (cf.

Assumption 1), the pooled estimator has infinite adversarial loss while the adversarial loss of the CORE estimator converges to the optimal adversarial loss as n → ∞.

We perform an array of different experiments: in §5.1 and §5.2 we study how CORE can handle confounded training data sets and changing style features in test distributions.

For the assessment we explicitly control the level of confounding.

In §5.3, we consider classifying elephants and horses where X ⊥ ≡ color.

In §B, we include two additional experiments: in the first one, Y ≡ gender and X ⊥ ≡ wearing glasses; in the second one, Y ≡ wearing glasses and X ⊥ ≡ brightness.

Additional experimental results for the settings introduced in §2 can be found in §C.2 and §C.3.

A TensorFlow BID0 implementation of CORE will be made available as well as further code necessary to reproduce the experiments.

In addition to the details provided below, information on the employed architectures can be found in §C.7.

An open question is how to set the value of the tuning parameter τ or the penalty λ in Lagrangian form.

We show in §C.1 that performance is typically not very sensitive to the choice of λ.

In this example we consider synthetically generated stickmen images (cf.

FIG0 .

The target of interest is Y ∈ {adult, child} and X ci ≡ height.

The class Y is causal for height and height cannot be easily intervened on, so we consider it to be a core feature-it is a robust predictor for differentiating between children and adults.

Additionally, there is a dependence between age and X ⊥ ≡ movement in the training dataset which arises through the hidden common cause D ≡ place of observation.

The data generating process is illustrated in FIG0 .9.

For instance, the images of children might mostly show children playing while the images of adults typically show them in more "static" postures.

If the learned model exploits this dependence for predicting Y , it will fail when presented images of, say, dancing adults.

FIG0 shows examples from the training set where large movements are associated with children and small movements are associated with adults.

Test set 1 follows the same distribution.

In test sets 2 and 3 X ⊥ is intervened on such that the edge from D to X ⊥ is removed and the dependence between Y and X ⊥ vanishes.

In test sets 2 and 3 large movements are associated with both children and adults, while the movements are heavier in test set 3 than in test set 2.

FIG0 .10 shows examples from all test sets.

FIG0 shows misclassification rates for CORE and the pooled estimator for c = 50 with a total sample size of m = 20000.

For as few as 50 counterfactual observations, CORE succeeds in achieving good predictive performance on test sets 2 and 3 where the pooled estimator fails (test errors > 40%).

These results suggest that the learned representation of the pooled estimator uses movement as a predictor for age while CORE does not use this feature due to the counterfactual regularization.

Importantly, including more counterfactual examples would not improve the performance of the pooled estimator as these would be subject to the same bias and hence also predominantly have examples of heavily moving children and "static" adults (also see FIG0 .10 which shows results for c ∈ {20, 500, 2000}).

As in §2.1, we use the CelebA dataset and consider the problem of classifying whether the person in the image is wearing eyeglasses.

Here, X⊥ is the quality of the image which differs conditional on Y 7 -if the image shows a person wearing glasses, the image quality tends to be lower.

This setting mimics the confounding that occurred in the Russian tank legend (cf.

§1).

The strength of the image quality intervention is governed by sampling the new image quality as a percentage of the original image's quality from a Gaussian distribution N (µ = 30, σ = 10).

Images of people without glasses are not changed.

Thus, we only have counterfactual observations for Y ≡ glasses.

FIG0 shows examples from the training set.

Here, we use as the counterfactual observation the same image but with a newly sampled image quality value from N (30, 10).

We call using the same image as a counterfactual "CF setting 1".

Two alternatives for constructing counterfactual observations for this setting are discussed in §B.2.1.

Here, c = 5000 and m = 20000.

FIG0 shows misclassification rates for CORE and the pooled estimator on different test sets.

Examples from all test sets can be found in FIG0 .11.

Test set 1 follows the same distribution as the training set.

In test set 2 the class of the quality intervention is reversed, i.e. the quality of images showing people without glasses tends to be lower.

In test set 3 all images are left unchanged and in test set 4 the quality of all images is decreased.

First, we notice that the pooled estimator performs better than CORE on test set 1.

This can be explained by the fact that it can exploit the predictive information contained in an image's quality while CORE is restricted not to do so.

Second, we observe that the pooled estimator does not perform well on test sets 2-4 as its learned representation seems to use the image's quality as a predictor for the target.

In contrast, the predictive performance of CORE is hardly affected by the changing image quality distributions.

More experimental details are provided in §C.5.

Results for quality interventions of different strengths (µ ∈ {30, 40, 50}) are shown in FIG0 .12.

In this example, we want to assess whether invariance with respect to X ⊥ ≡ color can be achieved.

In the children's book "Elmer the elephant" 8 one instance of a colored elephant suffices to recognize it as being an elephant, making the color "gray" no longer an integral part of the object "elephant".

Motivated by this process of concept formation, we would like to assess whether CORE can exclude "color" from its learned representation by including a few counterfactuals of different color.

We work with the "Animals with attributes 2" (AwA2) dataset (Xian et al., 2017) and consider classifying images of horses and elephants.

The data generating process is illustrated in FIG0 .14.

We include counterfactual examples by adding grayscale images for c = 250 images of elephants, i.e. counterfactuals are only available for one class and the shift in color is quite subtle.

The total sample size is 1850.

FIG0 shows examples from the training set and FIG0 shows misclassification rates for CORE and the pooled estimator on different test sets.

Examples from all test sets can be found in FIG0 .13.

Test set 1 contains original, colored images only.

In test set 2 images of horses are in grayscale and the colorspace of elephant images is modified, effectively changing the color gray to red-brown.

Test set 3 contains grayscale images only and in test set 4 the colorspace of all images is shifted towards red.

The details are given in §C.6 .

We observe that the pooled estimator does not perform well on test sets 2 and 3 as its learned representation seems to exploit the fact that "gray" is predictive for the target in the training set.

Using this information helps its predictive accuracy on test set 1.

In contrast, the predictive performance of CORE is hardly affected by the changing color distributions.

It is noteworthy that a colored elephant can be recognized as an elephant by adding a few examples of a grayscale elephant to the very lightly colored pictures of natural elephants.

If we just pool over these examples, there is still a strong bias that elephants are gray.

The CORE estimator, in contrast, demands invariance of the prediction for instances of the same elephant and we can learn color invariance with a few added grayscale images.

While a thorough analysis in terms of fairness considerations is beyond the scope of this work, we would like to draw the following connection.

If "color" was a protected attribute or a proxy for one, CORE would satisfy fairness in the sense that it would not include it in its learned representation.

In contrast, there is no way to avoid that the pooled estimator extracts and uses "color" for its decisions.

Distinguishing the latent features in an image into core and style features, we have proposed counterfactual regularization (CORE) to achieve robustness with respect to arbitrarily large interventions on the style or conditionally invariant features.

The main idea of the CORE estimator is to exploit the fact that we often have instances of the same object in the training data.

By demanding invariance of the classifier amongst a group of instances that relate to the same object, we can achieve invariance of the classification performance with respect to adversarial interventions on style features such as image quality, fashion type, color, or body posture.

The training also works despite sampling biases in the data.

There are two main applications areas.

If the style features are known explicitly, we can achieve the same classification performance as standard data augmentation approaches but using fewer instances which, on top, do not have to be carefully balanced in the training data.

Perhaps more interestingly, if the style features are unknown, the regularization of CORE avoids usage of them automatically by penalizing features that vary strongly between different instances of the same object in the training data.

An interesting line of work would be to use larger models such as Inception or large ResNet architectures BID15 BID19 .

These models have been trained to be invariant to an array of explicitly defined style features.

In §B.1 we include results which show that using Inception V3 features does not guard against interventions on more implicit style features.

We would thus like to assess what benefits CORE can bring for training Inception-style models end-to-end, both in terms of sample efficiency and in terms of generalization performance.

While we showed some examples where the necessary grouping information is available, an interesting possible future direction would be to use video data since objects display temporal constancy and the temporal information can hence be used for grouping and counterfactual regularization.

Potentially an analogous approach could also help to debias word embeddings.

Assume the structural equation for the image X ∈ R p is linear in the style features X ⊥ ∈ R q (with generally p q) and we use logistic regression to predict a class label Y ∈ {−1, 1}.

Let the interventions ∆ ∈ R q act additively on the style features X ⊥ (this is only for notational convenience) and let the style features X ⊥ act in a linear way on the image X via a matrix W ∈ R p×q (this is an important assumption without which results are more involved).

The core or 'conditionally invariant' features are X ci ∈ R r , where in general r ≤ p but this is not important for the following.

For independent ε Y , ε ID , ε X ⊥ , ε X in R, R q , R r , R p respectively with positive density on their support and continuously differentiable functions DISPLAYFORM0 Of these, Y , X and ID are observed whereas D, X ci , ∆, X ⊥ and the noise variables are latent.

We assume a logistic regression as a prediction of Y from the image data X: DISPLAYFORM1 Given training data with m samples, we estimate θ withθ and use here a logistic loss θ (y i , x i ) = log(1 + exp(−y i (x t i θ))) for training and testing.

Some interesting expected losses on test data include DISPLAYFORM2 where the X in the first loss is a shorthand notation for X(∆ = 0), that is the images in absence of interventions on the style variables.

The first loss is thus a standard logistic loss in absence of adversarial interventions.

The second loss is the loss under adversarial style or domain interventions as we allow arbitrarily large interventions on X ⊥ here.

The corresponding benchmarks are DISPLAYFORM3 The formulation of Theorem 1 relies on the following assumptions.

Assumption 1.

We require the following conditions to hold:(A1) Assume ∆ is sampled from a distribution for training data in R q with positive density on an -ball in 2 -norm around the origin for some > 0.(A2) Assume the matrix W has full rank q.(A3) Assume c ≥ q, that is the number c = m − n of counterfactual examples in the samples is at least as large as the dimension of the style variables.

Regarding (A3): the sampling process is as follows.

We collect n independent samples (y i , id i , δ i,1 ) from a distribution of (Y, ID, ∆) that satisfies the constraints above.

Then, for c = m − n of the samples we select each time i ∈ {1, . . .

, n} at random, keep (y i , id i ) fixed (and hence also the realization of X ⊥ is fixed) and redraw a new value of ∆ as δ i,ui+1 if u i is the current number of counterfactual examples for sample i.

This leads to m samples in total with in general n distinct values of (y i , id i ) and m i counterfactuals at each sample with corresponding x i,j with i ∈ {1, . . .

, n} and j ∈ {1, . . .

, m i }.

Theorem 1.

Under Assumption 1, with probability 1 with respect to the training data, the pooled estimator has infinite adversarial loss DISPLAYFORM4 For the CORE estimator, for n → ∞, DISPLAYFORM5 An equivalent results can be derived for misclassification loss instead of logistic loss (with infinity replaced by 1).Proof.

First part.

To show the first part, namely that with probability 1, DISPLAYFORM6 we need to show that W tθpool = 0 with probability 1.

The reason this is sufficient is as follows: if DISPLAYFORM7 To show that W tθpool = 0 with probability 1, letθ * be the oracle estimator that is constrained to be orthogonal to the column space of W : DISPLAYFORM8 We show W tθpool = 0 by contradiction.

Assume hence that W tθpool = 0.

If this is indeed the case, then the constraint W t θ = 0 in (6) becomes non-active and we haveθ pool =θ * .

This would imply that taking the directional derivative of the training loss with respect to any δ ∈ R p in the column space of W should vanish at the solutionθ * .

Define r i (θ) := (y i + 1)/2 − fθ * .

For all i = 1, . . .

, n we have r i = 0.

The derivative g(δ) of L n (θ) in direction of δ is proportional to DISPLAYFORM9 x i,j ∈ R p is the j-th counterfactual for training sample i (with j ∈ {1, . . .

, m i }).

Let x i,j (0) = x i,1 (0) for i = 1, . . .

, n be the counterfactual training data in absence of any interventions (∆ i,j = 0).

Since the interventions only have an effect on the column space of W in X, the oracle estimator θ * is identical under the true training data and the counterfactual training data x(0).

Hence, for any δ in R p , the derivative g(δ) in FORMULA24 can also be written as DISPLAYFORM10 Taking the difference between FORMULA24 and FORMULA25 , DISPLAYFORM11 Now, by the model assumptions, x i,j − x i,j (0) = W ∆ i,j .

Since δ is in the column-space of W , there exists u ∈ R q such that δ = W u. then (9) can be written as DISPLAYFORM12 From (A2) we have that the eigenvalues of W t W are all positive.

Also r i (θ * ) is not a function of the interventions ∆ i,j since, as already argued above, the estimatorθ * is identical whether trained on the original data x i,j or on the counterfactual data x i,j (0).

If we condition on (x i (0), y i ) for i = 1, . . .

, n (that is everything except for the random ∆ i,j , i = 1, . . .

, n), then the interventions ∆ i,j are by (A1) drawn from a continuous distribution.

Hence the left hand side of (10) has a continuous distribution, and the probability of the left hand side of (10) being not identically 0 is 1.

This completes the proof of the first part by contradiction.

Second part.

For the second part, we first show that with probability 1,θ core =θ * withθ * defined as in (6).

Note that the invariant space is for this model the linear subspace I = {θ : W t θ = 0}. Note that by their respective definitions, DISPLAYFORM13 By (A2) and (A3), with probability 1, I n = {θ : W t θ = 0} since the number of counterfactuals examples is equal to or exceeds the rank q of W and X ⊥ has a linear influence on X. Hence with probability 1, we have I = I n and henceθ core =θ * .

We thus need to show that DISPLAYFORM14 Sinceθ * is in I, we have (y, x(∆)) = (y, x(0)), where x(0) are the previously discussed counterfactual data in the absence of interventions.

Hencê DISPLAYFORM15 that is the estimator is unchanged if we use the data without interventions (∆ i = 0) as training data.

Define the population-optimal vector as DISPLAYFORM16 which can for the same reason be written as DISPLAYFORM17 Hence FORMULA3 and FORMULA3 can be written aŝ DISPLAYFORM18 Comparing FORMULA3 and FORMULA3 , by uniform convergence of Ln to the population loss L (0) under the assumed sampling where n samples of (Y, ID) are drawn independently then c = m − n samples are redrawn from this empirical sample at random, we have DISPLAYFORM19 By definition of I and θ DISPLAYFORM20 adv .

This completes the proof, using the previous fact thatθ core =θ * with probability 1 under (A3).

We work with the CelebA dataset BID26 and consider the problem of classifying whether the person in the image is male or female.

We create a confounding by including mostly images of men wearing glasses while the images of women do not include photos of women with glasses.

As counterfactuals, we use an image of the same person without glasses if the person is male and with glasses if the person is female.

We call using an image of the same person as counterfactual "CF setting 2".

Examples from the training and test sets are shown in FIG0 .2.

Test set 1 follows the same distribution as the training set.

In test set 2 the association between gender and glasses is flipped: women always wear glasses while men never wear glasses.

In this example, we would like to assess whether the results will differ when (a) training a fourlayer CNN (as detailed in TAB1 .1) end-to-end versus (b) using Inception V3 features and merely retraining the softmax layer.

FIG0 .2 shows the results for varying numbers of m and c-in the left column for training a four-layer CNN; in the right column for using Inception V3 features.

Overall, we see the same trends: As c increases, the performance difference between CORE and the pooled estimator becomes smaller.

This is due to the fact that X ⊥ is binary in this example and, therefore, including counterfactual examples corresponds to data augmentation.

Interestingly, the pooled estimator performs worse on test set 2 as m becomes larger.

It thus seems to exploit X ⊥ to a larger extent as m grows.

As in §5.2 we work with the CelebA dataset and consider the problem of classifying whether the person in the image is wearing eyeglasses.

Here we analyze a confounded setting that could arise as follows.

Say the hidden common cause of Y and X ⊥ , D indicates whether the image was taken outdoors or indoors.

If it was taken outdoors, then the person wears glasses and the image tends to be brighter.

If the image was taken indoors, then the person does not wear glasses and the image tends to be darker.

In other words, X ⊥ ≡ brightness and the structure of the data generating process is equivalent to the one shown in FIG0 Examples from all test sets can be found in FIG0 .4.

Test set 1 follows the same distribution as the training set.

In test set 2 the sign of the brightness intervention is reversed, i.e. images of people with glasses tend to be darker; images of people without glasses tend to be brighter.

In test set 3 all images are left unchanged and in test set 4 the brightness of all images is increased.

First, we notice that the pooled estimator performs better than CORE on test set 1.

This can be explained by the fact that it can exploit the predictive information contained in the brightness of an image while CORE is restricted not to do so.

Second, we observe that the pooled estimator does not perform well on test sets 2 and 4 as its learned representation seems to use the image's brightness as a predictor for the response which fails when the brightness distribution in the test set differs significantly from the training set.

In contrast, the predictive performance of CORE is hardly affected by the changing brightness distributions.

Results for β ∈ {5, 10, 20} and c ∈ {200, 5000} can be found in FIG0 .5.

Above we used the same image to create a counterfactual observation by sampling a different value for the brightness intervention.

A plausible alternative is to use a different image of the same person as counterfactual.

We call this "CF setting 2".

For comparison, we also evaluate using an image of a different person as counterfactual as a baseline ("CF setting 3").

Examples from the training sets using CF setting 2 and 3 can be found in FIG0 .4.Results for all counterfactual settings, β ∈ {5, 10, 20} and c ∈ {200, 5000} can be found in FIG0 .5.

We see that using counterfactual setting 1 works best since we could explicitly control that only X ⊥ ≡ brightness varies between counterfactual examples.

In counterfactual setting 2, different images of the same person can vary in many factors, making it more challenging to isolate brightness as the factor to be invariant against.

Lastly, we see that even grouping images of different persons can still help predictive performance to some degree.

INTRODUCED IN §2 AND §5

An open question is how to set the value of the tuning parameter τ in Eq. (4) or the penalty λ in the Lagrangian form.

FIG0 .6 shows the misclassification rates of CORE on the subsampled and augmented AwA2 dataset as a function of the penalty λ.

We see that performance is not very sensitive to the choice of λ.

Here, we show further results for the experiment introduced in §2.1.

We vary the number of identities included in the training data set n ∈ {10, 20, 40, 80, 160}. This results in total sample sizes m ranging from 321 for n = 10 to 4386 for n = 160, implying that the average number of counterfactual observations per person varies between 27 and 32.

FIG0 .7b shows the misclassification rates for the test set which consists of 5000 examples.

We see that CORE helps predictive performance compared to the estimator which just pools all images, notably when n is very small.

It thus successfully mitigates the effect of potential confounders arising due to small sample sizes.

As n and m increase the performance of CORE and the pooled estimator become comparable-the larger sample sizes ensure that fewer confounding factors are present in the training data and exploited by the pooled estimator.

Here, we show further results for the experiment introduced in §2.2.

We vary the number of augmented training examples c from 100 to 5000 for n = 10000 and c ∈ {100, 200, 500, 1000} for n = 1000.

The degree of the rotations is sampled uniformly at random from [35, 70] .

FIG0 .8 shows the misclassification rates.

Test set 1 contains rotated digits only, test set 2 is the usual MNIST test set.

We see that the misclassification rates of CORE are always lower on test set 1, showing that it makes data augmentation more efficient.

For n = 1000, it even turns out to be beneficial for performance on test set 2.

Here, we show further results for the experiment introduced in §5.1.

FIG0 10b shows results for different numbers of counterfactual examples.

For c = 20 the misclassification rate of CORE estimator has a large variance.

For c ∈ {50, 500, 2000}, the CORE estimator shows similar results.

Its performance is thus not sensitive to the number of counterfactual examples, once there are sufficiently many counterfactual observations in the training set.

The pooled estimator fails to achieve good predictive performance on test sets 2 and 3 as it seems to use "movement" as a predictor for "age".

We run experiments for counterfactual settings 1-3 and for c = 5000.

FIG0 .11 shows examples from the respective training and test sets and FIG0 .12 shows the corresponding misclassification rates.

Again, we observe that counterfactual setting 1 works best while there are only small differences in predictive performance between counterfactual settings 2 and 3.

Interestingly, there is a large performance difference between µ = 40 and µ = 50 for the pooled estimator.

Possibly, with µ = 50 the image quality is not sufficiently predictive for the target.

Under review as a conference paper at ICLR 2018 latter command is applied to all images.

It rotates the colors of the image, in a cyclic manner 10 .

In test set 3, all images are changed to grayscale.

We implemented the considered models in TensorFlow BID0 .

The model architectures used are detailed in TAB1 .1.

CORE and the pooled estimator thus use the same network architecture and training procedure; merely the loss function differs by the counterfactual regularization term.

In all experiments we use the Adam optimizer BID22 .All experimental results are based on training the respective model five times (using the same data) to assess the variance due to the randomness in the training procedure.

In each epoch of the training, the training data x i,· , i = 1, . . . , n is randomly shuffled, keeping the counterfactual observations x i,j , j = 1, . . .

, m i together to ensure that mini batches will contain counterfactual observations.

In all experiments the mini batch size is set to 120.

For small c this implies that not all mini batches contain counterfactual observations, making the optimization more challenging.

@highlight

We propose counterfactual regularization to guard against adversarial domain shifts arising through shifts in the distribution of latent "style features" of images.

@highlight

The paper discusses ways to guard against adversarial domain shifts with counterfactual regularization by learning a classifier that is invariant to superficial changes (or "style" features) in imagess.

@highlight

This paper aims at robust image classification against adversarial domain shifts and the goal is achieved by avoiding using the changing style features.