Local explanation frameworks aim to rationalize particular decisions made by a black-box prediction model.

Existing techniques are often restricted to a specific type of predictor or based on input saliency, which may be undesirably sensitive to factors unrelated to the model's decision making process.

We instead propose sufficient input subsets that identify minimal subsets of features whose observed values alone suffice for the same decision to be reached, even if all other input feature values are missing.

General principles that globally govern a model's decision-making can also be revealed by searching for clusters of such input patterns across many data points.

Our approach is conceptually straightforward, entirely model-agnostic, simply implemented using instance-wise backward selection, and able to produce more concise rationales than existing techniques.

We demonstrate the utility of our interpretation method on neural network models trained on text and image data.

The rise of neural networks and nonparametric methods in machine learning (ML) has driven significant improvements in prediction capabilities, while simultaneously earning the field a reputation of producing complex black-box models.

Vital applications, which could benefit most from improved prediction, are often deemed too sensitive for opaque learning systems.

Consider the widespread use of ML for screening people, including models that deny defendants' bail [1] or reject loan applicants [2] .

It is imperative that such decisions can be interpretably rationalized.

Interpretability is also crucial in scientific applications, where it is hoped that general principles may be extracted from accurate predictive models [3, 4, 5].One simple explanation for why a particular black-box decision is reached may be obtained via a sparse subset of the input features whose values form the basis for the model's decision -a rationale.

For text (or image) data, a rationale might consist of a subset of positions in the document (or image) together with the words (or pixel-values) occurring at these positions (see FIG0 .

To ensure interpretations remain fully faithful to an arbitrary model, our rationales do not attempt to summarize the (potentially complex) operations carried out within the model, and instead merely point to the relevant information it uses to arrive at a decision [6] .

For high-dimensional inputs, sparsity of the rationale is imperative for greater interpretability.

Here, we propose a local explanation framework to produce rationales for a learned model that has been trained to map inputs x P X via some arbitrary learned function f : X Ñ R. Unlike many other interpretability techniques, our approach is not restricted to vector-valued data and does not require gradients of f .

Rather, each input example is solely presumed to have a set of indexable features x " rx 1 , . . .

, x p s, where each x i P R d for i P rps " t1, . . . , pu.

We allow for features that are unordered (set-valued input) and whose number p may vary from input to input.

A rationale corresponds to a sparse subset of these indices S Ď rps together with the specific values of the features in this subset.

To understand why a certain decision was made for a given input example x, we propose a particular rationale called a sufficient input subset (SIS).

Each SIS consists of a minimal input pattern present in x that alone suffices for f to produce the same decision, even if provided no other information about the rest of x. Presuming the decision is based on f pxq exceeding some pre-specified threshold τ P R, we specifically seek a minimal-cardinality subset S of the input features such that f px S q ě τ .

Throughout, we use x S P X to denote a modified input example in which all information about the values of features outside subset S has been masked with features in S remaining at their original values.

Thus, each SIS characterizes a particular standalone input pattern that drives the model toward this decision, providing sufficient justification for this choice from the model's perspective, even without any information on the values of the other features in x.

In classification settings, f might represent the predicted probability of class C where we decide to assign the input to class C if f pxq ě τ , chosen based on precision/recall considerations.

Each SIS in such an application corresponds to a small input pattern that on its own is highly indicative of class C, according to our model.

Note that by suitably defining f and τ with respect to the predictor outputs, any particular decision for input x can be precisely identified with the occurrence of f pxq ě τ , where higher values of f are associated with greater confidence in this decision.

For a given input x where f pxq ě τ , this work presents a simple method to find a complete collection of sufficient input subsets, each satisfying f px S q ě τ , such that there exists no additional SIS outside of this collection.

Each SIS may be understood as a disjoint piece of evidence that would lead the model to the same decision, and why this decision was reached for x can be unequivocally attributed to the SIS-collection.

Furthermore, global insight on the general principles underlying the model's decision-making process may be gleaned by clustering the types of SIS extracted across different data points (see FIG4 and TAB0 ).

Such insights allow us to compare models based not only on their accuracy, but also on human-determined relevance of the concepts they target.

Our method's simplicity facilitates its utilization by non-experts who may know very little about the models they wish to interrogate.

Certain neural network variants such as attention mechanisms [7] and the generator-encoder of [6] have been proposed as powerful yet human-interpretable learners.

Other interpretability efforts have tailored decompositions to certain convolutional/recurrent networks [8, 9, 10, 11] , but these approaches are model-specific and only suited for ML experts.

Many applications necessitate a model outside of these families, either to ensure supreme accuracy, or if training is done separately with access restricted to a black-box API [12, 13 ].An alternative model-agnostic approach to interpretability produces local explanations of f for a particular input x (e.g. an individual classification decision).

Popular local explanation techniques produce attribution scores that quantify the importance of each feature in determining the output of f at x. Examples include LIME, which locally approximates f [14], saliency maps based on f -gradients [15, 16] , Layer-wise Relevance Propagation [17] , as well as the discrete DeepLIFT approach [5] and its continuous variant -Integrated Gradients (IG), developed to ensure attributions reflect the cumulative difference in f at x vs. a reference input [18] .

A separate class of input-signal-based explanation techniques such as DeConvNet [19] , Guided Backprop [20] , and PatternNet [21] employ gradients of f in order to identify input patterns that cause f to output large values.

However, many such gradient-based saliency methods have been found unreliable, depending not only on the learned function f , but also on its specific architectural implementation and how inputs are scaled [22, 21] .

More similar to our approach are recent techniques [23, 24, 25] which also aim to identify input patterns that best explain certain decisions, but additionally require either a predefined set of such patterns or an auxiliary neural network trained to identify them.

In comparison with the aforementioned methods, our SIS approach presented here is conceptually simple, completely faithful to any type of model, requires no access to gradients of f , requires no additional training of the underlying model f , and does not require training any auxiliary explanation model.

Also related to our subset-selection methodology are the ideas of Li et al. [26] and Fong & Veldadi [27] , which for a particular input example aim to identify a minimal subset of features whose deletion causes a substantial drop in f such that a different decision would be reached.

However, this objective can undesirably produce adversarial artifacts that are not easy to interpret [27] .

In contrast, we focus on identifying disjoint minimal subsets of input features whose values suffice to ensure f outputs significantly positive predictions, even in the absence of any other information about the rest of the input.

While the techniques used in [26, 27] produce rationales that remain strongly dependent on the rest of the input outside of the selected feature subset, each rationale revealed by our SIS approach is independently considered by f as an entirely sufficient justification for a particular decision in the absence of other information.

Our approach to rationalizing why a particular black-box decision is reached only applies to input examples x P X that meet the decision criterion f pxq ě τ .

For such an input x, we aim to identify a SIS-collection of disjoint feature subsets S 1 , . . .

, S K Ď rps that satisfy the following criteria:(1) f px S k q ě τ for each k " 1, . . .

, K (2) There exists no feature subset S 1 Ă S k for some k " 1, . . .

, K such that f px S 1 q ě τ (3) f px R q ă τ for R " rps z Ť K k"1 S k (the remaining features outside of the SIS-collection) Criterion (1) ensures that for any SIS S k , the values of the features in this subset alone suffice to justify the decision in the absence of any information regarding the values of the other features.

To ensure information that is not vital to reach the decision is not included within the SIS, criterion (2) encourages each SIS to contain a minimal number of features, which facilitates interpretability.

Finally, we require that our SIS-collection satisfies a notion of completeness via criterion (3), which states that the same decision is no longer reached for the input after the entire SIS-collection has been masked.

This implies the remaining feature values of the input no longer contain sufficient evidence for the same decision.

FIG1 show SIS-collections found in text/image inputs.

Recall that x S P X denotes a modified input in which the information about the values of features outside subset S is considered to be missing.

We construct x S as new input whose values on features in S are identical to those in the original x, and whose remaining features x i P rpszS are each replaced by a special mask z i P R di used to represent a missing observation.

While certain models are specially adapted to handle inputs with missing observations [28] , this is generally not the case.

To ensure our approach is applicable to all models, we draw inspiration from data imputation techniques which are a common way to represent missing data [29] .Two popular strategies include hot-deck imputation, in which unobserved values are sampled from their marginal feature distribution, and mean imputation, in which each z i simply fixed to the average value of feature i in the data.

Note that for a linear model, these two strategies are expected to produce an identical change in prediction f pxq´f px S q. We find in practice that the change in predictions resulting from either masking strategy is roughly equivalent even for nonlinear models such as neural networks ( FIG0 ).

In this work, we favor the mean-imputation approach over samplingbased imputation, which would be computationally-expensive and nondeterministic (undesirable for facilitating interpretability).

One may also view z as the baseline input value used by feature attribution methods [18, 5] , a value which should not lead to particularly noteworthy decisions.

Since our interests primarily lie in rationalizing atypical decisions, the average input arising from mean imputation serves as a suitable baseline.

Zeros have also been used to mask image/categorical data [26] , but empirically, this mask appears undesirably more informative than the mean (predictions more affected by zero-masking).For an arbitrarily complex function f over inputs with many features p, the combinatorial search to identify sets which satisfy objectives (1)-(3) is computationally infeasible.

To find a SIS-collection in DISPLAYFORM0 Update S Ð S Y tiu 5 if f px S q ě τ : return S 6 else: return None practice, we employ a straightforward backward selection strategy, which is here applied separately on an example-by-example basis (unlike standard statistical tools which perform backward selection globally to find a fixed set of features for all inputs).

The SIScollection algorithm details our straightforward procedure to identify disjoint SIS subsets that satisfy (1)-(3) approximately (as detailed in §3.1) for an input x P X where f pxq ě τ .Our overall strategy is to find a SIS subset S k (via BackSelect and FindSIS), mask it out, and then repeat these two steps restricting each search for the next SIS solely to features disjoint from the currently found SIS-collection S 1 , . . .

, S k , until the decision of interest is no longer supported by the remaining feature values.

In the BackSelect procedure, S Ă rps denotes the set of remaining unmasked features that are to be considered during backward selection.

For the current subset S, step 3 in BackSelect identifies which remaining feature i P S produces the minimal reduction in f px S q´f px Sztiu q (meaning it least reduces the output of f if additionally masked), a question trivially answered by running each of the remaining possibilities through the model.

This strategy aims to gradually mask out the least important features in order to reveal the core input pattern that is perceived by the model as sufficient evidence for its decision.

Finally, we build our SIS up from the last features omitted during the backward selection, selecting a value just large enough to meet our sufficiency criterion (1).

Because this approach always queries a prediction over the joint set of remaining features S, it is better suited to account for interactions between these features and ensure their sufficiency (i.e. that f px S q ě τ ) compared to a forward selection in the opposite direction which builds the SIS upwards one feature at a time by greedily maximizing marginal gains.

Throughout its execution, BackSelect attempts to maintain the sufficiency of x S as the set S shrinks.

Given p input features, our algorithm requires Opp 2 kq evaluations of f to identify k SIS, but we can achieve Oppkq by parallelizing each argmax in BackSelect (e.g. batching on GPU).

Throughout, let S 1 , . . .

, S K denote the output of SIScollection when applied to a given input x for which f pxq ě τ .

Disjointness of these sets is crucial to ensure computational tractability and that the number of SIS per example does not grow huge and hard to interpret.

Proposition 1 below proves that each SIS produced by our procedure will satisfy an approximate notion of minimality.

Because we desire minimality of the SIS as specified by (2), it is not appropriate to terminate the backward elimination in BackSelect as soon as the sufficiency condition f px S q ě τ is violated, due to the possible presence of local minima in f along the path of subsets encountered during backward selection (as shown in FIG1 ).Proposition 2 additionally guarantees that masking out the entirety of the feature values in the SIScollection will ensure the model makes a different decision.

Given f pxq ě τ , it is thus necessarily the case that the observed values responsible for this decision lie within the SIS-collection S 1 , . . .

, S K .

We point out that for an easily reached decision, where f pzq ě τ (i.e. this decision is reached even for the average input), our approach will not output any SIS.

Because this same decision would likely be anyway reached for a vast number of inputs in the training data (as a sort of default decision), it is conceptually difficult to grasp what particular aspect of the given x is responsible.

Proposition 1.

There exists no feature i in any set S 1 , . . .

, S K that can be additionally masked while retaining sufficiency of the resulting subset (i.e. f px S k ztiu q ă τ for any k " 1, . . .

, K, i P S k ).

Also, among all subsets S considered during the backward selection phase used to produce S k , this set has the smallest cardinality of those which satisfy f px S q ě τ .

Proposition 2.

For x rpszS˚, modified by masking all features in the entire SIS-collection S˚" Ť K k"1 S k , we must have: f px rpszS˚q ă τ when S˚‰ rps.

Unfortunately, nice assumptions like convexity/submodularity are inappropriate for estimated functions in ML.

We present various simple forms of practical decision functions for which our algorithms are guaranteed to produce desirable explanations.

Example 1 considers interpreting functions of a generalized linear form, Examples 2 & 3 describe functions whose operations resemble generalized logical OR & AND gates, and Example 4 considers functions that seek out a particular input pattern.

Note that features ignored by f are always masked in our backward selection and thus never appear in the resulting SIS-collection.

Example 1.

Suppose the input data are vectors and f pxq " gpβ T x`β 0 q, where g is monotonically increasing.

We also presume τ ą gpβ 0 q and the data were centered such that each feature has mean zero (for ease of notation).

In this case, S 1 , . . .

, S K must satisfy criteria (1)-(3).

S 1 will consist of the features whose indices correspond to the largest entries of tβ 1 x 1 , . . .

, β p x p u for some suitable that depends on the value of τ .

It is also guaranteed that f px S1 q ě f px S q for any subset S Ď rps of the same cardinality |S| " .

For each individual feature i where gpβ i x i`β0 q ě τ , there will be exist a corresponding SIS S k consisting only of tiu.

No SIS will include features whose coefficient β i " 0, or those whose difference between the observed and average value z i (" 0 here) is of an opposite sign than the corresponding model coefficient (i.e. β i px i´zi q ď 0).

DISPLAYFORM0 . ., g L , such that for the given x and threshold τ : DISPLAYFORM1 Such f might be functions that model strong interactions between the features in each S k or look for highly specific value patterns to occur these subsets.

In this case, SIScollection will return L sets such that DISPLAYFORM2 qu and the same conditions from Example 2 are met, then SIScollection will return a single set DISPLAYFORM3 p with f pxq " hp||x S´cS ||q where h is monotonically decreasing and c S specifies a fixed pattern of input values for features in a certain subset S. For input x and threshold choice τ " f pxq, SIScollection will return a single set S 1 " ti P S : |x i´ci | ă |z i´ci |u.

We apply our methods to analyze neural networks for text and image data.

SIScollection is compared with alternative subset-selection methods for producing rationales (see descriptions in Supplement §S1).

Note that our BackSelect procedure determines an ordering of elements, R, subsequently used to construct the SIS.

Depictions of each SIS are shaded based on the feature order in R (darker = later), which can indicate relative feature importance within the SIS.

In the "Suff.

IG," "Suff.

LIME," and "Suff.

Perturb." (sufficiency constrained) methods, we instead compute the ordering of elements R according to the feature attribution values output by integrated gradients [18] , LIME [14], or a perturbative approach that measures the change in prediction when individually masking each feature (see §S1).

The rationale subset S produced under each method is subsequently assembled using FindSIS exactly as in our approach and thus is guaranteed to satisfy f px S q ě τ .

In the "IG," "LIME," and "Perturb." (length constrained) methods, we use the same previously described ordering R, but always select the same number of features in the rationale as in the SIS produced by our method (per example).

We first consider a dataset of beer reviews from BeerAdvocate [30] .

Taking the text of a review as input, different LSTM networks [31] are trained to predict user-provided numerical ratings of aspects like aroma, appearance, and palate (details in §S2).

FIG0 shows a sample beer review where we highlight the SIS identified for the LSTM that predicts each aspect.

Each SIS only captures sentiment toward the relevant aspect.

FIG1 depicts the SIS-collection identified from a review the LSTM decided to flag for positive aroma.

FIG2 shows that when the alternative methods described in §4 are length constrained, the rationales they produce often badly fail to meet our sufficiency criterion.

Thus, even though the same number of feature values are preserved in the rationale and these alternative methods select the features to which they have assigned the largest attribution values, their rationales lead to significantly reduced f outputs compared to our SIS subsets.

If the sufficiency constraint is instead enforced for these alternative methods, the rationales they identify become significantly larger than those produced by SIScollection, and also contain many more unimportant features (Table S2, FIG1 ).Benchmarking interpretability methods is difficult because a learned f may behave counterintuitively such that seemingly unreasonable model explanations are in fact faithful descriptions of a model's decision-making process.

For some reviews, a human annotator has manually selected which sentences carry the relevant sentiment for the aspect of interest, so we treat these annotations as an alternative rationale for the LSTM prediction.

For a review x whose true and predicted aroma exceed our decision threshold, we define the quality of human-selected sentences for model explanation QHS " f px S q´f pxq where S is the human-selected-subset of words in the review (see examples in FIG6 ).

High variability of QHS in the annotated reviews FIG3 indicates the human rationales often do not contain sufficient information to preserve the LSTM's decision.

FIG3 shows the LSTM makes many decisions based on different subsets of the text than the parts that humans find appropriate for this task.

Reassuringly, our SIS more often lie within the selected annotation for reviews with high QHS scores.

We also study a 10-way CNN classifier trained on the MNIST handwritten digits data [32] .

Here, we only consider predicted probabilities for one class of interest at a time and always set τ " 0.7 as the probability threshold for deciding that an image belongs to the class.

We extract the SIS-collection from all corresponding test set examples (details in §S3).

Example images and corresponding SIScollections are shown in Figures 6, 7, and S27.

FIG5 illustrates how the SIS-collection drastically changes for an example of a correctly-classified 9 that has been adversarially manipulated [33] to become confidently classified as the digit 4.

Furthermore, these SIS-collections immediately enable us to understand why certain misclassifications occur FIG5 ).

Identifying the different input patterns that justify a decision can help us better grasp the general operating principles of a model.

To this end, we cluster all of the SIS produced by SIScollection applied across a large number of data examples that received the same decision.

Clustering is done via DBSCAN, a widely applicable algorithm that merely requires specifying pairwise distances between points [34].We first apply this procedure to the SIS found across all held-out beer reviews (Test-Fold in TAB0 ) that received positive aroma predictions from our LSTM network.

The distance between two SIS is taken as the Jaccard distance between their bag of words representations.

Three clusters depicted in TAB0 ) reveal isolated phrases that the LSTM associates with positive aromas in the absence of other context.

We also apply DBSCAN clustering to the SIS found across all MNIST test-examples confidently identified by the CNN as a particular class.

Pairwise distances are here defined as the energy distance [35] over pixel locations between two SIS subsets (see §S3.3).

FIG4 depicts the SIS clusters identified for digit 4 (others in FIG1 ).

These reveal distinct feature patterns learned by the CNN to distinguish 4 from other digits, which are clearly present in the vast majority of test set images confidently classified as a 4.

For example, cluster C 8 depicts parallel slanted lines, a pattern that never occurs in other digits.

The general insights revealed by our SIS-clustering can also be used to compare the operatingbehavior of different models.

For the beer reviews, we also train a CNN to compare with our existing LSTM (see §S2.6).

For MNIST, we train a multilayer perceptron (MLP) and compare to our existing CNN (see §S3.5).

Both networks exhibit similar performance in each task, so it is not immediately clear which model would be preferable in practice.

FIG8 shows the SIS extracted under one model are typically insufficient to receive the same decision from the other model, indicating these models base their positive predictions on different evidence.

TAB1 contains results of jointly clustering the SIS extracted from beer reviews with positive aroma predictions under our LSTM or text-CNN.

This CNN tends to learn localized (unigram/bigram) word patterns, while the LSTM identifies more complex multi-word interactions that truly seem more relevant to the target aroma value.

Many CNN-SIS are simply phrases with universally-positive sentiment, indicating this model is less capable at distinguishing between positive sentiment toward

This work introduced the idea of interpreting black-box decisions on the basis of sufficient input subsets -minimal input patterns that alone provide sufficient evidence to justify a particular decision.

Our methodology is easy to understand for non-experts, applicable to all ML models without any additional training steps, and remains fully faithful to the underlying model without making approximations.

While we focus on local explanations of a single decision, clustering the SISpatterns extracted from many data points reveals insights about a model's general decision-making process.

Given multiple models of comparable accuracy, SIS-clustering can uncover critical operating differences, such as which model is more susceptible to spurious training data correlations or will generalize worse to counterfactual inputs that lie outside the data distribution.

[1] Kleinberg J, Lakkaraju H, Leskovec J, Ludwig J, Mullainathan S (2018) Human decisions and machine predictions.

The Quarterly Journal of Economics 133: 237-293.[2]

Sirignano JA, Sadhwani A, Giesecke K (2018) Deep learning for mortgage risk.

arXiv:160702470 .[3] Doshi-Velez F, Kim B (2017) Towards a rigorous science of interpretable machine learning.

arXiv:170208608 .[4]

Lipton ZC (2016) The mythos of model interpretability.

In: ICML Workshop on Human Interpretability of Machine Learning.[5]

Shrikumar A, Greenside P, Kundaje A (2017) Learning important features through propagating activation differences.

In: International Conference on Machine Learning.[6]

Lei T, Barzilay R, Jaakkola T (2016) Rationalizing neural predictions.

In: Empirical Methods in Natural Language Processing. [34] Ester M, Kriegel HP, Sander J, Xu X (1996) A density-based algorithm for discovering clusters a density-based algorithm for discovering clusters in large spatial databases with noise.

In: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining.[ List of TAB0

In Section 3, we describe a number of alternative methods for identifying rationales for comparison with our method.

We use methods based on integrated gradients BID0 , LIME BID1 , and feature perturbation.

Note that integrated gradients is an attribution method which assigns a numerical score to each input feature.

LIME likewise assigns a weight to each feature using a local linear regression model for f around x. In the perturbative approach, we compute the change in prediction when each feature is individually masked, as in Equation 1 (of Section S2.4).

Each of these feature orderings R is used to construct a rationale using the FindSIS procedure (Section 3) for the "Suff.

IG," "Suff.

LIME," and "Suff.

Perturb." (sufficiency constrained) methods.

Note that our text classification architecture (described in Section S2.2) encodes discrete words as 100-dimensional continuous word embeddings.

The integrated gradients method returns attribution scores for each coordinate of each word embedding.

For each word embedding x i P x (where each x i P R 100 ), we summarize the attributions along the corresponding embedding into a single score y i using the L 1 norm: y i " ř d |x id | and compute the ordering R by sorting the y i values.

We use an implementation of integrated gradients for Keras-based models from https://github.

com/hiranumn/IntegratedGradients.

In the case of the beer review dataset (Section 4.1), we use the mean embedding vector as a baseline for computing integrated gradients.

As suggested in BID0 , we verified that the prediction at the baseline and the integrated gradients sum to approximately the prediction of the input.

For LIME and our beer reviews dataset, we use the approach described in BID1 for textual data, where individual words are removed entirely from the input sequence.

We use the implementation of LIME at: https://github.com/marcotcr/lime.

The LimeTextExplainer module is used with default parameters, except we set the maximal number of features used in the regression to be the full input length so we can order all input features.

Additionally, we explore methods in which we use the same ordering R by these alternative methods but select the same number of input features in the rationale to be the median SIS length in the SIS-collection computed by our method on each example: the "IG," "LIME," and "Perturb.

" (length constrained) methods.

We compute the feature ordering based on the absolute value of the non-zero integrated gradient attributions.

Note that for the length constrained methods, there is no guarantee of sufficiency f px S q ě τ for any input subset S.

As done in BID2 , we use a preprocessed version of the BeerAdvocate 2 dataset 3 which contains decorrelated numerical ratings toward three aspects: aroma, appearance, and palate (each normalized to r0, 1s).

Dataset statistics can be found in TAB0 .

Reviews were tokenized by converting to lowercase and filtering punctuation, and we used a vocabulary containing the top 10,000 most common words.

The data also contain subset of human-annotated reviews, in which humans manually selected full sentences in each review that describe the relevant aspects BID3 .

This annotated set was never seen during training and used solely as part of our evaluation.

Long short-term memory (LSTM) networks are commonly employed for natural language tasks such as sentiment analysis BID4 BID5 .

We use a recurrent neural network (RNN) architecture with two stacked LSTMs as follows:1.

Input/Embeddings Layer: Sequence with 500 timesteps, the word at each timestep is represented by a (learned) 100-dimensional embedding 2.

LSTM Layer 1: 200-unit recurrent layer with LSTM (forward direction only) 3.

LSTM Layer 2: 200-unit recurrent layer with LSTM (forward direction only) 4.

Dense: 1 neuron (sentiment output), sigmoid activationWith this architecture, we use the Adam optimizer BID6 to minimize mean squared error (MSE) on the training set.

We use a held-out set of 3,000 examples for validation (sampled at random from the pre-defined test set used in BID2 ).

Our test set consists of the remaining 7,000 test examples.

Training results are shown in TAB0 .

In Section 3, we discuss the problem of masking input features.

Here, we show that the meanimputation approach (in which missing inputs are masked with a mean embedding, taken over the entire vocabulary) produces a nearly identical change in prediction to a nondeterministic hot-deck approach (in which missing inputs are replaced by randomly sampling feature-values from the data).

FIG0 shows the change in prediction f pxztiuq´f pxq by both imputation techniques after drawing a training example x and word x i P x (both uniformly at random) and replacing x i with either the mean embedding or a randomly selected word (drawn from the vocabulary, based on counts in the training corpus).

This procedure is repeated 10,000 times.

Both resulting distributions have mean near zero (µ mean-embedding "´7.0e´4, µ hot-deck "´7.4e´4), and the distribution for mean embedding is slightly narrower (σ mean-embedding " 0.013, σ hot-deck " 0.018).

We conclude that mean-imputation is a suitable method for masking information about particular feature values in our SIS analysis.

We also explored other options for masking word information, e.g. replacement with a zero embedding, replacement with the learned <PAD> embedding, and simply removing the word entirely from the input sequence, but each of these alternative options led to undesirably larger changes in predicted values as a result of masking, indicating they appear more informative to f than replacement via the feature-mean.

For each feature i in the input sequence, we quantify its marginal importance by individually perturbing only this feature:Feature Importancepiq " prediction on original input´prediction with feature i masked (1) FIG0 : Change in prediction (f pxztiuq´f pxq) after masking a randomly chosen word with mean imputation or hot-deck imputation.

10,000 replacements were sampled from the aroma beer reviews training set.

Note that these marginal Feature Importance scores are identical to those of the Perturb.

method described in Section S1.

The marginal Feature Importance scores are summarized in TAB1 and FIG1 .

Compared to the Suff.

IG and Suff.

LIME methods, our SIScollection technique produces rationales that are much shorter and contain fewer irrelevant (i.e. not marginally important) features TAB1 , FIG1 .

Note that by construction, the rationales of the Suff.

Perturb.

method contain features with the greatest Feature Importance, since this precisely how the ranking in Suff.

Perturb. is defined.

We apply our method to the set of reviews containing sentence-level annotations.

Note that these reviews (and the human annotations) were not seen during training.

We choose thresholds τ`" 0.85, FIG1 : Importance of individual features in the rationales for aroma prediction in beer reviews FIG2 : Length of rationales for aroma prediction FIG3 : Predictive distribution on the annotation set (held-out) using the LSTM model for aroma.

Vertical lines indicate decision thresholds (τ`" 0.85, τ´" 0.45) selected for SIScollection.

τ´" 0.45 for strong positive and strong negative sentiment, respectively, and extract the complete set of sufficient input subsets using our method.

Note that in our formulation above, we apply our method to inputs x where f pxq ě τ .

For the sentiment analysis task, we analogously apply our method for both f pxq ě τ`and´f pxq ě´τ´, where the model predicts either strong positive or strong negative sentiment, respectively.

These thresholds were set empirically such that they were sufficiently apart, based on the distribution of predictions ( FIG3 ).

For most reviews, SIScollection outputs just one or two SIS sets ( FIG4 ).We analyzed the predictor output following the elimination of each feature in the BackSelect procedure (Section 3).

FIG5 shows the LSTM output on the remaining unmasked text f px Szti˚u q at each iteration of BackSelect, for all examples.

This figure reveals that only a small number of features are needed by the model in order to make a strong prediction (most features can be removed without changing the prediction).

We see that as those final, critical features are removed, there is a rapid, monotonic decrease in output values.

Finally, we see that the first features to be removed by BackSelect are those which generally provide negative evidence against the decision.

We demonstrate how our SIS-clustering procedure can be used to understand differences in the types of concepts considered important by different neural network architectures.

In addition to the LSTM (see Section S2.2), we trained a convolutional neural network (CNN) on the same sentiment analysis task (on the aroma aspect).

The CNN architecture is as follows: FIG6 : Beer reviews (aroma) in which human-selected sentences (underlined) are aligned well (top) and poorly (bottom) with predictive model.

Fraction of SIS in the human sentences corresponds accordingly.

In the bottom example (poor alignment between human-selection and predictive model), our procedure has surfaced a case where the LSTM has learned features that diverge from what a human would expect (and may suggest overfitting).

1. Input/Embeddings Layer: Sequence with 500 timesteps, the word at each timestep is represented by a (learned) 100-dimensional embedding 2.

Convolutional Layer 1: Applies 128 filters of window size 3 over the sequence, with ReLU activation 3.

Max Pooling Layer 1: Max-over-time pooling, followed by flattening, to produce a p128, q representation 4.

Dense: 1 neuron (sentiment output), sigmoid activation Note that a new set of embeddings was learned with the CNN.

As with the LSTM model, we use Adam BID6 to minimize MSE on the training set.

For the aroma aspect, this CNN achieves 0.016 (0.850), 0.025 (0.748), 0.026 (0.741), 0.014 (0.662) MSE (and Pearson ρ) on the Train, Validation, Test, and Annotation sets, respectively.

We note that this performance is very similar to that from the LSTM (see TAB0 ).We apply our procedure to extract the SIS-collection from all applicable test examples using the CNN, as in Section 4.1.

FIG8 shows the predictions from one model (LSTM or CNN) when fed input examples that are SIS extracted with respect to the other model (for reviews predicted to have positive sentiment toward the aroma aspect).

For example, in FIG8 , "CNN SIS Preds by LSTM" refers to predictions made by the LSTM on the set of sufficient input subsets produced by applying our SIScollection procedure on all examples x P X test for which f CNN pxq ě τ`.

4 Since the word embeddings are model-specific, we embed each SIS using the embeddings of the model making the prediction (note that while the embeddings are different, the vocabulary is the same across the models).In TAB1 , we show five example clusters (and cluster composition) resulting from clustering the combined set of all sufficient input subsets extracted by the LSTM and CNN on reviews in the test set for which a model predicts positive sentiment toward the aroma aspect.

The complete clustering on reviews receiving positive sentiment predictions is shown in TAB7 for reviews receiving negative sentiment predictions.

DISPLAYFORM0

For posterity, we include results here from repeating the analysis in our paper for the two other non-aroma aspects measured in the beer reviews data: appearance and palate.

FIG7 : Change in appearance prediction (f pxztiuq´f pxq) after masking a randomly chosen word with mean imputation or hot-deck imputation.

10,000 replacements were sampled from the appearance beer reviews training set.

FIG0 : Change in palate prediction (f pxztiuq´f pxq) after masking a randomly chosen word with mean imputation or hot-deck imputation.

10,000 replacements were sampled from the palate beer reviews training set.

Figure S19: Length of rationales for palate prediction FIG1 : Importance of individual features in beer review palate rationales S3 Details of the MNIST Analysis

The MNIST database of handwritten digits contains 60k training images and 10k test images BID7 .

All images are 28x28 grayscale, and we normalize them such that all pixel values are between 0 and 1.

We use the convolutional architecture provided in the Keras MNIST CNN example.

5 The architecture is as follows:1.

Input: (28 x 28 x 1) image, all values P r0, 1s 2.

Convolutional Layer 1: Applies 32 3x3 filters with ReLU activation 3.

Convolutional Layer 2: Applies 64 3x3 filters, with ReLU activation 4.

Pooling Layer 1: Performs max pooling with a 2x2 filter and dropout probability 0.25 5.

Dense Layer 1: 128 neurons, with ReLU activation and dropout probability 0.5 6.

Dense Layer 2: 10 neurons (one per digit class), with softmax activation The Adadelta optimizer BID8 is used to minimize cross-entropy loss on the training set.

The final model achieves 99.7% accuracy on the train set and 99.1% accuracy on the held-out test set.

Original image (class 9).

(c) SIS if backward selection were to terminate the first time prediction on remaining image drops below 0.7, corresponding to point C in (a) (CNN predicts class 9 with probability 0.700 on this SIS).

(d) Actual SIS produced by our FindSIS algorithm, corresponding to point D in (a) (CNN predicts class 9 with probability 0.704 on this SIS).

FIG1 demonstrates an example MNIST digit for which there exists a local minimum in the backward selection phase of our algorithm to identify the initial SIS.

Note that if we were to terminate the backward selection as soon as predictions drop below the decision threshold, the resulting SIS would be overly large, violating our minimality criterion.

It is also evident from FIG1 that the smaller-cardinality SIS in (d), found after the initial local optimum in (c), presents a more interpretable input pattern that enables better understanding of the core motifs influencing our classifier's decisions.

To avoid suboptimal results, it is important to run a complete backward selection sweep until the entire input is masked before building the SIS upward, as done in our SIScollection procedure.

To cluster SIS from the image data, we compute the pairwise distance between two SIS subsets S 1 and S 2 as the energy distance BID9 between two distributions over the image pixel coordinates that comprise the SIS, X 1 and X 2 P R 2 : DISPLAYFORM0 Here, X i is uniformly distributed over the pixels that are selected as part of the SIS subset S i , X 1 i is an i.i.d.

copy of X i , and ||¨|| represents the Euclidean norm.

Unlike a Euclidean distance between images, our usage of the energy distance takes into account distances between the similar pixel coordinates that comprise each SIS.

The energy distance offers a more efficiently computable integral probability metric than the optimal transport distance, which has been widely adopted as an appropriate measure of distance between images.

We set the threshold τ " 0.7 for SIS to ensure that the model is confident in its class prediction (probability of the predicted class is ě 0.7).

Almost all test examples initially have f pxq ě τ for the top class ( FIG1 ).

We identify all test examples that satisfy this condition and use SIS to identify all sufficient input subsets.

The number of sufficient input subsets per digit is shown in FIG1 .We apply our SIScollection algorithm to identify sufficient input subsets on MNIST test digits (Section 4.2).

Examples of the complete SIS-collection corresponding to randomly chosen digits are shown in FIG1 .

We also cluster all the sufficient input subsets identified for each class (Section 4.3), depicting the results in FIG1 .In FIG5 , we show an MNIST image of the digit 9, adversarially perturbed to 4, and the sufficient subsets corresponding to the adversarial prediction.

Although a visual inspection of the perturbed image does not really reveal exactly how it has been manipulated, it becomes immediately clear from the SIS-collection for the adversarial image.

These sets shows that the perturbation modifies pixels in such a way that input patterns similar to the typical SIS-collection for a 4 ( FIG4 ) become embedded in the image.

The adversarial manipulation was done using the Carlini-Wagner L 2 (CW2) attack 6 BID11 with a confidence parameter of 10.

The CW2 attack tries to find the minimal change to the image, with respect to the L 2 norm, that will lead the image to be misclassified.

It has been demonstrated to be one of the strongest extant adversarial attacks BID12 .

<|TLDR|>

@highlight

We present a method for interpreting black-box models by using instance-wise backward selection to identify minimal subsets of features that alone suffice to justify a particular decision made by the model.