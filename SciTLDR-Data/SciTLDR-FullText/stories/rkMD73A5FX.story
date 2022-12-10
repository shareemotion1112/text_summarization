Interactions such as double negation in sentences and scene interactions in images are common forms of complex dependencies captured by state-of-the-art machine learning models.

We propose Mahé, a novel approach to provide Model-Agnostic Hierarchical Explanations of how powerful machine learning models, such as deep neural networks, capture these interactions as either dependent on or free of the context of data instances.

Specifically, Mahé provides context-dependent explanations by a novel local interpretation algorithm that effectively captures any-order interactions, and obtains context-free explanations through generalizing context-dependent interactions to explain global behaviors.

Experimental results show that Mahé obtains improved local interaction interpretations over state-of-the-art methods and successfully provides explanations of interactions that are context-free.

State-of-the-art machine learning models, such as deep neural networks, are exceptional at modeling complex dependencies in structured data, such as text BID44 BID40 , images BID6 BID12 , and DNA sequences BID0 BID48 .

However, there has been no clear explanation on what type of dependencies are captured in the black-box models that perform so well BID30 .In this paper, we make one of the first attempts at solving this important problem through interpreting two forms of structures, i.e., context-dependent representations and context-free representations.

A context-dependent representation is the one in which a model's prediction depends specifically on a data instance level (such as a sentence or an image).

In order to illustrate the concept, we consider an example in image analysis.

A yellow round-shape object can be identified as the sun or the moon given its context, either bright blue sky or dark night.

A context-free representation is one where the representation behaves similarly independent of instances (i.e., global behaviors).

In a hypothetical task of classifying sentiment in sentences, each sentence carries very different meaning, but when "not" and "bad" depend on each other, their sentiment contribution is almost always positive -i.e., the structure is context-free.

To investigate context-dependent and context-free structure, we lend to existing definitions in interpretable machine learning BID29 BID13 .

A context-dependent interpretation is a local interpretation of the dependencies at or within the vicinity of a single data instance.

Conversely, a context-free interpretation is a global interpretation of how those dependencies behave in a model irrespective of data instances.

In this work, we study a key form of dependency: an interaction relationship between the prediction and input features.

Interactions can describe arbitrarily complex relationships between these variables and are commonly captured by state-of-the-art models like deep neural networks BID42 .

Interactions which are context-dependent or context-free are therefore local or global interactions, respectively.

We propose Mahé, a framework for explaining the context-dependent and context-free structures of any complex prediction model, with a focus on explaining neural networks.

The context-dependent explanations are built based on recent work on local intepretations (such as BID29 ).

Specifically, Mahé takes as input a model to explain and a data instance, and returns a hierarchical explanation, a format proposed by to show local group-variable relationships used in predictions (Figure 1 ).

To provide context-free Input into complex ML model:Step FORMULA0 Step FORMULA3 Step 3 Fit ( )

Hierarchical Explanation 0.6 none (linear LIME) this movie is not bad 0.8 { 4 , 5 } not bad 0.9 { 2 , 4 , 5 } movie not bad … … … { 4 , 5 } are interacting linear LIMEMah ƴ e (not, bad) Figure 1: An overview of the steps used to obtain a context-dependent hierarchical explanation.

Step 1 inputs a data instance of interest (e.g. a sentence) into a complex model, in this case a classifier.

Step 2 locally perturbs the data instance and obtains their predictions results from the model.

Instead of only fitting a linear model as linear LIME does to the perturbed samples and outputs, Mahé fits a neural network to them to learn the highly nonlinear decision boundary used to classify the instance.

The nonlinearity indicates that there should be an interaction between variables, and an interpretation of the neural network is used to extract the interactions BID42 .

Attribution scores of those interactions can then be shown for the data instance, as displayed in Step 3.explanations, Mahé generalizes those context-dependent interactions with consistent behavior in a model and determines whether a local representation in the model is responsible for the global behavior.

In this case, Mahé takes as input a model and representative data corresponding to an interaction of interest and returns whether or not that interaction is context-free.

We conduct experiments on both synthetic datasets and real-world application datasets, which shows that Mahé's context-dependent explanations can significantly outperform state-of-the-art methods for local interaction interpretation, and Mahé is capable of successfully finding context-free explanations of interactions.

In addition, we identify promising cases where the methodology for context-free explanations can successfully edit models.

Our contributions are as follows: 1) Mahé achieves the task of improved context-dependent explanations based on interaction detection and fitting performance and model-agnostic generality, compared to state-of-the-art methods for local interaction interpretation, 2) Mahé is the first to provide context-free explanations of interactions in deep learning models, and 3) Mahé provides a promising direction for modifying context-free interactions in deep learning models without significant performance degradation.

Attribution Interpretability: A common form of interpretation is feature attribution, which is concerned with how features of a data instance contribute to a model output.

Within this category, there are two distinct approaches: additive and sensitivity attribution.

Additive attribution interprets how much each feature contributes to the model output when these contributions are summed.

In contrast, sensitivity attribution interprets how sensitive a model output is to changes in features.

Examples of additive attribution techniques include LIME BID29 and CD .

Examples of sensitivity attribution methods include Integrated Gradients BID39 , DeepLIFT BID34 , and SmoothGrad BID36 .

Unlike previous approaches, Mahé provides additive attribution interpretations that consist of non-additive groups of variables (interactions) in addition to the normal additive contributions of each variable.

Interaction Interpretability: An interaction in its generic form is a non-additive effect between features on an outcome variable.

Only until recently has there been development in interpreting nonadditive interactions despite often being learned in complex machine learning models.

The difficulty interpreting non-additive interactions stems from their lack of exact functional identity compared to, for example, a multiplicative interaction.

Methods that exist to interpret non-additive interactions are NID BID42 and Additive Groves BID38 .

In contrast, many more methods exist to interpret specific interactions, namely multiplicative ones.

Notable methods include CD , Tree-Shap (Lundberg & Lee, 2017) , and GLMs with multiplicative interactions BID27 .

Unlike previous methods, our approach provides local interpretations of the more challenging non-additive interaction.

Locally Interpretable Model-Agnostic Explanations (LIME): LIME BID29 ) is a very popular type of model interpretation.

Its popularity comes from additive attribution interpretations to explain the output of any prediction model.

The original and most popular version of LIME uses a linear model to approximate model predictions in the local vicinity of a data instance.

Since its introduction, variants of LIME have been proposed, for example Anchors (Ribeiro et al., 2018) and LIME-SUP BID11 .

While Anchors generates a form of context-free explanation, its method of selecting fully representative features for a prediction does not consider interactions.

For example, Anchors assumes that (not, bad) "virtually guarentees" a sentiment prediction to be positive, whereas in Mahé this is not necessarily true; only their interaction is positive (See Table 6 for an example).

LIME-SUP touches upon interactions but does not study their interpretation.

Let f (·) be a target function (model) of interest, e.g. a classifier, and φ(·) be a local approximation of f and is interpretable in contrast to f .

A common choice for φ is a linear model, which is interpretable in each linear term.

Namely, for an data instance x ∈ R p , weights w ∈ R p and bias b, interpretations are given by w i x i , known as additive attributions (Lundberg & Lee, 2017) from DISPLAYFORM0 Given a set of n data points {x DISPLAYFORM1 ))} will accurately fit to the functional surface of f at the data instance, such that φ(x) = f (x).

Because it is possible in such scenarios that φ(x) = f (x) ≈ b, there must be some nonzero distances between x and x (i) to obtain informative attribution scores.

LIME, as it was originally proposed, uses a linear approximation as above where samples are generated in a nonzero local vicinity of x BID29 .

The drawback of linear LIME is that there is often an error = |f (x) − φ(x)| > 0.For complex models f , the functional surface at x can be nonlinear.

Because D consists of x (i) with distance d > 0 from x, a closer fit to f (x) in its nonlinear vicinity, i.e. {f ( DISPLAYFORM2 , can be achieved with the following generalization of Eq. 1: DISPLAYFORM3 where g i (·) can be any function, for example one that is arbitrarily nonlinear.

This function is called a generalized additive model (GAM) BID5 , and now attribution scores can be given by g i (x i ) for each feature i.

For the purposes of interpreting individual feature attribution, the GAM may be enough.

However, if we would like broader explanations, we can also obtain nonadditive attributions or interactions between variables BID17 , which can provide an even better fit to the complex local vicinity.

Expanding Eq. 2 with interactions yields: DISPLAYFORM4 where g i (·) can again be any function, x I ∈ R |I| are interacting variables corresponding to the variable indices I, and DISPLAYFORM5 is a set of K interactions.

Attribution scores are now generated from both g i and g i .

In this paper, we learn g i and g i using Multilayer Perceptrons (MLPs).

φ or φ K can be converted to classification by applying a sigmoid function.

Adding non-additive interactions, I, that are truly present in the local vicinity increases the representational capacity of φ K (x).

I corresponds to non-additive interacting features if and only if g (·) (Eq. 3) cannot be decomposed into a sum of |I| arbitrary subfunctions δ, each not depending on a corresponding interacting variable BID42 , i.e. DISPLAYFORM6 4 Mahé FRAMEWORK In this section, we introduce our Mahé framework, which can provide context-dependent and context-free explanations of interactions.

To provide context-dependent explanations, we propose to use a two-step procedure that first identifies what variables interact locally, then learns a model of interactions (as Eq. 3) to provide a local interaction score at the data instance in question.

The procedure of first detecting interactions then building non-additive models for them has been studied previously BID17 BID42 ; however, previous works have not focused on using the same non-additive models to provide local interaction attribution scores, which enable us to visualize interactions of any size as demonstrated later in §5.2.3.

Local Interaction Detection: To perform interaction detection on samples in the local vicinity of data instance x, we first sample n points in the -neighborhood of x with a maximum neighborhood distance under a distance metric d. While the choice of d depends on the feature type(s) of x, we always set = σ, i.e. one standard deviation from the mean of a Gaussian weighted sampling kernel.

When all features are continuous, neighborhood points are sampled with mean x ∈ R p and d = 2 to generate x (1) , . . .

, DISPLAYFORM0 where N is a normal distribution truncated at .

When features are categorical, they are converted to one-hot binary representation.

For x of binary features, we sample each point around x by first selecting a number of random features to flip (or perturb) from a uniform distribution between 0 and min(p, ).

The max number of flips is derived from for a distance metric that is usually cosine distance BID29 .

Distances between local samples and x are then weighted by a Gaussian kernel to become sample weights (e.g. the frequency each sample appears in the sampled dataset).1 For context-dependent explanations, the exact choice of σ depends on the stability and interaction orders of explanations.

The interaction orders may become too large and uninformative because the local vicinity area covers too much complex representation from f (·).

Thus we recommend tuning σ to the task at hand.

Our framework is flexible to any interaction detection method that applies to the dataset DISPLAYFORM1 Since we seek to detect non-additive interactions, we use the neural interaction detection (NID) framework BID42 , which interprets learned neural network weights to obtain interactions.

To the best of our knowledge, this detection method is the only polynomial-time algorithm that accurately ranks any-order non-additive interactions after training one model, compared to alternative methods that must train an exponential number O(2 p ) of models.

The basic idea of NID is to interpret an MLP's accurate representation of data to accurately identify the statistical interactions present in this data.

Because MLPs learn interactions at nonlinear activation functions, NID performs feature interaction detection by tracing high-strength 1 -regularized weights from features to common hidden units.

In particular, NID efficiently detects any-order interactions by first assuming each first layer hidden unit in a trained MLP captures at most one interaction, then NID greedily identifies these interactions and their strengths through a 2D traversal over the MLP's input weight matrix, W ∈ R p×h .

The result is that instead of testing for interactions by training O(2 p ) models, now only O(1) models and O(ph) tests are needed.

In addition to its efficiency, applying NID to our framework Mahé has several advantages.

One is the universal approximation capabilities of MLPs BID10 , allowing them to approximate arbitrary interacting functions in the potentially complex local vicinity of f (x).

Another advantage is the independence of features in the sampled points of D. Normally, interaction detection methods cannot identify high interaction strengths involving a feature that is correlated with others because interaction signals spread and weaken among correlated variables BID38 .

Without facing correlations, NID can focus more on interpreting the data-generating function, the target model f .

One disadvantage of our application of NID is the curse of dimensionality for MLPs when p is large (e.g. p > n) BID41 , which is oftentimes the case for images.

In general, large input dimensions should be reduced as much as possible to avoid overfitting.

For images, p is normally reduced in model-agnostic explanation methods by using segmented aggregations of pixels called superpixels as features BID29 BID18 BID30 .

Upon obtaining an interaction ranking from NID, GAMs with interactions (Eq. 3) can be learned for different top-K interactions ranked by their strengths BID42 .

In the Mahé framework, there are L + 1 different levels of a hierarchical explanation which constitutes our context-dependent explanation, where L is the number of levels with interaction explanations, and K = L at the last level.

When presenting the hierarchy such as Figure 1 Step 3, the first level shows the additive attributions of individual features from by a trained φ(·) in Eqs. 1 or 2, such as the explanation from linear LIME.

Subsequently, the parameters w of φ(·; w, b) are frozen before interaction models are added to construct φ K (·) in Eq. 3.

The next levels of the hierarchy can be presented either as the interaction attribution of g K (·) as in Figure 1 or 1 In cases where features are a mixture of continuous and one-hot categorical variables, a way of sampling points is to adapt the approach for binary features to handle the mixture of feature types BID29 .

The main difference now is that continuous features are drawn from a uniform distribution truncated at σ and are standard scaled to have similar magnitudes as the binary features.

Since continuous features are present, d can be 2 distance, then a Gaussian kernel can be applied to sample distances as before.

DISPLAYFORM0 The practice of training interaction models g i on the residual of φ is used to prevent degeneracy of univariate functions in φ in the presence of any overlapping interaction functions BID17 .

Since φ K is trained at each hierarchical level on D, the fit of each φ K can also be explained via predictive performance, such as R 2 performance in Figure 1 Step 3.

The stopping criteria for the number of hierarchical levels can depend on the predictive performance or user preference.

In order to provide context-free explanations, we propose determining whether the local interactions assumed to be context-dependent in §4.1 can generalize to explain global behavior in f .

To this end, we first define ideal conditions for which a generic local explanation can generalize.

For choosing distance metric d and sampling points in the local vicinity of x, please refer to §4.1 and our considerations for generalizing explanations at the end of this section.

Definition 1 (Generalizing Local Explanations).

Let f (·) be the model output we wish to explain, and X f be the data domain of f .

Let a local explanation of f at x ∈ X f be some explanation E that is true for f (x) and depends on samples x ∈ X f that are only in the local vicinity of x, i.e. d(x, x ) ≤ provided a distance metric d and distance ≥ 0.

The local explanation E is a global explanation if the following two conditions are met: 1) Explanation E is true for f at all data samples in X f , including samples outside the local vicinity of x, i.e. all samples DISPLAYFORM0 2) There exists a sample x ∈ X f and a local modification to f (x ) (modifying f (x ) in the vicinity d(x , x ) ≤ ) that changes E for all samples in X f while still meeting condition 1).For example, consider a simple linear regression model we wish to explain, f (x) = w 1 x 1 + w 2 x 2 .

Let its local explanation be the feature attributions w 1 x 1 and w 2 x 2 .

This local explanation is a global explanation because 1) for all values of x 1 and x 2 , the feature attributions are still w 1 x 1 and w 2 x 2 , and 2) if any of the weights are changed, e.g. w 1 → w 1 , the attribution explanation will change, but the feature attributions are still w 1 x 1 and w 2 x 2 for all values of x 1 and x 2 .Our context-free explanation of interaction I is:

whenever local interaction I exists, its attribution will in general have the same polarity (or sign).

Since it is impossible to empirically prove that a local explanation is true for all data instances globally (via Definition 1), this work is focused on providing evidence of context-free interactions.

This evidence can be obtained by checking whether our explanation is consistent with the two conditions from Definition 1 for the interaction of interest I: 1) For representative data instances in the domain of f , if local interaction I exists, does it always have the same attribution polarity?

The representative data instances should be separated from each other at an average distance beyond .

2) Can local interaction I at a single data instancex be used to negate I's attribution polarity for all representative data instances where I exists?The advantage of checking the response of f to local modification is determining if consistent explanations across data instances are more than just coincidence.

This is especially important when only a limited number of data instances are available to test on.

We propose to modify an interaction attribution of the model's output f (x) at data instance x by utilizing a trained model g k (x I ) of interaction I k , where 1 ≤ k ≤ K (Eq. 3).

Letg k (·) be a modified version of g k (·).

We can then define a modified form of Eq. 3: DISPLAYFORM1 Without retrainingφ k (·), we useφ k and the same local vicinity {x DISPLAYFORM2 Finally, we can modify the interaction attribution of f (x) by fine-tuning f (·) on datasetD. In this paper, we modify interactions by negating them:g k (·) = −cg k (·), where −c negates the interaction attribution with a specified magnitude c.

How can modifying a local interaction affect interactions outside its local vicinity?

This would suggest that the manifold hypothesis is true for f (·)'s representations of these interactions FIG1 .

The manifold hypothesis states that similar data lie near a lowdimensional manifold in a high-dimensional space BID43 BID15 BID3 .

Studies have suggested that the hypothesis applies to the data representations learned by neural networks BID31 BID1 .

The hypothesis is frequently used to visualize how deep networks represent data clusters BID21 BID14 , and it has been applied to representations of interactions BID28 , but not for neural networks.

Part of our objective is to generalize our explanation as much as possible.

In the case of languagerelated tasks, we additionally generalize based on our meaning of a local interaction and the distance metric we use, d. In this paper, local interactions for language tasks do not have word interactions fixed to specific positions; instead, these interactions are only defined by the words themselves (the interaction values) and their positional order.

For example, the ("not", "bad") interaction would match in the sentences: "this is not bad" and "this does not seem that bad".

For comparing texts and measuring vicinity sizes, we use edit distance BID16 , which allows us to compare sentences with different word counts.2 Although we define distance metrics for each domain ( §5.1), we found that our results were not very sensitive to the exact choice of valid distance metric.

We evaluate the effectiveness of Mahé first on synthetic data and then on four real-world datasets.

To evaluate context-dependent explanations of Mahé, we first evaluate the accuracy of Mahé at local interaction detection and modeling on the outputs of complex base models trained on synthetic ground truth interactions.

We compare Mahé to Shap-Tree (Lundberg et al., 2018), ACD-MLP , and ACD-LSTM , which are local interaction modeling baselines for the respective models they explain: XGBoost BID4 , multilayer perceptrons (MLP), and long short-term memory networks (LSTM) BID7 .

Synthetic datasets have p = 10 features (Table 2) .

In all other experiments, we study Mahé's explanations of state-of-the-art level models trained on real-world datasets.

The state-of-the-art models are: 1) DNA-CNN, a 2-layer 1D convolutional neural network (CNN) trained on MYC-DNA binding data 3 BID24 BID47 BID0 BID48 BID46 , 2) Sentiment-LSTM, a 2-layer bi-directional LSTM trained on the Stanford Sentiment Treebank (SST) BID37 BID40 , 3) ResNet152, an image classifier pretrained on ImageNet '14 BID32 BID6 , and 4) Transformer, a machine translation model pretrained on WMT-14 En→ Fr BID44 BID26 .

Avg.

p for our context-dependent evaluations, similar to our context-free tests, are shown in TAB0 .The following hyperparameters are used in our experiments.

We use n = 1k local-vicinity samples in D for synthetic experiments and n = 5k samples for experiments explaining models of real-world datasets, with 80%-10%-10% train-validation-test splits to train and evaluate Mahé.

The distance metrics for vicinity size are: 2 distance for synthetic experiments, cosine distance for DNA-CNN and ResNet152, and edit distance for Sentiment-LSTM and Transformer.

We use on-off superpixel and word approaches to binary feature representation for explaining ResNet152 and Sentiment-LSTM respectively BID29 BID18 , and the other experiments for real-world datasets use perturbation distributions that randomly perturbs features to belong to the same categories of original features, as in BID30 .The superpixel segmenter we use is quick-shift BID45 BID29 .For the hyperparameters of the neural networks in Mahé, we use MLPs with 50-30-10 first-tolast hidden layer sizes to perform interaction detection in the NID framework BID42 .

These MLPs are trained with 1 regularization λ 1 = 5e−4.

The learning rate used is always 5e−3 except for Transformer experiments, whose learning rate of 5e−4 helped with interaction detection under highly unbalanced output classes.

The MLP-based interaction models in the GAM (Eq. 3) always have architectures of 30-10.

They are trained with 2 regularization of λ 2 = 1e−5 and learning rate of 1e−3.

Because learning GAMs can be slow, we make a linear approximation of the univariate functions in Eq. 3, such that g i (x i ) = x i .

This approximation also allows us to make direct comparisons between Mahé and linear LIME, since x i is exactly the linear part (Eq. 1).

All neural networks train with early stopping, and Level L + 1 is decided where validation performance does not improve more than 10% with a patience of 2 levels.

c ranges from 3 to 4 in our experiments.

Table 2 : Data generating functions with interactions

In order to evaluate Mahé's context-dependent explanations, we first compare them to state-of-the-methods for local interaction interpretation.

A standard way to evaluate the accuracy of interaction detection and modeling methods has been to experiment on synthetic data because ground truth interactions are generally unknown in real-world data BID8 BID38 BID17 BID42 .

Similar to BID9 , we evaluate interactions in a subset region of a synthetic function domain.

We generate synthetic data using functions F 1 − F 4 (Table 2 ) with continuous features uniformly distributed between −1 to 1, train complex base models (as specified in §5.1) on this data, and run different local interaction interpretation methods on 10 trials of 20 data instances at randomly sampled locations on the synthetic function domain.

Between trials, base models with different random initializations are trained to evaluate the stability of each interpretation method.

We evaluate how well each method fits to interactions by first assuming the true interacting variables are known, then computing the Mean Squared Error (MSE) between the predicted interaction attribution of each interpretation method and the ground truth at 1000 uniformly drawn locations within the local vicinity of a data instance, averaged over all randomly sampled data instances and trials ( FIG2 .

We also evaluate the interaction detection performance of each method by comparing the average R-precision BID22 of their interaction rankings across the same sampled data instances FIG2 ).

R-precision is the percentage of the top-R items in a ranking that are correct out of R, the number of correct items.

Since F 1 − F 4 only ever have 1 ground truth interaction, R is always 1.

Compared to Shap-Tree, ACD-MLP, and ACD-LSTM, the Mahé framework is the only one capable of detection and fitting, and it is the only model-agnostic approach.

In this section, we demonstrate our approaches to evaluating Mahé's context-dependent explanations on real-world data.

We first evaluate the prediction performance of Mahé on the test set of D as interactions are added in Eq. 3, i.e. K increases.

For a given value of σ, we run Mahé 10 times on each of 40 randomly selected data instances from the test sets associated with DNA-CNN, Sentiment-LSTM, and ResNet152.

For Transformer, performance is examined on a specific grammar (cet) translation, to be detailed in §5.3.

The local vicinity samples and model initializations in Mahé are randomized in every trial.

We select the σ that gives the worst performance for Mahé at K = L in each base model, out of σ = 0.4σ , 0.6σ , 0.8σ , and 1.0σ , where σ is the average pairwise distance between data instances in respective test sets.

Results are shown in TAB2 for K starting from 0, which is linear LIME, and increasing to the last hierarchical level L. An alternative approach to evaluating Mahé is to determine out of LIME and Mahé explanations, could human evaluators prefer Mahé explanations?

We recruit a total of 60 Amazon Mechanical Turk users to participate in comparing explanations of Sentiment-LSTM predictions.

While the presented LIME explanations are standard, we adjust Mahé to only show the K = 1 interaction and merge its attribution with subsumed features' attributions to make the difference between LIME and Mahé subtle FIG3 ).

We present evaluators with explanations for randomly selected test sentences under the main condition that these sentences must have at least one detected interaction, which is the case for > 95% of sentences.

In total, there are explanations for 40 sentences, each of which is examined by 5 evaluators, and a majority vote of their preference is taken.

Each evaluator is only allowed to pick between explanations for a maximum of 4 sentences.

Please see Appendix B for additional conditions used to select sentences for evaluators and more examples like FIG3 .

The result of this experiment is that the majority of preferred explanations (65%, p = 0.029) is with interactions, supporting their inclusion in hierarchical explanations.

Examples of context-dependent hierarchical explanations for ResNet152, Sentiment-LSTM, and Transformer are shown in FIG6 , Table 6 , and Appendix E respectively after page 9.

For the image explanations in FIG6 , superpixels belonging to the same entity often interact to support its prediction.

One interesting exception is FIG6 ) because water is not detected as an important interaction with buffalo in the prediction of water buffalo.

This could be due to various reasons.

For example, water may not be a discriminatory feature because there are a mix of training images of water buffalo in ImageNet with and without water.

The same is true for related classes like bison.

Explanations may also appear unintuitive when a model misbehaves.

Therefore, quantitative validations, such as the predictive performance of adding interactions in each hierarchical level (e.g. R 2 scores in FIG6 ), can be critical for trusting explanations.

In this section, we show examples of context-free explanations of interactions found by Mahé.

We first study the context-free interactions learned by Sentiment-LSTM.

To have enough sentences for this evaluation, we use data from IMDB movie reviews BID20 in addition to the test set of SST.

Based on our results FIG5 ), we observe that the polarities of certain local interactions are almost always the same, where the words of matching interactions can be separated by any number of words in-between.

To ensure that this global behavior is not a coincidence, we modify local interaction behavior in Sentiment-LSTM to check for a global change in this behavior ( §4.2).

As a result, when the model's local interaction attribution at a single data instance is negated, the attribution is almost always the opposite sign for the rest of the sentences.

A notable insight about Sentiment-LSTM is that it appears to represent (too, bad) and (only, worse) as globally positive sentiments, and Mahé's modification in large part rectifies this misbehavior ( FIG5 ).

The modifications to Sentiment-LSTM only cause an average reduction of 1.5% test accuracy, indicating that the original learned representation stays largely intact.

Results for σ = 16 are shown with the average pairwise edit distance between sentences being σ = 24.8.

Words in detected interactions are separated by 1.3 words on average.

Next, we study the possibility of identifying context-free interactions in Transformer on a known form of interaction in English-to-French translations: translations into a special French word for "this" or "that", cet, which only appears when the noun it modifies begins with a vowel.

Some examples of cet interactions are (this, event), (this, article), and (this, incident), whose nouns have the same starting vowels in French.

For our explanation task, the presence of cet in a translation is used as a binary prediction variable for local interaction extraction.

To minimize the sources of cet, we limit original sentence lengths to 15 words, and we perform translations on WikiText-103 BID23 to evaluate on enough sentences.

The results of context-free experiments on cet interactions of adjacent English words are shown in TAB3 .

The interactions always have positive polarities towards cet, and after modifying Transformer at a single data instance for a given interaction, its polarity almost always become negative, just like the context-free interactions in Sentiment-LSTM.

Examples of new translations from the modified Transformer are shown in the "after" rows in TAB4 , where cet now disappears from the translations.

The test BLEU score of Transformer only decreases by an average percent difference of −2.7% from modification, which is done through differentiating the max value of cet output neurons over all translated words.

Results for σ = 6, σ = 10.5 are shown.

Experiments on DNA-CNN and ResNet152 show similar results at fixed interaction positions ( §4.2).

For DNA-CNN, out of the 94 times a 6-way interaction of the CACGTG motif BID33 was detected in the test set, every time yielded a positive attribution polarity towards DNA-protein affinity, and the same was true after modifying the model in the opposite polarity (cosine distance σ = 0.35, σ = 0.408).

For ResNet152, context-free interactions are also found (cosine distance σ = 0.4, σ = 0.663).

However, because superpixels are used, the interactions found may contain artifacts caused by superpixel segmenters, yielding less intuitive interactions (see Appendix A).

Although Mahé obtains accurate local interactions on synthetic data using NID, there is no guarantee that NID finds correct interactions.

Mahé faces common issues of model-agnostic perturbation methods in interpreting high-dimensional feature spaces, choice of perturbation distribution, and speed BID29 .

Finally, an exhaustive search is used for context-free explanations.

In this work, we proposed Mahé, a model-agnostic framework of providing context-dependent and context-free explanations of local interactions.

Mahé has demonstrated the capability of outperforming existing approaches to local interaction interpretation and has shown that local interactions can be context-free.

In future work, we wish to make the process of finding context-free interactions more efficient, and study to what extent model behavior can be changed by editing its interactions or univariate effects.

Finally, we would like to study the interpretations provided by Mahé more closely to find new insights into structured data.

Table 6 : Examples of context-dependent hierarchical explanations on Sentiment-LSTM.

The interaction attribution of g K (·) is shown at each K − 1 level, K ≥ 1 ( §4.1) in color.

Green means positively contributing to sentiment, and red the opposite.

Visualized attributions of linear LIME and Mahé are normalized to the max attribution magnitudes (max magn.) shown.

Top-5 attributions by magnitude are shown for LIME.

Besides requiring detected interactions, several other conditions were used to choose sentences for Mechanical Turk evaluators.

We ensure that there is a significant attribution difference between LIME and Mahé by only choosing among sentences that have a polarity difference between Mahé's interaction and LIME's corresponding linear attributions.

To reduce ambiguities of uninterpretable explanations arising from a misbehaving model -an issue also faced by BID39 in interpretation evaluation -we only show explanations of sentences that the model classified correctly.

We also attempt to limit the effort that evaluators need to analyze explanations by only showing sentences with 5-12 words with uniform representation of each sentence length.

An example of the interface that evaluators select from is shown in Figure 8 .

Figure 9 shows randomly selected examples that evaluators analyze.

The visualization tool for presenting additive attribution explanations is graciously provided by the official code repository of LIME 4 .Figure 8: Example of Mechanical Turk interface used by workers to select between explanations provided by linear LIME and Mahé.

C RUNTIME Figure 10 : Average runtime of linear LIME versus Mahé on context-dependent explanations.

Runtimes for experiments in TAB2 are shown.

"local inference" is the runtime for sampling in the local vicinity of a data instance and running inference though a black-box model for every sampled point.

"NID" is the runtime for running NID interaction detection.

"linear model" is the runtime for training a linear model (Eq. 1) to get linear attributions with LIME.

"interaction model(s)" is the runtime for sequentially training interaction models (Eq. 3) to get interaction attributions with Mahé.

Figure 11: Runtime of Mahé for determining whether an interaction is context-free for a randomly selected interaction and 40 different contexts, run sequentially.

Runtimes for checking interaction consistency before and after model retraining (fine-tuning) are shown, resulting in tests on 80 contexts total.

DNA-CNN takes longer here because we needed to relax the cutoff criteria of identifying the last hierarchical level to find the CACGTG interaction.

For context-free experiments, a cutoff patience ( §5.1) for Sentiment-LSTM, ResNet152, and Transformer was not needed in our experiments and is excluded in this runtime analysis.

The patience for DNA-CNN was 2.

FIG5 .

The baselines are GLM and GA 2 M. GLM is a lasso-regularized generalized linear model with all pairs of multiplicative interaction terms BID2 , and GA 2 M is a tree-based generalized additive model with pairwise non-additive interactions BID17 .

Mahé shows more significant improvements over baselines than before negating interactions.

Table 7 : Examples of context-dependent hierarchical explanations on Transformer.

The interaction attribution of g K (·) is shown at each K − 1 level, K ≥ 1 ( §4.1) in color.

Green contributes towards cet translations, and red contributes the opposite.

Visualized attributions of linear LIME and Mahé are normalized to the max attribution magnitudes (max magn.) shown.

Top-5 attributions by magnitude are shown for LIME.

@highlight

A new framework for context-dependent and context-free explanations of predictions

@highlight

The authors extend the linear local attribution method LIME for interpreting black box models, and propose a method to discern between context-dependent and context-free interactions.

@highlight

A method that can provide hierarchical explanations for a model, including both context-dependent and context-free explanations by a local interpretation algorithm.