Instancewise feature scoring is a method for model interpretation, which yields, for each test instance, a vector of importance scores associated with features.

Methods based on the Shapley score have been proposed as a fair way of computing feature attributions, but incur an exponential complexity in the number of features.

This combinatorial explosion arises from the definition of Shapley value and prevents these methods from being scalable to large data sets and complex models.

We focus on settings in which the data have a graph structure, and the contribution of features to the target variable is well-approximated by a graph-structured factorization.

In such settings, we develop two algorithms with linear complexity for instancewise feature importance scoring on black-box models.

We establish the relationship of our methods to the Shapley value and a closely related concept known as the Myerson value from cooperative game theory.

We demonstrate on both language and image data that our algorithms compare favorably with other methods using both quantitative metrics and human evaluation.

Although many black box machine learning models, such as random forests, deep neural networks, and kernel methods, can produce highly accurate prediction in many applications, such prediction often comes at the cost of interpretability.

Ease of interpretation is a crucial criterion when these tools are applied in areas such as medicine, financial markets, and criminal justice; for more background, see the discussion paper by Lipton (2016) as well as references therein.

In this paper, we study instancewise feature importance scoring as a specific approach to the problem of interpreting the predictions of black-box models.

Given a predictive model, such a method yields, for each instance to which the model is applied, a vector of importance scores associated with the underlying features.

The instancewise property means that this vector, and hence the relative importance of each feature, is allowed to vary across instances.

Thus, the importance scores can act as an explanation for the specific instance, indicating which features are the key for the model to make its prediction on that instance.

There is now a large body of research focused on the problem of scoring input features based on the prediction of a given instance (see, e.g., Shrikumar et al., 2017; BID0 Ribeiro et al., 2016; Lundberg & Lee, 2017; Štrumbelj & Kononenko, 2010; BID1 BID4 Sundararajan et al., 2017) .

Of most relevance to this paper is a line of recent work (Štrumbelj & Kononenko, 2010; Lundberg & Lee, 2017; BID4 ) that has developed methods for model interpretation based on Shapley value (Shapley, 1953) from cooperative game theory.

The Shapley value was originally proposed as an axiomatic characterization of a fair distribution of a total surplus from all the players, and can be applied to predictive models, in which case each feature is modeled as a player in the underlying game.

While the Shapley value approach is conceptually appealing, it is also computationally challenging: in general, each evaluation of a Shapley value requires an exponential number of model evaluations.

Different approaches to circumventing this complexity barrier have been proposed, including those based on Monte Carlo approximation (Štrumbelj & Kononenko, 2010; BID4 and methods based on sampled least-squares with weights (Lundberg & Lee, 2017) .In this paper, we take a complementary point of view, arguing that the problem of explanation is best approached within a model-based paradigm.

In this view, explanations are cast in terms of a model, which may or may not be the same model as used to fit the data.

Criteria such as Shapley value, which are intractable to compute when no assumptions are made, can be more effectively computed or approximated within the framework of a model.

We focus specifically on settings in which a graph structure is appropriate for describing the relations between features in the data (e.g., chains for sequences and grids for images), and distant features according to the graph have weak interaction during the computation of Shapley values.

We propose two methods for instancewise feature importance scoring in this framework, which we term L-Shapley and C-Shapley; here the abbreviations "L" and "C" refer to "local" and "connected," respectively.

By exploiting the underlying graph structure, the number of model evaluations is reduced to linear-as opposed to exponential-in the number of features.

We demonstrate the relationship of these measures with a constrained form of Shapley value, and we additionally relate C-Shapley with another solution concept from cooperative game theory, known as the Myerson value (Myerson, 1977) .

The Myerson value is commonly used in graph-restricted games, under a local additivity assumption of the model on disconnected subsets of features.

Finally, we apply our feature scoring methods to several state-of-the-art models for both language and image data, and find that our scoring algorithms compare favorably to several existing sampling-based algorithms for instancewise feature importance scoring.

We begin by introducing some background and notation for instancewise feature importance scoring and the Shapley value.

We are interested in studying models that are trained to perform prediction, taking as input a feature vector x ∈ X ⊂ R d and predicting a response or output variable y ∈ Y.

We assume access to the output of a model via a conditional distribution, denoted by P m (·|x), that provides the distribution of the response Y ∈ Y conditioned on a given vector X = x of inputs.

For any given subset S ⊂ {1, 2, . . .

, d}, we use x S = {x j , j ∈ S} to denote the associated sub-vector of features, and we let P m (Y | x S ) denote the induced conditional distribution when P m is restricted to using only the sub-vector x S .

In the corner case in which S = ∅, we define P m (Y | x ∅ ) : = P m (Y ).

In terms of this notation, for a given feature vector x ∈ X , subset S and fitted model distribution P m (Y | x), we introduce the importance score v x (S) : = E m − log 1 DISPLAYFORM0 where E m [· | x] denotes the expectation over P m (· | x).

The importance score v x (S) has a codingtheoretic interpretation: it corresponds to the negative of the expected number of bits required to encode the output of the model based on the sub-vector x S .

It will be zero when the model makes a deterministic prediction based on x S , and larger when the model returns a distribution closer to uniform over the output space.

There is also an information-theoretic interpretation to this definition of importance scores, as discussed in BID2 .

In particular, suppose that for a given integer k < d, there is a function x → S * (x) such that, for all almost all x, the k-sized subset S * (x) maximizes v x (S) over all subsets of size k; then we are guaranteed that the mutual information I(X S * (X) , Y ) between X S * (X) and Y is maximized, over any conditional distribution that generates a subset of size k given X. The converse is also true.

In many cases, class-specific importance is favored, where one is interested in seeing how important a feature subset S is to the predicted class, instead of the prediction as a conditional distribution.

In order to handle such cases, it is convenient to introduce the degenerate conditional distribution P m (y | x) : = 1 if y ∈ arg max y P m (y | x), 0 otherwise.

We can then define the importance of a subset S with respect toP m using the modified score v x (S) : =Ê m − log 1 DISPLAYFORM1 which is the expected log probability of the predicted class given the features in S.Estimating the conditional distribution: In practice, we need to estimate-for any given feature vectorx ∈ X -the conditional probability functions P m (y |x S ) based on observed data.

Past work has used one of two approaches: either estimation based on empirical averages (Štrumbelj & Kononenko, 2010) , or plug-in estimation using a reference point BID4 Lundberg & Lee, 2017) .

In this approach, we first draw a set of feature vector {x j } M j=1 by sampling with replacement from the full data set.

For each sample x j , we define a new vector x j ∈ R d with components (x j ) i equal to x j i if i ∈ S andx i otherwise.

Taking the empirical mean of P m (y |x j ) over {x j } then provides an estimate of P m (y |x S ).

In this approach, the first step is to specify a reference vector x 0 ∈ R d is specified.

We then define the vectorx ∈ R d with components (x) i equal to x i if i ∈ S and x 0 iotherwise.

Finally, we use the conditional probability P m (y |x) as an approximation to P m (y |x S ).The plug-in estimate is more computationally efficient than the empirical average estimator, and works well when there exist appropriate choices of reference points.

We use this method for our experiments, where we use the index of padding for language data, and the average pixel strength of an image for vision data.

Consider the problem of quantifying the importance of a given feature index i for feature vector x. A naive way of doing so would be by computing the importance score v x ({i}) of feature i on its own.

However, doing so ignores interactions between features, which are likely to be very important in applications.

As a simple example, suppose that we were interested in performing sentiment analysis on the following sentence: It is not heartwarming or entertaining.

It just sucks.( ) This sentence is contained in a movie review from the IMDB movie data set (Maas et al., 2011) , and it is classified as negative sentiment by a machine learning model to be discussed in the sequel.

Now suppose we wish to quantify the importance of feature "not" in prediction.

The word "not" plays an important role in the overall sentence as being classified as negative, and thus should be attributed a significant weight.

However, viewed in isolation, the word "not" has neither negative nor positive sentiment, so that one would expect that v x ({"not"}) ≈ 0.Thus, it is essential to consider the interaction of a given feature i with other features.

For a given subset S containing i, a natural way in which to assess how i interacts with the other features in S is by computing the difference between the importance of all features in S, with and without i. This difference is called the marginal contribution of i to S, and given by DISPLAYFORM0 (1) In order to obtain a simple scalar measure for feature i, we need to aggregate these marginal contributions over all subsets that contain i. The Shapley value (Shapley, 1953) is one principled way of doing so.

For each integer k = 1, . . .

, d, we let S k (i) denote the set of k-sized subsets that contain i. The Shapley value is obtained by averaging the marginal contributions, first over the set S k (i) for a fixed k, and then over all possible choices of set size k: DISPLAYFORM1 Since the model P m remains fixed throughout our analysis, we frequently omit the dependence of φ x on P m , instead adopting the more compact notation φ x (i).The concept of Shapley value was first introduced in cooperative game theory (Shapley, 1953) , and it has been used in a line of recent work on instancewise feature importance ranking (Štrumbelj & Kononenko, 2010; BID4 Lundberg & Lee, 2017) .

It can be justified on an axiomatic basis (Shapley, 1953; Young, 1985) as being the unique function from a collection of 2 d numbers (one for each subset S) to a collection of d numbers (one for each feature i) with the following properties: (i) [Additivity]

The sum of the Shapley values DISPLAYFORM2 Given two models P m and P m , let m x and m x denote the associated marginal contribution functions, and let φ x and φ x denote the associated Shapley values.

If m x (S, i) ≥ m x (S, i) for all subsets S, then we are guaranteed that φ x (i) ≥ φ x (i).

Note that all three of these axioms are reasonable in our feature selection context.

The exact computation of the Shapley value φ x (i) takes into account the interaction of feature i with all 2 d−1 subsets that contain i, thereby leading to computational difficulties.

Various approximation methods have been developed with the goal of reducing complexity.

For example, Štrumbelj & Kononenko (2010) proposed to estimate the Shapley values via a Monte Carlo approximation built on an alternative permutation-based definition of the Shapley value.

Lundberg & Lee (2017) proposed to evaluate the model over randomly sampled subsets and use a weighted linear regression to approximate the Shapley values based on the collected model evaluations.

In practice, such sampling-based approximations may suffer from high variance when the number of samples to be collected per instance is limited.

(See Appendix E for an empirical evaluation.)

For large-scale predictive models, the number of features is often relatively large, meaning that the number of samples required to obtain stable estimates can be prohibitively large.

The main contribution of this paper is to address this challenge in a model-based paradigm, where the contribution of features to the response variable respects the structure of an underlying graph.

In this setting, we propose efficient algorithms and provide bounds on the quality of the resulting approximation.

As we discuss in more detail later, our approach should be viewed as complementary to samplingbased or regresssion-based approximations of the Shapley value.

In particular, these methods can be combined with the approach of this paper so as to speed up the computation of the L-Shapley and C-Shapley values that we propose.

In many applications, the features can be associated with the nodes of a graph, and we can define distances between pairs of features based on the graph structure.

Intuitively, features distant in the graph have weak interactions with each other, and hence excluding those features in the computation of Shapley value has little effect.

For instance, each feature vector x in sequence data (such as language, music etc.), can be associated with a line graph, where positions too far apart in a sequence may not affect each other in Shapley value computation; similarly, each image data is naturally modeled with a grid graph, such that pixels that are far apart may have little effect on each other in the computation of Shapley value.

In this section, we propose modified forms of the Shapley values, referred to as L-Shapley and CShapley values, that can be computed more efficiently than the Shapley value by excluding those weak interactions in the structured data.

We also show that under certain probabilistic assumptions on the marginal distribution over the features, these quantities yield good approximations to the original Shapley values.

More precisely, given feature vectors x ∈ R d , we let G = (V, E) denote a connected graph with nodes V and edges E ⊂ V × V , where each feature i is associated with a a node i ∈ V , and edges represent interactions between features.

The graph induces a distance function on V × V , given by d G ( , m) = number of edges in shortest path joining to m. (3) In the line graph, this graph distance corresponds to the number of edges in the unique path joining them, whereas it corresponds to the Manhattan distance in the grid graph.

For a given node i ∈ V , its k-neighborhood is the set DISPLAYFORM0 of all nodes at graph distance at most k. See FIG0 for an illustration for the 2D grid graph.

We propose two algorithms for approximating Shapley value in which features that are either far apart on the graph or features that are not directly connected have an accordingly weaker interaction.

In order to motivate our first graph-structured Shapley score, let us take a deeper look at Example ( ).

In order to compute the importance score of "not," the most important words to be included are "heartwarming" and "entertaining."

Intuitively, the words distant from them have a weaker influence on the importance of a given word in a document, and therefore have relatively less effect on the Shapley score.

Accordingly, as one approximation, we propose the L-Shapley score, which only perturbs the neighboring features of a given feature when evaluating its importance: Definition 1.

Given a model P m , a sample x and a feature i, the L-Shapley estimate of order k on a graph G is given byφ DISPLAYFORM0 The coefficients in front of the marginal contributions of feature i are chosen to match the coefficients in the definition of the Shapley value restricted to the neighborhood N k (i).

We show in Section 4 that this choice controls the error under certain probabilistic assumptions.

In practice, the choice of the integer k is dictated by computational considerations.

We also propose a second algorithm, C-Shapley, that further reduces the complexity of approximating the Shapley value.

Coming back to Example ( ) where we evaluate the importance of "not," both the L-Shapley estimate of order larger than two and the exact Shapley value estimate would evaluate the model on the word subset "It not heartwarming," which rarely appears in real data and may not make sense to a human or a model trained on real-world data.

The marginal contribution of "not" relative to "It not heartwarming" may be well approximated by the marginal contribution of "not" to "not heartwarming."

This motivates us to proprose C-Shapley: Definition 2.

Given a model P m , a sample x and a feature i, the C-Shapley estimate of order k on a graph G is given byφ DISPLAYFORM0 where C k (i) denotes the set of all subsets of N k (i) that contain node i, and are connected in G.The coefficients in front of the marginal contributions are a result of using Myerson value to characterize a new coalitional game over the graph G, in which the influence of disconnected subsets of features are additive.

The error between C-Shapley and the Shapley value can also be controlled under certain statistical assumptions.

See Section 4 for details.

For text data, C-Shapley is equivalent to only evaluating n-grams in a neighborhood of the word to be explained.

By the definition of k-neighborhoods, evaluating the C-Shapley scores for all d features takes O(k 2 d) model evaluations on a line graph, as each feature takes O(k 2 ) model evaluations.

In this section, we study some basic properties of the L-Shapley and C-Shapley values.

In particular, under certain probabilistic assumptions on the features, we show that they provide good approximations to the original Shapley values.

We also show their relationship to another concept from cooperative game theory, namely that of Myerson values, when the model satisfies certain local additivity assumptions.

In order to characterize the relationship between L-Shapley and the Shapley value in terms of some conditional independence assumption between features, we introduce absolute mutual information as a measure of dependence.

Given two random variables X and Y , the absolute mutual information I a (X; Y ) between X and Y is defined as DISPLAYFORM0 where the expectation is taken jointly over X, Y .

Based on the definition of independence, we have I a (X; Y ) = 0 if and only if X ⊥ ⊥ Y .

Recall the mutual information (Cover & BID3 ) is defined as I(X; Y ) = E[log DISPLAYFORM1 The new measure is more stringent than the mutual information in the sense that I(X; Y ) ≤ I a (X; Y ).

The absolute conditional mutual information can be defined in an analogous way.

Given three random variables X, Y and Z, we define the absolute conditional mutual information to be Theorem 1 and Theorem 2 show that L-Shapley and C-Shapley values, respectively, are related to the Shapley value whenever the model obeys a Markovian structure that is encoded by the graph.

We leave their proofs to Appendix B. Theorem 1.

Suppose there exists a feature subset S ⊂ N k (i) with i ∈ S, such that sup DISPLAYFORM2 DISPLAYFORM3 where we identify I a (X i ; X V |X ∅ ) with I a (X i ; X V ) for notational convenience.

Then the expected error between the L-Shapley estimateφ k X (i) and the true Shapley-value-based importance score φ i (P m , x) is bounded by 4ε: DISPLAYFORM4 In particular, we haveφ DISPLAYFORM5 Theorem 2.

Suppose there exists a neighborhood S ⊂ N k (i) of i, with i ∈ S, such that Condition 8 is satisfied.

Moreover, for any connected subset U ⊂ S with i ∈ U , we have sup DISPLAYFORM6 where DISPLAYFORM7 Then the expected error between the C-Shapley estimateφ k X (i) and the true Shapley-value-based importance score φ i (P m , x) is bounded by 6ε: DISPLAYFORM8 In particular, we haveφ DISPLAYFORM9

Let us now discuss how the C-Shapley value can be related to the Myerson value, which was introduced by Myerson (1977) as an approach for characterizing a coalitional game over a graph G. Given a subset of nodes S in the graph G, let C G (S) denote the set of connected components of S. Thus, if S is a connected subset of G, then C G (S) consists only of S; otherwise, it contains a collection of subsets whose disjoint union is equal to S.Consider a score function T → v(T ) that satisfies the following decomposability condition: for any subset of nodes S, the score v(S) is equal to the sum of the scores over the connected components DISPLAYFORM0 For any such score function, we can define the associated Shapley value, and it is known as the Myerson value on G with respect to v. Myerson (1977) showed that the Myerson value is the unique quantity that satisfies both the decomposability property, as well as the properties additivity, equal contributions and monotonicity given in Section 2.2.In our setting, if we use a plug-in estimate for conditional probability, the decomposability condition (12) is equivalent to assuming that the influence of disconnected subsets of features are additive at sample x, and C-Shapley of order k = d is exactly the Myerson value over G. In fact, if we partition each subset S into connected components, as in the definition of Myerson value, and sum up the coefficients (using Lemma 1 in Appendix B), then the Myerson value is equivalent to equation 6.

Let us how methods useful for approximating the Shapley value can be used to speed up the evaluation of approximate L-Shapley and C-Shapley values.

FORMULA3 propose a Monte Carlo approximation, based on randomly sampling permutations.

While L-Shapley is deterministic in nature, it is possible to combine it with this and other sampling-based methods.

For example, if one hopes to consider the interaction of features in a large neighborhood N k (i) with a feature i, where exponential complexity in k becomes a barrier, sampling based on random permutation of local features may be used to alleviate the computational burden.

Regression-based methods Lundberg & Lee (2017) proposed to sample feature subsets based on a weighted kernel, and carry out a weighted linear regression to estimate the Shapley value.

Strong empirical results were provided using the regression-based approximation, referred to as KernelSHAP; see, in particular, Section 5.1 and FIG5 of their paper.

We can combine such a regression-based approximation with our modified Shapley values to further reduce the evaluation complexity of the C-Shapley values.

In particular, for a chain graph, we evaluate the score function over all connected subsequences of length ≤ k; similarly, on a grid graph, we evaluate it over all connected squares of size ≤ k × k.

We evaluate the performance of L-Shapley and C-Shapley on real-world data sets involving text and image classification.

We compare L-Shapley and C-Shapley with several competitive algorithms for instancewise feature importance scoring on black-box models, including the regressionbased approximation known as KernelSHAP (Lundberg & Lee, 2017), SampleShapley (Štrumbelj & Kononenko, 2010) , and the LIME method (Ribeiro et al., 2016) .

We emphasize that our focus is model-agnostic interpretation, and we omit the comparison with methods requiring additional assumptions or specific to a certain class models (e.g., (Sundararajan et al., 2017; Shrikumar et al., 2017; BID0 Karpathy et al., 2015; Strobelt et al., 2018; Murdoch & Szlam, 2017) ).

For all methods, we choose the objective to be the log probability of the predicted class, and use the plug-in estimate of conditional probability across all methods (see Section 2.1).

See Appendix C and D for more experiments on a direct evaluation of the correlation with the Shapley value, and an analysis of sensitivity.

Text classification is a classical problem in natural language processing, in which text documents are assigned to predefined categories.

We study the performance of L-Shapley and C-Shapley on three popular neural models for text classification: word-based CNNs (Kim, 2014), characterbased CNNs (Zhang et al., 2015) , and long-short term memory (LSTM) recurrent neural networks (Hochreiter & Schmidhuber, 1997) , with the following three data sets on different scales: Table 1 : A summary of data sets and models in three experiments.

"

Average #w" is the average number of words per sentence.

"

Accuracy" is the model accuracy on test samples.

Table 2 : Each word is highlighted with the RGB color as a linear function of its importance score.

The background colors of words with positive and negative scores are linearly interpolated between blue and white, red and white respectively.

DISPLAYFORM0

We train a bidirectional LSTM on the Yahoo!

Answers Topic Classification Dataset (Zhang et al., 2015) , which achieves a test accuracy of 70.84%.

See Table 1 for a summary, and Appendix A for all of the details.

We choose zero paddings as the reference point for all methods, and make 4 × d model evaluations, where d is the number of words for each input.

Given the average length of each input (see Table 1 ), this choice controls the number of model evaluations under 1, 000, taking less than one second in TensorFlow on a Tesla K80 GPU for all the three models.

For L-Shapley, we are able to consider the interaction of each word i with the two neighboring words in N 1 (i) given the budget.

For C-Shapley, the budget allows the regression-based version to evaluate all n-grams with n ≤ 4.The change in log-odds scores before and after masking the top features ranked by importance scores is used as a metric for evaluating performance, where masked words are replaced by zero paddings.

This metric has been used in previous literature in model interpretation (Shrikumar et al., 2017; Lundberg & Lee, 2017) .

We study how the average log-odds score of the predicted class decreases as the percentage of masked features over the total number of features increases on 1, 000 samples from the test set.

Results are plotted in FIG4 .On IMDB with Word-CNN, the simplest model among the three, L-Shapley, achieves the best performance while LIME, KernelSHAP and C-Shapley achieve slightly worse performance.

On AG's news with Char-CNN, L-Shapley and C-Shapley both outperform other algorithms.

On Yahoo!

Answers with LSTM, C-Shapley outperforms the rest of the algorithms by a large margin, followed by LIME.

L-Shapley with order 1, SampleShapley, and KernelSHAP do not perform well for LSTM model, probably because some of the signals captured by LSTM are relatively long n-grams.

We also visualize the importance scores produced by different Shapley-based methods on Example ( ), which is part of a negative movie review taken from IMDB.

The result is shown in Table 2 .More visualizations by our methods can be found in Appendix H and Appendix I. We take each pixel as a single feature for both MNIST and CIFAR10.

We choose the average pixel strength and the black pixel strength respectively as the reference point for all methods, and make 4 × d model evaluations, where d is the number of pixels for each input image, which keeps the number of model evaluations under 4, 000.

LIME and L-Shapley are not used for comparison because LIME takes "superpixels" instead of raw pixels segmented by segmentation algorithms as single features, and L-Shapley requires nearly sixteen thousand model evaluations when applied to raw pixels.1 For C-Shapley, the budget allows the regression-based version to evaluate all n × n image patches with n ≤ 4.

FIG5 shows the decrease in log-odds scores before and after masking the top pixels ranked by importance scores as the percentage of masked pixels over the total number of pixels increases on 1, 000 test samples on MNIST and CIFAR10 data sets.

C-Shapley consistently outperforms other methods on both data sets.

FIG5 also shows two misclassified digits by the CNN model.

Interestingly, the top pixels chosen by C-Shapley visualize the "reasoning" of the model: the important pixels to the model are exactly those which could form a digit from the opposite class.

Table 3 : Results of human evaluation.

"Selected" and "Masked" indicate selected words and masked reviews respectively.

Results are averaged over 200 samples.

(The best numbers are highlighted.)

FIG6 provides additional visualization of the results.

By masking the top pixels ranked by various methods, we find that the pixels picked by C-Shapley concentrate around and inside the digits in MNIST.

For SampleShapley and KernelSHAP, unactivated pixels in MNIST are attributed nonzero scores when evaluated jointly with activated pixels.

While one could use post-processing by not choosing unactivated pixels, we choose to visualize the original outputs from all algorithms for fairness of comparison.

The C-Shapley also yields the most interpretable results in CIFAR10.

In particular, C-Shapley tends to mask the parts of head and body that distinguish deers and horses, and the human riding the horse.

More visualization results are available in Appendix F.

We use human annotators on Amazon Mechanical Turk (AMT) to compare L-Shapley, C-Shapley and KernelSHAP on IMDB movie reviews.

We aim to address two problems: (i) Are humans able to make a decision with top words alone? (ii) Are humans unable to make a decision with top words masked?We randomly sample 200 movie reviews that are correctly classified by the model.

Each review is assigned to five annotators.

We ask humans on AMT to classify the sentiment of texts into five categories: strongly positive (+2), positive (+1), neutral (0), negative (-1), strongly negative (-2).

See Appendix G for an example interface.

Texts have three types: (i) raw reviews; (ii) top ten words of each review ranked by L-Shapley, C-Shapley and KernelSHAP, where adjacent words, like "not satisfying or entertaining", keep their adjacency if selected simultaneously; and (iii) reviews with top words being masked.

In the third type of texts, words are replaced with "[MASKED]" one after another, in the order produced by the respective algorithms, until the probability score of the correct class produced by the model is lower than 10%.

We adopt the above design to make sure the majority of key words sensitive to the model have been masked.

On average, around 14% of words in each review are masked for L-Shapley and C-Shapley, while 31.6% for KernelSHAP.We measure the consistency (0 or 1) between the true labels and labels from human annotators, where a human label is positive if the average score over five annotators are larger than zero.

Reviews with an average score of zero are neither put in the positive nor in the negative class.

We also employ the standard deviation of scores on each review as a measure of disagreement between humans.

Finally, the absolute value of the average scores from five annotators is used as a measure of confidence of decision.

The results of the two experiments are shown in Table 3 .

We observe humans become more consistent with the truth and more confident, and also have less disagreement with each other when they are presented with top words.

Among the three algorithms, C-Shapley yields the highest performance in terms of consistency, agreement, and confidence.

On the other hand, when top words are masked, humans are easier to make mistakes and are less certain about their judgement.

L-Shapley harms the human judgement the most among the three algorithms, although KernelSHAP masks two times more words.

The above experiments show that (i) Key words to the model convey an attitude toward a movie;, and (ii) Our algorithms find the key words more accurately.

We have proposed two new algorithms-L-Shapley and C-Shapley-for instancewise feature importance scoring, making use of a graphical representation of the data.

We have demonstrated the superior performance of these algorithms compared to other methods on black-box models for instancewise feature importance scoring in both text and image classification with both quantitative metrics and human evaluation.

Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky.

Neural networks for machine learninglecture 6a-overview of mini-batch gradient descent.

et al., 2011) , which contains 50, 000 binary labeled movie reviews, with a split of 25, 000 for training and 25, 000 for testing.

AG news with Char-CNN The AG news corpus is composed of titles and descriptions of 196, 000 news articles from 2, 000 news sources (Zhang et al., 2015) .

It is segmented into four classes, each containing 30, 000 training samples and 1, 900 testing samples.

Yahoo!

Answers with LSTM The corpus of Yahoo!

Answers Topic Classification Dataset is divided into ten categories, each class containing 140, 000 training samples and 5, 000 testing samples.

Each input text includes the question title, content and best answer.

MNIST The MNIST data set contains 28 × 28 images of handwritten digits with ten categories 0 − 9 (LeCun et al., 1998).

A subset of MNIST data set composed of digits 3 and 8 is used for better visualization, with 12, 000 images for training and 1, 000 images for testing.

The CIFAR10 data set (Krizhevsky, 2009) contains 32 × 32 images in ten classes.

A subset of CIFAR10 data set composed of deers and horses is used for better visualization, with 10, 000 images for training and 2, 000 images for testing.

MNIST A simple CNN model is trained on the data set, which achieves 99.7% accuracy on the test data set.

It is composed of two convolutional layers of kernel size 5 × 5 and a dense linear layer at last.

The two convolutional layers contain 8 and 16 filters respectively, and both are followed by a max-pooling layer of pool size two.

CIFAR10 A convolutional neural network modified from AlexNet Krizhevsky et al. FORMULA3 is trained on the subset.

It is composed of six convolutional layers of kernel size 3 × 3 and two dense linear layers of dimension 512 and 256 at last.

The six convolutional layers contain 48, 48, 96, 96, 192, 192 filters respectively, and every two convolutional layers are followed by a maxpooling layer of pool size two and a dropout layer.

The CNN model is trained with the Adam optimizer Kingma & Ba FORMULA3 and achieves 96.1% accuracy on the test data set.

In this appendix, we collect the proofs of Theorems 1 and 2.

We state an elementary combinatorial equality required for the proof of the main theorem:Lemma 1 (A combinatorial equality).

For any positive integer n, and any pair of non-negative integers with s ≥ t, we have Proof.

By the binomial theorem for negative integer exponents, we have DISPLAYFORM0 The identity can be found by examination of the coefficient of x n in the expansion of DISPLAYFORM1 In fact, equating the coefficients of x n in the left and the right hand sides, we get DISPLAYFORM2 Moving n+s n to the right hand side and expanding the binomial coefficients, we have DISPLAYFORM3 which implies n j=0 n j s t DISPLAYFORM4 Taking this lemma, we now prove the theorem.

We split our analysis into two cases, namely DISPLAYFORM5 For notational convenience, we extend the definition of L-Shapley estimate for feature i to an arbitrary feature subset S containing i. In particular, we definê DISPLAYFORM6 Case 1: First, suppose that S = N k (i).

For any subset A ⊂ [d], we introduce the shorthand notation U S (A) : = A ∩ S and V S (A) : = A ∩ S c , and note that A = U S (A) ∪ V S (A).

Recalling the definition of the Shapley value, let us partition all the subsets A based on U S (A), in particular writing DISPLAYFORM7 Based on this partitioning, the expected error betweenφ S X (i) and φ X (i) can be written as DISPLAYFORM8 Partitioning the set {A : U S (A) = U } by the size of V S (A) = A ∩ S c , we observe that DISPLAYFORM9 where we have applied Lemma 1 with n = d − |S|, s = |S| − 1, and t = |U | − 1.

Substituting this equivalence into equation equation 18, we find that the expected error can be upper bounded by DISPLAYFORM10 where we recall that A = U S (A) ∪ V S (A).

A for notational simplicity, we now write the difference as DISPLAYFORM0 Substituting this equivalence into our earlier bound equation 19 and taking an expectation over X on both sides, we find that the expected error is upper bounded as DISPLAYFORM1 Recalling the definition of the absolute mutual information, we see that DISPLAYFORM2 ≤ 2ε, which completes the proof of the claimed bound.

Case 2: We now consider the general case in which S ⊂ N k (i).

Using the previous arguments, we can show DISPLAYFORM3 Appylying the triangle inequality yields E|φ k X (i) − φ X (i)| ≤ 4ε, which establishes the claim.

As in the previous proof, we divide our analysis into two cases.

DISPLAYFORM0 For any subset A ⊂ S with i ∈ A, we can partition A into two components U S (A) and V S (A), such that i ∈ U S (A) and U S (A) is a connected subsequence.

V S (A) is disconnected from U S (A).

We also define DISPLAYFORM1 , U is a connected subsequence.} (20) We partition all the subsets A ⊂ S based on U S (A) in the definition of the Shapley value: DISPLAYFORM2 The expected error betweenφ DISPLAYFORM3 (21) Partitioning {A : U S (A) = U } by the size of V S (A), we observe that DISPLAYFORM4 where we apply Lemma 1 with n = d − |U | − 2, s = |U | + 1 and t = |U | − 1.

From equation equation 21, the expected error can be upper bounded by DISPLAYFORM5 We omit the dependence of U S (A) and V S (A) on the pair (A, S) for notational simplicity, and observe that the difference between m x (A, i) and m x (U, i) is DISPLAYFORM6 Taking an expectation over X at both sides, we can upper bound the expected error by DISPLAYFORM7 Therefore, we haveφ

We address the problem of how the rank of features produced by various approximation algorithms correlates with the rank produced by the true Shapley value.

We sample a subset of test data from Yahoo!

Answers with 9-12 words, so that the underlying Shapley scores can be accurately computed.

We employ two common metrics, Kendall's τ and Spearman's ρ (Kendall, 1975) , to measure the similarity (correlation) between two ranks.

The result is shown in Figure 5 .

The rank correlation between L-Shapley and the Shapley value is the highest, followed by C-Shapley, consistent across both of the two metrics.

Given the limited length of each instance, the search space for sampling based algorithms is relatively small.

Thus there is only a slight performance gain of our algorithms over KernelSHAP and SampleShapley.

Figure 6 shows how Kendall's τ and Spearman's ρ between the proposed algorithms and the Shapley value vary with the radius of neighborhood.

We observe the bias of the proposed algorithms decreases gradually with increasing neighborhood radius.

Figure 7 plots the number of model evaluations as a function of neighborhood radius for both algorithms, on an example instance with ten features 3 .

The complexity of L-Shapley grows exponentially with the neighborhood radius while the complexity of C-Shapley grows linearly.

We empirically evaluate the variance of SampleShapley and KernelSHAP in the setting where the sample size is linear in the number of features.

The experiment is carried out on the test data set of IMDB movie reviews.

For each method, we run 30 replications on every instance, which generates 30 scores.

Given the varied scalability of underlying Shapley values, we again seek a nonparametric approach to measure the variability of sampling based algorithms.

On each instance, we compute

Only the ten words with the largest scores and the ten words with the smallest scores are colorized.

The words with largest scores with respect to the predicted class are highlighted with red.

The ten words with smallest scores with respect to the predicted class are highlighted with blue.

(In other words, red words tend to contain positive attitude for a positive prediction, but negative attitude for a negative prediction.)

The corresponding RGB entries are linearly interpolated based on importance scores.

The lighter a color is, the less information with respect to the prediction the corresponding word is.

that everyone always gave a good performance , the production design was spectacular , the costumes were well _ designed , and the writing was always very strong .

In conclusion , even though new episodes can currently be seen , I strongly recommend you catch it just in case it goes off the air for good .

Mike and the bots so it made it bearable .

Horrid acting , unsettling mother / daughter moment , silly premise , if you want a bad movie here it is .

Be warned though watch it with Mike and the bots or you will suffer .

1 out of 10 .

I still cant believe it won an award , and the director is defending this movie !

negative I had never heard of this one before it turned up on Cable TV . were in it .

They were in House Party 3 when they were 11 , but they are all grown up now !

I was a little shocked at some of the things they were doing in the movie ( almost ready to tear my hair out ), but I had to realize that they were not my little boys anymore .

I think ChrisStokes did a pretty good job , considering that is was his first movie .

Only the ten words with the largest scores and the ten words with the smallest scores are colorized.

The words with largest scores with respect to the predicted class are highlighted with red.

The ten words with smallest scores with respect to the predicted class are highlighted with blue.

The corresponding RGB entries are linearly interpolated based on importance scores.

The lighter a color is, the less information with respect to the prediction the corresponding word is.

less longer summers not so bloody freezing in winter oh and the small matter of maybe wales flooding n n n nso this global warming is a bad thing yeah Politics, Government so if our borders need fixing and let's agree that they do how do we pay for it the united states congress seems to come up with all kinds of money for a lot of silly things here are some examples n 75 000 for seafood waste research n 500 000 for the arctic winter games n 300 000for sunset and beaches in california n 350 000 for the chicago program for the design installation and maintenance of over 950 hanging baskets n 600 000 for the abraham lincoln commission n 100 000 for the police department has a population of 400 people n 2 500 000 for the space flight center for process dry cleaning capability n 500 000 for construction of the museum nand the list goes on and on n i think we could find a few places to make cuts to pay for securing our borders Society, Culture why do filipinos are using language yet there is no such a language some of them are forced to resort back to filipino you should have to pay to get it and be a citizen of the us to reap our benefits i don't know who's bright idea that was but i'm sure as soon as they let us know he will not be a very popular man Politics, Government whats a good way to raise money to get someone out of jail i need to get 1500 to get them out no i would assume you mean to make bail not pay for an escape but also remember if they make bail in most places they can not use a pubic defender since making bail shows they have or had the money to hire thier own attorney n nwork second job sell your computer tv Politics, Government what does the aclu think it is doing other than being a i mean honestly free speech is important but people also have to have decency they are helping to strip the nation of our the values and that make us americans they are ensuring that no one is judged based on their actions that anything and everything goes n nthey used to protect americans right to free speech but now they are so far left they make the 9th circus court of appeals appear right wingPolitics, Government what is a a is the holder of various important jobs including n n formerly the head priest in an when it had responsibilities n n the chief academic officer at various universities in north america n n an officer of local government including the scottish equivalent of a mayor the lord is the scottish equivalent of lord mayor in edinburgh glasgow and n n the officer in charge of military police n n sergeant a sergeant in charge of police in the british and commonwealth armies n n the administrator of a prison n n Entertainment, Music if your husband had cheap on his breath and wanted to take you in bed would you like it you mean like the mother on that movie carrie huh n nand i liked it i liked it n n Entertainment, Music how does a band register to play the 2006 sorry to tell you this but the deadline for a band to register for a at this

@highlight

We develop two linear-complexity algorithms for model-agnostic model interpretation based on the Shapley value, in the settings where the contribution of features to the target is well-approximated by a graph-structured factorization.

@highlight

The paper proposes two approximations to the Shapley value used for generating feature scores for interpretability.

@highlight

This paper proposes two methods for instance-wise feature importance scoring using Shapely values, and provides two efficient methods of computing approximate Shapely values when there is a known structure relating the features.