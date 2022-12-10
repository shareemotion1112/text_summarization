An important question in task transfer learning is to determine task transferability, i.e. given a common input domain, estimating to what extent representations learned from a source task can help in learning a target task.

Typically, transferability is either measured experimentally or inferred through task relatedness, which is often defined without a clear operational meaning.

In this paper, we present a novel metric, H-score, an easily-computable evaluation function that estimates the performance of transferred representations from one task to another in classification problems.

Inspired by a principled information theoretic approach, H-score has a direct connection to the asymptotic error probability of the decision function based on the transferred feature.

This formulation of transferability can further be used to select a suitable set of source tasks in task transfer learning problems or to devise efficient transfer learning policies.

Experiments using both synthetic and real image data show that not only our formulation of transferability is meaningful in practice, but also it can generalize to inference problems beyond classification, such as recognition tasks for 3D indoor-scene understanding.

Transfer learning is a learning paradigm that exploits relatedness between different learning tasks in order to gain certain benefits, e.g. reducing the demand for supervision BID22 ).

In task transfer learning, we assume that the input domain of the different tasks are the same.

Then for a target task T T , instead of learning a model from scratch, we can initialize the parameters from a previously trained model for some related source task T S .

For example, deep convolutional neural networks trained for the ImageNet classification task have been used as the source network in transfer learning for target tasks with fewer labeled data BID7 ), such as medical image analysis BID24 ) and structural damage recognition in buildings (Gao & Mosalam) .

An imperative question in task transfer learning is transferability, i.e. when a transfer may work and to what extent.

Given a metric capable of efficiently and accurately measuring transferability across arbitrary tasks, the problem of task transfer learning, to a large extent, is simplified to search procedures over potential transfer sources and targets as quantified by the metric.

Traditionally, transferability is measured purely empirically using model loss or accuracy on the validation set (Yosinski et al. (2014) ; Zamir et al. (2018) ; BID5 ).

There have been theoretical studies that focus on task relatedness BID1 ; BID19 ; BID21 ; BID2 ).

However, they either cannot be computed explicitly from data or do not directly explain task transfer performance.

In this study, we aim to estimate transferability analytically, directly from the training data.

We quantify the transferability of feature representations across tasks via an approach grounded in statistics and information theory.

The key idea of our method is to show that the error probability of using a feature of the input data to solve a learning task can be characterized by a linear projection of this feature between the input and output domains.

Hence we adopt the projection length as a metric of the feature's effectiveness for the given task, and refer to it as the H-score of the feature.

More generally, H-score can be applied to evaluate the performance of features in different tasks, and is particularly useful to quantify feature transferability among tasks.

Using this idea, we define task transferability as the normalized H-score of the optimal source feature with respect to the target task.

As we demonstrate in this paper, the advantage of our transferability metric is threefold.

(i) it has a strong operational meaning rooted in statistics and information theory; (ii) it can be computed directly and efficiently from the input data, with fewer samples than those needed for empirical learning; (iii) it can be shown to be strongly consistent with empirical transferability measurements.

In this paper, we will first present the theoretical results of the proposed transferability metric in Section 2-4.

Section 5 presents several experiments on real image data , including image classificaton tasks using the Cifar 100 dataset and 3D indoor scene understanding tasks using the Taskonomy dataset created by Zamir et al. (2018) .

A brief review of the related works is included in Section 6.

In this section, we will introduce the notations used throughout this paper, as well as some related concepts in Euclidean information theory and statistics.

X, x, X and P X represent a random variable, a value, the alphabet and the probability distribution respectively.

√ P X denotes the vector with entries P X (x) and [ √ P X ] the diagonal matrix of P X (X).

For joint distribution P Y X , P Y X represents the |Y| × |X | probability matrix.

Depending on the context, f(X) is either a |X |-dimensional vector whose entries are f (x), or a |X | × k feature matrix.

Further, we define a task to be a tuple T = (X, Y, P XY ), where X is the training features and Y is the training label, and P XY the joint probability (possibly unknown).

Subscripts S and T are used to distinguish the source task from the target task.

Our definiton of transferability uses concepts in local information geometry developed by BID18 , which characterizes probability distributions as vectors in the information space.

Consider the following binary hypothesis testing problem: test whether i.i.d.

samples DISPLAYFORM0 are drawn from distribution P 1 or distribution P 2 , where P 1 , P 2 belong to an -neighborhood N (P 0 ) {P | x∈X DISPLAYFORM1 as the information vector corresponding to P i for i = 1, 2.

DISPLAYFORM2 for the binary hypothesis testing problem.

Let E f be the error exponent of decision region {x m | l(x m ) > T } for T ≥ 0, which characterizes the asymptotic error probability P e of l (i.e. lim m→∞ − 1 m log(P e ) = E k f ).

E f can be written as the squared length of a projection: DISPLAYFORM3 When f (x) = log DISPLAYFORM4 P2(x) is the log likelihood ratio, l is the minimum sufficient statistics that achieves the largest error exponent DISPLAYFORM5 by the Chernoff theorem.

(See Appendix A for details.)

In the rest of this paper, we assume is small.

Definition 1.

MatrixB is the Divergence Transition Matrix (DTM) of a joint probability DISPLAYFORM0 The singular values ofB satisfy that DISPLAYFORM1 be the left and right singular vectors ofB. Define functions DISPLAYFORM2 for each i = 1, . . .

, K − 1.

BID18 further proved that f * i and g * i are solutions to the maximal HGR correlation problem studied by BID12 BID10 ; BID23 , defined as follows: DISPLAYFORM3 The maximal HGR problem finds the K strongest, independent modes in P XY from data.

It can be solved efficiently using the Alternating Conditional Expectation (ACE) algorithm with provable error bound (see Appendix B).

BID13 further showed that f * and g * are the universal minimum error probability features in the sense that they can achieve the smallest error probability over all possible inference tasks.

In this section, we present a performance metric of a given feature representation for a learning task.

For a classification task involving input variable X and label Y , most learning algorithms work by finding a k-dimensional functional representation f (x) of X that is most discriminative for the classification.

To measure how effective f (x) is in predicting Y , rather than train the model via gradient descent and evaluate its accuracy, we present an analytical approach based on the definition below: Definition 2.

Given data matrix X ∈ R m×d and label Y ∈ {1, . . . , |Y|}. Let f (x) be a kdimensional, zero-mean feature function.

The H-Score of f (x) with respect to the learning task represented by P Y X is: DISPLAYFORM0 This definition is intuitive from a nearest neighbor search perspective.

i.e. a high H-score implies the inter-class variance cov(E P X|Y [f (X)|Y ]) of f is large, while feature redundancy tr(cov(f (X))) is small.

Such a feature is discriminative and efficient for learning label Y. More importantly, H(f ) has a deeper operational meaning related to the asymptotic error probability for a decision rule based on f in the hypothesis testing context, discussed in the next section.

Without loss of generality, we consider the binary classification task as a hypothesis testing problem defined in Section 2.1, with P 1 = P X|Y =0 , and P 2 = P X|Y =1 .

For any k-dimensional feature representation f (x), we can quantify its performance with respect to the learning task using its error DISPLAYFORM0 See Appendix C for the proof.

The above theorem shows that H-score H(f ) is proportional to the error exponent of the decision region based on f (x) when f (x) is zero-mean with identity covariance.

To compute the H-score of arbitrary f , we can center the features f S (x) − E[f S (x)], and incorporate normalization into the computation of the error exponent, which results in Definition 2.

The details are presented in Appendix D.The proof for Theorem 1 uses the fact that H(f ) = B Ξ 2 F , whereB is the DTM matrix, Ξ [ξ 1 · · · ξ k ] is the matrix composed of information vectors ξ i and c is a constant.

This allows us to infer an upper bound for the H-score of a given learning task: The first inequality is achieved when Ξ is composed of the right singular vectors ofB, i.e. DISPLAYFORM1 DISPLAYFORM2 The corresponding feature functions f opt (X)is in fact the same as the universal minimum error probability features from the maximum HGR correlation problem.

The final inequality in Corollary 1 is due to the fact all singular values ofB are less than or equal to 1.

Next, we apply H-score to efficiently measure the effectiveness of task transfer learning.

We will also discuss how this approach can be used to solve the source task selection problem.

A typical way to transfer knowledge from the source task T S to target task T T is to train the target task using source feature representation f S (x).

In a neural network setting, this idea can be implemented by copying the parameters from the first N layers in the trained source model to the target model, assuming the model architecture on those layers are the same.

The target classifier then can be trained while freezing parameters in the copied layers FIG1 ).

Under this model, a natural way to quantify transferability is as follows: Definition 3 (Task transferability).

Given source task T S and target task T T , and trained source feature representation f S (x), the transferability from T S to T T is T(S, T ) DISPLAYFORM0 , where f Topt (x) is the minimum error probability feature of the target task.

The statement T(S, T ) = r means the error exponent of transfering from T S via feature representation f S is 1 r of the optimal error exponent for predicting the target label Y T .

This definition also implies 0 ≤ T(S, T ) ≤ 1, which satisfies the data processing inequality if we consider the transferred feature f S (X) as post-processing of input X for solving the target task.

And it can not increase information about predicting the target task T .

A common technique in task transfer learning is fine-tuning, which adds before the target classifier additional free layers, whose parameters are optimized with respect to the target label.

For the operational meaning of transferability to hold exactly, we require the fine tuning layers consist of only linear transformations, such as the model illustrated in Figure 2 .a.

It can be shown that under the local assumption, H-score is equivalent to the log-loss of the linear transfer model up to a constant offset (Appendix E).

Nevertheless, later we will demonstrate empirically that this transferability metric can still be used for comparing the relative task transferability with fine-tuning.

With a known f , computing H-score from m sample data only takes O(mk 2 ) time, where k is the dimension of f (x) for k < m. The majority of the computation time is spent on computing the sample covariance matrix cov(f (X)).The remaining question is how to obtain f Topt efficiently.

We use the fact that DISPLAYFORM0 , where f and g are the solutions of the HGR-Maximum Correlation problem.

This problem can be solved efficiently using the ACE algorithm for discrete variable X. For a continuous random variable X, we can obtain f opt through a different formulation of the HGR maximal correlation problem: DISPLAYFORM1 DISPLAYFORM2 This is also known as the soft HGR problem studied by BID26 , who reformulated the original maximal HGR correlation objective to eliminate the whitening constraints while having theoretically equivalent solution.

In practice, we can utilize neural network layers to model functions f and g, as shown in Figure 2 .b.

Given two branches of k output units for both f and g, the loss function can be evaluated in O(mk 2 ), where m is the batch size.

BID18 showed that the sample complexity of ACE is only 1/k of the complexity of estimating P Y,X directly.

This result also applies to the soft HGR problem due to their theoretical equivalence.

Hence transferability can be computed with much less samples than actually training the transfer network.

It's also worth noting that, when f is fixed, maximizing the objective in Equation 3 with respect to zero-mean function g results in the definition of H-score.

In many cases though, the computation of H T (f opt ) can even be skipped entirely, such as the problem below: Definition 4 (Source task selection).

Given N source tasks T S1 , . . .

, T S N with labels Y S1 , . . .

, Y S N and a target task T T with label Y T .

Let f S1 , . . .

, f S N be the minimum error probability feature functions of the source tasks.

Find the source task T Si that maximizes the testing accuracy of predicting Y T with feature f Si .We can solve this problem by selecting the source task with the largest transferability to T T .

In fact, we only need to compute the numerator in the transferability definition since the denominator is the same for all source tasks, i.e. DISPLAYFORM3

Under the local assumption that P X|Y ∈ N (P X ), we can show that mutual information DISPLAYFORM0 (See Appendix F for details.)

Hence H-score is related to mutual information by H(f (x)) ≤ 2I(X; Y ) for any zero-mean features f (x) satisfying the aforementioned conditions.

Figure 3 compares the optimal H-score of a synthesized task when |Y| = 6 with the mutual information between input and output variables, when the feature dimension k changes.

The value of H-score increases as k increases, but reaches the upper bound when k ≥ 6, since the rank of the joint probability between X and Y T , as well as the rank of its DTM is 6.

As expected, the H-score values are below 2I(X; Y ), with a gap due to the constant o(2 ).

This relationship shows that H-score is consistent with mutual information with a sufficiently large feature dimension k.

In practice, H-score is much easier to compute than mutual information when the input variable X (or f S (X)) is continous, as mutual information are either computed based on binning, which has extra bias due to bin size, or more sophisticated methods such as kernel density estimation or neural networks BID8 ).

On the other hand, H-score only needs to estimate conditional expectations, which requires less samples.

In this section, we validate and analyze our transferability metric through experiments on real image data.1 The tasks considered cover both object classification and non-classification tasks in computer vision, such as depth estimation and 3D (occlusion) edge detection.

To demonstrate that our transferability metric is indeed a suitable measurement for task transfer performance, we compare it with the empirical performance of transfering features learned from ImageNet 1000-class classification BID16 ) to Cifar 100-class classification BID15 ), using a network similar to FIG1 .

Comparing to ImageNet-1000, Cifar-100 has smaller sample size and its images have lower resolution.

Therefore it is considered to be a more challenging task than ImageNet, making it a suitable case for transfer learning.

In addition, we use a pretrained ResNet-50 as the source model due to its high performance and regular structure.

Validation of H-score.

The training data for the target task in this experiemnt consists of 20, 000 images randomly sampled from Cifar-100.

It is further split 9:1 into a training set and a testing set.

The transfer network was trained using stochastic gradient descent with batch size 20, 000 for 100 epochs.

FIG3 .a compares the H-score of transferring from five different layers (4a-4f) in the source network with the target log-loss and test accuracy of the respective features.

As H-score increases, log-loss of the target network decreases almost linearly while the training and testing accuracy increase.

Such behavior is consistent with our expectation that H-score reflects the learning performance of the target task.

We also demonstrated that target sample size does not affect the relationship between H-score and log-loss FIG3 .b).This experiment also showed another potential application of H-score for selecting the most suitable layer for fine-tuning in transfer learning.

In the example, transfer performance is better when an upper layer of the source networks is transferred.

This could be because the target task and the source task are inherently similar such that the representation learned for one task can still be discriminative for the other.

Validation of Transferability.

We further tested our transferability metric for selecting the best target task for a given source task.

In particular, we constructed 4 target classification tasks with 3, 5, 10, and 20 object categories from the Cifar-100 dataset.

We then computed the transferability from ImageNet-1000 (using the feature representation of layer 4f) to the target tasks.

The results are compared to the empirical transfer performance trained with batch size 64 for 50 epochs in Figure4.c.

We observe a similar behavior as the H-score in the case of a single target task in FIG3 .a, showing that transferability can directly predict the empirical transfer performance.

In this experiment, we solve the source task selection problem for a collection of 3D sceneunderstanding tasks using the Taskonomy dataset from Zamir et al. (2018) .

In the following, we will introduce the experiment setting and explain how we adapt the transferability metric to pixel-to-pixel recognition tasks.

Then we compare transferability with task affinity, an empirical transferability metric proposed by Zamir et al. (2018) .Data and Tasks.

The Taskonomy dataset contains 4,000,000 images of indoor scenes of 600 buildings.

Every image has annotations for 26 computer vision tasks.

We randomly sampled 20, 000 images as training data.

Eight tasks were chosen for this experiment, covering both classifications and lower-level pixel-to-pixel tasks.

Table 6 summaries the specifications of these tasks and sample outputs are shown in FIG4 .

For classification tasks, H-score can be easily calculated given the source features.

But for pixel-topixel tasks such as Edges and Depth, their ground truths are represented as images, which can not be quantized easily.

As a workaround, we cluster the pixel values in the ground truth images into a palette of 16 colors.

Then compute the H-score of the source features with respect to each pixel, before aggregating them into a final score by averaging over the whole image.

We ran the experiment on a workstation with 3.40 GHz ×8 CPU and 16 GB memory.

Each pairwise H-score computation finished in less than one hour including preprocessing.

Then we rank the source tasks according to their H-scores of a given target task and compare the ranking with that in Zamir et al. (2018) .

Pairwise Transfer Results.

Source task ranking results using transferability and affinity are visualized side by side in Figure 7 , with columns representing source tasks and rows representing target tasks.

For classification tasks (the bottom two rows in the transferability matrix), the top two transferable source tasks are identical for both methods.

The best source task is the target task itself, as the encoder is trained on a task-specific network with much larger sample size.

Scene Class. and Object Class.

are ranked second for each other, as they are semantically related.

Similar observations can be found in 2D pixel-to-pixel tasks (top two rows).

The results on lower rankings are noisier.

A slightly larger difference between the two rankings can be found in 3D pixel-to-pixel tasks, especially 3D Occlusion Edges and 3D Keypoints.

Though the top four ranked tasks of both methods are exactly the four 3D tasks.

It could indicate that these low level vision tasks are closely related to each other so that the transferability among them are inherently ambiguous.

We also computed the ranking correlations between transferability and affinity using Spearman's R and Discounted Cumulative Gain (DCG).

Both criterion show positive correlations for all target tasks.

The correlation is especially strong with DCG as higher ranking entities are given larger weights.

The above observations inspire us to define a notion of task relatedness, as some tasks are frequently ranked high for each other.

Specifically, we represent each task with a vector consisting of H-scores of all the source tasks, then apply agglomerative clustering over the task vectors.

As shown in the dendrogram in Figure 7 , 2D tasks and most 3D tasks are grouped into different clusters, but on a higher level, all pixel-to-pixel tasks are considered one category compared to the classifications tasks.

Higher Order Transfer.

Sometimes we need to combine features from two or more source tasks for better transfer performance.

A common way to combine features from multiple models in deep neural networks is feature concatenation.

For such problems, our transferability definition can be easily adapted to high order feature transfer, by computing the H-score of the concatenated features.

Figure 8 shows the ranking results of all combinations of source task pairs for each target task.

For all tasks except for 3D Occlusion Edges and Depth, the best seond-order source feature is the combination of the top two tasks of the first-order ranking.

We examine the exception in Figure 9 , by visualizing the pixel-by-pixel H-scores of first and second order transfers to Depth using a heatmap (lighter color implies a higher H-score).

Note that different source tasks can be good at predicting different parts of the image.

The top row shows the results of combining tasks with two different "transferability patterns" while the bottom row shows those with similar patterns.

Combining tasks with different transferability patterns has a more significant improvement to the overall performance of the target task.

Transfer learning.

Transfer learning can be devided into two categories: domain adaptation, where knowledge transfer is achieved by making representations learned from one input domain work on a different input domain, e.g. adapt models for RGB images to infrared images BID27 ); and task transfer learning, where knowledge is transferred between different tasks on the same input domain BID25 ).

Our paper focus on the latter prolem.

Empirical studies on transferability.

Yosinski et al. (2014) compared the transfer accuracy of features from different layers in a neural network between image classification tasks.

A similar study was performed for NLP tasks by BID5 .

Zamir et al. (2018) determined the optimal transfer hierarchy over a collection of perceptual indoor scene understanidng tasks, while transferability was measured by a non-parameteric score called "task affinity" derived from neural network transfer losses coupled with an ordinal normalization scheme.

Task relatedness.

One approach to define task relatedness is based on task generation.

Generalization bounds have been derived for multi-task learning BID1 ), learning-to-learn BID19 ) and life-long learning BID21 ).

Although these studies show theoretical results on transferability, it is hard to infer from data whether the assumptions are satisfied.

Another approach is estimating task relatedness from data, either explicitly BID3 ; Zhang Representation learning and evaluation.

Selecting optimal features for a given task is traditionally performed via feature subset selection or feature weight learning.

Subset selection chooses features with maximal relevance and minimal redundancy according to information theoretic or statistical criteria BID20 ; Hall (1999)).

The feature weight approach learns the task while regularizing feature weights with sparsity constraints, which is common in multi-task learningLiao & Carin FORMULA9 ; BID0 .

In a different perspective, BID13 consider the universal feature selection problem, which finds the most informative features from data when the exact inference problem is unknown.

When the target task is given, the universal feature is equivalent to the minimum error probability feature used in this work.

In this paper, we presented H-score, an information theoretic approach to estimating the performance of features when transferred across classification tasks.

Then we used it to define a notion of task transferability in multi-task transfer learning problems, that is both time and sample complexity efficient.

The resulting transferability metric also has a strong operational meaning as the ratio between the best achievable error exponent of the transferred representation and the minium error exponent of the target task.

Our transferability score successfully predicted the performance for transfering features from ImageNet-1000 classification task to Cifar-100 task.

Moreover, we showed how the transferability metric can be applied to a set of diverse computer vision tasks using the Taskonomy dataset.

In future works, we plan to extend our theoretical results to non-classification tasks, as well as relaxing the local assumptions on the conditional distributions of the tasks.

We will also investigate properties of higher order transferability, developing more scalable algorithms that avoid computing the H-score of all task pairs.

On the application side, as transferability tells us how different tasks are related, we hope to use this information to design better task hierarchies for transfer learning.

DISPLAYFORM0 x m with the following hypotheses: DISPLAYFORM1 Let P x m be the empirical distribution of the samples.

The optimal test, i.e., the log likelihood ratio test can be stated in terms of information-theoretic quantities as follows: DISPLAYFORM2 Figure 10: The binary hypothesis testing problem.

The blue curves shows the probility density functions for P 1 and P 2 .

The rejection region and the acceptance region are highlighted in red and blue, respectively.

The vertical line indicates the decision threshold.

Further, using Sannov's theorem, we have that asymptotically the probability of type I error DISPLAYFORM3 where P * DISPLAYFORM4 m log T } denotes the rejection region.

Similarly, for type II error DISPLAYFORM5 where P * 2 = argmin P ∈A D(P ||P 2 ) and A = {x m : FIG1 The overall probability of error is P (m) e = αP r(H 0 ) + βP r(H 1 ) and the best achievable exponent in the Bayesian probability of error (a.k.a.

Chernoff exponent) is defined as: DISPLAYFORM6 DISPLAYFORM7 See Cover & BID6 for more background information on error exponents and its related theorems.

Under review as a conference paper at ICLR 2019

Now consider the same binary hypothesis testing problem, but with the local constraint DISPLAYFORM0 denote the information vectors corresponding to P i for i = 1, 2.Makur et al. FORMULA3 uses local information geometry to connect the error exponent in hypothesis testing to the length of certain information vectors, summarized in the following two lemmas.

Lemma 1.

Given zero-mean, unit variance feature function f (x) : X → R, the optimal error exponent (a.k.a.

Chernoff exponent) of this hypothesis testing problem is DISPLAYFORM1 Lemma 2.

Given zero-mean, unit variance feature function f (x) : X → R, the error exponent of a mismatched decision function of the form l = DISPLAYFORM2 where ξ(x) = P 0 (x)f (x) is the feature vector associated with f (x).As our discussion of transferability mostly concerns with multi-dimensional features, we present the k-dimensional generalization of Lemma 2 below: (Equation 1 in the main paper.) DISPLAYFORM3 for all i, and cov(f (X)) = I , we define a k-d statistics of the form l k = (l 1 , ..., l k ) where DISPLAYFORM4 be the corresponding feature vectors with DISPLAYFORM5 Proof.

According to Cramér's theorem, the error exponent under P i is DISPLAYFORM6 .

With the techniques developed in local information geometry, the above above problem is equivalent to the following problem: DISPLAYFORM7 , it is easy to show that E 1 (λ) = E 1 (λ) when λ = 1 2 .

Then the overall error probability has the exponent as shown in Equation (1).

Given random variables X and Y , the HGR maximal correlation ρ(X; Y ) defined in Equation 2 is a generalization of the Pearson's correlation coefficient to capture non-linear dependence between random variables.

According to BID23 , it satisfies all seven natural postulates of a suitable dependence measure.

Some notable properties are listed below: When the feature dimension is 1, the solution of the maximal HGR correlation is ρ(X; Y ) = σ 1 , the largest singular value of the DTM matrixB. For k-dimensional features, ρ(X; Y ) = k i (σ i ).

However, computingB requires estimating the joint probability P Y X from data, which is inpractical in real applications.

BID4 proposed an efficient algorithm, alternating condition expectation (ACE), inspired by the power method for computing matrix singular values.

Require: training samples {(( DISPLAYFORM0 In Algorithm 1, we first initialize g as a random k-dimensional zero-mean function.

Then iteratively update f (x) and g(y) for all x ∈ X and y ∈ Y .

The conditional distributions on Line 4 and 6 are computed as the empirical average over m samples.

The normalization steps on Lines 5 and 7 can also be implemented using the Gram-Schmidt process.

Note that the ACE algorithm has several variations in previous works, including a kernel-smoothed version BID4 ) and a parallel version with improved efficiency BID13 ).

An alternative formulation that supports continuous X and large feature dimension k has also been proposed recently BID26 ).Next we look at the convergence property of the ACE algorithm.

Let f (X), g(Y ) be the true maximal correlation functions, and letf (X),g(Y ) be estimations computed with Algorithm 1 from m i.i.d.

sampled training data.

Similarly, denote by DISPLAYFORM1 ] the true and estimated maximal correlations, respectively.

Using Sanov's Theorem, we can show that for a small ∆ > 0, the probability that the ratio between the true and estimated maximal correlation is within 1 ± ∆ drops exponentially as the number of samples increases.

Hence the ACE algorithm converges in exponential time.

The following theorem gives the precise sampling complexity for k = 1.

Theorem 2.

For any random variables X and Y with joint probability distribution P Y X , if X and Y are not independent, then for any f : X → R and g : DISPLAYFORM2 for any given ∆ > 0.

To simplify the proof, we first consider the case when the feature function is 1-dimensional.i.e.

f : X → R. We have the following lemma: DISPLAYFORM0 where ξ is the feature vector corresponding to f .Proof.

Since ξ(x) = P X (x)f (x), we have DISPLAYFORM1 The last equality uses the assumption that E[f (x)] = 0.Theorem 3 (1D version of Theorem 1).

Given P X|Y =0 , P X|Y =1 ∈ N X (P 0,X ) and features f : DISPLAYFORM2 we first derive the following properties of the conditional expectations of f (x): DISPLAYFORM3 On the R.H.S. of Equation 5, we apply Lemma 4 to write B ξ DISPLAYFORM4 Next consider the L.H.S. of the equation, by Lemma 2, we have DISPLAYFORM5 2 for some constant c 0 .

DISPLAYFORM6 For k ≥ 2, Lemma 4 can be restated as follows: DISPLAYFORM7 where columns of Ξ are the information vectors corresponding to f 1 (x), . . .

, f k (x).Proof.

First, note that DISPLAYFORM8 T where 1 is a column vector with all entries 1 and length |Y|.

Since E[f (X)] = 0, we havẽ DISPLAYFORM9 It follows that DISPLAYFORM10 Finally, we derive the multi-dimensional case for Theorem 1.Proof of Theorem 1.

Using Lemma 5 and a similar argument as in the simplified proof, the R.H.S of the equation becomes DISPLAYFORM11 By Lemma 3, the L.H.S. of the equation can be written as DISPLAYFORM12 Equation FORMULA51 gives a more understandable expression of the normalization term.

We can also writẽ BΞ as follows:B DISPLAYFORM13 T where 1 is a column vector with all entries 1 and length |Y|, we have DISPLAYFORM14 On the other hand, DISPLAYFORM15 The last equality is derived by substituting (6) and (7) into (8).

In softmax regression, given m training examples {( DISPLAYFORM0 , the cross-entropy loss of the model is DISPLAYFORM1 Minimizing is equivalent to minimizing D P Y X ||P X Q Y |X where P Y X is the joint empirical distribution of (X, Y ).Using information geometry, it can be shown that under a local assumption DISPLAYFORM2 which reveals a close connection between log loss and the modal decomposition ofB. In consequence, it is reasonable to measure the classification performance with B − ΨΦ T 2 F given a pair of (f, θ) associated with (Ψ, Φ).In the context of estimating transferability, we are interested in a one-sided problem, where Φ S is given by the source feature and Ψ becomes the only variable.

Training the network is equivalent to finding the optimal weight Ψ * that minimizes the log-loss.

By taking the derivative of the Objective function with respect to Ψ, we get DISPLAYFORM3 Substituting FORMULA3 in the Objective of FORMULA3 , we can derive the following close-form solution for the log loss.

DISPLAYFORM4 The first term in (15) is fixed given T T while the second term has exactly the form of H-score.

This implies that log loss is negatively linearly related to H-score.

We demonstrates this relationship experimentally, using a collection of synthesized tasks FIG1 ).

In particular, the target task is generated based on a random stochastic matrix P Y0|X , and 20 source tasks are generated with the conditional probability matrix P Yi|X = P Y0|X + iλI for some positive constant λ.

The universal minimum error probability features for each source task are used as the source features f Si (x), while the respective log-loss are obtained through training a simple neural network in Figure 2 with cross-entropy loss.

The relationship is clearly linear with a constant offset.

Proposition 1.

Under the local assumption that DISPLAYFORM0 , whereB of the DTM matrix of X and Y .Proof.

First, we define φ DISPLAYFORM1 and let Φ X|Y ∈ R |X |×|Y| denote its matrix version.

Then we have DISPLAYFORM2 Next, we express the mutual information in terms of information vector φ X|Y , Here we present some detailed results on the comparison between H-score and the affinity score in Zamir et al. (2018) for pairwise transfer.

DISPLAYFORM3 The results of the classification tasks are shown in FIG1 and the results of Depth is shown in 13.We can see in general, although affinity and transferability have totally different value ranges, they tend to agree on the top few ranked tasks.

During the quantization process of the pixel-to-pixel task labels (ground truth images), we are primarily concerned with two factors: computational complexity and information loss.

Too much information loss will lead to bad approximation of the original problems.

On the other hand, having little information loss requires larger cluster size and computation cost.

Figure FORMULA3 shows that even after quantization, much of the information in the images are retained.

To test the sensitivity of the cluster size, we used cluster centroids to recover the ground truth image pixel-by-pixel.

The 3D occlusion Edge detection results on a sample image is shown in FIG1 .

When the cluster number is set to N = 5 (right), most detected Edges in the ground truth image (left) are lost.

We found that N = 16 strikes a good balance between recoverability and computation cost.

<|TLDR|>

@highlight

We present a provable and easily-computable evaluation function that estimates the performance of transferred representations from one learning task to another in task transfer learning.