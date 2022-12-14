Interpreting neural networks is a crucial and challenging task in machine learning.

In this paper, we develop a novel framework for detecting statistical interactions captured by a feedforward multilayer neural network by directly interpreting its learned weights.

Depending on the desired interactions, our method can achieve significantly better or similar interaction detection performance compared to the state-of-the-art without searching an exponential solution space of possible interactions.

We obtain this accuracy and efficiency by observing that interactions between input features are created by the non-additive effect of nonlinear activation functions, and that interacting paths are encoded in weight matrices.

We demonstrate the performance of our method and the importance of discovered interactions via experimental results on both synthetic datasets and real-world application datasets.

Despite their strong predictive power, neural networks have traditionally been treated as "black box" models, preventing their adoption in many application domains.

It has been noted that complex machine learning models can learn unintended patterns from data, raising significant risks to stakeholders BID43 .

Therefore, in applications where machine learning models are intended for making critical decisions, such as healthcare or finance, it is paramount to understand how they make predictions BID6 BID17 .Existing approaches to interpreting neural networks can be summarized into two types.

One type is direct interpretation, which focuses on 1) explaining individual feature importance, for example by computing input gradients BID37 BID34 BID40 or by decomposing predictions BID2 BID36 , 2) developing attention-based models, which illustrate where neural networks focus during inference BID20 BID27 BID47 , and 3) providing model-specific visualizations, such as feature map and gate activation visualizations BID48 BID21 .

The other type is indirect interpretation, for example post-hoc interpretations of feature importance BID32 and knowledge distillation to simpler interpretable models BID7 .It has been commonly believed that one major advantage of neural networks is their capability of modeling complex statistical interactions between features for automatic feature learning.

Statistical interactions capture important information on where features often have joint effects with other features on predicting an outcome.

The discovery of interactions is especially useful for scientific discoveries and hypothesis validation.

For example, physicists may be interested in understanding what joint factors provide evidence for new elementary particles; doctors may want to know what interactions are accounted for in risk prediction models, to compare against known interactions from existing medical literature.

In this paper, we propose an accurate and efficient framework, called Neural Interaction Detection (NID), which detects statistical interactions of any order or form captured by a feedforward neural network, by examining its weight matrices.

Our approach is efficient because it avoids searching over an exponential solution space of interaction candidates by making an approximation of hidden unit importance at the first hidden layer via all weights above and doing a 2D traversal of the input weight matrix.

We provide theoretical justifications on why interactions between features are created at hidden units and why our hidden unit importance approximation satisfies bounds on hidden unit gradients.

Top-K true interactions are determined from interaction rankings by using a special form of generalized additive model, which accounts for interactions of variable order BID46 BID25 .

Experimental results on simulated datasets and real-world datasets demonstrate the effectiveness of NID compared to the state-of-the-art methods in detecting statistical interactions.

The rest of the paper is organized as follows: we first review related work and define notations in Section 2.

In Section 3, we examine and quantify the interactions encoded in a neural network, which leads to our framework for interaction detection detailed in Section 4.

Finally, we study our framework empirically and demonstrate its practical utility on real-world datasets in Section 5.

Statistical interaction detection has been a well-studied topic in statistics, dating back to the 1920s when two-way ANOVA was first introduced BID11 .

Since then, two general approaches emerged for conducting interaction detection.

One approach has been to conduct individual tests for each combination of features BID25 .

The other approach has been to pre-specify all interaction forms of interest, then use lasso to simultaneously select which are important BID42 BID4 BID30 .Notable approaches such as ANOVA and Additive Groves BID39 belong to the first group.

Two-way ANOVA has been a standard method of performing pairwise interaction detection that involves conducting hypothesis tests for each interaction candidate by checking each hypothesis with F-statistics BID45 .

Besides two-way ANOVA, there is also threeway ANOVA that performs the same analyses but with interactions between three variables instead of two; however, four-way ANOVA and beyond are rarely done because of how computationally expensive such tests become.

Specifically, the number of interactions to test grows exponentially with interaction order.

Additive Groves is another method that conducts individual tests for interactions and hence must face the same computational difficulties; however, it is special because the interactions it detects are not constrained to any functional form e.g. multiplicative interactions.

The unconstrained manner by which interactions are detected is advantageous when the interactions are present in highly nonlinear data BID38 .

Additive Groves accomplishes this by comparing two regression trees, one that fits all interactions, and the other that has the interaction of interest forcibly removed.

In interaction detection, lasso-based methods are popular in large part due to how quick they are at selecting interactions.

One can construct an additive model with many different interaction terms and let lasso shrink the coefficients of unimportant terms to zero BID42 .

While lasso methods are fast, they require specifying all interaction terms of interest.

For pairwise interaction detection, this requires O(p 2 ) terms (where p is the number of features), and O(2 p ) terms for higherorder interaction detection.

Still, the form of interactions that lasso-based methods capture is limited by which are pre-specified.

Our approach to interaction detection is unlike others in that it is both fast and capable of detecting interactions of variable order without limiting their functional forms.

The approach is fast because it does not conduct individual tests for each interaction to accomplish higher-order interaction detection.

This property has the added benefit of avoiding a high false positive-, or false discovery rate, that commonly arises from multiple testing BID3 .

The interpretability of neural networks has largely been a mystery since their inception; however, many approaches have been developed in recent years to interpret neural networks in their traditional feedforward form and as deep architectures.

Feedforward neural networks have undergone multiple advances in recent years, with theoretical works justifying the benefits of neural network depth (Telgarsky, 2016; BID23 BID33 and new research on interpreting feature importance from input gradients BID18 BID34 BID40 .

Deep architectures have seen some of the greatest breakthroughs, with the widespread use of attention mechanisms in both convolutional and recurrent architectures to show where they focus on for their inferences BID20 BID27 BID47 .

Methods such as feature map visualization BID48 , de-convolution (Zeiler & Fergus, 2014) , saliency maps BID37 , and many others have been especially important to the vision community for understanding how convolutional networks represent images.

With long short-term memory networks (LSTMs), a research direction has studied multiplicative interactions in the unique gating equations of LSTMs to extract relationships between variables across a sequence BID1 BID28 .

DISPLAYFORM0 Figure 1: An illustration of an interaction within a fully connected feedforward neural network, where the box contains later layers in the network.

The first hidden unit takes inputs from x 1 and x 3 with large weights and creates an interaction between them.

The strength of the interaction is determined by both incoming weights and the outgoing paths between a hidden unit and the final output, y.

Unlike previous works in interpretability, our approach extracts generalized non-additive interactions between variables from the weights of a neural network.

Vectors are represented by boldface lowercase letters, such as x, w; matrices are represented by boldface capital letters, such as W. The i-th entry of a vector w is denoted by w i , and element (i, j) of a matrix W is denoted by W i,j .

The i-th row and j-th column of W are denoted by W i,: and W :,j , respectively.

For a vector w ??? R n , let diag(w) be a diagonal matrix of size n ?? n, where {diag(w)} i,i = w i .

For a matrix W, let |W| be a matrix of the same size where DISPLAYFORM0 Let [p] denote the set of integers from 1 to p.

An interaction, I, is a subset of all input features [p] with |I| ??? 2.

For a vector w ??? R p and I ??? [p], let w I ??? R |I| be the vector restricted to the dimensions specified by I.

1 Consider a feedforward neural network with L hidden layers.

Let p be the number of hidden units in the -th layer.

We treat the input features as the 0-th layer and p 0 = p is the number of input features.

There are L weight matrices DISPLAYFORM0 and L bias vectors b ( ) ??? R p , = 1, 2, . . .

, L. Let ?? (??) be the activation function (nonlinearity), and let w y ??? R p L and b y ??? R be the coefficients and bias for the final output.

Then, the hidden units h ( ) of the neural network and the output y with input x ??? R p can be expressed as: DISPLAYFORM1 We can construct a directed acyclic graph G = (V, E) based on non-zero weights, where we create vertices for input features and hidden units in the neural network and edges based on the non-zero entries in the weight matrices.

See Appendix A for a formal definition.

A statistical interaction describes a situation in which the joint influence of multiple variables on an output variable is not additive BID8 BID39 DISPLAYFORM0 For example, in x 1 x 2 +sin (x 2 + x 3 + x 4 ), there is a pairwise interaction {1, 2} and a 3-way higherorder interaction {2, 3, 4}, where higher-order denotes |I| ??? 3.

Note that from the definition of statistical interaction, a d-way interaction can only exist if all its corresponding (d ??? 1)-interactions exist BID39 .

For example, the interaction {1, 2, 3} can only exist if interactions {1, 2}, {1, 3}, and {2, 3} also exist.

We will often use this property in this paper.

In feedforward neural networks, statistical interactions between features, or feature interactions for brevity, are created at hidden units with nonlinear activation functions, and the influences of the interactions are propagated layer-by-layer to the final output (see Figure 1) .

In this section, we propose a framework to identify and quantify interactions at a hidden unit for efficient interaction detection, then the interactions are combined across hidden units in Section 4.

In feedforward neural networks, any interacting features must follow strongly weighted connections to a common hidden unit before the final output.

That is, in the corresponding directed graph, interacting features will share at least one common descendant.

The key observation is that nonoverlapping paths in the network are aggregated via weighted summation at the final output without creating any interactions between features.

The statement is rigorized in the following proposition and a proof is provided in Appendix A. The reverse of this statement, that a common descendant will create an interaction among input features, holds true in most cases.

Proposition 2 (Interactions at Common Hidden Units).

Consider a feedforward neural network with input feature DISPLAYFORM0 , there exists a vertex v I in the associated directed graph such that I is a subset of the ancestors of v I at the input layer (i.e., = 0).In general, the weights in a neural network are nonzero, in which case Proposition 2 blindly infers that all features are interacting.

For example, in a neural network with just a single hidden layer, any hidden unit in the network can imply up to 2 Wj,: 0 potential interactions, where W j,: 0 is the number of nonzero values in the weight vector W j,: for the j-th hidden unit.

Managing the large solution space of interactions based on nonzero weights requires us to characterize the relative importance of interactions, so we must mathematically define the concept of interaction strength.

In addition, we limit the search complexity of our task by only quantifying interactions created at the first hidden layer, which is important for fast interaction detection and sufficient for high detection performance based on empirical evaluation (see evaluation in Section 5.2 and TAB3 ).Consider a hidden unit in the first layer: ?? w x + b , where w is the associated weight vector and x is the input vector.

While having the weight w i of each feature i, the correct way of summarizing feature weights for defining interaction strength is not trivial.

For an interaction I ???

[p], we propose to use an average of the relevant feature weights w I as the surrogate for the interaction strength: ?? (|w I |), where ?? (??) is the averaging function for an interaction that represents the interaction strength due to feature weights.

We provide guidance on how ?? should be defined by first considering representative averaging functions from the generalized mean family: maximum value, root mean square, arithmetic mean, geometric mean, harmonic mean, and minimum value BID5 .

These options can be narrowed down by accounting for intuitive properties of interaction strength : 1) interaction strength is evaluated as zero whenever an interaction does not exist (one of the features has zero weight); 2) interaction strength does not decrease with any increase in magnitude of feature weights; 3) interaction strength is less sensitive to changes in large feature weights.

While the first two properties place natural constraints on interaction strength behavior, the third property is subtle in its intuition.

Consider the scaling between the magnitudes of multiple feature weights, where one weight has much higher magnitude than the others.

In the worst case, there is one large weight in magnitude while the rest are near zero.

If the large weight grows in magnitude, then interaction strength may not change significantly, but if instead the smaller weights grow at the same rate, then interaction strength should strictly increase.

As a result, maximum value, root mean square, and arithmetic mean should be ruled out because they do not satisfy either property 1 or 3.

Our definition of interaction strength at individual hidden units is not complete without considering their outgoing paths, because an outgoing path of zero weight cannot contribute an interaction to the final output.

To propose a way of quantifying the influence of an outgoing path on the final output, we draw inspiration from Garson's algorithm BID14 BID15 , which instead of computing the influence of a hidden unit, computes the influence of features on the output.

This is achieved by cumulative matrix multiplications of the absolute values of weight matrices.

In the following, we propose our definition of hidden unit influence, then prove that this definition upper bounds the gradient magnitude of the hidden unit with its activation function.

To represent the influence of a hidden unit i at the -th hidden layer, we define the aggregated weight z DISPLAYFORM0 where z ( ) ??? R p .

This definition upper bounds the gradient magnitudes of hidden units because it computes Lipschitz constants for the corresponding units.

Gradients have been commonly used as variable importance measures in neural networks, especially input gradients which compute directions normal to decision boundaries BID34 BID16 BID37 .

Thus, an upper bound on the gradient magnitude approximates how important the variable can be.

A full proof is shown in Appendix C. Lemma 3 (Neural Network Lipschitz Estimation).

Let the activation function ?? (??) be a 1-Lipschitz function.

Then the output y is z DISPLAYFORM1

We now combine our definitions from Sections 3.1 and 3.2 to obtain the interaction strength ?? i (I) of a potential interaction I at the i-th unit in the first hidden layer h DISPLAYFORM0 .(Note that ?? i (I) is defined on a single hidden unit, and it is agnostic to scaling ambiguity within a ReLU based neural network.

In Section 4, we discuss our scheme of aggregating strengths across hidden units, so we can compare interactions of different orders.

In this section, we propose our feature interaction detection algorithm NID, which can extract interactions of all orders without individually testing for each of them.

Our methodology for interaction detection is comprised of three main steps: 1) train a feedforward network with regularization, 2) interpret learned weights to obtain a ranking of interaction candidates, and 3) determine a cutoff for the top-K interactions.

Data often contains both statistical interactions and main effects BID44 .

Main effects describe the univariate influences of variables on an outcome variable.

We study two architectures: MLP and MLP-M. MLP is a standard multilayer perceptron, and MLP-M is an MLP with additional univariate networks summed at the output ( Figure 2 ).

The univariate networks are intended to discourage the modeling of main effects away from the standard MLP, which can create spurious interactions using the main effects.

When training the neural networks, we apply sparsity regularization on the MLP portions of the architectures to 1) suppress unimportant interacting paths and 2) push the modeling of main effects towards any univariate networks.

We note that this approach can also generalize beyond sparsity regularization (Appendix G).

Input:

input-to-first hidden layer weights W (1) , aggregated weights z DISPLAYFORM0 Output: ranked list of interaction candidates DISPLAYFORM1 1: d ??? initialize an empty dictionary mapping interaction candidate to interaction strength 2: for each row w of W (1) indexed by r do 3:for j = 2 to p do

I ??? sorted indices of top j weights in w 5: DISPLAYFORM0 DISPLAYFORM1 (1)Figure 2: Standard feedforward neural network for interaction detection, with optional univariate networksWe design a greedy algorithm that generates a ranking of interaction candidates by only considering, at each hidden unit, the top-ranked interactions of every order, where 2 ??? |I| ??? p, thereby drastically reducing the search space of potential interactions while still considering all orders.

The greedy algorithm (Algorithm 1) traverses the learned input weight matrix W (1) across hidden units and selects only top-ranked interaction candidates per hidden unit based on their interaction strengths (Equation 1 ).

By selecting the top-ranked interactions of every order and summing their respective strengths across hidden units, we obtain final interaction strengths, allowing variable-order interaction candidates to be ranked relative to each other.

For this algorithm, we set the averaging function ?? (??) = min (??) based on its performance in experimental evaluation (Section 5.1).With the averaging function set to min (??), Algorithm 1's greedy strategy automatically improves the ranking of a higher-order interaction over its redundant subsets 2 (for redundancy, see Definition 1).

This allows the higher-order interaction to have a better chance of ranking above any false positives and being captured in the cutoff stage.

We justify this improvement by proving Theorem 4 under a mild assumption.

Theorem 4 (Improving the ranking of higher-order interactions).

Let R be the set of interactions proposed by Algorithm 1 with ?? (??) = min (??), let I ??? R be a d-way interaction where d ??? 3, and let S be the set of subset (d ??? 1)-way interactions of I where |S| = d. Assume that for any hidden unit j which proposed s ??? S ??? R, I will also be proposed at the same hidden unit, and DISPLAYFORM2 .

Then, one of the following must be true: a) ???s ??? S ??? R ranked lower than I, i.e., ??(I) > ??(s), or b) ???s ??? S where s / ??? R.The full proof is included in Appendix D. Under the noted assumption, the theorem in part a) shows that a d-way interaction will improve over one its d ??? 1 subsets in rankings as long as there is no sudden drop from the weight of the (d ??? 1)-way to the d-way interaction at the same hidden units.

We note that the improvement extends to b) as well, when d = |S ??? R| > 1.Lastly, we note that Algorithm 1 assumes there are at least as many first-layer hidden units as there are the true number of non-redundant interactions.

In practice, we use an arbitrarily large number of first-layer hidden units because true interactions are initially unknown.

In order to predict the true top-K interactions DISPLAYFORM0 , we must find a cutoff point on our interaction ranking from Section 4.2.

We obtain this cutoff by constructing a Generalized Additive Model (GAM) with interactions: DISPLAYFORM1 x 1 x 2 + 2 x3+x5+x6 + 2 x3+x4+x5+x7 + sin(x 7 sin(x 8 + x 9 )) + arccos(0.9x 10 ) DISPLAYFORM2 where g i (??) captures the main effects, g i (??) captures the interactions, and both g i and g i are small feedforward networks trained jointly via backpropagation.

We refer to this model as MLP-Cutoff .We gradually add top-ranked interactions to the GAM, increasing K, until GAM performance on a validation set plateaus.

The exact plateau point can be found by early stopping or other heuristic means, and we report DISPLAYFORM3 as the identified feature interactions.

A variant to our interaction ranking algorithm tests for all pairwise interactions.

Pairwise interaction detection has been a standard problem in the interaction detection literature BID25 BID9 due to its simplicity.

Modeling pairwise interactions is also the de facto objective of many successful machine learning algorithms such as factorization machines BID31 and hierarchical lasso BID4 .We rank all pairs of features {i, j} according to their interaction strengths ??({i, j}) calculated on the first hidden layer, where again the averaging function is min (??), and ??({i, j}) = p1 s=1 ?? s ({i, j}).

The higher the rank, the more likely the interaction exists.

In this section, we discuss our experiments on both simulated and real-world datasets to study the performance of our approach on interaction detection.

Averaging Function Our proposed NID framework relies on the selection of an averaging function (Sections 3.1, 4.2, and 4.4).

We experimentally determined the averaging function by comparing representative functions from the generalized mean family BID5 : maximum, root mean square, arithmetic mean, geometric mean, harmonic mean, and minimum, intuitions behind which were discussed in Section 3.1.

To make the comparison, we used a test suite of 10 synthetic functions, which consist of a variety of interactions of varying order and overlap, as shown in TAB2 .

We trained 10 trials of MLP and MLP-M on each of the synthetic functions, obtained interaction rankings with our proposed greedy ranking algorithm (Algorithm 1), and counted the total number of correct interactions ranked before any false positive.

In this evaluation, we ignore predicted interactions that are subsets of true higher-order interactions because the subset interactions are redundant (Section 2).

As seen in FIG0 , the number of true top interactions we recover is highest with the averaging function, minimum, which we will use in all of our experiments.

A simple analytical study on a bivariate hidden unit also suggests that the minimum is closely correlated with interaction strength (Appendix B).

TAB2 ) over 10 trials.

x-axis labels are maximum, root mean square, arithmetic mean, geometric mean, harmonic mean, and minimum.

Neural Network Configuration We trained feedforward networks of MLP and MLP-M architectures to obtain interaction rankings, and we trained MLP-Cutoff to find cutoffs on the rankings.

In our experiments, all networks that model feature interactions consisted of four hidden layers with first-to-last layer sizes of: 140, 100, 60, and 20 units.

In contrast, all individual univariate networks had three hidden layers with sizes of: 10, 10, and 10 units.

All networks used ReLU activation and were trained using backpropagation.

In the cases of MLP-M and MLP-Cutoff , summed networks were trained jointly.

The objective functions were meansquared error for regression and cross-entropy for classification tasks.

On the synthetic test suite, MLP and MLP-M were trained with L1 constants in the range of 5e-6 to 5e-4, based on parameter tuning on a validation set.

On real-world datasets, L1 was fixed at 5e-5.

MLP-Cutoff used a fixed L2 constant of 1e-4 in all experiments involving cutoff.

Early stopping was used to prevent overfitting.

Datasets We study our interaction detection framework on both simulated and real-world experiments.

For simulated experiments, we used a test suite of synthetic functions, as shown in TAB2 .

The test functions were designed to have a mixture of pairwise and higher-order interactions, with varying order, strength, nonlinearity, and overlap.

F 1 is a commonly used function in interaction detection literature BID19 BID39 BID25 .

All features were uniformly distributed between ???1 and 1 except in F 1 , where we used the same variable ranges as reported in literature BID19 .

In all synthetic experiments, we used random train/valid/test splits of 1/3 each on 30k data points.

We use four real-world datasets, of which two are regression datasets, and the other two are binary classification datasets.

The datasets are a mixture of common prediction tasks in the cal housing and bike sharing datasets, a scientific discovery task in the higgs boson dataset, and an example of very-high order interaction detection in the letter dataset.

Specifically, the cal housing dataset is a regression dataset with 21k data points for predicting California housing prices BID29 .

The bike sharing dataset contains 17k data points of weather and seasonal information to predict the hourly count of rental bikes in a bikeshare system BID10 .

The higgs boson dataset has 800k data points for classifying whether a particle environment originates from the decay of a Higgs Boson BID0 .

Lastly, the letter recognition dataset contains 20k data points of transformed features for binary classification of letters on a pixel display BID12 .

For all real-world data, we used random train/valid/test splits of 80/10/10.Baselines We compare the performance of NID to that of three baseline interaction detection methods.

Two-Way ANOVA (Wonnacott & Wonnacott, 1972) utilizes linear models to conduct significance tests on the existence of interaction terms.

Hierarchical lasso (HierLasso) BID4 applies lasso feature selection to extract pairwise interactions.

RuleFit BID13 contains a statistic to measure pairwise interaction strength using partial dependence functions.

Additive Groves (AG) BID39 ) is a nonparameteric means of testing for interactions by placing structural constraints on an additive model of regression trees.

AG is a reference method for interaction detection because it directly detects interactions based on their non-additive definition.

As discussed in Section 4, our framework NID can be used for pairwise interaction detection.

To evaluate this approach, we used datasets generated by synthetic functions F 1 -F 10 ( TAB2 that contain a mixture of pairwise and higher-order interactions, where in the case of higher-order interactions we tested for their pairwise subsets as in BID39 ; BID25 .

AUC scores of interaction strength proposed by baseline methods and NID for both MLP and MLP-M are shown in TAB3 .

We ran ten trials of AG and NID on each dataset and removed two trials with highest and lowest AUC scores.

When comparing the AUCs of NID applied to MLP and MLP-M, we observe that the scores of MLP-M tend to be comparable or better, except the AUC for F 6 .

On one hand, MLP-M performed better on F 2 and F 4 because these functions contain main effects that MLP would model as spurious interactions with other variables.

On the other hand, MLP-M performed worse on F 6 because it modeled spurious main effects in the {8, 9, 10} interaction.

Specifically, {8, 9, 10} can be approximated as independent parabolas for each variable (shown in Appendix I).

In our analyses of NID, we mostly focus on MLP-M because handling main effects is widely considered an important problem in interaction detection BID4 BID24 BID22 .

Comparing the AUCs of AG and NID for MLP-M, the scores tend to close, except for F 5 , F 6 , and F 8 , where NID performs significantly better than AG.

This performance difference may be due to limitations on the model capacity of AG, which is tree-based.

In comparison to ANOVA, HierLasso and RuleFit, NID-MLP-M generally performs on par or better.

This is expected for ANOVA and HierLasso because they are based on quadratic models, which can have difficulty approximating the interaction nonlinearities present in the test suite.

In Figure 4 , heat maps of synthetic functions show the relative strengths of all possible pairwise interactions as interpreted from MLP-M, and ground truth is indicated by red cross-marks.

The interaction strengths shown are normally high at the cross-marks.

An exception is F 6 , where NID proposes weak or negligible interaction strengths at the cross-marks corresponding to the {8, 9, 10} interaction, which is consistent with previous remarks about this interaction.

Besides F 6 , F 7 also shows erroneous interaction strengths; however, comparative detection performance by the baselines is similarly poor.

Interaction strengths are also visualized on real-world datasets via heat maps ( FIG1 ).

For example, in the cal housing dataset, there is a high-strength interaction between x 1 and x 2 .

These variables mean longitude and latitude respectively, and it is clear to see that the outcome variable, California housing price, should indeed strongly depend on geographical location.

We further observe high-strength interactions appearing in the heat maps of the bike sharing, higgs boson dataset, and letter datasets.

For example, all feature pairs appear to be interacting in the letter dataset.

The binary classification task from the letter dataset is to distinguish letters A-M from N-Z using 16 pixel display features.

Since the decision boundary between A-M and N-Z is not obvious, it would make sense that a neural network learns a highly interacting function to make the distinction.

We use our greedy interaction ranking algorithm (Algorithm 1) to perform higher-order interaction detection without an exponential search of interaction candidates.

We first visualize our higherorder interaction detection algorithm on synthetic and real-world datasets, then we show how the predictive capability of detected interactions closes the performance gap between MLP-Cutoff and MLP-M. Next, we discuss our experiments comparing NID and AG with added noise, and lastly we verify that our algorithm obtains significant improvements in runtime.x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 TAB2 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 DISPLAYFORM0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 DISPLAYFORM1 Figure 4: Heat maps of pairwise interaction strengths proposed by our NID framework on MLP-M for datasets generated by functions F 1 -F 10 ( We visualize higher-order interaction detection on synthetic and real-world datasets in Figures 6 and 7 respectively.

The plots correspond to higher-order interaction detection as the ranking cutoff is applied (Section 4.3).

The interaction rankings generated by NID for MLP-M are shown on the x-axes, and the blue bars correspond to the validation performance of MLP-Cutoff as interactions are added.

For example, the plot for cal housing shows that adding the first interaction significantly reduces RMSE.

We keep adding interactions into the model until reaching a cutoff point.

In our experiments, we use a cutoff heuristic where interactions are no longer added after MLP-Cutoff 's validation performance reaches or surpasses MLP-M's validation performance (represented by horizontal dashed lines).As seen with the red cross-marks, our method finds true interactions in the synthetic data of F 1 -F 10 before the cutoff point.

Challenges with detecting interactions are again mainly associated with F 6 and F 7 , which have also been difficult for baselines in the pairwise detection setting TAB3 .

For the cal housing dataset, we obtain the top interaction {1, 2} just like in our pairwise test ( FIG1 , cal housing), where now the {1, 2} interaction contributes a significant improvement in MLP-Cutoff performance.

Similarly, from the letter dataset we obtain a 16-way interaction, which is consistent with its highly interacting pairwise heat map ( FIG1 .

For the bike sharing and higgs boson datasets, we note that even when considering many interactions, MLP-Cutoff eventually reaches the cutoff point with a relatively small number of superset interactions.

This is because many subset interactions become redundant when their corresponding supersets are found.

In our evaluation of interaction detection on real-world data, we study detected interactions via their predictive performance.

By comparing the test performance of MLP-Cutoff and MLP-M with respect to MLP-Cutoff without any interactions (MLP-Cutoff ?? ), we can compute the relative test performance improvement obtained by including detected interactions.

These relative performance improvements are shown in TAB7 for the real-world datasets as well as four selected synthetic datasets, where performance is averaged over ten trials per dataset.

The results of this study show DISPLAYFORM2 Figure 6: MLP-Cutoff error with added top-ranked interactions (along x-axis) of F 1 -F 10 TAB2 , where the interaction rankings were generated by the NID framework applied to MLP-M. Red crossmarks indicate ground truth interactions, and ?? denotes MLP-Cutoff without any interactions.

Subset interactions become redundant when their true superset interactions are found.

TAB2 , where the interaction rankings were generated by the NID framework on MLP-M. ?? denotes MLP-Cutoff without any interactions.that a relatively small number of interactions of variable order are highly predictive of their corresponding datasets, as true interactions should.

We further study higher-order interaction detection of our NID framework by comparing it to AG in both interaction ranking quality and runtime.

To assess ranking quality, we design a metric, toprank recall, which computes a recall of proposed interaction rankings by only considering those interactions that are correctly ranked before any false positive.

The number of top correctly-ranked interactions is then divided by the true number of interactions.

Because subset interactions are redundant in the presence of corresponding superset interactions, only such superset interactions can count as true interactions, and our metric ignores any subset interactions in the ranking.

We compute the top-rank recall of NID on MLP and MLP-M, the scores of which are averaged across all tests in the test suite of synthetic functions TAB2 with 10 trials per test function.

For each test, we remove two trials with max and min recall.

We conduct the same tests using the stateof-the-art interaction detection method AG, except with only one trial per test because AG is very computationally expensive to run.

In FIG4 , we show top-rank recall of NID and AG at different Gaussian noise levels 3 , and in FIG4 , we show runtime comparisons on real-world and synthetic datasets.

As shown, NID can obtain similar top-rank recall as AG while running orders of magnitude times faster.

In higher-order interaction detection, our NID framework can have difficulty detecting interactions from functions with interlinked interacting variables.

For example, a clique x 1 x 2 +x 1 x 3 +x 2 x 3 only TAB2 , (b) comparison of runtimes, where NID runtime with and without cutoff are both measured.

NID detects interactions with top-rank recall close to the state-of-the-art AG while running orders of magnitude times faster.contains pairwise interactions.

When detecting pairwise interactions (Section 5.2), NID often obtains an AUC of 1.

However, in higher-order interaction detection, the interlinked pairwise interactions are often confused for single higher-order interactions.

This issue could mean that our higherorder interaction detection algorithm fails to separate interlinked pairwise interactions encoded in a neural network, or the network approximates interlinked low-order interactions as higher-order interactions.

Another limitation of our framework is that it sometimes detects spurious interactions or misses interactions as a result of correlations between features; however, correlations are known to cause such problems for any interaction detection method BID39 BID25 .

We presented our NID framework, which detects statistical interactions by interpreting the learned weights of a feedforward neural network.

The framework has the practical utility of accurately detecting general types of interactions without searching an exponential solution space of interaction candidates.

Our core insight was that interactions between features must be modeled at common hidden units, and our framework decoded the weights according to this insight.

In future work, we plan to detect feature interactions by accounting for common units in intermediate hidden layers of feedforward networks.

We would also like to use the perspective of interaction detection to interpret weights in other deep neural architectures.

A PROOF AND DISCUSSION FOR PROPOSITION 2Given a trained feedforward neural network as defined in Section 2.3, we can construct a directed acyclic graph G = (V, E) based on non-zero weights as follows.

We create a vertex for each input feature and hidden unit in the neural network: V = {v ,i |???i, }, where v ,i is the vertex corresponding to the i-th hidden unit in the -th layer.

Note that the final output y is not included.

We create edges based on the non-zero entries in the weight matrices, i.e., DISPLAYFORM0 Note that under the graph representation, the value of any hidden unit is a function of parent hidden units.

In the following proposition, we will use vertices and hidden units interchangeably.

Proposition 2 (Interactions at Common Hidden Units).

Consider a feedforward neural network with input feature DISPLAYFORM1 , there exists a vertex v I in the associated directed graph such that I is a subset of the ancestors of v I at the input layer (i.e., = 0).Proof.

We prove Proposition 2 by contradiction.

Let I be an interaction where there is no vertex in the associated graph which satisfies the condition.

Then, for any vertex v L,i at the L-th layer, the value f i of the corresponding hidden unit is a function of its ancestors at the input layer I i where I ??? I i .Next, we group the hidden units at the L-th layer into non-overlapping subsets by the first missing feature with respect to the interaction I. That is, for element i in I, we create an index set S i ??? [p L ]: DISPLAYFORM2 Note that the final output of the network is a weighed summation over the hidden units at the L-th layer: DISPLAYFORM3 Since that j???Si w y j f j x Ij is not a function of x i , we have that ?? (??) is a function without the interaction I, which contradicts our assumption.

The reverse of this statement, that a common descendant will create an interaction among input features, holds true in most cases.

The existence of counterexamples is manifested when early hidden layers capture an interaction that is negated in later layers.

For example, the effects of two interactions may be directly removed in the next layer, as in the case of the following expression: max{w 1 x 1 + w 2 x 2 , 0} ??? max{???w 1 x 1 ??? w 2 x 2 , 0} = w 1 x 1 + w 2 x 2 .

Such an counterexample is legitimate; however, due to random fluctuations, it is highly unlikely in practice that the w 1 s and the w 2 s from the left hand side are exactly equal.

We can provide a finer interaction strength analysis on a bivariate ReLU function: max{?? 1 x 1 + ?? 2 x 2 , 0}, where x 1 , x 2 are two variables and ?? 1 , ?? 2 are the weights for this simple network.

We quantify the strength of the interaction between x 1 and x 2 with the cross-term coefficient of the best quadratic approximation.

That is, ?? 0 , . . .

, ?? 5 = argmin ??i,i=0,...,5 DISPLAYFORM0 Then for the coefficient of interaction {x 1 , x 2 }, ?? 5 , we have that, DISPLAYFORM1 Note that the choice of the region (???1, 1) ?? (???1, 1) is arbitrary: for larger region (???c, c) ?? (???c, c) with c > 1, we found that |?? 5 | scales with c ???1 .

By the results of Proposition B, the strength of the interaction can be well-modeled by the minimum value between |?? 1 | and |?? 2 |.

Note that the factor before min{|?? 1 |, |?? 2 |} in Equation FORMULA13 Proof.

For non-differentiable ?? (??) such as the ReLU function, we can replace it with a series of differentiable 1-Lipschitz functions that converges to ?? (??) in the limit.

Therefore, without loss of generality, we assume that ?? (??) is differentiable with |??? x ??(x)| ??? 1.

We can take the partial derivative of the final output with respect to h ( ) i , the i-th unit at the -th hidden layer: DISPLAYFORM2 We can conclude the Lemma by proving the following inequality: DISPLAYFORM3 The left-hand side can be re-written as DISPLAYFORM4 The right-hand side can be re-written as DISPLAYFORM5 We can conclude by noting that |??? x ??(x)| ??? 1.

Theorem 4 (Improving the ranking of higher-order interactions).

Let R be the set of interactions proposed by Algorithm 1 with ?? (??) = min (??), let I ??? R be a d-way interaction where d ??? 3, and let S be the set of subset (d ??? 1)-way interactions of I where |S| = d. Assume that for any hidden unit j which proposed s ??? S ??? R, I will also be proposed at the same hidden unit, and DISPLAYFORM0 .

Then, one of the following must be true: a) ???s ??? S ??? R ranked lower than I, i.e., ??(I) > ??(s), or b) ???s ??? S where s / ??? R.Proof.

Suppose for the purpose of contradiction that S ??? R and ???s ??? S, ??(s) ??? ??(I).

Because DISPLAYFORM1 which is a contradiction.

E ROC CURVES We evaluate our approach in a large p setting with pairwise interactions using the same synthetic function as in BID30 .

Specifically, we generate a dataset of n samples and p features { X (i) , y (i) } using the function DISPLAYFORM2 where X (i) ??? R p is the i th instance of the design matrix X ??? R p??n , y (i) ??? R is the i th instance of the response variable y ??? R n??1 , W ??? R p??p contains the weights of pairwise interactions, ?? ??? R p contains the weights of main effects, (i) is noise, and i = 1, . . .

, n. W was generated as a sum of K rank one matrices, W = K k=1 a k a k .

In this experiment, we set p = 1000, n = 1e4, and K = 5.

X is normally distributed with mean 0 and variance 1, and (i) is normally distributed with mean 0 and variance 0.1.

Both a k and ?? are sparse vectors of 2% nonzero density and are normally distributed with mean 0 and variance 1.

We train MLP-M with the same hyperparameters as before (Section 5.1) but with a larger main network architecture of five hidden layers, with first-to-last layers sizes of 500, 400, 300, 200, and 100.

Interactions are then extracted using the NID framework.

From this experiment, we obtain a pairwise interaction strength AUC of 0.984 on 950 ground truth pairwise interactions, where the AUC is measured in the same way as those in TAB3 .

The corresponding ROC curve is shown in Figure 10 .

We compare the average performance of NID for different regularizations on MLP-M networks.

Specifically, we compare interaction detection performance when an MLP-M network has L1, L2, or group lasso regularization 4 .

While L1 and L2 are common methods for regularizing neural network weights, group lasso is used to specifically regularize groups in the input weight matrix because weight connections into the first hidden layer are especially important in this work.

In particular, we study group lasso by 1) forming groups associated with input features, and 2) forming groups associated with hidden units in the input weight matrix.

In this experimental setting, group lasso effectively conducts variable selection for associated groups.

BID35 define group lasso regularization for neural networks in Equation 5.

Denote group lasso with input groups a R (i) GL and group lasso with hidden unit groups as R (h) GL .

In order to apply both group and individual level sparsity, BID35 further define sparse group lasso in Equation 7.

Denote sparse group lasso with input groups as R (i) SGL and sparse group lasso with hidden unit groups as R (h)

Networks that had group lasso or sparse group lasso applied to the input weight matrix had L1 regularization applied to all other weights.

In our experiments, we use large dataset sizes of 1e5 and tune the regularizers by gradually increasing their respective strengths from zero until validation performance worsens from underfitting.

The rest of the neural network hyperparameters are the same as those discussed in Section 5.1.

In the case of the group lasso and sparse group lasso experiments, L1 norms were tuned the same as in the standard L1 experiments.

In Table 4 , we report average pairwise interaction strength AUC over 10 trials of each function in our synthetic test suite TAB2 for the different regularizers.

Table 4 : Average AUC of pairwise interaction strengths proposed by NID for different regularizers.

Evaluation was conducted on the test suite of synthetic functions TAB2 .

DISPLAYFORM0 average 0.94 ?? 2.9e???2 0.94 ?? 2.5e???2 0.95 ?? 2.4e???2 0.94 ?? 2.5e???2 0.93 ?? 3.2e???2 0.94 ?? 3.0e???2

We perform experiments with our NID approach on synthetic datasets that have binary class labels as opposed to continuous outcome variables (e.g. TAB2 ).

In our evaluation, we compare our method against two logistic regression methods for multiplicative interaction detection, Factorization Based High-Order Interaction Model (FHIM) BID30 and Sparse High-Order Logistic Regression (Shooter) .

In both comparisons, we use dataset sizes of p = 200 features and n = 1e4 samples based on MLP-M's fit on the data and the performance of the baselines.

We also make the following modifications to MLP-M hyperparameters based on validation performance: the main MLP-M network has first-to-last layer sizes of 100, 60, 20 hidden units, the univariate networks do not have any hidden layers, and the L1 regularization constant is set to 5e???4.

All other hyperparameters are kept the same as in Section 5.1.

When used in a logistic regression model, FHIM detects pairwise interactions that are predictive of binary class labels.

For this comparison, we used data generated by Equation 5 in BID30 , with K = 2 and sparsity factors being 5% to generate 73 ground truth pairwise interactions.

In TAB8 , we report average pairwise interaction detection AUC over 10 trials, with a maximum and a minimum AUC removed.

FHIM NID average 0.925 ?? 2.3e???3 0.973 ?? 6.1e???3Shooter Min et al. FORMULA13 developed Shooter, an approach of using a tree-structured feature expansion to identify pairwise and higher-order multiplicative interactions in a L1 regularized logistic regression model.

This approach is special because it relaxes our hierarchical hereditary assumption, which requires subset interactions to exist when their corresponding higher-order interaction also exists (Section 3).

Specifically, Shooter relaxes this assumption by only requiring at least one (d ??? 1)-way interaction to exist when its corresponding d-way interaction exists.

With this relaxed assumption, Shooter can be evaluated in depth per level of interaction order.

We compare NID and Shooter under the relaxed assumption by also evaluating NID per level of interaction order, where Algorithm 1 is specifically being evaluated.

We note that our method of finding a cutoff on interaction rankings (Section 4.3) strictly depends on the hierarchical hereditary assumption both within the same interaction order and across orders, so instead we set cutoffs by thresholding the interaction strengths by a low value, 1e???3.For this comparison, we generate and consider interaction orders up to degree 5 (5-way interaction) using the procedure discussed in , where the interactions do not have strict hierarchical correspondence.

We do not extend beyond degree 5 because MLP-M's validation performance begins to degrade quickly on the generated dataset.

The sparsity factor was set to be 5%, and to simplify the comparison, we did not add noise to the data.

In TAB9 , we report precision and recall scores of Shooter and NID, where the scores for NID are averaged over 10 trials.

While Shooter performs near perfectly, NID obtains fair precision scores but generally low recall.

When we observe the interactions identified by NID per level of interaction order, we find that the interactions across levels are always subsets or supersets of another predicted interaction.

This strict hierarchical correspondence would inevitably cause NID to miss many true interactions under this experimental setting.

The limitation of Shooter is that it must assume the form of interactions, which in this case is multiplicative.

87.5% 100% 90% ?? 11% 21% ?? 9.6% 3 96.0% 96.0% 91% ?? 8.4% 19% ?? 8.5% 4 100% 100% 60% ?? 13% 21% ?? 9.3% 5 100% 100% 73% ?? 8.4% 30% ?? 13%

In the synthetic function F 6 TAB3 , the {8, 9, 10} interaction, x 2 8 + x 2 9 + x 2 10 , can be approximated as main effects for each variable x 8 , x 9 , and x 10 when at least one of the three variables is close to ???1 or 1.

Note that in our experiments, these variables were uniformly distributed between ???1 and 1. , and x 10 .

The MLP-M was trained on data generated from synthetic function F 6 ( TAB3 ).

Note that the plots are subject to different levels of bias from the MLP-M's main multivariate network.

For example, let x 10 = 1 and z 2 = x 2 8 + x By symmetry under the assumed conditions, where c is a constant.

In FIG10 , we visualize the x 8 , x 9 , x 10 univariate networks of a MLP-M (Figure 2 ) that is trained on F 6 .

The plots confirm our hypothesis that the MLP-M models the {8,9,10} interaction as spurious main effects with parabolas scaled by In FIG13 , we visualize the interaction between longitude and latitude for predicting relative housing price in California.

This visualization is extracted from the longitude-latitude interaction network within MLP-Cutoff 5 , which was trained on the cal housing dataset BID29 .

We can see that this visualization cannot be generated by longitude and latitude information in additive form, but rather the visualization needs special joint information from both features.

The highly interacting nature between longitude and latitude confirms the high rank of this interaction in our NID experiments (see the {1, 2} interaction for cal housing in FIG1 .

<|TLDR|>

@highlight

We detect statistical interactions captured by a feedforward multilayer neural network by directly interpreting its learned weights.