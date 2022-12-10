Recommendation is a prevalent application of machine learning that affects many users; therefore, it is crucial for recommender models to be accurate and interpretable.

In this work, we propose a method to both interpret and augment the predictions of black-box recommender systems.

In particular, we propose to extract feature interaction interpretations from a source recommender model and explicitly encode these interactions in a target recommender model, where both source and target models are black-boxes.

By not assuming the structure of the recommender system, our approach can be used in general settings.

In our experiments, we focus on a prominent use of machine learning recommendation: ad-click prediction.

We found that our interaction interpretations are both informative and predictive, i.e., significantly outperforming existing recommender models.

What's more, the same approach to interpreting interactions can  provide new insights into domains even beyond recommendation.

Despite their impact on users, state-of-the-art recommender systems are becoming increasingly inscrutable.

For example, the models that predict if a user will click on an online advertisement are often based on function approximators that contain complex components in order to achieve optimal recommendation accuracy.

The complex components come in the form of modules for better learning relationships among features, such as interactions between user and ad features (Cheng et al., 2016; Guo et al., 2017; Wang et al., 2017; Lian et al., 2018; Song et al., 2018) .

Although efforts have been made to understand the feature relationships, there is still no method that can interpret the feature interactions learned by a generic recommender system, nor is there a strong commercial incentive to do so.

In this work, we identify and leverage feature interactions that represent how a recommender system generally behaves.

We propose a novel approach, Global Interaction Detection and Encoding for Recommendation (GLIDER), which detects feature interactions that span globally across multiple data-instances from a source recommender model, then explicitly encodes the interactions in a target recommender model, both of which can be black-boxes.

GLIDER achieves this by first utilizing feature interaction detection with a data-instance level interpretation method called LIME (Ribeiro et al., 2016 ) over a batch of data samples.

GLIDER then explicitly encodes the collected global interactions into a target model via sparse feature crossing.

In our experiments on ad-click recommendation, we found that the interpretations generated by GLIDER are informative, and the detected global interactions can significantly improve the target model's prediction performance, even in a setting where the source and target models are the same.

Because our interaction interpretation method is very general, we also show that the interpretations are informative in domains outside of recommendation, such as image and text classification.

Our contributions are as follows:

1.

We propose GLIDER to detect and explicitly encode global feature interactions in blackbox recommender systems.

Through experiments, we demonstrate the overall interpretability of detected feature interactions and show that they can be leveraged to improve recommendation accuracy.

A simplified overview of GLIDER.

1 GLIDER uses interaction detection and LIME together to interpret feature interactions learned by a source black-box (recommender) model at a data instance, denoted by the large green plus sign.

2 GLIDER identifies interactions that consistently appear over multiple data samples, then explicitly encodes these interactions in a target black-box recommender model f rec .

Interaction Interpretations: A variety of methods exist to detect feature interactions learned in specific models but not black-box models.

For example, RuleFit (Friedman et al., 2008) , Additive Groves (Sorokina et al., 2008) , and Tree-Shap (Lundberg et al., 2018) detect interactions learned in trees, and Neural Interaction Detection (Tsang et al., 2017) detects interactions learned in a multilayer perceptron.

Some methods have attempted to interpret feature groups in black-box models, such as Anchors (Ribeiro et al., 2018) , Agglomerative Contextual Decomposition (Singh et al., 2019) , and Context-Aware methods (Singla et al., 2019) ; however, these methods were not intended to identify feature interactions.

Explicit Interaction Representation: There are increasingly methods for explicitly representing interactions in models.

Cheng et al. (2016) , Guo et al. (2017) , Wang et al. (2017) , and Lian et al. (2018) directly incorporate multiplicative cross terms in neural network architectures and Song et al. (2018) use attention as an interaction module, all of which are intended to improve the neural network's function approximation.

This line of work found that predictive performance can improve with dedicated interaction modeling.

Luo et al. (2019) followed up by detecting interactions in data then explicitly encoding them via feature crossing.

Our work approaches this problem from a model interpretation standpoint to show that interaction interpretations are also useful in explicit encoding.

Black-Box Local vs. Global Interpretations: Data-instance level local interpretation methods are more flexible at explaining general black-box models; however, global interpretations, which cover multiple data instances, have become increasingly desirable to better summarize model behavior.

Locally Interpretable Model-Agnostic Explanations (LIME) (Ribeiro et al., 2016) and Integrated Gradients (Sundararajan et al., 2017) are some of most used methods to locally interpret any classifier and neural predictor respectively.

There are some methods for global black-box interpretations, such as shuffle-based feature importance (Fisher et al., 2018) , submodular pick (Ribeiro et al., 2016) , and visual concept extraction (Kim et al., 2018) .

§4.1 of this paper discusses local interaction interpretations, and §4.2-4.4 explains how we extract and utilize global interaction interpretations.

Notations: Vectors are represented by boldface lowercase letters, such as x or w. The i-th entry of a vector x is denoted by x i .

For a set S, its cardinality is denoted by |S|.

Let d be the number of features in a dataset.

An interaction, I, is the indices of a feature subset: I ⊆ {1, 2, . . .

, d}, where interaction order |I| is always ≥ 2.

A higher-order interaction always has order ≥ 3.

For a vector x ∈ R d , let x I ∈ R |I| be restricted to the dimensions of x specified by I.

Let a black-box model be f (·) : R p → R. A black-box recommender model uses tabular feature types, as discussed later in this section.

In classification tasks, we assume f is a class logit.

p and d may be different depending on feature transformations.

Feature Interactions: By definition, a model f learns a statistical (non-additive) feature interaction I if and only if f cannot be decomposed into a sum of |I| arbitrary subfunctions δ i , each not depending on a corresponding interaction variable (Friedman et al., 2008; Sorokina et al., 2008; Tsang et al., 2017) , i.e., f (x) = i∈I δ i (x {1,...,d}\i ).

For example, a multiplication between two features, x 1 and x 2 , is a feature interaction because it cannot be represented as an addition of univariate functions, i.e., x 1 x 2 = δ 1 (x 2 ) + δ 2 (x 1 ).

Recommendation Systems: A recommender system, f rec (·), is a model of two feature types: dense numerical features and sparse categorical features.

Since the one-hot encoding of categorical feature x c can be high-dimensional, it is commonly represented in a low-dimensional embedding e c = one hot(x c )v c via embedding matrix v c .

We now discuss the different components of GLIDER, starting from data-instance level (local) interpretations of interactions in §4.1, then global interaction detection in §4.2, and finally explicitly encoding the global interactions in §4.3.

While our methodology is focused on recommender systems, it is not necessarily limited to this model type.

Nonetheless, recommender systems are interesting because they have pervasive application in real-world systems, and their features are often very sparse.

By sparse features, we mean features with many categories, e.g., millions of user IDs.

The sparsity makes interaction detection challenging especially when applied directly on raw data because the one-hot encoding of sparse features creates an extremely large space of potential feature combinations (Fan et al., 2015) .

We start by explaining how to detect feature interactions in a black-box model at the data-instance level via interaction detection on feature perturbations.

LIME Perturbation and Inference: Given a data instance x ∈ R p , LIME proposed to perturb the data instance by sampling a separate binary representation x ∈ {0, 1} d of the same data instance.

Let ξ : {0, 1} d → R p be the map from the binary representation to the perturbed data instance.

Starting from a binary vector of all ones that map to the original features values in the data instance, LIME uniformly samples the number of random features to switch to 0 or the "off" state.

In the data instance, "off" could correspond to a 0 embedding vector for categorical features or mean value over a batch for numerical features.

It is possible for d < p by grouping features in the data instance to correspond to single binary features in x .

A key step in LIME interpretations is obtaining black-box predictions for the perturbed data instances to generate a dataset with binary inputs and prediction targets:

Feature Interaction Detection: Feature interaction detection is concerned with identifying feature interactions in a dataset (Bien et al., 2013; Purushotham et al., 2014; Lou et al., 2013; Friedman et al., 2008) .

Typically, proper interaction detection requires an expensive pre-processing step of feature selection to remove correlated features that adversely affect detection performance (Sorokina et al., 2008) .

Since the features in D are sampled randomly, they are uncorrelated by default, so we can directly use dataset D to detect feature interactions from black-box model f at a data instance x.

f can be an arbitrary function and can generate highly nonlinear targets in D, so we focus on detecting interactions that could have generic forms.

In light of this, we use a state-of-the-art method called Neural Interaction Detection (NID), which accurately and efficiently detects generic nonadditive and arbitrary-order statistical feature interactions (Tsang et al., 2017) .

NID detects these interactions by training a lasso-regularized multilayer perceptron (MLP) on a dataset, then identifying the features that have high-magnitude weights to common hidden units.

NID is efficient by greedily testing the top-interaction candidates of every order at each of h first-layer hidden units, enabling arbitrary-order interaction detection in O(hd) tests within one MLP.

We can now define a function, MADEX(f, x), that inputs black-box f and data instance x, and outputs

, a set of top-k detected feature interactions.

MADEX stands for "ModelAgnostic Dependency Explainer".

As the name suggests, MADEX is not limited to recommender models; it can also be used for general black-box models.

In some cases, it is necessary to identify a k threshold.

Because of the importance of speed for local interpretations, we simply use a linear regression with additional multiplicative terms to approximate

global interactions I i and their counts c i over the dataset 1: G ← initialize occurrence dictionary for global interactions 2: for each data sample x within dataset B do 3:

G ← increment the occurrence count of I j ∈ S, ∀j = 1, 2, . . .

, |S| 5: sort G by most frequently occurring interactions 6: [optional] prune subset interactions in G within a target number of interactions K the gains given by interactions in S, where k starts at 0 and is incremented until the linear model's predictions stop improving.

In this section, we discuss the first step of GLIDER.

As defined in §4.1, MADEX takes as input a blackbox model f and data instance x. In the context of this section, MADEX inputs a source recommender system f rec and data instance x = [x 1 , x 2 , . . .

, x p ].

x i is the i-th feature field and is either a dense or sparse feature.

p is both the total number of feature fields and the number of perturbation variables (p = d).

We define global interaction detection as repeatedly running MADEX over a batch of data instances, then counting the occurrences of the same detected interactions, shown in Alg.

1.

The occurrence counts are not only a useful way to rank global interaction detections, but also a sanity check to rule out the chance that the detected feature combinations are random selections.

One potential concern with Alg.

1 is that it could be slow depending on the speed of MADEX.

In our experiments, the entire process took less than 3 hours when run in serial over a batch of 1000 samples with ∼ 40 features on a 32-CPU server.

In addition, this algorithm is fully parallelizable and only needs to be run once to obtain the summary of global interactions.

Each global interaction I i from Alg.

1 is used to create a synthetic feature x Ii for a target recommender system.

The synthetic feature x Ii is created by explicitly crossing sparse features indexed in I i .

If interaction I i involves dense features, we bucketize the dense features before crossing them.

The synthetic feature is sometimes called a cross feature (Wang et al., 2017; Luo et al., 2019) or conjunction feature (Rosales et al., 2012; Chapelle et al., 2015) .

In this context, a cross feature is an n-ary Cartesian product among n sparse features.

If we denote X 1 , X 2 , . . .

, X n as the set of IDs for each respective feature x 1 , x 2 , . . .

, x n , then their cross feature x {1,...,n} takes on all possible values in

Accordingly, the cardinality of this cross feature is |X 1 | × · · · × |X n | and can be extremely large, yet many combinations of values in the cross feature are likely unseen in the training data.

Therefore, we generate a truncated form of the cross feature with only seen combinations of values, x (j) I , where j is a sample index in the training data, and x (j) I is represented as a sparse ID in the cross feature x I .

We further reduce the cardinality by requiring the same cross feature ID to occur more than T times in a batch of samples, or set to a default ID otherwise.

These truncation steps significantly reduce the embedding sizes of each cross feature while maintaining their representation power.

Once cross features {x Ii } i are included in a target recommender system, it can be trained as per usual.

There are dual perspectives of GLIDER: as a method for model distillation or model enhancement.

If a strong source model is used to detect global interactions which are then encoded in more resourceconstrained target models, then GLIDER adopts a teacher-student type distillation process.

If interaction encoding augments the same model where the interactions were detected from, then GLIDER tries to enhance the model's ability to represent the interactions.

In our experiments, we study the effectiveness of GLIDER on real-world data.

The hyperparameters for local interaction interpretation in our experiments are as follows.

For all experiments, we use 5000 perturbation samples to train the models used for interaction detection.

We use NID as the interaction detector, which requires training an MLP to detect each set of interactions.

The MLPs for §5.3 have architectures of 50-30-10 first-to-last hidden layer sizes, and in §5.2, architectures of 256-128-64.

We apply an 1 regularization of λ 1 = 5e−5, and the learning rate is 5e−3.

In general, models are trained with early stopping on the validation set.

For LIME perturbations, we need to establish what a binary 0 maps to via ξ in the raw data instance ( §4.1).

In domains involving embeddings, i.e., sparse features and word embeddings, the 0 ("off") state is the zeroed embedding vector.

For dense features, it is the mean feature value over a batch; for images, the mean of each RGB of the image.

For our DNA experiment, we use a random nucleotide other than the original one.

These settings correspond to what is used in literature (Ribeiro et al., 2016; .

In our graph experiment, the nodes within the neighborhood of a test node are perturbed, where each node is zeroed during perturbation.

In this section, we provide experiments with GLIDER on models trained for clickthrough-rate (CTR) prediction.

The recommender models we study include commonly reported baselines, which all use neural networks: Wide&Deep (Cheng et al., 2016) , DeepFM (Guo et al., 2017) , Deep&Cross (Wang et al., 2017) , xDeepFM (Lian et al., 2018) , and AutoInt (Song et al., 2018) .

AutoInt is the reported state-of-the-art in academic literature, so we use the model settings and data splits provided by AutoInt's official public repository 1 .

For all other recommender models, we use public implementations 2 with the same original architectures reported in literature, set all embedding sizes to 16, and tune the learning rate and optimizer to try to reach or surpass the test logloss reported by the AutoInt paper (on AutoInt's data splits).

From tuning, we use the Adagrad optimizer (Duchi et al., 2011) with learning rates in {0.01, 0.001}.

The datasets we use are benchmark CTR datasets with the largest number of features: Criteo 3 and Avazu 4 , whose data statistics are shown in Table 1 .

Criteo and Avazu both contain 40+ millions of user records on clicking ads, with Criteo being the primary benchmark in CTR research (Cheng Table 3 : Test prediction performance by encoding top-K global interactions in baseline recommender systems on the Criteo and Avazu datasets (5 trials).

K are 40 and 20 for Criteo and Avazu respectively.

"+ GLIDER" means the inclusion of detected global interactions to corresponding baselines.

The "Setting" column is labeled relative to the source of detected interactions: AutoInt.

Guo et al., 2017; Wang et al., 2017; Lian et al., 2018; Song et al., 2018; Luo et al., 2019) .

For each dataset, we train a source AutoInt model, f rec , then run global interaction detection via Algorithm 1 on a batch of 1000 samples from the validation set.

A full global detection experiment finishes between 2-3 hours when run in serial on either Criteo or Avazu datasets in a 32-CPU Intel Xeon E5-2640 v2 @ 2.00GHz server, and significant speed-ups can be achieved by fully parallelizing Algorithm 1.

The detection results across datasets are shown in Figure 2 as plots of detection counts versus rank.

Because the Avazu dataset contains non-anonymized features, we directly show its top-10 detected global interactions in Table 2 .

From Figures 2, we see that the top interactions are detected very frequently across data instances, once appearing across more than half of the batch.

In Table 2 , the top-interactions can be explained.

For example, the interaction between "hour" (in UTC time) and "device ip" makes sense because users -here identified by an IP address -have ad-click behaviors dependent on their time zones.

We hypothesize that the global interaction detections are also informative for modeling purposes.

Based on our results from the previous section ( §5.2.1), we turn our attention to explicitly encoding the detected global interactions in target baseline models via truncated feature crosses (detailed in §4.3).

In order to generate valid cross feature IDs, we bucketize dense features into a maximum of 100 bins before crossing them and require that final cross feature IDs occur more than T = 100 times over a training batch of one million samples.

We take AutoInt's top-K global interactions on each dataset from §5.2.1 with subset interactions excluded (Algorithm 1, line 6) and encode the interactions in each baseline model including AutoInt Table 5 : Prediction performance (mean-squared error; lower is better) with (k > 0) and without (k = 0) interactions for random data instances in the test sets of respective black-box models.

k = L corresponds to the interaction at a rank threshold.

2 ≤ k < L are excluded because not all instances have 2 or more interactions.

Only results with detected interactions are shown.

At least 80% (≥ 320) of the data instances possessed interactions over 10 trials for each model/performance statistic.

linear LIME 0 9.8e−3 ± 9e−4 0.101 ± 7e−3 0.25 ± 0.07 0.080 ± 3.0e−4 MADEX 1 8e−3 ± 1e−3 0.056 ± 9e−3 0.22 ± 0.06 0.062 ± 8.1e−3 MADEX L 6e−3 ± 1e−3 0.024 ± 7e−3 0.16 ± 0.05 0.038 ± 9.6e−3

itself.

There is consensus that 0.001 logloss or AUC improvements are significant in CTR prediction tasks (Cheng et al., 2016; Guo et al., 2017; Wang et al., 2017; Song et al., 2018) .

K is tuned on valiation sets, and model hyperparameters are the same between a baseline and one with encoded interactions.

We set K = 40 for Criteo and K = 20 for Avazu.

We found that using GLIDER can often reach or exceed the 0.001 significance level, especially for the main Criteo benchmark dataset, as shown in Table 3 .

These performance gains are obtained at limited cost of extra model parameters (Table 4) thanks to the truncations applied to our feature crosses.

In Figure 3 , we also show how the test performance of AutoInt varies with different K on the Criteo dataset.

One one hand, the evidence that AutoInt's detected interactions can improve other baselines' performance suggests the viability of interaction distillation.

On the other hand, evidence that AutoInt's performance on Criteo can improve using its own detected interactions suggests that AutoInt may benefit from learning interactions more explicitly.

In either model distillation or enhancement settings, we found that GLIDER performs especially well on industry production models trained on large private datasets with thousands of features.

Since the proposed interaction interpretations by GLIDER are not entirely limited to recommender systems, we demonstrate interpretations on more general black-box models.

Specifically, we experiment with the function MADEX(·) defined in §4.1, which inputs a black-box f , data-instance x, and outputs a set of interactions.

The models we use are trained on very different tasks: ResNet152-an image classifier pretrained on ImageNet '14 (Russakovsky et al., 2015; He et al., 2016) , Sentiment-LSTM-a 2-layer bi-directional long short-term memory network (LSTM) trained on the Stanford Sentiment Treebank (SST) (Socher et al., 2013; Tai et al., 2015) , DNA-CNN-a 2-layer 1D convolutional neural network (CNN) trained on MYC-DNA binding data (Mordelet et al., 2013; Yang et al., 2013; Alipanahi et al., 2015; Zeng et al., 2016; Wang et al., 2018) , and GCN-a 3-layer Graph Convolutional Network trained on the Cora dataset (Kipf & Welling, 2016; Sen et al., 2008) .

We first provide quantitative validation for the detected interactions of all four models in §5.3.1, followed by qualitative results for ResNet152, Sentiment-LSTM, and DNA-CNN in §5.3.2.

To provide quantitative validation of interaction interpretations of black-box models, we evaluate the predictive power of the interactions at the data instance level.

As suggested in §4.1 and §4.3, encoding feature interactions is a way to increase a model's function representation, but this also means that prediction performance gains over simpler first-order models (e.g., linear regression) is a way to test the significance of the detected interactions.

In this section, we use neural network function approximators for each top-interaction from the ranking {

I i } given by MADEX's interaction detector (in this case NID).

Similar to the k-thresholding description in §4.1, we start at k = 0, which is a linear regression, then increment k with added MLPs for each

until validation performance stops improving, denoted at k = L. The MLPs all have architectures of 30-10 first-to-last hidden layer sizes and use the binary perturbation dataset D (introduced in §4.1).

Test prediction performances are shown in Table 5 for k ∈ {0, 1, L}. The average number of features of D among the black-box models ranges from 16 to 189.

Our quantitative validation shows that adding feature interactions for DNA-CNN, Sentiment-LSTM, and ResNet152, and adding node in-teractions for GCN result in significant performance gains when averaged over 40 randomly selected data instances in the test set.

Figure 4 : Qualitative results of the detected interactions by MADEX and the selected features by LIME's original linear regression ("LIME selection") on (a) ResNet152 and (b) Sentiment-LSTM.

The interpretations between MADEX and LIME selection are complementary.

For our qualitative analysis, we provide interaction interpretations via MADEX(·) of ResNet152, Sentiment-LSTM, and DNA-CNN on test examples.

The interpreations are given by

, a set of k detected interactions, which are shown in Figures 4a and 4b for ResNet152 and Sentiment-LSTM respectively.

Interactions that have majority overlap among S are merged, i.e., overlap coefficient > 0.5 (Vijaymeena & Kavitha, 2016) .

For reference, we also show the selected features by LIME's original linear regression, which takes the top-5 features that attribute towards the predicted class 5 .

In Figure 4a , the MADEX columns show selected features from the detected interactions between Quickshift superpixels (Vedaldi & Soatto, 2008; Ribeiro et al., 2016) .

We see that the interactions can form a single region or multiple regions of the image, and they are complementary to LIME's feature selection.

For example, the interpretations of the "deskop computer" classification show that interaction detection finds one of the computers and feature selection finds the other.

For Sentiment-LSTM interpretations in Figure 4b , we also see that MADEX's interactions can complement LIME's selected features.

Here, the interactions show salient combinations of words, such as "science fiction" and "I like pug".

In our experiments on DNA-CNN, we consistently detected the interaction between "CACGTG" nucleotides, which form a canonical DNA sequence (Staiger et al., 1989) .

The interaction was detected 76.5% out of 187 CACGTG appearances in the test set.

We proposed GLIDER that detects and explicitly encodes global feature interactions in black-box recommender systems.

In our experiments, we found that the detected global interactions are informative and that explicitly encoding interactions can improve the accuracy of CTR predictions.

We further validated interaction interpretations on image, text, and graph classifiers.

We hope GLIDER encourages investigation into the complex interaction behaviors of recommender models to understand why certain feature interactions are very predictive.

For future research, we wish to understand how feature interactions play a role in the integrity of automatic recommendations.

In this section, we study whether increasing embedding size can obtain similar prediction performance gains as explicitly encoding interactions via GLIDER.

We increase the embedding dimension sizes of every sparse feature in baseline recommender models to match the total number of model parameters of baseline + GLIDER as close as possible.

The embedding sizes we used to obtain similar parameter counts are shown in Table 6 .

For the Avazu dataset, most of the embedding sizes remain unchanged because they were already the target size.

The corresponding prediction performances of all models are shown in Table 7 .

We observed that directly increasing embedding size / parameter counts generally did not give the same level of performance gains that GLIDER provided.

Table 6 : Comparison of # model parameters between baseline models with enlarged embeddings and original baselines + GLIDER (from Tables 3 and 4 ).

The models with enlarged embeddings are denoted by the asterick (*).

The embedding dimension of sparse features is denoted by "emb.

size".

Percent differences are relative to baseline* models.

M denotes million, and the ditto mark (") means no change in the above line.

Table 7 : Test prediction performance coresponding to the models shown in Table 6 Model Criteo Avazu

We examine the effect of dense feature bucketization on the parameter efficiency and prediction performance of AutoInt.

Results are provided for the Criteo dataset, which contains 13 dense features.

Figure 5 shows the effects of varying the number of dense feature buckets on the total number of parameters and the test logloss of AutoInt.

Figure 6 shows the effects of varying the number of dense buckets on the embedding sizes of the cross features involving dense features.

Both the effects on the average and individual embedding size are shown.

20 of the cross features involved a dense feature.

Patterns to note include the largely asymptotic behavior of the parameter plots as the number of buckets increases (Figures 5a and 6 ).

Our requirement that a valid cross feature ID occurs more than T times ( §4.3) restricts the growth in parameters.

In some cases, the number of cross feature IDs (embedding size) decreases (Figure 6b ), which happens when the dense bucket size becomes too small to satisfy the T occurrence restriction.

In Figure 5b , prediction performance degrades beyond 100 buckets, yet it is still an improvement over the baseline without cross features (0.4434 in Table 3 ).

The degradation may be caused by overfitting.

In this section, we show ranked results for interactions discovered by MADEX in Sentiment-LSTM.

The top-1 interactions are provided on random sentences from the SST test set.

For every sentence, we preprocess it to remove stop words.

If interactions are not detected in a sentence, that sentence is excluded.

Results are shown in Table 8 .

We use the same stop words suggested by (Manning et al., 2008), i.e., {a, an, and, are, as, at, be, by, for, from, has, he, in, is, it, its, of, on, that , the, to, was, were, will, with}. In order to obtain occurrence counts of word interactions, we need to detect the same word interactions across different sentences.

Naturally, different sentences will often have different words, so it is nontrivial to identify consistent word interactions.

We start by collecting interaction candidates by running MADEX over all sentences in the SST test set, then identifying the word interactions that appear multiple times.

Here, we make the assumption that word interactions are ordered but not necessarily adjacent or positionally bound.

Therefore, a hypothetical interaction, (not, good), could be found in either sentence: "This is not good", or "This movie is not so good".

The order of the words matter, so (not, good) = (good, not).

After collecting interaction candidates, we then look for sentences in the larger IMDB dataset (Maas et al., 2011) that contain the same ordered words of an interaction candidate.

Therefore, each interaction candidate I i will have its own set of sentences W i that could potentially yield that interaction.

Let W i contain a random selection of 40 viable sentences for I i .

The interactions {I i } with the highest detection counts across corresponding W i are shown in

We compare the detection performances between MADEX and baselines on identifying feature interactions learned by complex models, i.e., XGBoost (Chen & Guestrin, 2016) , Multilayer Perceptron (MLP), and Long Short-Term Memory Network (LSTM) (Hochreiter & Schmidhuber, 1997) .

The baselines are Tree-Shap: a method to identify interactions in tree-based models like XGBoost (Lundberg et al., 2018) , MLP-ACD+: a modified version of ACD (Singh et al., 2019; Murdoch et al., 2018) to search all pairs of features in MLP to find the best interaction candidate, and LSTM-ACD+: the same as MLP-ACD+ but for LSTMs.

All baselines are local interpretation methods.

For MADEX, we sample continuous features from a truncated normal distribution N (x, σ 2 I) centered at a specified data instance x and truncated at σ.

We evaluate interaction detection performance by using synthetic data where ground truth interactions are known (Hooker, 2004; Sorokina et al., 2008) .

We generate 10e3 samples of synthetic data using functions F 1 − F 4 (Table 10 ) with continuous features uniformly distributed between −1 to 1.

Next, we train complex models (XGBoost, MLP, and LSTM) on this data.

Lastly, we run MADEX and the baselines on 10 trials of 20 data instances at randomly sampled locations on the synthetic function domain.

Between trials, the complex models are trained with different random initialization to test the stability of each interpretation method.

Interaction detection performance is computed by the average R-precision (Manning et al., 2008) 6 of interaction rankings across the sampled data instances.

Results are shown in Table 11 .

On the tree-based model, MADEX can compete with the tree-specific baseline Tree-Shap, which only detects pairwise interactions.

On MLP and LSTM, MADEX performs significantly better than ACD+.

The performance gain is especially large in the LSTM setting.

Table 11 : Detection Performance in R-Precision (higher the better).

σ = 0.6 (max: 3.2).

"Tree" is XGBoost.

*Does not detect higher-order interactions.

†Requires an exhaustive search of all feature combinations.

Tree-Shap MADEX MLP-ACD+ MADEX LSTM-ACD+ MADEX F 1 (x) 1 ± 0 1 ± 0 0.63 ± 0.08 1 ± 0 0.3 ± 0.2 1 ± 0 F 2 (x) 1 ± 0 0.14 ± 0.09 0.41 ± 0.06 0.97 ± 0.03 0.01 ± 0.02 0.96 ± 0.03 F 3 (x) 1 ± 0 1 ± 0 0.3 ± 0.2 1 ± 0 0.05 ± 0.08 1 ± 0 F 4 (x) * 0.2 ± 0.1 † 0.61 ± 0.07 † 0.54 ± 0.09

This section shows how often higher-order interactions are identified by GLIDER / MADEX.

Figure 8 plots the occurrence counts of global interactions detected in AutoInt for the Criteo and Avazu dataset, which correspond to the results in Figure 2 .

Here we only show the occurrence counts of higher-order interactions, where the exact interaction order is annotated besides each data point.

3rd-order interactions are the most common type, and interestingly, an 8th-order interaction appears 13 times in the Avazu dataset.

Figure 9 plots histograms of interaction orders for all interactions detected from ResNet152 and Sentiment-LSTM across 1000 random samples in their test sets.

The average number of features are 67 and 19 for ResNet152 and Sentiment-LSTM respectively.

Higherorder interactions are especially common in ResNet152.

@highlight

Proposed a method to extract and leverage interpretations of feature interactions