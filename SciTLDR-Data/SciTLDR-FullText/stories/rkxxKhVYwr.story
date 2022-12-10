In this paper, we diagnose deep neural networks for 3D point cloud processing to explore the utility of different network architectures.

We propose a number of hypotheses on the effects of specific network architectures on the representation capacity of DNNs.

In order to prove the hypotheses, we design five metrics to diagnose various types of DNNs from the following perspectives, information discarding, information concentration, rotation robustness, adversarial robustness, and neighborhood inconsistency.

We conduct comparative studies based on such metrics to verify the hypotheses, which may shed new lights on the architectural design of neural networks.

Experiments demonstrated the effectiveness of our method.

The code will be released when this paper is accepted.

Recently, a series of works use the deep neural network (DNN) for 3D point cloud processing and have achieved superior performance in various 3D tasks.

However, traditional studies usually design network architectures based on empiricism.

There does not exist a rigorous and quantitative analysis about the utility of specific network architectures for 3D point cloud processing.

Exploring and verifying the utility of each specific intermediate-layer architecture from the perspective of a DNN's representation capacity still present significant challenges for state-of-the-art algorithms.

In this study, we aim to bridge the gap between the intermediate-layer network architecture and its utility.

Therefore, we propose a few hypotheses of the utility of specific network architectures.

Table 1 lists the hypotheses to be verified in this study, towards three kinds of utilities, i.e. rotation robustness, adversarial robustness, and neighborhood inconsistency.

We design and conduct comparative studies to verify the hypotheses.

Finally, we obtain some new insights into the utility of specific network architectures as follows.

• The specific architecture in , which uses the local density information to reweight features (Figure 1 (a) ), improves adversarial robustness (Table 1 (a)).

• Another specific architecture in , which uses local 3D coordinates' information to reweight features (Figure 1 (b) ), improves rotation robustness (Table 1 (b)).

• The specific architecture in (Qi et al., 2017b; Liu et al., 2018) , which extracts multi-scale features (Figure 1 (c) ), improves adversarial robustness and neighborhood consistency (Table 1 (c)).

Neighborhood consistency measures whether a DNN assigns similar attention to neighboring points.

• The specific architecture in (Jiang et al., 2018) , which encodes the information of different orientations (Figure 1 (d) ), improves rotation robustness (Table 1 (d)) .

More specifically, in order to verify the above hypotheses, we design the following five evaluation metrics and conduct a number of comparative experiments to quantify the utility of different network architectures.

1.

Information discarding and 2.

information concentration: Information discarding measures how much information of an input point cloud is forgotten during the computation of a specific intermediate-layer feature.

From the perspective of information propagation, the forward propagation through layers can be regarded as a hierarchical process of discarding input information (Shwartz-Ziv & Tishby, 2017) .

Ideally, the DNN is supposed to discard information that is not related to the task.

Let us take the task of object classification for example.

The information of

Rotation robustness

Neighborhood inconsistency (a) Modules of using information of local density to reweight features.

--(b) Modules of using information of local coordinates to reweight features.

--(c) Modules of concatenating multi-scale features.

-(d) Modules of computing orientation-aware features.

-- Table 1 : Illustration of the verified utilities of specific architectures.

Blank regions correspond to utilities that have not been examined, instead of indicating non-existence of the utilities.

Please see Figure 1 for architectural details.

foreground points is usually supposed to be related to the task, while that of background points is not related to the task and is discarded.

To this end, we further propose information concentration to measure the gap between the information related to the task and the information not related to the task.

Information concentration can be used to evaluate a DNN's ability to focus on points related to the task.

3.

Rotation robustness: Rotation robustness measures whether a DNN will use the same logic to recognize the same object when a point cloud has been rotated by a random angle.

In other words, if two point clouds have the same global shape but different orientations, the DNN is supposed to select the same regions/points to compute the intermediate-layer feature.

Unlike images with rich color information, point clouds usually only use spatial contexts for classification.

Therefore, a well-trained DNN is supposed to have rotation robustness.

A reliable DNN should be robust to adversarial attacks.

Neighborhood inconsistency measures whether adjacent points have similar importance in the computation of an intermediate-layer feature.

Adjacent points usually have similar shape contexts, so they are supposed to have similar importance.

Therefore, ideally, a well-trained DNN should have a low value of neighborhood inconsistency.

Contributions of our study are summarized as follows.

We propose a few hypotheses on the utility of specific network architectures.

Then, we design five metrics to conduct comparative studies for verifying these hypotheses, which provide a new insightful understanding of architectural utility.

3D Point Cloud Processing: Recently, a number of approaches use DNNs for 3D point cloud processing and have exhibited superior performance in various 3D tasks (Qi et al., 2017a; Su et al., 2018; Valsesia et al., 2018; Yu et al., 2018; Yang et al., 2018; Gadelha et al., 2018; Wang et al., 2018a; Komarichev et al., 2019; Shi et al., 2019) .

PointNet (Qi et al., 2017a ) was a pioneer in this direction, which used a max pooling layer to aggregate all individual point features into a global feature.

However, such architecture fell short of capturing local features.

PointNet++ (Qi et al., 2017b) hierarchically used PointNet as a local descriptor to extract contextual information.

Some studies Jiang et al., 2018; Wang et al., 2018b; Komarichev et al., 2019) further improved the networks' ability to capture local geometric features.

Other researches focused on the correlations between different regions of the 3D point cloud (Liu et al., 2018) or interaction between points (Zhao et al., 2019a) .

In comparison, our study focuses on the utility analysis of intermediate-layer network architectures for point cloud processing.

Visualization or diagnosis of representations: The visualization of visual patterns corresponding to a feature map or the network output is the most intuitive way of interpreting DNNs (Zeiler & Fergus, 2014; Mahendran & Vedaldi, 2015; Dosovitskiy & Brox, 2016; Zhou et al., 2014) , such as gradient-based methods (Fong & Vedaldi, 2018; Selvaraju et al., 2017) , and the estimation of the saliency map (Ribeiro et al., 2016; Lundberg & Lee, 2017; Kindermans et al., 2017; Qi et al., 2017a; Zheng et al., 2019) .

In comparison, our study aims to explore the utility of intermediate-layer network architectures by diagnosing the information-processing logic of DNNs.

Quantitative evaluation of representations: Recently, some studies quantified the representation similarity to help understand the neural networks (Gotmare et al., 2018; Kornblith et al., 2019; Morcos et al., 2018; Raghu et al., 2017) .

The method quantitated the importance of different feature dimensions to guide model compression.

Other studies evaluated the representations via quantifying the information they contain.

The information-bottleneck theory (Tishby et al., 2000; Shwartz-Ziv & Tishby, 2017; Cheng et al., 2018) explained the trade-off between the information compression and the discrimination power of features in a neural network.

Achille & Soatto (2018) We extend the method of (Ma et al., 2019) as the technical foundation.

Based on which, a number of metrics are designed to diagnose the DNN.

The method quantifies the discarding of the input information during the layerwise forward propagation by computing the entropy of the input information given the specific feature of an intermediate layer.

Given a point cloud X, let f = h(X) denote the feature of a specific intermediate layer.

It is assumed that f and f represent the same object concept 1 when f satisfies f − f 2 < , where feature f = h(X ), X = X + δ.

δ denotes a random noise.

The conditional entropy of the input information given a specific feature, which represents a specific object concept, is computed, i.e. calculating entropy

, where Σ can be regarded as the maximum perturbation added to X following the maximum-entropy principle, which subjects to a specific concept.

Considering the assumption of independent and identically distributed variables of each dimension of X , the overall entropy H(X ) can be decomposed into point-wise entropies.

where Hi = log σi + 1 2 log(2πe) denotes the entropy of the i-th point, which quantifies how much information of the i-th point can be discarded, when the feature h(X ) is required to represent the concept of the target object.

Metric 1, information discarding: The information discarding is defined as H(X ) in Equation (1).

The information discarding is measured at the point level, i.e. Hi, which quantifies how much information of the i-th point is discarded during the computation of an intermediate-layer feature.

The point with a lower value of Hi is regarded more important in the computation of the feature.

Metric 2, information concentration: The information concentration is based on the metric of information discarding.

The information concentration is used to analyze a DNN's ability to maintain the input information related to the task, and discard redundant information unrelated to the task.

For example, in the task of object classification, the background points are usually supposed not to be related to the task and are therefore more likely to be discarded by the DNN.

Let Λ

where C(·) is the predicted label; l * is the correct label of X;l is a target incorrect label.

In this study, we perform adversarial attacks against all incorrect classes.

We use the average of 2 over all incorrect classes to measure the adversarial robustness.

Metric 5, neighborhood inconsistency: The neighborhood inconsistency is proposed to evaluate a DNN's ability to assign similar attention to neighboring points during the computation of an intermediate-layer feature, i.e. a well-trained DNN should have a low value of neighborhood inconsistency.

Ideally, for a DNN, except for special points (e.g. those on the edge), most neighboring points in a small region of a point cloud usually have similar shape contexts, so they are supposed to make similar contributions to the classification and receive similar attention.

Let N(i) denote a set of K nearest neighboring points of the i-th point.

We define the neighborhood inconsistency as the difference between the maximum and minimum point-wise information discarding within N(i).

4 HYPOTHESES AND COMPARATIVE STUDY

• Notation: Let xi ∈ R 3 denote the i-th point, i = 1, 2, . . .

, n; let N(i) denote a set of K nearest points of xi; let Fi ∈ R d×K denote intermediate-layer features that correspond to neighboring points in N(i), where each column of Fi represents the feature of a specific point in N(i).

• Architecture 1, features reweighted by the information of the local density: Architecture 1 focuses on the use of the local density information to reweight features .

As shown in Figure 1 (a), for each point xi, Architecture 1 uses the local density w.r.t.

neighboring points of xi to compute W H1 ∈ R K , which reweights intermediate-layer features Fi.

where diag[W H1 ] transforms the vector W H1 into a diagonal matrix; density(N(i)) is a density vector w.r.t.

points in N(i); the M LP is a two-layer perceptron network.

• Architecture 2, features reweighted by the information of local coordinates: As shown in Figure 1 (

where the M LP is a single-layer perceptron network.

• Architecture 3, multi-scale features: Architecture 3 focuses on the use of multi-scale contextual information (Qi et al., 2017b; Liu et al., 2018) .

As illustrated in Figure 1 (c),

} denote features that are extracted using contexts of xi at different scales, .

Architecture 3 concatenates these multi-scale features to obtain f

where

where concat indicates the concatenation operator; g(·) is a function for feature extraction (Qi et al., 2017a) .

Please see Appendix B for details about this function.

• Architecture 4, orientation-aware features: Architecture 4 focuses on the use of orientation information (Jiang et al., 2018) .

As illustrated in Figure

where Conv oe is a special convolution operator.

Please see (Jiang et al., 2018) or Appendix C.4 for details about this operator and the computation of f oe i .

Hypothesis 1: Architecture 1 designed by , as shown in Figure 1 (a), increases the adversarial robustness.

This hypothesis is proposed based on the observation that PointConv has good performance in adversarial robustness, which may stem from Architecture 1.

To verify this hypothesis, we design comparative studies on PointConv, PointNet++ (Qi et al., 2017b) , and Point2Sequence (Liu et al., 2018) .

For each network, we construct two versions for comparison, i.e. one with Architecture 1 and the other without Architecture 1.

PointConv w/ and w/o Architecture 1: To obtain the PointConv without Architecture 1, we remove all the modules of Architecture 1 from the original network (see the footnote 2 ), which are located behind the 2-nd, 5-th, 8-th, 11-th, and 14-th nonlinear transformation layers.

Please see Appendix D.2 for the global architectures of different versions of PointConv.

PointNet++ w/ and w/o Architecture 1: To obtain the PointNet++ with Architecture 1, we add three modules of Architecture 1, which are located behind the 3-rd, 6-th, and 9-th nonlinear transformation layers.

Please see Appendix D.1 for the global architectures of different versions of PointNet++.

Point2Sequence w/ and w/o Architecture 1: To obtain the Point2Sequence with Architecture 1, we add the module of Architecture 1 behind the last nonlinear transformation layer, as shown in Equation (11).

Please see Appendix D.3 for the global architectures of different versions of Point2Sequence.

Hypothesis 2: Architecture 2 designed by , as shown in Figure 1 (b), increases the rotation robustness.

This hypothesis is proposed based on the observation that PointConv has good performance in rotation robustness, which may stem from Architecture 2.

To verify this hypothesis, we design comparative studies on PointConv, PointNet++, and Point2Sequence.

PointConv w/ and w/o Architecture 2: To obtain the PointConv without Architecture 2, we remove all the modules of Architecture 2, which are located before the 3-rd, 6-th, 9-th, 12-th, and 15-th nonlinear transformation layers.

Please see Appendix D.2 for the global architectures of different versions of PointConv.

PointNet++ w/ and w/o Architecture 2: Just like Hypothesis 1, to obtain the PointNet++ with Architecture 2, we add three modules of Architecture 2, which are located behind the 3-rd, 6-th, and 9-th nonlinear transformation layers.

Please see Appendix D.1 for the global architectures of different versions of PointNet++.

Point2Sequence w/ and w/o Architecture 2: Just like Hypothesis 1, to obtain the Point2Sequence with Architecture 2, we add the module of Architecture 2 behind the last nonlinear transformation layer, as shown in Equation (13).

Please see Appendix D.3 for the global architectures of different versions of Point2Sequence.

Hypothesis 3: Architecture 3 designed by (Qi et al., 2017b) , as shown in Figure 1 (c), increases the adversarial robustness and the neighborhood consistency.

This hypothesis is proposed inspired by (Qi et al., 2017b; Liu et al., 2018) , which encode multi-scale contextual information.

To verify this hypothesis, we design the following comparative studies on Point2Sequence and PointNet++.

PointNet++ w/ and w/o Architecture 3: To obtain the PointNet++ with Architecture 3, we use the multi-scale version of PointNet++ (Qi et al., 2017b) , which extracts features at three scales.

Point2Sequence w/ and w/o Architecture 3: To obtain different versions of Point2Sequence, we use three networks as follows.

The baseline network of Point2Sequence concatenates features of 4 different scales to compute the feature in the upper layer,

In this study, we set K1 = 128, K2 = 64, K3 = 32, and K4 = 16.

The first network for comparison extracts three different scale features, {f This hypothesis is proposed based on the observation that PointSIFT (Jiang et al., 2018) performs well in rotation robustness, which may stem from Architecture 4.

Because Architecture 4 ensures that features contain information from various orientations.

To verify this hypothesis, we design comparative studies on PointSIFT, PointNet++, and Point2Sequence as follows.

Figure 2 : Comparisons of layerwise information discarding and layerwise information concentration between DNNs.

Table 3 : Comparisons of the rotation non-robustness between DNNs trained using the ModelNet40, ShapeNet, and 3D MNIST datasets.

The column of ∆ denotes the increase of the rotation robustness of the network with the specific architecture w.r.t.

the network without the specific architecture.

∆ > 0 indicates that the corresponding hypothesis has been verified.

Experimental results show that Architecture 2 and Architecture 4 improved the rotation robustness, i.e. reporting relatively lower values of rotation non-robustness.

Please see Appendix F (Table 14) for the classification accuracy of these DNNs.

To demonstrate the broad applicability of our method, we applied our method to diagnose six widely used DNNs, including PointNet, PointNet++, PointConv, DGCNN, PointSIFT, and Point2Sequence.

These DNNs were trained using three benchmark datasets, including the ModelNet40 dataset (Wu et al., 2015) , the ShapeNet 4 dataset (Chang et al., 2015) , the 3D MNIST 5 dataset.

In sum, we conducted four comparative experiments.

The first experiment was to analyze DNNs using the proposed five metrics.

The remaining three experiments were conducted to verify the effects of different network architectures on DNNs' rotation robustness, adversarial robustness, and neighborhood inconsistency, respectively.

Implementation details: To analyze the information concentration of DNNs, we generated a new dataset that contained both the foreground objects and the background, since most widely used benchmark datasets for point cloud classification only contain foreground objects.

Specifically, for each sample (i.e. the foreground object) in the ModelNet40, we used the following three steps to generate the background.

First, we randomly sampled a set of 500 points from point clouds, which had different labels from the foreground object.

Second, we resized this set of points to the density of the foreground object.

Finally, we randomly located it around the foreground object.

The dataset will be released when this paper is accepted.

The entropy-based method (Ma et al., 2019) quantified the layerwise information discarding.

This method assumed the feature space of the concept of a specific object satisfied f − f 2 < , where f = h(X), f = h(X ), X = X + δ.

δ denotes a random noise.

For point cloud processing, each dimension of the intermediate-layer feature is computed using the context of a specific point xi.

However, adding noise to a point cloud will change the context of each point.

In order to extend the entropy-based method to point cloud processing, we selected the same set of points as the contexts w.r.t.

xi and x i , so as to generate a convincing evaluation.

Please see Appendix E for details.

Experiment 1, quantifying the representation capacity of DNNs: As shown in Table 2 , we measured information discarding, information concentration, rotation robustness, and neighborhood inconsistency of the representation of the fully connected layer close to the network output, which had 512 hidden units.

We measured adversarial robustness by performing adversarial attacks over all Figure 2 compares layerwise information discarding and layerwise information concentration of different layers in different DNNs.

We found that PointNet and Point2Sequence had relatively higher values of information discarding.

PointConv and PointSIFT discarded more information of points in the background.

PointConv, DGCNN, and PointSIFT performed well in rotation robustness.

PointConv, PointSIFT, and Point2Sequence exhibited higher adversarial robustness.

DGCNN and PointSIFT exhibited lower neighborhood inconsistency.

Experiment 2, verifying the effects on rotation robustness: For the computation of rotation robustness, during the training and testing phases, each point cloud was rotated by random angles.

Table 3 shows that Architecture 2 and Architecture 4 improved the rotation robustness.

Experiment 3, verifying the effects on adversarial robustness: Table 4 shows that both Architecture 1 and Architecture 3 improved the adversarial robustness.

For Architecture 3, we found that the adversarial robustness increased with the scale number of features.

Experiment 4, verifying the effects on neighborhood inconsistency:

For the computation of neighborhood inconsistency, we used k-NN search to select 16 neighbors for each point.

Table 5 shows that networks with Architecture 3 usually had lower neighborhood inconsistency than those without Architecture 3.

Besides, DNNs, which extracted features from contexts of more scales, usually exhibited lower neighborhood inconsistency.

In this paper, we have verified a few hypotheses of the utility of four specific network architectures for 3D point cloud processing.

Comparative studies are conducted to prove the utility of the specific architectures, including rotation robustness, adversarial robustness, and neighborhood inconsistency.

In preliminary experiments, we have verified that Architecture 2 and Architecture 4 mainly improve the rotation robustness; Architecture 1 and Architecture 3 have positive effects on adversarial robustness; Architecture 3 usually alleviates the neighborhood inconsistency.

This Appendix provides more details about comparative studies in the main paper and includes more implementation details about experiments.

In Section B, we introduce a special element-wise max operator widely used in point cloud processing.

In Section C, we briefly introduce DNNs used in comparative studies.

In Section D, we show details about different versions of DNNs for comparison.

In Section E, we show implementation details about extending the entropy-based method (Ma et al., 2019) to point cloud processing.

In Section F, we compare the accuracy of different versions of DNNs.

In Section G, we supplement related work about learning interpretable representations.

In point cloud processing, a special element-wise max operator is widely used for aggregating a set of neighboring points' features into a local feature.

As shown in Figure 3 , given a set of K nearest neighboring points of xi, N(i), let Fi ∈ R d×K denote intermediate-layer features that correspond to the set of neighboring points in N(i) w.r.t.

the point xi.

Each specific column of Fi represents the feature of a specific point in N(i).

The process of extracting the feature in the upper layer, i.e. , can be formulated as follows, which is the local feature of N(i).

where M LP is an MLP network with a few layers; M LP (Fi) ∈ R D×K ; MAX is an element-wise max operator as follows.

Let

For a better understanding of different versions of DNNs in the next section, we briefly introduce DNNs used in comparative studies, including PointNet++, PointConv, Point2Sequence, and PointSIFT.

PointNet++ (Qi et al., 2017b ) is a hierarchical structure composed of a number of set abstraction modules (SA module).

For each SA module, a set of points is processed and abstracted to produce a new set with fewer elements.

An SA module includes four parts: the Sampling layer, the Grouping layer, the MLP, and the Maxpooling layer.

Given a set of N input points, the Sampling layer uses the farthest point sampling algorithm to select a subset of points from the input points, which defines the centroids of local regions, {xi}, i = 1, . . .

, N .

Then, for each selected point, the Grouping layer constructs a local region by using ball query search to find K neighboring points within a radius r. For each local region N(i) centered at xi, Fi ∈ R d×K denotes the intermediate-layer features that correspond to points in N(i).

The MLP transforms Fi into higher dimension features F i ∈ R D×K , where D > d. Finally, the Maxpooling layer encodes F i into a local feature f upper i , which will be fed to the upper SA module.

Please see Appendix B for details about the Maxpooling layer.

In this study, the baseline network of PointNet++ is composed of three SA modules and a few fully connected layers.

Please see Table 6 (left column) for details about the network architecture.

PointConv has a similar architecture with PointNet++, i.e. hierarchically using a few blocks to extract contextual information.

In this study, the baseline network of PointConv is composed of five blocks.

Each block is constructed as [Sample layer→Group layer→MLP→Architecture 1→Architecture 2→Conv layer].

The Sampling layer uses the farthest point sampling algorithm to select a subset of points from the input points, which defines the centroids of local regions.

Then, for each selected point, the Grouping layer constructs a local region by using k-NN search to find K neighboring points.

For each local region, the MLP transforms features of points in the local region into higher dimension features.

Different from PointNet++, PointConv uses the information of density (i.e. Architecture 1) and local 3D coordinates (i.e. Architecture 2) to reweight the features learned by the MLP.

Finally, a 1 × 1 convolution is used to compute the output feature of each local region.

Please see Table 8 (left column) for details about the network architecture.

Point2Sequence (Liu et al., 2018 ) is composed of five parts: (a) multi-scale area establishment, (b) area feature extraction, (c) encoder-decoder feature aggregation, (d) local region feature aggregation, and (e) shape classification, where parts (a) and (b) makes up Architecture 3 in our study.

Specifically, given a point cloud X = {xi}, i = 1, 2, ..., N , Point2Sequence first uses the farthest point sampling algorithm to select N points from the input point cloud, X = {x j }, j = 1, 2, ..., N , to define the centroids of local regions {N(j)}, j = 1, 2, ...N .

For each local region N(j), T different scale areas {A(j) ∈ R d for each scale area A(j) t by the MLP and the Maxpooling layer introduced in Appendix C.1.

Therefore, for each local region N(j), a feature sequence f

is aggregated into a d-dimensional feature rj by the encoder-decoder feature aggregation part.

The sequence encoder-decoder structure used here is an LSTM network, where an attention mechanism is proposed to highlight the importance of different area scales (please see (Liu et al., 2018) for details).

Then, a 1024-dimensional global feature is aggregated from the features rj of all local regions by the local region feature aggregation part.

Finally, the global feature is used for shape classification.

Please see Table 9 for details about the network architecture.

PointSIFT (Jiang et al., 2018) adopts the similar hierarchical structure as PointNet++, which is composed of a number of SA modules.

The difference is that PointSIFT uses a special orientation encoding unit, i.e., Architecture 4, to learn an orientation-aware feature for each point.

Architecture 4 is a point-wise local feature descriptor that encodes information of eight orientations.

Unlike the unordered operator, e.g. max pooling, which discards all inputs except for the maximum, Architecture 4 is an ordered operator, which could be more informative.

Architecture 4 first selects 8-nearest points of x i from eight octants partitioned by the ordering of three coordinates.

Since distant points provide little information for the description of local patterns, when no point exists within searching radius r in some octant, x i will be duplicated as the nearest neighbor of itself.

Then, Architecture 4 processes features of 8-nearest neighboring points, F oe i ∈ R d×2×2×2 , which reside in a 2 × 2 × 2 cube for local pattern description centering at x i (as shown in Figure 1 (d) ), the three dimensions 2 × 2 × 2 correspond to three axes.

An orientation- Table 6 : Different versions of PointNet++, including the original one, the one with Architecture 1, the one with Architecture 2, and the one with Architecture 4.

Sample (N ) indicates the Sample layer, which selects a subset of N points from the input point cloud.

Group (r, K) indicates the Group layer, which uses the ball query search to find K neighboring points around each sampled point within a radius r. Group (all) means constructing a region with all the input points.

MLP (512) Sample (512) Sample (512) Sample (512) encoding convolution, i.e. Conv oe , which is a three-stage operator, is used to convolve the 2 × 2 × 2 cube along x, y, and z axis.

The three-stage convolution Conv oe is formulated as:

where W x ∈ R d×1×1×2 , W y ∈ R d×1×2×1 , and W z ∈ R d×2×1×1 are weights of the convolution operator.

In this way, Architecture 4 learns the orientation-aware feature f oe i for each point x i .

Such orientation-aware features will be fed to SA modules introduced in Appendix C.1 to extract contextual information.

In this study, we reconstructed the PointNet++ (Qi et al., 2017b) using four specific modules.

Table 6 and Table 7 compare the different versions of PointNet++, including the original one, the one with Architecture 1 , the one with Architecture 2 , the one with Architecture 4 (Jiang et al., 2018) , and the one with Architecture 3 (Liu et al., 2018) .

To obtain the PointNet++ with Architecture 1 (as shown in Table 6 ), we added modules of Architecture 1 after all the MLPs in PointNet++, i.e. the output of the MLP was reweighted by the weights learned by Architecture 1.

Architecture 1 used in this study was an MLP with two layers, the first layer contained 16 hidden units, and the second layer contained 1 hidden unit.

This network was designed to verify the effect of Architecture 1 on the adversarial robustness.

To obtain the PointNet++ with Architecture 2 (as shown in Table 6 ), we added modules of Architecture 2 after all the MLPs in PointNet++, i.e. the output of the MLP was reweighted by the weights learned by Architecture 2.

Architecture 2 used in this study was an MLP with a single-layer, which contained 32 hidden units.

This network was designed to verify the effect of Architecture 2 on the rotation robustness.

To obtain the PointNet++ with Architecture 4 (as shown in Table 6 ), we added the module of Architecture 4 before the last Sample layer in PointNet++.

This network was designed to verify the effect of Architecture 4 on the rotation robustness. (1024) Sample (1024) Sample (1024) Group (32) Group (32) Group ( [512, 128, 40] Softmax

To obtain the PointNet++ with Architecture 3 (as shown in Table 7 ), we used the multi-scale version of PointNet++ designed in (Qi et al., 2017b) .

Compared with the single-scale version of PointNet++ (as shown in This network was used to verify the effect of Architecture 3 on the adversarial robustness and the neighborhood inconsistency.

D.2 POINTCONV: Table 8 compares different versions of PointConv , including the original one, the one without Architecture 1 , and the one without Architecture 2 .

To obtain the PointConv without Architecture 1 (as shown in Table 8 (middle column)), we removed all the five modules of Architecture 1 from the original PointConv architecture.

This network was designed to verify the effect of Architecture 1 on the adversarial robustness.

To obtain the PointConv without Architecture 2 (as shown in Table 8 (right column)), we removed all the five modules of Architecture 2 from the original PointConv architecture.

This network was designed to verify the effect of Architecture 2 on the rotation robustness.

The baseline network of Point2Sequence (as shown in To obtain the Point2Sequence with Architecture 1 (as shown in Table 10 ), we added the module of Architecture 1 after the last MLP, i.e. MLP [256, 512, 1024] , in Point2Sequence.

This network was designed to verify the effect of Architecture 1 on the adversarial robustness.

To obtain the Point2Sequence with Architecture 2 (as shown in Table 11 ), we added the module of Architecture 2 after the last MLP, i.e. MLP [256,512,1024] , in Point2Sequence.

This network was designed to verify the effect of Architecture 2 on the rotation robustness.

To obtain the Point2Sequence with Architecture 4 (as shown in Table 12 ), we added the module of Architecture 4 after the LSTM.

This network was designed to verify the effect of Architecture 4 on the rotation robustness.

To obtain the PointSIFT without Architecture 4 (as shown in Table 13 ), we removed all the four modules of Architecture 4 from the original PointSIFT.

This network was designed to verify whether Architecture 4 can improve the rotation robustness.

In this study, we used the entropy-based method (Ma et al., 2019) to quantify the layerwise information discarding of DNNs.

This method assumed the feature space of the concept of a specific object satisfied f − f 2 < , where f = h(X), f = h(X ), X = X + δ.

δ denotes a random noise.

For image processing, changing the pixel values will not change the receptive field of an interneuron, thereby features f and f are computed using the same set of pixels (as shown in Figure 4 (a) ).

However, for point cloud processing, changing the coordinates of points will change the "receptive field" of an interneuron, i.e. features f and f are computed using contexts of different set of points (as shown in Figure 4 (b)).

Compared to the visualization or diagnosis of representations, directly learning interpretable representations is more meaningful to improving the transparency of DNNs.

In the capsule nets (Sabour et al., 2017; Zhao et al., 2019b) , meaningful capsules, which were composed of a group of neurons, were learned to represent specific entities.

Vaughan et al. (2018) learned explainability features with additive nature.

The infoGAN (Chen et al., 2016) learned disentangled representations for generative models.

The β- VAE Higgins et al. (2017) further developed a measure to quantitatively compare the degree of disentanglement learnt by different models.

Zhang et al. (2018) proposed an interpretable CNN, where filters were mainly activated by a certain object part.

Fortuin et al. (2018) learned interpretable low-dimensional representations of time series and provided additional explanatory insights.

Mott et al. (2019) presented a soft attention mechanism for the reinforcement learning domain, the interpretable output of which can be used by the agent to decide its action.

@highlight

We diagnose deep neural networks for 3D point cloud processing to explore the utility of different network architectures. 

@highlight

The paper investigates different neural network architectures for 3D point cloud processing and proposes metrics for adversarial robustness, rotational robustness, and neighborhood consistency.