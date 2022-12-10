In weakly-supervised temporal action localization, previous works have failed to locate dense and integral regions for each entire action due to the overestimation of the most salient regions.

To alleviate this issue, we propose a marginalized average attentional network (MAAN) to suppress the dominant response of the most salient regions in a principled manner.

The MAAN employs a novel marginalized average aggregation (MAA) module and learns a set of latent discriminative probabilities in an end-to-end fashion.

MAA samples multiple subsets from the video snippet features according to a set of latent discriminative probabilities and takes the expectation over all the averaged subset features.

Theoretically, we prove that the MAA module with learned latent discriminative probabilities successfully reduces the difference in responses between the most salient regions and the others.

Therefore, MAAN is able to generate better class activation sequences and identify dense and integral action regions in the videos.

Moreover, we propose a fast algorithm to reduce the complexity of constructing MAA from $O(2^T)$ to $O(T^2)$. Extensive experiments on two large-scale video datasets show that our MAAN achieves a superior performance on weakly-supervised temporal action localization.

Weakly-supervised temporal action localization has been of interest to the community recently.

The setting is to train a model with solely video-level class labels, and to predict both the class and the temporal boundary of each action instance at the test time.

The major challenge in the weakly-supervised localization problem is to find the right way to express and infer the underlying location information with only the video-level class labels.

Traditionally, this is achieved by explicitly sampling several possible instances with different locations and durations BID2 BID11 .

The instance-level classifiers would then be trained through multiple instances learning BID4 BID40 or curriculum learning BID1 ).

However, the length of actions and videos varies too much such that the number of instance proposals for each video varies a lot and it can also be huge.

As a result, traditional methods based on instance proposals become infeasible in many cases.

Recent research, however, has pivoted to acquire the location information by generating the class activation sequence (CAS) directly BID17 , which produces the classification score sequence of being each action for each snippet over time.

The CAS along the 1D temporal dimension for a video is inspired by the class activation map (CAM) BID46 BID19 BID18 in weakly-supervised object detection.

The CAM-based models have shown that despite being trained on image-level labels, convolutional neural networks (CNNs) have the remarkable ability to localize objects.

Similar to object detection, the basic idea behind CAS-based methods for action localization in the training is to sample the non-overlapping snippets from a video, then to aggregate the snippet-level features into a video-level feature, and finally to yield a video-level class prediction.

During testing, the model generates a CAS for each class that identifies the discriminative action regions, and then applies a threshold on the CAS to localize each action instance in terms of the start time and the end time.

In CAS-based methods, the feature aggregator that aggregates multiple snippet-level features into a video-level feature is the critical building block of weakly-supervised neural networks.

A model's ability to capture the location information of an action is primarily determined by the design of the aggregators.

While using the global average pooling over a full image or across the video snippets has shown great promise in identifying the discriminative regions BID46 BID19 BID18 , treating each pixel or snippet equally loses the opportunity to benefit from several more essential parts.

Some recent works BID17 BID49 have tried to learn attentional weights for different snippets to compute a weighted sum as the aggregated feature.

However, they suffer from the weights being easily dominated by only a few most salient snippets.

In general, models trained with only video-level class labels tend to be easily responsive to small and sparse discriminative regions from the snippets of interest.

This deviates from the objective of the localization task that is to locate dense and integral regions for each entire action.

To mitigate this gap and reduce the effect of the domination by the most salient regions, several heuristic tricks have been proposed to apply to existing models.

For example, BID35 BID44 attempt to heuristically erase the most salient regions predicted by the model which are currently being mined, and force the network to attend other salient regions in the remaining regions by forwarding the model several times.

However, the heuristic multiple-run model is not end-to-end trainable.

It is the ensemble of multiple-run mined regions but not the single model's own ability that learns the entire action regions.

"Hide-and-seek" BID28 randomly masks out some regions of the input during training, enforcing the model to localize other salient regions when the most salient regions happen to be masked out.

However, all the input regions are masked out with the same probability due to the uniform prior, and it is very likely that most of the time it is the background that is being masked out.

A detailed discussion about related works can be found in Appendix D.To this end, we propose the marginalized average attentional network (MAAN) to alleviate the issue raised by the domination of the most salient region in an end-to-end fashion for weakly-supervised action localization.

Specifically, MAAN suppresses the action prediction response of the most salient regions by employing marginalized average aggregation (MAA) and learning the latent discriminative probability in a principled manner.

Unlike the previous attentional pooling aggregator which calculates the weighted sum with attention weights, MAA first samples a subset of features according to their latent discriminative probabilities, and then calculates the average of these sampled features.

Finally, MAA takes the expectation (marginalization) of the average aggregated subset features over all the possible subsets to achieve the final aggregation.

As a result, MAA not only alleviates the domination by the most salient regions, but also maintains the scale of the aggregated feature within a reasonable range.

We theoretically prove that, with the MAA, the learned latent discriminative probability indeed reduces the difference of response between the most salient regions and the others.

Therefore, MAAN can identify more dense and integral regions for each action.

Moreover, since enumerating all the possible subsets is exponentially expensive, we further propose a fast iterative algorithm to reduce the complexity of the expectation calculation procedure and provide a theoretical analysis.

Furthermore, MAAN is easy to train in an end-to-end fashion since all the components of the network are differentiable.

Extensive experiments on two large-scale video datasets show that MAAN consistently outperforms the baseline models and achieves superior performance on weakly-supervised temporal action localization.

In summary, our main contributions include: (1) a novel end-to-end trainable marginalized average attentional network (MAAN) with a marginalized average aggregation (MAA) module in the weaklysupervised setting; (2) theoretical analysis of the properties of MAA and an explanation of the reasons MAAN alleviates the issue raised by the domination of the most salient regions; (3) a fast iterative algorithm that can effectively reduce the computational complexity of MAA; and (4) a superior performance on two benchmark video datasets, THUMOS14 and ActivityNet1.3, on the weakly-supervised temporal action localization.

incorporates MAA, and introduce the corresponding inference process on weakly-supervised temporal action localization in Sec. 2.4.

Let {x 1 , x 2 , · · · x T } denote the set of snippet-level features to be aggregated, where x t ∈ R m is the m dimensional feature representation extracted from a video snippet centered at time t, and T is the total number of sampled video snippets.

The conventional attentional weighted sum pooling aggregates the input snippet-level features into a video-level representation x. Denote the set of attentional weights corresponding to the snippet-level features as {λ 1 , λ 2 , · · · λ T }, where λ t is a scalar attentional weight for x t .

Then the aggregated video-level representation is given by DISPLAYFORM0 as illustrated in FIG0 (a).

Different from the conventional aggregation mechanism, the proposed MAA module aggregates the features by firstly generating a set of binary indicators to determine whether a snippet should be sampled or not.

The model then computes the average aggregation of these sampled snippet-level representations.

Lastly, the model computes the expectation (marginalization) of the aggregated average feature for all the possible subsets, and obtains the proposed marginalized average aggregated feature.

Formally, in the proposed MAA module, we first define a set of probabilities {p 1 , p 2 , · · · p T }, where each p t ∈ [0, 1] is a scalar corresponding to x t , similar to the notation λ t mentioned previously.

We then sample a set of random variables {z 1 , z 2 , · · · z T }, where z t ∼ Bernoulli(p t ), i.e., z t ∈ {0, 1} with probability P (z t = 1) = p t .

The sampled set is used to represent the subset selection of snippet-level features, in which z t = 1 indicates x t is selected, otherwise not.

Therefore, the average aggregation of the sampled subset of snipped-level representations is given by s = DISPLAYFORM1 z i , and our proposed aggregated feature, defined as the expectation of all the possible subset-level average aggregated representations, is given by DISPLAYFORM2 which is illustrated in FIG0 (b).

Direct learning and prediction with the attention weights λ in Eq.(1) in weakly-supervised action localization leads to an over-response in the most salient regions.

The MAA in Eq. (2) has two properties that alleviate the domination effect of the most salient regions.

First, the partial order preservation property, i.e., the latent discriminative probabilities preserve the partial order with respect to their attention weights.

Second, the dominant response suppression property, i.e., the differences in the latent discriminative probabilities between the most salient items and others are smaller than the differences between their attention weights.

The partial order preservation property guarantees that it does not mix up the action and non-action snippets by assigning a high latent discriminative probability to a snippet with low response.

The dominant response suppression property reduces the dominant effect of the most salient regions and encourages the identification of dense and more integral action regions.

Formally, we present the two properties in Proposition 1 and Proposition 2, respectively.

Detailed proofs can be found in Appendix A and Appendix B respectively.

Proposition 1.

Let z i ∼ Bernoulli(p i ) for i ∈ {1, ..., T }.

Then for T ≥ 2, Eq. (3) holds true, and DISPLAYFORM0 where DISPLAYFORM1 Proposition 1 shows that the latent discriminative probabilities {p i } preserve the partial order of the attention weights {λ i }.

This means that a large attention weight corresponds to a large discriminative probability, which guarantees that the latent discriminative probabilities preserve the ranking of the action prediction response.

Eq. (3) can be seen as a factorization of the attention weight λ i into the multiplication of two components, p i and c i , for i ∈ {1, ..., T }.

p i is the latent discriminative probability related to the feature of snippet i itself.

The factor c i captures the contextual information of snippet i from the other snippets.

This factorization can be considered to be introducing structural information into the aggregation.

Factor c i can be considered as performing a structural regularization for learning the latent discriminative probabilities p i for i ∈ {1, ..., T }, as well as for learning a more informative aggregation.

DISPLAYFORM2 as an index set.

Then I = ∅ and for ∀i ∈ I, ∀j ∈ {1, ..., T } inequality (4) holds true.

DISPLAYFORM3 The index set I can be viewed as the most salient features set.

Proposition 2 shows that the difference between the normalized latent discriminative probabilities of the most salient regions and others is smaller than the difference between their attention weights.

It means that the prediction for each snippet using the latent discriminative probability can reduce the gap between the most salient featuress and the others compared to conventional methods that are based on attention weights.

Thus, MAAN suppresses the dominant responses of the most salient featuress and encourages it to identify dense and more integral action regions.

Directly learning the attention weights λ leans to an over response to the most salient region in weakly-supervised temporal localization.

Namely, the attention weights for only a few snippets are too large and dominate the others, while attention weights for most of the other snippets that also belong to the true action are underestimated.

Proposition 2 shows that latent discriminative probabilities are able to reduce the gap between the most salient features and the others compared to the attention weights.

Thus, by employing the latent discriminative probabilities for prediction instead of the attention weights, our method can alleviate the dominant effect of the most salient region in weakly-supervised temporal localization.

Given a video containing T snippet-level representations, there are 2 T possible configurations for the subset selection.

Directly summing up all the 2 T configurations to calculate x has a complexity of O(2 T ) .

In order to reduce the exponential complexity, we propose an iterative method to calculate x with O(T 2 ) complexity.

Let us denote the aggregated feature of {x 1 , x 2 , · · · x t } with length t as h t , and denote DISPLAYFORM0 z i for simplicity, then we have a set of and the aggregated feature of {x 1 , x 2 , · · · x T } can be obtained as x = h T .

In Eq. FORMULA8 , Z t is the summation of all the z i , which indicates the number of elements selected in the subset.

Although there are 2 t distinct configurations for {z 1 , z 2 , · · · z t }, it has only t + 1 distinct values for Z t , i.e. 0, 1, · · · , t. Therefore, we can divide all the 2 t distinct configurations into t + 1 groups, where the configurations sharing with the same Z t fall into the same group.

Then the expectation h t can be calculated as the summation of the t + 1 parts.

That is, DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 where the m t i , indicating the i th part of h t for group Z t = i, is shown in Eq. FORMULA11 .

DISPLAYFORM4 In order to calculate DISPLAYFORM5 The key idea here is that m The latter case is also related to the probability P (Z t = i − 1).

By denoting q t i−1 = P (Z t = i − 1) for simplicity, we can obtain m t+1 i as a function of several elements: DISPLAYFORM6 Similarly, the computation of q t+1 i = P (Z t+1 = i) comes from two cases: the probability of selecting i − 1 items from the first t items and selecting the (t + 1) th item, i.e., q t i−1 p t+1 ; and the probability of selecting i items all from the first t items and not selecting the (t + 1) th item, i.e., DISPLAYFORM7 We derive the function of m t+1 iand q t+1 i in Proposition 3.

Detailed proofs can be found in Appendix C. DISPLAYFORM8 i ∈ {0, 1, · · · , t + 1} can be obtained recurrently by Eq. FORMULA16 and Eq. FORMULA17 .

DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 Proposition 3 provides a recurrent formula to calculate m t i .

With this recurrent formula, we calculate the aggregation h T by iteratively calculating m t i from i = 1 to t and t = 1 to T .

Therefore, we can obtain the aggregated feature of {x 1 , DISPLAYFORM12 The iterative computation procedure is summarized in Algorithm 1 in Appendix E. The time complexity is O(T 2 ).With the fast iterative algorithm in Algorithm 1, the MAA becomes practical for end-to-end training.

A demonstration of the computation graph for q DISPLAYFORM13

Network Architecture: We now describe the network architecture that employs the MAA module described above for weakly-supervised temporal action localization.

We start from a previous stateof-the-art base architecture, the sparse temporal pooling network (STPN) BID17 .

As shown in FIG4 , it first divides the input video into several non-overlapped snippets and extracts the I3D feature for each snippet.

Each snippet-level feature is then fed to an attention module to generate an attention weight between 0 and 1.

STPN then uses a feature aggregator to calculate a weighted sum of the snippet-level features with these class-agnostic attention weights to create a video-level representation, as shown on the left in FIG5 .

The video-level representation is then passed through an FC layer followed by a sigmoid layer to obtain class scores.

Our MAAN uses the attention module to generate the latent discriminative probability p t and replaces the feature aggregator from the weighted sum aggregation by the proposed marginalized average aggregation, which is demonstrated on the right in FIG5 Training with video-level class labels: Formally, the model first performs aggregation of the snippet-level features (i.e. DISPLAYFORM0 Then, it applies a logistic regression layer (FC layer + sigmoid) to output video-level classification prediction probability.

Specifically, the prediction probability for class c ∈ {1, 2, · · · C} is parameterized as σ c j = σ(w c x j ), where x j is the aggregated feature for video j ∈ {1, ..., N }.

Suppose each video x j is i.i.d and each action class is independent from the other, the negative log-likelihood function (cross-entropy loss) is given as follows: DISPLAYFORM1 where y c j ∈ {0, 1} is the ground-truth video-level label for class c happening in video j and W = [w 1 , ..., w C ].Temporal Action Localization: Let s c = w c x be the video-level action prediction score, and σ(s c ) = σ(w c x) be the video-level action prediction probability.

In STPN, asx = T t=1 λ t x t , the s c can be rewritten as: DISPLAYFORM2 In STPN, the prediction score of snippet t for action class c in a video is defined as: DISPLAYFORM3 where σ(·) denotes the sigmoid function.

In MAAN, asx = E[ DISPLAYFORM4 , according to Proposition 1, the s c can be rewritten as: DISPLAYFORM5 The latent discriminative probability p t corresponds to the class-agnostic attention weight for snippet t. According to Proposition 1 and Proposition 2, c t does not relate to snippet t, but captures the context of other snippets.

w c corresponds to the class-specific weights for action class c for all the snippets, and w c x t indicates the relevance of snippet t to class c. To generate temporal proposals, we compute the prediction score of snippet t belonging to action class c in a video as: DISPLAYFORM6 We denote the s c = (s as the class activation sequence (CAS) for class c. Similar to STPN, the threshold is applied to the CAS for each class to extract the one-dimensional connected components to generate its temporal proposals.

We then perform non-maximum suppression among temporal proposals of each class independently to remove highly overlapped detections.

Compared to STPN (Eq. FORMULA0 ), MAAN (Eq. FORMULA0 ) employs the latent discriminative probability p t instead of directly using the attention weight λ t (equivalent to c t p t ) for prediction.

Proposition 2 suggests that MAAN can suppress the dominant response s c t compared to STPN.

Thus, MAAN is more likely to achieve a better performance in weakly-supervised temporal action localization.

This section discusses the experiments on the weakly-supervised temporal action localization problem, which is our main focus.

We have also extended our algorithm on addressing the weakly-supervised image object detection problem and the relevant experiments are presented in Appendix F.

Datasets.

We evaluate MAAN on two popular action localization benchmark datasets, THU-MOS14 BID10 and ActivityNet1.3 BID8 .

THUMOS14 contains 20 action classes for the temporal action localization task, which consists of 200 untrimmed videos (3,027 action instances) in the validation set and 212 untrimmed videos (3,358 action instances) in the test set.

Following standard practice, we train the models on the validation set without using the temporal annotations and evaluate them on the test set.

ActivityNet1.3 is a large-scale video benchmark for action detection which covers a wide range of complex human activities.

It provides samples from 200 activity classes with an average of 137 untrimmed videos per class and 1.41 activity instances per video, for a total of 849 video hours.

This dataset contains 10,024 training videos, 4,926 validation videos and 5,044 test videos.

In the experiments, we train the models on the training videos and test on the validation videos.

Evaluation Metrics.

We follow the standard evaluation metric by reporting mean average precision (mAP) values at several different levels of intersection over union (IoU) thresholds.

We use the benchmarking code provided by ActivityNet 1 to evaluate the models.

Implementation Details.

We use two-stream I3D networks pre-trained on the Kinetics dataset BID12 ) to extract the snippet-level feature vectors for each video.

All the videos are divided into sets of non-overlapping video snippets.

Each snippet contains 16 consecutive frames or optical flow maps.

We input each 16 stacked RGB frames or flow maps into the I3D RGB or flow models to extract the corresponding 1024 dimensional feature vectors.

Due to the various lengths of the videos, in the training, we uniformly divide each video into T non-overlapped segments, and randomly sample one snippet from each segment.

Therefore, we sample T snippets for each video as the input of the model for training.

We set T to 20 in our MAAN model.

The attention module in FIG4 consists of an FC layer of 1024 × 256, a LeakyReLU layer, an FC layer of 256 × 1, and a sigmoid non-linear activation, to generate the latent discriminative probability p t .

We pass the aggregated video-level representation through an FC layer of 1024 × C followed by a sigmoid activation to obtain class scores.

We use the ADAM optimizer BID14 with an initial learning rate of 5 × 10 −4 to optimize network parameters.

At the test time, we first reject classes whose video-level probabilities are below 0.1.

We then forward all the snippets of the video to generate the CAS for the remaining classes.

We generate the temporal proposals by cutting the CAS with a threshold th.

The combination ratio of two-stream modalities is set to 0.5 and 0.5.

Our algorithm is implemented in PyTorch 2 .

We run all the experiments on a single NVIDIA Tesla M40 GPU with a 24 GB memory.

We first compare our MAAN model on the THUMOS14 dataset with several baseline models that use different feature aggregators in FIG4 to gain some basic understanding of the behavior of our proposed MAA.

The descriptions of the four baseline models are listed below.(1) STPN.

It employs the weighed sum aggregationx = T t=1 λ t x t to generate the video-level representation.

(2) Dropout.

It explicitly performs dropout sampling with dropout probability p = 0.5 in STPN to obtain the video-level representation,x = T t=1 r t λ t x t , r t ∼ Bernoulli(0.5).

We test all the models with the cutting threshold th as 0.2 of the max value of the CAS.

We compare the detection average precision (%) at IoU = [0.1 : 0.1 : 0.9] and the video-level classification mean average precision (%) (denoted as Cls mAP) on the test set in TAB0 .

From TAB0 , we can observe that although all the methods achieve a similar video-level classification mAP, their localization performances vary a lot.

It shows that achieving a good video-level classification performance cannot guarantee obtaining a good snippet-level localization performance because the former only requires the correct prediction of the existence of an action, while the latter requires the correct prediction of both its existence and its duration and location.

Moreover, TAB0 demonstrates that MAAN consistently outperforms all the baseline models at different levels of IoUs in the weakly-supervised temporal localization task.

Both the "Norm" and "SoftmaxNorm" are the normalized weighted average aggregation.

However, the "SoftmaxNorm" performs the worst, because the softmax function over-amplifies the weight of the most salient snippet.

As a result, it tends to identify very few discriminative snippets and obtains sparse and non-integral localization.

The "Norm" also performs worse than our MAAN.

It is the normalized weighted average over the snippet-level representation, while MAAN can be considered as the normalized weighted average (expectation) over the subsetlevel representation.

Therefore, MAAN encourages the identification of dense and integral action segments as compared to "Norm" which encourages the identification of only several discriminative snippets.

MAAN works better than "Dropout" because "Dropout" randomly drops out the snippets with different attention weights by uniform probabilities.

At each iteration, the scale of the aggregated feature varies a lot, however, MAAN samples with the learnable latent discriminative probability and conducts the expectation of keeping the scale of the aggregated feature stable.

Compared to STPN, MAAN also achieves superior results.

MAAN implicitly factorizes the attention weight into c t p t , where p t learns the latent discriminative probability of the current snippet, and c t captures the contextual information and regularizes the network to learn a more informative aggregation.

The properties of MAA disallow the predicted class activation sequences to concentrate on the most salient regions.

The quantitative results show the effectiveness of the MAA feature aggregator.

The temporal CAS generated by MAAN can cover large and dense regions to obtain more accurate action segments.

In the example in FIG8 , MAAN can discover almost all the actions that are annotated in the ground-truth; however, the STPN have missed several action segments, and also tends to only output the more salient regions in each action segment.

Other methods are much sparser compared to MAAN.

The first row of FIG8 shows several action segments in red and in green, corresponding to action segments that are relatively difficult and easy to be localized, respectively.

We can see that all the easily-localized segments contain the whole person who is performing the "HammerThrow" action, while the difficultly-localized segments contain only a part of the person or the action.

Our MAAN can successfully localize the easy segments as well as the difficult segments; however, all the other methods fail on the difficult ones.

It shows that MAAN can identify several dense and integral action regions other than only the most discriminative region which is identified by the other methods.

We also compare our model with the state-of-the-art action localization approaches on the THU-MOS14 dataset.

The numerical results are summarized in TAB1 .

We include both fully and weakly-supervised learning, as in BID17 .

As shown in TAB1 , our implemented STPN performs slightly better than the results reported in the original paper BID17 .

From TAB1 , our proposed MAAN outperforms the STPN and most of the existing weakly-supervised action localization approaches.

Furthermore, our model still presents competitive results compared with several recent fully-supervised approaches even when trained with only video-level labels.

We train the MAAN model on the ActivityNet1.3 training set and compare our performance with the recent state-of-the-art approaches on the validation set in TAB3 .

The action segment in ActivityNet is usually much longer than that of THUMOS14 and occupies a larger percentage of a video.

We use a set of thresholds, which are [0.2, 0.15, 0.1, 0.05] of the max value of the CAS, to generate the proposals from the one-dimensional CAS.

As shown in TAB3 , with the set of thresholds, our implemented STPN performs slightly better than the results reported in the original paper (Nguyen BID23 47.7 43.5 36.3 28.7 19.0 10.3 5.3 --Yeung et al. BID38 48.9 44.0 36.0 26.4 17.1 ----Yuan et al. BID39 51.4 42.6 33.6 26.1 18.8 ----Shou et al. BID24 --40.1 29.4 23.3 13.1 7.9 --Yuan et al. BID41 51.0 45.2 36.5 27.8 17.8 ----Xu et al. BID37 54.5 51.5 44.8 35.6 28.9 ----Zhao et al. 66.0 59.4 51.9 41.0 29.8 ----

Wang et al. 44.4 37.7 28.2 21.1 13.7 ----Singh & Lee BID28 36.4 27.8 19.5 12.7 6.8 ----STPN BID17 BID34 45.1 4.1 0.0 Shou et al. BID24 45.3 26.0 0.2 Xiong et al. 39.1 23.5 5.5Weakly-supervised STPN BID17 29.3 16.9 2.6 STPN BID17 , 2018) .

With the same threshold and experimental setting, our proposed MAAN model outperforms the STPN approach on the large-scale ActivityNet1.3.

Similar to THUMOS14, our model also achieves good results that are close to some of the fully-supervised approaches.

We have proposed the marginalized average attentional network (MAAN) for weakly-supervised temporal action localization.

MAAN employs a novel marginalized average aggregation (MAA) operation to encourage the network to identify the dense and integral action segments and is trained in an end-to-end fashion.

Theoretically, we have proved that MAA reduces the gap between the most discriminant regions in the video to the others, and thus MAAN generates better class activation sequences to infer the action locations.

We have also proposed a fast algorithm to reduce the computation complexity of MAA.

Our proposed MAAN achieves superior performance on both the THUMOS14 and the ActivityNet1.3 datasets on weakly-supervised temporal action localization tasks compared to current state-of-the-art methods.

We thank our anonymous reviewers for their helpful feedback and suggestions.

Prof. Ivor W. Tsang was supported by ARC FT130100746, ARC LP150100671, and DP180100106.A PROOF OF PROPOSITION 1

Proof.

DISPLAYFORM0 In addition, DISPLAYFORM1 Thus, we achieve DISPLAYFORM2 A DISPLAYFORM3 where 1(·) denotes the indicator function.

We achieve Eq. FORMULA2 by partitioning the summation into t + 1 groups .

Terms belonging to group i have DISPLAYFORM4 , and we achieve Eq. (28).

We now give the proof of the recurrent formula of Eq. (29) DISPLAYFORM0 Proof.

DISPLAYFORM1 Then, we have DISPLAYFORM2 Since DISPLAYFORM3 C.3 PROOF OF RECURRENT FORMULA OF q t+1 iWe present the proof of Eq. (39) DISPLAYFORM4 Proof.

DISPLAYFORM5 = z1,z2,···zt,zt+1 DISPLAYFORM6 D RELATED WORK Video Action Analysis.

Researchers have developed quite a few deep network models for video action analysis.

Two-stream networks BID26 and 3D convolutional neural networks (C3D) BID29 are popular solutions to learn video representations and these techniques, including their variations, are extensively used for video action analysis.

Recently, a combination of two-stream networks and 3D convolutions, referred to as I3D , was proposed as a generic video representation learning method, and served as an effective backbone network in various video analysis tasks such as recognition , localization BID23 , and weakly-supervised learning .Weakly-Supervised Temporal Action Localization.

There are only a few approaches based on weakly-supervised learning that rely solely on video-level class labels to localize actions in the temporal domain.

Wang et al. proposed a UntrimmedNet framework, where two softmax functions are applied across class labels and proposals to perform action classification and detect important temporal segments, respectively.

However, using the softmax function across proposals may not be effective for identifying multiple instances.

Singh et al. BID28 designed a Hide-and-Seek model to randomly hide some regions in a video during training and force the network to seek other relevant regions.

However, the randomly hiding operation, as a data augmentation, cannot guarantee whether it is the action region or the background region that is hidden during training, especially when the dropout probabilities for all the regions are the same.

Nguyen et al. BID17 proposed a sparse temporal pooling network (STPN) to identify a sparse set of key segments associated with the actions through attention-based temporal pooling of video segments.

However, the sparse constraint may force the network to focus on very few segments and lead to incomplete detection.

In order to prevent the model from focusing only on the most salient regions, we are inspired to propose the MAAN model to explicitly take the expectation with respect to the average aggregated features of all the sampled subsets from the video.

Feature Aggregators.

Learning discriminative localization representations with only video-level class labels requires the feature aggregation operation to turn multiple snippet-level representations into a video-level representation for classification.

The feature aggregation mechanism is widely adopted in the deep learning literature and a variety of scenarios, for example, neural machine translation BID0 , visual question answering BID9 , and so on.

However, most of these cases belong to fully-supervised learning where the goal is to learn a model that attends the most relevant features given the supervision information corresponding to the task directly.

Many variant feature aggregators have been proposed, ranging from nonparametric max pooling and average pooling, to parametric hard attention BID6 , soft attention BID30 BID22 , second-order pooling BID5 BID15 , structured attention BID13 BID16 , graph aggregators BID43 BID7 , and so on.

Different from the fullysupervised setting where the feature aggregator is designed for the corresponding tasks, we develop a feature aggregator that is trained only with class labels, and then to be used to predict the dense action locations for test data.

Different from the heuristic approaches BID35 BID44 which can be considered as a kind of hard-code attention by erasing some regions with a hand-crafted threshold, we introduce the end-to-end differentiable marginalized average aggregation which incorporates learnable latent discriminative probabilities into the learning process.

E MARGINALIZED AVERAGE AGGREGATION We also evaluate the proposed model on the weakly-supervised object localization task.

For weaklysupervised object localization, we are given a set of images in which each image is labeled only with its category label.

The goal is to learn a model to predict both the category label as well as the bounding box for the objects in a new test image.

Based on the model in BID46 (denoted as CAM model), we replace the global average pooling feature aggregator with other kinds of feature aggregator, such as the weighted sum pooling and the proposed MAA by extending the original 1D temporal version in temporal action localization into a 2D spatial version.

We denote the model with weighted sum pooling as the weighted-CAM model.

For the weighted-CAM model and the proposed MAAN model, we use an attention module to generate the attention weight λ in STPN or the latent discriminative probability p in MAAN.

The attention module consists of a 2D convolutional layer of kernel size 1 × 1, stride 1 with 256 units, a LeakyReLU layer, a 2D convolutional layer of kernel size 1 × 1, stride 1 with 1 unit, and a sigmoid non-linear activation.

We evaluate the weakly-supervised localization accuracy of the proposed model on the CUB-200-2011 dataset BID31 .

The CUB-200-2011 dataset has 11,788 images of 200 categories with 5,994 images for training and 5,794 for testing.

We leverage the localization metric suggested by BID21 for comparison.

This metric computes the percentage of images that is misclassified or with bounding boxes with less than 50% IoU with the groundtruth as the localization error.

We compare our MAA aggregator (MAAN) with the weighted sum pooling (weighted-CAM) and global average pooling (CAM BID48 ).

For MAAN and weighted-CAM, we pool the convolutional feature for aggregation into two different sizes, 4 × 4 and 7 × 7.

We fix all other factors (e.g. network structure, hyper-parameters, optimizer), except for the feature aggregators to evaluate the models.

The localization errors for different methods are presented in TAB5 , where the GoogLeNet-GAP is the CAM model.

Our method outperforms GoogLeNet-GAP by 5.06% in a Top-1 error.

Meanwhile, MAAN achieves consistently lower localization error than weighted-CAM on the two learning schemes.

It demonstrates that the proposed MAAN can improve the localization performance in the weakly-supervised setting.

Moreover, both MAAN and weighted-CAM obtain smaller localization error when employing the 7 × 7 learning scheme than the 4 × 4 learning scheme.

FIG11 visualizes the heat maps and localization bounding boxes obtained by all the compared methods.

The object localization heat maps generated by the proposed MAAN can cover larger object regions and obtain more accurate bounding boxes.

@highlight

A novel marginalized average attentional network for weakly-supervised temporal action localization 