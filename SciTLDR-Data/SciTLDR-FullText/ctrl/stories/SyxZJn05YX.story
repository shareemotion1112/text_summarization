A well-trained model should classify objects with unanimous score for every category.

This requires the high-level semantic features should be alike among samples, despite a wide span in resolution, texture, deformation, etc.

Previous works focus on re-designing the loss function or proposing new regularization constraints on the loss.

In this paper, we address this problem via a new perspective.

For each category, it is assumed that there are two sets in the feature space: one with more reliable information and the other with less reliable source.

We argue that the reliable set could guide the feature learning of the less reliable set during training - in spirit of student mimicking teacher’s behavior and thus pushing towards a more compact class centroid in the high-dimensional space.

Such a scheme also benefits the reliable set since samples become more closer within the same category - implying that it is easilier for the classifier to identify.

We refer to this mutual learning process as feature intertwiner and embed the spirit into object detection.

It is well-known that objects of low resolution are more difficult to detect due to the loss of detailed information during network forward pass.

We thus regard objects of high resolution as the reliable set and objects of low resolution as the less reliable set.

Specifically, an intertwiner is achieved by minimizing the distribution divergence between two sets.

We design a historical buffer to represent all previous samples in the reliable set and utilize them to guide the feature learning of the less reliable set.

The design of obtaining an effective feature representation for the reliable set is further investigated, where we introduce the optimal transport (OT) algorithm into the framework.

Samples in the less reliable set are better aligned with the reliable set with aid of OT metric.

Incorporated with such a plug-and-play intertwiner, we achieve an evident improvement over previous state-of-the-arts on the COCO object detection benchmark.

Classifying complex data in the high-dimensional feature space is the core of most machine learning problems, especially with the emergence of deep learning for better feature embedding (Krizhevsky et al., 2012; BID3 BID10 .

Previous methods address the feature representation problem by the conventional cross-entropy loss, l 1 / l 2 loss, or a regularization constraint on the loss term to ensure small intra-class variation and large inter-class distance (Janocha & Czarneck, 2017; BID16 BID29 BID15 .

The goal of these works is to learn more compact representation for each class in the feature space.

In this paper, we also aim for such a goal and propose a new perspective to address the problem.

Our observation is that samples can be grouped into two sets in the feature space.

One set is more reliable, while the other is less reliable.

For example, visual samples may be less reliable due to low resolution, occlusion, adverse lighting, noise, blur, etc.

The learned features for samples from the reliable set are easier to classify than those from the less reliable one.

Our hypothesis is that the reliable set can guide the feature learning of the less reliable set, in the spirit of a teacher supervising the student.

We refer to this mutual learning process as a feature intertwiner.

In this paper, a plug-and-play module, namely, feature intertwiner, is applied for object detection, which is the task of classifying and localizing objects in the wild.

An object of lower resolution will inevitably lose detailed information during the forward pass in the network.

Therefore, it is well-known that the detection accuracy drops significantly as resolutions of objects decrease.

We can treat samples with high resolution (often corresponds to large objects or region proposals) as the reliable set and samples with low resolution (small instances) as the less reliable set 1 .

Equipped with these two 'prototypical' sets, we can apply the feature intertwiner where the reliable set is leveraged to help the feature learning of the less reliable set.

Without intertwiner in (a), samples are more scattered and separated from each other.

Note there are several samples that are far from its own class and close to the samples in other categories (e.g., class person in blue), indicating a potential mistake in classification.

With the aid of feature intertwiner in (b), there is barely outlier sample outside each cluster.

the features in the lower resolution set approach closer to the features in the higher resolution set -achieving the goal of compact centroids in the feature space.

Empirically, these two settings correspond to the baseline and intertwiner experiments (marked in gray) in TAB3 .

The overall mAP metric increases from 32.8% to 35.2%, with an evident improvement of 2.6% for small instances and a satisfying increase of 0.8% for large counterparts.

This suggests the proposed feature intertwiner could benefit both sets.

Two important modifications are incorporated based on the preliminary intertwiner framework.

The first is the use of class-dependent historical representative stored in a buffer.

Since there might be no large sample for the same category in one mini-batch during training, the record of all previous features of a given category for large instances is recorded by a representative, of which value gets updated dynamically as training evolves.

The second is an inclusion of the optimal transport (OT) divergence as a deluxe regularization in the feature intertwiner.

OT metric maps the comparison of two distributions on high-dimensional feature space onto a lower dimension space so that it is more sensible to measure the similarity between two distributions.

For the feature intertwiner, OT is capable of enforcing the less reliable set to be better aligned with the reliable set.

We name the detection system equipped with the feature intertwiner as InterNet.

Full code suite is available at https://github.com/hli2020/feature intertwiner.

For brevity, we put the descriptions of dividing two sets in the detection task, related work (partial), background knowledge on OT theory and additional experiments in the appendix.

Object detection BID3 He et al., 2017; BID22 BID22 BID12 ) is one of the most fundamental computer vision tasks and serves as a precursor step for other high-level problems.

It is challenging due to the complexity of features in high-dimensional space (Krizhevsky et al., 2012) , the large intra-class variation and inter-class similarity across categories in benchmarks BID4 BID27 .

Thanks to the development of deep networks structure BID25 BID3 and modern GPU hardware acceleration, this community has witnessed a great bloom in both performance and efficiency.

The detection of small objects is addressed in concurrent literature mainly through two manners.

The first is by looking at the surrounding context BID18 since a larger receptive filed in the surrounding region could well compensate for the information loss on a tiny instance during down-sampling in the network.

The second is to adopt a multiscale strategy BID12 BID14 BID24 to handle the scale problem.

This is probably the most effective manner to identify objects in various sizes and can be seen in (almost) all detectors.

Such a practice is a "sliding-window" version of warping features across different stages in the network, aiming for normalizing the sizes of features for objects of different resolutions.

The proposed feature intertwiner is perpendicular to these two solutions.

We provide a new perspective of addressing the detection of small objects -leveraging the feature guidance from high-resolution reliable samples.

Designing loss functions for learning better features.

The standard cross-entropy loss does not have the constraint on narrowing down the intra-class variation.

Several works thereafter have focused on adding new constraints to the intra-class regularization.

Liu et al. BID15 proposed the angular softmax loss to learn angularly discriminative features.

The new loss is expected to have smaller maximal intra-class distance than minimal inter-class distance.

The center loss BID29 ) approach specifically learns a centroid for each class and penalizes the distances between samples within the category and the center.

Our feature intertwiner shares some spirit with this work in that, the proposed buffer is also in charge of collecting feature representatives for each class.

A simple modification BID16 to the inner product between the normalized feature input and the class centroid for the softmax loss also decreases the inner-class variation and improves the classification accuracy.

Our work is from a new perspective in using the reliable set for guiding the less reliable set.

In this paper, we adopt the Faster RCNN pipeline for object detection BID3 BID22 .

In Faster RCNN, the input image is first fed into a backbone network to extract features; a region proposal network BID22 is built on top of it to generate potential region proposals, which are several candidate rectangular boxes that might contain objects.

These region proposals vary in size.

Then the features inside the region are extracted and warped into the same spatial size (by RoI-pooling).

Finally, the warped features are used by the subsequent CNN layers for classifying whether an object exists in the region.

We now explicitly depict how the idea of feature intertwiner could be adapted into the object detection framework.

FIG2 describes the overall pipeline of the proposed InterNet.

A network is divided into several levels based on the spatial size of feature maps.

For each level l, we split the set of region proposals into two categories: one is the large-region set whose size is larger than the output size of RoI-pooling layer and another the small-region set whose size is smaller.

These two sets corresponds to the reliable and less reliable sets, respectively.

For details on the generation of these two sets in object detection, refer to Sec. 6.2 in the appendix.

Feature map P l at level l is fed into the RoI layer and then passed onto a make-up layer.

This layer is designed to fuel back the lost information during RoI and compensate necessary details for instances of small resolution.

The refined high-level semantics after this layer is robust to factors (such as pose, lighting, appearance, etc.) despite sample variations.

It consists of one convolutional layer without Blue blobs stands for the less reliable set (small objects) and green for the reliable set (large ones).

For current level l, feature map P l of the small set is first passed into a RoI-pooling layer.

Then it is fed into a make-up layer, which fuels back the information lost during RoI; it is optimized via the intertwiner module (yellow rectangle), with aid of the reliable set (green).

'OT' (in red) stands for the optimal transport divergence, which aligns information between levels (for details see Sec. 3.3).

P m|l is the input feature map of the reliable set for the RoI layer; m indicates higher level(s) than l.altering the spatial size.

The make-up unit is learned and optimized via the intertwiner unit, with aid of features from the large object set, which is shown in the upstream (green) of FIG2 .The feature intertwiner is essentially a data distribution measurement to evaluate divergence between two sets.

For the reliable set, the input is directly the outcome of the RoI layer of the large-object feature maps P m|l , which correspond to samples of higher level/resolution.

For the less reliable set, the input is the output of the make-up layer.

Both inputs are fed into a critic module to extract further representation of these two sets and provide evidence for intertwiner.

The critic consists of two convolutions that transfer features to a larger channel size and reduce spatial size to one, leaving out of consideration the spatial information.

A simple l 2 loss can be used for comparing difference between two sets.

The final loss is a combination of the standard detection losses BID22 and the intertwiner loss across all levels.

The detailed network structure of the make-up and critic module in the feature intertwiner is shown in the appendix (Sec. 6.6).

There are two problems when applying the aforementioned pipeline into application.

The first is that the two sets for the same category often do not occur simultaneously in one mini-batch; the second is how to choose the input source for the reliable set, i.e., feature map P m|l for the large object set.

We address these two points in the following sections.

The goal of the feature intertwiner is to have samples from less reliable set close to the samples within the same category from the reliable set.

In one mini-batch, however, it often happens that samples from the less reliable set are existent while samples of the same category from the reliable set are non-existent (or vice versa).

This makes it difficult to calculate the intertwiner loss between two sets.

To address this problem, we use a buffer B to store the representative (prototype) for each category.

Basically the representative is the mean feature representation from large instances.

Let the set of features from the large-region object on all levels be F (large) critic ; each sample consisting of the large set F be f (j) , where j is the sample index and its feature dimension is d. The buffer could be generated as a mapping from sample features to class representative: DISPLAYFORM0 DISPLAYFORM1 where the total number of classes is denoted as N cls .

Each entry b i in the buffer B is referred to as the representative of class i. Every sample, indexed by j in the large object set, contributes to the class representative i * if its label belongs to i * .

Here we denote i * as the label of sample j; and Z in Eqn.

(2) denotes the total number of instances whose label is i * .

The representative is deemed as a reliable source of feature representation and could be used to guide the learning of the less reliable set.

There are many options to design the mapping M, e.g., the weighted average of all features in the past iterations during training within the class as shown in Eqn.

FORMULA1 , feature statistics from only a period of past iterations, etc.

We empirically discuss different options in TAB3 .Equipped with the class buffer, we define the intertwiner loss between two sets as: DISPLAYFORM2 where D is a divergence measurement; f (small,l,j) critic denotes the semantic feature after critic of the j-th sample at level l in the less reliable set (small instances).

Note that the feature intertwiner is proposed to optimize the feature learning of the less reliable set for each level.

During inference, the green flow as shown in FIG2 for obtaining the class buffer will be removed.

Discussion on the intertwiner.

(a) Through such a mutual learning, features for small-region objects gradually encode the affluent details from large-region counterparts, ensuring that the semantic features within one category should be as much similar as possible despite the visual appearance variation caused by resolution change.

The resolution imperfection of small instances inherited from the RoI interpolation is compensated by mimicking a more reliable set.

Such a mechanism could be seen as a teacher-student guidance in the self-supervised domain

It is observed that if the representative b i is detached in back-propagation process (i.e., no backward gradient update in buffer), performance gets better.

The buffer is used as the guidance for less reliable samples.

As contents in buffer are changing as training evolves, excluding the buffer from network update would favorably stabilize the model to converge.

Such a practice shares similar spirit of the replay memory update in deep reinforcement learning.

(c) The buffer statistics come from all levels.

Note that the concept of "large" and "small" is a relative term: large proposals on current level could be deemed as "small" ones on the next level.

However, the level-agnostic buffer would always receive semantic features for (strictly) large instances.

This is why there are improvements across all levels (large or small objects) in the experiments.

How to acquire the input source, denoted as P (large,l) , i.e., feature maps of large proposals, to be fed into the RoI layer on current level l?

The feature maps, denoted by P l or P m , are the output of ResNet at different stages, corresponding to different resolutions.

Altogether we use four stages, i.e., P 2 to P 5 ; P 2 corresponds to feature maps of the highest resolution and P 5 has the lowest resolution.

The inputs are crucial since they serve as the guidance targets to be learned by small instances.

There are several choices, which is depicted in Fig. 3 .

Option (a): P (large,l) = P l .

The most straightforward manner would be using features on current level as input for large object set.

This is inappropriate since P l is trained in RPN specifically for identifying small objects; adopting it as the source could contain noisy details of small instances.

DISPLAYFORM0 Option (b): P (large,l) = P m .

Here m and l denote the index of stage/level in ResNet and m > l. One can utilize the higher level feature map(s), which has the proper resolution for large objects.

Compared with P l , P m have lower resolution and higher semantics.

For example, consider the large instances assigned to level l = 2 (how to assign large and small instances is discussed in the appendix Sec. 6.2), P m indicates three stages m = 3, 4, 5.

However, among these large instances, some of them are deemed as small objects on higher level m -implying that those feature maps P m might not carry enough information.

They would also have to be up-sampled during the RoI operation for updating the buffer on current level l. TAB6 in the appendix for example, among the assigned 98 proposals on level 2, there are 31 (11 on level 3 and 20 on level 4) objects that have insufficient size (smaller than RoI's output).

Hence it might be inappropriate to directly use the high-level feature map as well.

Option (c): P (large,l) = P m|l F(P m ).

P m is first up-sampled to match the size at P l and then is RoI-pooled with outcome denoted as P m|l .

The up-sampling operation aims at optimizing a mapping F : P m → P m|l that can recover the information of large objects on a shallow level.

F could be as simple as a bilinear interpolation or a neural network.

These three options are empirically reported in Table 1 .

The baseline model in (b) corresponds to the default setting in cases 2d, 2e of TAB3 , where the feature intertwiner is adopted already.

There is a 0.8% AP boost from option (b) to (c), suggesting that P m for large objects should be converted back to the feature space of P l .

The gain from (a) to (c) is more evident, which verifies that it might not be good to use P l directly.

More analysis is provided in the appendix.

Option (c) is a better choice for using the reliable feature set of large-region objects.

Furthermore, we build on top of this choice and introduce a better alternative to build the connection between P l and P m|l , since the intertwiner is designed to guide the feature learning of the less reliable set on the current level.

If some constraint is introduced to keep information better aligned between two sets, the modified input source P m|l for large instance would be more proper for the other set to learn.

(large,l) = OT(P l , P m|l ).

The spirit of moving one distribution into another distribution optimally in the most effective manner fits well into the optimal transport (OT) domain BID19 .

In this work, we incorporate the OT unit between feature map P l and P m|l , which serve as inputs before the RoI-pooling operation.

A discretized version BID6 BID2 of the OT divergence is employed as an additional regularization to the loss: DISPLAYFORM0 where the non-positive P serves as a proxy for the coupling and satisfies P T 1 C2 = 1 C1 , P 1 C1 = 1 C2 .

·, · indicates the Frobenius dot-product for two matrices and 1 m := (1/m, . . .

, 1/m) ∈ R m + .

Now the problem boils down to computing P given some ground cost Q. We adopt the Sinkhorn algorithm BID26 in an iterative manner to compute W Q , which is promised to have a differentiable loss function.

The OT divergence is hence referred to as Sinkhorn divergence.

Given features maps P m from higher level, the generator network F up-samples them to match the size of P l and outputs P m|l .

The channel dimension of P l and P m|l is denoted as C. The critic unit H (not the proposed critic unit in the feature intertwiner) is designed to reduce the spatial dimensionality of input to a lower dimension k while keeping the channel dimension unchanged.

The number of samples in each distribution is C. The outcome of the critic unit in OT module is denoted as p l , p m|l , respectively.

We choose cosine distance as the measurement to calculate the distance between manifolds.

The output is known as the ground cost Q x,y , where x, y indexes the sample in these two distributions.

The complete workflow to compute the Sinkhorn divergence is summarized in Alg.

1. Note that each level owns their own OT module W l Q (P l , P m ) = OT(P l , P m|l ).

The total loss for the detector is summarized as: DISPLAYFORM1 where L standard is the classification and regression losses defined in most detectors BID22 .Algorithm 1 Sinkhorn divergence W Q adapted for object detection (red rectangle in FIG2 Input: Feature maps on current and higher levels, P l , P m The generator network F and the critic unit in OT module H Output: Sinkhorn loss W l Q (P l , P m ) = OT(P l , P m|l ) Upsample via generator P m|l = F(P m ) Feed both inputs into critic p l = H(P l ), p m|l = H(P m|l ) DISPLAYFORM2 DISPLAYFORM3 known as Sinkhorn iterate end for Compute the proxy matrix PWhy prefer OT to other alternatives.

As proved in BID0 , the OT metric converges while other variants (KL or JS divergence) do not in some scenarios.

OT provides sensible cost functions when learning distributions supported by low-dim manifolds (in our case, p l and p m|l ) while other alternatives do not.

As verified via experiments in Table 1 , such a property could facilitate the training towards a larger gap between positive and false samples.

In essence, OT metric maps the comparison of two distributions on high-dimensional feature space onto a lower dimension space.

The use of Euclidean distance could improve AP by around 0.5% (see Table 1 , (d) l 2 case), but does not gain as much as OT does.

This is probably due to the complexity of feature representations in high-dimension space, especially learned by deep models.

We evaluate InterNet on the object detection track of the challenging COCO benchmark (TsungYi Lin, 2015) .

For training, we follow common practice as in BID22 He et al., 2017) and use the trainval35k split (union of 80k images from train and a random 35k subset of images from 40k val split) for training.

The lesion and sensitivity studies are reported by evaluating on the minival split (the remaining 5k images from val).

For all experiments, we use depth 50 or 101 ResNet BID3 with FPN BID12 ) constructed on top.

We base the framework on Mask-RCNN (He et al., 2017) without the segmentation branch.

All ablative analysis adopt austere settings: training and test image scale only at 512; no multi-scale and data augmentation (except for horizontal flip).

Details on the training and test procedure are provided in the appendix (Sec. 6.5).

Baseline comparison.

TAB3 lists the comparison of InterNet to baseline, where both methods shares the same setting.

On average it improves by 2 points in terms of mAP.

The gain for small objects is much more evident.

Note that our method also enhances the detection of large objects (by 0.8%), since the last level also participates in the intertwiner update by comparing its similarity feature to the history buffer, which requires features of the same category to be closer to each other.

The last level does not contribute to the buffer update though.

Assignment strategy (analysis based on Sec. 6.2).

Table 2a also investigates the effect of different region proposal allocations. 'by RoI size' divides proposals whose area is below the RoI threshold in TAB6 as small and above as large; 'more on higher' indicates the base value in Eqn.

(6) is smaller (=40); the default setting follows BID12 where the base is set to 224.

Preliminary, we think putting more proposals on higher levels (the first two cases) would balance the workload of the intertwiner; since the default setting leans towards too many proposals on level 2.

However, there is no gain due to the mis-alignment with RPN training.

The distribution of anchor templates in RPN does not alter accordingly, resulting in the inappropriate use of backbone feature maps.

Intertwinter loss.

Upper block in TAB3 shows a factor of 1.0 to be merged on the total loss whereas lower block depicts a specific factor that achieves better AP than others.

The simple l 2 loss achieves slightly better than the KL divergence, where the latter is computed as DISPLAYFORM0 The l 1 option is by around 1 point inferior than these two and yet still verifies the effectiveness of the intertwiner module compared with baseline (34.2 vs 32.8) -implying the generalization ability of our method in different loss options.

How does the intertwiner module affect learning?

By measuring the divergence between two sets (i.e., small proposals in the batch and large references in the buffer), we have gradients, as the influence, back-propagated from the critic to make-up layer.

In the end, the make-up layer is optimized to enforce raw RoI outputs recovering details even after the loss from reduced resolution.

The naive design denoted by 'separate' achieves 34.0% AP as shown in TAB3 .

To further make the influence of the intertwiner stronger, we linearly combine the features after critic with the original detection feature (with equal weights, aka 0.5; not shown in FIG2 ) and feed this new combination into the final detection heads.

This improves AP by 1 point (denoted as 'linear' in TAB3 ).

The 'naive add' case with equal weights 1 does not work (loss suddenly explodes during training), since the amplitude of features among these two sources vary differently if we simply add them.

TAB3 shows that it does not.

A natural thought could be having a window size of K and sliding the window to keep the most recent features recorded.

In general, larger size improves performance (see case '2000' vs the size of 'one epoch' where batch size is 8, 37.3% → 38.8%).

In these cases, statistics of large object features for one category cannot reflect the whole training set and it keeps alternating as network is updated.

Using 'all history' data by running averaging not only saves memory but also has the whole picture of the data.

Preliminary, we choose a decayed scheme that weighs more to recent features than ones in the long run, hoping that the model would be optimized better as training evolves.

However, experiments does not accord with such an assumption: AP is better where features are equally averaged (c.f., 40.5% and 39.2%) in terms of network evolution.

Unified or level-based buffer?

Unified.

TAB3 upper block reports such a perspective.

In early experiments, we only have one unified buffer in order to let objects on the last level also involved in the intertwiner.

Besides, the visual features of large objects should be irrelevant of scale variation.

This achieves a satisfying AP already.

We also try applying different buffers on each level 3 .

The performance improvement is slight, although the additional memory cost is minor.

Other investigations.

As discussed at the end of Sec. 3.1, detaching buffer transaction from gradient update attracts improvement (40.5% vs 40.1% in TAB3 ).

Moreover, we tried imposing stronger supervision on the similarity feature of large proposals by branching out a cross-entropy loss, for purpose of diversifying the critic outputs among different categories.

However, it does not work and this additional loss seems to dominate the training process.

Performance.

We list a comparison of our InterNet with previous state-of-the-arts in TAB7 in the appendix.

Without multi-scale technique, ours (42.5%) still favorably outperforms other two-stage detectors (e.g., Mask-RCNN, 39.2%) as well as one-stage detector (SSD, 31.2%).

Moreover, we showcase in FIG3 the per-class improvement between the baseline and the improved model after adopting feature intertwiner in distinct drop for the 'couch' class, we find that for a large couch among samples on COCO, usually there sit a bunch of people, stuff, pets, etc.

And yet the annotations in these cases would cover the whole scenario including these noises, making the feature representation of the large couch quite inaccurate.

The less accurate features would guide the learning of their small counterparts, resulting in a lower AP for this class.

Model complexity and timing.

The feature intertwiner only increases three light-weight conv.

layers at the make-up and critic units.

The usage of class buffer could take up a few GPU memory on-the-fly; however, since we adopt an 'all-history' strategy, the window size is just 1 instead of a much larger K. The additional cost to the overall model parameters is also from the OT module for each level; however, we find using just one conv.

layer for the critic H and two conv.

layers with small kernels for generator F is enough to achieve good result.

Training on 8 GPUs with batch size of 8 takes around 3.4 days; this is slower than Mask-RCNN reported in (He et al., 2017) .

The memory cost on each card is 9.6 GB, compared with baseline 8.3 GB.

The inference runs at 325ms per image (input size is 800) on a Titan Pascal X, increasing around 5% time compared to baseline (308 ms).

We do not intentionally optimize the codebase, however.

In this paper, we propose a feature intertwiner module to leverage the features from a more reliable set to help guide the feature learning of another less reliable set.

This is a better solution for generating a more compact centroid representation in the high-dimensional space.

It is assumed that the high-level semantic features within the same category should resemble as much as possible among samples with different visual variations.

The mutual learning process helps two sets to have closer distance within the cluster in each class.

The intertwiner is applied on the object detection task, where a historical buffer is proposed to address the sample missing problem during one mini-batch and the optimal transport (OT) theory is introduced to enforce the similarity among the two sets.

Since the features in the reliable set serve as teacher in the feature learning, careful preparation of such features is required so that they would match the information in the small-object set.

This is why we design different options for the large set and finally choose OT as a solution.

With aid of the feature intertwiner, we improve the detection performance by a large margin compared to previous state-of-the-arts, especially for small instances.

Feature intertwiner is positioned as a general alternative to feature learning.

As long as there exists proper division of one reliable set and the other less reliable set, one can apply the idea of utilizing the reliable set guide the feature learning of another, based on the hypothesis that these two sets share similar distribution in some feature space.

One direction in the future work would be applying feature intertwiner into other domains, e.g., data classification, if proper set division are available.

Self-supervised learning.

The buffer in the feature intertwiner can be seen as utilizing non-visual domain knowledge on a set of data to help supervise the feature learning for another set in highdimensional space.

Such a spirit falls into the self-supervised learning domain.

In , Chen et al. proposed a knowledge distillation framework to learn compact and accurate object detectors.

A teacher model with more capacity is designed to provide strong information and guide the learning of a lite-weight student model.

The center loss BID29 ) is formulated to learn a class center and penalize samples that have a larger distance with the centroid.

It aims at enlarging inter-class resemblance with cross-entropy (CE) loss as well as narrowing down innerclass divergence for face recognition.

In our work, the feature intertwiner gradually aggregates statistics of a meta-subset and utilizes them as targets during the feature learning of a less accurate (yet holding a majority) subset.

We are inspired by the proposal-split mechanism in object detection domain to learn recognition at separate scales in the network.

The self-paced learning framework (Kumar et al., 2010) deals with two sets as well, where the easy examples are first introduced to optimize the hidden variable and later on during training, the hard examples are involved.

There is no interaction between the two sets.

The division is based on splitting different samples.

In our framework, the two sets mutually help and interact with each other.

The goal is towards optimizing a more compact class centroid in the feature space.

These are two different branches of work.

Optimal transport (OT) has been applied in two important tasks.

One is for transfer learning in the domain adaption problem.

Lu et al. BID17 explored prior knowledge in the cost matrix and applied OT loss as a soft penalty for bridging the gap between target and source predictions.

Another is for estimating generative models.

In BID23 , a metric combined with OT in primal form with an energy distance results in a highly discriminative feature representation with unbiased gradients.

Genevay et al. BID6 presents the first tractable method to train large-scale generative models using an OT-based loss.

We are inspired by these works in sense that OT metric is favorably competitive to measure the divergence between two distributions supported on low-dimensional manifolds.

In this paper we adopt the ResNet model BID3 with feature pyramid dressings BID12 ) constructed on top.

It generates five levels of feature maps to serve as inputs for the subsequent RPN and detection branches.

Denote the level index as l = {1, . . . , 5} and the corresponding feature maps as P l .

Level l = 1 is the most shallow stage with more local details for detecting tiny objects and level l = 5 is the deepest stage with high-level semantics.

Let A = {a j } denote the whole set of proposals generated by RPN from l 2 to l 6 (level six is generated from l 5 , for details refer to BID12 ).

The region proposals are divided into different levels from l 2 to l 5 : DISPLAYFORM0 where a 0 =4 as in BID12 ; base=224 is the canonical ImageNet pre-training setting.

TAB6 shows a detailed breakdown 4 of the proposal allocation based on Eqn.

(6).

We can see most proposals from RPN focus on identifying small objects and hence are allocated at shallow level l = 2.

The threshold is set to be the ratio of RoI output's area over the area of feature map.

For example, threshold on l = 3 is obtained by (14/64) 2 , where 14 is the RoI output size as default setting.

Proposals whose area is below the threshold suffer from the inherent design during RoI operation -these feature outputs are up-sampled by a simple interpolation.

The information of small regions is already lost and RoI layer does not help much to recover them back.

As is shown on the fourth row ("below # / above #"), such a case holds the majority.

This observation brings in the necessity of designing a meta-learner to provide guidance on feature learning of small objects due to the loophole during the RoI layer.

For level l in the network, we define small proposals (or RoIs) to be those already assigned by (6) and large to be those above l: DISPLAYFORM1 where the superscript s,b denotes the set of small and large proposals, respectively.

The last two rows in TAB6 show an example of the assignment.

These RoIs are then fed into the RoI-pooling layer 5 to generate output features maps for the subsequent detection pipeline to process.

One may wonder the last level do not have large objects for reference based on Eqn.

FORMULA10 .

In preliminary experiments, leaving proposals on the last level out of the intertwiner could already improve the overall performance; however, if the last level is also involved (since the buffer is shared across all levels), AP for large objects also improves.

See the experiments in Sec. 4.1 for detailed analysis.

Let u , u indicate the individual sample after degenerating high-dimensional features P m|l , P l from two spaces into low manifolds.

u , u are vectors of dimension k. The number of samples in these two distributions is denoted by C 1 and C 2 , respectively.

The OT metric between two joint probability distributions supported on two spaces (U, U) is defined as the solution of the linear program BID2 .

Denote the data and reference distribution as P ψ , P r ∈ Prob(U) 6 , respectively, we have the continuous form of OT divergence: DISPLAYFORM0 where γ is a coupling; Γ is the set of couplings that consists of joint distributions.

Intuitively, γ(u , u) implies how much "mass" must be transported from u to u in order to transform the distribution P ψ into P r ; Q is the "ground cost" to move a unit mass.

Eqn. (8) above becomes the p-Wasserstein distance (or loss, divergence) between probability measures when U is equipped with a distance D U and Q = D U (u , u) p , for some exponent p.

The biased version of Sinkhorn divergence used in Table 1 is defined by: DISPLAYFORM1 More analysis on Table 1 .

All these options have been discussed explicitly at the beginning of Sec. 3.3.

Option (a) is inferior due to the inappropriateness of feature maps; (b) serves as the baseline and used as the default setting in TAB3 .

Options in (c) verifies that up-sampling feature maps from higher-level onto current level is preferable; F being a neural net ensures better improvement.

Options in (d) illustrates the case where a supervision signal is imposed onto pair (P l , P m|l ) to make better alignment between them.

We can observe that OT outperforms other variants in this setup.

Moreover, we tried a biased version BID6 of the Sinkhorn divergence.

However, it does not bring in much gain compared to the previous setup.

Besides, it could burden system efficiency during training (although it is minor considering the total time per iteration).

Such a phenomenon could result from an improper update of critic and generator inside the OT module, since the gradient flow would be iterated twice more for the last two terms above.

Extending OT divergence to image classification.

We also testify OT divergence on CIFAR-10 (Krizhevsky & Hinton, 2009) where feature maps between stages are aligned via OT.

Test error decreases by around 1.3%.

This suggests the potential application of OT in various vision tasks.

Different from OT in generative models, we deem the channel dimension as different samples to compare, instead of batch-wise manner as in BID23 ; and treat the optimization of F and H in a unified min problem, as opposed to the adversarial training BID6 .6.4 COMPARISON TO STATE-OF-THE-ARTS ON COCO AND PASCAL VOC To further verify the effectiveness of the feature intertwiner, we further conduct experiments on the PASCAL VOC 2007 dataset.

The results are shown in Table 5 .

Two network structures are adopted.

For ResNet-101, the division of the four levels are similar as ResNet-101-FPN on COCO; for VGG-16, we take the division similarly as stated in SSD BID14 .

Specifically, the output of layer 'conv7', 'conv8 2', 'conv9 2' and 'conv10 2' are used for P 2 to P 5 , respectively.

Our method performs favorably against others in both backbone structures on the PASCAL dataset.

We adopt the stochastic gradient descent as optimizer.

Initial learning rate is 0.01 with momentum 0.9 and weight decay 0.0001.

Altogether there are 13 epoches for most models where the learning rate is dropped by 90% at epoch 6 and 10.

We find the warm-up strategy BID8 barely improves the performance and hence do not adopt it.

The gradient clip is introduced to prevent training loss to explode in the first few iterations, with maximum gradient norm to be 5.

Batch size is set to 8 and the system is running on 8 GPUs.

The object detector is based on Mask-RCNN (or Faster-RCNN).

RoIAlign is adopted for better performance.

The model is initialized with the corresponding ResNet model pretrained on ImageNet.

The new proposed feature intertwiner module is trained from scratch with standard initialization.

The basic backbone structure for extracting features is based on FPN network BID12 , where five ResNet blocks are employed with up-sampling layers.

The region proposal network consists of one convolutional layer with one classification and regression layer.

The classifier structure is similar as RPN's -one convolution plus one additional classification/regression head.

Non-maximum suppression (NMS) is used during RPN generation and detection test phase.

Threshold for RPN is set to 0.7 while the value is 0.3 during test.

We do not adopt a dense allocation of anchor templates as in some literature BID14 ); each pixel on a level only has the number of anchors the same as the number of aspect ratios (set to 0.5, 1 and 2).

Each level l among the five stages owns a unique anchor size: 32, 64, 128, 256, and 512.

The detailed network architecture on the make-up layer and critic layer are shown below.

Output size Layers in the make-up module B × C l × 14 × 14 conv2d(C l , C l , k = 3, padding = 1) B × C l × 14 × 14 batchnorm2d(C l ) B × C l × 14 × 14 relu(·) Table 6 : Network structure of the make-up unit, which consists of one convolutional layer without altering the spatial size.

Input: RoI output of the small-set feature map P l .

We denote the output of the make-up layer as P l .

B is the batch size in one mini-batch; C l is the number of channels after the feature extractor in ResNet blocks for each level.

For example, when l = 2, C l = 256, etc.

Layers in the critic module B × 512 × 7 × 7 conv2d(C l , 512, k = 3, padding = 1, stride = 2) B × 512 × 7 × 7 batchnorm2d(512) B × 512 × 7 × 7 relu(·) B × 1024 × 1 × 1 conv2d(512, 1024, k = 7) B × 1024 × 1 × 1 batchnorm1d(1024) B × 1024 × 1 × 1 relu(·) B × 1024 × 1 × 1 sigmoid(·) Table 7 : Network structure of the critic unit.

Input: for large set, it is the RoI output of the large-set feature map P m|l and for small set, it is the output of the make-up layer P l .

B is the batch size in one mini-batch; C l is the number of channels in ResNet blocks.

<|TLDR|>

@highlight

(Camera-ready version) A feature intertwiner module to leverage features from one accurate set to help the learning of another less reliable set.