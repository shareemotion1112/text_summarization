We address the problem of learning to discover 3D parts for objects in unseen categories.

Being able to learn the geometry prior of parts and transfer this prior to unseen categories pose fundamental challenges on data-driven shape segmentation approaches.

Formulated as a contextual bandit problem, we propose a learning-based iterative grouping framework which learns a grouping policy to progressively merge small part proposals into bigger ones in a bottom-up fashion.

At the core of our approach is to restrict the local context for extracting part-level features, which encourages the generalizability to novel categories.

On a recently proposed large-scale fine-grained 3D part dataset, PartNet, we demonstrate that our method can transfer knowledge of parts learned from 3 training categories to 21 unseen testing categories without seeing any annotated samples.

Quantitative comparisons against four strong shape segmentation baselines show that we achieve the state-of-the-art performance.

Perceptual grouping has been a long-standing problem in the study of vision systems (Hoffman & Richards, 1984) .

The process of perceptual grouping determines which regions of the visual input belong together as parts of higher-order perceptual units.

Back to the 1930s, Wertheimer (1938) listed several vital factors, such as similarity, proximity, and good continuation, which lead to visual grouping.

To this era of deep learning, grouping cues can be learned from massive annotated datasets.

However, compared with human visual system, these learning-based segmentation algorithms are far inferior for objects from unknown categories.

We are interested in attacking a specific problem of this kind -zero-shot part discovery for 3D shapes.

We choose to study the zero-shot learning problem on 3D shape data instead of 2D image data, because part-level similarity across object categories in 3D is more salient and less affected by various distortions introduced in the imaging process.

Work done while Tiange Luo, Kaichun Mo, Jiarui Xu, and Siyu Hu were visiting UC San Diego.

To motive our approach, we first review the key idea and limitation of existing 3D part segmentation methods.

With the power of big data, deep neural networks that learn data-driven features to segment shape parts, such as (Kalogerakis et al., 2010; Graham et al., 2018; Mo et al., 2019c) , have demonstrated the state-of-the-art performance on many shape segmentation benchmarks (Yi et al., 2016; Mo et al., 2019c) .

These networks usually have large receptive fields that cover the whole input shape, so that global context can be leveraged to improve the recognition of part semantics and shape structures.

While learning such features leads to superior performance on the training categories, they often fail miserably on unseen categories (Figure 1 ) due to the difference of global shapes.

On the contrary, classical shape segmentation methods, such as (Kaick et al., 2014 ) that use manually designed features with relatively local context, can often perform much better on unseen object categories, although they tend to give inferior segmentation results on training categories.

In fact, many globally different shapes share similar part-level structures.

For example, airplanes, cars, and swivel chairs all have wheels, even though their global geometries are totally different.

Having learned the geometry of wheels from airplanes should help recognize wheels for cars and swivel chairs.

In this paper, we aim to invent a learning-based framework that will by design avoid using excessive context information that hurts cross-category generalization.

We start from learning to propose a pool of superpixel-like sub-parts for each shape.

Then, we learn a grouping policy that seeks to progressively group sub-parts and increase recognition context.

What lies in the heart of our algorithm is to learn a function to assess whether two parts should be grouped.

Different from prior deep segmentation work that learns point features for segmentation mask prediction, our formulation essentially learns part-level features.

Borrowing ideas from Reinforcement Learning (RL), we formalize the process as a contextual bandit problem and train a local grouping policy to iteratively pick a pair of most promising sub-parts for grouping.

In this way, we restrict that our features only convey information within the local context of a part.

Our learning-based agglomerative clustering framework deviates drastically from the prevailing deep segmentation pipelines and makes one step towards generalizable part discovery in unseen object categories.

To summarize, we make the following contributions:

??? We formulate the task of zero-shot part discovery on a large-scale fine-grained shape segmentation benchmark PartNet (Mo et al., 2019c ); ??? We propose a learning-based agglomerative clustering framework that learns to do part proposals and grouping from training categories and generalizes to unseen novel categories; ??? We quantitatively compare our approach to several baseline methods and demonstrate the state-of-the-art results for part discovery in unseen object categories.

Shape segmentation has been a classic and fundamental problem in computer vision and graphics.

Dated back to 1990s, researchers have started to design heuristic geometric criterion for segmenting 3D meshes, including methods based on morphological watersheds (Mangan & Whitaker, 1999) , K-means (Shlafman et al., 2002) , core extraction (Katz et al., 2005) , graph cuts (Golovinskiy & Funkhouser, 2008) , random walks (Lai et al., 2008) , spectral clustering (Liu & Zhang, 2004) and primitive fitting (Attene et al., 2006a) , to name a few.

See Attene et al. (2006b) ; Shamir (2008) ; Chen et al. (2009) for more comprehensive surveys on mesh segmentation.

Many papers study mesh cosegmentation that discover consistent part segmentation over a collection of shapes (Golovinskiy & Funkhouser, 2009; Huang et al., 2011; Sidi et al., 2011; Hu et al., 2012; Wang et al., 2012; Van Kaick et al., 2013) .

Our approach takes point clouds as inputs as they are closer to the real-world scanners.

Different from meshes, point cloud data lacks the local vertex normal and connectivity information.

Kaick et al. (2014) segments point cloud shapes under the part convexity constraints.

Our work learns shared part priors from training categories and thus can adapt to different segmentation granularity required by different end-stream tasks.

In recent years, with the increasing availability of annotated shape segmentation datasets (Chen et al., 2009; Yi et al., 2016; Mo et al., 2019c) , many supervised learning approaches succeed in refreshing the state-of-the-arts.

Kalogerakis et al. (2010); Guo et al. (2015) ; Wang et al. (2018a) learn to label mesh faces with semantic labels defined by human.

See Xu et al. (2016) for a recent survey.

More recent works propose novel 3D deep network architectures segmenting shapes represented as 2D images (Kalogerakis et al., 2017 ), 3D voxels (Maturana & Scherer, 2015 , sparse volumetric representations (Klokov & Lempitsky, 2017; Riegler et al., 2017; Wang et al., 2017; Graham et al., 2018) , point clouds (Qi et al., 2017a; b; Wang et al., 2018b; Yi et al., 2019b) and graph-based representations (Yi et al., 2017) .

These methods take advantage of sufficient training samples of seen categories and demonstrate appealing performance for shape segmentation.

However, they often perform much worse when testing on unseen categories, as the networks overfit their weights to the global shape context in training categories.

Our work focus on learning context-free part knowledges and perform part discovery in a zero-shot setting on unseen object classes.

There are also a few relevant works trying to reduce supervisions for shape part segmentation.

Makadia & Yumer (2014) learns from sparsely labeled data that only one vertex per part is given the ground-truth.

Yi et al. (2016) proposes an active learning framework to propogate part labels from a selected sets of shapes with human labeling.

Lv et al. (2012) proposes a semi-supervised Conditional Random Field (CRF) optimization model for mesh segmentation.

Shu et al. (2016) proposes an unsupervised learning method for learning features to group superpixels on meshes.

Our work processes point cloud data and focus on a zero-shot setting, while part knowledge can be learned from training categories and transferred to unseen categories.

Our work is also related to many recent research studying learning based bottom-up methods for 2D instance segmentation.

These methods learn an per-pixel embedding and utilize a clustering algorithm (Newell et al., 2017; Fathi et al., 2017) as post-process or integrating a recurrent meanshift module (Kong & Fowlkes, 2018) to generate final instances.

Bai & Urtasun (2017) predicts the energy of the watershed transform and Liu et al. (2017) predicts object breakpoints and use a cascade of networks to group the pixels into lines, components and objects sequentially.

Our work is significant different from previous methods as our method does not rely on an fully convolutional neural network to process the whole scene.

Our work can generalize better to unseen categories as our method reduces the influences of context.

Some works in the 3D domain try to use part-level information are also related to our work (Yi et al., 2019a; Achlioptas et al., 2019; Mo et al., 2019b; a) .

Achlioptas et al. (2019) shows that the shared part-based structure of objects enables zero-shot 3D recognition based on language.

To reduce the overfitting of global contextual information, our approach would exploit the part prior encoded in the dataset and involve only part-level inductive biases.

We consider the task of zero-shot shape part discovery on 3D point clouds in unseen object categories.

For a 3D shape S (e.g. a 3D chair model), we consider the point cloud C S = {p 1 , p 2 , ?? ?? ?? , p N } sampled from the surface of the 3D model.

A part P i = {p i1 , p i2 , ?? ?? ?? , p it } ??? C S defines a group of points that has certain interesting semantics for some specific downstream task.

A set of part proposal P S = {P 1 , P 2 , ?? ?? ?? , P S } comprises of several interesting part regions on S that are useful for various tasks.

The task of shape part discovery on point clouds is to produce P pred S for each input shape point cloud C S .

Ground-truth proposal set P gt S is a manually labeled set of parts that are useful for some human-defined downstream tasks.

A good algorithm should predict P pred S such that P gt S ??? P pred S within an upper-bound limit of part numbers M .

A category of shapes T = {S 1 , S 2 , ?? ?? ?? } gathers all shapes that belong to one semantic category.

For example, T chair includes all chair 3D models in a dataset.

Zero-shot shape part discovery considers two sets of object categories T train = {T 1 , T 2 , ?? ?? ?? , T u } and T test = {T u+1 , T u+2 , ?? ?? ?? , T v }, where T i ??? T j = ??? for any i = j. For each shape S ??? T ??? T train , a manually labeled part proposal subset P gt S ??? P S is given for algorithms to use.

It provides algorithms an opportunity to develop the concept of parts in the training categories.

No ground-truth part proposals are provided for shapes in testing categories T test .

Algorithms are expected to predict P pred S for any shape S ??? T ??? T test .

Our method starts with proposing a set of small superpixel-like (Ren & Malik, 2003) sub-parts of the given shape.

We refer readers to Appendix A for more details of our sub-part proposing method.

Given a set of sub-parts, our method iteratively groups together the sub-parts belonging to the same parts in ground-truth and produce larger sub-parts, until no sub-part can further group each other.

The remaining sub-parts in the final stage become a pool of part proposals for the input shape.

Our perceptual grouping process is a sequential decision process.

We formulate the perceptual grouping process as a contextual bandit (one-step Markov Decision Process) (Langford & Zhang, 2007) .

In each iteration, we use a policy network to select a pair of sub-parts and send it to the verification network to verify whether we should group the selected pair of sub-parts.

If yes, we group the selected pair of sub-parts into a larger sub-part.

Otherwise, we will not consider this pair in the latter grouping process.

Our policy network is composed of two sub-modules: a purity module and a rectification module.

The purity module inputs unary information and measures how likely two sub-parts belong to the same part in ground-truth after grouping and the rectification module inputs binary information and further decides the pair to select.

We describe more technical network design choices in Section 4.1.

To train the entire pipeline, we borrow the on-policy training scheme from Reinforcement Learning (RL) to train these networks, in order to match the data distribution during training and inference stages, as described in Section 4.2.

Rectification Score Purity Module:

Two sub-parts that belong to the same ground-truth part should group together.

We define purity score U (P ) for a sub-part P as the maximum ratio of the intersection of P with the ground-truth parts {P gt i }.

More formally,

where p enumerates all points in the shape point cloud and I is the indicator function.

We train a purity module to predict the purity score.

It employs a PointNet that takes as input a merged sub-part P ij = P i ??? P j and predicts the purity score.

Figure 2 (a) shows the architecture.

Rectification Module: We observe that a purity module is not enough to fully select the best pair of sub-parts to group in practice.

For example, when a large sub-part tries to group with a small one from a different ground-truth part, the part geometry of the grouping outcome is primarily dominated by the large sub-part, and thus the purity module tends to produce a high purity score, which results in selecting a pair that should not be grouped.

To address this issue, we consider learning a rectification module to correct the failure case given by the purity module.

We design the rectification module as in Figure 2 (b).

The rectification module takes two sub-parts as inputs, extracts features using a shared PointNet, concatenates the two part features and outputs a real-valued rectification score R(P ), based purely on local information.

Different from the purity module that takes the grouped subpart as input, the rectification module explicitly takes two subparts as inputs in order to compare the two sub-part features for decision making.

Algorithm 1 Sub-Part Pair Selection and Grouping.

Input: A sub-parts pool P = {P i } i???n Input: Purity module U ; Rectification module R; Verification network V 1: for i, j ??? n do 2:

Group two shapes:

Calculate the purity score u i,j ??? U (P ij )

Calculate the rectification score r ij ??? R(P i , P j ) 5: end for 6: Calculate policy ??(P i , P j ) ??? Sample pair P i , P j ??? ??(P i , P j ) 9: else 10:

Select the P i , P j = arg max ??(P i , P j ) 11: end if 12: if V (P i , P j ) is True then 13:

Delete P i , P j from the pool 14:

Add P ij into the pool 15: end if Policy Network: We define policy score by making the product of purity score and rectification score.

We define the policy ??(P i , P j |P) as a distribution over all possible pairs characterized by a softmax layer as shown in line 6 of Algorithm 1.

The goal of the policy is to maximize the objective maximize

The reward, or the merge-ability score M (P i , P j ) defines whether we could group two sub-parts P i and P j .

To compute the reward M (P i , P j ): we first calculate the instance label of the corresponding ground-truth part for sub-parts P i , P j as l i and l j .

We set M (P i , P j ) to be one if the two sub-parts have the same instance label and the purity scores of two sub-parts are greater than 0.8.

Verification Network: Since the policy scores sum to one overall pairs of sub-parts, there is no explicit signal from the policy network on whether the pair should be grouped.

We train a separate verification network that is specialized to verify whether we should group the selected pair.

Here also exists a cascaded structure where the verification network will focus on the pairs selected by the policy network and make a double verification.

The verification network takes a pair of shape as input and outputs values from zero to one after a Sigmoid layer.

Figure 2 (c) illustrates the network architecture: a PointNet first extracts the part feature for each sub-part, then two sub-part point clouds are augmented with the extracted part features and concatenated together to pass through another PointNet to obtain the final score.

Notice that our design of the verification network is a combination of the purity module and rectification module.

We want to extract both the input sub-part features and the part feature after grouping.

In this section, we illustrate how to train the two networks jointly as an entire pipeline.

We use Reinforcement Learning (RL) on-policy training and borrow the standard RL training techniques, such as epsilon-greedy exploration and replay buffer sampling.

We also discuss the detailed loss designs for training the policy network and the verification network.

RL On-policy Training Borrowing ideas from the field of Reinforcement Learning (RL), we train the policy network and the verification network in an on-policy fashion.

On-policy training alternates between the data sampling step and the network training step.

The data sampling step fixes the network parameters and then runs the inference-time pipeline to collect the grouping trajectories, including all pairs of sub-parts seen during the process and all the grouping operations taken by the pipeline.

The network training step uses the trajectory data collected from the data sampling step to compute losses for different network modules and performs steps of gradient descents to update the network parameters.

We fully describe the on-policy training algorithm in Algorithm 2.

We adapt epsilon-greedy strategy (Mnih et al., 2013) into the training stage.

We start from involving 80% random sampling samples during inference as selected pairs and decay the ratio with 10% step Sample shape S and its ground truth-label gt.

Preprocess S to get a sub-parts pool P = {P i } i???n

while ??? Groupable sub-parts do 6:

Select and group two sub-parts P i , P j with Algorithm 1 7:

Store (P i , P j , P) in B and update sub-part pool P

Sample batch of data (P

Set purity score

Update rectification module with policy gradient:

Update purity module by minimizing the l 2 loss with purity score U k gt :

Update verification network by minimizing the cross entropy loss :

end while 15: end while size in each epoch.

We find that random actions not only improve the exploration in the action space and but also serve as the data-augmentation role.

The random actions collect more samples to train the networks, which improves the transfer performance in unseen categories.

Also, purely on-policy training would drop all experience but only use the data sampled by current policy.

This is not data efficient, so we borrow the idea from DQN (Mnih et al., 2013) and use the replay buffer to store and utilize the experience.

The replay buffer stores all the states and actions during the inference stage.

When updating the policy networks, we sample a batch of transitions, i.e. , the grouped sub-parts, and the sub-part pools when the algorithm groups the sub-parts from the replay buffer.

The batch data is used to compute losses and gradients to update the two networks.

Training Losses As shown in Algorithm 2, to train the networks, we sample a batch of data (P k i , P k j , P k ) k???N from the replay buffer, where P k i , P k j is the grouped pair and P k is the corresponding sub-parts pool.

We first calculate the reward M k gt and ground-truth purity score U k gt for each data in the batch.

For updating the rectification module, we fix the purity module and calculate the policy gradient (Sutton et al., 2000) of the policy network with the reward M k gt shown in line 11.

As the rectification module is a part of the policy network, the gradient will update the rectification module by backpropagation.

We then use the l 2 loss in line 12 to train the purity module and use the cross entropy loss in line 13 to train the verification network.

In this section, we conduct quantitative evaluations of our proposed framework and present extensive comparisons to four previous state-of-the-art shape segmentation methods using PartNet dataset (Mo et al., 2019c) in zero-shot part discovery setting.

We also show a diagnostic analysis of how the discovered part knowledge transfers across different object categories.

We use the recently proposed PartNet dataset (Mo et al., 2019c) as the main testbed.

PartNet provides fine-grained, hierarchical and instance-level part annotations for 26,671 3D models from 24 object categories.

PartNet defines up to three levels of non-overlapping part segmentation for each object category, from coarse-grained parts (e.g. chair back, chair base) to fine-grained ones (e.g. chair back vertical bar, swivel chair wheel).

Unless otherwise noticed, we use 3 categories (i.e. Chair, Lamp, and Storage Furniture) 1 for training and take the rest 21 categories as unseen categories for testing.

In zero-shot part discovery setting, we aim to propose parts that are useful under various different use cases.

PartNet provides multi-level human-defined semantic parts that can serve as a sub-sampled pool of interesting parts.

Thus, we adopt Mean Recall (Hosang et al., 2015; Sung et al., 2018) as the evaluation metric to measure how the predicted part pool covers the PartNet-defined parts.

To elaborate on the calculation of Mean Recall, we first define R t as the fraction of ground-truth parts that have Intersection-over-Union (IoU) over t with any predicted part.

Mean Recall is then defined as the average values of R t 's where t varies from 0.5 to 0.95 with 0.05 as a step size.

We compare our approach to four previous state-of-the-art methods as follows:

??? PartNet-InsSeg: Mo et al. (2019c) proposed a part instance segmentation network that employs a PointNet++ (Qi et al., 2017b) as the backbone that takes as input the whole shape point cloud and directly predicts multiple part instance masks.

The method is a top-down label-prediction method that uses the global shape information. (2014) is a non-learning based method based on the convexity assumption of parts.

The method leverages hand-engineered heuristics relying on local statistics to segment shapes, thus is more agnostic to the object categories.

All the three deep learning-based methods take advantage of the global shape context to achieve state-of-the-art shape part segmentation results on PartNet.

However, these networks are prone to over-fitting to training categories and have a hard time transferring part knowledge to unseen categories.

WCSeg, as a non-learning based method, demonstrates good generalization capability to unseen categories, but is limited by the part convexity assumption.

We compare our proposed framework to the four baseline methods under the Mean Recall metric.

For PartNet-InsSeg, SGPN, GSPN and our method, we train three networks corresponding to three levels of segmentation for training categories (e.g. Chair, Lamp, and Storage Furniture).

We remove the part semantics prediction branch from the three baseline methods, as semantics are not transferable to novel testing categories.

For WCSeg, point normals are required by the routine to check local patch continuity.

PartNet experiments (Mo et al., 2019c) usually assume no such point normals as inputs.

Thus, we approximately compute normals based on the input point clouds by reconstructing surface with ball pivoting (Bernardini et al., 1999) .

Then, to obtain three-levels of part proposals for WCSeg, we manually tune hyperparameters in the procedure at each level of part annotations on training categories to have the best performance on three seen categories.

Since the segmentation levels for different categories may not share consistent part granularity (e.g. display level-2 parts may correspond to chair level-3 parts), we gather together the part proposals generated by methods at all three levels as a joint pool of proposals for evaluation on levels of unseen categories.

For the proposed method, we involve limited context only on seen categories as presented in Appendix B.1.

We present quantitative and qualitative evaluations to baseline methods in Table 1 , Figure 3 and Appendix C. For each testing category, we report the average values of Mean Recall scores at all levels.

See the appendix Table 6 for detailed numbers at all levels.

We observe that our approach achieves the best performance on average among all testing novel categories, while championing 11 out of 20 categories.

The core of our method is to learn local-context part knowledge from training categories that is able to transfer to novel unseen categories.

Such learned part knowledge may also include non-transferable category-specific information, such as the part geometry and the part boundary types.

Training our framework on more various object categories is beneficial to learn more generalizable knowledge that shares in common.

However, due to the difficulties in acquiring human annotated fine-grained parts (e.g. PartNet (Mo et al., 2019c )), we can often conduct training on a few training categories.

Thus, we are interested to know how to select categories to achieve the best performance in all categories.

Different object categories have different part patterns that block part knowledge transfers across category boundaries.

However, presumably, similar categories, such as tables and chairs, often share common part patterns that are easier to transfer.

For example, tables and chairs are both composed of legs, surfaces, bar stretchers and wheels, which offers a good opportunity for transferring local-context part knowledge.

We analyze the capability of transferring part knowledge across category boundaries under our framework.

Table 2 presents experimental results of doing cross-validation using chairs, tables and lamps by training on one category and testing on another.

We observe that, chairs and tables transfer part knowledge to each other as expected, while the network trained on lamps demonstrates much worse performance on generalizing to chairs and tables.

In this paper, we introduced a data-driven iterative perceptual grouping pipeline for the task of zero-shot 3D shape part discovery.

At the core of our method is to learn part-level features within part local contexts, in order to generalize the part discovery process to unseen novel categories.

We conducted extensive evaluation and analysis of our method and presented thorough quantitative comparisons to four state-of-the-art shape segmentation algorithms.

We demonstrated that our method successfully extracts locally-aware part knowledge from training categories and transfers the knowledge to unseen novel categories.

Our method achieved the best performance over all four baseline methods on the PartNet dataset.

A SUB-PART PROPOSAL MODULE Given a shape represented as a point cloud, we first propose a pool of small superpixel-like (Ren & Malik, 2003) sub-parts as the building blocks.

We employ furthest point sampling to sample 128 seed points on each input shape.

To capture the local part context, we extract PointNet (Qi et al., 2017a) features with 64 points sampling within a local 0.04-radius 2 neighborhood around each seed point.

In the training phase, all the 64 points will be sampled from the same instance.

Then, we train a local PointNet segmentation network that takes as inputs 512 points within a 0.2-radius ball around every seed point and output a binary segmentation mask indicating a sub-part proposal.

If the point belongs to the instance is the same as the 0.04-radius ball, it will be classified into 1.

We call this module as the sub-part proposal module and illustrate it in Figure 4 .

In the inference phase, we can not guarantee the 64 points sampled within a 0.04-radius ball are all coming from the same part.

However, in our experiments, we observe those sub-part proposals will have a low purity score due to the poor center feature extracted from the 64 points across different parts.

Also, even the center feature extraction is good, some sub-parts may also cover multiple parts in ground-truth.

To obtain high-quality sub-parts, we remove the sub-parts whose purity score lower than 0.8, and the remain sub-parts form our initial sub-part pool.

The input of this learning module is constrained in a local region, thus will not be affected by the global context.

To validate the transferring performance of this module, we train the module on Chair, Storage Furniture, and Lamp of level-3 annotations and test on all categories with evaluating by the most fine-grained level annotations of each category.

The results are listed in Table 3 .

Since the part patterns in Table 3 : Quantitative evaluation of the sub-part proposal module.

PosAcc and NegAcc refer to positive accuracy and negative accuracy of the binary segmentation.

In this section, we conduct experiments about the effects of involving more context for training models on seen categories and unseen categories.

We add a branch to the verification network and extend it into Figure 5 .

This branch takes all the sub-parts where the minimum distance with the input sub-part pair 2 ??? 0.01 as input and thus encodes more context information for decision making.

Note that the involved context is still restricted in a very local region and the module can not "see" the whole shape.

Now, there are two branches can be used to determine whether we should group the pair.

The original binary branch is driven by purely local contextual information.

The newly added one encodes more context information.

To test the effectiveness of involving more context, we train the model with the extended verification network on Chair, Lamp, and Storage Furniture and test them in two ways.

1) We make decisions by only using the original binary branch.

2) We make decisions by using the original binary branch when the size of the sub-part pool is ??? 32 and using the newly added branch when the size of the sub-part pool is ??? 32.

We choose 32 as an empirical threshold here based on our observation that when the size of the pool is ??? 32, the sub-parts in the pool will be relative large and the additional local context will help make grouping decisions in most scenarios.

From the results listed in Table 4 , we can point out that the involved context helps to consistently improve the performance on seen categories, but has negative effects on most unseen categories.

When the context is similar between the seen categories and unseen categories, such as patterns between Storage Furniture and Bed, the involved context can help make decisions.

The phenomenon indicates a future direction that we train the model with involving more context but use them on unseen categories only when we found a similar context we have seen during training.

It also enables the proposed method to achieve higher performance on seen categories without degrading the performance on unseen categories by involving contexts only when testing on seen categories and discarding it for unseen categories.

We adopt this way for obtaining final scores.

Table 4 : Quantitative evaluation of involving more context.

w/ and w/o denote making decision with and without involving more context, respectively.

Note that we only introduce more context in the late grouping process and the involved context is restricted in a very local region.

The number is the mean recall of segmentation results.

The L1, L2 and L3 refer to the three levels of segmentation defined in PartNet.

Avg is the average among mean recall of three levels segmentation results.

In our pipeline, we use the policy network to learn to pick pairs of sub-parts and adopt on-policy training for boosting the performance.

For the policy network, it consists of the purity module and the rectification module, which process the unary information and binary information respectively.

Here, we show the quantitative results and validate the effectiveness of these components.

The results are listed in Table 5 , where we train the model on the Chair of level-3 annotations.

??? Purity Module: The purity module takes the unary information (a grouped pair) as input and output the purity score.

Similar to the objectness score Alexe et al. (2012) used in object detection, the purity score serves as the partness score to measure the quality of the proposal.

We optimize the purity module by regressing the ground-truth purity scores and use such meaningful supervision to help learn the policy.

The results of "no purity" row in Table 5 show the effectiveness of this module.

??? Rectification Module: The rectification module is involved to rectify the failure cases for the purity network.

Our experiments shows that without the rectification module, our decision process will easily converge to a trajectory that a pair of sub-part with unbalanced size will usually be chosen to group results in situations that one huge sub-part dominate the sub-part pool and bring in performance drop as shown in Table 5 , the "no rectification" row.

Please also refer to Appendix B.3 to see some relating qualitative results.

??? On-Policy Training: The on-policy training will sample training data that matches the inference process without the requirement of carefully designing the sampling strategy.

Without it, our networks suffer from a decrease in performance as shown in Table 5 , the "off-policy" row.

Table 5 : Quantitative results of ablation studies.

We train the models on the Chair of level-3 annotations and test on the listed categories.

The number is the mean recall of the most fine-grained annotations of each category.

We involve the rectification module and learn the policy to pick pairs of sub-parts for grouping.

The rectification module may bring several benefits.

Here we demonstrate the effectiveness of the rectification module from one aspect that this module will encourage to pick equal size pairs of sub-parts.

In our experiments, we found if we only follow the guidance of the purity network, our policy will tend to choose the pair comprising one big sub-part and one small sub-part.

Like the descriptions in Section 4.1, the geometry of such unequal size pairs will be dominated by the big sub-part and thus raise the possibilities of errors.

The rectification module can alleviate this situation and encourage the learned policy to choose more equal size pairs.

To evaluate this point, we define the relative size for the selected pairs.

Given a pair of sub-parts P i and P j , we define the relative size as

Pi where the smaller value means the size of the pair P i and P j are more equal, and the minimum value is 2.

We train two models with and without the rectification module separately on Chair of level-3 annotations and test on Chair.

We plot the relative size for the process of grouping and show some randomly sampled results in Figure 6 .

Every picture shows the grouping process for one shape, the x-axis is the iteration number of the process, the y-axis is the defined metric.

From the results, we can clearly see the rectification module helps to choose more equal size pairs.

Only when it comes to the late stage, where the size of parts is various and it is hard to find size equal pairs, our policy will pick size unequal pairs.

Therefore, the rectification module helps to prevent the trajectory converging to catastrophic cases in which larger sub-parts dominate the feature for purity score prediction and fail to predict the purity for the grouped sub-parts.

Also, intuitively, the intermediate sub-parts generated during the grouping process may have various patterns and are irregular.

This increases the burden of models to recognize, and the learned "equal-size selection" like rule may help to form regular intermediate sub-parts and alleviate this issue.

We will train our model on three categories (Chair, Lamp, and Storage Furniture) and test on all categories.

For each method, we train three models corresponding to three levels of segmentation annotations for training categories.

All the point clouds of shapes used in our experiments have 10000 points.

For the compared baselines, the input is the whole shape point cloud, where the size is 10000.

For the proposed method, the input is the points sampling from a sub-part, where the size is 1024.

Therefore, the proposed method has advantages in GPU memory cost.

Since our task does not need semantic labels, we remove the semantic segmentation loss for all the deep learning methods.

??? PartNet-InsSeg:

We follow the default settings and hyper-parameters described in the paper where the output instance number is 200, and loss weights are 1 except 0.1 for the regularization term.

We train the model for a total of 120 epochs with the learning rate starting from 0.001 and decaying 0.5 every 15 epochs.

The whole training process will take 4 days on a single 1080Ti GPU.

The model has a total of 1.93 ?? 10 6 parameters.

??? SGPN: Following the same experiment setting in Mo et al. (2019c) , where the max group number is set to 200.

The learning rate is 0.0001 initially and decays by a factor of 0.8 every 40000 iterations.

The model is trained on one 1080Ti GPU for 50 epochs and has a total of 1.55 ?? 10 6 parameters.

It takes 3 days to train the network and 20 seconds to process each shape in the inference phase.

??? GSPN: The maximum number of detection per shape is 200.

The number of points of each generated instance is set to 512.

NMS ( Non-Maximum Suppression) of threshold 0.8 is applied after proposal generation.

As in Yi et al. (2019b) , we train GSPN first for 20 epochs and then fine-tune it jointly with R-PointNet for another 20 epochs.

The learning rate is initialized to 0.001 and decays by a factor of 0.7 every 10000 steps.

Each stage of training takes 2 days respectively.

The model has a total of 14.80 ?? 10 6 parameters where the Shape Proposal Net has 13.86 ?? 10 6 parameters.

??? WCSeg: This method requires out-ward normal as input which is lacked in the point clouds.

In order to perform this method and compare it as fair as possible, we first generate out-ward normals for input point clouds as follows: a) we employ ball-pivoting Bernardini et al. (1999) to reconstruct surface for input point clouds.

b) we keep one face fixed and re-orient the faces order coherently so that the generated face normal is all out-ward.

c) we transfer the face normals back to the vertices' normals.

As a traditional method, the performance is very sensitive to all hyper-parameters, we tune four parameters recommended in the paper by grid searching on seen categories and then test on unseen categories.

More specifically, we randomly select 100 object instances for each of the three seen categories (i.e. Chair, Lamp, and Storage Furniture) as our grid search dataset.

Then we conduct a grid search on the 300 instances regarding the parameters (?? 1 , ?? 2 , ?? 3 , ??) in WCSeg.

Based on the recommended parameters from the original paper Kaick et al. (2014) , we apply relative shifts with the range of [???20%, +20%] on each parameter to form 3 4 = 81 sets of parameters.

Among these parameters, we choose the set with the highest mean recall on fore-mentioned grid search dataset as the parameter for each level.

We eventually select (?? 1 = 0.950, ?? 2 = 0.704, ?? 3 = 0.403, ?? = 0.069) for Level-1, (?? 1 = 1.426, ?? 2 = 0.845, ?? 3 = 0.504, ?? = 0.086) for Level-2, and (?? 1 = 1.188, ?? 2 = 0.563, ?? 3 = 0.504, ?? = 0.069) for Level-3 in our experiments.

We use the MATLAB code provided by the paper and perform it on our Intel(R) Xeon(R) Gold 6126 CPU cluster with 16 CPU cores used.

For the inference, WCSeg takes about 2.2 minutes to process each shape per CPU core and about 4 days to finish testing over PartNet's part instance segmentation dataset.

??? Our:

For each shape, we first use the sub-part proposal module to generate sub-part proposals as described in Appendix A. For training the proposal module, we use batch size 12 and learning rate starting from 0.001 and decaying 0.5 every 15 epochs for a total of 120 epochs.

After this, we will gain 128 proposals and then remove the proposals whose purity score is lower than 0.8.

The rest proposals form our initial sub-parts pool.

In the training phase, we can calculate the ground-truth purity score according to the annotations.

In the inference phase, we will use the trained purity module to predict the purity score.

To short the whole training time, we train the proposal module on level-3 annotations and use it for training the policy network and verification network on all three levels.

After gaining the initial sub-part pool, we begin our grouping process.

During the process, we will use the policy network comprising the purity module and the rectification module to pick the pair of sub-parts and use the verification network to determine whether we should group the pair.

If a pair of sub-parts should be grouped, we will add the new grouped part into the sub-parts pool and remove the input two sub-parts from the pool.

We will iteratively group the sub-parts until no more available pairs.

So for each shape, we generate a trajectory and collect training data from the trajectory.

In the training phase, for each iteration, we sample 64 pairs and calculate the policy score by using both the purity module and the rectification module.

To accelerate the training process, we sample 10 pairs not 1 pair from the 64 pairs and send them to the verification network to determine whether we should group the pair of sub-parts.

The 10 pairs comprise rank n pairs and 10 ??? n random sampling pairs since we adapt epsilon-greedy strategy where start from involving 8 random sampling samples and decay the number with 1 in each epoch.

The minimal number of random sampling pairs is 1.

In the inference phase, for each iteration, we will calculate the policy score for all pairs and send the pair with the highest score to the verification network.

Note that we use the prediction of the verification network not the ground-truth annotations to determine whether we should group and generate the trajectory.

According to ablation studies in Appendix B.2, this on-policy manual is important for the final performance.

For the level-3 model, we choose the pairs where two sub-parts are close to each other within the minimum distance 2 ??? 0.01.

For the level-1 and level-2 model, we will first choose the neighboring pairs and group them.

When all neighboring pairs have been processed, we remove the neighboring constraints and group the pairs until no more available pairs.

We collect the training data from the trajectories and form the replay buffer for each module, where each replay buffer only can hold up to data from 4 trajectories.

The size of the input point cloud to all three modules is 1024 by sampling from the sub-parts.

In each iteration, we train all the modules on corresponding replay buffers.

We train the modules for a total of 1200 iterations with the learning rate starting from 0.001 and decaying 0.5 every 150 epochs.

The batch size for the purity module, the verification network is 128, for the rectification module is 2.

The whole training process will take 4 days on a single 1080Ti GPU.

For the inference, the method takes about 3 seconds to process each shape.

Our model has a total of 0.64 ?? 10 6 parameters, which is fewer than all compared deep learning methods.

We present the full table including Mean Recall scores at all levels and the performance on seen categories in Table 6 .

We involve more context only on seen categories the same as the way presented in Appendix B.1.

Published as a conference paper at ICLR 2020

We provide qualitative results of GSPN, SGPN, WCSeg, PartNet-InsSeg, and our proposed methods for the zero-shot part discovery.

We train the models on Chair, Lamp, and Storage Furniture of level-3 (the most fine-grained level) of PartNet Dataset and test on the other unseen categories.

We also list the corresponding most fine-grained ground-truth annotations for reference (Some categories may only have the level-1 annotation).

Note that the ground-truth annotation only provides one possible segmentation that satisfies category-specific human-defined semantic meanings.

Table   Table   Table   Vase Table 6 : Quantitative Evaluation.

Algorithm P, S, G, W, O refer to PartNet-InsSeg, SGPN, GSPN, WCSeg and Ours, respectively.

The number 1, 2 and 3 refer to the three levels of segmentation defined in PartNet.

We put short lines for the levels that are not defined.

Avg is the average among mean recall of three levels segmentation results in PartNet.

SAvg and WSAvg are average among seen categories and weighted average among seen categories over shape numbers, respectively.

UAvg and WUAvg are average among unseen categories and weighted average among unseen categories over shape numbers, respectively.

@highlight

A zero-shot segmentation framework for 3D object part segmentation. Model the segmentation as a decision-making process and solve as a contextual bandit problem.

@highlight

A method for segmenting 3D point clouds of objects into component parts, focused on generalizing part groupings to novel object categories unseen during training, that shows strong performance relative to baselines.

@highlight

This paper proposes a method for part segmentation in object point clouds.