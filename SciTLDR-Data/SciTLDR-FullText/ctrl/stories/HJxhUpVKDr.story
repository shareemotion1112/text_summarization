In the context of multi-task learning, neural networks with branched architectures have often been employed to jointly tackle the tasks at hand.

Such ramified networks typically start with a number of shared layers, after which different tasks branch out into their own sequence of layers.

Understandably, as the number of possible network configurations is combinatorially large, deciding what layers to share and where to branch out becomes cumbersome.

Prior works have either relied on ad hoc methods to determine the level of layer sharing, which is suboptimal, or utilized neural architecture search techniques to establish the network design, which is considerably expensive.

In this paper, we go beyond these limitations and propose a principled approach to automatically construct branched multi-task networks, by leveraging the employed tasks' affinities.

Given a specific budget, i.e. number of learnable parameters, the proposed approach generates architectures, in which shallow layers are task-agnostic, whereas deeper ones gradually grow more task-specific.

Extensive experimental analysis across numerous, diverse multi-tasking datasets shows that, for a given budget, our method consistently yields networks with the highest performance, while for a certain performance threshold it requires the least amount of learnable parameters.

Deep neural networks are usually trained to tackle different tasks in isolation.

Humans, in contrast, are remarkably good at solving a multitude of tasks concurrently.

Biological data processing appears to follow a multi-tasking strategy too; instead of separating tasks and solving them in isolation, different processes seem to share the same early processing layers in the brain -see e.g. V1 in macaques (Gur & Snodderly, 2007) .

Drawing inspiration from such observations, deep learning researchers began to develop multi-task networks with branched architectures.

As a whole, multi-task networks (Caruana, 1997) seek to improve generalization and processing efficiency through the joint learning of related tasks.

Compared to the typical learning of separate deep neural networks for each of the individual tasks, multi-task networks come with several advantages.

First, due to their inherent layer sharing (Kokkinos, 2017; Lu et al., 2017; Kendall et al., 2018; Guo et al., 2018; , the resulting memory footprint is typically substantially lower.

Second, as features in the shared layers do not need to be calculated repeatedly for the different tasks, the overall inference speed is often higher (Neven et al., 2017; Lu et al., 2017) .

Finally, multi-task networks may outperform their single-task counterparts (Kendall et al., 2018; Xu et al., 2018; Sener & Koltun, 2018; Maninis et al., 2019) .

Evidently, there is merit in utilizing multi-task networks.

When it comes to designing them, however, a significant challenge is to decide on the layers that need to be shared among tasks.

Assuming a hard parameter sharing setting 1 , the number of possible network configurations grows quickly with the number of tasks.

As a result, a trial-and-error procedure to define the optimal architecture becomes unwieldy.

Resorting to neural architecture search (Elsken et al., 2019) techniques is not a viable option too, as in this case, the layer sharing has to be jointly optimized with the layers types, their connectivity, etc., rendering the problem considerably expensive.

Instead, researchers have recently explored more viable alternatives, like routing (Rosenbaum et al., 2018) , stochastic filter grouping (Bragman et al., 2019) , and feature partitioning (Newell et al., 2019) , which are, however, closer to the soft parameter sharing setting.

Previous works on hard parameter sharing opted for the simple strategy of sharing the initial layers in the network, after which all tasks branch out simultaneously.

The point at which the branching occurs is usually determined ad hoc (Kendall et al., 2018; Guo et al., 2018; Sener & Koltun, 2018) .

This situation hurts performance, as a suboptimal grouping of tasks can lead to the sharing of information between unrelated tasks, known as negative transfer .

In this paper, we go beyond the aforementioned limitations and propose a novel approach to decide on the degree of layer sharing between tasks in order to eliminate the need for manual exploration.

To this end, we base the layer sharing on measurable levels of task affinity or task relatedness: two tasks are strongly related, if their single task models rely on a similar set of features.

Zamir et al. (2018) quantified this property by measuring the performance when solving a task using a variable sets of layers from a model pretrained on a different task.

However, their approach is considerably expensive, as it scales quadratically with the number of tasks.

Recently, Dwivedi & Roig (2019) proposed a more efficient alternative that uses representation similarity analysis (RSA) to obtain a measure of task affinity, by computing correlations between models pretrained on different tasks.

Given a dataset and a number of tasks, our approach uses RSA to assess the task affinity at arbitrary locations in a neural network.

The task affinity scores are then used to construct a branched multitask network in a fully automated manner.

In particular, our task clustering algorithm groups similar tasks together in common branches, and separates dissimilar tasks by assigning them to different branches, thereby reducing the negative transfer between tasks.

Additionally, our method allows to trade network complexity against task similarity.

We provide extensive empirical evaluation of our method, showing its superiority in terms of multi-task performance vs computational resources.

Multi-task learning.

Multi-task learning (MTL) (Caruana, 1997; Ruder, 2017) is associated with the concept of jointly learning multiple tasks under a single model.

This comes with several advantages, as described above.

Early work on MTL often relied on sparsity constraints (Yuan & Lin, 2006; Argyriou et al., 2007; Lounici et al., 2009; Jalali et al., 2010; Liu et al., 2017) to select a small subset of features that could be shared among all tasks.

However, this can lead to negative transfer when not all tasks are related to each other.

A general solution to this problem is to cluster tasks based on prior knowledge about their similarity or relatedness (Evgeniou & Pontil, 2004; Abernethy et al., 2009; Agarwal et al., 2010; Zhou et al., 2011; Kumar & Daume III, 2012) .

In the deep learning era, MTL models can typically be classified as utilizing soft or hard parameter sharing.

In soft parameter sharing, each task is assigned its own set of parameters and a feature sharing mechanism handles the cross-task talk.

Cross-stitch networks softly share their features among tasks, by using a linear combination of the activations found in multiple single task networks.

Sluice networks (Ruder et al., 2019) extend cross-stitch networks and allow to learn the selective sharing of layers, subspaces and skip connections.

In a different vein, multi-task attention networks use an attention mechanism to share a general feature pool amongst task-specific networks.

In general, MTL networks using soft parameter sharing are limited in terms of scalability, as the size of the network tends to grow linearly with the number of tasks.

In hard parameter sharing, the parameter set is divided into shared and task-specific parameters.

MTL models using hard parameter sharing are often based on a generic framework with a shared off-the-shelf encoder, followed by task-specific decoder networks (Neven et al., 2017; Kendall et al., 2018; Chen et al., 2018; Sener & Koltun, 2018) .

Multilinear relationship networks (Long et al., 2017) extend this framework by placing tensor normal priors on the parameter set of the fully connected layers.

Guo et al. (2018) proposed the construction of a hierarchical network, which predicts increasingly difficult tasks at deeper layers.

A limitation of the aforementioned approaches is that the branching points are determined ad hoc, which can easily lead to negative transfer if the predefined task groupings are suboptimal.

In contrast, in our branched multi-task networks, the degree of layer sharing is automatically determined in a principled way, based on task affinities.

Our work bears some similarity to fully-adaptive feature sharing (Lu et al., 2017) , which starts from a thin network where tasks initially share all layers, but the final one, and dynamically grows the model in a greedy layer-by-layer fashion.

Task groupings, in this case, are decided on the probability of concurrently simple or difficult examples across tasks.

Differently, (1) our method clusters tasks based on feature affinity scores, rather than example difficulty, which is arguably a better criterion; (2) the tree structure is determined offline using the precalculated affinities for the whole network, and not online in a greedy layer-by-layer fashion, which promotes task groupings that are optimal in a global, rather than local, sense; (3) our approach achieves significantly better results, especially on challenging datasets featuring numerous tasks, like Taskonomy.

Neural architecture search.

Neural architecture search (NAS) (Elsken et al., 2019) aims to automate the construction of the network architecture.

Different algorithms can be characterized based on their search space, search strategy or performance estimation strategy.

Most existing works on NAS, however, are limited to task-specific models (Zoph & Le, 2017; Liu et al., 2018b; Pham et al., 2018; Liu et al., 2018a; Real et al., 2019) .

This is to be expected as when using NAS for MTL, layer sharing has to be jointly optimized with the layers types, their connectivity, etc., rendering the problem considerably expensive.

To alleviate the heavy computation burden, a recent work implemented an evolutionary architecture search for multi-task networks, while other researchers explored more viable alternatives, like routing (Rosenbaum et al., 2018) , stochastic filter grouping (Bragman et al., 2019) , and feature partitioning (Newell et al., 2019) .

In contrast to traditional NAS, the proposed methods do not build the architecture from scratch, but rather start from a predefined backbone network for which a layer sharing scheme is automatically determined.

Transfer learning.

Transfer learning (Pan et al., 2010 ) makes use of the knowledge obtained when solving one task, and applies it to a different but related task.

Our work is loosely related to transfer learning, as we use it to measure levels of task affinity.

Zamir et al. (2018) provided a taxonomy for task transfer learning to quantify such relationships.

However, their approach scales unfavorably w.r.t.

the number of tasks, and we opted for a more efficient alternative proposed by Dwivedi & Roig (2019) .

The latter uses RSA to obtain a measure of task affinity, by computing correlations between models pretrained on different tasks.

In our method, we use the performance metric from their work to compare the usefulness of different feature sets for solving a particular task.

Loss weighting.

One of the known challenges of jointly learning multiple tasks is properly weighting the loss functions associated with the individual tasks.

Early work (Kendall et al., 2018) used the homoscedastic uncertainty of each task to weigh the losses.

Gradient normalization (Chen et al., 2018) balances the learning of tasks by dynamically adapting the gradient magnitudes in the network.

Dynamic task prioritization (Guo et al., 2018) prioritizes the learning of difficult tasks.

Zhao et al. (2018) observed that two competing tasks can cause the destructive interference of the gradient, and proposed a modulation module to alleviate this problem.

Sener & Koltun (2018) cast multi-task learning as a multi-objective optimization problem, with the overall objective of finding a Pareto optimal solution.

Note that, addressing the loss weighting issue in MTL is out of the scope of this work.

In fact, all our experiments are based on a simple uniform loss weighing scheme.

In this paper, we aim to jointly solve N different tasks T = {t 1 , . . .

, t N } given a computational budget C, i.e. number of parameters.

Consider a backbone architecture: an encoder, consisting of a sequence of shared layers or blocks f l , followed by a decoder with a few task-specific layers.

We assume an appropriate structure for layer sharing to take the shape of a tree.

In particular, the first layers are shared by all tasks, while later layers gradually split off as they show more task-specific behavior.

The proposed method aims to find an effective task grouping for the sharable layers f l of the encoder, i.e. grouping related tasks together in the same branches of the tree.

When two tasks are strongly related, we expect their single-task models to rely on a similar feature set (Zamir et al., 2018) .

Based on this viewpoint, the proposed method derives a task affinity score at various locations in the sharable encoder.

The resulting task affinity scores are used for the automated construction of a branched multi-task network that fits the computational budget C. Fig. 1 illustrates our pipeline, while Algorithm 1 summarizes the whole procedure.

We use RSA to measure the task affinity at D predefined locations in the sharable encoder.

In particular, we calculate the representation dissimilarity matrices (RDM) for the features at D locations using K images, which gives a D × K × K tensor per task. (right) The affinity tensor A is found by calculating the correlation between the RDM matrices, which results in a three-dimensional tensor of size D ×N ×N , with N the number of tasks.

(b) Our pipeline's output is a branched multi-task network, similar to how NAS techniques output sample architectures.

An example branched multi-task network is visualized here.

Algorithm 1 Branched Multi-Task Networks -Task clustering 1: Input:

Tasks T , K images I, a sharable encoder E with D locations where we can branch, a set of task specific decoders Dt and a computational budget C. 2: for t in T do 3:

Train the encoder E and task-specific decoder Dt for task t. 4:

for ti, tj in T and d in locations Task affinity 7: D = 1 − A Task dissimilarity 8: Return: Task-grouping with minimal task dissimilarity that fits within C

As mentioned, we rely on RSA to measure task affinity scores.

This technique has been widely adopted in the field of neuroscience to draw comparisons between behavioral models and brain activity.

Inspired by how Dwivedi & Roig (2019) applied RSA to select tasks for transfer learning, we use the technique to assess the task affinity at predefined locations in the sharable encoder.

Consequently, using the measured levels of task affinity, tasks are assigned in the same or different branches of a branched multi-task network, subject to the computational budget C.

The procedure to calculate the task affinity scores is the following.

As a first step, we train a singletask model for each task t i ∈ T .

The single-task models use an identical encoder E -made of all sharable layers f l -followed by a task-specific decoder D ti .

The decoder contains only taskspecific operations and is assumed to be significantly smaller in size compared to the encoder.

As an example, consider jointly solving a classification and a dense prediction task.

Some fully connected layers followed by a softmax operation are typically needed for the classification task, while an additional decoding step with some upscaling operations is required for the dense prediction task.

Of course, the appropriate loss functions are applied in each case.

Such operations are part of the task-specific decoder D ti .

The different single-task networks are trained under the same conditions.

At the second step, we choose D locations in the sharable encoder E where we calculate a twodimensional task affinity matrix of size N × N .

When concatenated, this results in a threedimensional tensor A of size D × N × N that holds the task affinities at the selected locations.

To calculate these task affinities, we have to compare the representation dissimilarity matrices (RDM) of the single-task networks -trained in the previous step -at the specified D locations.

To do this, a held-out subset of K images is required.

The latter images serve to compare the dissimilarity of their feature representations in the single-task networks for every pair of images.

Specifically, for every task t i , we characterize these learned feature representations at the selected locations by filling a tensor of size D × K × K. This tensor contains the dissimilarity scores 1 − ρ between feature representations, with ρ the Pearson correlation coefficient.

Specifically, RDM d,i,j is found by calculating the dissimilarity score between the features at location d for image i and j.

For a specific location d in the network, the computed RDMs are symmetrical, with a diagonal of zeros.

For every such location, we measure the similarity between the upper or lower triangular part of the RDMs belonging to the different single-task networks.

We use the Spearman's correlation coefficient r s to measure similarity.

When repeated for every pair of tasks, at a specific location d, the result is a symmetrical matrix of size N × N , with a diagonal of ones.

Concatenating over the D locations in the sharable encoder, we end up with the desired task affinity tensor of size D × N × N .

Note that, in contrast to prior work (Lu et al., 2017) , the described method focuses on the features used to solve the single tasks, rather than the examples and how easy or hard they are across tasks, which is arguably a better measure of task affinity.

Given a computational budget C, we need to derive how the layers (or blocks) f l in the sharable encoder E should be shared among the tasks in T .

Each layer f l ∈ E is represented as a node in the tree, i.e. the root node contains the first layer f 0 , and nodes at depth l contain layer(s) f l .

The granularity of the layers f l corresponds to the intervals at which we measure the task affinity in the sharable part of the model, i.e. the D locations.

When the encoder is split into b l branches at depth l, this is equivalent to a node at depth l having b l children.

The task-specific decoders D t can be found in the leaves of the tree.

Fig. 1b shows an example of such a tree using the aforementioned notation.

Each node is responsible for solving a unique subset of tasks.

The branched multi-task network is built with the intention to separate dissimilar tasks by assigning them to separate branches.

To this end, we define the dissimilarity score between two tasks t i and t j at location d as 1 − A d,i,j , with A the task affinity tensor 2 .

The branched multi-task network is found by minimizing the sum of the task dissimilarity scores at every location in the sharable encoder.

In contrast to prior work (Lu et al., 2017) , the task affinity (and dissimilarity) scores are calculated a priori.

This allows us to determine the task clustering offline.

Since the number of tasks is finite, we can enumerate all possible trees that fall within the given computational budget C. Finally, we select the tree that minimizes the task dissimilarity score.

The task dissimilarity score of a tree is defined as C cluster = l C l cluster , where C l cluster is found by averaging the maximum distance between the dissimilarity scores of the elements in every cluster.

The use of the maximum distance encourages the separation of dissimilar tasks.

By taking into account the clustering cost at all depths, the procedure can find a task grouping that is considered optimal in a global sense.

This is in contrast to the greedy approach in (Lu et al., 2017) , which only minimizes the task dissimilarity locally, i.e. at isolated locations in the network.

An exhaustive search becomes intractable when the number of tasks is extremely large.

For such cases, we propose to derive the tree in a top-down manner, starting at the most outer layer.

At every step l, we can perform spectral clustering for each possible number of groups m where 1 ≤ m ≤ b l+1 , with b l+1 the number of branches at layer l + 1.

Before proceeding to the next step, we select the top-n task groupings with minimal cost.

This constrains the number of possible groupings at the next layer.

When we proceed to cluster the tasks at the next layer, we select the top-n groupings from the ones that are still eligible to be constructed.

This beam search is used in CelebA experiments.

In this section, we quantitatively and qualitatively evaluate the proposed method on a number of diverse multi-tasking datasets, that range from real to semi-real data, from few to many tasks, from dense prediction to classification tasks, and so on.

Dataset.

The Cityscapes dataset (Cordts et al., 2016) considers the challenging scenario of urban scene understanding.

The train, validation and test set contain respectively 2975, 500 and 1525 real images, taken by driving a car in Central European cities.

It considers a few dense prediction tasks: semantic segmentation (S), instance segmentation (I) and monocular depth estimation (D).

As in prior works (Kendall et al., 2018; Sener & Koltun, 2018) , we use a ResNet-50 encoder with dilated convolutions, followed by a Pyramid Spatial Pooling (PSP) (He et al., 2015) decoder.

Every input image is rescaled to 512 x 256 pixels.

We reuse the approach from Kendall et al. (2018) for the instance segmentation task, i.e. we consider the proxy task of regressing each pixel to the center of the instance it belongs to.

We obtained all results after a grid search on the hyperparameter space, to ensure a fair comparison across the compared approaches.

For more details please visit Appendix A.

Results.

We measure the task affinity after every block (1 to 4) in the ResNet-50 model (see Fig. 2a ).

The task affinity decreases in the deeper layers of the model, due to the features becoming more taskspecific.

We compare the performance of the task groupings generated by our method with those by other approaches.

As in (Maninis et al., 2019) , the performance of a multi-task model m is defined as the average per-task performance drop/increase w.r.t.

a single-task baseline b.

We trained all possible task groupings that can be derived from branching the model in the last three ResNet blocks.

Fig. 2b visualizes performance vs number of parameters for the trained architectures.

Depending on the available computational budget C, our method generates a specific task grouping.

We visualize these generated groupings as a path in Fig. 2b , when gradually increasing the computational budget C. Similarly, we consider the task groupings when branching the model based on the task affinity measure proposed by Lu et al. (2017) .

We find that, in comparison, the task groupings devised by our method achieve higher performance within a given computational budget C. Furthermore, in the majority of cases, for a fixed budget C the proposed method is capable of selecting the best performing task grouping w.r.t.

performance vs parameters metric.

We also compare our branched multi-task networks with cross-stitch networks and NDDR-CNNs in Table 1 3 .

While the latter give higher multi-task performance, attributed to their computationally expensive soft parameter sharing setting, our branched multi-task networks can strike a better trade-off between the performance and number of parameters.

In particular, Fig. 2b shows that we can effectively sample architectures which lie between the extremes of a baseline multi-task model and a cross-stitch or NDDR-CNN architecture.

It is worth noting that soft parameter sharing does not scale when the number of tasks increases greatly.

Dataset.

The Taskonomy dataset (Zamir et al., 2018) contains semi-real images of indoor scenes, annotated for 26 (dense preciction, classification, etc.) tasks.

Out of the available tasks, we select scene categorization (C), semantic segmentation (S), edge detection (E), monocular depth estimation (D) and keypoint detection (K).

The task dictionary was selected to be as diverse as possible, while still keeping the total number of tasks reasonable for all computations.

We use the tiny split of the dataset, containing 275k train, 52k validation and 54k test images.

We reuse the architecture and training setup from Zamir et al. (2018) : the encoder is based on ResNet-50; a 15-layer fullyconvolutional decoder is used for the pixel-to-pixel prediction tasks.

Appendix B contains a more detailed description on the training setup of the Taskonomy experiments.

Results.

The task affinity is again measured after every ResNet block.

Since the number of tasks increased to five, it is very expensive to train all task groupings exhaustively, as done above.

Instead, we limit ourselves to three architectures that are generated when gradually increasing the parameter budget.

As before, we compare our task groupings against the method from Lu et al. (2017) .

The numerical results can be found in Table 2 .

The task groupings themselves are shown in Appendix B.

The effect of the employed task grouping technique can be seen from comparing the performance of our models against the corresponding FA models, generated by (Lu et al., 2017) .

The latter are consistently outperformed by our models.

Compared to the results on Cityscapes (Fig. 2b) , we find that the multi-task performance is much more susceptible to the employed task groupings, possibly due to negative transfer.

Furthermore, we observe that cross-stitch networks and NDDR-CNNS can not handle the larger, more diverse task dictionary: the performance decreases when using these models, while the number of parameters increases.

This is in contrast to our branched multi-task networks, which seem to handle the diverse set of tasks rather positively.

As opposed to (Zamir et al., 2018) , but in accordance with (Maninis et al., 2019) , we show that it is possible to solve many heterogeneous tasks simultaneously when the negative transfer is limited, by separating dissimilar tasks from each other in our case.

In fact, our approach is the first to show such consistent performance across different multi-tasking scenarios and datasets.

Existing approaches seem to be tailored for particular cases, e.g. few/correlated tasks, synthetic-like data, binary classification only tasks, etc., whereas we show stable performance across the board of different experimental setups.

Table 3 : Quantitative analysis on the CelebA test set. (bold) The Ours-Thin-32 architecture is found by optimizing the task clustering for the parameter budget that is used in the Branch-32-2.0 model. (italic) The Ours-Thin-64 architecture is found by optimizing the task clustering for the parameter budget that is used in the GNAS-Shallow-Wide model.

Accuracy (%) Parameters (Millions) LNet+ANet (Wang et al., 2016) 87 -Walk and Learn (Wang et al., 2016) 88 -MOON (Rudd et al., 2016)

90.94 119.73 Independent Group (Hand & Chellappa, 2017)

91.06 -MCNN (Hand & Chellappa, 2017) 91.26 -MCNN-AUX (Hand & Chellappa, 2017) 91.29 -VGG-16 Baseline (Lu et al., 2017) 91.44 134.41 Branch-32-2.0 (Lu et al., 2017)

90.79 2.09 GNAS-Shallow-Thin (Hand & Chellappa, 2017) 91.30 1.57 GNAS-Shallow-Wide (Hand & Chellappa, 2017) 91.63 7.73 GNAS- Deep-Thin (Hand & Chellappa, 2017) 90.90 1.47 GNAS- Deep-Wide (Hand & Chellappa, 2017) 91.36 6.41 ResNet-18 (Uniform weighing) (Sener & Koltun, 2018) 90.38 11.2 ResNet-18 (MGDA-UB) (Sener & Koltun, 2018) 91

Dataset.

The CelebA dataset (Liu et al., 2015) contains over 200k real images of celebrities, labeled with 40 facial attribute categories.

The training, validation and test set contain 160k, 20k and 20k images respectively.

We treat the prediction of each facial attribute as a single binary classification task, as in (Lu et al., 2017; Sener & Koltun, 2018; .

To ensure a fair comparison: we reuse the thin-ω model from Lu et al. (2017) in our experiments on CelebA; the parameter budget C is set for the model to have the same amount of parameters as prior work.

As mentioned in Sec. 3.2, we use the beam search adaptation of our optimization procedure due to the very large number of tasks.

We set n = 10, with the top-n being the number of groupings to keep at every layer during the optimization.

Note that, the final result remains unchanged when applying small changes to n. Appendix C provides more details on the training setup.

Results.

Table 3 shows the results on the CelebA test set.

The task groupings themselves are visualized in Appendix C. Our branched multi-task networks outperform earlier works (Lu et al., 2017; when using a similar amount of parameters.

Since our Thin-32 model only differs from the model in (Lu et al., 2017) on the employed task grouping technique, we can conclude that the proposed method devises more effective task groupings for the attribute classification tasks on CelebA. Furthermore, our Thin-32 model performs on par with the VGG-16 baseline, while using 64 times less parameters.

We also compare our results with the ResNet-18 model from Sener & Koltun (2018) .

Our Thin-64 models performs 1.35% better than the ResNet-18 model when trained with a uniform loss weighing scheme.

More noticeably, our Thin-64 model performs on par with the state-of-the-art ResNet-18 model that was trained with the loss weighing scheme from Sener & Koltun (2018) , while at the same time using 31% less parameters (11.2 vs 7.7 M).

In this paper, we introduced a principled approach to automatically construct branched multi-task networks for a given computational budget.

To this end, we leverage the employed tasks' affinities as a quantifiable measure for layer sharing.

The proposed approach can be seen as an abstraction of NAS for MTL, where only layer sharing is optimized, without having to jointly optimize the layers types, their connectivity, etc., as done in traditional NAS, which would render the problem considerably expensive.

Extensive experimental analysis shows that our method outperforms existing ones w.r.t.

the important metric of multi-tasking performance vs number of parameters, while at the same time showing consistent results across a diverse set of multi-tasking scenarios and datasets.

MTAN We tried re-implementing the MTAN model ) using a ResNet-50 backbone.

The architecture was based on the Wide-ResNet architecture that is used in the original paper.

After extensive hyperparameter tuning, we were unable to get a meaningful result on the Cityscapes dataset when trying to solve all three tasks jointly.

Note that, the authors have only shown results in their paper when training semantic segmentation and monocular depth estimation.

We reuse the setup from Zamir et al. (2018) .

All input images were rescaled to 256 x 256 pixels.

We use a ResNet-50 encoder and replace the last stride 2 convolution by a stride 1 convolution.

A 15-layer fully-convolutional decoder is used for the pixel-to-pixel prediction tasks.

The decoder is composed of five convolutional layers followed by alternating convolutional and transposed convolutional layers.

We use ReLU as non-linearity.

Batch normalization is included in every layer except for the output layer.

We use Kaiming He's initialization for both encoder and decoder.

We use an L1 loss for the depth (D), edge detection (E) and keypoint detection (K) tasks.

The scene categorization task is learned with a KL-divergence loss.

We report performance on the scene categorization task by measuring the overlap in top-5 classes between the predictions and ground truth.

The multi-task models were optimized with task weights w s = 1, w d = 1, w k = 10, w e = 10 and w c = 1.

Notice that the heatmaps were linearly rescaled to lie between 0 and 1.

During training we normalize the depth map by the standard deviation.

Single-task models We use an Adam optimizer with initial learning rate 1e-4.

The learning rate is decayed by a factor of10 after 80000 iterations.

We train the model for 120000 iterations.

The batch size is set to 32.

No additional data augmentation is applied.

The weight decay term is set to 1e-4.

Baseline multi-task model We use the same optimization procedure as for the single-task models.

The multi-task performance is calculated using Eq. 1.

Branched multi-task models We use the same optimization procedure as for the single-task models.

The architectures that were generated by our method are shown in Fig. 4 .

Fig. 5 shows the architectures that are found when using the task grouping method from Lu et al. (2017) .

We show some of the predictions made by our third branched multi-task network in figure 6 for the purpose of qualitative evaluation.

Figure 4: Task groupings generated by our method.

The numerical results can be found in Table 2 .

Figure 5: Task groupings generated using the method from Lu et al. (2017) .

The numerical results can be found in Table 2 .

Cross-stitch networks / NDDR-CNN We reuse the hyperparameter settings that were found optimal on Cityscapes.

We reuse the thin-ω model from Lu et al. (2017) .

The CNN architecture is based on the VGG-16 model (Simonyan & Zisserman, 2015) .

The number of convolutional features is set to the minimum between ω and the width of the corresponding layer in the VGG-16 model.

The fully connected layers contain 2 · ω features.

We train the branched multi-task network using stochastic gradient descent with momentum 0.9 and initial learning rate 0.05.

We use batches of size 32 and weight decay 0.0001.

The model is trained for 120000 iterations and the learning rate divided by 10 every 40000 iterations.

The loss function is a sigmoid cross-entropy loss with uniform weighing scheme.

<|TLDR|>

@highlight

A method for the automated construction of branched multi-task networks with strong experimental evaluation on diverse multi-tasking datasets.

@highlight

This paper proposes a novel soft parameter sharing Multi-task Learning framework based on a tree-like structure.

@highlight

This paper presents a method to infer multi-task networks architecture to determine which part of the network should be shared among different tasks.