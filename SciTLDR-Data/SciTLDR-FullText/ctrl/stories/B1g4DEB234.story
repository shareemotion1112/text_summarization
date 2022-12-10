Our work presents empirical evidence that layer rotation, i.e. the evolution across training of the cosine distance between each layer's weight vector and its initialization, constitutes an impressively consistent indicator of generalization performance.

Compared to previously studied indicators of generalization, we show that layer rotation has the additional benefit of being easily monitored and controlled, as well as having a network-independent optimum: the training procedures during which all layers' weights reach a cosine distance of 1 from their initialization consistently outperform other configurations -by up to 20% test accuracy.

Finally, our results also suggest that the study of layer rotation can provide a unified framework to explain the impact of weight decay and adaptive gradient methods on generalization.

In order to understand the intriguing generalization properties of deep neural networks highlighted by BID22 BID33 BID15 , the identification of numerical indicators of generalization performance that remain applicable across a diverse set of training settings is critical.

A well-known and extensively studied example of such indicator is the width of the minima the network has converged to BID11 BID15 .In this paper, we present empirical evidence supporting the discovery of a novel indicator of generalization: the evolution across training of the cosine distance between each layer's weight vector and its initialization (denoted by layer rotation).

Indeed, we show across a diverse set of experiments (with varying datasets, networks and training procedures), that larger layer rotations (i.e. larger cosine distance between final and initial weights of each layer) consistently translate into better generalization performance.

In addition to providing an original perspective on generalization, our experiments suggest that layer rotation also BID0 ICTEAM, Université catholique de Louvain, Louvain-LaNeuve, Belgium.

<simon.carbonnelle@uclouvain.be>.benefits from the following properties compared to alternative indicators of generalization:• It is easily monitored and, since it only depends on the evolution of the network's weights, can be controlled along the optimization through appropriate weight update adjustments • It has a network-independent optimum (all layers reaching a cosine distance of 1) •

It provides a unified framework to explain the impact of weight decay and adaptive gradient methods on generalization.

In comparison, other indicators usually provide a metric to optimize (e.g. the wider the minimum, the better) but no clear optimum to be reached (what is the optimal width?), nor a precise methodology to tune it (how to converge to a minimum with a specific width?).

By disclosing simple guidelines to tune layer rotations and an easy-to-use controlling tool, our work can also help practitioners get the best out of their network with minimal hyper-parameter tuning.

The presentation of our experimental study is structured according to three successive steps:1.

Development of tools to monitor and control layer rotation (Section 2); 2.

Systematic study of layer rotation configurations in a controlled setting (Section 3); 3.

Study of layer rotation configurations in standard training settings, with a special focus on SGD, weight decay and adaptive gradient methods (Section 4).Related work is discussed in Supplementary Material.

This section describes the tools for monitoring and controlling layer rotation during training, such as its relation with generalization can be studied in Sections 3 and 4.

Layer rotation is defined as the evolution of the cosine distance between each layer's weight vector and its initialization during training.

More precisely, let w t (t 0 corresponding to initialization), then the rotation of layer l at training step t is defined as the cosine distance between w t0 l and w t l .

BID0 In order to visualize the evolution of layer rotation during training, we record how the cosine distance between each layer's current weight vector and its initialization evolves across training steps.

We denote this visualization tool by layer rotation curves hereafter.

The ability to control layer rotations during training would enable a systematic study of its relation with generalization.

Therefore, we present Layca (LAYer-level Controlled Amount of weight rotation), an algorithm where the layerwise learning rates directly determine the amount of rotation performed by each layer's weight vector during each optimization step (the layer rotation rates), in a direction specified by an optimizer (SGD being the default choice).

Inspired by techniques for optimization on manifolds BID0 , and on spheres in particular, Layca applies layer-wise orthogonal projection and normalization operations on SGD's updates, as detailed in Algorithm 1 in Supplementary Material.

These operations induce the following simple relation between the learning rate ρ l (t) of layer l at training step t and the angle θ l (t) between w t l and w DISPLAYFORM0 Our controlling tool is based on a strong assumption: that controlling the amount of rotation performed during each individual training step (i.e. the layer rotation rate) enables control of the cumulative amount of rotation performed since the start of training (i.e. layer rotation).

This assumption is not trivial since the aggregated rotation is a priori very dependent on the structure of the loss landscape.

As will be attested by the inspection of the layer rotation curves, our assumption however appeared to be sufficiently valid, and the control of layer rotation was effective in our experiments.

Section 2 provides tools to monitor and control layer rotation.

The purpose of this section is to use these tools to conduct a systematic experimental study of layer rotation configurations.

We adopt SGD as default optimizer, but use Layca (cfr.

Algorithm 1) to vary the relative rotation rates (faster rotation for first layers, last layers, or no prioritization) and the global rotation rate value (high or low rate, for all layers).

The experiments are conducted on five different tasks which vary in network architecture and dataset complexity, and are further described in TAB0 .

Layca enables us to specify layer rotation rate configurations by setting the layer-wise learning rates.

To explore the large space of possible layer rotation rate configurations, our study restricts itself to two directions of variation.

First, we vary the initial global learning rate ρ(0), which affects the layer rotation rate of all the layers.

During training, the global learning rate ρ(t) drops following a fixed decay scheme (hence the dependence on t), as is common in the literature (cfr.

Supp.

Mat.

A.6).

The second direction of variation tunes the relative rotation rates between different layers.

More precisely, we apply static, layer-wise learning rate multipliers that exponentially increase/decrease with layer depth (which is typical of exploding/vanishing gradients).

The multipliers are parametrized by the layer index l (in forward pass ordering) and a parameter α ∈ [−1, 1] such that the learning rate of layer l becomes: DISPLAYFORM0 Values of α close to −1 correspond to faster rotation of first layers, 0 corresponds to uniform rotation rates, and values close to 1 to faster rotation of last layers.

Visualization of the layer-wise multipliers for different α values is provided in Supplementary Material.

Figure 1a depicts the layer rotation curves (cfr.

Section 2.1) and the corresponding test accuracies obtained with different layer rotation rate configurations.

While each configuration solves the classification task on the training data (≈ 100% training accuracy in all configurations, cfr.

Supp.

Mat.), we observe huge differences in generalization ability (differences of up to 30% test accuracy).

More importantly, these differences in generalization ability seem to be tightly connected to differences in layer rotations.

In particular, we extract the following rule of thumb that is applicable across the five considered tasks: the larger the layer rotations, the better the generalization performance.

The best performance is consistently obtained when nearly all layers reach the largest possible distance from their initialization: a cosine distance of 1 (cfr. fifth column of FIG0 ).This observation would have limited value if many configurations (amongst which the best one) lead to cosine distances of 1.

However, we notice that most configurations do not.

In particular, rotating the layers weights very slightly is sufficient for the network to achieve 100% training accuracy (cfr.

third column of FIG0 ).

Moreover, one could imagine training procedures with large layer rotations that do not generalize well, e.g. if large rotations are performed in a classification on the training set (100% accuracy).

random direction.

It is indeed necessary that the rotations performed coincide with improvements in the training error.

In particular, configurations with too high layer rotation rates can prevent training from happening, thereby escaping the scope of our rule of thumb (cfr.

Figure 3 ).

Section 3 uses Layca to study the relation between layer rotations and generalization in a controlled setting.

This section investigates the layer rotation configurations that naturally emerge when using SGD, weight decay or adaptive gradient methods for training.

First of all, these experiments will provide supplementary evidence for the rule of thumb proposed in Section 3.

Second, we'll see that studying training methods from the perspective of layer rotation can provide useful insights to explain their behaviour.

The experiments are performed on the five tasks of TAB0 .

The learning rate parameter is tuned independently for each training setting through grid search over 10 logarithmically spaced values (3 −7 , 3 −6 , ..., 3 2 ), except for C10-CNN2 and C100-WRN where learning rates are taken from their original implementations when using SGD + weight decay, and from BID28 when using adaptive gradient methods for training.

The test accuracies obtained in standard settings are compared to the best results obtained with Layca, as provided in the 5th column of FIG0 .

FIG0 (1 st line) depicts the layer rotation curves and the corresponding test accuracies generated by SGD for each of the five tasks.

We observe that the curves are far from the ideal scenario disclosed in Section 3, where the majority of the layers' weights reached a cosine distance of 1 from their initialization.

Moreover, in accordance with our rules of thumb, SGD reaches a considerably lower test performance than Layca.

Extensive tuning of the learning rate did not help SGD to solve its two systematic problems: 1) layer rotations are not uniform and 2) the layers' weights stop rotating before reaching a cosine distance of 1.

Several papers have recently shown that, in batch normalized networks, the regularization effect of weight decay was caused by an increase of the effective learning rate BID27 BID12 BID34 .

More generally, reducing the norm of weights increases the amount of rotation induced by a given training step.

It is thus interesting to see how weight decay affects layer rotations, and if its impact on generalization is coherent with our rule of thumb.

FIG0 nd line) displays, for the 5 tasks, the layer rotation curves generated by SGD when combined with weight decay (in this case, equivalent to L 2 -regularization).

We observe that SGD's problems are solved: all layers' weights are rotated synchronously and reach a cosine distance of 1 from their initialization.

Moreover the observations confirm our rule of thumb: the resulting test performances are on par with the ones obtained with Layca.

The recent years have seen the rise of adaptive gradient methods in the context of machine learning (e.g. RMSProp BID26 , Adagrad BID5 , Adam BID16 ).

Initially introduced for improving training speed, BID28 observed that these methods also had a considerable impact on generalization.

Since these methods affect the rate at which individual parameters change, they might also influence layer rotations.

We will thus verify if their influence on generalization is coherent with our rule of thumb.

FIG0 (1 st line) provides the layer rotation curves and test accuracies obtained when using adaptive gradient methods to train the 5 tasks described in TAB0 .

We observe an overall worse generalization ability compared to Layca's optimal configuration and small and/or non-uniform layer rotations.

We also observe that the layer rotations of adaptive gradient methods are considerably different from the ones induced by SGD (cfr.

FIG0 .

For example, adaptive gradient methods seem to induce larger rotations of the last layers' weights, while SGD usually favors rotation of the first layers' weights.

Could these differences explain the impact of parameter-level adaptivity on generalization in deep learning?

In FIG0 (2 nd line), we show that when Layca is used on top of adaptive methods (to control layer rotation), adaptive methods can reach test accuracies on par with SGD + weight decay.

Our observations thus offer a novel perspective on adaptive gradient methods' poor generalization properties, and provide supplementary evidence for our rule of thumb.

TAB0 .

∆η is computed with respect to Layca's best configuration (last column of (a)).

Colour code and axes are provided in the upper right.

Training accuracies are provided in Supplementary Material (≈ 100% in all configurations).

Overall, the visualizations unveil large differences in generalization ability across configurations which seem to follow a simple yet consistent rule of thumb: the larger the layer rotation for each layer, the better the generalization performance.

<|TLDR|>

@highlight

This paper presents empirical evidence supporting the discovery of an indicator of generalization: the evolution across training of the cosine distance between each layer's weight vector and its initialization.