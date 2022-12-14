We show implicit filter level sparsity manifests in convolutional neural networks (CNNs) which employ Batch Normalization and ReLU activation, and are trained using adaptive gradient descent techniques with L2 regularization or weight decay.

Through an extensive empirical study (Anonymous, 2019) we hypothesize the mechanism be hind the sparsification process.

We find that the interplay  of  various  phenomena  influences  the strength of L2 and weight decay regularizers, leading the supposedly non sparsity inducing regularizers to induce filter sparsity.

In this workshop article we summarize some of our key findings and experiments, and present additional results on modern network architectures such as ResNet-50.

In this article we discuss the findings from BID7 regarding filter level sparsity which emerges in certain types of feedforward convolutional neural networks.

Filter refers to the weights and the nonlinearity associated with a particular feature, acting together as a unit.

We use filter and feature interchangeably throughout the document.

We particularly focus on presenting evidence for the implicit sparsity, our experimentally backed hypotheses regarding the cause of the sparsity, and discuss the possible role such implicit sparsification plays in the adaptive vs vanilla (m)SGD generalization debate.

For implications on neural network speed up, refer to the original paper BID7 .In networks which employ Batch Normalization and ReLU activation, after training, certain filters are observed to not activate for any input.

Importantly, the sparsity emerges in the presence of regularizers such as L2 and weight decay (WD) which are in general understood to be non sparsity inducing, and the sparsity vanishes when regularization is 1 Max Planck Institute For Informatics, Saarbrücken, Germany 2 Saarland Informatics Campus, Germany 3 Ulsan National Institute of Science and Technology, South Korea.

th International Conference on Machine Learning, 2019. removed.

We experimentally observe the following:• The sparsity is much higher when using adaptive flavors of SGD vs. (m)SGD.

The sparsity exists even with leaky ReLU.• Adaptive methods see higher sparsity with L2 regularization than with WD.

No sparsity emerges in the absence of regularization.• In addition to the regularizers, the extent of the emergent sparsity is also influenced by hyperparameters seemingly unrelated to regularization.

The sparsity decreases with increasing mini-batch size, decreasing network size and increasing task difficulty.• The primary hypothesis that we put forward is that selective features 1 see a disproportionately higher amount of regularization than non-selective ones.

This consistently explains how unrelated parameters such as mini-batch size, network size, and task difficulty indirectly impact sparsity by affecting feature selectivity.• A secondary hypothesis to explain the higher sparsity observed with adaptive methods is that Adam (and possibly other) adaptive approaches learn more selective features.

Though threre is evidence of highly selective features with Adam, this requires further study.• Synthetic experiments show that the interaction of L2 regularizer with the update equation in adaptive methods causes stronger regularization than WD.

This can explain the discrepancy in sparsity between L2 and WD.Quantifying Feature Sparsity: Feature sparsity can be measured by per-feature activation and by per-feature scale.

For sparsity by activation, the absolute activations for each feature are max pooled over the entire feature plane.

If the value is less than 10 −12 over the entire training corpus, the feature is inactive.

For sparsity by scale, we consider the scale γ of the learned affine transform in the Batch Norm layer.

We consider a feature inactive if |γ| for the feature is less than 10 −3 .

Explicitly zeroing the features thus marked inactive does not affect the test error, which ensures the validity of our chosen thresholds.

The thresholds chosen are purposefully conservative, and comparable levels of sparsity are observed for a higher feature activation threshold of 10 −4 , and a higher |γ| threshold of 10 −2 .

Figure 1 .

BasicNet: Structure of the basic convolution network studied in this paper.

We refer to the convolution layers as C1-7.

Preliminary Experiments: We use a 7-layer convolutional network with 2 fully connected layers as shown in Figure 1 .

We refer to this network as BasicNet in the rest of the document.

For the basic experiments on CIFAR-10/100, we use a variety of gradient descent approaches, a mini-batch size of 40, with a method specific base learning rate for 250 epochs which is scaled down by 10 for an additional 75 epochs.

The base learning rates and other hyperparameters are as follows: Adam (1e-3, β 1 =0.9, β 2 =0.99, =1e-8), Adadelta (1.0, ρ=0.9, =1e-6), SGD (0.1, mom.=0.9), Adagrad (1e-2), AMSGrad (1e-3), AdaMax (2e-3), RMSProp (1e-3).

We study the effect of varying the amount and type of regularization 2 on the extent of sparsity and test error in TAB0 .

It shows significant convolutional filter sparsity emerges with adaptive gradient descent methods when combined with L2 regularization.

The extent of sparsity is reduced when using Weight Decay instead, and absent entirely in the case of SGD with moderate levels of regularization.

Table 2 shows that using leaky ReLU does not prevent sparsification.

The emergence of sparsity is not an isolated phenomenon specifc to CIFAR-10/100 and BasicNet.

We show in tables 3, 4, and 5 that sparsity manifests in VGG-11/16 ((Simonyan & Zisserman, 2014) ), and ResNet-50 ( BID0 ) on ImageNet and Tiny-ImageNet.

ResNet-50 shows a significantly higher overall filter sparsity than nonresidual VGG networks.

We see in TAB3 , 3, 4, and 5 that decreasing the minibatch size (while maintaining the same number of iterations) leads to increased sparsity across network architectures and datasets.

Table 4 .

Effect of different mini-batch sizes on sparsity (by γ) in VGG-11, trained on ImageNet.

Same network structure employed as (Liu et al., 2017 Norm outputs {x i } N i=1 of a particular convolutional kernel, where N is the size of the training corpus, due to the use of ReLU, a gradient is only seen for those datapoints for whichx i > −β/γ.

Both SGD and Adam (L2: 1e-5) learn positive γs for layer C6, however βs are negative for Adam, while for SGD some of the biases are positive.

This implies that all features learned for Adam (L2: 1e-5) in this layer activate for ≤ half the activations from the training corpus, while SGD has a significant number of features activate for more than half of the training corpus, i.e., Adam learns more selective features in this layer.

Features which activate only for a small subset of the training corpus, and consequently see gradient updates from the main objective less frequently, continue to be acted upon by the regularizer.

If the regularization is strong enough (Adam with L2: 1e-4 in FIG2 ), or the gradient updates infrequent enough (feature too selective), the feature may be pruned away entirely.

The propensity of later layers to learn more selective features with Adam would explain the higher degree of sparsity seen for later layers as compared to SGD.

Understanding the reasons for emergence of higher feature selectivity in Adam than SGD, and verifying if other adaptive gradient descent flavours also exhibit higher feature selectivity remains open for future investigation.

Quantifying Feature Selectivity:

Similar to feature sparsity by activation, we apply max pooling to a feature's absolute activations over the entire feature plane.

For a particular feature, we consider these pooled activations over the entire training corpus to quantify feature selectivity.

See the original paper BID7 for a detailed discussion.

Unlike the selectivity metrics employed in literature BID9 , ours is class agnostic, and provides preliminary quantitative evidence that Adam (and perhaps other adaptive gradient descent methods) learn more selective features than (m)SGD, which consequently see a higher relative degree of regularization.

Interaction of L2 Regularizer with Adam: Next, we consider the role of the L2 regularizer vs. weight decay.

In the original paper we study the behaviour of L2 regularization in the low gradient regime for different optimizers through synthetic experiments and find that coupling of L2 regularization with certain adaptive gradient update equations yields a faster decay than weight decay, or L2 regularization with SGD, even for smaller regularizer values.

This is an additional source of regularization disparity between parameters which see frequent updates and those which don't see frequent updates or see lower magnitude gradients.

It manifests for certain adaptive gradient descent approaches.

Task 'Difficulty' Dependence: As per the hypothesis developed thus far, as the task becomes more difficult, for a given network capacity, we expect the fraction of features pruned to decrease corresponding to a decrease in selectivity of the learned features BID13 .

Since the task difficulty cannot be cleanly decoupled from the number of classes, we devise a synthetic experiment based on grayscale renderings of 30 object classes from ObjectNet3D BID11 .

We construct 2 identical sets of ≈ 50k 64×64 pixel renderings, one with a clean background (BG) and the other with a cluttered BG.

We train BasicNet with a mini-batch size of 40, and see that as expected there is a much higher sparsity (70%) with the clean BG set than with the more difficult cluttered set (57%).

See the original paper BID7 for representative images and a list of the object classes selected.

BID12 Liu et al., 2017) employ explicit filter sparsification heuristics that make use of the learned scale parameter γ in Batch Norm for enforcing sparsity on the filters.

BID12 argue that BatchNorm makes feature importance less susceptible to scaling reparameterization, and the learned scale parameters (γ) can be used as indicators of feature importance.

We thus adopt γ as the criterion for studying implicit feature pruning.

Morcos et al. BID9 suggest based on extensive experimental evaluation that good generalization ability is linked to reduced selectivity of learned features.

They further suggest that individual selective units do not play a strong role in the overall performance on the task as compared to the less selective ones.

They connect the ablation of selective features to the heuristics employed in neural network feature pruning literature which prune features whose removal does not impact the overall accuracy significantly BID8 BID2 .

The findings of Zhou et al. BID13 concur regarding the link between emergence of feature selectivity and poor generalization performance.

They further show that ablation of class specific features does not influence the overall accuracy significantly, however the specific class may suffer significantly.

We show that the emergence of selective features in Adam, and the increased propensity for pruning the said selective features when using L2 regularization may thus be helpful both for better generalization performance and network speedup.

Our findings would help practitioners and theoreticians be aware that seemingly unrelated hyperparameters can inadvertently affect the underlying network capacity, which interplays with both the test accuracy and generalization gap, and could partially explain the practical performance gap between Adam and SGD.

Our work opens up future avenues of theoretical and practical exploration to further validate our hypotheses, and attempt to understand the emergence of feature selectivity in Adam and other adaptive SGD methods.

As for network speed up due to sparsification, the penalization of selective features can be seen as a greedy local search heuristic for filter pruning.

While the extent of implicit filter sparsity is significant, it obviously does not match up with some of the more recent explicit sparsification approaches BID1 BID3 which utilize more expensive model search and advanced heuristics such as filter redundancy.

Future work should reconsider the selective-feature pruning criteria itself, and examine nonselective features as well, which putatively have comparably low discriminative information as selective features and could also be pruned.

These non-selective features are however not captured by greedy local search heuristics because pruning them can have a significant impact on the accuracy.

Though the accuracy can presumably can be recouped after fine-tuning.

@highlight

Filter level sparsity emerges implicitly in CNNs trained with adaptive gradient descent approaches due to various phenomena, and the extent of sparsity can be inadvertently affected by different seemingly unrelated hyperparameters.