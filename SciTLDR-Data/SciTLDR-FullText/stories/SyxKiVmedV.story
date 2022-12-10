We introduce an attention mechanism to improve feature extraction for deep active learning (AL) in the semi-supervised setting.

The proposed attention mechanism is based on recent methods to visually explain predictions made by DNNs.

We apply the proposed explanation-based attention to MNIST and SVHN classification.

The conducted experiments show accuracy improvements for the original and class-imbalanced datasets with the same number of training examples and faster long-tail convergence compared to uncertainty-based methods.

Deep active learning (AL) minimizes the number of expensive annotations needed to train DNNs by selecting a subset of relevant data points from a large unlabeled dataset BID7 .

This subset is annotated and added to the training dataset in a single pool of data points or, more often, in an iterative fashion.

The goal is to maximize prediction accuracy while minimizing the product of pool size × number of iterations.

A proxy for this goal could be the task of matching feature distributions between the validation and the AL-selected training datasets.

In density-based AL approaches, data selection is typically performed using a simple L 2 -distance metric BID10 .

The image retrieval field BID17 has advanced much further in this area.

For example, recent state-of-the-art image retrieval systems are based on DNNbased feature extraction BID0 with attention mechanisms BID8 .

The latter estimates an attention mask to weight importance of the extracted features and it is trained along with the feature extraction.

Inspired by this, we employ image retrieval techniques and propose a novel attention mechanism for deep AL.

Unlike supervised self-attention in BID8 BID14 , our attention mechanism is not trained with the model.

It relies on recent methods to generate visual explanations and to attribute feature importance values BID13 .

We show the effectiveness of such explanation-based attention (EBA) mechanism for AL when combined with multi-scale feature extraction on a number of image classification datasets.

We also conduct experiments for distorted class-imbalanced training data which is a more realistic assumption for unlabeled data.

AL is a well-studied approach to decrease annotation costs in a traditional machine learning pipelines BID11 .

Recently, AL has been applied to data-demanding DNN-based systems in semi-supervised or weakly-supervised settings.

Though AL is an attractive direction, existing methods struggle to deal with high-dimensional data e.g. images.

We believe this is related to the lack of class and instance-level feature importance information as well as the inability to capture spatially-localized features.

To overcome these limitations, we are interested in estimating spatiallymultiscale features and using our EBA mechanism to select only the most discriminative features.

BID16 proposed to augment the training dataset by labeling the least confident data points and heuristically pseudo-labeling high confidence predictions.

We believe the softmax output is not a reliable proxy for the goals of AL i.e. for selecting images using feature distribution matching between validation and train data.

Unlike BID16 , we use pseudo labels only to estimate EBA vectors and find similarities between discriminative features.

BID2 introduced a measure of uncertainty for approximate Bayesian inference that can be estimated using stochastic forward passes through a DNN with dropout layers.

An acquisition function then selects data points with the highest uncertainty which is measured at the output of softmax using several metrics.

Recent work BID1 extended this method by using an ensemble of networks for uncertainty estimation and achieved superior accuracy.

Sener & Savarese (2018) formulated feature similarity-based selection as a geometric core-set approach which outperforms greedy k-center clustering.

Though their method can complement our approach, we are focusing on the novel feature extraction.

For instance, they employed a simple L 2 distance similarity measure for the activations of the last fully-connected layer.

The most similar work to ours, by BID15 , uses the gradients as a measure of importance for dataset subsampling and analysis.

However, our approach formulates the problem as a multi-scale EBA for AL application and goes beyond a less robust single-step gradient attention.

Other related works are online importance sampling methods BID9 and the influence functions approach in BID4 .

Online importance sampling upweights samples within the mini-batch during supervised training using gradient similarity while influence functions analyze data point importance using computationally challenging second-order gradient information.

Pool-based AL.

Let (X, y) be an input-label pair.

There is a validation dataset {(X v i , y v i )} i∈M of size M and a collection of training pairs {(X i , y i )} i∈N of size N for which, initially, only a small random subset or pool of labels indexed by N 1 is known.

The validation dataset approximates the distribution of test data.

At every bth iteration the AL algorithm selects a pool of P new labels to be annotated and added to existing training pairs which creates a training dataset indexed by N b .A DNN Φ(X, θ) is optimized by minimizing a loss function ( DISPLAYFORM0 to model parameters θ.

However, the actual task is to minimize validation loss expressed by M DISPLAYFORM1 Therefore, an oracle AL algorithm achieves minimum of validation loss using the smallest b × P product.

In this work, we are interested not in finding an oracle acquisition function, but in a method to extract relevant features for such function.

We use a low-complexity greedy k-center algorithm to select the data points in the unlabeled training collection which are most similar to the misclassified entries in the validation dataset.

Feature descriptors.

Let F j i ∈ R C×H×W , where C, H, and W are the number of channels, the height, and the width, respectively be the output of the jth layer of DNN for input image X i .

Then, a feature vector or descriptor of length L can be defined as DISPLAYFORM2 , where function φ(·) is a conventional average pooling operation from BID0 .

In a multi-scale case, descriptor is a concatenation of multiple feature vectors DISPLAYFORM3 A descriptor matrix for the validation dataset V d ∈ R L×M and training dataset S d ∈ R L×N can be calculated using forward passes.

Practically, descriptors can be compressed for storage efficiency reasons using PCA, quantization, etc.

Then, a match kernel BID6 , e.g. cosine similarity, can be used to match features in both datasets.

Assuming that vectors d i are L 2 -normalized, the cosine-similarity matrix is simply DISPLAYFORM4 Explanation-based attention.

Feature maps F i extracted by Φ(X, θ) and pooled by φ(·) contain features that: a) are not class and instance-level discriminative (in other words, not disentangled), b) spatially represent features for a plurality of objects in the input.

We would like to upweight discriminative features that satisfy a) and b) using an attention mechanism.

One approach would be to use self-attention BID14 at the cost of modifying network architecture and intervening into the training process.

Instead, we propose to use EBA that is generated only for feature selection.

The EBA mechanism attributes feature importance values w.r.t.

to the output predictions.

Unlike a visual explanation task, which estimates importance heatmaps in the input (image) space, we propose to estimate feature importance tensors A i of the internal DNN representations F i .

Attention tensors A i can be efficiently calculated using a series of backpropagation passes.

Using one of backpropagation-based methods called integrated gradients (IG) from BID13 , A j i can be estimated as DISPLAYFORM5 where K is the number of steps to approximate the continuous integral by a linear path.

Other forms of (1) are possible: from the simplest saliency method for which K = 1 BID12 to more advanced methods with randomly sampled input features BID3 .Due to lack of labels y i in (1), we use common pseudo-labeling strategy: y i = 1 arg maxŷi .

It is schematically shown in FIG0 .

Unlike BID16 , pseudo-labels are used only to calculate similarity without additional hyperparameters rather than to perform a threshold-selected greedy augmentation.

The EBA A i can be converted to multi-scale attention vector using the same processing a i = φ(A i ) ∈ R L×1 , which, by analogy, forms validation V a ∈ R L×M and train attention matrices S a ∈ R L×N .

The latter processing is implemented in most modern frameworks and, therefore, the complexity to generate A i is only K forward-backward passes.

Summary for the proposed method.

A random subset of N 1 training data points is annotated and a DNN Φ(X, θ) optimized for this subset.

Then, the AL algorithm iteratively (b = 2, 3 . . .)

performs following steps: 1) generates descriptor-attention matrix pairs DISPLAYFORM6 is element-wise product, 3) selects P relevant data points from the remaining subset using acquisition function arg max i∈N\N b−1 (R(X i ), Φ) and 4) retrains Φ(X, θ) using augmented subset N b .

Our method as well as uncertainty-based methods from BID2 are applied to the MNIST and SVHN classification.

We evaluate AL with the original and distorted training data because unlabeled collection of data points cannot be a-priori perfectly selected.

Hence, we introduce a class imbalance which is defined as the ratio of {0 . . .

4} to {5 . . .

9} digits.

The following methods have been employed: random sampling, uncertainty-based (uncert), greedy selection using similarity matching without (top-P:none) and with EBA.

The latter is estimated by saliency (top-P:grad) or IG (top-P:ig).

We rerun experiments 10 times for MNIST and 5 times for SVHN with all-randomized initial parameters.

Mean accuracy and standard deviation are reported.

DNN parameters are trained from scratch initially and after each AL iteration.

Mini-batch size is chosen by cross-validation.

MNIST.

The dataset train/val/test split is 50K/10K/10K.

The LeNet is used with the following hyperparameters: epochs=50, batch-size=25, lr=0.05, lr-decay=0.1 every 15 epochs, uncert methods and IG EBA use K = 128 passes and L is 20 for single-scale (before fc1 layer) and 90 for multiscale descriptors (all layers are concatenated).

Figure 2(a) shows that feature-only matching (top-P:none L20) outperforms random selection by ≈ 1% while EBA (top-P:ig L90) adds another 1% of accuracy when there is no class imbalance.

High class imbalance (Figure 2(c) ) increases that gap: up to 20% for feature-only matching and 25% with EBA.

The highest accuracy is achieved by multi- 90 91 92 93 94 95 96 97 98 99 Top-1 Accuracy, % full random uncert:varMC uncert:entMC top-P:none_L20 top-P:grad_L20 top-P:ig_L20 top-P:ig_L90 top-P:igAbl_L20 top-P:igAbl_L90 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 Fraction of full training dataset, % (b) 75 77 79 81 83 85 87 89 91 93 95 97 Top-1 Accuracy, % full random uncert:varMC uncert:entMC top-P:none_L20 top-P:grad_L20 top-P:ig_L20 top-P:ig_L90 top-P:igAbl_L20 top-P:igAbl_L90 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 Fraction of full training dataset, % (c) 58 61 64 67 70 73 76 79 82 85 88 91 Top-1 Accuracy, % full random uncert:varMC uncert:entMC top-P:none_L20 top-P:grad_L20 top-P:ig_L20 top-P:ig_L90 top-P:igAbl_L20 top-P:igAbl_L90Figure 2: MNIST test dataset accuracy for 3 class imbalance ratios: a) 1 (no imbalance), b) 10 and c) 100.

Total 9 AL iterations (b = 10) are performed each with P = 250 pool size.

Top-1 Accuracy, % full random uncert:entMC uncert:baldMC top-P:none_L256 top-P:grad_L256 top-P:ig_L256 top-P:ig_L384Figure 3: SVHN test dataset accuracy for 3 class imbalance ratios: a) 1 (no imbalance), b) 10 and c) 100.

Total 9 AL iterations (b = 10) are performed each with P = 2, 500 pool size.scale EBA estimated by IG.

EBA-based methods outperform the best uncertainty-based variation ratio (uncert:varMC) approach for all class imbalance settings except the last one where its accuracy is higher by less than 1% when b = 4.

This might be related to small-scale MNIST and pseudo-label noise for EBA.

To study the effects of pseudo-labeling, we plot true-label configurations (marked by "Abl") as well.

The accuracy gap between EBA using true-and pseudo-labels is small with no class imbalance, but much larger (up to 25%) when class imbalance ratio is 100 during first AL iterations.

The dataset train/validation/test split is 500K/104K/26K.

A typical 8-layer CNN is used with the following hyperparameters: epochs=35, batch-size=25, lr=0.1, lr-decay=0.1 every 15 epochs, uncert methods and IG EBA use K = 128 and L is 256 for single-scale (before fc1 layer) and 384 for two-scale descriptors (+ layer before conv7).

Figure 3 shows that the gap between random selection and the best EBA-based AL method grows from 2% to more than 12% when the unlabeled training collection has more class imbalance.

The gap between full training dataset accuracy increases for larger-scale SVHN as well.

This results in even faster convergence for the proposed AL relative to random selection.

Accuracies of the uncert methods are closer to each other than for MNIST, which may signal their declining effectiveness for large-scale data.

The proposed EBA-based methods outperform all uncertainty-based methods for SVHN in the first AL iterations (up to +2.5%) and later arrive at approximately equal results.

We applied recent image retrieval feature-extraction techniques to deep AL and introduced a novel EBA mechanism to improve feature-similarity matching.

First feasibility experiments on MNIST and SVHN datasets showed advantages of EBA to improve density-based AL.

Rather than performing AL for the well-picked training datasets, we also considered more realistic and challenging scenarios with class-imbalanced training collections where the proposed method emphasized the importance of additional feature supervision.

In future research, EBA could be evaluated with other types of data distortions and biases: within-class bias, adversarial examples, etc.

Furthermore, such applications as object detection and image segmentation may benefit more from EBA because multiscale attention can focus on spatially-important features.

@highlight

We introduce an attention mechanism to improve feature extraction for deep active learning (AL) in the semi-supervised setting.