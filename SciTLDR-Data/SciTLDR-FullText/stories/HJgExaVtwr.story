Deep neural networks are known to be annotation-hungry.

Numerous efforts have been devoted to reducing the annotation cost when learning with deep networks.

Two prominent directions include learning with noisy labels and semi-supervised learning by exploiting unlabeled data.

In this work, we propose DivideMix, a novel framework for learning with noisy labels by leveraging semi-supervised learning techniques.

In particular, DivideMix models the per-sample loss distribution with a mixture model to dynamically divide the training data into a labeled set with clean samples and an unlabeled set with noisy samples, and trains the model on both the labeled and unlabeled data in a semi-supervised manner.

To avoid confirmation bias, we simultaneously train two diverged networks where each network uses the dataset division from the other network.

During the semi-supervised training phase, we improve the MixMatch strategy by performing label co-refinement and label co-guessing on labeled and unlabeled samples, respectively.

Experiments on multiple benchmark datasets demonstrate substantial improvements over state-of-the-art methods.

Code is available at https://github.com/LiJunnan1992/DivideMix .

The remarkable success in training deep neural networks (DNNs) is largely attributed to the collection of large datasets with human annotated labels.

However, it is extremely expensive and time-consuming to label extensive data with high-quality annotations.

On the other hand, there exist alternative and inexpensive methods for mining large-scale data with labels, such as querying commercial search engines (Li et al., 2017a) , downloading social media images with tags (Mahajan et al., 2018) , leveraging machine-generated labels (Kuznetsova et al., 2018) , or using a single annotator to label each sample (Tanno et al., 2019) .

These alternative methods inevitably yield samples with noisy labels.

A recent study (Zhang et al., 2017) shows that DNNs can easily overfit to noisy labels and results in poor generalization performance.

Existing methods on learning with noisy labels (LNL) primarily take a loss correction approach.

Some methods estimate the noise transition matrix and use it to correct the loss function (Patrini et al., 2017; Goldberger & Ben-Reuven, 2017) .

However, correctly estimating the noise transition matrix is challenging.

Some methods leverage the predictions from DNNs to correct labels and modify the loss accordingly (Reed et al., 2015; Tanaka et al., 2018) .

These methods do not perform well under high noise ratio as the predictions from DNNs would dominate training and cause overfitting.

To overcome this, Arazo et al. (2019) adopt MixUp augmentation.

Another approach selects or reweights samples so that noisy samples contribute less to the loss (Jiang et al., 2018; Ren et al., 2018) .

A challenging issue is to design a reliable criteria to select clean samples.

It has been shown that DNNs tend to learn simple patterns first before fitting label noise (Arpit et al., 2017) .

Therefore, many methods treat samples with small loss as clean ones (Jiang et al., 2018; Arazo et al., 2019) .

Among those methods, Co-teaching (Han et al., 2018) and Co-teaching+ train two networks where each network selects small-loss samples in a mini-batch to train the other.

Another active area of research that also aims to reduce annotation cost is semi-supervised learning (SSL).

In SSL, the training data consists of unlabeled samples in addition to the labeled samples.

Significant progress has been made in leveraging unlabeled samples by enforcing the model to produce low entropy predictions on unlabeled data (Grandvalet & Bengio, 2004) or consistent predictions on perturbed input (Laine & Aila, 2017; Tarvainen & Valpola, 2017; Miyato et al., 2019) .

Recently, Berthelot et al. (2019) propose MixMatch, which unifies several dominant SSL approaches in one framework and achieves state-of-the-art performance.

Despite the individual advances in LNL and SSL, their connection has been underexplored.

In this work, we propose DivideMix, which addresses learning with label noise in a semi-supervised manner.

Different from most existing LNL approaches, DivideMix discards the sample labels that are highly likely to be noisy, and leverages the noisy samples as unlabeled data to regularize the model from overfitting and improve generalization performance.

The key contributions of this work are:

??? We propose co-divide, which trains two networks simultaneously.

For each network, we dynamically fit a Gaussian Mixture Model (GMM) on its per-sample loss distribution to divide the training samples into a labeled set and an unlabeled set.

The divided data is then used to train the other network.

Co-divide keeps the two networks diverged, so that they can filter different types of error and avoid confirmation bias in self-training.

??? During SSL phase, we improve MixMatch with label co-refinement and co-guessing to account for label noise.

For labeled samples, we refine their ground-truth labels using the network's predictions guided by the GMM for the other network.

For unlabeled samples, we use the ensemble of both networks to make reliable guesses for their labels.

??? We experimentally show that DivideMix significantly advances state-of-the-art results on multiple benchmarks with different types and levels of label noise.

We also provide extensive ablation study and qualitative results to examine the effect of different components.

2 RELATED WORK

Most existing methods for training DNNs with noisy labels seek to correct the loss function.

The correction can be categorized in two types.

The first type treats all samples equally and correct loss either explicitly or implicitly through relabeling the noisy samples.

For relabeling methods, the noisy samples are modeled with directed graphical models (Xiao et al., 2015) , Conditional Random Fields (Vahdat, 2017) , knowledge graph (Li et al., 2017b) , or DNNs (Veit et al., 2017; Lee et al., 2018) .

However, they require access to a small set of clean samples.

Recently, Tanaka et al. (2018) and Yi & Wu (2019) propose iterative methods which relabel samples using network predictions.

For explicit loss correction.

Reed et al. (2015) propose a bootstrapping method which modifies the loss with model predictions, and improve the bootstrapping method by exploiting the dimensionality of feature subspaces.

Patrini et al. (2017) estimate the label corruption matrix for loss correction, and Hendrycks et al. (2018) improve the corruption matrix by using a clean set of data.

The second type of correction focuses on reweighting training samples or separating clean and noisy samples, which results in correcting the loss function (Thulasidasan et al., 2019; Konstantinov & Lampert, 2019) .

A common method is to consider samples with smaller loss as clean ones (Shen & Sanghavi, 2019) .

Jiang et al. (2018) train a mentor network to guide a student network by assigning weights to samples.

Ren et al. (2018) reweight samples based on their gradient directions.

Chen et al. (2019) apply cross validation to identify clean samples.

Arazo et al. (2019) calculate sample weights by modeling per-sample loss with a mixture model.

Han et al. (2018) train two networks which select small-loss samples within each mini-batch to train each other, and improve it by updating the network on disagreement data to keep the two networks diverged.

Contrary to all aforementioned methods, our method discards the labels that are highly likely to be noisy, and utilize the noisy samples as unlabeled data to regularize training in a SSL manner.

Ding et al. (2018) and Kong et al. (2019) have shown that SSL method is effective in LNL.

However, their methods do not perform well under high levels of noise, whereas our method can better distinguish and utilize noisy samples.

Besides leveraging SSL, our method also introduces other advantages.

Compared to self-training methods (Jiang et al., 2018; Arazo et al., 2019) , our method can avoid the confirmation bias problem (Tarvainen & Valpola, 2017) by training two networks to filter error for each other.

Compared to Co-teaching (Han et al., 2018) and Co-teaching+ , our method is more robust to noise by enabling the two networks to teach each other implicitly at each epoch (co-divide) and explicitly at each mini-batch (label co-refinement and co-guessing).

its per-sample loss distribution with a GMM to divide the dataset into a labeled set (mostly clean) and an unlabeled set (mostly noisy), which is then used as training data for the other network (i.e. co-divide).

At each mini-batch, a network performs semi-supervised training using an improved MixMatch method.

We perform label co-refinement on the labeled samples and label co-guessing on the unlabeled samples.

SSL methods aim to improve the model's performance by leveraging unlabeled data.

Current state-of-the-art SSL methods mostly involve adding an additional loss term on unlabeled data to regularize training.

The regularization falls into two classes: consistency regularization (Laine & Aila, 2017; Tarvainen & Valpola, 2017; Miyato et al., 2019) enforces the model to produce consistent predictions on augmented input data; entropy minimization (Grandvalet & Bengio, 2004; Lee, 2013) encourages the model to give high-confidence predictions on unlabeled data.

Recently, Berthelot et al. (2019) propose MixMatch, which unifies consistency regularization, entropy minimization, and the MixUp regularization into one framework.

In this section, we introduce DivideMix, our proposed method for learning with noisy labels.

An overview of the method is shown in Figure 1 .

To avoid confirmation bias of self-training where the model would accumulate its errors, we simultaneously train two networks to filter errors for each other through epoch-level implicit teaching and batch-level explicit teaching.

At each epoch, we perform co-divide, where one network divides the noisy training dataset into a clean labeled set (X ) and a noisy unlabeled set (U), which are then used by the other network.

At each mini-batch, one network utilizes both labeled and unlabeled samples to perform semi-supervised learning guided by the other network.

Algorithm 1 delineates the full algorithm.

Deep networks tend to learn clean samples faster than noisy samples (Arpit et al., 2017) , leading to lower loss for clean samples (Han et al., 2018; Chen et al., 2019) .

Following Arazo et al. (2019) , we aim to find the probability of a sample being clean by fitting a mixture model to the per-sample

denote the training data, where x i is an image and yi ??? {0, 1} C is the one-hot label over C classes.

Given a model with parameters ??, the cross-entropy loss (??) reflects how well the model fits the training samples:

where p c model is the model's output softmax probability for class c. Arazo et al. (2019) fit a two-component Beta Mixture Model (BMM) to the max-normalized loss to model the distribution of clean and noisy samples.

However, we find that BMM tends to produce undesirable flat distributions and fails when the label noise is asymmetric.

Instead, Gaussian Mixture Model (GMM) (Permuter et al., 2006) can better distinguish clean and noisy samples due to its flexibility in the sharpness of distribution.

Therefore, we fit a two-component GMM to using the Expectation-Maximization algorithm.

For each sample, its clean probability w i is the posterior probability p(g| i ), where g is the Gaussian component with smaller mean (smaller loss).

We divide the training data into a labeled set and an unlabeled set by setting a threshold ?? on w i .

However, training a model using the data divided by itself could lead to confirmation bias (i.e. the 1 Input: ??

(1) and ?? (2) , training dataset (X , Y), clean probability threshold ?? , number of augmentations M , sharpening temperature T , unsupervised loss weight ??u, Beta distribution parameter ?? for MixMatch.

// standard training (with confidence penalty) 3 while e < MaxEpoch do

) // model per-sample loss with ?? (1) to obtain clean proabability for ??

) // model per-sample loss with ?? (2) to obtain clean proabability for ??

6 for k = 1, 2 do // train the two networks one by one

// refine ground-truth label guided by the clean probability produced by the other network

// apply temperature sharpening to the refined label

)

// co-guessing: average the predictions from both networks across augmentations of u b

// apply temperature sharpening to the guessed label model is prone to confirm its mistakes (Tarvainen & Valpola, 2017) ), as noisy samples that are wrongly grouped into the labeled set would keep having lower loss due to the model overfitting to their labels.

Therefore, we propose co-divide to avoid error accumulation.

In co-divide, the GMM for one network is used to divide training data for the other network.

The two networks are kept diverged from each other due to different (random) parameter initialization, different training data division, different (random) mini-batch sequence, and different training targets.

Being diverged offers the two networks distinct abilities to filter different types of error, making the model more robust to noise.

Confidence Penalty for Asymmetric Noise.

For initial convergence of the algorithm, we need to "warm up" the model for a few epochs by training on all data using the standard cross-entropy loss.

The warm up is effective for symmetric (i.e. uniformly random) label noise.

However, for asymmetric (i.e. class-conditional) label noise, the network would quickly overfit to noise during warm up and produce over-confident (low entropy) predictions, which leads to most samples having near-zero normalized loss (see Figure 2a) .

In such cases, the GMM cannot effectively distinguish clean and noisy samples based on the loss distribution.

To address this issue, we penalize confident predictions from the network by adding a negative entropy term, ???H (Pereyra et al., 2017) , to the cross-entropy loss during warm up.

The entropy of a model's prediction for an input x is defined as:

By maximizing the entropy, becomes more evenly distributed (see Figure 2b ) and easier to be modeled by the GMM.

Furthermore, in Figure 2c we show when the model is trained with DivideMix for 10 more epochs after warm up.

The proposed method can significantly reduce the loss for clean samples while keeping the loss larger for most noisy samples.

To account for label noise, we make two improvements to MixMatch which enable the two networks to teach each other.

First, we perform label co-refinement for labeled samples by linearly combining the ground-truth label y b with the network's prediction p b (averaged across multiple augmentations of x b ), guided by the clean probability w b produced by the other network:

Then we apply a sharpening function on the refined label to reduce its temperature:

Second, we use the ensemble of predictions from both networks to "co-guess" the labels for unlabeled samples (algorithm 1, line 20), which can produce more reliable guessed labels.

Having acquiredX (and??) which consists of multiple augmentations of labeled (unlabeled) samples and their refined (guessed) labels, we follow MixMatch to "mix" the data, where each sample is interpolated with another sample randomly chosen from the combined mini-batch ofX and??. Specifically, for a pair of samples (x 1 , x 2 ) and their corresponding labels (p 1 , p 2 ), the mixed (x , p ) is computed by:

MixMatch transformsX and?? into X and U .

Equation 6 ensures that X are "closer" toX than??. The loss on X is the cross-entropy loss and the loss on U is the mean squared error:

Under high levels of noise, the network would be encouraged to predict the same class to minimize the loss.

To prevent assigning all samples to a single class, we apply the regularization term used by Tanaka et al. (2018) and Arazo et al. (2019) , which uses a uniform prior distribution ?? (i.e. ?? c = 1/C) to regularize the model's average output across all samples in the mini-batch:

Finally, the total loss is:

In our experiments, we set ?? r as 1 and use ?? u to control the strength of the unsupervised loss.

We extensively validate our method on four benchmark datasets, namely CIFAR-10, CIFAR-100 (Krizhevsky & Hinton, 2009 ), Clothing1M (Xiao et al., 2015) , and WebVision (Li et al., 2017a) .

Both CIFAR-10 and CIFAR-100 contain 50K training images and 10K test images of size 32 ?? 32.

Following previous works (Tanaka et al., 2018; Li et al., 2019) , we experiment with two types of label noise: symmetric and asymmetric.

Symmetric noise is generated by randomly replacing the labels for a percentage of the training data with all possible labels.

Note that there is another criterion for symmetric label noise injection where the true labels cannot be maintained (Jiang et al., 2018; , for which we also report the results (Table 6 in Appendix).

Asymmetric noise is designed to mimic the structure of real-world label noise, where labels are only replaced by similar classes (e.g. deer???horse, dog???cat).

We use an 18-layer PreAct Resnet (He et al., 2016) and train it using SGD with a momentum of 0.9, a weight decay of 0.0005, and a batch size of 128.

The network is trained for 300 epochs.

We set the initial learning rate as 0.02, and reduce it by a factor of 10 after 150 epochs.

The warm up period is 10 epochs for CIFAR-10 and 30 epochs for CIFAR-100.

We find that most hyperparameters introduced by DivideMix do not need to be heavily tuned.

For all CIFAR experiments, we use the same hyperparameters M = 2, T = 0.5, and ?? = 4.

?? is set as 0.5 except for 90% noise ratio when it is set as 0.6.

We choose ?? u from {0, 25, 50, 150} using a small validation set.

Clothing1M and WebVision 1.0 are two large-scale datasets with real-world noisy labels.

Clothing1M consists of 1 million training images collected from online shopping websites with labels generated from surrounding texts.

We follow previous work (Li et al., 2019) and use ResNet-50 with ImageNet pretrained weights.

WebVision contains 2.4 million images crawled from the web using the 1,000 concepts in ImageNet ILSVRC12.

Following previous work (Chen et al., 2019) , we compare baseline methods on the first 50 classes of the Google image subset using the inception-resnet v2 (Szegedy et al., 2017) .

The training details are delineated in Appendix B.

We compare DivideMix with multiple baselines using the same network architecture.

Here we introduce some of the most recent state-of-the-art methods: Meta-Learning (Li et al., 2019) proposes a gradient based method to find model parameters that are more noise-tolerant; Joint-Optim (Tanaka et al., 2018) and P-correction (Yi & Wu, 2019) jointly optimize the sample labels and the network parameters; M-correction (Arazo et al., 2019) Table 1 : Comparison with state-of-the-art methods in test accuracy (%) on CIFAR-10 and CIFAR-100 with symmetric noise.

Methods marked by * denote re-implementations based on public code.

Note that none of these methods can consistently outperform others across different datasets.

Mcorrection excels at symmetric noise, whereas Meta-Learning performs better for asymmetric noise.

Table 1 shows the results on CIFAR-10 and CIFAR-100 with different levels of symmetric label noise ranging from 20% to 90%.

We report both the best test accuracy across all epochs and the averaged test accuracy over the last 10 epochs.

DivideMix outperforms state-of-the-art methods by a large margin across all noise ratios.

The improvement is substantial (???10% in accuracy) for the more challenging CIFAR-100 with high noise ratios.

Appendix A shows comparison with more methods in Table 6 .

The results on CIFAR-10 with asymmetric noise is shown in Table 2 .

We use 40% because certain classes become theoretically indistinguishable for asymmetric noise larger than 50%.

Cross-Entropy 85.0 72.3 F-correction (Patrini et al., 2017) 87.2 83.1 M-correction (Arazo et al., 2019) 87.4 86.3 Iterative-CV (Chen et al., 2019) 88.6 88.0 P-correction (Yi & Wu, 2019) 88.5 88.1 Joint-Optim (Tanaka et al., 2018) 88.9 88.4 Meta-Learning (Li et al., 2019) 89.2 88.6 DivideMix 93.4 92.1 Table 2 : Comparison with state-of-the-art methods in test accuracy (%) on CIFAR-10 with 40% asymmetric noise.

We re-implement all methods under the same setting.

Table 3 and Table 4 show the results on Clothing1M and WebVision, respectively.

DivideMix consistently outperforms state-of-the-art methods across all datasets with different types of label noise.

For WebVision, we achieve more than 12% improvement in top-1 accuracy.

Cross-Entropy 69.21 F-correction (Patrini et al., 2017) 69.84 M-correction (Arazo et al., 2019) 71.00 Joint-Optim (Tanaka et al., 2018) 72.16 Meta-Cleaner 72.50 Meta-Learning (Li et al., 2019) 73.47 P-correction (Yi & Wu, 2019) 73.49 DivideMix 74.76 62.68 84.00 57.80 81.36 MentorNet (Jiang et al., 2018) 63.00 81.40 57.80 79.92 Co-teaching (Han et al., 2018) 63.58 85.20 61.48 84.70 Iterative-CV (Chen et al., 2019) 65 Table 5 : Ablation study results in terms of test accuracy (%) on CIFAR-10 and CIFAR-100.

??? To study the effect of model ensemble during test, we use the prediction from a single model ??

instead of averaging the predictions from both networks as in DivideMix.

Note that the training process remains unchanged.

The decrease in accuracy suggests that the ensemble of two diverged networks consistently yields better performance during inference.

??? To study the effect of co-training, we train a single network using self-divide (i.e. divide the training data based on its own loss).

The performance further decreases compared to ?? (1) .

??? We find that both label refinement and input augmentation are beneficial for DivideMix.

??? We combine self-divide with the original MixMatch as a naive baseline for using SLL in LNL.

Appendix A also introduces more in-depth studies in examining the robustness of our method to label noise, including the AUC for clean/noisy sample classification on CIFAR-10 training data, qualitative examples from Clothing1M where our method can effectively identify the noisy samples and leverage them as unlabeled data, and visualization results using t-SNE.

In this paper, we propose DivideMix for learning with noisy labels by leveraging SSL.

Our method trains two networks simultaneously and achieves robustness to noise through dataset co-divide, label co-refinement and co-guessing.

Through extensive experiments across multiple datasets, we show that DivideMix consistently exhibits substantial performance improvements compared to state-of-the-art methods.

For future work, we are interested in incorporating additional ideas from SSL to LNL, and vice versa.

Furthermore, we are also interested in adapting DivideMix to other domains such as NLP.

In In Figure 3 , we show the Area Under a Curve (AUC) for clean/noisy sample classification on CIFAR-10 training data from one of the GMMs during the first 100 epochs.

Our method can effectively separate clean and noisy samples as training proceeds, even for high noise ratio.

In Figure 4 , we show example images in Clothing1M identified by our method as noisy samples.

Our method achieves noise filtering by discarding the noisy labels (shown in red) and using the co-guessed labels (shown in blue) to regularize training.

In Figure 5 , we visualize the features of training images using t-SNE (Maaten & Hinton, 2008) .

The model is trained using DivideMix for 200 epochs on CIFAR-10 with 80% label noise.

The embeddings form 10 distinct clusters corresponding to the true class labels, not the noisy training labels, which demonstrates our method's robustness to label noise.

For CIFAR experiments, the only hyperparameter that we tune on a per-experiment basis is the unsupervised loss weight ?? u .

Table 7 shows the value that we use.

A larger ?? u is required for stronger regularization under high noise ratios or with more classes.

For both Clothing1M and WebVision, we use the same set of hyperparameters M = 2, T = 0.5, ?? = 0.5, ?? u = 0, ?? = 0.5, and train the network using SGD with a momentum of 0.9, a weight decay of 0.001, and a batch size of 32.

The warm up period is 1 epoch.

For Clothing1M, we train the network for 80 epochs.

The initial learning rate is set as 0.002 and reduced by a factor of 10 after 40 epochs.

For each epoch, we sample 1000 mini-batches from the training data while ensuring the labels (noisy) are balanced.

For WebVision, we train the network for 100 epochs.

The initial learning rate is set as 0.01 and reduced by a factor of 10 after 50 epochs.

Table 7 : Unsupervised loss weight ?? u for CIFAR experiments.

Higher noise ratio requires stronger regularization from unlabeled samples.

Here we clarify some details for the baseline methods in the ablation study.

First, DivideMix w/o co-training still has dataset division, label refinement and label guessing, but performed by the same model.

Thus, the performance drop (especially for CIFAR-100 with high noise ratio) suggests the disadvantage of self-training.

Second, label refinement is important for high noise ratio because more noisy samples would be mistakenly divided into the labeled set.

Third, augmentation improves performance through both producing more reliable predictions and achieving consistency regularization.

In addition, same as Berthelot et al. (2019) , we also find that temperature sharpening is essential for our method to perform well.

We analyse the training time of DivideMix to understand its efficiency.

In Table 8 , we compare the total training time of DivideMix on CIFAR-10 with several state-of-the-art methods, using a single Nvidia V100 GPU.

DivideMix is slower than Co-teaching+ , but faster than P-correction (Yi & Wu, 2019) and Meta-Learning (Li et al., 2019) which involve multiple training iterations.

In Table 9 , we also break down the computation time for each operation in DivideMix.

@highlight

We propose a novel semi-supervised learning approach with SOTA performance on combating learning with noisy labels.