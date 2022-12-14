Unsupervised embedding learning aims to extract good representations from data without the use of human-annotated labels.

Such techniques are apparently in the limelight because of the challenges in collecting massive-scale labels required for supervised learning.

This paper proposes a comprehensive approach, called Super-AND, which is based on the Anchor Neighbourhood Discovery model.

Multiple losses defined in Super-AND make similar samples gather even within a low-density space and keep features invariant against augmentation.

As a result, our model outperforms existing approaches in various benchmark datasets and achieves an accuracy of 89.2% in CIFAR-10 with the Resnet18 backbone network, a 2.9% gain over the state-of-the-art.

Deep learning and convolutional neural network have become an indispensable technique in computer vision (LeCun et al., 2015; Krizhevsky et al., 2012; Lawrence et al., 1997) .

Remarkable developments, in particular, were led by supervised learning that requires thousands or more labeled data.

However, high annotation costs have become a significant drawback in training a scalable and practical model in many domains.

In contrast, unsupervised deep learning that requires no label has recently started to get attention in computer vision tasks.

From clustering analysis (Caron et al., 2018; Ji et al., 2018) , and self-supervised model (Gidaris et al., 2018; Bojanowski & Joulin, 2017) to generative model (Goodfellow et al., 2014; Kingma & Welling, 2013; Radford et al., 2016) , various learning methods came out and showed possibilities and prospects.

Unsupervised embedding learning aims to extract visually meaningful representations without any label information.

Here "visually meaningful" refers to finding features that satisfy two traits: (i) positive attention and (ii) negative separation (Ye et al., 2019; Zhang et al., 2017c; Oh Song et al., 2016) .

Data samples from the same ground truth class, i.e., positive samples, should be close in the embedding space (Fig. 1a) ; whereas those from different classes, i.e., negative samples, should be pushed far away in the embedding space (Fig. 1b) .

However, in the setting of unsupervised learning, a model cannot have knowledge about whether given data points are positive samples or negative samples.

Several new methods have been proposed to find 'visually meaningful' representations.

The sample specificity method considers all data points as negative samples and separates them in the feature space (Wu et al., 2018; Bojanowski & Joulin, 2017) .

Although this method achieves high performance, its decisions are known to be biased from learning only from negative separation.

One approach utilizes data augmentation to consider positive samples in training (Ye et al., 2019) , which efficiently reduces any ambiguity in supervision while keeping invariant features in the embedding space.

Another approach is called the Anchor Neighborhood Discovery (AND) model, which alleviates the complexity in boundaries by discovering the nearest neighbor among the data points (Huang et al., 2019) .

Each of these approaches overcomes different limitations of the sample specificity method.

However, no unified approach has been proposed.

This paper presents a holistic method for unsupervised embedding learning, named Super-AND.

Super-AND extends the AND algorithm and unifies various but dominant approaches in this domain with its unique architecture.

Our proposed model not only focuses on learning distinctive features across neighborhoods, but also emphasizes edge information in embeddings and maintains the unchanging class information from the augmented data.

Besides combining existing techniques, we newly introduce Unification Entropy loss (UE-loss), an adversary of sample specificity loss, which is able to gather similar data points within a low-density space.

Extensive experiments are conducted on several benchmark datasets to verify the superiority of the model.

The results show the synergetic advantages among modules of Super-AND.

The main contributions of this paper are as follows:

??? We effectively unify various techniques from state-of-the-art models and introduce a new loss, UE-loss, to make similar data samples gather in the low-density space.

??? Super-AND outperforms all baselines in various benchmark datasets.

It achieved an accuracy of 89.2% in the CIFAR-10 dataset with the ResNet18 backbone network, compared to the state-of-the-art that gained 86.3%.

??? The extensive experiments and the ablation study show that every component in Super-AND contributes to the performance increase, and also indicate their synergies are critical.

Our model's outstanding performance is a step closer to the broader adoption of unsupervised techniques in computer vision tasks.

The premise of data-less embedding learning is at its applicability to practical scenarios, where there exists only one or two examples per cluster.

Codes and trained data for Super-AND are accessible via a GitHub link.

Generative model.

This type of model is a powerful branch in unsupervised learning.

By reconstructing the underlying data distribution, a model can generate new data points as well as features from images without labels.

Generative adversarial network (Goodfellow et al., 2014) has led to rapid progress in image generation problems Arjovsky et al., 2017) .

While some attempts have been made in terms of unsupervised embedding learning (Radford et al., 2016) , the main objective of generative models lies at mimicking the true distribution of each class, rather than discovering distinctive categorical information the data contains.

Self-supervised learning.

This type of learning uses inherent structures in images as pseudo-labels and exploits labels for back-propagation.

For example, a model can be trained to create embeddings by predicting the relative position of a pixel from other pixels (Doersch et al., 2015) or the degree of changes after rotating images (Gidaris et al., 2018) .

Predicting future frames of a video can benefit from this technique (Walker et al., 2016) .

Wu et al. (2018) proposed the sample specificity method that learns feature representation from capturing apparent discriminability among instances.

All of these methods are suitable for unsupervised embedding learning, although there exists a risk of false knowledge from generated labels that weakly correlate with the underlying class information.

Learning invariants from augmentation.

Data augmentation is a strategy that enables a model to learn from datasets with an increased variety of instances.

Popular techniques include flipping, scaling, rotation, and grey-scaling.

These techniques do not deform any crucial features of data, but only change the style of images.

Some studies hence use augmentation techniques and train models Clustering analysis.

This type of analysis is an extensively studied area in unsupervised learning, whose main objective is to group similar objects into the same class.

Many studies either leveraged deep learning for dimensionality reduction before clustering (Schroff et al., 2015; Baldi, 2012) or trained models in an end-to-end fashion (Xie et al., 2016; Yang et al., 2016) .

Caron et al. (2018) proposed a concept called deep cluster, an iterative method that updates its weights by predicting cluster assignments as pseudo-labels.

However, directly reasoning the global structures without any label is error-prone.

The AND model, which we extend in this work, combines the advantages of sample specificity and clustering strategy to mitigate the noisy supervision via neighborhood analysis (Huang et al., 2019) .

Problem definition.

Assume that there is an unlabeled image set I, and a batch set B with n images: B = {x 1 , x 2 , ..., x n } ??? I. Our goal is to get a feature extractor f ?? whose representations (i.e., v i = f ?? (x i )) are "visually meaningful," a definition we discussed earlier.

LetB = {x 1 ,x 2 , ...,x n } be the augmentation set of input batches B. Super-AND projects images x i ,x i from batches B,B to 128 dimensional embeddings v i ,v i .

During this process, the Sobelprocessed images (Maini & Aggarwal, 2008) are also used, and feature vectors from both images are concatenated to emphasize edge information in embeddings (see the left side in Fig. 2) .

Then, the model computes the probability of images p i ,p i being recognized as its own class with a nonparametric classifier (see the right side in Fig. 2) .

A temperature parameter (?? < 1) was added to ensure a label distribution with low entropy .

To reduce the computational complexity in calculating embeddings from all images, we set up a memory bank M to save instance embeddings m i accumulated from the previous steps, as similarly proposed by Wu et al. (2018) .

The memory bank M is updated by exponential moving average (Lucas & Saccucci, 1990 ).

The probability vector p i is defined in Eq 1, where the superscript j in vector notation (i.e., v j ) represents the j-th component value of a given vector.

We define the neighborhood relationship vectors r i ,r i , and compute these vectors by the cosine similarity between the embedding vectors v i ,v i , and the memory bank M (Eq 2).

The extracted vectors are used to define the loss term that detects a discrepancy between neighborhoods.

The loss term also enforces features v to remain unchanged even after data augmentation.

The loss term is written as

where N is the set of progressively selected pairs discovered by the nearest neighbor algorithm, and V, R,R, P are matrices of concatenated embedded vectors v i , r i ,r i , p i from the batch image set, respectively.

w(t) is the hyper-parameter that controls the weights of UE-loss.

The algorithm below describes how to train Super-AND.

Algorithm 1: Main algorithm for training Super-AND.

Input : Unlabeled image set I, encoder f ?? to train, the number of total rounds for training:

Rounds, and the number of total epochs for training:

Compute gradient and update weights by backpropagation 14 end 15 end

Existing clustering methods like (Caron et al., 2018; Xie et al., 2016) train networks to find an optimal mapping.

However, their learned decisions are unstable due to initial randomness, and some overfitting can occur during the training period (Zhang et al., 2017a) .

To tackle these limitations, the AND model suggests a finer-grained clustering focusing on 'neighborhoods'.

By regarding the nearest-neighbor pairs as local classes, AND can separate data points that belong to different neighborhood sets from those in the same neighborhood set.

We adopt this neighborhood discovery strategy in our Super-AND.

The AND algorithm has three main steps: (1) neighborhood discovery, (2) progressive neighborhood selection with curriculum, and (3) neighborhood supervision.

For the first step, the k nearest neighborhood (k-NN) algorithm is used to discover all neighborhood pairs (Eq 7 and Eq 8), and these pairs are progressively selected for curriculum learning.

We choose a small part of neighborhood pairs at the first round, and gradually increase the amount of selection for training (Current/Total rounds ?? 100%).

Since we cannot assure that every neighborhood is visually similar, this progressive method helps provide a consistent view of local class information for training at each round.

When selecting candidate neighborhoods for local classes, the entropy of probability vector H(x i ) is utilized as a criterion (Eq 9).

Probability vector p i , obtained from softmax function (Eq 1), shows the visual similarity between training instances in a probabilistic manner.

Data points with low entropy represent they reside in a relatively low-density area and have only a few surrounding neighbors.

Neighborhood pairs containing such data points likely share consistent and easily distinguishable features from other pairs.

We select neighborhood set N from?? that is in a lower entropy order.

The AND-loss function is defined to distinguish neighborhood pairs from one another.

Data points from the same neighborhoods need to be classified in the same class (i.e., the left-hand term in Eq 10).

If any data point is present in the selected pair, it is considered to form an independent class (i.e., the right-hand term in Eq 10).

Existing sample specificity methods (Wu et al., 2018; Bojanowski & Joulin, 2017) consider every single data point as a prototype for a class.

They use the cross-entropy loss to separate all data points in the L2-normalized embedding space.

Due to its confined space by normalization, data points cannot be placed far away from one another, and this space limitation induces an effect that leads to a concentration of positive samples, as shown in Fig. 1a .

The unification entropy loss (UE-loss) is able to even strengthen the concentration-effect above.

We define the UE-loss as the entropy of the probability vectorp i .

Probability vectorp i is calculated from the softmax function and represents the similarity between instances except for instance itself (Eq 11).

By excluding the class of one's own, minimizing the loss makes nearby data points attract each other -a concept that is contrary to minimizing the sample specificity loss.

Employing both AND-loss and the UE-loss will enforce similar neighborhoods to be positioned close while keeping the overall neighborhoods as separated as possible.

This loss is calculated as in Eq 12.

Unsupervised embedding learning aims at training encoders to extract visually meaningful features that are consistent with ground truth labels.

Such learning cannot use any external guidance on features.

Several previous studies tried to infer which features are substantial in a roundabout way; data augmentation is one such solution (Ye et al., 2019; Ji et al., 2018; Perez & Wang, 2017; Volpi et al., 2018) .

Since augmentation does not deform the underlying data characteristics, invariant features learned from the augmented data will still contain the class-related information.

Naturally, a training network based on these features will show performance gain.

We define the Augmentation-loss to learn invariant image features.

Assume that there is an image along with its augmented versions.

We may regard every augmentation instance as a positive sample.

The neighborhood relationship vectors, which show the similarity between all instances stored in memory, should also be similar to initial data points than other instances in the same batch.

In Eq 13, the probability of an augmented instance that is correctly identified as class-i is denoted asp i i ; and that of i-th original instance that is wrongly identified as class-j (j = i),p j i .

The Augmentation-loss is then defined to minimize misclassification over instances in all batches (Eq 14).

The evaluation involved extensive experiments.

We enumerated the model with different backbone networks on two kinds of benchmarks: coarse-grained and fine-grained.

Our ablation study helps speculate which components of the model are critical in performance.

Finally, the proposed model is compared to the original AND from different perspectives.

Datasets.

A total of six image datasets are utilized, where three are coarse-grained datasets: (6) is used for qualitative analysis.

Training.

We used AlexNet (Krizhevsky et al., 2012) and ResNet18 (He et al., 2016) as the backbone networks.

Hyper-parameters were tuned in the same way as the AND algorithm.

We used SGD with Nesterov momentum 0.9 for the optimizer.

We fixed the learning rate as 0.03 for the first 80 epochs, and scaled-down 0.1 every 40 epochs.

The batch size is set as 128, and the model was trained in 5 rounds and 200 epochs per round.

Weights for UE-loss w(t) (Eq 6) are initialized from 0 and increased 0.2 every 80 epochs.

For Augmentation-loss, we used four types: Resized Crop, Grayscale, ColorJitter, and Horizontal Flip.

Horizontal Flip was not used in the case of the SVHN dataset because the SVHN dataset is digit images.

Update momentum of the exponential moving average for memory bank was set to 0.5.

Evaluation.

Following the method from Wu et al. (2018) , we used the weighted k-NN classifier for making prediction.

Top k-nearest neighbors N top were retrieved and used to predict the final outcome in a weighted fashion.

We set k = 200 and the weight function for each class c as

, where c i is the class index for i-th instance.

Top-1 classification accuracy was used for evaluation.

Baseline models.

We adopt six state-of-the-art baselines for comparison.

They are (1) SplitBrain (Zhang et al., 2017b) , (2) Counting (Noroozi et al., 2017) , (3) DeepCluster (Caron et al., 2018) , (4) Instance (Wu et al., 2018) , (5) ISIF (Ye et al., 2019) , and (6) AND (Huang et al., 2019) .

For fair comparison, the same backbone networks were used.

Coarse-grained evaluation.

Table 1 describes the object classification performance of seven models, including the proposed Super-AND on three coarse-grained datasets: CIFAR-10, CIFAR-100, and SVHN.

Super-AND surpasses state-of-the-art baselines on all datasets except for one case, where the model underperforms marginally on CIFAR-100 with AlexNet.

One notable observa- Table 1 : k-NN Evaluation on coarse-grained datasets.

Results that are marked as * are borrowed from the previous works (Huang et al., 2019; Ye et al., 2019 tion is that the difference between previous models and super-AND is mostly larger in the case of ResNet18 than the AlexNet backbone network.

These results reveal that our model is superior to other methods and may indicate that our methodology can give more benefits to stronger CNN architectures.

Fine-grained evaluation.

We perform evaluations on fine-grained datasets that require the ability to discriminate subtle differences between classes.

Table 2 shows that Super-AND achieves an outstanding performance compared to three baselines with the ResNet18 backbone network.

We excerpted the results of Instance and AND model from the previous work.

Backbone network.

We tested the choice of backbone networks in terms of classification performance.

AlexNet, ResNet18, and ResNet101 are used and evaluated on CIFAR-10, as shown in Table 3 .

From the results, we can infer that the stronger the backbone network our model has, the better the performance model can produce.

Ablation study.

To verify every component does its role and has some contribution to the performance increase, an ablation study was conducted.

Since Super-AND combines various mechanisms based on AND algorithm, we study the effect of removing each component: (1) Super-AND without the UE-loss, (2) Super-AND without the Sobel filter, (3) Super-AND without the Augmentation-loss.

Table 4 displays the evaluation result based on the CIFAR-10 dataset and the ResNet18 backbone network.

We found that every component contributes to the performance increase, and a particularly dramatic decrease in performance occurs when removing the Augmentation-loss.

Initialization.

Instead of running the algorithm from an arbitrary random model, we can pre-train the network with "good" initial data points to discover consistent neighborhoods.

We investigate two different initialization methods and check whether the choice is critical.

Three models were compared: (1) a random model, (2) an initialized model with instance loss (Wu et al., 2018) from AND, and (3) an initialized model with multiple losses from Super-AND.

Table 5 shows that the choice of initialization is not significant, and solely using the instance loss even has an adverse effect on performance.

This finding implies that Super-AND is robust to random initial data points, yet the model will show an unexpected outcome if initialization uses ambiguous knowledge.

Embedding quality analysis.

Super-AND leverages the synergies from learning both similarities in neighborhoods and invariant features from data augmentation.

Super-AND, therefore, has a high capability of discovering cluster relationships, compared to the original AND model that only uses the neighborhood information.

Fig. 3 exploits t-SNE (Maaten & Hinton, 2008) to visualize the learned representations of three of the selected classes based on the two algorithms in CIFAR-10.

The plot demonstrates that Super-AND discovers consistent and discriminative clusters.

We investigated the embedding quality by evaluating the class consistency of selected neighborhoods.

Cheat labels are used to check whether neighborhood pairs come from the same class.

Since both algorithms increase the selection ratio every round when gathering the part of discovered neighborhoods, the consistency of selected neighborhoods will naturally decrease.

This relationship is drawn in Fig. 4 .

The reduction for Super-AND, nonetheless, is not significant compared to AND: our model maintains high-performance throughout the training rounds.

Qualitative study.

Fig. 5 illustrates the top-5 nearest retrievals of AND (i.e., upper rows) and Super-AND (i.e., lower rows) based on the STL-10 dataset.

The example queries shown are dump trucks, airplanes, horses, and monkeys.

Images with red frames, which indicate negative samples, appear more frequently for AND than Super-AND.

This finding implies that Super-AND excels in capturing the class information compared to AND.

Its clusters are robust to misleading color information and well recognize the shape of objects within images.

For example, in the case of the airplane query, pictures retrieved from Super-AND are consistent in shape while AND results confuse a cruise picture as an airplane.

The color composition in Super-AND is also more flexible and can find a red dump truck or a spotted horse, as shown in the examples.

This paper presents Super-AND, a holistic technique for unsupervised embedding learning.

Besides the synergetic advantage combining existing methods brings, the newly proposed UE-loss that groups nearby data points even in a low-density space while maintaining invariant features via data augmentation.

The experiments with both coarse-grained and fine-grained datasets demonstrate our model's outstanding performance against the state-of-the-art models.

Our efforts to advance unsupervised embedding learning directly benefit future applications that rely on various image clustering tasks.

The high accuracy achieved by Super-AND makes the unsupervised learning approach an economically viable option where labels are costly to generate.

@highlight

We proposed a comprehensive approach for unsupervised embedding learning on the basis of AND algorithm.