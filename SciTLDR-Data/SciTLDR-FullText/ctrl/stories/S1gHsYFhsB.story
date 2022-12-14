We present and discuss a simple image preprocessing method for learning disentangled latent factors.

In particular, we utilize the implicit inductive bias contained in features from networks pretrained on the ImageNet database.

We enhance this bias by explicitly fine-tuning such pretrained networks on tasks useful for the NeurIPS2019 disentanglement challenge, such as angle and position estimation or color classification.

Furthermore, we train a VAE on regionally aggregate feature maps, and discuss its disentanglement performance using metrics proposed in recent literature.

Fully unsupervised methods, that is, without any human supervision, are doomed to fail for tasks such as learning disentangled representations (Locatello et al., 2018) .

In this contribution, we utilize the implicit inductive bias contained in models pretrained on the ImageNet database (Russakovsky et al., 2014) , and enhance it by finetuning such models on challenge-relevant tasks such as angle and position estimation or color classification.

In particular, our submission for challenge stage 2 builds on our submission from stage 1 1 , in which we employed pretrained CNNs to extract convolutional feature maps as a preprocessing step before training a VAE (Kingma and Welling, 2013) .

Although this approach already results in partial disentanglement, we identified two issues with the feature vectors extracted this way.

Firstly, the feature extraction network is trained on ImageNet, which is rather dissimilar to the MPI3d dataset used in the challenge.

Secondly, the feature aggregation mechanism was chosen ad-hoc and likely does not retain all information needed for disentanglement.

We attempt to fix these issues by finetuning the feature extraction network as well as learning the aggregation of feature maps from data by using the labels of the simulation datasets MPI3d-toy and MPI3d-realistic.

Our method consists of the following three steps: (1) supervised finetuning of the feature extraction CNN (section 2.1), (2) extracting a feature vector from each image in the dataset using the finetuned network (section 2.2), (3) training a VAE to reconstruct the feature vectors and disentangle the latent factors of variation (section 2.3).

In this step, we finetune the feature extraction network offline (before submission to the evaluation server).

The goal is to adapt the network such that it produces aggregated feature vectors that capture the latent variables well.

In particular, the network is finetuned by learning to predict the value of each latent factor from the aggregated feature vector of an image.

To this end, we use the simulation datasets MPI3d-toy and MPI3d-realistic 2 , namely the images as inputs and the labels as supervised classification targets.

For the feature extraction network, we use the VGG19-BN architecture (Simonyan and Zisserman, 2014) of the torchvision package.

The input images are standardized using mean and variance across each channel computed from the ImageNet dataset.

We use the output feature maps of the last layer before the final average pooling (dimensionality 512??2??2) as the input to a feature aggregation module which reduces the feature map to a 512-dimensional vector 3 .

This aggregation module consists of three convolution layers with 1024, 2048, 512 feature maps and kernel sizes 1, 2, 1 respectively.

Each layer is followed by batch normalization and ReLU activation.

We also employ layerwise dropout with rate 0.1 before each convolution layer.

Finally, the aggregated feature vector is 2-normalized, which was empirically found to be important for the resulting disentanglement performance.

Then, for each latent factor, we add a linear classification layer computing the logits of each class from the aggregated feature vector.

These linear layers are discarded after this step.

We use both MPI3d-toy and MPI3d-realistic for training to push the network to learn features that identify the latent factors in a robust way, regardless of details such as reflections or specific textures.

In particular, we use a random split of 80% of each dataset as the training set, and the remaining samples as a validation set.

VGG19-BN is initialized with a set of weights resulting from ImageNet training 4 , and the aggregation module and linear layers were randomly initialized using uniform He initialization (He et al., 2015) .

The network is trained for 5 epochs using the RAdam optimizer (Liu et al., 2019) with learning rate 0.001, ?? 0 " 0.999, ?? 1 " 0.9, a batch size of 512 and a weight decay of 0.01.

We use a multi-task classification loss consisting of the sum of cross entropies between the prediction and the ground truth of each latent factor.

After training, the classification accuracy on the validation set is around 98% for the two degrees of freedom of the robot arm, and around 99.9% for the remaining latent factors.

In this step, we use the finetuned feature extraction network to produce a set of aggregated feature vectors.

We simply run the network detailed in the previous step on each image of the dataset and store the aggregated 512-dimensional vectors in memory.

Again, inputs to the feature extractor are standardized such that mean and variance across each channel correspond to the respective ones from the ImageNet dataset.

2.

Pretraining using any data was explicitly stated to be allowed by the challenge organizers.

3. Using aggregated feature vectors instead of feature maps is necessitated by the memory requirements of the challenge.

4. https://download.pytorch.org/models/vgg19_bn-c79401a0.pth

Finally, we train a standard ??-VAE (Higgins et al., 2017) on the set of aggregated feature vectors resulting from the previous step.

The encoder network consists of 4 fully-connected layers with 1024 neurons each, followed by two fully-connected layers parametrizing mean and log variance of the factorized Gaussian approximate posterior q pz | xq " N`??, ?? 2w

ith C " 16 latent factors.

The number of latent factors was experimentally determined.

The decoder network consists of 4 fully-connected layers with 1024 neurons each, followed by a fully-connected layer parametrizing the mean of the factorized Gaussian conditional distribution p px | zq " N p??, Iq.

The mean is constrained to range p0, 1q using the sigmoid activation.

All fully-connected layers but the final ones use batch normalization and are followed by ReLU activation functions.

We use orthogonal initialization Saxe et al. (2013) for all layers and assume a factorized Gaussian prior p pzq " N p0, Iq on the latent variables.

For optimization, we use the RAdam optimizer (Liu et al., 2019 ) with a learning rate of 0.001, ?? 0 " 0.999, ?? 1 " 0.9 and a batch size of B " 256.

The VAE is trained for N " 100 epochs by minimizing

where ?? is a hyperparameter to balance the losses of the MSE reconstruction and the KLD penalty terms.

As the scale of the KLD term depends on the numbers of latent factors C, we normalize it by C such that ?? can be varied independently of C. It can be harmful to start training with too much weight on the KLD term (Bowman et al., 2015) .

Therefore, we use the following cosine schedule to smoothly anneal ?? from ?? start " 0.001 to ?? end " 0.4 over the course of training:

where ??ptq is the value for ?? in training episode t P t0, . . .

, N??1u, and annealing runs from epoch t start " 10 to epoch t end " 49.

This schedule lets the model initially learn to reconstruct the data well and only then puts pressure on the latent variables to be factorized which we found to considerably improve performance.

On the public leaderboard (i.e. on MPI3D-real ), our best submission achieves the first rank on the FactorVAE (Kim and Mnih, 2018) , and DCI (Eastwood and Williams, 2018 ) metrics, with a large gap to the second-placed entry.

See appendix A for a discussion of the results.

Unsurprisingly, introducing prior knowledge simplifies the disentanglement task considerably, reflected in improved scores.

To do so, our approach makes use of task-specific supervision obtained from simulation, which restricts its applicability.

Nevertheless, it constitutes a demonstration that this type of supervision can transfer to better disentanglement on real world data, which was one of the goals of the challenge.

<|TLDR|>

@highlight

We use supervised finetuning of feature vectors to improve transfer from simulation to the real world