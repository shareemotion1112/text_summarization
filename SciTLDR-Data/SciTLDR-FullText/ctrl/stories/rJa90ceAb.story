Conventionally, convolutional neural networks (CNNs) process different images with the same set of filters.

However, the variations in images pose a challenge to this fashion.

In this paper, we propose to generate sample-specific filters for convolutional layers in the forward pass.

Since the filters are generated on-the-fly, the model becomes more flexible and can better fit the training data compared to traditional CNNs.

In order to obtain sample-specific features, we extract the intermediate feature maps from an autoencoder.

As filters are usually high dimensional, we propose to learn a set of coefficients instead of a set of filters.

These coefficients are used to linearly combine the base filters from a filter repository to generate the final filters for a CNN.

The proposed method is evaluated on MNIST, MTFL and CIFAR10 datasets.

Experiment results demonstrate that the classification accuracy of the baseline model can be improved by using the proposed filter generation method.

Variations exist widely in images.

For example, in face images, faces present with different head poses and different illuminations which are challenges to most face recognition models.

In the conventional training process of CNNs, filters are optimized to deal with different variations.

The number of filters increases if more variations are added to the input data.

However, for a test image, only a small number of the neurons in the network are activated which indicates inefficient computation BID13 ).Unlike CNNs with fixed filters, CNNs with dynamically generated sample-specific filters are more flexible since each input image is associated with a unique set of filters.

Therefore, it provides possibility for the model to deal with variations without increasing model size.

However, there are two challenges for training CNNs with dynamic filter generation.

The first challenge is how to learn sample-specific features for filter generation.

Intuitively, filter sets should correspond to variations in images.

If the factors of variations are restricted to some known factors such as face pose or illumination, we can use the prior knowledge to train a network to represent the variation as a feature vector.

The main difficulty is that besides the factors of variations that we have already known, there are also a number of them that we are not aware of.

Therefore, it is difficult to enumerate all the factors of variations and learn the mapping in a supervised manner.

The second challenge is that how to map a feature vector to a set of new filters.

Due to the high dimension of the filters, a direct mapping needs a large number of parameters which can be infeasible in real applications.

In response, we propose to use an autoencoder for variation representation leaning.

Since the objective of an autoencoder is to reconstruct the input images from internal feature representations, each layer of the encoder contains sufficient information about the input image.

Therefore, we extract features from each layer in the encoder as sample-specific features.

For the generation of filters , given a sample-specific feature vector, we firstly construct a filter repository.

Then we learn a matrix that maps the feature vector to a set of coefficients which will be used to linearly combine the base filters in the repository to generate new filters.

Our model has several elements of interest.

Firstly, our model bridges the gap between the autoencoder network and the prediction network by mapping the autoencoder features to the filters in the prediction network.

Therefore, we embed the knowledge from unsupervised learning to supervised learning.

Secondly, instead of generating new filters directly from a feature vector, we facilitate the generation with a filter repository which stores a small number of base filters.

Thirdly, we use linear combination of the base filters in the repository to generate new filters.

It can be easily implemented as a convolution operation so that the whole pipeline is differentiable with respect to the model parameters.

The essential part of the proposed method is the dynamical change of the parameters of a CNN.

In general, there are two ways to achieve the goal including dynamically changing the connection and dynamically generating the weights, both of which are related to our work.

In this section, we will give a brief review of the works from these two aspects.

There are several works in which only a subset of the connections in a CNN are activated in a forward pass.

We term this kind of strategy dynamic connection.

Since the activation of connections depends on input images, researchers try to find an efficient way to select subsets of the connections.

The benefit of using dynamical connection is the reduction in computation cost.

BID13 propose a conditional convlutional neural network to handle multimodal face recognition.

They incorporate decision trees to dynamically select the connections so that images from different modalities activate different routes.

BID6 present deep neural decision forests that unify classification trees with representation learning functionality.

Each node of the tree performs routing decisions via a decision function.

For each route, the input images are passed through a specific set of convolutional layers.

BID5 and BID0 also propose similar frameworks for combining decision forests and deep CNNs.

Those hybrid models fuse the high representation learning capability of CNNs and the computation efficiency of decision trees.

We refer to weights that are dynamically generated as dynamic weights.

Furthermore, since the weights are the parameters of a CNN, learning to generate those weights can also be viewed as a meta learning approach.

BID1 propose to use dynamic weights in the scenario of one-shot learning.

They construct a learnet to generate the weights of another deep model from a single exemplar.

A number of factorizations of the parameters are proposed to reduce the learning difficulty.

BID4 present hypernetworks which can also generate weights for another network, especially a deep convolutional network or a long recurrent network.

The hypernetworks can generate non-shared weights for LSTM and improve its capability of sequence modelling.

There are several other similar architectures BID11 BID3 ).Results from those works demonstrate that dynamical weights help learn feature representation more effectively.

The work that most resembles ours is the work of De BID3 .

However, our work is different in the following aspects.

(i) The feature vectors we used for filter generation are extracted from the feature maps of an autoencoder network. (ii) New filters are generated by the linear combination of base filters in a filter repository.

The rest of the paper is structured as follows.

Section 3 presents the details of the proposed method.

Section 4 shows the experiment results and Section 5 concludes the paper.

The framework of the proposed method is illustrated in FIG0 .

The description of our model will be divided into three parts, i.e. sample-specific feature learning, filter generation, and final prediction.

The framework of the proposed method.

The autoencoder network in the first row is used to extract features from the input image.

The obtained feature maps are fed to a dimension reduction module to reduce the dimension of the feature maps.

Then the reduced features are used to generate new filters in the filter generation module.

Finally, the prediction network takes in the same input image and the generated filters to make the final prediction for high level tasks such as detection, classification and so on.

"*" indicates the convolution operation.

It is difficult to quantify variations in an image sample.

Thus, we adopt an autoencoder to learn sample-specific features.

Typically, an autoender consists of an encoder and a decoder.

The encoder extracts features from the input data layer by layer while the decoder plays the role of image reconstruction.

Therefore, we use the features from each layer of the encoder as representations of the input image.

Since the feature maps from the encoder are three-dimensional, we use dimension reduction modules to reduce the dimension of the feature maps.

For each dimension reduction module, there are several convolutional layers with stride larger than 1 to reduce the spatial size of the feature maps to 1 × 1.

After dimension reduction, we obtained the sample-specific features at different levels.

The loss function for the autoencoder network is the binary cross entropy loss DISPLAYFORM0 N pix is the number of pixels in the image.

o i is the value of the ith element in the reconstructed image and t i is the value of the ith element in the input image.

Both the input image and the output image are normalized to [0, 1].

The filter generation process is shown in FIG1 .

The input to the filter generation module is the sample-specific feature vector and the output is a set of the generated filters.

If we ignore the bias term, a filter can be flatten to a vector.

Given an input feature vector, the naive way to generate filters is to use a fully connected layer to directly map the input vector to the filters.

However, it is infeasible when the number of filters is large.

Let the length of each filter be L k and the length of the feature vector be L f .

If we need to generate N filter vectors from the feature vector.

We need DISPLAYFORM0 In order to tackle the problem, we refactor each filter vector k i as DISPLAYFORM1 w j is the coefficient of the base filter b j which is from a filter repository.

M is the number of filters in the filter repository.

Equation 2 assumes that each filter vector can be generated by a set of base filters.

The assumption holds true if M = L K and those base filters are orthogonal.

However, in real applications of CNNs, each convolutional layer has limited number of filters which indicates that compared to the large dimension of the filter vector space, only a small subspace is used in the final trained model.

Based on this observation, we set M << L k in this work.

The total number of parameters in the transformation matrix is N ×L f ×M which is much smaller than the original size.

The filters in the repositories are orthogonally initialized and optimized during the training process.

The prediction network is the network for the high level task, such as image classification, recognition, detection and so on.

The filters used in the prediction network are provided by the filter generation module while the weights of the classifier in the prediction network are learned during the training process.

Loss functions for high level tasks are task-dependent.

In this work, we will use classification task for demonstration and the loss is the negative log likelihood loss DISPLAYFORM0 where t is the image label and p t is the softmax probability of the tth label.

Therefore, the entire loss function for training our model is DISPLAYFORM1

The proposed method aims for generating dynamic filters to deal with variations and improve the performance of a baseline network.

In the following experiments, we evaluate our method on three tasks, i.e. digit classification on MNIST dataset(Section 4.1), facial landmark detection on MTFL dataset (Section 4.1) and image classification on CIFAR10 dataset (Section 4.3).

The number of the base filters in each filter repository is the same as the number of the filters in each layer of the prediction network if not specified.

We will also present further analysis on the generated filters in Section 4.4.

Details of all network structures are given in Appendix A.1.

To begin our evaluation, we firstly set up a simple experiment on digit classification using MNIST dataset BID8 ).

We will show the accuracy improvement brought by our dynamic filters by comparing the performance difference of a baseline network with and without our dynamic filters.

We will also analyze how the size of the encoder network and the size of the filter repository (the number of filters in the repository) effect the accuracy of digit classification.

The baseline model used in this experiment is a small network with two convolutional layers followed by a fully connected layer that outputs ten dimensions.

For simplicity, we only use five filters in each convolutional layer.

Details of the network structures are shown in Appendix A.1.1.

To evaluate the effect of the size of the encoder network, we compare the classification accuracy obtained when the encoder network has different numbers of filters in each layer.

Let n enc be the number of filters in each layer of the encoder network.

We choose n enc from {5, 10, 20}. We also choose different repository size s from {2, 5, 10}. In the evaluation of the effect of s, we fix n enc = 20 and we fix s = 5 to evaluate the effect of n enc .

We train this baseline model with and without filter generation for 20 epochs respectively.

We show the classification accuracy on the test set in TAB0 .

The first row shows the test accuracy after training the network for only one epoch and the second row shows the final test accuracy.

From both tables, we can find that the final test accuracy of the baseline model using our dynamically generated filters is higher than that using fixed filters.

The highest accuracy obtained by our generated filters is 99.1% while the accuracy of the fixed filters is 98.25%.Interestingly, the test accuracies after the first epoch (first row in TAB0 ) show that our dynamically generated filters help the network fit the data better than the original baseline model.

It could be attribute to the flexibility of the generated filters.

Though there are only a small number of base filters in the repository, linear combination of those base filters can provide filters that efficiently extract discriminative features from input images.

In TAB0 , when s = 5, the classification accuracy increases as encoder network has more filters.

It is straightforward because with more filters, the encoder network can better capture the variation in the input image.

So it can provide more information for the generation of filters.

Based on the observation from TAB1 , it seems that the final classification accuracy is less dependent on the repository size given n enc = 20.

In this section, we apply our filter generation to the task of facial landmark detection.

To give a more straightforward understanding of the usefulness of our filter generation, we firstly investigate the performance difference of a baseline model before and after some explicit variations are added to the dataset.

Then we show the detection performance improvement with respect to the size of the detection network.

Dataset.

MTFL dataset BID15 ) contains 10,000 face images with ground truth landmark locations.

In order to compare the performance difference of baseline models with respect to variations, we construct two datasets from the original MTFL dataset.

Rotation variation is used here since it can be easily introduced to the images by manually rotating the face images.

Dataset D-Align.

We follow BID14 to aligned all face images and crop the face region to the size of 64 × 64.Dataset D-Rot.

This dataset is constructed based on D-Align.

we randomly rotate all face images within [−45 DISPLAYFORM0 Some image samples for both datasets are shown in Appendix A.2 Figure 6 .We split both datasets into the training dataset containing 9,000 images and the test dataset containing 1,000 images.

Note that the train-test splits in D-Align and D-Rot are identical.

Models Here we train two baseline models based on UNet BID10 ).

The baseline models are M odel 32 with 32 filters in each convolutional layer and M odel 64 with 64 filters in each convolutional layer.

M odel 32 and M odel 64 share the same architecture.

Details of the network structures are shown in Appendix A.1.2.We firstly trained M odel 32 and M odel 64 on D-Align and D-Rot without our filter generation module.

Then we train them on D-Rot with our filter generation module.

For evaluation, we use two metrics here.

One is the mean error which is defined as the average landmark distance to groundtruth, normalized as percentages with respect to interocular distance (Burgos-Artizzu et al. FORMULA0 ).

The other is the maximal normalized landmark distance.

Since there are more rotation variations in D-Rot than D-Align, we can consider landmark detection task on D-Rot is more challenging than that on D-Align.

This is also proved by the increase in detection error when the dataset is switched from D-Align to D-Rot as shown in FIG2 and TAB2 .

However, when we train the same baseline model with our generated filters, the detection error decreases, compared to the same model trained on D-Rot.

There is also a large error drop in maximal detection error.

These results indicate that using filters conditioned on the input image can reduces the effect of variations in the dataset.

Comparing the averaged mean error in FIG2 and FIG2 , we find that the performance gain brought by filter generation is larger on M odel 32 than that on M odel 64 .

It could be explained by the capacity of the baseline models.

The capacity of M odel 64 is larger than that of M odel 32 .

So M odel 64 can handle more variations than M odel 32 , so the performance gain on M odel 64 is smaller.

FORMULA2 ) dataset consists of natural images with more variations.

We evaluate models on this dataset to show the effectiveness of our dynamic filter generation in this challenging scenario.

We construct a small baseline network with only four convolutional layers followed by a fully connected layer.

We train this model on CIFAR10 firstly without filter generation and then with filter generation.

We also train a VGG11 model BID12 ) on this dataset.

The results are shown in FIG3 .

From the training accuracy curves, we observe that the baseline model trained without filter generation doesn't fit the data as well as other models.

This is because there are only five layers in the network which limits the network's capacity.

When the baseline model is trained with filter generation, the model can fit the data well, reaching more than 98% training accuracy.

VGG11 also achieves high training accuracy which is not supervising since there are more layers (eleven layers) in the models.

The test accuracy curves also show the benefit of adopting our dynamic filter generation.

The baseline classification accuracy is improved by ∼1% by using filter generation and the test accuray is comparable to VGG11.Based on the above evaluations on different datasets, we claim that dynamically generated filters can help improve the performance of the baseline models.

Using linear combination of base filters from filter repositories can generate effective filters for high level tasks.

In this section, we visualize the distributions of the coefficients, the generated filters and the feature maps using MNIST dataset.

Then we conduct another experiment on CIFAR10 dataset to demonstrate that the generated filters are sample-specific.

DISPLAYFORM0 Figure 5: Visualization of the distributions of the generated coefficients, filters, and feature maps from the first (top row) and the second (bottom row) convolutional layer.

The model we used for visualization is the baseline model trained in MNIST experiment with n ae = 20 and s = 5.

TSNE BID9 ) is applied to project the high dimensional features into a two-dimensional space.

The visualization results are shown in Figure 5 .

In the first row, we show the distributions of the coefficients, the filters, and the feature maps from the first convolutional layer of the model.

We observe that the generated filters are shared by certain categories but not all the categories.

It is clear in Figure 5a that the coefficients generated by some digits are far away from those by other digits.

Nevertheless, the feature maps from the first convolution layer show some separability.

In the second row, the generated coefficients and the generated filters forms into clusters which means digits from different categories activate different filters.

This behavior makes the final feature maps more separable.

We further analyze the generated filters to show that those filters are sample-specific.

We take CI-FAR10 dataset as an example.

The model used here is the same trained model used in the CIFAR10 experiment (Section 4.3).

In this experiment, we feed a test image A to the classification network and another different image B to generate filters.

In other words, the filters that will be used in the classification network are not generated from A but from B. This time the classification accuracy of the classification network falls to 15.24%, which is nearly the random guess.

This accuracy drop demonstrates that the generated filters are sample-specific.

Filters generated from one image doesn't work on the other image.

In this paper, we propose to learn to generate filters for convolutional neural networks.

The filter generation module transforms features from an autoencoder network to sets of coefficients which are then used to linearly combine base filters in filter repositories.

Dynamic filters increase model capacity so that a small model with dynamic filters can also be competitive to a deep model.

Evaluation on three tasks show the accuracy improvement brought by our filter generation.

In this section, we show the details of the network structures used in our experiments.

When we extract sample-specific features, we directly take the convolution feature maps (before LReLU layer) from the autoencoder network as input and feed them to the dimension reduction network.

The entire process of sample-specific feature extraction is split into the autoencoder network and the dimension reduction network for the purpose of plain and straightforward illustration.

The networks used in the MNIST experiment are shown from TAB3 to TAB6 .

The networks used in the MTFL experiment are shown from TAB0 .

The networks used in the CIFAR10 experiment are shown from TAB0 A.2 IMAGE SAMPLES FROM DATASET D-Align AND DATASET D-Rot

<|TLDR|>

@highlight

dynamically generate filters conditioned on the input image for CNNs in each forward pass 