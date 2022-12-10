We apply multi-task learning to image classification tasks on MNIST-like datasets.

MNIST dataset has been referred to as the {\em drosophila} of machine learning and has been the testbed of many learning theories.

The NotMNIST dataset and the FashionMNIST dataset have been created with the MNIST dataset as reference.

In this work, we exploit these MNIST-like datasets for multi-task learning.

The datasets are pooled together for learning the parameters of joint classification networks.

Then the learned parameters are used as the initial parameters to retrain disjoint classification networks.

The baseline recognition model are all-convolution neural networks.

Without multi-task learning, the recognition accuracies for MNIST, NotMNIST and FashionMNIST are 99.56\%, 97.22\% and 94.32\% respectively.

With multi-task learning to pre-train the networks, the recognition accuracies are respectively 99.70\%, 97.46\% and 95.25\%.

The results re-affirm that multi-task learning framework, even with data with different genres, does lead to significant improvement.

Multi-task learning BID1 ) enjoys the idea of pooling information that can be learned from data collected for multiple related tasks.

Multiple sources of information can stem from multiple datasets, or even a single dataset, for multiple tasks.

In this work, we focus on the case of using multiple datasets for multiple tasks.

Namely, we use MNIST, FashionMNIST, and NotMNIST image datasets collected for digit recognition, fashion item recognition, and letter recognition, respectively.

Information sharing in multi-task training can be achieved in various formality.

For neural-network based deep learning, the sharing can happen at the input layer, the hidden layers, or the output layer.

Input-layer multi-tasking combines heterogeneous input data, hidden-layer multi-tasking shares multiple groups of hidden layer units, and output-layer multi-tasking pools multiple output groups of categories.

The implementation of a multi-task learning system depends on the data and the tasks at hand.

Multi-task learning has been successfully applied to many applications of machine learning, from natural language processing BID4 ) and speech recognition BID5 ) to computer vision ) and drug discovery BID7 ).

A recent review of multi-task learning in deep learning can be found in BID9 ).

The MNIST dataset BID6 ) consists of a training set of 60,000 images, and a test set of 10,000 images.

MNIST is often referred to as the drosophila of machine learning, as it is an ideal testbed for new machine learning theories or methods on real-world data.

BID3 0.23% BID10 0.23% BID2 0.24% 2.2 FASHIONMNIST BID14 presents the FashionMNIST dataset.

It consists of images from the assortment on Zalandos website.

As given out by name, the configuration of the FashionMNIST dataset completely parallels the configuration of the MNIST dataset.

FashionMNIST consists of a training set of 60,000 images and a test set of 10,000 images.

Each image is a 28 × 28 grayscale image associated with a label from 10 classes.

FashionMNIST poses a more challenging classification task than the MNIST digits data.

A leaderboard for FashionMNIST has been created and maintained at https://github.com/zalandoresearch/fashion-mnist.

Bulatov (2011) presents the NotMIST dataset.

NotMNIST dataset consists of more than 500k 28×28 greyscale images of English letters from A to J, including a small hand-cleaned subset and a large uncleaned subset.

From the uncleaned subset, we randomly select 60,000 examples as train set and 10,000 examples as test set.

In short, we apply multi-task learning to learn from the MNIST-like datasets to pre-train the parameters of the all-convolution neural networks for individual image recognition tasks.

The overall framework is depicted in FIG0 .

DISPLAYFORM0 where the first 2 subscripts index position in the map, and the third subscript indexed the channel.

If a convolution operation with the same stride were applied to f , we would have a tensor c with DISPLAYFORM1 where t is the convolution kernel tensor, and σ(·) is an activation function.

Thus, a pooling operation can be seen as a convolution operation with uniform kernel tensor and with L p -norm as the activation function.

The architecture of an all-convolution neural network for a single task is shown in Figure 2 .

The multi-task learning classifier has the same architecture as a single-task classifier except that the width of the output layer is proportional to the number of tasks.

The target label is enhanced accordingly Each network is trained with 50 epochs.

A two-stage learning rate decay scheme is implemented.

The initial learning rate is 10 −3 for the first stage of 25 epochs, and 10 −5 for the second stage of 25 epochs.

When training a model, it is often recommended to lower the learning rate as the training progresses.

So, we let learning rate decay in each epochs.

The decay rate is 1 1+d×n where d is the learning rate set in the begin divided by 25.

n is the number of current epochs.

FIG2 shows the learning rate scheme.

The size of a mini-batch is set to 100, and the Adam optimizer is used.

When multi-task learning is complete, the parameters in the network is used to initialize single-task classifiers.

FIG3 illustrates how parameters are handed over.

The single-task classifiers are then re-trained to perform their respective classification tasks.

The experimental results are summarized in TAB3 .

We are happy to see that multi-task learning works, even though the MNIST, NotMNIST, and FashionMNIST datasets are images from totally different classes.

The bi-task learning systems are always better than the single-task systems.

Furthermore, the tri-task learning systems are the best, except for the MNIST (0.01% difference).

The relative reduction in error rates by tri-task learning are respectively 31.8% for MNIST (99.56% to 99.70%), 16.4% for FashionMNIST (94.32% to 95.25%), and 8.6% for NotMNIST (97.22% to 97.46%).

The results confirm that multi-task learning is able to learn representation which is universal and robust to different tasks.

To better understand the effect of multi-task learning, we plot the distribution of the high dimensional-data of the classes by the t-distributed stochastic neighbor embedding (t-SNE) BID12 ).

Figure 6 shows the t-SNE of the output values of several hidden layers of the classifiers.

The separation between the learned class manifolds appears to increase with the integration of multi-task learning.

The representation learned with multi-task learning in the loop looks better indeed.

In this paper, we use multi-task learning in pre-training an all-convolution neural network model.

We pass the parameters of trained multi-task models to single-task models.

Evaluation on MNISTlike datasets show that using multi-task learning can improve image recognition accuracy.

The more data we use, the better results we get.

This agrees with statistical learning theory that using more data reduces the generalization gap, thus improving test set performance, even if the data comes from a different domain.

The classification tasks of the images of digits, letters, and fashion items share parts of their hierarchical representations.

By multi-task learning, it is possible to make such common representation robust to help individual classification tasks.

Figure 6: Visualization of data manifolds with t-SNE.

The left column is the case with multi-task learning, and the right column is the case without multi-task learning.

@highlight

multi-task learning works 

@highlight

This paper presents a multi-task neural network for classification on MNIST-like datasets