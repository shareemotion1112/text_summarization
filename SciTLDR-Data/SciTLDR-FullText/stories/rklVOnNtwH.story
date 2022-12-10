In this paper, we tackle the problem of detecting samples that are not drawn from the training distribution, i.e., out-of-distribution (OOD) samples, in classification.

Many previous studies have attempted to solve this problem by regarding samples with low classification confidence as OOD examples using deep neural networks (DNNs).

However, on difficult datasets or models with low classification ability, these methods incorrectly regard in-distribution samples close to the decision boundary as OOD samples.

This problem arises because their approaches use only the features close to the output layer and disregard the uncertainty of the features.

Therefore, we propose a method that extracts the uncertainties of features in each layer of DNNs using a reparameterization trick and combines them.

In experiments, our method outperforms the existing methods by a large margin, achieving state-of-the-art detection performance on several datasets and classification models.

For example, our method increases the AUROC score of prior work (83.8%) to 99.8% in DenseNet on the CIFAR-100 and Tiny-ImageNet datasets.

Deep neural networks (DNNs) have achieved high performance in many classification tasks such as image classification (Krizhevsky et al., 2012; Simonyan & Zisserman, 2014) , object detection (Lin et al., 2017; Redmon & Farhadi, 2018) , and speech recognition Hannun et al., 2014) .

However, DNNs tend to make high confidence predictions even for samples that are not drawn from the training distribution, i.e., out-of-distribution (OOD) samples (Hendrycks & Gimpel, 2016) .

Such errors can be harmful to medical diagnosis and automated driving.

Because it is not generally possible to control the test data distribution in real-world applications, OOD samples are inevitably included in this distribution.

Therefore, detecting OOD samples is important for ensuring the safety of an artificial intelligence system (Amodei et al., 2016) .

There have been many previous studies (Hendrycks & Gimpel, 2016; Liang et al., 2017; Lee et al., 2017; DeVries & Taylor, 2018; Lee et al., 2018; Hendrycks et al., 2018) that have attempted to solve this problem by regarding samples that are difficult to classify or samples with low classification confidence as OOD examples using DNNs.

Their approaches work well and they are computationally efficient.

The limitation of these studies is that, when using difficult datasets or models with low classification ability, the confidence of inputs will be low, even if the inputs are in-distribution samples.

Therefore, these methods incorrectly regard such in-distribution samples as OOD samples, which results in their poor detection performance (Malinin & Gales, 2018) , as shown in Figure 1 .

One cause of the abovementioned problem is that their approaches use only the features close to the output layer and the features are strongly related to the classification accuracy.

Therefore, we use not only the features close to the output layer but also the features close to the input layer.

We hypothesize that the uncertainties of the features close to the input layer are the uncertainties of the feature extraction and are effective for detecting OOD samples.

For example, when using convolutional neural networks (CNNs), the filters of the convolutional layer close to the input layer extract features such as edges that are useful for in-distribution classification.

In other words, indistribution samples possess more features that convolutional filters react to than OOD samples.

Therefore, the uncertainties of the features will be larger when the inputs are in-distribution samples.

Another cause of the abovementioned problem is that their approaches disregard the uncertainty of the features close to the output layer.

We hypothesize that the uncertainties of the latent features close Baseline (Hendrycks & Gimpel, 2016) UFEL (ours) max softmax probability Baseline UFEL (ours) degree of uncertainty Figure 1 : Comparison of existing and proposed methods.

We visualized scatter plots of the outputs of the penultimate layer of a CNN that can estimate the uncertainties of latent features using the SVHN dataset (Netzer et al., 2011) .

We used only classes 0, 1, and 2 for the training data.

Classes 0, 1, 2, and OOD, indicated by red, yellow, blue, and black, respectively, were used for the validation data.

We plot the contour of the maximum output of the softmax layer of the model.

Left: Because the image of "204" includes the digits "2" and "0," the maximum value of the softmax output decreases because the model does not know to which class the image belongs.

Right: The sizes of points in the scatter plots indicate the value of the combined uncertainties of features.

We can classify the image of "204" as an in-distribution image according to the value of the combined uncertainties.

to the output layer are the uncertainties of classification and are also effective for detecting OOD samples.

For example, in-distribution samples are embedded in the feature space close to the output layer to classify samples.

In contrast, OOD samples have no fixed regions for embedding.

Therefore, the uncertainties of the features of OOD samples will be larger than those of in-distribution samples.

Based on the hypotheses, we propose a method that extracts the Uncertainties of Features in Each Layer (UFEL) and combines them for detecting OOD samples.

Each uncertainty is easily estimated after training the discriminative model by computing the mean and the variance of their features using a reparameterization trick such as the variational autoencoder (Kingma & Welling, 2013) and variational information bottleneck (Alemi et al., 2016; .

Our proposal is agnostic to the model architecture and can be easily combined with any regular architecture with minimum modifications.

We visualize the maximum values of output probability and the combined uncertainties of the latent features in the feature space of the penultimate layer in Figure 1 .

The combined uncertainties of the features discriminate the in-distribution and OOD images that are difficult to classify.

For example, although the images that are surrounded by the red line are in-distribution samples, they have low maximum softmax probabilities and could be regarded as OOD samples in prior work.

Meanwhile, their uncertainties are smaller than those of OOD samples and they are regarded as in-distribution samples in our method.

In experiments, we validate the hypothesis demonstrating that each uncertainty is effective for detecting OOD examples.

We also demonstrate that UFEL can obtain state-of-the-art performance in several datasets including CIFAR-100, which is difficult to classify, and models including LeNet5 with low classification ability.

Moreover, UFEL is robust to hyperparameters such as the number of in-distribution classes and the validation dataset.

Methods based on the classification confidence Hendrycks & Gimpel (2016) proposed the baseline method to detect OOD samples without the need to further re-train and change the structure of the model.

They define low-maximum softmax probabilities as indicating the low confidence of in-distribution examples and detect OOD samples using the softmax outputs of a pre-trained deep classifier.

Building on this work, many models have recently been proposed.

Liang et al. (2017) proposed ODIN, a calibration technique that uses temperature scaling (Guo et al., 2017) in the

Figure 2: Network structure of UFEL when using DenseNet.

Black arrow: Extracting the variance of latent features using the reparameterization trick.

Blue arrow: Combining these features.

softmax function and adds small controlled perturbations to the inputs to widen the gap between indistribution and OOD features, which improves the performance of the baseline method.

Likewise, Lee et al. (2018) ; DeVries & Taylor (2018); Lee et al. (2017) ; Hendrycks et al. (2018) also extended the baseline method.

Like Hendrycks & Gimpel (2016) , we use the feature of maximum softmax probability as one of our features.

Methods based on the uncertainty Malinin & Gales (2018) attempted to solve the problem of classifying in-distribution samples close to the decision boundary as OOD samples by distinguishing between data uncertainty and distributional uncertainty.

Data uncertainty, or aleatoric uncertainty (Kendall & Gal, 2017) , is irreducible uncertainty such as class overlap, whereas distributional uncertainty arises because of the mismatch between training and testing distributions.

They argue that the value of distributional uncertainty depends on the difference in the Dirichlet distribution of the categorical parameter.

Further, they estimate the parameter of the Dirichlet distribution using a DNN and train the model with in-distribution and OOD datasets.

The motivation for our work is similar to that of Malinin & Gales (2018) .

In our work, the distribution of the logit of the categorical parameters is modeled as a Gaussian distribution, which enables us to train the model without an OOD dataset.

Furthermore, we estimate the parameters of the Gaussian distribution of latent features close to the input layer.

In this section, we present UFEL, which extracts the uncertainties of features in each layer and combines them for detecting OOD samples.

First, we use the maximum of the softmax output, as in Hendrycks & Gimpel (2016) , as one of our features.

Second, we also use the distribution of the categorical parameter, as in Malinin & Gales (2018) , using the uncertainty of logits.

Furthermore, we use the uncertainty of the feature extraction extracted from the latent space close to the input layer because they will not be relevant to the classification accuracy.

We probabilistically model the values of these features, estimate their uncertainties, and combine them.

Let x ∈ X be an input, y ∈ Y = {1, · · · , K} be its label, and l ∈ {1, · · · , L} be the index of the block layers.

The objective function of normal deep classification is as follows:

where p(x, y) is the empirical data distribution, L is a cross entropy loss function, and f φ is a DNN.

We use the following notation Figure 2 .

To extract the uncertainties of features in each layer, we model the lth block layer's output z l as a Gaussian whose parameters depend on the l-1th block layer's output z l−1 as follows:

, where f φ l is the lth block layer, which outputs both the mean µ and covariance matrix Σ. In this paper, we use a diagonal covariance matrix to reduce the model parameters.

We use the reparameterization trick (Kingma & Welling, 2013) to write z l = µ l + σ l , where

, and is the Gaussian noise.

Then, our objective function is as follows:

where z 0 = x. Because of the reparameterization trick, the loss gradient is backpropagated directly through our model, and we can train our model like the regular classification models in Equation 1.

Next, we explain the two methods of combining the features extracted in each layer.

In the first method, we sum the uncertainties of each value of the features in each layer and linearly combine them.

Because the feature maps of a convolutional block layer are three dimensional, each element is computed as z

Moreover, because the output of a fully connected layer is one dimensional, each element is formed as z

We use a weighted summation of the scale of each feature and the maximum value of the softmax scores as a final feature d LR as follows:

We choose the parameter λ l by training a logistic regression (LR) using in-distribution and OOD validation samples.

In the second method, we combine the features directly and nonlinearly using a CNN as follows:

We train the CNN parameter θ with in-distribution and OOD validation samples using binary crossentropy loss.

The detailed structures of the CNN are given in Table A .3.

We use the values of these feature d(x) to test the performance of detecting OOD samples.

In this section, we present the details of the experiments, which includes the datasets, metrics, comparison methods, and models.

Because of space limitations, more details are given in Appendix A.

Datasets We used several standard datasets for detecting OOD samples and classifying indistribution samples.

The SVHN, CIFAR-10, and CIFAR-100 datasets were used as in-distribution datasets, whereas Tiny ImageNet (TIM), LSUN, iSUN, Gaussian noise, and uniform noise were used as OOD datasets.

These data were also used in Liang et al. (2017) ; DeVries & Taylor (2018) .

We applied standard augmentation (cropping and flipping) in all experiments.

We used 5,000 validation images split from each training dataset and chose the parameter that can obtain the best accuracy in the validation dataset.

We also used 68,257 training images from the SVHN dataset and 45,000 training images from the CIFAR-10 and CIFAR-100 datasets.

All the hyperparameters of ODIN and UFEL were tuned on a separate validation set, which consists of 100 OOD images from the test dataset and 1,000 images from the in-distribution validation set.

We tuned the parameters of the CNN in Equation 4 using 50 validation training images taken from the 100 validation images.

The best parameters were chosen by validating the performance using the rest of 50 validation images.

Finally, we tested the models with a test dataset that consisted of 10,000 in-distribution images and 9,900 OOD images.

Evaluation metrics We used several standard metrics for testing the detection of OOD samples and the classification of in-distribution samples.

We used TNR at 95% TPR, AUROC, AUPR, and accuracy (ACC), which were also used in Lee et al. (2017; .

Comparison method We compare UFEL with the baseline (Hendrycks & Gimpel, 2016) and ODIN (Liang et al., 2017) methods.

For the baseline method, we used max k p(y = k|x) as the detection metric.

For ODIN, we used the same detection metric and calibrated it using temperature scaling and small perturbations to the input.

The temperature parameter T ∈ {1, 10, 100, 1000} and the perturbation parameter ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1} were chosen using the in-distribution and OOD validation datasets.

Model training details We adopted LeNet5 (LeCun et al., 1998) and two state-of-the-art models, WideResNet (He et al., 2016) and DenseNet (Huang et al., 2017) , in this experiment.

In all experiments, we used the same model and conditions to compare UFEL with existing methods.

Only the structure used to extract the variance parameters differs.

For LeNet5, we increased the number of channels of the original LeNet5 to improve accuracy.

See Table A .3 for model details.

We inserted the reparameterization trick to the second convolutional layer and the softmax layer.

LeNet5 was trained using the Adam (Kingma & Ba, 2014) optimizer for 10 epochs and the learning rate was set to 5e-4.

Both DenseNet and WideResNet were trained using stochastic gradient descent, with a Nesterov momentum of 0.9.

We inserted the reparameterization trick to the first convolutional block, the second convolutional block, and the softmax layer.

For WideResNet, we used a WideResNet with a depth of 40 and width of 4 (WRN-40-4), which was trained for 50 epochs.

The learning rate was initialized to 0.1 and reduced by a factor of 10× after the 40th epoch.

For DenseNet, we used a DenseNet with depth L = 100 (Dense-BC), growth rate of 12, and drop rate of 0.

DenseNet-BC was trained for 200 epochs with batches of 64 images, and a weight decay of 1e-4 for the CIFAR-10 and CIFAR-100 datasets.

It was trained for 10 epochs for the SVHN dataset.

The learning rate was initialized to 0.1 and reduced by a factor of 10× after the 150th epoch.

In this section, we demonstrate the performance of UFEL by conducting five experiments.

In the first experiment, we show that UFEL performs better than the baseline (Hendrycks & Gimpel, 2016) and ODIN (Liang et al., 2017 ) methods on several datasets and models.

In the second experiment, we confirm that the features of UFEL have almost no relationship with the ACC.

In the third experiment, we demonstrate that UFEL has a strong ability to detect OOD data, even if the number of classes of in-distribution data is small.

In the fourth experiment, we confirm that UFEL is robust to the number of OOD samples, and in the fifth experiment, we test the performance of UFEL on unseen OOD datasets.

The objective of these experiments is to show the uncertainties of the features obtained in each CNN layer distinguish the in-distribution and OOD data.

Moreover, we obtain state-of-the-art performance for OOD sample detection by combining these features.

Detecting OOD samples on several datasets and models In this experiment, we evaluate the performance of OOD detection using Equation 3 and Equation 4.

In this study, var l is used to denote σ

, and UFEL (CNN) denotes d CN N in Equation 4.

We measured the detection performance using a DenseNet trained on CIFAR-100 when the iSUN dataset is used to provide the OOD images.

Table 1 shows that var 1 and var 3 are strong features that, by themselves, can outperform ODIN.

This indicates that the uncertainties of the feature extraction and classification are effective for detecting OOD samples.

Moreover, the combination of these features yields state-of-the-art performance.

In Table 2 , we demonstrate that UFEL outperforms the baseline and ODIN methods on several datasets and models.

Furthermore, UFEL is also slightly superior to them with respect to indistribution accuracy, which indicates that our model is robust to noise because of the reparameterization trick.

Here, we do not report ODIN accuracy because the model of ODIN is the same as that of the baseline.

We conducted this experiment three times and used the average of the results.

We used the CIFAR-10, CIFAR-100, and SVHN datasets as the in-distribution datasets and the other datasets as the OOD samples.

Note that our UFEL outperformed the baseline and ODIN methods by a large margin, especially when using CIFAR-100, which is difficult to classify, or LeNet5 which Relationship between the performance of detecting OOD samples and in-distribution accuracy In this experiment, we show that the features of our method are not related to the in-distribution accuracy.

We used CIFAR-10 as the in-distribution dataset and TIM as the OOD dataset.

We trained DenseNet-BC for nine epochs and tested the performance at each epoch.

As shown in Figure 3 , each variance (var l) is less related to the accuracy than the baseline and ODIN methods.

The var 1 of the feature close to the input layer has the highest ability to detect OOD samples in this experiment.

Out-of-distribution: iSUN Figure 4 : Plot of AUROC (y-axis) when changing the number of in-distribution dataset classes (xaxis).

We used SVHN as in-distribution dataset, TIM, LSUN, and iSUN as OOD datasets, and the LeNet5 model.

All plots were averaged over three runs and the error bar indicates one standard deviation.

These results also indicate that we can discriminate in-distribution and OOD examples when using a dataset that is difficult to classify.

Detecting OOD samples while changing the number of in-distribution classes In this experiment, we show that UFEL is robust to the number of class labels.

We used SVHN as indistribution dataset and changed the number of in-distribution classes in training as {0,1}, {0,1,2},..., {0,1,2,...

,9}. We also used TIM, LSUN and iSUN datasets as OOD samples, and LeNet5 as a model.

We compared the proposed method with the baseline and ODIN methods, as shown in Figure 4 .

This graph shows the AUROC score of each model when changing the number of training data classes.

As this graph shows, UFEL outperforms other methods in all cases and is robust to the number of in-distribution classes, whereas the performance of ODIN drops as the number of class labels decreases.

These results suggest that UFEL is effective for small datasets because the number of samples can be decreased to one fifth of the original number when there are two in-distribution classes and the cost of label annotation is reduced.

Detecting OOD samples while changing the number of OOD samples In this experiment, we present the performance of UFEL while changing the number of OOD validation examples.

All the hyperparameters of ODIN and UFEL were tuned on a separate validation set, which consists of 30, 50, and 100 OOD images in the test dataset and 1,000 images from the in-distribution validation set.

As shown in Figure 5 , although UFEL (CNN) outperforms other methods including UFEL (LR) in most cases, it performs worse than ODIN in part of the results because some tuning for OOD samples is needed.

Meanwhile, UFEL (LR) outperforms prior methods constantly because the number of hyperparameters is small and tuning samples are almost unneeded.

Figure 5: Plot of AUROC (y-axis) when changing the OOD dataset (x-axis).

We used CIFAR-10 and CIFAR-100 as the in-distribution dataset.

All plots are averaged over three runs and the error bar indicates one standard deviation.

Generalization to unseen OOD dataset Because OOD validation samples might not be available in practice, we used only uniform noise as the validation OOD dataset and tested the ability of our model to detect another OOD dataset.

We added a binary classification as a comparison method.

This method was trained using an in-distribution dataset (positive) and uniform noise (negative).

Table 3 shows that UFEL outperforms prior work in all cases and generalize well.

Table 3 also indicates that the binary classification method does not generalize well because it cannot distinguish in-distribution dataset and OOD datasets TIM, LSUN, and iSUN, although it can distinguish Gaussian noise, which is similar to uniform noise.

In this paper, we demonstrated that the uncertainties of features extracted in each hidden layer are important for detecting OOD samples.

We combined these uncertainties to obtain state-of-the-art OOD detection performance on several models and datasets.

The approach proposed in this paper has the potential to increase the safety of many classification systems by improving their ability to detect OOD samples.

In future work, our model could be used in an unsupervised model by training it to minimize reconstruction error, which would avoid the need to use in-distribution labels to detect OOD samples.

Furthermore, although we compared our model with ODIN, UFEL will perform better if we combine UFEL with ODIN because they are orthogonal methods.

CIFAR.

The CIFAR dataset (Krizhevsky et al., 2009 ) contains 32 × 32 natural color images.

The training set has 50,000 images and the test set has 10,000 images.

CIFAR-10 has 10 classes, whilst CIFAR-100 has 100 classes.

SVHN.

The Street View Housing Numbers (SVHN) dataset (Netzer et al., 2011) contains 32 × 32 color images of house numbers.

The training set has 604,388 images and the test set has 26,032 images.

SVHN has 10 classes comprising the digits 0-9.

TIM.

The Tiny ImageNet dataset consists of a subset of ImageNet images (Deng et al., 2009 ).

It contains 10,000 test images from 200 different classes.

We downsampled the images to 32 × 32 pixels.

LSUN.

The Large-scale Scene UNderstanding (LSUN) dataset (Yu et al., 2015) has 10,000 test images of 10 different scenes.

We downsampled the images to 32 × 32 pixels.

iSUN.

The iSUN (Xu et al., 2015) dataset consists of a subset of 8,925 SUN images.

We downsampled the images to 32 × 32 pixels.

Gaussian Noise.

The Gaussian noise dataset consists of 10,000 random two-dimensional Gaussian noise images, where each value of every pixel is sampled from an i.i.d Gaussian distribution with mean 0.5 and unit variance.

Uniform Noise.

The uniform noise dataset consists of 10,000 images, where each value of every pixel is independently sampled from a uniform distribution on [0, 1].

In channels Out channels Ksize Stride Padding  Conv2d  3  64  5  1  0  ReLU  -----MaxPool2d  ---2  -Conv2d  64  128  5  1  0  ReLU  -----MaxPool2d  ---2  -Linear  128*5*5  120  ---ReLU  -----Linear  120  84  ---ReLU  -----Linear  84  10  ---softmax -----

@highlight

We propose a method that extracts the uncertainties of features in each layer of DNNs and combines them for detecting OOD samples when solving classification tasks.