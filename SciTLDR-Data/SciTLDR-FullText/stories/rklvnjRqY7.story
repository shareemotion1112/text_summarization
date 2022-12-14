Real world images often contain large amounts of private / sensitive information that should be carefully protected without reducing their utilities.

In this paper, we propose a privacy-preserving deep learning framework with a learnable ob- fuscator for the image classification task.

Our framework consists of three mod- els: learnable obfuscator, classifier and reconstructor.

The learnable obfuscator is used to remove the sensitive information in the images and extract the feature maps from them.

The reconstructor plays the role as an attacker, which tries to recover the image from the feature maps extracted by the obfuscator.

In order to best protect users’ privacy in images, we design an adversarial training methodol- ogy for our framework to optimize the obfuscator.

Through extensive evaluations on real world datasets, both the numerical metrics and the visualization results demonstrate that our framework is qualified to protect users’ privacy and achieve a relatively high accuracy on the image classification task.

In the past few years, deep neural networks (DNNs) have achieved great breakthroughs in computer vision, speech recognition and many other areas.

To support the training of DNNs, large datasets have been collected, e.g., ImageNet BID6 , MNIST (LeCun et al., 1998) and CIFAR-10/CIFAR-100 BID15 ) as image datasets, Youtube-8M (Abu-El-Haija et al., 2016) as video datasets, and AudioSet BID8 as audio datasets.

These datasets are usually crowdsourced from the real world, and may carry sensitive private information, thus, leading to serious privacy problems.

The new European Union's General Data Protection Regulation (GDPR) (Regulation, 2016) stipulates that personal data cannot be stored for long periods of time, and personal data requests, such as deleting personal images, should be handled within 30 days.

In other words, this regulation prevents long-term storage of video/image data (e.g., from CCTV cameras), which hinders the collection of real-world datasets for training deep learning models.

However, the data storage limitations do not apply if the data is anonymized.

This regulation considers the trade-off between the utility and the privacy of the data.

However, 30 days may not be a long enough period to collect image data and train a complex deep learning model, and deletion of data hinders re-training later when the model structure is updated or more data becomes available.

GPDR allows anonymized data to be stored indefinitely, which inspires us to design a framework where an image is converted into an obfuscated intermediate representation that removes sensitive personal information while retaining suitable discriminative features for the learning task.

Thus the obfuscated intermediate representation can be stored indefinitely for model training in compliance with GDPR.

Contributions In this paper, we design a obfuscator-adversary framework to obtain a trainable obfuscator that fulfills the dual goals of removing sensitive information and extracting useful features for the learning task.

Here, we mainly focus on image classification as the learning task, since it is a more general task in computer vision -the framework could be extended to other tasks.

Our framework consists of three models, each with its own objective: the obfuscator, the classifier and the reconstructor, shown in Figure 1 .

The obfuscator works as an information remover, which takes the input image and extracts feature maps that carry enough primary information for the classification task while removing sensitive private information.

These feature maps are the obfuscated representation of the input image.

The classifier uses the obfuscated representation to perform classification of the input image.

Finally, the reconstructor plays the role as an adversary whose goal is to extract the sensitive information from the obfuscated representation.

Privacy Attack:Step 1:

Step 2:Users' Feature Maps

Attacker's Eavesdropped image Figure 1 : (top) Our proposed framework learns an obfuscated representation (feature maps) for image classification that also prevents leakage of users' privacy.

The obfuscator extracts a feature map (the obfuscated representation) that both prevents the reconstruction of the image and keeps the primary information for the classification task.

The classifier uses the obfuscated feature map to perform the image classification task.

The reconstructor aims to reconstruct the original image from the feature map.

The three models are trained using an adversarial training process. (bottom) The attacker aims to reconstruct a users' images to eavesdrop their privacy.

We assume that the attacker has unlimited access to the obfuscator and the feature maps extracted from users' images.

The attacker trains their own reconstructor using their own set of images, and attempts to reconstruct the users' images from the stored feature maps.

As different kinds of images may contain different kinds of sensitive information (e.g., personal identity, location, etc), we choose image reconstruction quality as a general measure for privacy preservation.

The reconstructor, as the adversary, tries to reveal the sensitive information by restoring the image from the feature maps.

If even state-of-the-art reconstructors cannot restore the image, and the classification accuracy is still good, we can say that our framework has experimentally demonstrated enough security to protect users' privacy.

As the obfuscator and the reconstructor have opposite objectives, the training of our proposed framework can be formalized as an adversarial training paradigm.

The main contributions of this paper are: 1) To the best of our knowledge, this is the first study of using the adversarial training methodology for privacy-preserving image classification.2) We propose a brute-force experimental evaluation method to demonstrate the security-level performance of the proposed framework.3) The experiments on real-world datasets demonstrate that utility(classification accuracy)-privacy trade-off is perfectly handled via the adversarial training process.

Deep learning requires a tremendous amount of data that may contain a significnat private information.

Conventional works have already proposed several approaches to counter the privacy problem in learning tasks.

Prior works can be divided into three categories: privacy of datasets, privacy of models, and privacy of models' outputs BID26 .

In this paper, we mainly focus on the privacy of datasets.

One way to protect the privacy of data is to increase the amount of uncertainty, e.g., based on kanonymity BID28 , l-diversity BID20 and t-closeness BID18 .

Unfortunately, these approaches are only suitable for low-dimensional data because the quasi-identifiers and sensitive attributes are not easily defined for high-dimensional data.

This makes private information in multimedia (videos, images and audios, etc.) much harder to be protected.

Differential privacy BID7 , as the state-of-the-art privacy preserving mechanism, is a more formal way to open-source a database while keeping all individual records private by adding welldesigned noise.

However, differential privacy only affects inserting and deleting an individual data record.

BID0 investigated the application of differential privacy to deep learning, and extended the conventional Stochastic Gradient Descent (SGD) BID3 algorithm to a novel Differentially Private SGD (DPSGD) algorithm.

However, the inherent character of differential privacy implies that there will always be a data utility and privacy tradeoff.

The fact that more strict privacy guarantee always demands more noise added to the data often limits its application scenarios, especially when high accuracy of learning tasks is a must.

Another way for data-level privacy protection is to use cryptographic operations to encrypt the dataset.

Gilad-Bachrach et al. FORMULA0 proposed Cryptonets, a cloud based framework, in which the inference stage is applied on encrypted datum.

However, Cryptonets has some limitations.

First, it has a sensitive privacy-utility trade-off.

Second, low-degree polynomials using homomorphic encryption are not able to compute the non-linear activation function efficiently.

Focusing on these shortcomings, Rouhani et al. FORMULA0 proposed DeepSecure, a provably-secure framework for scalable deep learning based data analysis.

DeepSecure is also a cloud-client based framework, and it does not have the concern of privacy-utility trade-off.

However, this approach is only suitable for scenarios in which the number of samples submitted by each client is less than 2600, which extremely limits its application.

Other applications of homomorphic encryption to privacy preserving tasks, e.g., BID4 , BID2 and BID19 , have almost the same disadvantages and limitations as approaches mentioned above.

Recent works extend the common deep neural networks to protect the dataset privacy using pure machine learning techniques.

BID21 proposed a client-server model, which separates the common CNN into two parts and the first part becomes the feature extractor and the second part works as the classifier.

A Siamese network is used to ensure privacy protection.

However, this framework can only be deployed during the inference stage because the training of a neural network would require a large amount of communication throughput between the clients and servers.

BID17 uses the reconstruction quality as a measure for privacy preservation.

However, the reconstruction quality is only used for evaluation, and it not used in the loss function during training, which makes this work similar to that of BID21 .In contrast to these previous works, we propose a privacy protection framework based on an adversarial training procedure, where the obfuscator and classifier work together to preserve privacy while performing the classification task, and an adversarial reconstructor tries to reveal the private information by recovering the image.

As good reconstruction quality is highly related with the recovery of private information, in our framework, we include the reconstruction quality into the loss of the framework in order to better learn the obfuscator.

Experimental results demonstrate that our framework both preserves privacy well and achieves good classification accuracy.

the reconstructor, can somehow also be leveraged by an attacker.

However, compared to the attack domain of CIA, threats of other attacks that focus solely on the training data domain appear less imminent and hence those attacks are not covered in this paper.

We further assume within the attack model of CIA, an attacker would subsequently use a DNN to carry out image reconstruction due to the fact that the obfuscated feature maps may contain information unnoticed by humans.

For quality assessment of the reconstructed images in CIA, it is extremely difficult to directly define how successful an attack can be given the specific image because sensitive information contained within different images may vary from case to case.

To this end, we adopt the index measuring the quality of reconstruction, i.e., Peak Signal to Noise Ratio (PSNR), to roughly evaluate the strength of security against CIA in our design rather than define our own privacy measurement index.

We say the lower the PSNR value, the more secure it indicates that our design is able to defend against such attack model of CIA.

The PSNR between two images I o and I r with dimensions m×n is DISPLAYFORM0 where p max is the maximum range of pixel values in an image (typically 255 for 8-bit images).

Higher values of PSNR indicate that the two images are more similar.

In this section, we introduce the our proposed framework deep learning based privacy-preserving image classification.

Our approach is divided into three modules: the obfuscator, the classifier and the reconstructor.

The goal of the obfuscator is to produce a feature map that removes sensitive information from the image, while also preserving primary information for the classifier.

On the opposite, the reconstructor acts as an attacker that aims to reconstruct the original image from the feature map.

We formulate the training of our proposed framework as an adversarial training process.

Here we give the following notations.

Let D = {I i , . . .

I N } denote the images in our dataset, where N is the number of images, and Y = {y 1 , . . .

, y N } are the corresponding class labels, where the set of possible classes is Y = {1, . . .

, M }.

An obfuscator f (·; θ f ) : I → F is a function mapping from images I to feature maps F. θ f are the parameters (weights) of the obfuscator.

The classifier g(·; θ g ) : F → Y represents a mapping from feature maps F to class labels Y. The reconstructor r(·; θ r ) : F → I is a function mapping from feature maps back to images.

Conventional deep learning models for the image classification are usually based on convolutional neural networks (CNNs), which is a stack of multiple convolutional layers, pooling layers, activation functions and fully connected layers.

An intrinsic characteristic of the convolutional layers in CNNs is the ability to extract discriminative information from the input image into feature maps, while ignoring non-discriminative information.

This phenomenon inspires us to modify the objective of the convolutional layers to both extract discriminative features and remove sensitive information in the extracted feature maps.

Thus, in our framework, we divide a deep CNN architecture, VGG16 BID27 , into two parts.

The first part is used as the obfuscator, while the second part is used as the classifier.

The feature map between the two parts is the obfuscated representation.

FIG0 shows the structure of the obfuscator and the classifier.

As an feature extractor and sensitive information filter, the obfuscator has two objectives.

First, it should minimize the classification error to ensure the high utility of our framework.

Second, to protect the privacy in input images, it should minimize the PSNR between the original image and the reconstructed image in (1).

The goal of the classifier is to minimize the classification error, which is consistent with the obfuscator.

Hence, the obfuscator and classifier can be trained by minimizing the loss function, DISPLAYFORM0 where λ is a trade-off parameter.

The first term is the categorical cross-entropy loss between the ground-truth class label and the classifier prediction.

The second term is the reconstruction loss, based on the output of the reconstructor.

The loss in Eq.2 is the adversarial loss of our framework.

The reconstructor in our framework plays the role of an attacker.

According to the assumptions in Section 3, the attacker can access the feature maps of raw images extracted by a pre-trained obfuscator.

The attacker's goal is to recover private information stored in the feature maps through reconstruction of the image from the feature map.

Consequently, the objective of the reconstructor is to maximize the similarity between reconstructed images and raw images, as measured by PSNR, DISPLAYFORM0 which is the opposite objective of the loss function in (2).

The architecture of the reconstructors is discussed in the next section.

Intuitively, the roles of the obfuscator and the reconstructor are working against each other.

During the training period, the obfuscator tries its best to maximally remove the sensitive information in the input images so that the reconstructor cannot reconstruct images similar to raw input images.

Whereas the reconstructor will fine-tune its parameters to find the best reconstructions for given input feature maps.

This training formulation is exactly an adversarial training process, where two models play a minimax game BID5 .

The adversarial training methodology introduces an rebuttal procedure in the training process.

We formalize the training procedure in Algorithm 1.

The training set of images D = {I 1 , ..., I N } and their class labels Y = {y 1 , ..., y n }, number of main iterations T main , number of sub-iterations T sub Result: Weights of three models: θ f , θ g , θ r such that the given three objects are optimized Initialization: Initialize θ f , θ g , θ r using Xaiver initialization BID10 ; while not converged or reached T main do Generate augmented data from input images; if is the first epoch then train the obfuscator-classifier until it reaches its optimal performance (at least 200 epochs); else train the obfuscator-classifier for T sub epochs; end freeze the obfuscator, then train the reconstructor for T sub epochs; freeze the reconstructor and classifier, then train the obfuscator for T sub epochs; end Algorithm 1: Adversarial training algorithm for our frameworkAs the primary task of our framework is to classify images, we first train the classifier and obfuscator together without any security concern to obtain optimal performance at image classification.

The obfuscator working in this stage can be recognized as the first several layers of the classifier.

After the initial training of the obfuscator-classifier, we need to take the privacy problem into consideration, which is handled by our adversarial training framework.

In particular, the reconstructor is trained while holding the obfuscator fixed, and then the obfuscator is trained while keeping the reconstructor and classifier fixed.

In this way, the obfuscator can counteract the any improvements in PSNR from the reconstructor.

Finally, the whole procedure is repeated until convergence or the maximum number of epochs is reached.

In our experiments, we use three datasets: MNIST handwritten digits dataset BID16 and CIFAR-10/CIFAR-100 dataset BID15 .

MNIST consists of 70,000 handwritten digit images, of which 60,000 images belong to the training set, and 10,000 images belong to the testing set.

All these images are size-normalized and centered to a fixed-size (28 × 28).

CIFAR-10 is a tiny image dataset containing 60,000 32 × 32 colored images in 10 classes (50,000 for training and 10,000 for testing).

CIFAR-100 is similar to CIFAR-10 but contains 100 classes with 600 images per class.

In order to ensure the security under different kinds of attackers, we implement 4 state-of-the-art reconstructors as attackers, and train them separately within our framework.

Note that for space interest, we only report the results where we set trade-off parameter λ = 1 in the experiment.

We will include more enriched results with diverse choice of parameters in the full version.

The four reconstructors are (see TAB1 for architecture details): 1) Simple autoencoder reconstructor (SimRec): This is a simple reconstructor, which is just a reversed model of the obfuscator used in our framework.

As the structure of the obfuscator-reconstructor is similar to the structure of an autoencoder; 2) U-net reconstructor #1 (URec#1): U-net BID24 ) is a fully convolutional neural network structure used for biomedical image segmentation and image generation with GANs Isola et al. FORMULA0 ; 2) U-net reconstructor #2 (URec#2): This reconstructor is a simpler version of URec#1, which reduces the number of layers; 3) ResNet reconstructor (ResRec): ResNet BID12 is a deep learning model used for image recognition, as well as image restoration BID14 .

ResNet model involves the residual function and contains several residual blocks.

In this section, we will show the experimental results to demonstrate that our framework is a strong method to protect user's privacy while keeping the utility of a deep learning image classification model.

TAB2 presents the classification accuracy for each dataset, related to the number of epochs.

As the space is limited, here we only give the accuracy of the first 100 epochs.

During the training and testing process, we find that the accuracy is not highly related to the reconstructor, which means that although we changed the reconstructor in our framework, the classification accuracy is consistent (as shown in FIG1 ).

This suggests that the obfuscator is robust to the type of the reconstructor, so that our framework is able to be deployed in different scenarios.

In TAB2 , the accuracy of the first epoch represents the baseline of VGG16 on the given dataset.

The baseline for MNIST is 99.70% and our accuracy using privacy-preservation is 92.95%.

For CIFAR-10 it is 93.56% and 89.48%, and for CIFAR-100 it is 70.40% and 64.53%.

For comparison, the state-of-the-art work in privacy-preserving networks BID17 achieved only 70.1% average accuracy on CIFAR-10 dataset, which our framework outperforms.

The major difference between our work and BID17 is that we employ adversarial training where the goal is to both reduce reconstruction quality and improve classification accuracy.

Figure 4 gives some examples of the reconstruction results on CIFAR-10 using different reconstructors.

Comparing the reconstructed images with raw images, we find that the reconstructed images only contain the blurred outlines of target objects in raw images, which may represent the category information of the image.

The average PSNR values for the reconstructors on all datasets is shown in the diagonal of TAB3 -SimRec achieved 28.0306 as its average PSNR, while URec#1, URec#2 and ResRec, the PSNR values are 28.0020, 28.0129 and 28.0176, respectively.

Small values of PSNR indicate that the difference between the input image and the reconstructed image is large, and hence most of the information of the raw image is removed by the obfuscator.

In order to simulate the behavior of the attackers, besides the adversarial training and testing, we designed a complementary experiment that simulates a brute-force attack.

In this experiment, we assume that the attacker has a pre-trained obfuscator, and then trains multiple reconstructors to try to recover sensitive information from a given feature map.

During the training process, the obfuscator is not modified and only the reconstructors' weights are updated (simulating the situation that attackers wants to break the obfuscator).

The off-diagonal entries of TAB3 show the reconstruction PSNR when the attacker uses a different reconstruction method than the one used for training, The PSNR values are also low and comparable to the reconstructor used for training the obfuscator.

This suggests that the obfuscator is able to remove most of the sensitive information from the feature map, and that it is robust against different types of attackers on which it was not trained.

Finally, to demonstrate that adversarial training is useful, we train a variant of our framework that removes the reconstructor and the adversarial training process, and the PSNR loss term in Eq.2 is also removed.

Reconstruction results are shown in Figure 5 .

Compared to our method, the reconstructed images without the adversarial training have more detailed information about raw images, and thus allow more private information to be leaked.

Thus adversarial training plays an important role in learning the obfuscator.

We proposed a deep learning framework on privacy-preserving image classification tasks.

Our framework has three modules, the obfuscator, classifier, and reconstructor.

The obfuscator works as an feature extractor and sensitive information remover to protect users' privacy without decreasing the accuracy of the classifier.

The reconstructor is an attacker, and has an opposite objective to reveal the sensitive information.

Based on this antagonism, we designed an adversarial training methodology.

Experiments showed our framework is qualified to protect users' privacy and achieve a relatively high accuracy on the image classification task.

@highlight

We proposed a novel deep learning image classification framework that can both accurately classify images and protect users' privacy.

@highlight

This paper proposes a framework which preserves the private information in the image and doesn’t compromise the usability of the image.

@highlight

This current work suggests using adversarial networks to obfuscate images and thus allow collecting them without privacy concerns to use them for training machine learning models.