The fabrication of semiconductor involves etching process to remove selected areas from wafers.

However, the measurement of etched structure in micro-graph heavily relies on time-consuming manual routines.

Traditional image processing usually demands on large number of annotated data and the performance is still poor.

We treat this challenge as segmentation problem and use deep learning approach to detect masks of objects in etched structure of wafer.

Then, we use simple image processing to carry out automatic measurement on the objects.

We attempt Generative Adversarial Network (GAN) to generate more data to overcome the problem of very limited dataset.

We download 10 SEM (Scanning Electron Microscope) images of 4 types from Internet, based on which we carry out our experiments.

Our deep learning based method demonstrates superiority over image processing approach with  mean accuracy reaching over 96% for the measurements, compared with the ground truth.

To the best of our knowledge, it is the first time that deep learning has been applied in semiconductor industry for automatic measurement.

Typically, massive processing phases during fabrication of semiconductor can be roughly categorized into four manufacturing steps: deposition, removal, patterning, and modification.

Particularly for etching process, as one type of removal technologies, such as Reactive-ion etching (RIE) BID6 , aims to remove selected areas from the surface of wafer so that other materials can be deposited BID12 .

As the density of semiconductor device are continually increasing with miniaturization tendency in highly integrated circuits, features and size between etched structure also become dramatically smaller and leads to higher aspect ratio (means ratio of height to width), which in other words narrowing the horizontal width much faster than vertical height within the structure BID3 .

Therefore, the precision during removal and patterning phases is quite crucial for semiconductor devices to ensure ultimate quality and reliability of products.

However, how to guarantee the precision and accuracy during fabrication process remains a big challenge due to the following reasons: (1) Inaccurate measurement of critical dimensions, such as wrongly localized key points or failure of removing some material in etched process, can result in adverse consequence and impact on product yield and quality; (2) Even very slight deformation on patterns or features with irregular shape or bend line can cause some device defects, e.g., nonfunctionality or bad quality on next-generation products BID23 BID11 BID26 .Till now, although with the advent of high-resolution image using AFM (Atomic Force Microscopy) and advanced etched process like Deep Reactive-Ion Etching (DIRE) BID17 , the measurements, such as etch line, space depth, width, profile angle, etc., are still carried out manually by domain experts or process engineers.

Such process requires considerable efforts and it is timeconsuming, subjective, and very hard to reproduce BID21 .

Traditional approaches like thresholding BID24 and edge detecting BID7 are basically utilizing constraints on image intensity or object appearance which require relatively large training sets.

Consequently, automatic measurement and profile characterization of etched structures in semiconductor is highly desirable for semiconductor industry to achieve consistent, efficient, and accurate evaluation of the device quality, and at the same time reduce the demand for human beings.

With evolving of machine learning techniques, such as random forest BID22 , SVM BID8 ), AdaBoost (Lee et al., 2010 , etc., they can be widely applied in many different domains, such as clinical, object recognition, image segmentation BID15 , etc.

This paper conducts preliminary study on segmentation of silicon SEM image by using traditional machine learning approaches.

As results demonstrated in FIG0 , image analytics using the traditional methods are not so promising, which fail to detect several boundaries due to unknown variations in images.

Another challenge lies in irregular shapes resulted from unexpected variations, as the traditional image processing searches the boundaries based on predefined patterns.

Recently, explosion of interests is drawn on deep learning approaches and many state-of-the-art methods are leading the direction for automatic image segmentation and recognition in a broad fields, such as clinical like tumors detection BID2 , arts like music BID25 , nature language processing BID4 , to mention a few.

However, till now deep learning approach has not been used in etched structure detection for semiconductor wafer images yet.

Although the aforementioned deep learning network can achieve promising performance in segmentation and classification issues, it still suffers from learning abundant data without ground truth or obtaining labeled training dataset with expensive and considerable efforts.

Therefore, data augmentation is crucial to achieve excellent results by training model with desired properties and invariance with limited dataset.

To address such puzzle, BID9 designed Generative Adversarial networks (GAN) using adversarial process, which can be used to generate pseudo images to reduce demand for labeled data.

In this paper, fully convolutional network, U-Net, is adopted to the new issue on profile characterization of etched structure in semiconductor SEM image.

Problem from this domain is appropriate to select U-Net for object boundary detection as following reasons: Firstly, the dataset for neuronal structure segmentation BID19 and the SEM image used for etched structure segmentation are quite similar which are all in electron microscopic stacks.

Next, replacing fully connected layers with convolutional layers can generate seamless segmentation results with smooth lines rather than jagged line BID14 .

Third, the output images generated are with high resolution which resulting from replacement with upsampling operators, and this lays the foundation of measurement on key point sets and critical dimension of chips after segmentation.

Furthermore, data augmentation is performed by using GAN to acquire pseudo images for training.

Other approaches for increasing sample dataset involve cropping, contrast, flipping, etc.

Finally, key point localization is performed to measure critical dimensions of etched structures in wafers.

Concluded from the aforementioned literatures and studies, this paper has the following contributions:1.

Unlike the traditional boundary searching based on image processing, deep learning treats the problem as a segmentation problem; 2.

Many data augmentations have been explored, such as varying contrasts, flipping, rotation, adding salt-and-pepper, etc., in order to explore all potential variations;3.

Cross-validation methods are used to avoid any potential data-leakage, e.g., if there are three images of the same type, two out of three are used as training dataset and the remaining one are used for testing; if there is only one image for a type, the image is cropped into three smaller ones and again, two out of the three are used as training and the remaining one is used for testing;4.

GAN is also used to generate some data for training and around 0.5% improvement of pixel-wise accuracy on testing dataset has been observed.

The rest of the paper is organized as following.

In Section 2, some related work are discussed.

In section 3, we present the Network architecture of deep learning approach.

Section 4, training process and data augmentation are demonstrated.

In section 5, experiments and results are further discussed.

Finally, conclusion has been drawn in section 6.

Generally, on the basis of instruments used to scan semiconductor wafers for etched structure inspection, the technologies can be classified into three categories, which broadly are applied in practice, covering optical imaging, Scanning Electron Microscope (SEM), and Optical Microscopy.

Particularly, SEM with higher resolution comparing to the other two are able to observe geometric features and shapes even in extreme environments, such as high vacuum, low temperature, etc., so that it appears widely applied in semiconductor industry for inspection of wafers BID24 .

Conventionally, such inspection is manually done by process engineers using eyes.

Due to the drawbacks of unreliability, exhaustion, and bias from different expertise for manual inspection, many research work have been carried out for automatic inspection to reduce the traditional manual inspection using eyes BID18 BID7 BID20 BID5 .The research work on wafers inspection can be classified into two categories, direct and indirect approaches.

Direct approaches use reference images without flaws as benchmarks to compare with inspected images, to generate computed difference for defect identification, e.g., golden template using threshold for subtractive comparison, neighborhood template using dynamic reference image on account of neighboring structure BID24 .

Although direct methods are relatively fast with easy procedures, massive potential offset values can pose difficulty to adjust threshold value with appropriateness.

On the other hand, indirect approaches aim to compare two segmented images with masks to indicate defect object, instead of using reference and inspected images.

Such segmentation step to differentiate insulator and conductor is also one crucial pre-requisite procedure for further locating key point sets and measurement of critical dimensions in SEM images.

Regarding segmentation techniques for SEM images or within the field of semiconductor industry, although BID7 adopted hybrid ridge detector rather than normal approach of edge detector to avoid the effect of noise and double edge and demonstrated robustness to some extent, the method still suffers from issues of computational intensity when generating coefficients for regression in polynomial with high orders.

Based on this work, BID14 further proposed a segmentation method according to global-local thresholding approach and watershed segmentation algorithms on two types of SEM images, achieving relatively high accuracy.

However, this method may be efficient for semiconductor wafer inspection but can pose issues when conducting measurement of critical dimensions with jagged line instead of smooth line for object boundary.

Furthermore, requiring a large number of training dataset and annotated images can be another weakness of these methods.

Referring to related deep learning networks, BID13 training and appears robust in segmentation even with very limited original dataset, which leads to broad applications among different domains.

Fully convolutional network which was proposed by BID16 has raised extensive concerns by training end-to-end with images as both input and output, to achieve state-of-the-art performance, particularly for segmentation and semantic classification BID2 BID16 BID1 .

The key idea of BID16 lies in replacing the max pooling operation with up-sampling convolutional operations, which in turn supports the path of down sampling.

The down-sampling path aims to capture high resolution information while up-sampling path targets at localizing features with pixel-wise manner.

Inspired by BID16 , U-Net BID19 demonstrates superiority particularly with limited training samples but higher precision in masks.

Such promising results are reached via utilizing massive feature channels in up-sampling path and the corresponding network architecture is presented in FIG1 .The discriminating path (left half) and the localizing path (right half) together constitute the whole U-shaped framework with a total of 23 convolutional layers and massive different operations BID19 .

Referring to discriminating path in each dimension, a nonlinear activation ReLU and a 2 × 2 max pooling (stride=2) follows after 2 standard 3 × 3 convolutions.

For such convolutional layers, only valid part is utilized which represents a 1-pixel border lost in each 3 × 3 convolution to allow later large image processing in individual tiles.

Batch normalization using standard deviation and mean is further applied after each convolution to learn bias and scale for higher accuracy.

For max pooling operation, it conducts on each channel separately and doubles the feature channels in each step.

Consequently, discriminating path leads to a spatial contraction with captured abstraction information increased and spatial information decreased.

Regrading localizing path, a sequence of a 2 × 2 up-sampling convolutional operation, two standard 3 × 3 convolutions each followed with a ReLU, and a shortcut connection from corresponding highresolution features in the discriminating path together constitute every step of the localizing path.

The high-resolution segmentation map with two channels for foreground and background separately is generated after a 1 × 1 convolution with 64 channels as input.

The overlap-tile strategy is also conducted when output segmentation maps to ensure the seamless score masks BID19 .

The semiconductor images with etched structures are publicly available on the official website of Oxford Instruments (DAT).

The silicon wafers adopted in this paper consist of 4 types with a total of 10 raw pictures captured by Scanning Electron Microscope with diverse angels, as shown in 3.

For each type of etched structure, only two or three sample images are available with 8 ∼ 33 objects included in one single image.

Such limited training samples indicates data augmentation are significantly crucial for further experiments.

Each image also follows with a manual annotated segmentation result showing ground truth of object boundary.

Initial weights are known to be crucial for deep learning framework like fully convolutional networks to avoid either none or excessive contribution from network.

Considering our network with ReLU activation, the initialization approach from the work of BID10 is applied to calibrate the unit variance.

And such variance is concluded to be 2/N, with N equals to 576 in our case.

With very limited training dataset, data augmentation plays a crucial role in fighting overfitting while preserving robustness.

In our experiments with SEM image on etch structure of wafers, both generic image augmentation approach and Generative Adversarial Net (GAN) are applied to enlarge the training dataset.

For traditional method, a combination of random cropping, horizontally flipping, contrast adjusting, adding salt-and-pepper, etc., has been adopted to increase the number of training images and objects to at least 44 times.

GANs demonstrate efficiency in obtaining counterfeit images using generator which in turn can fool the discriminative network BID9 .

Even image with higher resolution can be generated during augmentation process.

In order to choose the appropriate augmentation approach to boost performance and accuracy, preliminary experiment is conducted using both GAN and traditional augmentation approach to enlarge the dataset for U-Net training.

The sample SEM data for such experiment involving 2 raw images on one type of wafers with each image containing 4 objects, and GAN would increase the original data to 2N in our case.

The SEM wafers images generated by GAN are shown in FIG4 and preliminary experimental results are presented in TAB0 .Revealing from the results, although the created pictures are quite similar to the original one, cases like disconnection in object boundary, double edge effect and some noise in background still appears frequently to impede the learning process of U-Net.

From TAB0 , we can observe that accuracy on training dataset decreases slightly and the result on testing dataset increases around 0.5%.

With GAN, we have more in the training dataset and it is reasonable that the result decreases.

However, less than 1% improvement is not significant.

Besides, with dataset increasing dramatically by utilizing GAN, considerate efforts and expense are required to manually label the ground truth for newly generated images.

As a consequent, further experiment on different types of semiconductor wafers only applied generic augmentation approach like clipping and flipping to extend the original dataset.

The distribution of 10 images is as follows: Type A (4 images), Type B (1 image), Type C (2 images), and Type D (3 images).

Cross-validation methods are used to avoid any potential data-leakage, e.g., if there are three images of the same type, two out of three are used as training dataset and the remaining one are used for testing; if there is only one image for a type, the image is cropped into three smaller ones and again, two out of the three are used as training and the remaining one is used for testing.

The input image tiles for training are 400 × 400 pixels while the output segmentation maps are 400 × 400 too.

Our network is implemented under the open-source deep network library Keras, and the server for experiments is equipped with Dual 8-Core Intel@Xeon Processors 2.4GHz, 128 GB memory, 4 × 2TB SATA3 hard disk, and 4× NvidiaTitan × 12GB GDDR5 GPU cards.

The OS is Ubuntu 14.04 with Keras 2.0.7 installed and Tensorflow 1.0.0 as backend.

The learning rate is adjusted to be 0.0001 throughout implementation process and pixels of segmentation image are within zero to one range generated by sigmoid activation function.

The final test accuracy extracts the best score over all epochs.

With the purpose of boost performance with limited dataset, transfer learning is applied to load the weights from pretrained models for further fine turning purpose so as to increase the segmentation accuracy on similar dataset.

The training speed is quite fast with a single image just costing less than 1 second by our server.

The corresponding segmentation results are displayed in FIG5 , which demonstrate superiority over traditional machine learning approach compared with the results shown in FIG0 .In addition to the pixel-wise accuracy, Intersection over Union (IoU) (also known as Jaccard index) and Dice metric (DM) are the most popularly used metrics to perform quantitative evaluation of image segmentation results BID19 .

We also use IoU and DM to evaluate the performance as shown in Equ.

1. DISPLAYFORM0 where A is the predicted mask and B is the corresponding ground truth.

After comparing the performance of the pre-trained models and the models training from scratch, the experimental results are presented in TAB1 , where experiments are conducted on different types of silicon wafer images and finally implemented on the whole dataset.

The Type D SEM image with a single circle is comparatively simple for learning comparing to the other 3 types to achieve the highest accuracy with 99.57%, then follows Type A wafer with 96.55% accuracy which indicates relatively clear and regular boundary.

Type B ranks next as sunk object boundary appears occasionally, and Type D obvious received least accuracy of 95.59% with uneven edge and diverse ring shapes.

In our case, the superiority of transfer learning is not so obvious due to diverse types of SEM image rather than in similar shapes.

Results on the whole dataset have been degraded to 94.23%, since U-Net cannot handle multiple different datasets simultaneously.

Regarding metrics of IoU and DM, we can observe that Type B has the worst performance and Type D has the best performance, but the results are all more than 90%.

Such results prove the hypothesis that U-Net is significantly appropriate to conduct shape modeling on etched structure of semiconductor wafers with SEM image, although such efficient networks which stem from clinical segmentation have not been applied in semiconductor field yet.

Promising segmentation results laid the foundation of further measurement on critical dimensions and key points of silicon wafers.

Refer to the suggestions from domain experts, Type A and Type B aim to identify the depth, width, and critical dimensions of etched structure, while diameter and recess for each hole are assessed for Type C and Type D wafers.

The corresponding measurement parameters are illustrated in FIG5 .

In this paper, we apply average, max, min, and variance to quantify and evaluate the measurement results, which are demonstrated in TAB2 .

Further more, in TAB3 , we present the comparisons of our measurements with the ground truth and we can see that less than 4% mean error or more than 96% mean accuracy has been achieved for all types in our experiments.

The overall measurement results demonstrate acceptable variability based on opinions from semiconductor experts and can be further broadly applied in automatic measurement of etched structure to ensure the quality and functionality of wafer products in the domain.

This paper has demonstrated deep learning methodes like fully convolutional network U-Net can be further applied in the new issue of etched structure segmentation even with very limited dataset, and have broad application prospects in semiconductor industry to replace the time-consuming manual measurements.

Segmentation results using deep learning approach illustrate superiority over traditional machine learning method to solve the puzzle of undetected boundary and irregular shape problems.

Although GAN is state-of-the-art deep learning technique, results in this paper indicate generic data augmentation methods are even more efficient and powerful to enlarge dataset, and can avoid additional manual labeling tasks.

The superiority of transfer learning is not so apparent in our case due to diverse data types.

In addition to high accuracy in profile characterization, fast prediction can also be guaranteed with testing time of less than 1 second on a single image.

<|TLDR|>

@highlight

Using deep learning method to carry out automatic measurement of SEM images in semiconductor industry