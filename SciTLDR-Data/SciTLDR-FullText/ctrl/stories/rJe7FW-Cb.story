We propose a novel attention mechanism to enhance Convolutional Neural Networks for fine-grained recognition.

The proposed mechanism reuses CNN feature activations to find the most informative parts of the image at different depths with the help of gating mechanisms and without part annotations.

Thus, it can be used to augment any layer of a CNN to extract low- and high-level local information to be more discriminative.



Differently, from other approaches, the mechanism we propose just needs a single pass through the input and it can be trained end-to-end through SGD.

As a consequence, the proposed mechanism is modular, architecture-independent, easy to implement, and faster than iterative approaches.



Experiments show that, when augmented with our approach, Wide Residual Networks systematically achieve superior performance on each of five different fine-grained recognition datasets: the Adience age and gender recognition benchmark, Caltech-UCSD Birds-200-2011, Stanford Dogs, Stanford Cars, and UEC Food-100, obtaining competitive and state-of-the-art scores.

Humans and animals process vasts amounts of information with limited computational resources thanks to attention mechanisms which allow them to focus resources on the most informative chunks of information.

These biological mechanisms have been extensively studied (see BID0 ; BID2 ), concretely those mechanisms concerning visual attention, e.g. the work done by BID25 .In this work, we inspire on the advantages of visual and biological attention mechanisms for finegrained visual recognition with Convolutional Neural Networks (CNN) (see BID11 ).

This is a particularly difficult task since it involves looking for details in large amounts of data (images) while remaining robust to deformation and clutter.

In this sense, different attention mechanisms for fine-grained recognition exist in the literature: (i) iterative methods that process images using "glimpses" with recurrent neural networks (RNN) or long short-term memory (LSTM) (e.g. the work done by BID22 ; BID35 ), (ii) feed-forward attention mechanisms that augment vanilla CNNs, such as the Spatial Transformer Networks (STN) by BID8 , or a top-down feed-forward attention mechanism (FAM) BID19 ).

Although it is not applied to fine-grained recognition, the Residual Attention introduced by BID28 is another example of feed-forward attention mechanism that takes advantage of residual connections BID6 ) to enhance or dampen certain regions of the feature maps in an incremental manner.

Inspired by all the previous research about attention mechanisms in computer vision, we propose a novel feed-forward attention architecture (see FIG0 ) that accumulates and enhances most of the desirable properties from previous approaches:1.

Detect and process in detail the most informative parts of an image: more robust to deformation and clutter.

2.

Feed-forward trainable with SGD: faster inference than iterative models, faster convergence rate than Reinforcement Learning-based (RL) methods like the ones presented by BID22 ; BID15 .

The proposed mechanism.

Feature maps at different levels are processed to generate spatial attention masks and use them to output a class hypothesis based on local information and a confidence score (C).

The final prediction consists of the average of all the hypotheses weighted by the normalized confidence scores.

.

Preserve low-level detail: unlike Residual Attention BID28 , where low-level features are subject to noise after traversing multiple residual connections, our architecture directly uses them to make predictions.

This is important for fine-grained recognition, where low-level patterns such as textures can help to distinguish two similar classes.

Moreover, the proposed mechanism possesses other interesting properties such as:1.

Modular and incremental: the attention mechanism can be replicated at each layer on any convolutional architecture, and it is easy to adapt to the task at hand.2.

Architecture independent: the mechanism can accept any pre-trained architecture such as VGG BID23 ) or ResNet.3.

Low computational impact: While STNs use a small convnet to predict affine-transform parameters and Residual Attention uses the hourglass architecture, our attention mechanism consists of a single 1 × 1 convolution and a small fully-connected layer.4.

Simple: the proposed mechanism can be implemented in few lines of code, making it appealing to be used in future work.

The proposed attention mechanism has been included in a strong baseline like Wide Residual Networks (WRN) BID31 ), and applied on five fine-grained recognition datasets.

The resulting network, called Wide Residual Network with Attention (WRNA) systematically enhances the performance of WRNs, obtains competitive results using low resolution training images, and surpasses the state of the art in the Adience gender recognition task, Stanford dogs, and UEC Food-100.

Table 1 shows the gain in performance of WRNA w.r.t.

WRN for all the datasets considered in this paper.

In the next section, we review the most relevant work concerning attention mechanisms for visual fine-grained recognition.

As reviewed by BID34 , there are different approaches to fine-grained recognition: (i) vanilla deep CNNs, (ii) CNNs as feature extractors for localizing parts and do alignment, (iii) ensembles, (iv) attention mechanisms.

In this paper we focus on (iv), the attention mechanisms, which aim to discover the most discriminative parts of an image to be processed in greater detail, thus ignoring clutter and focusing on the most distinctive traits.

These parts are central for fine-grained recognition, where the inter-class variance is small and the intra-class variance is high.

Different fine-grained attention mechanisms can be found in the literature.

BID29 proposed a two-level attention mechanism for fine-grained classification on different subsets of the ICLR2012 BID21 ) dataset, and the CUB200 2011.

In this model, images are first processed by a bottom-up object proposal network based on R-CNN BID32 ) and selective search BID24 ).

Then, the softmax scores of another ILSVRC2012 pretrained CNN, which they call FilterNet, are thresholded to prune the patches with the lowest parent class score.

These patches are then classified to fine-grained categories with a DomainNet.

Spectral clustering is also used on the DomainNet filters in order to extract parts (head, neck, body, etc.), which are classified with an SVM.

Finally, the part-and object-based classifier scores are merged to get the final prediction.

The two-level attention obtained state of the art results on CUB200-2011 with only class-level supervision.

However, the pipeline must be carefully fine-tuned since many stages are involved with many hyper-parameters.

Differently from two-level attention, which consists of independent processing and it is not endto-end, Sermanet et al. proposed to use a deep CNN and a Recurrent Neural Network (RNN) to accumulate high multi-resolution "glimpses" of an image to make a final prediction BID22 ), however, reinforcement learning slows down convergence and the RNN adds extra computation steps and parameters.

A more efficient approach was presented by BID15 ), where a fullyconvolutional network is trained with reinforcement learning to generate confidence maps on the image and use them to extract the parts for the final classifiers whose scores are averaged.

Compared to previous approaches, in the work done by BID15 , multiple image regions are proposed in a single timestep thus, speeding up the computation.

A greedy reward strategy is also proposed in order to increase the training speed.

The recent approach presented by BID4 uses a classification network and a recurrent attention proposal network that iteratively refines the center and scale of the input (RA-CNN).

A ranking loss is used to enforce incremental performance at each iteration.

Zhao et al. proposed Diversified Visual Attention Network (DVAN), i.e. enforcing multiple nonoverlapped attention regions BID35 ).

The overall architecture consists of an attention canvas generator, which extracts patches of different regions and scales from the original image; a VGG-16 BID23 ) CNN is then used to extract features from the patches, which are aggregated with a DVAN long short-term memory BID7 ) that attends to non-overlapping regions of the patches.

Classification is performed with the average prediction of the DVAN at each region.

All the previously described methods involve multi-stage pipelines and most of them are trained using reinforcement learning (which requires sampling and makes them slow to train).

In contrast, STNs, FAM, and our approach jointly propose the attention regions and classify them in a single pass.

Moreover, they possess interesting properties compared to previous approaches such as (i) simplicity (just a single model is needed), (ii) deterministic training (no RL), and (iii) feed-forward training (only one timestep is needed), see Table 2 .

In addition, since our approach only uses one CNN stream, it is far more computationally efficient than STNs and FAM, as described next.

Our approach consists of a universal attention module that can be added after each convolutional layer without altering pre-defined information pathways of any architecture.

This is helpful since it allows to seamlessly augment any architecture such as VGG and ResNet with no extra supervision, i.e. no part labels are necessary.

The attention module consists of three main submodules: (i) the attention heads H, which define the most relevant regions of a feature map, (ii) the output heads O, generate an hypothesis given the attended information, and (iii) the confidence gates G, which output a confidence score for each attention head.

Each of these modules is explained in detail in the following subsections.

As it can be seen in FIG0 , and 2b, a 1 × 1 convolution is applied to the output of the augmented layer, producing an attentional heatmap.

This heatmap is then element-wise multiplied with a copy of the layer output, and the result is used to predict the class probabilities and a confidence score.

This process is applied to an arbitrary number N of layers, producing N class probability vectors, and N confidence scores.

Then, all the class predictions are weighted by the confidence scores (softmax normalized so that they add up to 1) and averaged (using 9).

This is the final combined prediction of the network.

Inspired by DVAN BID35 ) and the transformer architecture presented by BID26 , and following the notation established by BID31 , we have identified two main dimensions to define attentional mechanisms: (i) the number of layers using the attention mechanism, which we call attention depth (AD), and (ii) the number of attention heads in each attention module, which we call attention width (AW).

Thus, a desirable property for any universal attention mechanism is to be able to be deployed at any arbitrary depth and width.

This property is fulfilled by including K attention heads H k (width), depicted in FIG2 , into each attention module (depth) 1 .

Then, the attention heads at layer l, receive the feature activations Z l of that layer as input, and output K weighted feature maps, see Equations 1 and 2: DISPLAYFORM0 DISPLAYFORM1 where H l is the output matrix of the l th attention module, W H is a 1 × 1 convolution kernel with output dimensionality K used to compute the attention masks corresponding to the attention heads H k , * denotes the convolution operator, and is the element-wise product.

Please note that M l k is a 2d flat mask and the product with each of the N input channels of Z is done by broadcasting.

Likewise, the dimensionality of H l k is the same as Z l .

The spatial softmax is used to enforce the model to learn the most relevant region of the image.

Sigmoid units could also be used at the risk of degeneration to all-zeros or all-ones.

Since the different attention heads in an attention module are sometimes focusing on the same exact part of the image, similarly to DVAN, we have introduced a regularization loss L R that forces the multiple masks to be different.

In order to simplify the notation, we set m l k to be k th flattened version of the attention mask M in Equation 1.

Then, the regularization loss is expressed as: DISPLAYFORM2 i.e., it minimizes the squared Frobenius norm of the off-diagonal cross-correlation matrix formed by the squared inner product of each pair of different attention masks, pushing them towards orthogonality (L R = 0).

This loss is added to the network loss L net weighted by a constant factor γ = 0.1 which was found to work best across all tasks: DISPLAYFORM3 .5 2 2 +1.6 Table 3 : Average performance impact across datasets on (in accuracy %) of the attention depth (AD), attention width (AW ), and the presence of gates (G) on WRN.

The output of each attention module consists of a spatial dimensionality reduction layer: DISPLAYFORM0 followed by a fully-connected layer that produces an hypothesis on the output space, see FIG2 .

DISPLAYFORM1 We consider two different dimensionality reductions: (i) a channel-wise inner product by W 1×n F , where W F is a dimensionality reduction projection matrix with n the number of input channels; and (ii) an average pooling layer.

We empirically found (i) to work slightly better than (ii) but at a higher computational cost.

W F is shared across all attention heads in an attention module.

Each attention module makes a class hypothesis given its local information.

However, in some cases, the local features are not good enough to output a good hypothesis.

In order to alleviate this problem, we make each attention module, as well as the network output, to predict a confidence score c by means of an inner product by the gate weight matrix W G : DISPLAYFORM0 The gate weights g are then obtained by normalizing the set of scores by means of a sof tmax function: DISPLAYFORM1 where |G| is the total number of gates, and c i is the i th confidence score from the set of all confidence scores.

The final output of the network is the weighted sum of the output heads: DISPLAYFORM2 where g net is the gate value for the original network output (output net ), and output is the final output taking the attentional predictions o l h into consideration.

Please note that setting the output of G to 1 |G| , corresponds to averaging all the outputs.

Likewise, setting {G \G output } = 0, G output = 1, i.e. the set of attention gates is set to zero and the output gate to one, corresponds to the original pre-trained model without attention.

In Table 3 we show the importance of each submodule of our proposal on WRN.

Instead of augmenting all the layers of the WRN, in order to have the minimal computational impact and to attend features of different levels, attention modules are placed after each pooling layer, where the spatial resolution is divided by two.

Attention Modules are thus placed starting from the fourth pooling layer and going backward when AD increases.

As it can be seen, just adding a single attention module with a single attention head is enough to increase the mean accuracy by 1.2%.

Adding extra heads and gates increase an extra 0.1% each.

Since the first and second pooling layers have a big spatial resolution, the receptive field for AD > 2 was too small and did not result in increased accuracy.

The fact that the attention mask is generated by just one 1 × 1 convolution and the direct connection to the output makes the module fast to learn, thus being able to generate foreground masks from the beginning of the training and refining them during the following epochs.

A sample of these attention masks for each dataset is shown on FIG3 .

As it can be seen, the masks help to focus on the foreground object.

In (c), the attention mask focuses on ears for gender recognition, possibly looking for earrings.

DISPLAYFORM3

In order to support the design decisions of Section 3, we follow the procedure of BID17 , and train a CNN on the Cluttered Translated MNIST dataset 2 , consisting of 40 × 40 images containing a randomly placed MNIST digit and a set of D randomly placed distractors, see FIG5 .

The distractors are random 8 × 8 patches from other MNIST digits.

The CNN consists of five 3 × 3 convolutional layers and two fully-connected in the end, the three first convolution layers are followed by a spatial pooling.

Batch-normalization was applied to the inputs of all these layers.

Attention modules were placed starting from the fifth convolution (or pooling instead) backwards until AD images validation set, and tested on 100k test images.

First, we tested the importance of AW and AD for our model.

As it can be seen in FIG5 , greater AD results in better accuracy, reaching saturation at AD = 4, note that for this value the receptive field of the attention module is 5 × 5 px, and thus the performance improvement from such small regions is limited.

FIG5 shows training curves for different values of AW .

As it can be seen, small performance increments are obtained by increasing the number of attention heads despite there is only one object present in the image.

Then, we used the best AD and AW to verify the importance of using softmax on the attention masks instead of sigmoid (1), the effect of using gates (Eq. 8), and the benefits of regularization (Eq. 3).

FIG5 confirms that, ordered by importance: gates, softmax, and regularization result in accuracy improvement, reaching 97.8%.

Concretely, we found that gates pay an important role discarding the distractors, especially for high AW and high AD.Finally, in order to verify that attention masks are not overfitting on the data, and thus generalize to any amount of clutter, we run our best model so far FIG5 ) on the test set with an increasing number of distractors (from 4 to 64).

For the comparison, we included the baseline model before applying our approach and the same baseline augmented with an STN Jaderberg et al. (2015) that reached comparable performance as our best model in the validation set.

All three models were trained with the same dataset with eight distractors.

Remarkably, as it can be seen in FIG5 , the attention augmented model demonstrates better generalization than the baseline and the STN.

In order to demonstrate that the proposed generalized attention can easily augment any recent architecture, we have trained a strong baseline, namely a Wide Residual Network (WRN) BID31 ) pre-trained on the ImageNet.

We chose to place attention modules after each pooling layer to extract different level features with minimal computational impact.

The modules described in the previous sections have been implemented on pytorch, and trained in a single workstation with two NVIDIA 1080Ti.

All the experiments are trained for 100 epochs, with a batch size of 64.

The learning rate is first set to 10 −3 to all layers except the attention modules and the classifier, for which it ten times higher.

The learning rate is reduced by a factor of 0.5 every 30 iterations and the experiment is automatically stopped if a plateau is reached.

The network is trained with For the sake of clarity and since the aim of this work is to demonstrate that the proposed mechanism universally improves CNNs for fine-grained recognition, we follow the same training procedure in all datasets.

Thus, we do not use 512 × 512 images which are central in STNs, RA-CNNs, or B-CNNs to reach state of the art performances.

Accordingly, we do not perform color jitter and other advanced augmentation techniques such as the ones used by BID5 for food recognition.

The proposed method is able to obtain state of the art results in Adience Gender, Stanford dogs and UEC Food-100 even when trained with lower resolution.

In the following subsections the proposed approach is evaluated on the five datasets.

Adience dataset.

The adience dataset consists of 26.5 K images distributed in eight age categories (02, 46, 813, 1520, 2532, 3843, 4853, 60+) , and gender labels.

A sample is shown in FIG7 .The performance on this dataset is measured by both the accuracy in gender and age recognition tasks using 5-fold cross-validation in which the provided folds are subject-exclusive.

The final score is given by the mean of the accuracies of the five folds.

This dataset is particularly challenging due to the high level of deformation of face pictures taken in the wild, occlusions and clutter such as sunglasses, hats, and even multiple people in the same image.

As it can be seen in Table 4 , the Wide ResNet augmented with generalized attention surpasses the baseline performance, etc.

The birds dataset (see FIG7 ) consists of 6K train and 5.8K test bird images distributed in 200 categories.

The dataset is especially challenging since birds are in different poses and orientations, and correct classification often depends on texture and shape details.

Although bounding box, bough segmentation, and attributes are provided, we perform raw classification as done by BID8 .In TAB2 , the performance of our approach is shown in context with the state of the art.

Please note that even our approach is trained in lower resolution crops, i.e. 224 × 224 instead of 448 × 448, we reach the same accuracy as the recent fully convolutional attention by BID15 .

FIG7 .

The data is split into 8K training images and 8K testing images.

The difficulty of this dataset resides in the identification of the subtle differences that distinguish between two car models.

In TAB4 the performance of our approach with respect to the baseline and other state of the art is shown.

The augmented WRN shows better performance than the baseline, and even surpasses recent approaches such as FCAN.Stanford Dogs.

The Stanford Dogs dataset consists of 20.5K images of 120 breeds of dogs, see FIG7 .

The dataset splits are fixed and they consist of 12k training images and 8.5K validation images.

Pictures are taken in the wild and thus dogs are not always a centered, unique, posenormalized object in the image but a small, cluttered region.

Table 7 shows the results on Stanford dogs.

As it can be seen, performances are low in general and nonetheless, our model was able to increase the accuracy by a 0.3% (0.1% w/o gates), being the highest score obtained on this dataset to the best of our knowledge.

This performance has been achieved thanks to the gates, which act as a detection mechanism, giving more importance to those attention masks that correctly guessed the position of the dog.

UEC Food 100 is a Japanese food dataset with 14K images of 100 different dishes, see FIG7 .

Pictures present a high level of variation in the form of deformation, rotation, clutter, and noise.

In Table 8 : Performance on UEC Food-100.

Table 8 shows the performance of our model compared to the state of the art.

As it can be seen, our model is able to improve the baseline by a relative 7% with a 85.5% of accuracy, the best-reported result compared to previous publications.

We have presented a novel attention mechanism to improve CNNs for fine-grained recognition.

The proposed mechanism finds the most informative parts of the CNN feature maps at different depth levels and combines them with a gating mechanism to update the output distribution.

Moreover, we thoroughly tested all the components of the proposed mechanism on Cluttered Translated MNIST, and demonstrate that the augmented models generalize better on the test set than their plain counterparts.

We hypothesize that attention helps to discard noisy uninformative regions, avoiding the network to memorize them.

Unlike previous work, the proposed mechanism is modular, architecture independent, fast, and simple and yet WRN augmented with it show higher accuracy in each of the following tasks: Age and Gender Recognition (Adience dataset), CUB200-2011 birds, Stanford Dogs, Stanford Cars, and UEC Food-100.

Moreover, state of the art performance is obtained on gender, dogs, and cars.

Figure 6: Test accuracy logs for the five fine-grained datasets.

As it can be seen, the augmented models (WRNA) achieve higher accuracy at similar convergence rates.

For the sake of space we only show one of the five folds of the Adience dataset.

<|TLDR|>

@highlight

We enhance CNNs with a novel attention mechanism for fine-grained recognition. Superior performance is obtained on 5 datasets.

@highlight

Describes a novel attentional mechanism applid to fine-grained recognition that consistently improves the recognition accuracy of the baseline

@highlight

This paper proposes a feed-forward attention mechanism for fine-grained image classification

@highlight

This paper presents an interesting attention mechanism for fine-grained image classification.