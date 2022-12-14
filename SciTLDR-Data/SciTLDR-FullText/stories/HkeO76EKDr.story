Pointwise localization allows more precise localization and accurate interpretability, compared to bounding box, in applications where objects are highly unstructured such as in medical domain.

In this work, we focus on  weakly supervised localization (WSL) where a model is trained to classify an image and localize regions of interest at pixel-level using only global image annotation.

Typical convolutional attentions maps are prune to high false positive regions.

To alleviate this issue, we propose a new deep learning method for WSL, composed of a localizer and a classifier, where the localizer is constrained to determine relevant and irrelevant regions using conditional entropy (CE) with the aim to reduce false positive regions.

Experimental results on a public medical dataset and two natural datasets, using Dice index, show that, compared to state of the art WSL methods, our proposal can provide significant improvements in terms of image-level classification and pixel-level localization (low false positive) with robustness to overfitting.

A public reproducible PyTorch implementation is provided.

Pointwise localization is an important task for image understanding, as it provides crucial clues to challenging visual recognition problems, such as semantic segmentation, besides being an essential and precise visual interpretability tool.

Deep learning methods, and particularly convolutional neural networks (CNNs), are driving recent progress in these tasks.

Nevertheless, despite their remarkable performance, their training requires large amounts of labeled data, which is time consuming and prone to observer variability.

To overcome this limitation, weakly supervised learning (WSL) has emerged recently as a surrogate for extensive annotations of training data (Zhou, 2017) .

WSL involves scenarios where training is performed with inexact or uncertain supervision.

In the context of pointwise localization or semantic segmentation, weak supervision typically comes in the form of image level tags (Kervadec et al., 2019; Kim et al., 2017; Pathak et al., 2015; Teh et al., 2016; Wei et al., 2017) , scribbles (Lin et al., 2016; Tang et al., 2018) or bounding boxes (Khoreva et al., 2017) .

Current state-of-the-art WSL methods rely heavily on pixelwise activation maps produced by a CNN classifier at the image level, thereby localizing regions of interest (Zhou et al., 2016) .

Furthermore, this can be used as an interpretation of the model's decision (Zhang & Zhu, 2018) .

The recent literature abounds of WSL works that relax the need of dense and prohibitively time consuming pixel-level annotations (Rony et al., 2019) .

Bottom-up methods rely on the input signal to locate regions of interest, including spatial pooling techniques over activation maps (Durand et al., 2017; Oquab et al., 2015; Sun et al., 2016; Zhang et al., 2018b; Zhou et al., 2016) , multi-instance learning (Ilse et al., 2018) and attend-and-erase based methods (Kim et al., 2017; Li et al., 2018; Pathak et al., 2015; Singh & Lee, 2017; Wei et al., 2017) .

While these methods provide pointwise localization, the models in (Bilen & Vedaldi, 2016; Kantorov et al., 2016; Shen et al., 2018; Tang et al., 2017; Wan et al., 2018 ) predict a bounding box instead, i.e., perform weakly supervised object detection.

Inspired by human visual attention, top-down methods rely on the input signal and a selective backward signal to determine the corresponding region of interest.

This includes special feedback layers (Cao et al., 2015) , backpropagation error (Zhang et al., 2018a) and Grad-CAM (Chattopadhyay et al., 2018; Selvaraju et al., 2017) .

In many applications, such as in medical imaging, region localization may require high precision such as cells, boundaries, and organs localization; regions that have an unstructured shape, and different scale that a bounding box may not be able to localize precisely.

In such cases, a pointwise localization can be more suitable.

The illustrative example in Fig.1 (bottom row) shows a typical case where using a bounding box to localize the glands is clearly problematic.

This motivates us to consider predicting a mask instead of a bounding box.

Consequently, our latter choice of evaluation datasets is constrained by the availability of both global image annotation for training and pixel-level annotation for evaluation.

In this work, we focus on the case where there is one object of interest in the image.

Often, within an agnostic-class setup, input image contains the object of interest among other irrelevant parts (noise, background).

Most the aforementioned WSL methods do not consider such prior, and feed the entire image to the model.

In such scenario, (Wan et al., 2018) argue that there is an inconsistency between the classification loss and the task of WSL; and that typically the optimization may reach sub-optimal solutions with considerable randomness in them, leading to high false positive localization.

False positive localization is aggravated when a class appears in different and random shape/structure, or may have relatively similar texture/color to the irrelevant parts driving the model to confuse between both parts.

False positive regions can be problematic in critical domains such as medical applications where interpretability plays a central role in trusting and understanding an algorithm's prediction.

To address this important issue, and motivated by the importance of using prior knowledge in learning to alleviate overfitting when training using few samples (Belharbi et al., 2017; Krupka & Tishby, 2007; Mitchell, 1980; Yu et al., 2007) , we propose to use the aforementioned prior in order to favorite models with low false positive localization.

To this end, we constrain the model to learn to localize both relevant and irrelevant regions simultaneously in an end-to-end manner within a WSL scenario, where only image-level labels are used for training.

We model the relevant (discriminative) regions as the complement of the irrelevant (non-discriminative) regions (Fig.1) .

Our model is composed of two sub-models: (1) a localizer that aims to localize both types of regions by predicting a latent mask, (2) and a classifier that aims to classify the visible content of the input image through the latent mask.

The localizer is driven through CE (Cover & Thomas, 2006) to simultaneously identify (1) relevant regions where the classifier has high confidence with respect to the image label, (2) and irrelevant regions where the classifier is being unable to decide which image label to assign.

This modeling allows the discriminative regions to pop out and be used to assign the corresponding image label, while suppressing non-discriminative areas, leading to more reliable predictions.

In order to localize complete discriminative regions, we extend our proposal by training the localizer to recursively erase discriminative parts during training only.

To this end, we propose a consistent recursive erasing algorithm that we incorporate within the backpropagation.

At each recursion, and within the backpropagation, the algorithm localizes the most discriminative region; stores it; then erases it from the input image.

At the end of the final recursion, the model has gathered a large extent of the object of interest that is fed next to the classifier.

Thus, our model is driven to localize complete relevant regions while discarding irrelevant regions, resulting in more reliable region localization.

Moreover, since the discriminative parts are allowed to be extended over different instances, our proposal handles multi-instances intrinsically.

The main contribution of this paper is a new deep learning framework for WSL at pixel level.

The framework is composed of two sequential sub-networks where the first one localizes regions of interest, whereas the second classifies them.

Based on CE, the end-to-end training of the framework allows to incorporate prior knowledge that, an image is more likely to contain relevant and irrelevant regions.

Throughout the CE measured at the classifier level, the localizer is driven to localize relevant regions (with low CE) and irrelevant regions (with high CE).

Such localization is achieved with the main goal of providing a more interpretable and reliable regions of interest with low false positive localization.

This paper also contributes a consistent recursive erasing algorithm that is incorporated within backpropagation, along with a practical implementation in order to obtain complete discriminative regions.

Finally, we conduct an extensive series of experiments on three public image datasets (medical and natural), where the results show the effectiveness of the proposed approach in terms of pointwise localization (measured with Dice index) while maintaining competitive accuracy for image-level classification.

In this section, we briefly review state of the art of WSL methods, divided into two main categories, aiming at pointwise localization of regions of interest using only image-level labels as supervision.

(1) Fully convolutional networks with spatial pooling have shown to be effective to obtain localization of discriminative regions (Durand et al., 2017; Oquab et al., 2015; Sun et al., 2016; Zhang et al., 2018b; Zhou et al., 2016) .

Multi-instance learning methods have been used within an attention framework to localize regions of interest (Ilse et al., 2018) . (Singh & Lee, 2017) propose to hide randomly large patches in training image in order to force the network to seek other discriminative regions to recover large part of the object of interest, since neural networks often provide small and most discriminative regions of object of interest (Kim et al., 2017; Singh & Lee, 2017; Zhou et al., 2016) . (Wei et al., 2017) use the attention map of a trained network to erase the most discriminative part of the original image. (Kim et al., 2017) use two-phase learning stage where the attention maps of two networks are combined to obtain a complete region of the object. (Li et al., 2018) propose a two-stage approach where the first network classifies the image, and provides an attention map of the most discriminative parts.

Such attention is used to erase the corresponding parts over the input image, then feed the resulting erased image to a second network to make sure that there is no discriminative parts left.

(2) Inspired by the human visual attention, top-down methods were proposed.

In (Simonyan et al., 2014; Springenberg et al., 2015; Zeiler & Fergus, 2014) , backpropagation error is used in order to visualize saliency maps over the image for the predicted class.

In (Cao et al., 2015) , an attention map is built to identify the class relevant regions using feedback layer. (Zhang et al., 2018a) propose Excitation backprop that allows to pass along top-down signals downwards in the network hierarchy through a probabilistic framework.

Grad-CAM (Selvaraju et al., 2017) generalize CAM (Zhou et al., 2016) using the derivative of the class scores with respect to each location on the feature maps; it has been furthermore generalized in (Chattopadhyay et al., 2018) .

In practice, top-down methods are considered as visual explanatory tools, and they can be overwhelming in term of computation and memory usage even during inference.

While the aforementioned approaches have shown great success mostly with natural images, they still lack a mechanism for modeling what is relevant and irrelevant within an image which is important to reduce false positive localization.

This is crucial for determining the reliability of the regions of interest.

Erase-based methods (Kim et al., 2017; Li et al., 2018; Pathak et al., 2015; Singh & Lee, 2017; Wei et al., 2017) follow such concept where the non-discriminative parts are suppressed through constraints, allowing only the discriminative ones to emerge.

Explicitly modeling negative evidence within the model has shown to be effective in WSL (Azizpour et al., 2015; Durand et al., 2017; 2016; Parizi et al., 2015) .

Our proposal is related to (Behpour et al., 2019; Wan et al., 2018) in using entropy-measure to explore the input image.

However, while (Wan et al., 2018) defines an entropy over the bounding boxes' position to minimize its variance, we define a CE over the classifier to be low over discriminative regions, while being high over non-discriminative ones.

Our recursive erasing algorithm follows general erasing and mining techniques (Kim et al., 2017; Li et al., 2018; Singh & Lee, 2017; Wan et al., 2018; Wei et al., 2017) , but places more emphasis on mining consistent regions, and being performed on the fly during backpropagation.

For instance, compared to (Wan et al., 2018) , our algorithm attempts to expand regions of interest, accumulate consistent regions while erasing, provide automatic mechanism to stop erasing over samples independently from each other.

However (Wan et al., 2018) aims to locate multiple instances without erasing, and use manual/empirical threshold for assigning confidence to boxes.

Our proposal can be seen as a guided dropout (Srivastava et al., 2014) .

While standard dropout is applied over a given input image to randomly zero out pixels, our proposed approach seeks to zero out irrelevant pixels and keep only the discriminative ones that support the image label.

From this perspective, our proposal mimics a discriminative gate that inhibits irrelevant and noisy regions while allowing only informative and discriminative regions to pass through the gate.

Notations and definitions: Let us consider a set of training samples

where X i is an input image with depth d, height h, and width w; a realization of the discrete random variable X with support set X ; y i is the image-level label (i.e., image class), a realization of the discrete random variable y with support set Y = {1, ?? ?? ?? , c}. We define a decidable region 1 of an image as any informative part of the image that allows predicting the image label.

An undecidable region is any noisy, uninformative, and irrelevant part of the image that does not provide any indication nor support for the image class.

To model such definitions, we consider a binary mask M + ??? {0, 1} h??w where a location (r, z) with value 1 indicates a decidable region, otherwise it is an undecidable region.

We model the decidability of a given location (r, z) with a binary random variable M. Its realization is m, and its conditional probability p m over the input image is defined as follows,

We note M ??? ??? {0, 1} h??w = U ??? M + a binary mask indicating the undecidable region, where

h??w .

We consider the undecidable region as the complement of the decidable one.

We can write:

where ?? 0 is the l 0 norm.

Following such definitions, an input image X can be decomposed into two images as X = X M + + X M ??? , where (?? ??) is the Hadamard product.

We note X + = X M + , and X ??? = X M ??? .

X + inherits the image-level label of X. We can write the pair (X + i , y i ) in the same way as (X i , y i ).

We note by R Following the previous discussion, predicting the image label depends only on the decidable region, i.e., X + .

Thus, knowing X ??? does not add any knowledge to the prediction, since X ??? does not contain any information about the image label.

This leads to: p(Y|X = X) = p(Y|X = X + ).

As a consequence, the image label is conditionally independent of

where X + , X ??? are the random variables modeling the decidable and the undecidable regions, respectively.

In the following, we provide more details on how to exploit such conditional independence property in order to estimate R + and R ??? .

We consider modeling the uncertainty of the model prediction over decidable, or undecidable regions using conditional entropy (CE).

Let us consider the CE of Y|X = X + , denoted H(Y|X = X + ) and computed as (Cover & Thomas, 2006) ,

Since the model is required to be certain about its prediction over X + , we constrain the model to have low entropy over X + .

Eq.2 reaches its minimum when the probability of one of the classes is certain, i.e.,p(Y = y|X = X + ) = 1 (Cover & Thomas, 2006) .

Instead of directly minimizing Eq.2, and in order to ensure that the model predicts the correct image label, we cast a supervised learning problem using the cross-entropy between p andp using the image-level label of X as a supervision,

Eq.3 reaches its minimum at the same conditions as Eq.2 with the true image label as a prediction.

We note that Eq.3 is the negative log-likelihood of the sample (X i , y i ).

In the case of X ??? , we consider the CE of Y|X = X ??? , denoted H(Y|X = X ??? ) and computed as,

Over irrelevant regions, the model is required to be unable to decide which image class to predict since there is no evidence to support any class.

This can be seen as a high uncertainty in the model decision.

Therefore, we consider maximizing the entropy of Eq.4.

The later reaches its maximum at the uniform distribution (Cover & Thomas, 2006) .

Thus, the inability of the model to decide is reached since each class is equiprobable.

An alternative to maximizing Eq.4 is to use a supervised target distribution since it is already known (i.e., uniform distribution).

To this end, we consider q as a uniform distribution, q(Y = y|X = X ??? i ) = 1/c , ???y ??? Y , and caste a supervised learning setup using a cross-entropy between q andp over X ??? ,

The minimum of Eq.5 is reached whenp(Y|X = X ??? i ) is uniform, thus, Eq.4 reaches its maximum.

Now, we can write the total training loss to be minimized as,

The posterior probabilityp is modeled using a classifier C(. , ?? C ) with a set of parameters ?? C ; it can operate either on X

is learned using another model M(X i ; ?? M ) with a set of parameters ?? M .

In this work, both models are based on neural networks (fully convolutional networks (Long et al., 2015) in particular).

The networks M and C can be seen as two parts of one single network G that localizes regions of interest using a binary mask, then classifies their content.

supervised gradient based only on the error made by C. In order to boost the supervised gradient at M, and provide it with more hints to select the most discriminative regions with respect to the image class, we consider using a secondary classification task at the output of M to classify the input X, following (Lee et al., 2015) .

M computes the posterior probabilityp s (Y |X) which is another estimate of p(Y |X).

To this end, M is trained to minimize the cross-entropy between p andp s ,

The total training loss to minimize is formulated as,

Mask computation and recursive erasing: The mask R + is computed using the last feature maps of M which contains high abstract descriminative activations.

We note such feature maps by a tensor A i ??? R c??h ??w that contains a spatial map for each class.

R + i is computed by aggregating the spatial activation of all the classes as,

is the continuous downsampled version of R + i , and A i (k) is the feature map of the class k of the input X i .

At convergence, the posterior probability of the winning class is pushed toward 1 while the rest is pushed down to 0.

This leaves only the feature map of the winning classe.

T i is upscaled using interpolation (Sec.

A.2) to T ??? i ??? R h??w which has the same size as the input X, then pseudo-thresholded using a sigmoid function to obtain a pseudo-binary R

where ?? is a constant scalar that ensures that the sigmoid approximately equals to 1 when T ??? i (r, z) is larger than ?? , and approximately equals to 0 otherwise.

At this point, R ??? may still contain discriminative regions.

To alleviate this issue, we propose a learning incremental and recursive erasing approach that drives M to mine complete discriminative regions.

The mining algorithm is consistent, sample dependent, it has a maximum recursion depth u, associates trust coefficients to each recursion, integrated within the backpropagation, operates only during training, and has a practical implementation.

Due to space limitation, we left it in the supplementary material (Sec.

A.1).

Our experiments focus simultaneously on classification and pointwise localization tasks.

Thus, we consider datasets that provide both image and pixel-level labels for evaluation.

Particularly, the following three datasets are considered: GlaS in medical domain, and CUB-200-2011 and Oxford flower 102 on natural scene images.

(1) GlaS dataset, one of the rare medical datasets that fits our scenario (Rony et al., 2019) , was provided in the 2015 Gland Segmentation in Colon Histology Images Challenge Contest 2 (Sirinukunwattana et al., 2017) .

The main task of the challenge is gland segmentation of microscopic images.

However, image-level labels were provided as well.

The dataset is composed of 165 images derived from 16 Hematoxylin and Eosin (H&E) histology sections of two grades (classes): benign, and malignant.

It is divided into 84 samples for training, and 80 samples for test.

Images have a high variation in term of gland shape/size, and overall H&E stain.

In this dataset, the glandes are the regions of interest that the pathologists use to prognosis the image grading of being benign or malignant.

(2) CUB-200-2011 dataset 3 (Wah et al., 2011) is a dataset for bird species with 11, 788 samples and 200 species.

Preliminary experiments were conducted on small version of this datatset where we selected randomly 5 species and build a small dataset with 150 samples for training, and 111 for test; referred to in this work as CUB5.

The entire dataset is referred to as CUB.

In this dataset, the regions of interest are the birds.

(3) Oxford flower 102 4 (Nilsback & Zisserman, 2007) datatset is collection of 102 species (classes) of flowers commonly occurring in United Kingdom; referred to here as OxF. It contains a total of 8, 189 samples.

We used the provided splits for training (1, 020 samples), validation (1, 020 samples) and test (6, 149 samples) sets.

Regions of interest are the flowers which were segmented automatically.

In GlaS, CUB5 and CUB datasets, we randomly select 80% of training samples for effective training, and 20% for validation to perform early stopping.

We provide in our public code the used splits and the deterministic code that generated them for the different datasets.

In all the experiments, image-level labels are used during training/evaluation, while pixel-level labels are used exclusively during evaluation.

The evaluation is conducted at two levels: at image-level where the classification error is reported, and at the pixel-level where we report F1 score (Dice index) over the foreground (region of interest), referred to as F1 + .

When dealing with binary data, F1 score is equivalent to Dice index.

We report as well the F1 score over the background, referred to as F1 ??? , in order to measure how well the model is able to identify irrelevant regions.

We compare our method to different methods of WSL.

Such methods use similar pre-trained backbone (resent18 (He et al., 2016) ) for feature extraction and differ mainly in the final pooling layer: CAM-Avg uses average pooling (Zhou et al., 2016) , CAM-Max uses max-pooling (Oquab et al., 2015) , CAM-LSE uses an approximation to maximum (Pinheiro & Collobert, 2015; Sun et al., 2016) , Wildcat uses the pooling in (Durand et al., 2017) , Grad-CAM (Selvaraju et al., 2017) , and Deep MIL is the work of (Ilse et al., 2018) with adaptation to multi-class.

We use supervised segmentation using U-Net (Ronneberger et al., 2015) as an upper bound of the performance for pixel-level evaluation (Full sup.).

As a simple baseline, we use a mask full of 1 with the same size of the image as a constant prediction of regions of interest to show that F1 + alone is not an efficient metric to evaluate pixel-level localization particularly over GlaS set (All-ones, Tab.2).

In our method, M and C share the same pre-trained backbone (resnet101 (He et al., 2016) ) to avoid overfitting while using (Durand et al., 2017 ) as a pooling function.

All methods are trained using stochastic gradient descent using momentum.

In our approach, we use the same hyper-parameters over all datasets, while other methods require adaptation to each dataset.

We provide the datasets splits, more experimental details, and visual results in the supplementary material (Sec.

B).

Our reproducible code is publicly available.

A comparison of the obtained results of different methods, over all datasets, is presented in Tab.1 and Tab.2 with visual results illustrated in Fig.3 .

In Tab.2, and compared to other WSL methods, our method obtains relatively similar F1 + score; while it obtains large F1 ??? over GlaS where it may be easy to obtain high F1

+ by predicting a mask full of 1 (Fig.3) .

However, a model needs to be very selective in order to obtain high F1 ??? score in order to localize tissues (irrelevant regions) where our model seems to excel at.

Cub5 set seems to be more challenging due to the variable size (from small to big) of the birds, their view, the context/surrounding environment, and the few training samples.

Our model outperforms all the WSL methods in both F1

+ and F1 ??? with a large gap due mainly to its ability to discard non-discriminative regions which leaves it only with the region of interest, the bird in this case.

While our model shows improvements in pointwise localization, it is still far behind full supervision.

Similar improvements are observed on CUB data.

In the case of OxF dataset, our approach provides low F1

+ values compared to other WSL methods.

However, the latter are not far from the performance of the All-ones that predicts a constant mask.

Given the large size of flowers, predicting a mask that is active over all the image will easily lead to 56.10% of F + .

The best WSL methods for OxF are only better than All-ones by ??? 2%, suggesting that such methods have predicted a full mask in many cases.

In term of F1 ??? , our approach is better than all the WSL techniques.

All methods achieve low classification error on GlaS which implies that it represents an easy classification problem.

Surprisingly, the other methods seem to overfit on CUB5, while our model shows a robustness.

The other methods outperform our approach on CUB and OxF, although ours is still in a competitive range to half WSL methods.

Results obtained on both these datasets indicate that, compared to WSL methods, our approach is effective in terms of image classification and pointwise localization with more reliability in the latter.

Visual quality of our approach (Fig.3) shows that the predicted regions of interest on GlaS agree with the doctor methodology of colon cancer diagnostics where the glands are used as diagnostic tool.

Additionally, it deals well with multi-instances when there are multiple glands within the image.

On CUB5/CUB, our model succeeds to locate birds in order to predict its category which one may do in such task.

We notice that the head, chest, tail, or body particular spots are often parts that are used by our model to decide a bird's species, which seems a reasonable strategy as well.

On OxF dataset, we observe that our approach mainly locates the central part of pistil.

When it is not enough, the model relies on the petals or on unique discriminative parts of the flower.

In term of time complexity, the inference time of our model is the same as a standard fully convolutional network since the recursive algorithm is disabled during inference.

However, one may expect a moderate increase in training time that depends mainly on the depth of the recursion (see Sec.

B.3.2).

In this work, we present a novel approach for WSL at pixel-level where we impose learning relevant and irrelevant regions within the model with the aim to reduce false positive localization.

Evaluated on three datasets, and compared to state of the art WSL methods, our approach shows its effectiveness in accurately localizing regions of interest with low false positive while maintaining a competitive classification error.

This makes our approach more reliable in term of interpetability.

As future work, we consider extending our approach to handle multiple classes within the image.

Different constraints can be applied over the predicted mask, such as texture properties, shape, or other region constraints.

Predicting bounding boxes instead of heat maps is considered as well since they can be more suitable in some applications where pixel-level accuracy is not required.

Our recursive erasing algorithm can be further improved by using a memory-like mechanism that provides spatial information to prevent forgetting the previously spotted regions and promote localizing the entire region (Sec.

B.3).

Deep classification models tend to rely on small discriminative regions (Kim et al., 2017; Singh & Lee, 2017; Zhou et al., 2016) .

Thus, in our proposal, R ??? may still contain discriminative parts.

Following (Kim et al., 2017; Li et al., 2018; Pathak et al., 2015; Singh & Lee, 2017) , and in particular (Wei et al., 2017) , we propose a learning incremental and recursive erasing approach that drives M to seek complete discriminative regions.

However, in the opposite of (Wei et al., 2017) where such mining is done offline, we propose to incorporate the erasing within the backpropagation using an efficient and practical implementation.

This allows M to learn to seek discriminative parts.

Therefore, erasing during inference is unnecessary.

Our approach consists in applying M recursively before applying C within the same forward.

The aim of the recursion, with maximum depth u, is to mine more discriminative parts within the non-discriminative regions of the image masked by R ??? .

We accumulate all discriminative parts in a temporal mask R +, .

At each recursion, we mine the most discriminative part, that has been correctly classified by M, and accumulate it in R +, .

However, with the increase of u, the image may run out of discriminative parts.

Thus, M is forced, unintentionally, to consider non-discriminative parts as discriminative.

To alleviate this risk, we introduce trust coefficients that control how much we trust a mined discriminative region at each step t of the recursion for each sample i as follows,

where ??(t, i) ??? R + computes the trust of the current mask of the sample i at the step t as follows,

where exp ???t ?? encodes the overall trust with respect to the current step of the recursion.

Such trust is expected to decrease with the depth of the recursion (Belharbi et al., 2016) .

?? controls the slop of the trust function.

The second part of Eq.11 is computed with respect to each sample.

It quantifies how much we trust the estimated mask for the current sample i,

In Eq.12, H(p i ,p

Eq.12 ensures that at a step t, for a sample i, the current mask is trusted only if M correctly classifies the erased image, and does not increase the loss.

The first condition ensures that the accumulated discriminative regions belong to the same class, and more importantly, the true class.

Moreover, it ensures that M does not change its class prediction through the erasing process.

This introduces a consistency between the mined regions across the steps and avoids mixing discriminative regions of different classes.

The second condition ensures maintaining, at least, the same confidence in the predicted class compared to the first forward without erasing (t = 0).

The given trust in this case is equal to the probability of the true class.

The regions accumulator is initialized to zero at t = 0, R

is not maintained through epoches; M starts over each time processing the sample i.

This prevents accumulating incorrect regions that may occur at the beginning of the training.

In order to automatize when to stop erasing, we consider a maximum depth of the recursion u. For a mini-batch, we keep erasing as along as we do not reach u steps of erasing, and there is at least one sample with a trust coefficient non-zero (Eq.12).

Once a sample is assigned a zero trust coefficient, it is maintained zero all along the erasing (Eq.10) (Fig.4) .

Direct implementation of Eq.10 is not practical since performing a recursive computation on a large model M requires a large memory that increases with the depth u. To avoid such issue, we propose a practical implementation using gradient accumulation at M through the loss Eq.7; such implementation requires the same memory size as in the case without erasing.

An illustration of our proposed recursive erasing algorithm is provided in Fig.4 .

Alg.1 illustrates our implementation using accumulated gradient through the backpropagation within the localizer M. We note that this erasing algorithm is performed only during training.

In most neural networks libraries (Pytorch (pytorch.org), Chainer (chainer.org)), the upsacling operations using interpolation/upsamling have a non-deterministic backward.

This makes training Figure 4 : Illustration of the implementation of the proposed recursive incremental mining of disciminative parts within the backpropagation.

The recursive mining is performed only during training.

Algorithm 1 Practical implementation of our incremental recursive erasing approach during training for one epoch (or one mini-batch) using gradient accumulation.

Make a copy of X i : X i .

# Perform the recursion.

Accumulate gradients, and masks.

while t ??? u and stop is False do 8: Accumulate gradient:

Erase the discriminative parts: Compute:

Compute:

Update the total gradient:

unstable due to the non-deterministic gradient; and makes reproducibility impossible as well.

To avoid such issues, we detach the upsacling operation, in Eq.9, from the training graph and consider it as input data for C.

In this section, we provide more details on our experiments, analysis, and discuss some of the drawbacks of our approach.

We took many precautions to make the code reproducible for our model up to Pytorch's terms of reproducibility.

Please see the README.md file for the concerned section in the code.

We checked reproducibility up to a precision of 10 ???16 .

All our experiments were conducted using the seed 0.

We run all our experiments over one GPU with 12GB 5 , and an environment with 10 to 64 GB of RAM (depending on the size of the dataset).

Finally, this section shows more visual results, analysis, training time, and drawbacks.

We provide in Fig.5 some samples from each dataset's test set along with their mask that indicates the region of interest.

As we mentioned in Sec.4, we consider a subset from the original CUB-200-2011 dataset for preliminary experiments, and we referred to it as CUB5.

To build it, we select, randomly, 5 classes from the original dataset.

Then, pick all the corresponding samples of each class in the provided train and test set to build our train and test set (CUB5).

Then, we build the effective train set, and validation set by taking randomly 80%, and the left 20% from the train set of CUB5, respectively.

We provide the splits, and the code used to generate them.

Our code generates the following classes:

The following is the configuration we used for our model over all the datasets: Data 1.

Patch size (hxw): 480 ?? 480. (for training sample patches, however, for evaluation, use the entire input image).

2.

Augment patch using random rotation, horizontal/vertical flipping.

(for CUB5 only horizontal flipping is performed).

3.

Channels are normalized using 0.5 mean and 0.5 standard deviation.

4.

For GlaS: patches are jittered using brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05.

Model Pretrained resnet101 (He et al., 2016) as a backbone with (Durand et al., 2017) as a pooling score with our adaptation, using 5 modalities per class.

We consider using dropout (Srivastava et al., 2014) (with value 0.75 over GlaS and 0.85 over CUB5, CUB, OxF over the final map of the pooling function right before computing the score).

High dropout is motivated by (Ghiasi et al., 2018; Singh & Lee, 2017) .

This allows to drop most discriminative parts at features with most abstract representation.

The dropout is not performed over the final mask, but only on the internal mask of the pooling function.

As for the parameters of (Durand et al., 2017) , we consider their ?? = 0 since most negative evidence is dropped, and use kmax = kmin = 0.09.

u = 0, u = 4, ?? = 10, ?? = 0.5, ?? = 8.

For evaluation, our predicted mask is binarized using a 0.5 threshold to obtain exactly a binary mask.

All our presented masks in this work follows this thresholding.

Our F1 + , and F1 ??? are computed over this binary mask.

In this section, we provide more visual results over the test set of each dataset.

Over GlaS dataset (Fig.7, 8 ), the visual results show clearly how our model, with and without erasing, can handle multi-instance.

Adding the erasing feature allows recovering more discriminative regions.

The results over CUB5 (Fig.9, 10, 11, 12, 13) while are interesting, they show a fundamental limitation to the concept of erasing in the case of one-instance.

In the case of multi-instance, when the model spots one instance, then, erases it, it is more likely that the model will seek another instance which is the expected behavior.

However, in the case of one instance, and where the discriminative parts are small, the first forward allows mainly to spot such small part and erase it.

Then, the leftover may not be sufficient to discriminate.

For instance, in CUB5, in many cases, the model spots only the head.

Once it is hidden, the model is unable to find other discriminative parts.

A clear illustration to this issue is in Fig.9 , row 13.

The model spots correctly the head, but was unable to spot the body while the body has similar texture, and it is located right near to the found head.

We believe that the main cause of this issue is that the erasing concept forgets where discriminative parts are located since the mining iterations are done independently from each other in a sens that the next mining iteration is unaware of what was already mined.

Erasing algorithms seem to be missing this feature that can be helpful to localize the entire region of interest by seeking around all the previously mined disciminative regions.

In our erasing algorithm, once a region is erased, the model forgets about its location.

Adding a memory-like, or constraints over the spatial distribution of the mined discriminative regions may potentially alleviate this issue.

Another parallel issue of erasing algorithms is that once the most discriminative regions are erased it may not be possible to discriminate using the leftover regions.

This may explain why our model was unable to spot other parts of the bird once its head is erased.

Probably using soft-erasing (blur the pixel for example) can be more helpful than hard-erasing (set pixel to zero).

It is interesting to notice the strategy used by our model to localize some types of birds.

In the case of the 099.Ovenbird, it relies on the texture of the chest (white doted with black), while it localizes the white spot on the bird neck in the case of 108.White_necked_Raven.

One can notice as well that our model seems to be robust to small/occluded regions.

In many cases, it was able to spot small birds in a difficult context where the bird is not salient.

Visual results over CUB and OxF are presented in Fig.14, and Fig.15 , respectively.

Tab.3 and Tab.4 show the boosting impact of our erasing recursive algorithm in both classification and pointwise localization performance.

From Tab.4, we can observe that using our recursive algorithm adds a large improvement in F1 + without degrading F1 ??? .

This means that the recursion allows the model to correctly localize larger portions of the region of interest without including false positive parts.

The observed improvement in localization allows better classification error as observed in Tab.3.

The localization improvement can be seen as well in the precision-recall curves in Fig.6 .

Adding recursive computation in the backpropagation loop is expected to add an extra computation time.

Tab.5 shows the training time (of 1 run) of our model with and without recursion over identical computation resource.

The observed extra computation time is mainly due to gradient accumulation (line 12.

Alg.1) which takes the same amount of time as parameters' update (which is expensive to compute).

The forward and the backward are practically fast, and take less time compared to gradient update.

We do not compare the running between the datasets since they have different number/size of samples, and different pre-processing that it is included in the reported time.

Moreover, the size of samples has an impact over the total time during the training over the validation set.

Table 5 : Comparison of training time, of 1 run, over 400 epochs over GlaS and CUB5 of our model using identical computation resources (NVIDIA Tesla V100 with 12GB memory) when using our erasing algorithm (u = 4) and without using it (u = 0).

Ours (u = 0) 49min 65min Ours (u = 4) 90min (??? ??1.83) 141min (??? ??2.16)

Post-processing the output of fully convolutional networks using a CRF often leads to smooth and better aligned mask with the region of interest (Chen et al., 2015) .

To this end, we use the CRF implementation of (Kr??henb??hl & Koltun, 2011) 6 .

The results are presented in Tab.6.

Following the notation in (Kr??henb??hl & Koltun, 2011), we set w (1) = w (2) = 1.

We set, over all the methods, ?? ?? = 13, ?? ?? = 3, ?? ?? = 3 for 2 iterations, over GlaS, and ?? ?? = 19, ?? ?? = 11, ?? ?? = 5 for 5 iterations, over CUB5, CUB, and OxF. Tab.6 shows a slight improvement in term of F1 + and slight degradation in term of F1 ??? .

When investigating the processed masks, we found that the CRF helps in improving the mask only when the mask covers precisely large part of the region of interest.

In this case, the CRF helps spreading the mask over the region.

In the case where there is high false positive, or the mask misses largely the region, the CRF does not help.

We can see as well that the CRF increases slightly the false positive by spreading the mask out of the region of interest.

Since our method has small false positive -i.e., the produced mask covers mostly the region of interest and avoids stepping outside it-using CRF helps in improving both F1

+ and F1 ??? in most cases.

et al., 2018 ) is discarded since the produced plans do not form probability over the classes axe at pixel level which is required for the CRF input (Kr??henb??hl & Koltun, 2011) .

To preserve horizontal space, we rename the methods CAM-Avg, CAM-Max, CAM-LSE, Grad-CAM to Avg, Max, LSE, G-C, respectively.

Average precision-recall curves of our model over different test sets.

Figure 6 : Average precision-recall curve of the foreground and the background of our proposal using u = 0, u = 4 over each test set.

To be able to compute an average curve, the recall axis is unified for all the images to the axis [0, 1] with a step 1e ??? 3.

Then, the precision axis is interpolated with respect to the recall axis.

@highlight

A deep learning method for weakly-supervised pointwise localization that learns using image-level label only. It relies on conditional entropy to localize relevant and irrelevant regions aiming to minimize false positive regions.

@highlight

This work explores the problem of WSL using a novel design of regularization terms and a recursive erasing algorithm.

@highlight

This paper presents a new weakly supervised approach for learning object segmentation with image-level class labels.