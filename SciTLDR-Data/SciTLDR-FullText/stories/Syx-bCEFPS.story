Performing controlled experiments on noisy data is essential in thoroughly understanding deep learning across a spectrum of noise levels.

Due to the lack of suitable datasets, previous research have only examined deep learning on controlled synthetic noise, and real-world noise has never been systematically studied in a controlled setting.

To this end, this paper establishes a benchmark of real-world noisy labels at 10 controlled noise levels.

As real-world noise possesses unique properties, to understand the difference, we conduct a large-scale study across a variety of noise levels and types, architectures, methods, and training settings.

Our study shows that: (1) Deep Neural Networks (DNNs) generalize much better on real-world noise.

(2) DNNs may not learn patterns first on real-world noisy data.

(3) When networks are fine-tuned, ImageNet architectures generalize well on noisy data.

(4) Real-world noise appears to be less harmful, yet it is more difficult for robust DNN methods to improve.

(5) Robust learning methods that work well on synthetic noise may not work as well on real-world noise, and vice versa.

We hope our benchmark, as well as our findings, will facilitate deep learning research on noisy data.

Deep Neural Networks (DNNs) trained on noisy data demonstrate intriguing properties.

For example, DNNs are capable of memorizing completely random training labels but generalize poorly on clean test data (Zhang et al., 2017) .

When trained with stochastic gradient descent, DNNs learn patterns first before memorizing the label noise (Arpit et al., 2017) .

These findings inspired recent research on noisy data.

As training data are usually noisy, the fact that DNNs are able to memorize the noisy labels highlights the importance of deep learning research on noisy data.

To study DNNs on noisy data, previous work often performs controlled experiments by injecting a series of synthetic noises into a well-annotated dataset.

The noise level p may vary in the range of 0%-100%, where p = 0% is the clean dataset whereas p = 100% represents the dataset of zero correct labels.

The most commonly used noise in the literature is uniform (or symmetric) labelflipping noise, in which the label of each example is independently and uniformly changed to a random (incorrect) class with probability p. Controlled experiments on noise levels are essential in thoroughly understanding a DNN's properties across a spectrum of noise levels and faithfully comparing the strengths and weaknesses of different methods.

The synthetic noise enables researchers to experiment on controlled noise levels, and drives the development of theory and methodology in this field.

On the other hand, some studies were also verified on real-world noisy datasets, e.g. on WebVision (Li et al., 2017a) , Clothing-1M (Xiao et al., 2015) , Fine-grained Images (Krause et al., 2016) , and Instagram hashtags (Mahajan et al., 2018) , where the images are automatically tagged with noisy labels according to their surrounding texts.

However, these datasets do not provide true labels for the training images.

Their underlying noise levels are not only fixed but also unknown, rendering them infeasible for controlled studies on noise levels.

In this paper, we refer image-search noise in these datasets as "real-world noise" to distinguish it from synthetic label-flipping noise.

To study real-world noise in a controlled setting, we establish a benchmark of controlled real-world noisy labels, building on two existing datasets for coarse and fine-grained image classification: MiniImageNet (Vinyals et al., 2016) and Stanford Cars (Krause et al., 2013) .

We collect noisy labels using text-to-image and image-to-image search via Google Image Search.

Every training image is independently annotated by 3-5 workers, resulting in a total of 527,489 annotations over 147,108 images.

We create ten different noise levels from 0% to 80% by gradually replacing the original images with our annotated noisy images.

Our new benchmark will enable future research on the real-world noisy data with a controllable noise level.

We find that real-world noise possesses unique properties in its visual/semantic relevance and underlying class distribution.

To understand the differences, we conduct a large-scale study comparing synthetic noise, namely blue-pilled noise (or Blue noise), and real-world noise (or Red noise 1 ).

Specifically, we train DNNs across 10 noise levels, 7 network architectures, 6 existing robust learning methods, and 2 training settings (fine-tuning and training from random initialization).

Our study reveals several interesting findings.

First, we find that DNNs generalize much better on real-world noise than synthetic noise.

Our results verify Zhang et al. (2017) 's finding of deep learning generalization on synthetic noise.

However, we observe a considerably smaller generalization gap on real-world noise.

This does not mean that real-world noise is easier to tackle.

On the contrary, we find that real-world noise is more difficult for robust DNNs to improve.

Second, our results substantiate Arpit et al. (2017) 's finding that DNNs learn patterns first on noisy data.

But we find this behavior becomes insignificant on real-world noise and completely disappears on the fine-grained classification dataset.

This finding lets us rethink the role of "early stopping" (Yao et al., 2007; Arpit et al., 2017) on real-world noisy data.

Third, we find that when networks are fine-tuned, ImageNet architectures generalize well on noisy data, with a correlation of r = 0.87 and 0.89 for synthetic and real-world noise, respectively.

This finding generalizes Kornblith et al. (2019) 's finding, i.e. ImageNet architectures generalize well across clean datasets, to the noisy data.

Our contribution is twofold.

First, we establish a large benchmark of controlled real image search noise.

Second, we conduct perhaps the largest study in the literature to understand DNN training across a wide variety of noise levels and types, architectures, methods, and training settings.

We hope our benchmark along with our findings, resulted from a considerable amount of manual labeling effort (∼520K annotations) and computing resources (∼3K experiments), will facilitate future deep learning research on real-world noisy data.

Our main findings are summarized as follows:

1.

DNNs generalize much better on real-world noise than synthetic noise.

Real-world noise appears to be less harmful, yet it is more difficult for robust DNN methods to improve.

2.

DNNs may not learn patterns first on the real-world noisy data.

3.

When networks are fine-tuned, ImageNet architectures generalize well on noisy data.

4.

Adding noisy examples to a clean dataset may improve performance as long as the noise level is below a certain threshold (30% in our experiments).

Noisy Datasets: to understand deep learning's properties on noisy training data, research often conducted experiments across a series of levels of synthetic noises.

The most common one is uniform label-flipping noise (aka.

symmetric noise), in which the label of each example is independently and uniformly changed to a random (incorrect) class with a probability (Zhang et al., 2017; Arpit et al., 2017; Vahdat, 2017; Shu et al., 2019; Jiang et al., 2018; Han et al., 2018; Li et al., 2019; Arazo et al., 2019) .

The synthetic noisy dataset enables us to experiment on controlled noise levels, and drive the development of theory and methodology in this field.

Research have also examined other types of noise to better approximate the real-world noise distribution, including class-conditional noises (Patrini et al., 2017; Rolnick et al., 2017) , noises from other datasets (Wang et al., 2018) , etc.

However, these noises are still synthetic, generated from artificial distributions.

Furthermore, different types of synthetic noises may lead to inconsistent or even contradicting observations.

For example, Rolnick et al. (2017) experimented on a slightly different type of uniform noise and surprisingly found that DNNs are robust to massive label noise.

On the other hand, studies have also verified DNNs on real-world noisy datasets.

While other noise types exist, e.g. image omission and registration noise (Mnih & Hinton, 2012) or image corruption (Hendrycks & Dietterich, 2019) , the most common type consists of images that are automatically tagged according to their surrounding texts either by directly crawling the web pages, e.g. Clothing-1M (Xiao et al., 2015) , Instagram (Mahajan et al., 2018) , or by querying an image search engine, e.g. WebVision (Li et al., 2017a) .

Several studies have used these datasets.

For example, Guo et al. (2018) , Jiang et al. (2018) and Song et al. (2018) verified their model on WebVision.

Mahajan et al. (2018) trained large DNNs on noisy Instagram hashtags.

As these datasets did not provide true labels for the training examples, methods could only be tested on a fixed and, moreover, unknown noise level.

To the best of our knowledge, there have been no studies focused on investigating real noisy labels in a controlled setting.

The closest work to ours is annotating a small set for evaluation or estimating the noise level of the training data (Krause et al., 2016) .

Robust Deep Learning Methods: robust learning is experiencing a renaissance in the deep learning era.

Since training data usually contain noisy examples, the ability of DNNs to memorize all noisy training labels often leads to poor generalization on the clean test data.

Recent contributions based on deep learning handled noisy data in multiple directions including, e.g., dropout (Arpit et al., 2017) and other regularization techniques (Azadi et al., 2016; Noh et al., 2017) , label cleaning/correction (Reed et al., 2014; Goldberger & Ben-Reuven, 2017; Li et al., 2017b; , example weighting (Jiang et al., 2018; Ren et al., 2018; Shu et al., 2019; Jiang et al., 2015; Liang et al., 2016) , semi-supervised learning (Hendrycks et al., 2018; Vahdat, 2017) , data augmentation (Zhang et al., 2018; Cheng et al., 2019) , etc.

Few studies have systematically compared these methods across different noise types and training conditions.

In our study, we select and compare six methods from four directions: regularization, label learning, example weighting, and mixup augmentation.

See Section 4 for details.

These examined methods are selected because they (i) represent a reasonable coverage of different ways of handling noisy data; (ii) are comparable to the state-of-the-art on the commonly used CIFAR-100 with synthetic noise.

Our benchmark is built on two existing datasets: Mini-ImageNet (Vinyals et al., 2016) for coarse image classification and Stanford Cars (Krause et al., 2013) for fine-grained classification.

MiniImageNet provides images of size 84x84 with 100 classes from the ImageNet dataset (Deng et al., 2009) 2 .

We select 50,000 images for training and 5,000 from ImageNet for testing.

Note that unlike few-shot learning, we train and test on the same 100 classes.

The Stanford Cars contain 16,185 high-resolution images of 196 classes of cars (Make, Model, Year) splitting in a 50-50 into training and test set.

The standard test split is used.

To recap, let us revisit the construction of existing noisy datasets in the literature.

For the realworld noisy datasets (Xiao et al., 2015; Li et al., 2017a) , one automatically collects images for a class by matching the class name to the images' surrounding text (by web crawling or equivalently querying the crawled index).

The retrieved images include false positive (or noisy) examples, i.e.text match/search thinks an image is a positive when it is not.

As their training images are not manually labeled, the data noise level is fixed and unknown.

As a result, these datasets are unsuitable for controlled studies.

On the other hand, the synthetic noisy dataset is built on the well-labeled dataset.

The label of each training example is independently changed to a random incorrect class 3 with a probability p, called noise level, which indicates the percentage of training examples with false labels.

Since the ground-truth labels for every image are known, previous studies enumerate p to obtain datasets of different noise levels and use them in controlled experiments.

On balanced datasets, such as MiniImageNet and Stanford Cars used in our study, the above process is equivalent to first sampling p% training images from a class and then replacing them with the images uniformly drawn from other classes.

The drawback is that their noisy labels are artificial and do not follow the distribution of the real-world noise (Xiao et al., 2015; Li et al., 2017a; Krause et al., 2016) .

For our datasets, we follow the construction of synthetic datasets with only one difference, i.e. we draw false positive (noisy) examples from similar noise distributions as in existing real-world noisy datasets (Li et al., 2017a; Xiao et al., 2015) .

To be specific, we draw images using Google text-to-image search, which is commonly used to get noisy labels in prior works (Bootkrajang & Kabán, 2012; Li et al., 2017a; Krause et al., 2016; Chen & Gupta, 2015; Wang et al., 2014) .

In addition, we also include noisy examples using image-to-image search to enrich the type of label noises in our dataset.

We manually annotate every retrieved image to identify the ones with false labels.

For each class, we replace p% training images in the Mini-ImageNet and Stanford Cars datasets with these false-positive images.

We enumerate p in 10 different levels: {0%, 5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 80%} to study noisy labels in the controlled setting.

Since Mini-ImageNet and Stanford Cars are all collected from the web images, their true positive examples should follow a similar distribution as the added false positive images.

The constructed datasets hence contain label noise similar to existing real-world datasets and are suitable to be used in controlled experiments.

We sample noisy images using Google Image Search 4 in three steps: images collection, deduplication, and annotation.

In the first step, we combine images independently retrieved from two sources (1) text-to-image and (2) image-to-image search.

For text-to-image (or text-search in short), we formulate a text query for each class using its class name and broader category (e.g. cars) to retrieve the top 5,000 images.

For image-to-image search (or image-search), we query the search engine using every training image in Mini-ImageNet (50,000) and Stanford Cars (8, 144) .

This collects a large pool of similar images.

As different query images may retrieve the same image, we rank the retrieved images by their number of occurrences in the pool and remove images that occur less than 3 times.

Finally, we union text-search and image-search images, where the text-search images accounts for 72% of our final dataset.

For deduplication, following (Kornblith et al., 2019) , we run a CNN-based duplicate detector over all images to remove near-duplicates to any of the images in the test set.

All images are under the usage rights "free to use or share" 5 .

These images are then annotated on a cloud labeling platform of high-quality labeling professionals.

The annotator is asked to provide a binary label to indicate whether an image is a true positive of its class.

Every image is independently annotated by 3-5 workers to improve the labeling quality, and the final label is reached by majority voting.

In total, we have 372,428 annotations over 94,906 images on Mini-ImageNet, and 155,061 annotations over 51,687 images on Stanford Cars, out of which there are 28,691 and 12,639 image with false (noisy) labels.

Using these noisy image, we replace p% of the original training images in Mini-ImageNet and Stanford Cars.

Similar to the synthetic noise, p is made uniform across classes, e.g. p = 20% means that every class has roughly 20% false labels.

Besides, we also append all annotated images to the original datasets and obtain two larger augmented datasets.

The last two rows of Table 1 list the two augmented datasets and their underlying noise levels (19% and 21%).

We report their performance in Section 5.3.

For comparison, we also construct 10 uniform label-flipping datasets under the same noise levels.

For convenience, we will use Blue Noise to denote the synthetic noise and Red Noise for the real-world image search noise.

Table 1 summarizes the datasets.

The test set in each dataset (i.e. Mini-ImageNet and Stanford) is shared across all training conditions such that their results are comparable.

Red Mini-ImageNet is smaller because we ran out of noisy images for some common classes like "carton" and "hotdog".

For common classes, it becomes more difficult to get noisy images.

On average, we can get only one noisy label after labeling every 22 images.

The size difference in Mini-ImageNet may not be a problem for our study as they have similar test performance and we also verify on Stanford Cars whose size is the same for Blue and Red noise. (Li et al., 2017a) in which the true labels are not provided.

There are two noticeable differences between the blue and red noise:

• Real-world noisy images are more visually or semantically relevant to the true positive images.

• Real-world noisy images may come outside the fixed set of classes in the dataset.

For example, the noisy images of "ladybug" include "fly" and other bugs that do not belong to any of the class in Mini-ImageNet.

To understand their differences, we will compare Blue and Red noise in the rest of this paper.

This paper evaluates robust deep learning methods on the introduced benchmark.

We select six methods from four directions that deal with noisy training data: (a) regularization, (b) label/prediction correction, (c) example weighting, and (d) vicinal risk minimization.

These methods are selected because they (i) represent a reasonable coverage of recent work, (ii) are competitive to the state-ofthe-art on the common CIFAR-100 dataset with synthetic noise.

Our study seeks the answer to the following questions:

1.

How does their performance differ on synthetic versus real-world noise?

2.

What is their real performance gap when each method is extensively tuned for every noise level?

As the noise levels span across a wide range from 0% to 80%, we find the hyperparameters of robust DNNs are important.

By extensively searching hyperparameters for every noise level, these methods can be very competitive to the state-of-the-art on the commonly used CIFAR-100 with synthetic noise.

See Table 3 in the Appendix.

To answer the second question, we need to train a formidable number of experiments, e.g. 920 experiments on a single dataset!

Daunting as it may seem, the experiments are, however, necessary to ensure the improvement stems from the methodology as opposed to favorable hyperparameter settings.

To briefly introduce these methods, consider a classification problem with training set D = {x 1 , y 1 ), · · · , (x n , y n )}, where x i denotes the i th training image and y i ∈ [1, m] is an integer-valued noisy label over m possible classes.

Let g s (x i ; w) denote the prediction of our DNN, parameterized by w ∈ R d .

In vanilla training, we optimize the following objective:

where (y i ,g s (x i ,w)), or i for short, is the cross-entropy loss with Softmax.

θ is the decay parameter on the l 2 norm of the model parameters.

Weight decay and dropout are two classical regularization methods.

In Weight Decay: we tune θ in {e −5 , e −4 , e −3 , e −2 } and set its default value to e −4 which is the best value found on the ILSVRC12 dataset (Deng et al., 2009 ).

For Dropout (Srivastava et al., 2014) , following (Arpit et al., 2017) , we apply a large dropout ratio for noisy data and tune its keep probability in {0.1, 0.2, 0.3, 0.4, 0.5}. By default, we disable dropout in training following the advice in (Kornblith et al., 2019) .

We select two methods for label/prediction correction.

Reed (Reed et al., 2014 ) is a method for correcting the loss with the learned label.

The soft version is used for its better performance.

Let softmax(g s (x i ; w)) = [q i1 , . . .

, q im ] denote the prediction for the i th image.

Reed replaces the loss in Equation 1 with:

where 1 condition is 1 if the condition is true, 0 otherwise.

Reed weights the cross-entropy loss computed over the noisy label (first term) and the learned label (second term).

We tune the hyperparameter β in {0.95, 0.75, 0.5, 0.3}. (Goldberger & Ben-Reuven, 2017 ) introduces a convenient way to append a new layer to a DNN to learn noise transformation so as to "correct" the predictions.

Let z i denote the unknown true label for the i th image, and q i andq i denote the original and the learned prediction.

It estimates the prediction over the true label by the learned conditional probability:

This is implemented as a label transition layer parameterized by B ∈ R m×m .

We haveq i = softmax(B)q i , where the softmax is applied over B i,: , ∀i ∈ [1, m].

According to the paper, we initialize B = log((1 − )I + × 1 m−1 J ), where I and J are the identity and the all-one matrix; is a small constant set to e −6 .

MentorNet (Jiang et al., 2018 ) is a competitive example-weighting method, which aims to assign smaller weights to noisy examples.

It introduces the learnable latent weight variable v for every training example and adds a regularization term over the weight variable:

When w is fixed, solving Equation 4 yields a weighting function that is monotonically decreasing with the example loss i :

where λ 1 and λ 2 are parameters.

We employ the predefined MentorNet to compute the example weight in Equation 5 at the mini-batch level.

It tracks the moving average of the p-percentile of the loss inside every mini-batch and sets λ 1 and λ 2 accordingly.

Following the paper, we set the burn-in epoch to 10-20% of the total training epochs and tune the hyperparameter p-percentile in {85%, 75%, 55%, 35%}.

Mixup (Zhang et al., 2018 ) is a simple and effective method for robust training.

It minimizes the vicinal risk (ŷ i ,g s (x i ,w)) calculated from:

where y i , y j ∈ R m are two one-hot label vectors and the pairs (x i , y i ) and (x j , y j ) are drawn at random from the same mini-batch.

The mixing weight λ is sampled from a conjugate prior Beta distribution λ ∼ Beta(α, α) for α > 0.

Following the paper, we search the hyperparameter α in {1, 2, 4, 8} for noisy training data.

For each method, we examine two training settings: (i) fine-tuning from the ImageNet checkpoint and (ii) training from scratch.

For method comparison, Inception-ResNet-V2 ) is used as the default network architectures.

For vanilla training, we also experiment with six other architectures: EfficientNet-B5 (Tan & Le, 2019) , MobileNet-V2 , ResNet-50 and ResNet-101 , Inception-V2 (Ioffe & Szegedy, 2015) , and Inception-V3 .

The top-1 accuracy of these architectures on the ImageNet ILSVRC 2012 validation ranges from 71.6% to 83.6%.

We first train our networks to get the best result on the clean training dataset and fix the setting across all noise levels, e.g. learning rate schedule and maximum epochs to train.

See Section A.1 in the Appendix for the detailed implementation.

5.1 VANILLA TRAINING Fig. 2 plots the training curve (gray) and the test curve (colored) on Blue and Red noisy benchmarks using vanilla training.

The first two columns of Fig. 2 show the synthetic (in blue) and real-world noise (in red), respectively, where the x-axis is the training step.

The colored belt plots the 95% confidence interval over 10 noise levels and the solid curve highlights the 40% noise level.

The first row in each sub-figure is training from scratch and the second row is fine-tuning.

Two classification accuracies on the test set are compared.

The peak accuracy denotes the maximum test accuracy throughout the training.

The converged accuracy is the test accuracy after training has converged, which, for most methods, means the training accuracy reaches 100%.

See Fig. 2a for an example.

DNNs generalize much better on the red noise.

By comparing the width of Red and Blue belt in Fig. 2 under the same training condition (row-wise), we can see that the test accuracy's standard deviation is considerably smaller on Red noise than on Blue noise.

This indicates a smaller difference in test accuracy between the clean and the noisy training data, suggesting that DNNs generalize better on Red noise.

For a clearer illustration, as an example, we plot networks trained from scratch on Mini-ImageNet in Fig. 3 .

Specifically, Fig. 3a shows the training accuracy of the 0%, 40%, and 60% noise levels along with the training step.

Fig. 3b shows the difference in final converged test accuracies, relative to the accuracy on the clean training data, under 10 noise levels.

As the training accuracy is perfect on all noise levels, the drop in the test accuracy can be regarded as an indicator of the generalization gap.

The blue curves in Fig 3b confirm Zhang et al. (2017) 's finding that DNNs generalize poorly as synthetic noise levels increase.

For example, the test accuracy of EfficientNet will drop by 85% when Blue noise reaches 60%.

On the real-world noise, however, the gap is considerably smaller, e.g. the drop on 60% Red noise is only 23%.

This pattern holds for all architectures in our study.

See the curves for EfficientNet, Inception-ResNet, and MobileNet in Fig. 3b .

Fig. 2 and Fig. 3b suggest that DNNs generalize better on real-world noisy data.

This phenomenon is probably due to the two properties discussed in Section 3: (i) red noisy images are similar to clean training images, and hence bring less change to the training (ii) red noisy images are often sampled out of the training classes.

This may make them less confusing for the fixed training classes.

DNNs may not learn patterns first on the red noise.

The third column of Fig. 2 illustrates the relative drop between the peak and converged test accuracy, where the x-axis is the noise level and the y-axis computes the relative difference in percentage, i.e. (peak -converged)/peak accuracy.

We see that there is almost no drop on the clean data (x = 0).

The drop starts to grow as the noise level increases.

This substantiates Arpit et al. (2017) 's finding that DNNs learn patterns first on noisy data.

Early stopping which terminates training at the peak accuracy is thus effective on Blue noise.

See the red curves in the third column of Fig. 2 .

This suggests DNNs may not learn patterns first on real-world noisy data, especially for the fine-grained classification task.

Our hypothesis is that Blue noise images are sampled uniformly from a fixed number of classes, and the uniform errors can be mitigated in the DNN's early training stage before it memorizes all noisy labels.

Real-world noisy images are sampled non-uniformly from an infinite number of classes, making it difficult for DNNs to identify meaningful patterns in the red noise.

ImageNet architectures generalize well on noisy data when the networks are fine-tuned.

Comparing the first and second rows in Fig. 2 , we observe that the test accuracy for fine-tuning is higher than that for training from scratch on both Red and Blue noise.

This is consistent with Table 6 in the Appendix.

In Fig. 4 , we compare the fine-tuning performance using ImageNet architectures and compute the correlation coefficient, where the y-axis is the peak accuracy and the x-axis is the top-1 accuracy of the architecture on ImageNet ILSVRC 2012 validation.

The bar plots the 95% confidence interval across 10 noise levels, where the center dot marks the mean.

As it shows, there is a decent correlation between the ImageNet accuracy and the test accuracy on noisy data

In Fig. 5, Fig. 6, Fig. 8 , and Fig. 9 , we compare the robust deep learning methods on Blue and Red noise, where the x-axis shows the peak accuracy and its corresponding 95% confidence interval over different hyperparameters.

We mainly compare the peak accuracy which, as shown in recent studies (Song et al., 2018; Arazo et al., 2019) , is more challenging to improve.

We also list their converged accuracies in the Appendix for reference.

First of all, we observe big performance variances (or confidence intervals) in most methods, suggesting that hyperparameters are important for robust learning methods.

The best hyperparameter usually varies for different noise levels, and our observation would be very different if the methods were not extensively tuned for each noise level.

Red noise is more difficult for robust DNNs to improve.

Although robust DNNs are able to improve the performance of vanilla training across all noise levels and types, the improvement is noticeably smaller on Red noise.

For example, the average improvement of the best robust method Comparing methods, we find that no single method performs the best across all noise levels and types.

Methods that work well on synthetic noise may not work as well on real-world noise, and vice versa.

To be specific, Dropout is effective for training from scratch on Stanford Cars and achieves the best accuracy in 20 trials.

Weight Decay mainly benefits fine-tuning but only to a small extent (6 best trials).

Reed achieves the best result in 10 trials, all of which are fine-tuning on MiniImageNet.

S-Model yields marginal gains over the vanilla training.

Finally, MentorNet and Mixup achieve the best accuracy in 21 and 23 trials, respectively.

Unlike MentorNet, Mixup seems to be more effective on Red noise, suggesting that pair-wise image mixing is more effective than example weighting on real-world noise.

Best accuracy on the original data (train from scratch)

Best accuracy on the original data (fine-tune) In practice, a simple technique to improve performance is to add noisy or weakly-labeled examples to an existing clean dataset.

It is convenient because noisy examples can be automatically collected without any manual labeling effort.

Multiple studies have shown that noisy examples can be beneficial for training e.g. (Krause et al., 2016; Liang et al., 2016; Mahajan et al., 2018) .

In particular, Guo et al. (2018) found that adding WebVision noisy images to clean ImageNet can lead to a performance gain.

On the other hand, it may be disadvantageous if the true labels of all added examples are incorrect.

An interesting question that has yet been answered is what is the maximum noise level at which the added noisy data can be useful?.

Our benchmark allows for investigating this question using real-world noisy data in a controlled setting.

To do so, we add ∼30K and ∼25K additional images to the original training sets of MiniImageNet and Stanford Cars, respectively.

The sizes are selected to be 60% and 300% of the original datasets to examine different settings.

We control the noise level of the added images in 10 different levels from 0% to 80%.

Fig. 7 shows the peak test accuracy (the y-axis) across ten noise levels (the x-axis).

The dashed line represents the best accuracy obtained on the original dataset.

As it shows, the test accuracy generally decreases as the noise level grows.

Small noise is useful but large noise can hurt the performance.

The equilibrium occurs between 30% to 50%.

In all cases, it is useful if the noise level is under 30%.

When the noise level is below 30%, it also improves the full augmented datasets in Table 1 , the accuracies of which are 0.770 (trained from scratch) and 0.865 (fine-tuned) on Mini-ImageNet, and 0.927 and 0.932 on Stanford Cars.

Note that the 30% threshold just represents the observation on our benchmark and should not be overinterpreted.

In this paper, we established a benchmark for controlled real-world noise.

On the benchmark, we conducted a large-scale study to understand deep learning on noisy data across a variety of settings.

Our studies revealed a number of new findings, improving our understanding of deep learning on noisy data.

By comparing six robust deep learning methods, we found that real-world noise is more difficult to improve and methods that work well on synthetic noise may not work as well on realworld noise, and vice versa.

This encourages future research to be also carried out on controlled real-world noise.

We hope our benchmark, as well as our findings, will facilitate deep learning research on real-world noisy data.

This subsection presents the implementation details.

Architectures: Table 2 lists the parameter count and input image size for each network architecture examined.

We obtained their model checkpoints trained on the ImageNet 2012 dataset from TensorFlow Slim 6 , EfficienNet TPU 7 , and from the authors of (Kornblith et al., 2019) .

The last two columns list the top-1 accuracy of our obtained models along with the accuracy reported in the original paper.

As shown, the top-1 accuracy of these architectures on the ImageNet ILSVRC 2012 validation ranges from 71.6% to 83.6%, making them suitable for our correlation study.

Training from scratch (random initialization): for vanilla training, we trained each architecture on the clean dataset (0% noise level) to find the optimal training setting by grid search.

Our grid search consisted of 6 start learning rates of {1.6, 0.16, 1.0, 0.5, 0.1, 0.01} and 3 learning rate decay epochs of {1, 2, 3}. The exponential learning rate decay factor was fixed to 0.975.

We trained each network to full convergence to ensure its training accuracy reached 1.0.

The maximum epoch to train was 200 on Mini-ImageNet (Red and Blue) and 300 epochs on Stanford Cars (Blue and Red), where the learning rate warmup (Goyal et al., 2017) was used in the first 5 epochs.

The training was using Nesterov momentum with a momentum parameter of 0.9 at a batch size of 64, taking an exponential moving average of the weights with a decay factor of 0.9999.

We had to reduce the batch size to 8 for EfficientNet for its larger image input.

Following Kornblith et al. (2019) , our vanilla training was with batch normalization layers but without label smoothing, dropout, or auxiliary heads.

We employed the standard prepossessing in EfficientNet 8 for data augmentation and evaluated on the central cropped images on the test set.

Training in this way, we obtained reasonable performance on the clean Stanford Cars test set.

For example, our Inception-ResNet-V2 got 90.8 (without dropout) and 92.4 (with dropout) versus 89.9 reported in (Kornblith et al., 2019) .

Fine-tuning from ImageNet checkpoint: for fine-tuning experiments, we initialized networks with ImageNet-pretrained weights.

We used a similar training protocol for fine-tuning as training from scratch.

The start learning rate was stable in fine-tuning so we fixed it to 0.01 and only searched the learning rate decay epochs in {1, 3, 5, 10}. Learning rate warmup was not used in fine-tuning.

As fine-tuning converges faster, we scaled down the maximum number of epoch to train by a factor of 2.

In this case, the training accuracy was still able to reach 1.0.

Training in this way, we obtained reasonable performance on the clean Stanford Cars test set.

For example, our Inception-ResNet-V2 got 92.4 versus 92.0 reported in (Kornblith et al., 2019) and our EfficientNet-B5 got 93.8% versus 93.6% reported in (Tan & Le, 2019) .

Robust deep learning method comparison: For method comparison, we used Inception-ResNet as the default network.

We fixed the optimal setting found on the clean training set for all methods and all noise levels.

We found the hyperparameter for robust DNNs is important.

Therefore, we extensively searched the hyperparameter of each method for every noise level and every noise type, using the range discussed in the main manuscript.

Comparing 6 methods using all hyperparameters, 10 noise levels, 2 noise types, and 2 training conditions led to a total of 1,840 experiments on two datasets.

The mean and 95% confidence interval over different hyperparameters were shown in Fig. 5, Fig. 6, Fig. 8 , and Fig. 9 in the main manuscript.

The best peak accuracy (along with the converged accuracy) could be found in Table 4 to Table 7 .

For Dropout, as it converges slower, we added another 100 epochs to its maximum epochs to train.

This subsection shows that our examined robust learning methods are able to achieve the performance that is comparable to, or even better than, the state-of-the-art on the commonly used synthetic noisy dataset in the literature.

Following (Jiang et al., 2018; Shu et al., 2019; Arazo et al., 2019) , the noisy CIFAR-100 data are of uniform label-flipping noise, where the label of each image is independently changed to a uniform (incorrect) class with probability p, where p is the noise level and is set to 20%, 40%, 60%, and 80%.

The clean test images on CIFAR-100 are used for evaluation.

Table 3 shows the results where † marks our implementation of MentorNet (Jiang et al., 2018) 9 and Mixup (Zhang et al., 2018) 10 under the best hyperparameter setting 11 .

First, the comparison between Row 2 and Row 7 in the table shows extensive hyperparameter search leads to about a 2% gain over the published results in (Jiang et al., 2018) .

Second, comparing Row 5 and 6 to Row 4, it shows that our examined methods are comparable to the state-of-the-art except for the 40% noise level.

Finally, comparing Row 7 and 8 to others, we find that our examined methods are able to achieve the best result on this dataset.

In this section, we exclude the images brought by image-to-image search and reconduct our main experiments.

This data subset contains only the images from Google text-to-image search and hence is similar to the noisy datasets used in previous studies (Li et al., 2017a; Krause et al., 2016; Chen & Gupta, 2015) (note that existing datasets do not have controlled noise).

Our goal is to verify whether our findings still hold on this new data subset.

We use dark red to denote the text-to-image search only noise and compare it with the red noise reported in the paper.

To be specific, the vanilla training and test curves are compared in Fig. 12 .

The generalization errors are compared in Fig. 13 .

The finetuning models on different ImageNet architectures are compared in Fig. 14.

As we see in all cases, text-to-image only noise (dark red) performs very similarly to red noise studied in the paper.

We can confirm the our findings still hold.

That is (1) DNNs generalize much better on the dark red noise (Fig. 12 and Fig. 13 ).

(2) DNNs may not learn patterns first on the dark red noise (Fig. 12) .

(3) ImageNet architectures generalize well on noisy data when the networks are fine-tuned (Fig. 14) .

The above results show that the red noise studied in our paper is consistent with the text-to-image only noise.

As red noise contains a more diverse types of label noises, we keep it as our main datasets.

This subsection presents the numerical results for the best trial in Fig. 5, Fig. 6, Fig. 8 , and Fig. 9 in the main manuscript.

@highlight

We establish a benchmark of controlled real noise and reveal several interesting findings about real-world noisy data.

@highlight

This paper compares 6 existing noisy label learning methods in two training settings: from scratch, and finetuning.

@highlight

The authors establish a large dataset and benchmark of controlled real-world noise for performing controlled experiments on noisy data in deep learning.