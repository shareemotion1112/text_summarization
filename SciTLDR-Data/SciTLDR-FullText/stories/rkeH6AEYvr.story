The available resolution in our visual world is extremely high, if not infinite.

Existing CNNs can be applied in a fully convolutional way to images of arbitrary resolution, but as the size of the input increases, they can not capture contextual information.

In addition, computational requirements scale linearly to the number of input pixels, and resources are allocated uniformly across the input, no matter how informative different image regions are.

We attempt to address these problems by proposing a novel architecture that traverses an image pyramid in a top-down fashion, while it uses a hard attention mechanism to selectively process only the most informative image parts.

We conduct experiments on MNIST and ImageNet datasets, and we show that our models can significantly outperform fully convolutional counterparts, when the resolution of the input is that big that the receptive field of the baselines can not adequately cover the objects of interest.

Gains in performance come for less FLOPs, because of the selective processing that we follow.

Furthermore, our attention mechanism makes our predictions more interpretable, and creates a trade-off between accuracy and complexity that can be tuned both during training and testing time.

Our visual world is very rich, and there is information of interest in an almost infinite number of different scales.

As a result, we would like our models to be able to process images of arbitrary resolution, in order to capture visual information with arbitrary level of detail.

This is possible with existing CNN architectures, since we can use fully convolutional processing (Long et al. (2015) ), coupled with global pooling.

However, global pooling ignores the spatial configuration of feature maps, and the output essentially becomes a bag of features 1 .

To demonstrate why this an important problem, in Figure 1 (a) and (b) we provide an example of a simple CNN that is processing an image in two different resolutions.

In (a) we see that the receptive field of neurons from the second layer suffices to cover half of the kid's body, while in (b) the receptive field of the same neurons cover area that corresponds to the size of a foot.

This shows that as the input size increases, the final representation becomes a bag of increasingly more local features, leading to the absence of coarselevel information, and potentially harming performance.

We call this phenomenon the receptive field problem of fully convolutional processing.

An additional problem is that computational resources are allocated uniformly to all image regions, no matter how important they are for the task at hand.

For example, in Figure 1 (b), the same amount of computation is dedicated to process both the left half of the image that contains the kid, and the right half that is merely background.

We also have to consider that computational complexity scales linearly with the number of input pixels, and as a result, the bigger the size of the input, the more resources are wasted on processing uninformative regions.

We attempt to resolve the aforementioned problems by proposing a novel architecture that traverses an image pyramid in a top-down fashion, while it visits only the most informative regions along the way.

The receptive field problem of fully convolutional processing.

A simple CNN consisted of 2 convolutional layers (colored green), followed by a global pooling layer (colored red), processes an image in two different resolutions.

The shaded regions indicate the receptive fields of neurons from different layers.

As the resolution of the input increases, the final latent representation becomes a bag of increasingly more local features, lacking coarse information.

(c) A sketch of our proposed architecture.

The arrows on the left side of the image demonstrate how we focus on image sub-regions in our top-down traversal, while the arrows on the right show how we combine the extracted features in a bottom-up fashion.

In Figure 1 (c) we provide a simplified sketch of our approach.

We start at level 1, where we process the input image in low resolution, to get a coarse description of its content.

The extracted features (red cube) are used to select out of a predefined grid, the image regions that are worth processing in higher resolution.

This process constitutes a hard attention mechanism, and the arrows on the left side of the image show how we extend processing to 2 additional levels.

All extracted features are combined together as denoted by the arrows on the right, to create the final image representation that is used for classification (blue cube).

We evaluate our model on synthetic variations of MNIST (LeCun et al., 1998 ) and on ImageNet (Deng et al., 2009 ), while we compare it against fully convolutional baselines.

We show that when the resolution of the input is that big, that the receptive field of the baseline 2 covers a relatively small portion of the object of interest, our network performs significantly better.

We attribute this behavior to the ability of our model to capture both contextual and local information by extracting features from different pyramid levels, while the baselines suffer from the receptive field problem.

Gains in accuracy are achieved for less floating point operations (FLOPs) compared to the baselines, due to the attention mechanism that we use.

If we increase the number of attended image locations, computational requirements increase, but the probability of making a correct prediction is expected to increase as well.

This is a trade-off between accuracy and computational complexity, that can be tuned during training through regularization, and during testing by stopping processing on early levels.

Finally, by inspecting attended regions, we are able to get insights about the image parts that our networks value the most, and to interpret the causes of missclassifications.

Attention.

Attention has been used very successfully in various problems (Bahdanau et al., 2014; Xu et al., 2015; Larochelle & Hinton, 2010; Gregor et al., 2015; Denil et al., 2012) .

Most similar to our work, are models that use recurrent neural networks to adaptively attend to a sequence of image regions, called glimpses (Ba et al. (2014) ; Mnih et al. (2014) ; Eslami et al. (2016) ; Ba et al. (2016) ; Ranzato (2014) ).

There are notable technical differences between such models and our approach.

However, the difference that we would like to emphasize, is that we model the image content as a hierarchical structure, and we implicitly create a parsing tree (Zhu et al. (2007) ), where each node corresponds to an attended location, and edges connect image regions with sub-regions (an example is provided in Appendix A.1).

If we decide to store and reuse information, building such a tree structure offers a number of potential benefits, e.g. efficient indexing.

We consider this an important direction to explore, but it is beyond the scope of the current paper.

Multi-scale representations.

We identify four broad categories of multi-scale processing methods.

(1) Image pyramid methods extract multi-scale features by processing multi-scale inputs (Eigen et al. (2014); Pinheiro & Collobert (2014); Najibi et al. (2018) ).

Our approach belongs to this category.

(2) Encoding schemes take advantage of the inherently hierarchical nature of deep neural nets, and reuse features from different layers, since they contain information of different scale (Liu et al. (2016) ; He et al. (2014) ; ).

(3) Encoding-Decoding schemes follow up the feed-forward processing (encoding) with a decoder, that gradually recovers the spatial resolution of early feature maps, by combining coarse with fine features (Lin et al. (2017); Ronneberger et al. (2015) ).

(4) Spatial modules are incorporated into the feed forward processing, to alter the feature extraction between layers (Yu & Koltun (2015) ; Chen et al. (2017) ; Wang et al. (2019) ).

Computational efficiency.

We separate existing methods on adjusting the computational cost of deep neural networks, into four categories.

Levi & Ullman (2018) ).

This is the strategy that we follow in our architecture.

We present our architecture by walking through the example in Figure 2 , where we process an image with original resolution of 128 × 128 px ( 1 in the top left corner).

In the fist level, we downscale the image to 32 × 32 px and pass it through the feature extraction module, in order to produce a feature vector V 1 that contains a coarse description of the original image.

The feature extraction module is a CNN that accepts inputs in a fixed resolution, that we call base resolution.

We can provide V 1 as direct input to the classification module in order to get a rapid prediction.

If we end processing here, our model is equivalent to a standard CNN operating on 32 × 32 px inputs.

However, since the original resolution of the image is 128 × 128 px, we can take advantage of the additional information by moving to the second processing level.

In the second level, we feed the last feature map, F 1 , of the feature extraction module, to the location module.

The location module considers a number of candidate locations within the image described by F 1 , and predicts how important it is to process in higher detail each one of them.

In this particular example, the candidate locations form a 2 × 2 regular grid ( 2 ), and the location module yields 4 probabilities, that we use as parameters of 4 Bernoulli distributions in order to sample a hard attention mask ( 3 ).

Based on this mask, we crop the corresponding regions, and we pass them through the feature extraction module, creating V 21 and V 23 .

If we want to stop at this processing level, we can directly pass V 21 and V 23 through the aggregation module (skipping the merging module 4 ).

The aggregation module combines the features from individual image regions into a single feature vector V , that describes the original image solely based on fine information.

This means that V agg 1 is complementary to V 1 , and both vectors are combined by the merging module, which integrates fine information (V agg 1 ) with its context (V 1 ), creating a single comprehensive representation V 1 .

Then, V 1 can be used for the final prediction.

We can extend our processing to a third level, where F 21 and F 23 are fed to the location module to create two binary masks.

No locations are selected from the image patch described by V 23 , and the aggregation module only creates V agg 21 .

Then, we start moving upwards for the final prediction.

In Appendix A.2 we provide additional details about the modules of our architecture.

In Appendix A.3 we express the feature extraction process with a single recursive equation.

Our model is not end-to-end differentiable, because of the Bernoulli sampling involved in the location selection process.

To overcome this problem, we use a variant of REINFORCE (Williams, 1992):

where N · M is the number of images we use for each update, x i is the ith image, y i is its label, and w are the parameters of our model.

p(l i |x i , w) is the probability that the sequence of locations l i is attended for image x i , and p(y i |l i , x i , w) is the probability of predicting the correct label after attending l i .

The size of our original batch B is N , but in the derivation of (1) we approximate with a Monte Carlo estimator of M samples, the expectation

for each image x i in B. Based on this, for simplicity we just consider that our batch has size N · M .

b is a baseline that we use to reduce the variance of our estimators, and λ f is a weighting hyperparameter.

The first term of L F allows us to update the parameters in order to maximize the probability of each correct label.

The second term allows us to update the location selection process, according to the utility of attended locations for the prediction of the correct labels.

In Appendix A.4.1 we provide the exact derivation of learning rule (1).

We experimentally identified two problems related to (1).

First, our model tends to attend all the available locations, maximizing the computational cost.

To regulate this, we add the following term:

where p l i t approximates the expected number of attended locations for image x i base on l i , and

t calculates the average expected number of attended locations per image in our augmented batch of N M images.

The purpose of R t is to make the average number of attended locations per image equal to c t , which is a hyperparameter selected according to our computational cost requirements.

λ t is simply a weighting hyperparameter.

The second problem we identified while using learning rule (1), is that the learned attention policy may not be diverse enough.

In order to encourages exploration, we add the following term:

where p k is the average probability of attending the kth out of the g candidate locations, that the location module considers every time it is applied during the processing of our N M images.

R r encourages the location module to attend with the same average probability c r , locations that are placed at different regions inside the sampling grid.

λ r is a weighting hyperparameter.

In Appendix A.4.2 we provide additional details about terms (2) and (3).

Our final learning rule is the following:

Gradual Learning.

The quality of the updates we make by using (4), depends on the quality of the Monte Carlo estimators.

When we increase the number of processing levels that our model can go through, the number of location sequences that can be attended increases exponentially.

Based on this, if we allow our model to go through multiple processing levels, and we use a small number of samples in order to have a moderate cost per update, we expect our estimators to have high variance.

To avoid this, we separate training into stages, where in the first stage we allow only 2 processing levels, and at every subsequent stage an additional processing level is allowed.

This way, we expect the location module to gradually learn which image parts are the most informative, narrowing down the location sequences that have a considerable probability to be attended at each training stage, and allowing a small number of samples to provide acceptable estimators.

We call this training strategy gradual learning, and the learning rule that we use at each stage s, is the following:

where L s r is equivalent to (4), with s superscripts indicating that the hyperparameters of each term can be adjusted at each training stage.

The maximum number of possible training stages depends on the resolution of the training images.

Multi-Level Learning.

By following the gradual learning paradigm, a typical behavior that we observe is the following.

After the first stage of training, our model is able to classify images with satisfactory accuracy by going through 2 processing levels.

After the second stage of training, our model can go through 3 processing levels, and as we expect, accuracy increases since finer information can be incorporated.

However, if we force our model to stop processing after 2 levels, the obtained accuracy is significantly lower than the one we were able to achieve after the first stage of training.

This is a behavior that we observe whenever we finish a new training stage, and it is an important problem, because we would like to have a flexible model that can make accurate predictions after each processing level.

To achieve this, we introduce the following learning rule:

where L z r is learning rule (5) with z + 1 processing levels allowed for our model.

Each λ s z is a hyperparameter that specifies the relative importance of making accurate predictions after processing level z + 1, while we are at training stage s.

We experiment with two datasets, MNIST (LeCun et al., 1998) and ImageNet (Deng et al., 2009) .

MNIST is a small dataset that we can easily modify in order to test different aspects of our models' Figure 3 : Example images from our MNIST-based datasets, along with the attended locations of M 28 3 models.

We blur the area outside the attended locations, because it is processed only in lower resolution during the first processing level.

This way we aim to get a better understanding of what our models "see".

behavior, e.g. the localization capabilities of our attention mechanism.

ImageNet has over a million training images, and allows us to evaluate our model on a large scale.

Data.

MNIST is a dataset with images of handwritten digits that range from 0 to 9, leading to 10 different classes.

All images are grayscale, and their size is 28 × 28 pixels.

The dataset is split in a training set of 55, 000 images, a validation set of 5, 000 images, and a test set of 10, 000 images.

We modify the MNIST images by placing each digit at a randomly selected location inside a black canvas of size 56 × 56.

We refer to this MNIST variation as plain MNIST, and we further modify it to create noisy MNIST and textured MNIST.

In the case of noisy MNIST, we add salt and pepper noise by corrupting 10% of the pixels in every image, with a 0.5 ratio between black and white noise.

In the case of textured MNIST, we add textured background that is randomly selected from a large image depicting grass.

Example images are provided in the first column of Figure 3 .

, each composed of a version M i of our architecture, and a corresponding fully convolutional baseline BL i for comparison.

Ideally, we would like the feature extraction processes of the models that we compare to be consisted of the same layers, in order to eliminate important factors of performance variation like the type or the number of layers.

To this end, in every pair of models, the baseline BL i is equivalent to the feature extraction module of M i followed by the classification module, with 2 fully connected layers in between.

The 2 fully connected layers account for the aggregation and merging modules, which could be considered part of the feature extraction process in M i .

We create 3 pairs of models because we want to study how our architecture performs relatively to fully convolutional models with different receptive fields.

To achieve this, BL 1 has 1 convolutional layer and receptive field 3 × 3 px, BL 2 has 2 convolutional layers and receptive field 8 × 8 px, and BL 3 has 3 convolutional layers and receptive field 20 × 20 px.

We would like to note that all models {M i } 3 i=1 have base resolution 14 × 14 px, and their location modules consider 9 candidate locations which belong to a 3 × 3 regular grid with 50% overlap.

In Appendix A.5.1 we provide the exact architectures of all models.

Training.

We describe how we train one pair of models (M i , BL i ) with one of the 3 MNIST-based datasets that we created.

The same procedure is repeated for all datasets, and for every pair of models.

The original resolution of our dataset is 56 × 56 px, and we rescale it to 2 additional resolutions that Figure 4 : Experimental results on plain, textured and noisy MNIST.

The differences in accuracy between many models were very small, and as a result, in the provided graphs we report the average of 20 different evaluations on the validation set, where each time we randomly change the positions of the digits inside the images.

For textured and noisy MNIST, we randomly change the background and the noise pattern of each image as well.

are equal to 28 × 28 px, and 14 × 14 px (example images are provided in the first 3 columns of Figure 3 ).

We split our training procedure in 3 sessions, where at each session we train our models with images of different resolution.

In the first session we use images of size 14 × 14 px, and we refer to the resulting models as BL 14 i and M 14 i .

We note that since the resolution of the input is equal to the base resolution of M i , our model can go through just 1 processing level and the location module is not employed.

In our second training session we use images of resolution 28 × 28 px, and the increased resolution of the input allows M i to use the location module and extend processing to 2 processing levels.

Based on this, we are able to train M i multiple times by assigning different values to hyperparameter c t (2), resulting to models that attend a different average number of locations per image, and as a result, they have different average computational cost and accuracy.

We refer to the models from this training session as BL 28 i and M

, where n c is the average number of locations that M i attended on the validation set, while trained with c t = c. We also define M 28 i as the set of all M 28,nc i models we trained in this session.

In the third training session we use images of resolution 56 × 56 px, and M i is able to go through 3 processing levels.

Following our previous notation, we refer to the models of this session as BL .

In Appendix A.5.2 we provide additional details about the training sessions, along with the exact hyperparameters that we used to obtain the results that follow.

Results.

In the first row of Figure 4 , we present performance graphs for all models

on plain MNIST.

We note that the annotations under all models

indicate the average number of locations n c that we described in the training section.

We start by examining Figure 4 (a), where , that demonstrate the interpretability of our models' decisions because of the employed attention mechanism.

we depict the performance of models M 3 and BL 3 on images of different resolution.

The overlapping blue markers indicate models M 14 3 and BL 14 3 , which achieve the same accuracy level.

The green markers denote models BL 28 3 and M 28,2.2 3 3 , and as we expect, they achieve higher accuracy compared to BL 3 , and by observing the performance of all M 3 models, we see that as the resolution of the input and the number of attended locations increases, accuracy increases as well, which is the expected trade-off between computation and accuracy.

This trade-off follows a form of logarithmic curve, that saturates in models M operates.

We observe that the location module is capable of focusing on the digit, and the generally good performance of the location module is reflected on the accuracy of the model, which is over 99%.

However, BL , but are correctly classified by BL 28 3 .

In both examples, the attended locations partially cover the digits, leading 5 to be interpreted as 1, and 3 to be interpreted as 7.

Of course, we are not always able to interpret the cause of a missclassification by inspecting the attended locations, but as the provided examples show, we may be able to get important insights.

Besides the performance of our attention mechanism, we don't expect M 28,2.2 3 to achieve higher accuracy compared to BL 28 3 .

We remind that the images provided to the models are of size 28 × 28 px, the digit within each image covers an area of maximum size 14 × 14 px, and the receptive field of the baseline is 20 × 20 px.

As a result, the receptive field can perfectly cover the area of the digit, and the extracted features contain both coarse and fine information.

Consequently, our model doesn't offer any particular advantage in terms of feature extraction that could be translated in better accuracy.

This is something that we expect to change if the resolution of the input increases, or if the receptive field of the baseline gets smaller, as in the cases of BL 2 and BL 1 .

In Fig. 4 (b) we present the performance of models M 2 and BL 2 , and the main difference we observe with (a), is that BL 56 2 demonstrates lower accuracy compared to BL 28 2 .

We attribute this behavior to the fact that the receptive field of BL 56 2 covers less than 10% of the area occupied by each digit, and as a result, BL 56 2 is unable to extract coarse level features which are valuable for classification.

Based on this hypothesis, we are able to interpret the behavior of the models in (c) as well.

We observe that M In the second and third row of Figure 4 , we present our results on textured and noisy MNIST respectively.

Our previous analysis applies to these results as well.

In addition, we would like to note that our attention mechanism is robust to textured background and salt and pepper noise, as we can see in the respresentative examples provided in the last column of Fig. 3 .

In Appendix A.5.3 we provide some additional remarks on the results reported in Fig. 4 .

Data.

We use the ILSVRC 2012 version of ImageNet, which contains 1, 000 classes of natural images.

The training and validation sets contain 1, 281, 167 and 50, 000 images, respectively.

The average resolution of the original images is over 256 px per dimension.

All images are color images, 3 We don't depict other M 28 3 models that were trained with different ct values, because they don't demonstrate any significant changes in accuracy, and would reduce the clarity of our graphs.

In the y-axis we provide the top-1 accuracy on the validation set, while in the x-axis we provide the required number of FLOPs (×10 6 ) per image.

but for simplicity, when we refer to resolution, we will drop the last dimension that corresponds to the color channels.

Models.

We create two pairs of models

by following the same design principles we presented in Section 5.1.

Model BL 1 has 3 convolutional layers and receptive field 18 × 18 px, while BL 2 has 4 convolutional layers and receptive field 38 × 38 px.

have base resolution 32 × 32 px, and their location modules consider 9 candidate locations which belong to a 3 × 3 regular grid with 50% overlap.

In Appendix A.6.1 we provide the architectures of all models.

Training.

We follow the training procedure we described in Section 5.1.

The main difference is that we rescale our images to 4 different resolutions {r × r|r ∈ {32, 64, 128, 256}}, resulting to 4 training sessions.

We follow the notation we introduced in Section 5.1l, and we denote the models that result from our first training session as M For the second training session we have r = 64, for the third r = 128, and for the fourth r = 256, while i ∈ {1, 2}. Finally, we use multi-level learning rule (6) to train models that are able to demonstrate high accuracy after each processing level.

The resulting models are denoted by {M

.

In Appendix A.6.2 we provide additional training details.

Results.

In Figure 6 we provide our experimental results.

As in Figure 4 , markers of different color denote models which are trained with images of different resolution, while the annotations next to markers that correspond to models

indicate the average number of locations n c .

In the first row of Fig. 6 , we depict the performance of models M 2 and BL 2 .

First, we would like to note that we observe the trade-off between accuracy and computatinal complexity that we have identified in Fig. 4 .

When the number of required FLOPs increases by processing inputs of higher resolution, or by attending a bigger number of locations, accuracy increases as well.

By inspecting the behavior of individual models, we observe that BL lack coarse information, and we expect this phenomenon to become even more intense for models BL 1 , since they have smaller receptive field.

Indeed, as we can see in the second row of Fig. 6 , the performance gap between M 1 and BL 1 models is bigger.

M We would also like to comment on the behavior of models M ml 2 and M ml 1 that we provide in the graphs of Fig. 6 .

We observe that both M ml 2 and M ml 1 are able to maintain comparable performance to models M r 2 and M r 1 respectively, in all processing levels.

This shows that we are able to adjust the computational requirements of our models during testing time, by controlling the number of processing levels that we allow them to go through.

This is beneficial when we face constraints in the available computational resources, but also when we are processing images which vary in difficultly.

Easier images can be classified after a few processing levels, while for harder ones we can extend processing to more levels.

Finally, in Figure 7 we provide examples of attended image locations.

We proposed a novel architecture that is able to process images of arbitrary resolution without sacrificing spatial information, as it typically happens with fully convolutional processing.

This is achieved by approaching feature extraction as a top-down image pyramid traversal, that combines information from multiple different scales.

The employed attention mechanism allows us to adjust the computational requirements of our models, by changing the number of locations they attend.

This way we can exploit the existing trade-off between computational complexity and accuracy.

Furthermore, by inspecting the image regions that our models attend, we are able to get important insights about the causes of their decisions.

Finally, there are multiple future research directions that we would like to explore.

These include the improvement of the localization capabilities of our attention mechanism, and the application of our model to the problem of budgeted batch classification.

In addition, we would like our feature extraction process to become more adaptive, by allowing already extracted features to affect the processing of image regions that are attended later on.

Figure 8 we provide the parsing tree that our model implicitly creates.

Feature extraction module.

It is a CNN that receives as input images of fixed resolution h × w × c, and outputs feature vectors of fixed size 1 × f .

In the example of Fig. 2 , h = w = 32.

Location module.

It receives as input feature maps of fixed size f h × f w × f c , and returns g probabilities, where g corresponds to the number of cells in the grid of candidate locations.

In the example of Fig. 2 , g = 4 since we are using a 2 × 2 grid.

Aggregation module.

It receives g vectors of fixed size 1 × f , and outputs a vector of fixed size 1 × f .

The g input vectors describe the image regions inside a k × k grid.

Image regions that were not selected by the location module, are described by zero vectors.

The g input vectors are reorganized into a 1 × k × k × f tensor, according to the spatial arrangement of the image regions they describe.

The tensor of the reorganized input vectors is used to produce the final output.

Merging module.

It receives two vectors of fixed size 1 × f , concatenates them into a single vector of size 1 × 2f , and outputs a vector of fixed size 1 × f .

Classification module.

It receives as input a vector of fixed size 1 × f , and outputs logits of fixed size 1 × c, where c is the number of classes.

The logits are fed into a softmax layer to yield class probabilities.

The feature extraction process of our model can be described with the following recursive equation:

where ⊕ denotes the merging module, agg(·) denotes the aggregation module, loc(·) denotes the outcome of the Bernoulli sampling that is based on the output of the location module, and l Vn is the set of indexes that denote the candidate locations related to the image region described by V n .

When recursion ends, V m = V m ∀m.

The REINFORCE rule naturally emerges if we optimize the log likelihood of the labels, while considering the attended locations as latent variables (Ba et al., 2014) .

For a batch of N images, the log likelihood is given by the following relation:

where x i is the ith image in the batch, y i is its label, and w are the parameters of our model.

p(l i |x i , w) is the probability that the sequence of locations l i is attended for image x i , and p(y i |l i , x i , w) is the probability of predicting the correct label after attending l i .

Equation 8 describes the log likelihood of the labels in terms of all location sequences that could be attended.

Equation 8a shows that

is the number of times the location module is applied while sequence l i is attended, and g is the number of candidate locations considered by the location module.

In the example of We use Jensen's inequality in equation 8 to derive the following lower bound on the log likelihood:

By maximizing the lower bound F , we expect to maximize the log likelihood.

The update rule that we use is the partial derivative of F with respect to w, normalized by the number of images in the batch.

We get:

To derive equation 10 we used the log derivative trick.

As we can see, for each image x i we need to calculate an expectation according to p(l i |x i , w).

We approximate each expectation with a Monte Carlo estimator of M samples:

We get samples from p(l i |x i , w) by repeating the processing of image x i .

l i,m is the sequence of locations that is attended during the mth time we process image x i .

In order to reduce the variance of the estimators, we use the baseline technique from Xu et al. (2015) .

In particular, the baseline we use is the exponential moving average of the log likelihood, and is updated after the processing of each batch during training.

Our baseline after the nth batch is the following:

where x n i is the ith image in the nth batch, y n i is its label, and l i,n is the corresponding attended sequence of locations.

Since we use M samples for the Monte Carlo estimator of each image, we simply consider that our batch has size NM to simplify the notation.

Our updated learning rule is the following:

For simplicity, we drop the indexes that indicate the batch we are processing. (13) is the learning rule we presented in (1), and this concludes our derivation.

We provide the exact equations that we use to calculate quantities p l i t and p k in regularization terms R t (2) and R r (3) respectively.

We have:

Based on the notation we introduced in Appendix A.4.1, p l i t approximates the expected number of attended locations for image x i , by summing the probabilities from all Bernoulli distributions considered during a single processing of x i under l i .

p k computes the average probability of attending the kth out of the g candidate locations, that the location module considers every time it is applied.

The average is calculated by considering all the times the location module is applied during the processing of the N M images in our augmented batch.

Finally, we would like to note that the values of c t and c r are interdependent.

A.5.1 ARCHITECTURES In Tables 1, 2 and 3, we provide the exact architectures of models (M 3 , BL 3 ), (M 2 , BL 2 ) and (M 1 , BL 1 ).

We provide details about training one pair of models (M i , BL i ) with one of the 3 MNIST-based datasets, and the exact same procedure applies to all pairs and to all datasets.

In our first training session we optimize the cross entropy loss to train BL i , and we use learning rule (4) for M i .

We remind that because of the input resolution, M i goes through only 1 processing level, and behaves as a regular CNN which is consisted of the feature extraction module followed by the classification module.

As a result, the only term of learning rule (4) that we actually use, is the first term of L F , and we end up optimizing the cross entropy of the labels.

For both models we use the following hyperparameters.

We use learning rate 0.01 that drops by a factor of 0.2 after the completion of 80% and 90% of the total number of training steps.

We train for 70 epochs, and we use batches of size 128.

We use the Adam optimizer (Kingma & Ba, 2014) with the default values of β 1 = 0.9, The architectures of models M 3 and BL 3 that we used in our experiments with MNIST.

"GAP" denotes a global average pooling layer.

The variable output sizes of the baseline CNN are approximately calculated to preserve the clarity of our tables.

Based on these approximate output sizes, the computation of the number of FLOPs in the corresponding layers is approximate as well.

β 2 = 0.999 and = 10 −8 .

We use xavier initialization (Glorot & Bengio, 2010) for the weights, and zero initialization for the biases.

For regularization purposes, we use the following form of data augmentation.

Every time we load an image, we randomly change the position of the digit inside the black canvas, as well as the noise pattern and the background, in case we are using noisy MNIST or textured MNIST.

This data augmentation strategy and the aforementioned hyperparameter's values, are used in the other two training sessions as well.

In the second training session we again optimize the cross entropy loss for BL i , and we use learning rule (4) for M i .

The resolution of the input allows M i to go through 2 processing levels, and all terms of (4) contribute to the updates of our models' parameters.

We would like to note that we stop the gradients' flow from the location module to the other modules, and as a result, the second term of (1), as well as the regularization terms (2) and (3), affect only the parameters of the location module.

We do this to have better control on how the location module learns, since we experienced the problems we described in Section 4.

In the same context, we set λ f = 10 −6 , λ r = 10 −7 and λ t = 10 −7 , which lead to very small updates at every training step.

For our Monte Carlo estimators we use 2 samples.

These hyperparameter values are used in the third training session as well.

The models M i which are reported in Figure 4 and result from this training session (green circles), are trained with c t = 2.

In the third training session M i can go through 3 processing levels, and we can train it by using either learning rule (4), or gradual learning (5).

Gradual learning evolves in 2 stages, where in the first stage M i can go through 2 processing levels, while in the second stage it can go through 3.

The first training stage is equivalent to the previous training session, since there is an equivalence between going through a different number of processing levels and processing inputs of different resolution.

Based on that, we can directly move to the second training stage of gradual learning, by initializing the variables of M i with the learned parameters from one of the M 28 i models that we already trained.

Gradual learning creates an imbalance in terms of how M i and BL i are trained, since it evolves in multiple stages.

However, if we see gradual learning as a method of training models with images of gradually increasing resolution, it can be applied to the baselines as well.

Based on this, we can i , and then apply our standard optimization to the cross entropy loss.

In practice, we observe that baselines are almost always benefited by gradual learning, while in some cases our models achieve higher accuracy when they are trained from scratch with learning rule (4).

In general, gradual learning and most modifications to the initial learning rule (11), resulted from our experimentation with ImageNet, which is a much bigger and more diverse dataset of natural images.

Consequently, our results on the MNIST-based datasets remain almost unchanged even if we simplify our training procedure, e.g. by excluding term (3) from our training rule.

However, we kept our training procedure consistent both on the MNIST-based datasets and on ImageNet, we experimented both with gradual learning and with training from scratch at every session, and in the results of Figure  4 we report the best performing models.

Finally, the models M i which are reported in Figure 4 and result from this training session (red circles), are trained with c t = 6 and c t = 12.

In Figure 4 (e), BL 56 2 achieves higher accuracy compared to BL 28 2 , which wasn't the case in (b).

We hypothesize that the fine information provided by the increased resolution, is valuable to disentangle the digits from the distracting background, and outweighs the lack of coarse level features that stems from the receptive field size of BL 56 2 .

In addition, the differences in accuracy between M 1 and BL 1 models in Fig. 4 (f) , are considerably bigger compared to the ones recorded in (c).

This shows that our models are more robust to distracting textured background compared to the baselines.

In Fig. 4 (g .

This is surprising, because M 56,6.1 3 is processing images of higher resolution, and it should be able to reach at least the same level of accuracy as M 28,2.2 3

.

Our explanation for this observation is based on the nature of the data.

As we can see in the example images that we provide in the last row of Figure 4 , when the resolution of an image is reduced, the noise is blurred out.

As a result, when M 56,6.1 3 is processing images of higher resolution, it is processing more intense high frequency noise, and since it is using a limited number of locations, its accuracy drops compared to M .

This phenomenon is observed in Fig. 4 respectively.

The explanation we provided adds a new dimension to the understanding of our models, because so far we were treating fine information as something that is by default beneficial for classification, while high frequency noise of any form may require special consideration in our design and training choices.

Tables 4 and 5 , we provide the exact architectures of models (M 2 , BL 2 ) and (M 1 , BL 1 ).

We provide details about training one pair of models (M i , BL i ).

We make a distinction between models M 1 and M 2 only when the process we follow, or the values of the hyperparameters that we use, differ.

In our first training session we optimize the cross entropy loss to train BL i , and we use learning rule (4) for M i .

The base resolution of M i matches the size of the input images (32 × 32 px), and our model goes through only 1 processing level, without using the location module.

As a result, the only term of learning rule (4) that we actually use, is the first term of L F , and we end up optimizing the cross entropy of the labels.

For both models we use the following hyperparameters.

We use learning rate 0.001 that drops by a factor of 0.2 after the completion of 80% and 90% of the total number of training steps.

We train for 200 epochs, and we use batches of size 128.

We use the Adam optimizer with the default values of β 1 = 0.9, β 2 = 0.999 and = 10 −8 .

We use xavier initialization for the weights, and zero initialization for the biases.

For regularization purposes, we use data augmentation that is very similar to the one used by Szegedy et al. (2015) .

In particular, given a training image, we get a random The architectures of models M 2 and BL 2 that we used in our experiments on ImageNet.

"GAP" denotes a global average pooling layer.

The variable output sizes of the baseline CNN are approximately calculated to preserve the clarity of our tables.

Based on these approximate output sizes, the computation of the number of FLOPs in the corresponding layers is approximate as well.

crop that covers at least 85% of the image area, while it has an aspect ration between 0.5 and 2.0.

Since we provide inputs of fixed size to our networks, we resize the image crops appropriately.

The resizing is performed by randomly selecting between bilinear, nearest neighbor, bicubic, and area interpolation.

Also, we randomly flip the resized image crops horizontally, and we apply photometric distortions according to Howard (2013) .

The final image values are scaled between −1 and 1.

This data augmentation strategy and the aforementioned hyperparameter's values, are used in the other training sessions as well, and in the stages of multi-level learning that we describe later.

In the second training session we again optimize the cross entropy loss for BL i , and we use learning rule (4) for M i .

The resolution of the input allows M i to go through 2 processing levels, and all terms of (4) contribute to the updates of our models' parameters.

As we described in Appendix A.5.2, we stop the gradients' flow from the location module to the other modules.

We set λ f = 10 −8 , λ r = 10 −9

and λ t = 10 −9 , and for our Monte Carlo estimators we use 2 samples.

These hyperparameter values are used in the two remaining training sessions as well.

The models M 64 i which are reported in Figure 6 are trained with c t ∈ {1.5, 4.5}.

In the third training session we use gradual learning (5) for M i , by initializing its variables with the learned parameters of a M 64 i model.

The models M 128 2 which are reported in Figure 6 are trained with c t ∈ {3.75, 7, 13.5, 24.75}, and models M 128 1 with c t ∈ {3.75, 8, 11, 13.5, 30}. We use gradual learning for BL i as well, by initializing its parameters with those of BL 64 i before we apply the standard optimization to the cross entropy loss.

In the fourth training session we again use gradual learning for M i , by initializing its variables with the learned parameters of a M with c t ∈ {30, 35, 40}. As in the previous session, we optimize the cross entropy loss to train BL i , and we initialize its parameters with those of BL 128 i .

@highlight

We propose a novel architecture that traverses an image pyramid in a top-down fashion, while it visits only the most informative regions along the way.