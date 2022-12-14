Artistic style transfer is the problem of synthesizing an image with content similar to a given image and style similar to another.

Although recent feed-forward neural networks can generate stylized images in real-time, these models produce a single stylization given a pair of style/content images, and the user doesn't have control over the synthesized output.

Moreover, the style transfer depends on the hyper-parameters of the model with varying ``optimum" for different input images.

Therefore, if the stylized output is not appealing to the user, she/he has to try multiple models or retrain one with different hyper-parameters to get a favorite stylization.

In this paper, we address these issues by proposing a novel method which allows adjustment of crucial hyper-parameters, after the training and in real-time, through a set of manually adjustable parameters.

These parameters enable the user to modify the synthesized outputs from the same pair of style/content images, in search of a favorite stylized image.

Our quantitative and qualitative experiments indicate how adjusting these parameters is comparable to retraining the model with different hyper-parameters.

We also demonstrate how these parameters can be randomized to generate results which are diverse but still very similar in style and content.

Style transfer is a long-standing problem in computer vision with the goal of synthesizing new images by combining the content of one image with the style of another BID8 BID12 BID0 .

Recently, neural style transfer techniques BID9 BID15 BID11 BID20 BID19 showed that the correlation between the features extracted from the trained deep neural networks is quite effective on capturing the visual styles and content that can be used for generating images similar in style and content.

However, since the definition of similarity is inherently vague, the objective of style transfer is not well defined and one can imagine multiple stylized images from the same pair of content/style images.

Existing real-time style transfer methods generate only one stylization for a given content/style pair and while the stylizations of different methods usually look distinct BID27 BID13 , it is not possible to say that one stylization is better in all contexts since people react differently to images based on their background and situation.

Hence, to get favored stylizations users must try different methods that is not satisfactory.

It is more desirable to have a single model which can generate diverse results, but still similar in style and content, in real-time, by adjusting some input parameters.

One other issue with the current methods is their high sensitivity to the hyper-parameters.

More specifically, current real-time style transfer methods minimize a weighted sum of losses from different layers of a pre-trained image classification model BID15 BID13 (check Sec 3 for details) and different weight sets can result into very different styles (Figure 6) .

However, one can only observe the effect of these weights in the final stylization by fully retraining the model with the new set of weights.

Considering the fact that the "optimal" set of weights can be different for any pair of style/content ( Figure 3 ) and also the fact that this "optimal" truly doesn't exist (since the goodness of the output is a personal choice) retraining the models over and over until the desired result is generated is not practical.

Content (Fixed) Figure 1:

Adjusting the output of the synthesized stylized images in real-time.

Each column shows a different stylized image for the same content and style image.

Note how each row still resembles the same content and style while being widely different in details.

The primary goal of this paper is to address these issues by providing a novel mechanism which allows for adjustment of the stylized image, in real-time and after training.

To achieve this, we use an auxiliary network which accepts additional parameters as inputs and changes the style transfer process by adjusting the weights between multiple losses.

We show that changing these parameters at inference time results to stylizations similar to the ones achievable by retraining the model with different hyperparameters.

We also show that a random selection of these parameters at run-time can generate a random stylization.

These solutions, enable the end user to be in full control of how the stylized image is being formed as well as having the capability of generating multiple stochastic stylized images from a fixed pair of style/content.

The stochastic nature of our proposed method is most apparent when viewing the transition between random generations.

Therefore, we highly encourage the reader to check the project website https://goo.gl/PVWQ9K to view the generated stylizations.

The strength of deep networks in style transfer was first demonstrated by BID10 .

While this method generates impressive results, it is too slow for real-time applications due to its optimization loop.

Follow up works speed up this process by training feed-forward networks that can transfer style of a single style image BID15 or multiple styles .

Other works introduced real-time methods to transfer style of arbitrary style image to an arbitrary content image BID11 BID13 .

These methods can generate different stylizations from different style images; however, they only produce one stylization for a single pair of content/style image which is different from our proposed method.

Generating diverse results have been studied in multiple domains such as colorizations BID6 BID2 , image synthesis BID3 , video prediction BID1 BID17 , and domain transfer BID14 BID32 .

Domain transfer is the most similar problem to the style transfer.

Although we can generate multiple outputs from a given input image BID14 , we need a collection of target or style images for training.

Therefore we can not use it when we do not have a collection of similar styles.

Style loss function is a crucial part of style transfer which affects the output stylization significantly.

The most common style loss is Gram matrix which computes the second-order statistics of the feature activations BID10 , however many alternative losses have been introduced to measure distances between feature statistics of the style and stylized images such as correlation alignment loss BID24 , histogram loss BID25 , and MMD loss BID18 .

More recent work BID22 has used depth similarity of style and stylized images as a part of the loss.

We demonstrate the success of our method using only Gram matrix; however, our approach can be expanded to utilize other losses as well.

To the best of our knowledge, the closest work to this paper is BID31 in which the authors utilized Julesz ensemble to encourage diversity in stylizations explicitly.

Although this Figure 2 : Architecture of the proposed model.

The loss adjustment parameters ?? ?? ?? c and ?? ?? ?? s is passed to the network ?? which will predict activation normalizers ?? ?? ?? ?? and ?? ?? ?? ?? that normalize activation of main stylizing network T .

The stylized image is passed to a trained image classifier where its intermediate representation is used to calculate the style loss L s and content loss L c .

Then the loss from each layer is multiplied by the corresponding input adjustment parameter.

Models ?? and T are trained jointly by minimizing this weighted sum.

At generation time, values for ?? ?? ?? c and ?? ?? ?? s can be adjusted manually or randomly sampled to generate varied stylizations.

method generates different stylizations, they are very similar in style, and they only differ in minor details.

A qualitative comparison in FIG6 shows that our proposed method is more effective in diverse stylization.

1e-2 1e-3 1e-4 Style Figure 3 : Effect of adjusting the style weight in style transfer network from BID15 .

Each column demonstrates the result of a separate training with all w l s set to the printed value.

As can be seen, the "optimal" weight is different from one style image to another and there can be multiple "good" stylizations depending on ones' personal choice.

Check supplementary materials for more examples.

Style transfer can be formulated as generating a stylized image p which its content is similar to a given content image c and its style is close to another given style image s. DISPLAYFORM0 The similarity in style can be vaguely defined as sharing the same spatial statistics in low-level features, while similarity in content is roughly having a close Euclidean distance in high-level features BID11 .

These features are typically extracted from a pre-trained image classification network, commonly VGG-19 (Simonyan & Zisserman, 2014) .

The main idea here is that the features obtained by the image classifier contain information about the content of the input image while the correlation between these features represents its style.

In order to increase the similarity between two images, Gatys et al. BID10 minimize the following distances between their extracted features: DISPLAYFORM1 where ?? l (x) is activation of a pre-trained classification network at layer l given the input image x, while L l c (p) and L l s (p) are content and style loss at layer l respectively.

G(?? l (p)) denotes the Gram matrix associated with ?? l (p).

The total loss is calculated as a weighted sum of losses across a set of content layers C and style layers S : DISPLAYFORM2 where w l c , w l s are hyper-parameters to adjust the contribution of each layer to the loss.

Layers can be shared between C and S .

These hyper-parameters have to be manually fine tuned through try and error and usually vary for different style images (Figure 3) .

Finally, the objective of style transfer can be defined as: DISPLAYFORM3 This objective can be minimized by iterative gradient-based optimization methods starting from an initial p which usually is random noise or the content image itself.

Solving the objective in Equation 3 using an iterative method can be very slow and has to be repeated for any given pair of style/content image.

A much faster method is to directly train a deep network T which maps a given content image c to a stylized image p BID15 .

T is usually a feed-forward convolutional network (parameterized by ??) with residual connections between downsampling and up-sampling layers BID26 and is trained on many content images using Equation 3 as the loss function: DISPLAYFORM0 The style image is assumed to be fixed and therefore a different network should be trained per style image.

However, for a fixed style image, this method can generate stylized images in realtime BID15 .

Recent methods BID11 BID13 introduced real-time style transfer methods for multiple styles.

But, these methods still generate only one stylization for a pair of style and content images.

In this paper we address the following issues in real-time feed-forward style transfer methods: 1.

The output of these models is sensitive to the hyper-parameters w l c and w l s and different weights significantly affect the generated stylized image as demonstrated in FIG3 .

Moreover, the "optimal" weights vary from one style image to another (Figure 3 ) and therefore finding a good set of weights should be repeated for each style image.

Please note that for each set of w l c and w l s the model has to be fully retrained that limits the practicality of style transfer models.

Top row demonstrates that randomizing ?? ?? ?? results to different stylizations however the style features appear in the same spatial position (e.g., look at the swirl effect on the left eye).

Middle row visualizes the effect of adding random noise to the content image in moving these features with fixed ?? ?? ??.

Combination of these two randomization techniques can generate highly versatile outputs which can be seen in the bottom row.

Notice how each image in this row differs in both style and the spatial position of style elements.

Look at FIG8 for more randomized results.2.

Current methods generate a single stylized image given a content/style pair.

While the stylizations of different methods usually look very distinct BID27 , it is not possible to say which stylization is better for every context since it is a matter of personal taste.

To get a favored stylization, users may need to try different methods or train a network with different hyper-parameters which is not satisfactory and, ideally, the user should have the capability of getting different stylizations in real-time.

We address these issues by conditioning the generated stylized image on additional input parameters where each parameter controls the share of the loss from a corresponding layer.

This solves the problem (1) since one can adjust the contribution of each layer to adjust the final stylized result after the training and in real-time.

Secondly, we address the problem (2) by randomizing these parameters which result in different stylizations.

We enable the users to adjust w DISPLAYFORM0 To learn the effect of ?? ?? ?? c and ?? ?? ?? s on the objective, we use a technique called conditional instance normalization (Ulyanov et al.) .

This method transforms the activations of a layer x in the feedforward network T to a normalized activation z which is conditioned on additional inputs ?? ?? ?? = [?? ?? ?? c , ?? ?? ?? s ]: DISPLAYFORM1 where ?? and ?? are mean and standard deviation of activations at layer x across spatial axes BID11 and ?? ?? ?? ?? , ?? ?? ?? ?? are the learned mean and standard deviation of this transformation.

These parameters can be approximated using a second neural network which will be trained end-to-end with T : Since L l can be very different in scale, one loss term may dominate the others which will fail the training.

To balance the losses, we normalize them using their exponential moving average as a normalizing factor, i.e. each L l will be normalized to: DISPLAYFORM2 DISPLAYFORM3 where L l (p) is the exponential moving average of L l (p).

In this section, first we study the effect of adjusting the input parameters in our method.

Then we demonstrate that we can use our method to generate random stylizations and finally, we compare our method with a few baselines in terms of generating random stylizations.

We implemented ?? as a multilayer fully connected neural network.

We used the same architecture as BID15 BID11 for T and only increased number of residual blocks by 3 (look at supplementary materials for details) which improved stylization results.

We trained T and ?? jointly by sampling random values for ?? ?? ?? from U (0, 1).

We trained our model on ImageNet BID5 ) as content images while using paintings from Kaggle Painter by Numbers (Kaggle) and textures from Descibable Texture Dataset BID4 as style images.

We selected random images form ImageNet test set, MS-COCO BID21 and faces from CelebA dataset as our content test images.

Similar to BID11 , we used the last feature set of conv3 as content layer C .

We used last feature set of conv2, conv3 and conv4 layers from VGG-19 network as style layers S .

Since there is only one content layer, we fix ?? ?? ?? c = 1.

Our implementation can process 47.5 fps on a NVIDIA GeForce 1080, compared to 52.0 for the base model without ?? sub-network.

The primary goal of introducing the adjustable parameters ?? ?? ?? was to modify the loss of each separate layer manually.

Qualitatively, this is demonstrable by increasing one of the input parameters from zero to one while fixing the rest of them to zero.

FIG0 shows one example of such transition.

Each row in this figure is corresponding to a different style layer, and therefore the stylizations at each row would be different.

Notice how deeper layers stylize the image with bigger stylization elements from the style image but all of them still apply the coloring.

We also visualize the effect of increasing two of the input parameters at the same time in FIG7 .

However, these transitions are best demonstrated interactively which is accessible at the project website https://goo.gl/PVWQ9K.

To quantitatively demonstrate the change in losses with adjustment of the input parameters, we rerun the same experiment of assigning a fixed value to all of the input parameters while gradually increasing one of them from zero to one, this time across 100 different content images.

Then we calculate the median loss at each style loss layer S .

As can be seen in , increasing ?? l s decreases the measured loss corresponding to that parameter.

To show the generalization of our method across style images, we trained 25 models with different style images and then measured median of the loss at any of the S layers for 100 different content images FIG4 -(bottom).

We exhibit the same drop trends as before which means the model can generate stylizations conditioned on the input parameters.

Finally, we verify that modifying the input parameters ?? ?? ?? s generates visually similar stylizations to the retrained base model with different loss weights w l s .

To do so, we train the base model BID15

One application of our proposed method is to generate multiple stylizations given a fixed pair of content/style image.

To do so, we randomize ?? ?? ?? to generate randomized stylization (top row of FIG1 ).

Changing values of ?? ?? ?? usually do not randomize the position of the "elements" of the style.

We can enforce this kind of randomness by adding some noise with the small magnitude to the content image.

For this purpose, we multiply the content image with a mask which is computed by applying an inverse Gaussian filter on a white image with a handful (< 10) random zeros.

This masking can shadow sensitive parts of the image which will change the spatial locations of the "elements" of style.

Middle row in FIG1 demonstrates the effect of this randomization.

Finally, we combine these two randomizations to maximizes the diversity of the output which is shown in the bottom row of FIG1 .

More randomized stylizations can be seen in FIG8 and at https://goo.gl/PVWQ9K.

BID31 .

Our method generates diverse stylizations while StyleNet results mostly differ in minor details.

To the best of our knowledge, generating diverse stylizations at real-time is only have been studied at BID31 before.

In this section, we qualitatively compare our method with this baseline.

Also, we compare our method with a simple baseline where we add noise to the style parameters.

The simplest baseline for getting diverse stylizations is to add noises to some parameters or the inputs of the style-transfer network.

In the last section, we demonstrate that we can move the locations of elements of style by adding noise to the content input image.

To answer the question that if we can get different stylizations by adding noise to the style input of the network, we utilize the model of which uses conditional instance normalization for transferring style.

We train this model with only one style image and to get different stylizations, we add random noise to the style parameters (?? ?? ?? ?? and ?? ?? ?? ?? parameters of equation 6) at run-time.

The stylization results for this baseline are shown on the top row of FIG6 .

While we get different stylizations by adding random noises, the stylizations are no longer similar to the input style image.

To enforce similar stylizations, we trained the same baseline while we add random noises at the training phase as well.

The stylization results are shown in the second row of FIG6 .

As it can be seen, adding noise at the training time makes the model robust to the noise and the stylization results are similar.

This indicates that a loss term that encourages diversity is necessary.

We also compare the results of our model with StyleNet BID31 .

As visible in FIG6 , although StyleNet's stylizations are different, they vary in minor details and all carry the same level of stylization elements.

In contrast, our model synthesizes stylized images with varying levels of stylization and more randomization.

Our main contribution in this paper is a novel method which allows adjustment of each loss layer's contribution in feed-forward style transfer networks, in real-time and after training.

This capability allows the users to adjust the stylized output to find the favorite stylization by changing input parameters and without retraining the stylization model.

We also show how randomizing these parameters plus some noise added to the content image can result in very different stylizations from the same pair of style/content image.

Our method can be expanded in numerous ways e.g. applying it to multi-style transfer methods such as BID11 , applying the same parametrization technique to randomize the correlation loss between the features of each layer and finally using different loss functions and pre-trained networks for computing the loss to randomize the outputs even further.

One other interesting future direction is to apply the same "loss adjustment after training" technique for other classic computer vision and deep learning tasks.

Style transfer is not the only task in which modifying the hyper-parameters can greatly affect the predicted results and it would be rather interesting to try this method for adjusting the hyper-parameters in similar problems.

Convolution 3 1 C SAME ReLU Convolution 3 1 C SAME Linear Add the input and the output Upsampling -C feature maps Nearest-neighbor interpolation, factor 2 Convolution 3 1 C SAME ReLUNormalization Conditional instance normalization after every convolution Optimizer Adam (?? = 0.001, ?? 1 = 0.9, ?? 2 = 0.999)

Batch size 8 Weight initialization Isotropic gaussian (?? = 0, ?? = 0.01) BID15 .

Each column demonstrates the result of a separate training.

As can be seen, the "optimal" weight is different from one style image to another and there can be more than one "good" stylization depending on ones personal choice.

Figure 13: Results of combining losses from different layers at generation time by adjusting their corresponding parameters.

The first column is the style image which is fixed for each row.

The content image is the same for all of the outputs.

The corresponding parameter for each one of the losses is zero except for the one(s) mentioned in the title of each column.

Notice how each layer enforces a different type of stylization and how the combinations vary as well.

Also note how a single combination of layers cannot be the "optimal" stylization for any style image and one may prefer the results from another column.

@highlight

Stochastic style transfer with adjustable features. 