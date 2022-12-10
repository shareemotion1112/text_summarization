Inspired by the combination of feedforward and iterative computations in the visual cortex, and taking advantage of the ability of denoising autoencoders to estimate the score of a joint distribution, we propose a novel approach to iterative inference for capturing and exploiting the complex joint distribution of output variables conditioned on some input variables.

This approach is applied to image pixel-wise segmentation, with the estimated conditional score used to perform gradient ascent towards a mode of the estimated conditional distribution.

This extends previous work on score estimation by denoising autoencoders to the case of a conditional distribution, with a novel use of a corrupted feedforward predictor replacing Gaussian corruption.

An advantage of this approach over more classical ways to perform iterative inference for structured outputs, like conditional random fields (CRFs), is that it is not any more necessary to define an explicit energy function linking the output variables.

To keep computations tractable, such energy function parametrizations are typically fairly constrained, involving only a few neighbors of each of the output variables in each clique.

We experimentally find that the proposed iterative inference from conditional score estimation by conditional denoising autoencoders performs better than comparable models based on CRFs or those not using any explicit modeling of the conditional joint distribution of outputs.

Based on response timing and propagation delays in the brain, a plausible hypothesis is that the visual cortex can perform fast feedforward BID21 inference when an answer is needed quickly and the image interpretation is easy enough (requiring as little as 200ms of cortical propagation for some object recognition tasks, i.e., just enough time for a single feedforward pass) but needs more time and iterative inference in the case of more complex inputs BID23 .

Recent deep learning research and the success of residual networks BID9 BID8 point towards a similar scenario BID16 : early computation which is feedforward, a series of non-linear transformations which map low-level features to high-level ones, while later computation is iterative (using lateral and feedback connections in the brain) in order to capture complex dependencies between different elements of the interpretation.

Indeed, whereas a purely feedforward network could model a unimodal posterior distribution (e.g., the expected target with some uncertainty around it), the joint conditional distribution of output variables given inputs could be more complex and multimodal.

Iterative inference could then be used to either sample from this joint distribution or converge towards a dominant mode of that distribution, whereas a unimodaloutput feedfoward network would converge to some statistic like the conditional expectation, which might not correspond to a coherent configuration of the output variables when the actual conditional distribution is multimodal.

This paper proposes such an approach combining a first feedforward phase with a second iterative phase corresponding to searching for a dominant mode of the conditional distribution while tackling the problem of semantic image segmentation.

We take advantage of theoretical results BID0 on denoising autoencoder (DAE), which show that they can estimate the score or negative gradient of the energy function of the joint distribution of observed variables: the difference between the reconstruction and the input points in the direction of that estimated gradient.

We propose to condition the autoencoder with an additional input so as to obtain the conditional score, Given an input image x, we extract a segmentation candidate y and intermediate feature maps h by applying a pre-trained segmentation network.

We add some noise to y and train a DAE that takes as input both y and h by minimizing Eq. 3.

Training scenario 1 (a) yields the best results and uses the corrupted prediction as input to the DAE during training.

Training scenario 2 (b) corresponds to the original DAE prescription in the conditional case, and uses a corruption of the ground truth as input to the DAE during training.i.e., the gradient of the energy of the conditional density of the output variables, given the input variables.

The autoencoder takes a candidate output y as well as an input x and outputs a valueŷ so thatŷ − y estimates the direction DISPLAYFORM0 .

We can then take a gradient step in that direction and update y towards a lower-energy value and iterate in order to approach a mode of the implicit p(y|x) captured by the autoencoder.

We find that instead of corrupting the segmentation target as input of the DAE, we obtain better results by training the DAE with the corrupted feedforward prediction, which is closer to what will be seen as the initial state of the iterative inference process.

The use of a denoising autoencoder framework to estimate the gradient of the energy is an alternative to more traditional graphical modeling approaches, e.g., with conditional random fields (CRFs) BID14 BID10 , which have been used to model the joint distribution of pixel labels given an image BID13 .

The potential advantage of the DAE approach is that it is not necessary to decide on an explicitly parametrized energy function: such energy functions tend to only capture local interactions between neighboring pixels, whereas a convolutional DAE can potentially capture dependencies of any order and across the whole image, taking advantage of the state-of-the-art in deep convolutional architectures in order to model these dependencies via the direct estimation of the energy function gradient.

Note that this is different from the use of convolutional networks for the feedforward part of the network, and regards the modeling of the conditional joint distribution of output pixel labels, given image features.

The main contributions of this paper are the following: 1.

A novel training framework for modeling structured output conditional distributions, which is an alternative to CRFs, inspired by denoising autoencoder estimation of energy gradients.

2.

Showing how this framework can be used in an architecture for image pixel-wise segmentation, in which the above energy gradient estimator is used to propose a highly probable segmentation through gradient descent in the output space.

3. Demonstrating that this approach to image segmentation outperforms or matches classical alternatives such as combining convolutional nets with CRFs and more recent state-of-theart alternatives on the CamVid dataset.

In this section, we describe the proposed iterative inference method to refine the segmentation of a feedforward network.

As pointed in section 1, DAE can estimate a density p(y) via an estimator of the score or negative gradient − ∂E ∂y of the energy function E BID25 BID24 BID0 .

These theoretical analyses of DAE are presented for the particular case where the corruption noise added to the input is Gaussian.

Results show that DAE can estimate the gradient of the energy function of a joint distribution of observed variables.

The main result is the following: DISPLAYFORM0 where σ 2 is the amount of Gaussian noise injected during training, y is the input of the autoencoder and r(y) is its output (the reconstruction).

The approximation becomes exact as σ → 0 and the autoencoder is given enough capacity, training examples and training time.

The direction of (r(y) − y) points towards more likely configurations of y. Therefore, the DAE learns a vector field pointing towards the manifold where the input data lies.

In our case, we seek to rapidly learn a vector field pointing towards more probable configurations of y|x.

We propose to extend the results summarized in subsection 2.1 and condition the autoencoder with an additional input.

If we condition the autoencoder with features h, which are a function of x, the DAE framework with Gaussian corruption learns to estimate DISPLAYFORM0 , where y is a segmentation candidate, x an input image and E is an energy function.

Gradient descent in energy can thus be performed in order to iteratively reach a mode of the estimated conditional distribution: DISPLAYFORM1 with step size .

In addition, whereas Gaussian noise around the target y true would be the DAE prescription for the corrupted input to be mapped to y true , this may be inefficient at visiting the configurations we really care about, i.e. those produced by our feedforward predictor, which we use to obtain a first guess for y, as initialization of the iterative inference towards an energy minimum.

Therefore, we propose that during training, instead of using a corrupted y true as input, the DAE takes as input a corrupted segmentation candidate y and either the input x or some features h extracted from a feedforward segmentation network applied to DISPLAYFORM2 , where f k is a non-linear function and l ∈ {1, ..., L} is the index of a layer in the feedforward segmentation network.

The output of the DAE is computed asŷ = r(ỹ, h), where r is a non-linear function which is trained to denoise conditionally andỹ is a corrupted form of y. During training,ỹ is y plus noise, while at test time (for inference) it is simply y itself.

In order to train the DAE, (1) we extract both y and h from a feedforward segmentation network; (2) we corrupt y intoỹ; and (3) we train the DAE by minimizing the following loss DISPLAYFORM3 where H is the categorical cross-entropy and y true is the segmentation ground truth.

Figure 1(a) depicts the pipeline during training.

First, a fully convolutional feedforward network for segmentation is trained.

In practice, we use one of the state-of-the-art pre-trained networks.

Second, given an input image x, we extract a segmentation proposal y and intermediate features h from the segmentation network.

Both y and h are fed to a DAE network (adding Gaussian noise to y).

The DAE is trained to properly reconstruct the clean segmentation (ground truth y true ).

FIG0 (b) presents the original DAE prescription , where the DAE is trained by taking as input y true and h.

Once trained, we can exploit the trained model to iteratively take gradient steps in the direction of the segmentation manifold.

To do so, we first obtain a segmentation proposal y from the feedforward network and then we iteratively refine this proposal by applying the following rule DISPLAYFORM4 For practical reasons, we collapsed the corruption noise σ 2 into the step size .

Given an input image x, we extract a segmentation candidate y and intermediate feature maps h by applying a pre-trained segmentation network.

We then feed x and h to the trained DAE and iteratively refine y by applying Eq. 4.

The final prediction is the last value of y computed in this way.

Figure 2 depicts the test pipeline.

We start with an input image x that we feed to a pre-trained segmentation network.

The segmentation networks outputs some intermediate feature maps h and a segmentation proposal y.

Then, both y and h are fed to the DAE to compute the outputŷ.

The DAE is used to take iterative gradient steps y = y − (y −ŷ) towards the manifold of segmentation masks, with no noise added at inference time.

On one hand, recent advances in semantic segmentation mainly focus on improving the architecture design BID19 BID1 BID4 BID12 , increasing the context understanding capabilities BID6 BID26 BID3 BID28 and building processing modules to enforce structure consistency to segmentation outputs BID13 BID3 BID31 .

Here, we are interested in this last research direction.

CRFs are among the most popular choices to impose structured information at the output of a segmentation network, being fully connected CRFs BID13 and CRFs as RNNs BID31 among best performing variants.

More recently, an alternative to promote structure consistency by decomposing the prediction process into multiple steps and iteratively adding structure information, was introduced by BID15 .

Another iterative approach was introduced by BID7 to tackle image semantic segmentation by repeatedly detecting, replacing and refining segmentation masks.

Finally, the reinterpretation of residual networks BID16 BID8 was exploited by , in the context of biomedical image segmentation, by iteratively refining learned pre-normalized images to generate segmentation predictions.

On the other hand, there has recently been some research devoted to exploit results of DAE on different tasks, such as image generation BID18 , high resolution image estimation BID20 and semantic segmentation BID27 .

BID18 propose plug & play generative networks, which, in the best reported results, train a fully-connected DAE to reconstruct a denoised version of some feature maps extracted from an image classification network.

The iterative update rule at inference time is performed in the feature space.

Sønderby et al. FORMULA1 use DAE in the context of image super-resolution to learn the gradient of the density of high resolution images and apply it to refine the output of an upsampled low resolution image.

BID27 exploit convolutional pseudo-priors trained on the ground-truth labels in semantic segmentation task.

During the training phase, the pseudo-prior is combined with the segmentation proposal from a segmentation model to produce joint distribution over data and labels.

At test time, the ground truth is not accessible, thus feedforward segmentation predictions are fed iteratively to the convolutional pseudo-prior network.

In this work, we exploit DAEs in the context of image segmentation and extend them in two ways, first by using them to learn a conditional score, and second by using a corrupted feedforward prediction as input during training to obtain better segmentations.

The main objective of these experiments is to answer the following questions: Can a conditional DAE be used successfully as the building block of iterative inference for image segmentation?

Does our proposed corruption model (based on the feedforward prediction) work better than the prescribed target output corruption?

Does the resulting segmentation system outperform more classical iterative approaches to segmentation such as CRFs?4.1 CAMVID DATASET CamVid 1 BID2 ) is a fully annotated urban scene understanding dataset.

It contains videos that are fully segmented.

We used the same split and image resolution as BID1 .

The split contains 367 images (video frames) for training, 101 for validation and 233 for test.

Each frame has a resolution of 360x480 and pixels are labeled with 11 different classes.

We experimented with two feedforward architectures for segmentation: the classical fully convolutional network FCN-8 of BID17 and the more recent state-of-the-art fully convolutional densenet (FC-DenseNet103) of BID12 , which do not make use of any additional synthetic data to boost their performances.

BID17 : FCN-8 is a feedforward segmentation network, which consists of a convolutional downsampling path followed by a convolutional upsampling path.

The downsampling path successively applies convolutional and pooling layers, and the upsampling path successively applies transposed convolutional layers.

The upsampling path recovers spatial information by merging features skipped from the various resolution levels on the downsampling path.

BID12 : FC-DenseNet is a feedforward segmentation network, that exploits the feature reuse idea of and extends it to perform semantic segmentation.

FC-DenseNet103 consists of a convolutional downsampling path, followed by a convolutional upsampling path.

The downsampling path iteratively concatenates all feature outputs in a feedforward fashion.

The upsampling path applies a transposed convolution to feature maps from the previous stage and recovers information from higher resolution features from the downsampling path of the network by using skip connections.

Our DAE is composed of a downsampling path and an upsampling path.

The downsampling path contains convolutions and pooling operations, while the upsampling path is built from unpooling with switches (also known as unpooling with index tracking) BID30 BID1 and convolution operations.

As discussed in , reverting the max pooling operations more faithfully, significantly improves the quality of the reconstructed images.

Moreover, while exploring potential network architectures, we found out that using fully convolutional-like architectures with upsampling and skip connections (between downsampling and upsampling paths) decreases segmentation results when compared to unpooling with switches.

This is not surprising, since we inject noise to the model's input when training the DAE.

Skip connections directly propagate this added noise to the end layers; making them responsible for the data denoising process.

Note that the last layers of the model might not have enough capacity to accomplish the denoising task.

In our experiments, we use DAE built from 6 interleaved pooling and convolution operations, followed by 6 interleaved unpooling and convolution operations.

We start with 64 feature maps in the first convolution and duplicate the number of feature maps in consecutive convolutions in the downsampling path.

Thus, the number of feature maps in the networks downsampling path is: 64, 128, 256, 512, 1024 and 2048.

In the upsampling path, we progressively reduce the number of feature maps up to the number of classes.

Thus, the number of feature maps in consecutive layers of the upsampling path is the following: 1024, 512, 256, 128, 64 and 11 (number of classes).

We concatenate the output of 4th pooling operation in downsampling path of DAE together with the feature maps h corresponding to 4th pooling operation in downsampling path of the segmentation network.

We train our DAE by means of stochastic gradient descent with RMSprop BID22 , initializing the learning rate to 10 −3 and applying an exponential decay of 0.99 after each epoch.

All models are trained with data augmentation, randomly applying crops of size 224 × 224 and horizontal flips.

We regularize our model with a weight decay of 10 −4 .

We use a minibatch size of 10.

While training, we add zero-mean Gaussian noise (σ = 0.1 or σ = 0.5) to the DAE input.

We train the models for a maximum of 500 epochs and monitor the validation reconstruction error to early stop the training using a patience of 100 epochs.

At test time, we need to determine the step size and the number of iterations to get the final segmentation output.

We select and the number of iterations by evaluating the pipeline on the validation set.

Therefore, we try ∈ {0.01, 0.02, 0.05, 0.08, 0.1, 0.5, 1} for up to 50 iterations (iteration ∈ {1, 2, ..., 50}).

For each iteration, we compute the mean intersection over union (mean IoU) on the validation set and keep the combination ( , number of iterations) that maximizes this metric to evaluate the test set.

TAB0 reports our results for FCN-8 and FC-DenseNet103 without any post-processing step, applying fully connected CRF BID13 , context network BID28 as trained post-processing step, CRF-RNN BID31 trained end-to-end with the segmentation network and DAE's iterative inference.

For CRF, we use publicly available implementation of BID13 .As shown in the table, using DAE's iterative inference on the segmentation candidates of a feedforward segmentation network (DAE(y)) outperforms state-of-the-art post-processing variants; improving upon FCN-8 by a margin of 3.0% IoU.

When applying CRF as a post-processor, the FCN-8 segmentation results improve 1.2%.

Note that similar improvements for CRF were reported on other architectures for the same dataset (e.g. BID1 ).

Comparable improvements are achieved when using the context module BID28 as post-processor (1.3%) and when applying CRF-RNN (1.7%).

It is worth noting that our method does not decrease the performance of any class with respect to FCN-8.

However, CRF loses 2.8% when segmenting column poles, whereas CRF-RNN loses 1.1% when segmenting signs.

When it comes to more recent state-ofthe-art architectures such as FC-DenseNet103, the post-processing increment on the segmentation metrics is lower, as expected.

Nevertheless, the improvement is still perceivable (+ 0.5% in IoU).

When comparing our method to other state-of-the-art post-processors, we observe a slight improvement.

End-to-end training of CRF-RNN with FC-DenseNet103 did not yield any improvement over FC-DenseNet103.It is worth comparing the performance of the proposed approach DAE(y) with DAE(y true ) trained from the ground truth.

As shown in the table, DAE(y) consistently outperforms DAE(y true ).

For FCN-8, the proposed method outperforms DAE(y true ) by a margin of 2.2%.

For FC-DenseNet103, differences are smaller but still noticeable.

In both cases, DAE(y) not only outperforms DAE(y true ) globally, but also in the vast majority of classes that exhibit an improvement.

Note that the model trained on the ground truth requires a bigger Gaussian noise σ in order to slightly increase the performance of the pre-trained feedforward segmentation networks.

It is worth mentioning that training our model end-to-end with the segmentation network didn't improve the results, while being more memory demanding.

FIG4 (d), the FCN-8 segmentation network fails to properly find the fence in the image, mistakenly classifying it as part of a building (highlighted with a white box on the image).

CRF is able to clean the segmentation candidate, for example, by filling in missing parts of the sidewalk but is not able to add non-existing structure (see FIG4 (e)).

Our method not only improves the segmentation candidate by smoothing large regions such as the sidewalk, but also corrects the prediction by incorporating missing objects such as the fence on FIG4

In this subsection, we analyze the influence of the two inference parameters of our method, namely the step size and the number of iterations.

This analysis is performed on the validation set of CamVid dataset, for the above-mentioned feedforward segmentation networks.

For the sake of comparison, we perform a similar analysis on densely connected CRF; by fixing the best configuration and only changing the number of CRF iterations.

plot the results in the case of FCN-8 and FC-DenseNet103, respectively.

As expected, there is a trade-off between the selected step size and the number of iterations.

The smaller the , the more iterations are required to achieve the best performance.

Interestingly, all within a reasonable range lead to similar maximum performances.

We have proposed to use a novel form of denoising autoencoders for iterative inference in structured output tasks such as image segmentation.

The autoencoder is trained to map corrupted predictions to target outputs and iterative inference interprets the difference between the output and the input as a direction of improved output configuration, given the input image.

Experiments provide positive evidence for the three questions raised at the beginning of Sec. 4: (1) a conditional DAE can be used successfully as the building block of iterative inference for image segmentation, (2) the proposed corruption model (based on the feedforward prediction) works better than the prescribed target output corruption, and (3) the resulting segmentation system outperforms state-of-the-art methods for obtaining coherent outputs.

<|TLDR|>

@highlight

Refining segmentation proposals by performing iterative inference with conditional denoising autoencoders.