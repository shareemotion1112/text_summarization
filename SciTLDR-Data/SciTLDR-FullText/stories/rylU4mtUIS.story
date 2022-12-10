Humans have the remarkable ability to correctly classify images despite possible degradation.

Many studies have suggested that this hallmark of human vision results from the interaction between feedforward signals from bottom-up pathways of the visual cortex and feedback signals provided by top-down pathways.

Motivated by such interaction, we propose a new neuro-inspired model, namely Convolutional Neural Networks with Feedback (CNN-F).

CNN-F extends CNN with a feedback generative network, combining bottom-up and top-down inference to perform approximate loopy belief propagation.

We show that CNN-F's iterative inference allows for disentanglement of latent variables across layers.

We validate the advantages of CNN-F over the baseline CNN.

Our experimental results suggest that the CNN-F is more robust to image degradation such as pixel noise, occlusion, and blur.

Furthermore, we show that the CNN-F is capable of restoring original images from the degraded ones with high reconstruction accuracy while introducing negligible artifacts.

Convolutional neural networks (CNNs) have been widely adopted for image classification and achieved impressive prediction accuracy.

While state-of-the-art CNNs can achieve near-or super-human classification performance [1] , these networks are susceptible to accuracy drops in the presence of image degradation such as blur and noise, or adversarial attacks, to which human vision is much more robust [2] .

This weakness suggests that CNNs are not able to fully capture the complexity of human vision.

Unlike the CNN, the human's visual cortex contains not only feedforward but also feedback connections which propagate the information from higher to lower order visual cortical areas as suggested by the predictive coding model [3] .

Additionally, recent studies suggest that recurrent circuits are crucial for core object recognition [4] .

A recently proposed model extends CNN with a feedback generative network [5] , moving a step forward towards more brain-like CNNs.

The inference of the model is carried out by the feedforward only CNN.

We term convolutional neural networks with feedback whose inference uses no iterations as CNN-F (0 iterations).

The generative feedback models the joint distribution of the data and latent variables.

This methodology is similar to how human brain works: building an internal model of the world [6] [7] .

Despite the success of CNN-F (0 iterations) in semi-supervised learning [5] and out-of-distribution detection [8] , the feedforward only CNN can be a noisy inference in practice and the power of the rendering top-down path is not fully utilized.

A neuro-inspired model that carries out more accurate inference is therefore desired for robust vision.

Our work is motivated by the interaction of feedforward and feedback signals in the brain, and our contributions are:

We propose the Convolutional Neural Network with Feedback (CNN-F) with more accurate inference.

We perform approximated loopy belief propagation to infer latent variables.

We introduce recurrent structure into our network by feeding the generated image from the feedback process back into the feedforward process.

We term the model with k-iteration inference as CNN-F (k iterations).

In the context without confusion, we will use the name CNN-F for short in the rest of the paper.

We demonstrate that the CNN-F is more robust to image degradation including noise, blur, and occlusion than the CNN.

In particular, our experiments show that CNN-F experiences smaller accuracy drop compared to the corresponding CNN on degraded images.

We verify that CNN-F is capable of restoring degraded images.

When trained on clean data, the CNN-F can recover the original image from the degraded images at test time with high reconstruction accuracy.

Convolutional Neural Network with Feedback (CNN-F) [5] is a generative model that generates images by coarse-to-fine rendering using the features computed by the corresponding CNN.

Latent variables in CNN-F account for the uncertainty of the rendering process.

The prior distribution of those latent variables is designed to capture the dependencies between them across layers.

Inference for the optimal latent variables given image x and label y matches a feedforward CNN in CNN-F (0 iterations) (see Fig. 1 ).

We provide mathematical description of CNN-F (0 iterations) below.

Let h(0) be the generated image, y ∈ {1, ..., K} be object category.

z( ) = {t( ), s( )}, = 1, ..., L are the latent variables at layer , where t( ) defines translation of rendering templates based on the position of local maximum from Maxpool, and s( ) decides whether to render a pixel or not based on whether it is activated (ReLU) in the feed-forward CNN.

T (t( )) denotes the translation matrix corresponding to the translation latent variable t( ).

W ( ) are rendering templates, where W is the weight matrix at layer in the corresponding CNN.

h( ) is the intermediate rendered image at layer .

The generation process in CNN-F (0 iterations) is given by:

The dependencies among latent variables {z( )} 1:L across different layers are captured by the structured prior π z|y Softmax

η exp(η) , and b( ) corresponds the bias after convolutions in CNN.

Under the assumption that the intermediate rendered images {h( )} 1:L are nonnegative, the joint maximum a posteriori (JMAP) inference of latent variable z in CNN-F (0 iterations) is a CNN [5] .

Convolutional Neural Networks with Feedback using k-iteration inference [CNN-F (k iterations)] performs approximated loopy belief propagation on CNN-F for k times (see Fig. 1 ).

Inference of latent variables is performed by propagating along both directions of the model.

In the following of this session, we will use CNN-F to denote CNN-F (k iterations) for short.

Inheriting the notation for the formulation in the CNN-F (0 iterations), we formulate CNN-F as follows.

The generation process of the top-down pathway in CNN-F is the same as in the CNN-F (0 iterations), i.e. h( − 1) = T (t( ))W ( )(s( ) h( )).

Different from the CNN-F (0 iterations), the generated image h(0) in the CNN-F is fed back to the bottomup pathway for approximated loopy belief propagation.

In other words, the CNN-F performs bottom-up followed by top-down inference such that the information at later layers in the CNNs can be used to update the noisy estimations at the early layers in the same network.

Specifically, the feedforward process in the CNN-F is g( ) = W ( ) AdaPool{AdaRelu(g( − 1))} + b( ), where g( ) denotes the network activations at layer .

The top-down messages correct for the noisy bottom-up inference by the adaptive operators

Input:

Input image x.

and object class y * .

where W ( ) is the rendering template at layer , and b( ) is the parameters of the structured prior π z|y at layer .

T (t( )) is the translation matrix corresponding to the translation latent variable t( ).

Repeat step 2 -3 until convergence or early stopping.

(see Algorithm 1):

We study the robustness and image restoration performance of CNN-F (10 iterations).

Additionally, we observe the disentanglement of information stored in latent variables.

In this section, we will refer to CNN-F (10 iterations) as CNN-F.

We train a 4 layer CNN and CNN-F (10 iterations) of corresponding architecture on the clean MNIST train set.

For the architecture, we use 3 convolutional layers followed by 1 fully connected layer.

We use 5x5 convolutional kernel for each convolutional layer with 8 channels in the first layer followed by 16 channels in the second layer followed by 8 channels in the third layer.

We use instance norm between layers to normalize the input.

We test the models on degraded test set images.

The CNN trained has test accuracy 99.1% while CNN-F has test accuracy 95.26%.

Our experimental study shows that the iterative inference in CNN-F promotes disentanglement of latent factors across layers.

In particular, we observe that the latent variables at each layer in CNN-F captures different essences of the reconstructed image.

For example, in the case of MNIST digits, those essences are different strokes that form the digits.

Those strokes differ from each other in their location, styles, or angles.

In our experiment, we trained a CNN-F with 3 convolutional layers on MNIST.

Then, we sent an MNIST image of digit 0 and an MNIST image of digit 1 into the trained networks and collected their corresponding sets of latent variables.

We denote z k to be the estimated latent variables from the image of digit 1 at layer k = 1, 2, 3 in CNN-F. Figure 2 illustrates that each set of latent variables z k captures strokes at a particular location in digit 1.

In the first column of Figure 2 , we use latent variables z 3 at the top layer in CNN-F to reconstruct the image.

Similarly, in the second column of Figure 2 , in addition to z 3 , we add z 2 into the reconstruction.

We observe that the latent variables z 3 capture the center of the digit 1 while the latent variables z 2 try to extend the digits to both ends.

Finally, we include z 1 into the reconstruction and observe that it completes the digit by filling in the two ends.

This observation suggests that CNN-F and its iterative inference algorithm lead to effective disentanglement of latent factors across the layers.

Robustness Table 1 shows the accuracy and percent accuracy drop on noisy, blurry and occluded input.

The accuracy of CNN-F drops less compared to CNN of same architecture, indicating that CNN-F is more robust.

Image Restoration Table 2 shows CNN-F's reconstruction of images with added gaussian noise, blur, and occlusion.

CNN-F is able to denoise, deblur, and do some degree of inpainting in on the degraded images.

We note that with more iterations of feedback, the reconstructed image becomes more clean.

The ability of CNN-F to restore images is consistent with studies in neuroscience which suggest that feedback signals contribute to automatic sharpening of images.

For example, Abdelhack and Kamitani [9] showed that the neural representation of blurry images is more similar to the latent representation of the clean version from a deep neural network than the latent representation of the blurry image.

CNN-F is able to sharpen blurry images, which is consistent with this study. [12] to understand better the role of feedback in robust vision.

To compare CNN-F with neuronal/psychological data, we will scale up the training to ImageNet.

A more challenging scenario for robust vision is adversarial attack.

We will study the robustness of the proposed CNN-F under various types of adversarial attacks.

We also plan to measure the similarity between the latent representations of the CNN-F with neural activity recorded from the brain in order to access whether CNN-F is a good model for human vision.

We propose the Convolutional Neural Networks with Feedback (CNN-F) which consists of both a classification pathway and a generation pathway similar to the feedforward and feedback connections in human vision.

Our model uses approximate loopy belief propagation for inferring latent variables, allowing for messages to be propagated along both directions of the model.

We also introduce recurrency by passing the reconstructed image and predicted label back into the network.

We show that CNN-F is more robust than CNN on corrupted images such as noisy, blurry, and occluded images and is able to restore degraded images when trained only on clean images.

@highlight

CNN-F extends CNN with a feedback generative network for robust vision.