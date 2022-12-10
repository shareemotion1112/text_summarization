Robustness of neural networks has recently been highlighted by the adversarial examples, i.e., inputs added with well-designed  perturbations which are imperceptible to humans but can cause the network to give incorrect outputs.

In this paper, we design a new CNN architecture that by itself has good robustness.

We introduce a simple but powerful technique, Random Mask, to modify existing CNN structures.

We show that CNN with Random Mask achieves state-of-the-art performance against black-box adversarial attacks without applying any adversarial training.

We next investigate the adversarial examples which “fool” a CNN with Random Mask.

Surprisingly, we find that these adversarial examples often “fool” humans as well.

This raises fundamental questions on how to define adversarial examples and robustness properly.

Deep learning (LeCun et al., 2015) , especially deep Convolutional Neural Network (CNN) (LeCun et al., 1998) , has led to state-of-the-art results spanning many machine learning fields, such as image classification BID12 BID15 Huang et al., 2017; Simonyan & Zisserman, 2014) , object detection (Redmon et al., 2016; BID7 Ren et al., 2015) , image captioning (Vinyals et al., 2015; Xu et al., 2015) and speech recognition BID1 BID13 .Despite the great success in numerous applications, recent studies have found that deep CNNs are vulnerable to some well-designed input samples named as Adversarial Examples (Szegedy et al., 2013) BID2 .

Take the task of image classification as an example, for almost every commonly used well-performed CNN, attackers are able to construct a small perturbation on an input image to cause the model to give an incorrect output label.

Meanwhile, the perturbation is almost imperceptible to humans.

Furthermore, these adversarial examples can easily transfer among different kinds of CNN architectures (Papernot et al., 2016b) .Such adversarial examples raise serious concerns on deep neural network models as robustness is crucial in many applications.

Just as BID8 suggests, both robustness and traditional supervised learning seem fully aligned.

Recently, there is a rapidly growing body of work on this topic.

One important line of research is adversarial training (Szegedy et al., 2013; Madry et al., 2017; BID9 Huang et al., 2015) .

Although adversarial training gains some success, a major difficulty is that it tends to overfit to the method of adversarial example generation used at training time BID3 .

Xie et al. (2017) and BID11 propose defense methods by introducing randomness and applying transformations to the inputs respectively.

BID5 introduces random drop during the evaluation of a neural network.

However, BID0 contends that such transformation and randomness only provide a kind of "obfuscated gradient" and can be attacked by taking expectation over transformation (EOT) to get a meaningful gradient.

Papernot et al. (2016a) and Katz et al. (2017) consider the non-linear functions in the networks and try to achieve robustness by adjusting them.

There are also detection-based defense Our main contributions are summarized as follows:• We develop a very simple but effective method, Random Mask.

We show that combining with Random Mask, existing CNNs can be significantly more robust while maintaining high generalization performance.

In fact, CNNs equipped with Random Mask achieve state-of-the-art performance against several black-box attacks, even when comparing with methods using adversarial training (See Table 1 ).•

We investigate the adversarial examples generated against CNNs with Random Mask.

We find that adversarial examples that can "fool" a CNN with Random Mask often fool humans as well.

This observation requires us to rethink what are the right definitions of adversarial examples and robustness.

We propose Random Mask, a method to modify existing CNN structures.

It randomly selects a set of neurons and removes them from the network before training.

Then the architecture of the network is fixed during the training and testing process.

To apply Random Mask on a selected layer Layer(j), suppose the input is X j and the output is conv j (X j ) ∈ R mj ×nj ×cj .

We randomly generate a binary mask mask(j) ∈ {0, 1} mj ×nj ×cj by sampling uniformly within each channel.

The drop rate of the sampling process is called the ratio (or drop ratio) of Random Mask.

Then we mask the neurons in position (x, y, c) of the output of Layer(j) if the (x, y, c) element of mask(j) is zero.

More specifically, after Random Mask, we will not compute these masked neurons and make the next layer regard these neurons as having value zero during computation.

A simple visualization of Random Mask is shown in Figure 2 .

The Random Mask in fact decreases the computational cost in each epoch since there are fewer effective connections.

Note that the number of parameters in the convolutional kernels remains unchanged, since we only mask neurons in the feature maps.

Figure 2: An illustration of Random Mask applied to three channels of a layer (neuron-wise).

Note that the number of parameters in the network is not reduced after applying Random Mask.

In the standard setting, convolutional filter will be applied uniformly to every position of the feature map of the former layer.

The success of this implementation is due to the reasonable assumption that if one feature is useful to be computed at some spatial position (x, y), then it should also be useful to be computed at a different position (x , y ).

Thus the original structure is powerful for feature extraction.

Moreover, this structure leads to parameter sharing which makes the training process more efficient.

However, the uniform application of filter also prevents the CNN from noticing the distribution of features.

In other words, the network focuses on the existence of a kind of feature but pays little attention to how this kind of feature distributes on the whole feature map (of the former layer).

Yet the pattern of feature distribution is important for humans to recognize and classify a photo, since empirically people would rely on some structured feature to perform classification.

With Random Mask, each filter may only extract features from partial positions.

More specifically, for one filter, only features which distribute consistently with the mask pattern can be extracted.

Hence filters in a network with Random Mask may capture more information on the spatial structures of local features.

Just think of a toy example: imagine Random Mask for a filter masks all the neurons but one row in the channel, if a kind of feature usually distributes in a column, it can not have strong response because the filter can only capture a small portion of the feature.

We do a straightforward experiment to verify our intuition.

We sample some images from ImageNet which can be correctly classified with high probability by both CNNs with and without Random Mask.

We then randomly shuffle the images by patches, and compare the accuracy of classifying the shuffled images (See Appendix A).

We find out that the accuracy of the CNN with Random Mask is consistently lower than that of normal CNN.

This result shows that CNNs without Random Mask cares more about whether a feature exists while CNNs with Random Mask will detect spatial structures and limit poorly-organized features from being extracted.

We further explore how Random Mask plays its role in defending against adversarial examples.

Recent observation (Liu et al., 2018) of adversarial examples found that these examples usually change a patch of the original image so that the perturbed patch looks like a small part of the incorrectly classified object.

This perturbed patch, although contains crucial features of the incorrectly classified object, usually appears at the wrong location and does not have the right spatial structure with other parts of the image.

For example (See FIG6 in Liu et al. (2018) ), the adversarial example of a panda image is misclassified as a monkey because a patch of the panda skin is perturbed adversarially so that it alone looks like the monkey's face.

However, this patch does not form a right structure of a monkey with other parts of the images.

By the properties of detecting spatial structures and limiting feature extraction, Random Mask can naturally help CNNs resist such adversarial perturbations.

In complement to the observation we mentioned above, we also find that most adversarial perturbations generated against normal CNNs look like random noises which do not change the semantic information of the original image.

In contrast, adversarial examples generated against CNNs with Random Mask tend to contain some well-organized features which sometimes change the classification results semantically (See Figure 3 and Figure 1 ).

This phenomenon also supports our intuition that Random Mask helps to detect spatial structures and extract well-organized features via imposing limitations.

Original Image Gaussian Noise Normal CNN Random Mask Figure 3 : The first image is the original image, and the other three contain different types of small perturbations.

Both the two adversarial examples on the right are predicted as frog by the corresponding models.

However, only the image generated by the randomly masked CNN is capable of fooling humans.

While the features that can be learned by each masked filter is limited, the randomness helps us get plenty of diversified patterns.

Our experiments show that these limited filters are enough for learning features.

In other words, CNNs will maintain a high test accuracy after being applied with Random Mask.

Besides, adding convolutional filters may help our CNN with Random Mask to increase test accuracy (See Section 3.3).

Furthermore, our structure is naturally compatible to ensemble methods, and randomness makes ensemble more powerful (See Section 3.3).However, it might not be appropriate to apply Random Mask to deep layers.

The distribution of features is meaningful only when the location in feature map is highly related to the location in the original input image, and the receptive field of each neuron in deep layers is too large.

In Section 3.3, there are empirical results which support our intuition.

In this section, we provide extensive experimental analyses on the performance and properties of Random Mask network structure.

We first test the robustness of Random Mask (See Section 3.1).

Then we take a closer look at the adversarial examples that can "fool" our proposed architecture (See Section 3.2).

After that we explore properties of Random Mask, including where and how to apply Random Mask, by a series of comparative experiments (See Section 3.3).

Some settings used in our experiments are listed below:Network Structure.

We apply Random Mask to several target networks, including ResNet-18 BID12 , ResNet-50, DenseNet-121 (Huang et al., 2017) , SENet-18 BID15 and VGG-19 (Simonyan & Zisserman, 2014) .

The effects of Random Mask on those network structures are quite consistent.

For brevity, we only show the defense performance on ResNet-18 in the main body and leave more experimental results in the Appendix F. The 5-block structure of ResNet-18 is shown in the Appendix C.1.

The blocks are labeled 0, 1, 2, 3, 4 and the 0 th block is the first convolution layer.

We divide these five blocks into two parts -the relative shallow ones (the 0 th , 1 st , 2 nd blocks) and the deep ones (the 3 rd , 4 th blocks).

For simplicity, we would like to regard each of these two parts as a whole in this section to avoid being trapped by details.

We use "σ-Shallow" and "σ-Deep" to denote that we apply Random Mask with drop ratio σ to the shallow blocks and to the deep blocks in ResNet-18 respectively.

Attack Framework.

The accuracy under black-box attack serves as a common criterion of robustness.

We will use it when selecting model parameters and comparing Random Mask to other similar structures.

To be more specific, by using FGSM BID9 , PGD (Kurakin et al., 2016) with ∞ norm and CW attack BID4 with 2 norm (See Appendix B for details on these attack approaches), we generate adversarial examples against different neural networks.

The performances on adversarial examples generated against different networks are quite consistent.

For brevity, we only show the defense performance against part of the adversarial examples generated by using DenseNet-121 on dataset CIFAR-10 in this section, and leave more experimental results obtained by using other adversarial examples in the Appendix F.

We use FGSM 16 , PGD 16 , PGD 32 , CW 40 to denote attack method FGSM with step size = 16, PGD with perturbation scale α = 16 and step number 20, PGD with perturbation scale α = 32 and step number 40, CW attack with confidence κ = 40 respectively.

The step size of both PGD methods are selected to be = 1.

We would like to point out that these attacks are really powerful that a normal network cannot resist these attacks.

Random Mask is not specially designed for adversarial defense, but as Random Mask introduces information that is essential for classifying correctly, it also brings robustness.

As mentioned in Section 2, normal CNN structures may allow adversary to inject features imperceptible to humans into images that can be recognized by CNN.

Yet Random Mask limits the process of feature extraction, so noisy features are less likely to be preserved.

The results of our experiments show the strengths of applying Random Mask to adversarial defense.

In fact, Random Mask can help existing CNNs reach state-of-the-art performance against the black-box attacks we use (See Table 1 ).

In Section 3.3, we will provide more experimental results to show that this asymmetric structure performs better than normal convolution and enhances robustness.

FIG0 .

We can see that networks with Random Mask always have higher accuracy than a normal network.

We evaluate the performance of CNNs with Random Mask under white-box attack (See Appendix F.3).

With neither obfuscated gradient nor gradient masking, Random Mask can still improve defense performance under various kinds of white-box attack.

Also, by checking adversarial images that are misclassified by our network, we find most of them have vague edges and can hardly be recognized by humans.

This result coincides with the theoretical analysis in Shafahi et al. (2018) ; BID6 that real adversarial examples may be inevitable in some way.

See Appendix E.2 for a randomly selected set of them.

In contrast, adversarial examples generated against normal CNNs are more like simply adding some non-sense noise which can be ignored by human.

This phenomenon also demonstrates that Random Mask really helps networks to catch more information related to real human perception.

Moreover, just as Figure 1 shows, with the help of Random Mask, we are able to find small perturbations that can actually change the semantic meaning of images for humans.

So should we still call them "adversarial examples"?

How can we get more reasonable definitions of adversarial examples and robustness?

These questions seem severe due to our findings.

We then show some properties of Random Mask including the appropriate positions to apply Random Mask, the benefit of breaking symmetry, the diversity introduced by randomness and the extensibility of Random Mask via structure adjustment and ensemble methods.

We conduct a series of comparative experiments and we will continue to use black-box defense performance as a criterion of robustness.

For brevity, we only present the results of a subset of our experiments in Table 2 .

Full information on all the experiments can be found in Appendix F.5.

Table 2 : A subset of our experiments presented in Appendix F.5 to show properties of Random Mask.

σ-Shallow DC , σ-Shallow SM , σ-Shallow ×n and σ-Shallow EN mean dropping channels with ratio σ, applying same mask with ratio σ, increasing channel number to n times with mask ratio σ for every channel and ensemble five models with different masks of same ratio σ respectively.

The entries in the middle four columns are success rates of defense under different settings.

This is also .

Masking Shallow Layers versus Masking Deep Layers.

In the last paragraph of Section 2 , we give an intuition that deep layers in a network should not be masked.

To verify this, we do extensive experiments on ResNet-18 with Random Mask applied to different parts.

We apply Random Mask with different ratios on the shallow blocks and on the deep blocks respectively.

Results in Table 2 accord closely with our intuition.

Comparing the success rate of black-box attacks on the model with the same drop ratio but different parts being masked, we find that applying Random Mask to shallow layers enjoys significantly lower adversarial attack success rates.

This verifies that shallow layers play a more important role in limiting feature extraction than the deep layers.

Moreover, only applying Random Mask on shallow blocks can achieve better performance than applying Random Mask on both shallow and deep blocks, which also verifies our intuition that dropping elements with large receptive fields is not beneficial for the network.

In addition, we would like to point out that ResNet-18 with Random Mask significantly outperforms the normal network in terms of robustness.

Random Mask versus Channel Mask.

As our Random Mask applies independent random masks to different channels in a layer, we actually break the symmetry of the original CNN structure.

To see whether this asymmetric structure would help, we try to directly drop whole channels instead of neurons using the same drop ratio as the Random Mask and train it to see the performance.

This channel mask does not hurt the symmetry while also leading to the same decrease in convolutional operations.

Table 2 shows that although our Random Mask network suffers a small drop in test accuracy due to the high drop ratio, we have a great gain in the robustness, compared with the channel-masking network.

Random Mask versus Same Mask.

The randomness in generating masks in different channels and layers allows each convolutional filter to focus on different patterns of feature distribution.

We show the essentialness of generating various masks per layer via experiments that compare Random Mask to a method that only randomly generates one mask per layer and uses it in every channel.

Table 2 shows that applying the same mask to each channel will decrease the test accuracy.

This may result from the limitation of expressivity due to the monotone masks at every masked layer.

In fact, we can illustrate such limitation using simple calculations.

Since the filters in our base network ResNet-18 is of size 3 × 3, each element of the feature maps after the first convolutional layer can extract features from at most 9 pixels in the original image.

This means that if we use the same mask and the drop ratio is 90%, only at most 9 × 10% of the input image can be caught by the convolutional layer, which would cause severe loss of input information.

Increase the Number of Channels.

In order to compensate the loss of masking many neurons in each channel, it is reasonable that we may need more convolutional filters for feature extraction.

Therefore, we try to increase the number of channels at masked layers.

Table 2 shows that despite ResNet-18 is a well-designed network structure, increasing channels does help the network with Random Mask to get higher test accuracy while maintaining good robustness performance.

Ensemble Methods.

Thanks to the diversity of Random Mask, we may directly use several networks with the same structure but different Random Masks and ensemble them.

Table 2 shows that such ensemble methods can improve a network with Random Mask in both test accuracy and robustness.

In conclusion, we introduce and experiment on Random Mask, a modification of existing CNNs that makes CNNs capture more information including the pattern of feature distribution.

We show that CNNs with Random Mask can achieve much better robustness while maintaining high test accuracy.

More specifically, by using Random Mask, we reach state-of-the-art performance in several black-box defense settings.

Another insight resulting from our experiments is that the adversarial examples generated against CNNs with Random Mask actually change the semantic information of images and can even "fool" humans.

We hope that this finding can inspire more people to rethink adversarial examples and the robustness of neural networks.

A RANDOM SHUFFLE Figure 5 : An example image that is randomly shuffled after being divided into 1 × 1, 2 × 2, 4 × 4 and 8 × 8 patches respectively.

In this part, we show results of our Random Shuffle experiment.

Intuitively, by dropping randomly selected neurons in the neural network, we may let the network learn the relative margins and features better than normal networks.

In randomly shuffled images, however, some global patterns of feature distributions are destroyed, so we expect that CNNs with Random Mask would have some trouble extracting feature information and might have worse performance than normal networks.

In order to verify our intuition, we compare the test accuracy of a CNN with Random Mask to that of a normal CNN on randomly shuffled images.

Specifically speaking, in the experiments, we first train a 0.7-Shallow network along with a normal network on ImageNet dataset.

Then we select 5000 images from the validation set which are predicted correctly with more than 99% confidence by both normal and masked networks.

We resize these images to 256 × 256 and then center crop them to 224 × 224.

After that, we random shuffle them by dividing them into k × k small patches k ∈ {2, 4, 8}, and randomly rearranging the order of patches.

Figure 5 shows one example of our test images after random shuffling.

Finally, we feed these shuffled images to the networks and see their classification accuracy.

The results are shown in Table 3 .

DISPLAYFORM0 Normal ResNet-18 99.58% 82.66% 17.56% 0.7-Shallow 97.36% 64.00% 11.94% Table 3 : The accuracy by using normal and masked networks to classify randomly shuffled test images.

From the results, we can see that our network with Random Mask always has lower accuracy than the normal network on these randomly shuffled test images, which indeed accords with our intuition.

By randomly shuffling the patches in images, we break the relative positions and margins of the objects and pose negative impact to the network with Random Mask since it may rely on such information to classify.

Note that randomly shuffled images are surely difficult for humans to classify, so this experiment might also imply that the network with Random Mask is more similar to human perception than the normal one.

We first give an overview of how to attack a neural network in some mathematical notations.

Let x be the input to the neural network and f θ be the function which represents the neural network with parameter θ.

The output label of the network to the input can be computed as c = arg max i f θ (x).

In order to perform an adversarial attack, we add a small perturbation δ x to the original image and get an adversarial image DISPLAYFORM0 The new input x adv should look visually similar to the original x. Here we use the commonly used ∞ -norm metric to measure similarity, i.e., we require that||δ x || ≤ .

The attack is considered successful if the predicted label of the perturbed image c adv = arg max i f θ (x adv ) is different from c.

Generally speaking, there are two types of attack methods: Targeted Attack, which aims to change the output label of an image to a specific (and different) one, and Untargeted Attack, which only aims to change the output label and does not restrict which specific label the modified example should let the network output.

In this paper, we mainly use the following three attack approaches.

J denotes the loss function of the neural network and y denotes the true label of x.• Fast Gradient Sign Method (FGSM).

FGSM BID9 ) is a one-step untargeted method which generates the adversarial example x adv by adding the sign of the gradients multiplied by a step size to the original benign image x. Note that FGSM controls the ∞ -norm between the adversarial example and the original one by the parameter .

DISPLAYFORM1 • Basic iterative method (PGD).

PGD is a multiple-step attack method which applies FGSM multiple times.

To make the adversarial example still stay "close" to the original image, the image is projected to the ∞ -ball centered at the original image after every step.

The radius of the ∞ -ball is called perturbation scale and is denoted by α.

DISPLAYFORM2 • CW Attack.

BID4 shows that constructing an adversarial example can be formulated as solving the following optimization problem: DISPLAYFORM3 where c · g(x ) is the loss function that evaluates the quality of x as an adversarial example and the term ||x − x|| 2 2 controls the scale of the perturbation.

More specifically, in the untargeted attack setting, the loss function g(x) can be defined as: DISPLAYFORM4 where the parameter κ is called confidence.

Here we briefly introduce the network architectures used in our experiments.

Generally, we apply Random Mask at the shallow layers of the networks and we have tried five different architectures, namely ResNet-18, ResNet-50, DenseNet-121 SENet-18 and VGG-19.

We next illustrate these architectures and show how we apply Random Mask to them.

ResNet-18 BID12 contains 5 blocks: the 0 th block is one single 3 × 3 convolutional layer, and each of the rest contains four 3 × 3 convolutional layers.

FIG1 shows the whole structure of ResNet-18.

In our experiment, applying Random Mask to a block means applying Random Mask to every layer in it.

we do for ResNet-50, we apply Random Mask to the 3 × 3 convolutional layers in the first three "shallow" blocks.

The growth rate is set to 32 in our experiments.

FIG4 .

Note that here we use the pre-activation shortcut version of SENet and we apply Random Mask to the convolutional layers in the first 3 SE-blocks. (Simonyan & Zisserman, 2014 ) is a typical neural network architecture with sixteen 3 × 3 convolutional layers and three fully-connected layers.

We slightly modified the architecture by replacing the final 3 fully connected layers with 1 fully connected layer as is suggested by recent architectures.

We apply Random Mask on the first four 3 × 3 convolutional layers.

To guarantee our experiments are reproducible, here we present more details on the training process in our experiments.

When training models on CIFAR-10, we first subtract per-pixel mean.

Then we apply a zero-padding of width 4, a random horizontal flip and a random crop of size 32 × 32 on train data.

No other data augmentation method is used.

We apply SGD with momentum parameter 0.9, weight decay parameter 5 × 10 −4 and mini-batch size 128 to train on the data for 350 epochs.

The learning rate starts from 0.1 and is divided by 10 when the number of epochs reaches 150 and 250.

When training models on MNIST, we first subtract per-pixel mean.

Then we apply random horizontal flip on train data.

We apply SGD with momentum parameter 0.9, weight decay parameter 5 × 10 −4 and mini-batch size 128 to train on the data for 50 epochs.

The learning rate starts from 0.1 and is divided by 10 when the number of epochs reaches 20 and 40.

FIG6 shows the train and test curves of a normal ResNet-18 and a Random Masked ResNet-18 on CIFAR-10 and MNIST.

Different network structures share similar tendency in terms of the train and test curves.

E ADVERSARIAL EXAMPLES GENERATED BY APPLYING RANDOM MASK E.1 ADVERSARIAL EXAMPLES THAT CAN "FOOL" HUMAN Figure 12 shows some adversarial examples generated from CIFAR-10 along with the corresponding original images.

These examples are generated from CIFAR-10 against ResNet-18 with Random Mask of drop ratio 0.8 on the 0 th , 1 st , 2 nd blocks and another ResNet-18 with Random Mask of drop ratio 0.9 on the 1 st , 2 nd blocks.

We use attack method PGD with perturbation scale α = 16 and α = 32.

We also show some adversarial examples generated from Tiny-ImageNet 1 along with the corresponding original images in Figure 13 .

Here we list the black-box settings in Madry's paper (Madry et al., 2017) .

In their experiments, ResNets are trained by minimizing the following loss: DISPLAYFORM0 The outer minimization is achieved by gradient descent and the inner maximization is achieved by generating PGD adversarial examples with step size 2, the number of steps 7 and the perturbation scale 8.

After training, in their black-box attack setting, they generate adversarial examples from naturally trained neural networks and test them on their models.

Both FGSM and PGD adversarial examples have step size or perturbation scale 8 and PGD runs for 7 gradient descent steps with step size 2.In Table 1 , we apply Random Mask to shallow blocks with drop ratio 0.85.

The ratio is selected by considering the trade-off of robustness and generalization performance, which is shown in FIG0 .

When doing attacks, we generate the adversarial examples in the same way as Madry's paper (Madry et al., 2017) does.

FIG0 : Relationship between defense rate against adversarial examples generated by PGD and test accuracy with respect to different drop ratios under Madry's setting (Madry et al., 2017) .

Each red star represents a specific drop ratio with its value written near the star.

We can see the trade-off between robustness and generalization.

In this part, we apply Random Mask to five popular network structures - , and test the black-box defense performance on CIFAR-10 and MNIST datasets.

Since both the intuition (see Section 2) and the extensive experiments (see Section 3.3 and Appendix F.5) show that we should apply Random Mask on the relatively shallow layers of the network structure, we would like to do so in this part of experiments.

Illustrations of Random Mask applied to these network structures can be found in Appendix C. In addition, the detailed experiments on ResNet-18 (See Appendix F.5) show that defense performances are consistent against adversarial examples generated under different settings.

Therefore, for brevity, we evaluate the defense performance on adversarial examples generated by PGD only in this subsection.

The results can be found in Table 4 and Table 5 .

Networks in the leftmost column are the target models which defend against adversarial examples.

Networks in the first row are the source models to generate adversarial examples by PGD.

0.5-shallow and 0.7-shallow mean applying Random Mask with drop ratio 0.5 and 0.7 to the shallow layers of the network structure whose name lies just above them.

The source and target networks are initialized differently if they share the same architecture.

All the numbers except the Acc column mean the success rate of defense.

The numbers in the Acc column mean the classification accuracy of the target model on clean test data.

These results show that Random Mask can consistently improve the black-box defense performance of different network structures.

ResNet Table 4 : Black-box experiments on CIFAR-10.

Networks in the leftmost column are the target models which defend against adversarial examples.

Networks in the first row are the source models to generate adversarial examples by PGD.

PGD runs for 20 steps with step size 1 and perturbation scale 16.

0.5-shallow and 0.7-shallow mean applying Random Mask with drop ratio 0.5 and 0.7 to the shallow layers of the network structure whose name lies just above them.

All the numbers except the Acc column mean the success rate of defense.

Table 5 : Black-box experiments on MNIST.

Networks in the leftmost column are the target models which defend against adversarial examples.

Networks in the first row are the source models to generate adversarial examples by PGD.

PGD runs for 40 steps with step size 0.01 × 255 and perturbation scale 0.3 × 255.

0.5-shallow and 0.7-shallow mean applying Random Mask with drop ratio 0.5 and 0.7 to the shallow layers of the network structure whose name lies just above them.

All the numbers except the Acc column mean the success rate of defense.

See Table 6 for the defense performance of ResNet-18 with Random Mask against white-box attacks on CIFAR-10 dataset.

All the numbers except the Acc column mean the success rate of defense.

The results on other network architectures are similar.

Table 6 : White-box defense performance.

FGSM 1 , FGSM 2 , FGSM 4 refer to FGSM with step size 1,2,4 respectively.

PGD 2 , PGD 4 , PGD 8 refer to PGD with perturbation scale 2,4,8 and step number 4,6,10 respectively.

The step size of all PGD are set to 1.

Here we show the gray-box defense ability of Random Mask and the transferability of the adversarial examples generated against Random Mask on CIFAR-10 dataset.

We generate gray-box attacks in the following two ways.

One way is to generate adversarial examples against one trained neural network and test those images on a network with the same structure but different initialization.

The other way is specific to our Random Mask models.

We generate adversarial examples on one trained network with Random Mask and test them on a network with the same drop ratio but different Random Mask.

In both of these two ways, the adversarial knows some information on the structure of the network, but does not know the parameters of it.

To see the transferability of the generated adversarial examples, we also test them on DenseNet-121 and VGG-19.

Table 7 : Results on gray-box attacks and transferability.

We use FGSM with step size 16 to generate the adversarial examples on source networks and test them on target networks.

For target networks, Normal ResNet-18, 0.5-Shallow and 0.7-Shallow represent the networks with the same structure as the corresponding source networks but with different initialization values.

0.5-Shallow DIF and 0.7-Shallow DIF represent the networks with the same drop ratios as the corresponding source networks but with different random masks.

Table 7 shows that Random Mask can also improve the performance under gray-box attacks.

In addition, we find that CNNs with Random Mask have similar performance on adversarial examples generated by our two kinds of gray-box attacks.

This phenomenon indicates that CNNs with Random Mask of same ratios have similar properties and catch similar information.

In this part, we will show more experimental results on Random Mask using different adversarial examples, different attack methods and different mask settings on ResNet-18.

More specifically, we choose 5000 test images from CIFAR-10 which are correctly classified by the original network to generate FGSM and PGD adversarial examples, and 1000 test images for CW attack.

For FGSM, we try step size ∈ {8, 16, 32}, namely FGSM 8 , FGSM 16 , FGSM 32 , to generate adversarial examples.

For PGD, we have tried more extensive settings.

Let { , T, α} be the PGD setting with step size , the number of steps T and the perturbation scale α, then we have tried PGD settings (1, 8, 4), (2, 4, 4), (4, 2, 4), (1, 12, 8) , (2, 6, 8), (4, 3, 8) , (1, 20, 16), (2, 10, 16), (4, 5, 16), (1, 40, 32) , (2, 20, 32), (4, 10, 32) to generate PGD adversarial examples.

From the experimental results, we observe the following phenomena.

First, we find that the larger the perturbation scale is, the stronger the adversarial examples are.

Second, for a fixed perturbation scale, the smaller the step size is, the more successful the attack is, as it searches the adversarial examples in a more careful way around the original image.

Based on these observation, we only show strong PGD attack results in the Appendix, namely the settings (1, 20, 16) (PGD 16 ), (2, 10, 16) (PGD 2,16 ) and (1, 40, 32) (PGD 32 ).

Nonetheless, our models also perform much better on weak PGD attacks.

For CW attack, we have also tried different confidence parameters κ.

However, we find that for large κ, the algorithm is hard to find adversarial examples for some neural networks such as VGG because of its logit scale.

For smaller κ, the adversarial examples have weak transfer ability, which means they can be easily defensed even by normal networks.

Therefore, in order to balance these two factors, we choose κ = 40 (CW 40 ) for ) for ResNet-18 as a good choice to compare our models with normal ones.

The step number for choosing the parameter c is set to 30.Note that the noise of FGSM and PGD is considered in the sense of ∞ norm and the noise of CW is considered in the sense of 2 norm.

All adversarial examples used to evaluate can fool the original network.

TAB9 ,9,10,11 and 12 list our experimental results.

DC means we replace Random Mask with a decreased number of channels in the corresponding blocks to achieve the same drop ratio.

SM means we use the same mask on all the channels in a layer.

×n means we multiply the number of the channels in the corresponding blocks by n times.

EN means we ensemble five models with different masks of the same drop ratio.

Adversarial examples generated against DenseNet-121.

The model trained on CIFAR-10 achieves 95.62% accuracy on test set.

σ-Shallow DC , σ-Shallow SM , σ-Shallow ×n and σ-Shallow EN mean dropping channels with ratio σ, applying same mask with ratio σ, increasing channel number to n times with mask ratio σ for every channel and ensemble five models with different masks of same ratio σ respectively.

The entries in the middle seven columns are success rates of defense under different settings.

The model trained on CIFAR-10 achieves 95.27% accuracy on test set.

σ-Shallow DC , σ-Shallow SM , σ-Shallow ×n and σ-Shallow EN mean dropping channels with ratio σ, applying same mask with ratio σ, increasing channel number to n times with mask ratio σ for every channel and ensemble five models with different masks of same ratio σ respectively.

The entries in the middle seven columns are success rates of defense under different settings.

The model trained on CIFAR-10 achieves 95.69% accuracy on test set.

σ-Shallow DC , σ-Shallow SM , σ-Shallow ×n and σ-Shallow EN mean dropping channels with ratio σ, applying same mask with ratio σ, increasing channel number to n times with mask ratio σ for every channel and ensemble five models with different masks of same ratio σ respectively.

The entries in the middle seven columns are success rates of defense under different settings.

The model trained on CIFAR-10 achieves 95.15% accuracy on test set.

σ-Shallow DC , σ-Shallow SM , σ-Shallow ×n and σ-Shallow EN mean dropping channels with ratio σ, applying same mask with ratio σ, increasing channel number to n times with mask ratio σ for every channel and ensemble five models with different masks of same ratio σ respectively.

The entries in the middle seven columns are success rates of defense under different settings.

Figure 15 : Randomly sampled images from Tiny-ImageNet dataset.

The network structure used to generate these images is ResNet-18 with Random Mask of ratio 0.9 on the 1 st , 2 nd blocks.

The attack method is PGD with perturbation scale 64, step size 1 and step number 80.

For each image, we show the image generated against network with Random Mask (upper), the image generated against the normal ResNet-18 (middle) and the original image (lower).

@highlight

We propose a technique that modifies CNN structures to enhance robustness while keeping high test accuracy, and raise doubt on whether current definition of adversarial examples is appropriate by generating adversarial examples able to fool humans.

@highlight

This paper proposes a simple technique for improving the robustness of neural networks against black-box attacks.

@highlight

The authors propose a simple method for increasing the robustness of convolutional neural networks against adversarial examples, with surprisingly good results.