Deep networks often perform well on the data distribution on which they are trained, yet give incorrect (and often very confident) answers when evaluated on points from off of the training distribution.

This is exemplified by the adversarial examples phenomenon but can also be seen in terms of model generalization and domain shift.

Ideally, a model would assign lower confidence to points unlike those from the training distribution.

We propose a regularizer which addresses this issue by training with interpolated hidden states and encouraging the classifier to be less confident at these points.

Because the hidden states are learned, this has an important effect of encouraging the hidden states for a class to be concentrated in such a way so that interpolations within the same class or between two different classes do not intersect with the real data points from other classes.

This has a major advantage in that it avoids the underfitting which can result from interpolating in the input space.

We prove that the exact condition for this problem of underfitting to be avoided by Manifold Mixup is that the dimensionality of the hidden states exceeds the number of classes, which is often the case in practice.

Additionally, this concentration can be seen as making the features in earlier layers more discriminative.

We show that despite requiring no significant additional computation, Manifold Mixup achieves large improvements over strong baselines in supervised learning, robustness to single-step adversarial attacks, semi-supervised learning, and Negative Log-Likelihood on held out samples.

Machine learning systems have been enormously successful in domains such as vision, speech, and language and are now widely used both in research and industry.

Modern machine learning systems typically only perform well when evaluated on the same distribution that they were trained on.

However machine learning systems are increasingly being deployed in settings where the environment is noisy, subject to domain shifts, or even adversarial attacks.

In many cases, deep neural networks which perform extremely well when evaluated on points on the data manifold give incorrect answers when evaluated on points off the training distribution, and with strikingly high confidence.

This manifests itself in several failure cases for deep learning.

One is the problem of adversarial examples (Szegedy et al., 2014) , in which deep neural networks with nearly perfect test accuracy can produce incorrect classifications with very high confidence when evaluated on data points with small (imperceptible to human vision) adversarial perturbations.

These adversarial examples could present serious security risks for machine learning systems.

Another failure case involves the training and testing distributions differing significantly.

With deep neural networks, this can often result in dramatically reduced performance.

To address these problems, our Manifold Mixup approach builds on following assumptions and motivations: (1) we adopt the manifold hypothesis, that is, data is concentrated near a lower-dimensional non-linear manifold (this is the only required assumption on the data generating distribution for Manifold Mixup to work); (2) a neural net can learn to transform the data non-linearly so that the transformed data distribution now lies on a nearly flat manifold; (3) as a consequence, linear interpolations between examples in the hidden space also correspond to valid data points, thus providing novel training examples. (a,b,c) shows the decision boundary on the 2d spirals dataset trained with a baseline model (a fully connected neural network with nine layers where middle layer is a 2D bottleneck layer), Input Mixup with ?? = 1.0, and Manifold Mixup applied only to the 2D bottleneck layer.

As seen in (b), Input Mixup can suffer from underfitting since the interpolations between two samples may intersect with a real sample.

Whereas Manifold Mixup (c), fits the training data perfectly (more intuitive example of how Manifold Mixup avoids underfitting is given in Appendix H).

The bottom row (d,e,f) shows the hidden states for the baseline, Input Mixup, and manifold mixup respectively.

Manifold Mixup concentrates the labeled points from each class to a very tight region, as predicted by our theory (Section 3) and assigns lower confidence classifications to broad regions in the hidden space.

The black points in the bottom row are the hidden states of the points sampled uniformly in x-space and it can be seen that manifold mixup does a better job of giving low confidence to these points.

Additional results in Figure 6 of Appendix B show that the way Manifold Mixup changes the representations is not accomplished by other well-studied regularizers (weight decay, dropout, batch normalization, and adding noise to the hidden states).Manifold Mixup performs training on the convex combinations of the hidden state representations of data samples.

Previous work, including the study of analogies through word embeddings (e.g. king -man + woman ??? queen), has shown that such linear interpolation between hidden states is an effective way of combining factors (Mikolov et al., 2013) .

Combining such factors in the higher level representations has the advantage that it is typically lower dimensional, so a simple procedure like linear interpolation between pairs of data points explores more of the space and with more of the points having meaningful semantics.

When we combine the hidden representations of training examples, we also perform the same linear interpolation in the labels (seen as one-hot vectors or categorical distributions), producing new soft targets for the mixed examples.

In practice, deep networks often learn representations such that there are few strong constraints on how the states can be distributed in the hidden space, because of which the states can be widely distributed through the space, (as seen in FIG0 ).

As well as, nearly all points in hidden space correspond to high confidence classifications even if they correspond to off-the-training distribution samples (seen as black points in FIG0 ).

In contrast, the consequence of our Manifold Mixup approach is that the hidden states from real examples of a particular class are concentrated in local regions and the majority of the hidden space corresponds to lower confidence classifications.

This concentration of the hidden states of the examples of a particular class into a local regions enables learning more discriminative features.

A low-dimensional example of this can be seen in FIG0 and a more detailed analytical discussion for what "concentrating into local regions" means is in Section 3.Our method provides the following contributions:??? The introduction of a novel regularizer which outperforms competitive alternatives such as Cutout BID4 , Mixup (Zhang et al., 2018) , AdaMix BID10 Dropout (Hinton et al., 2012) .

On CIFAR-10, this includes a 50% reduction in test Negative Log-Likelihood (NLL) from 0.1945 to 0.0957.??? Manifold Mixup achieves significant robustness to single step adversarial attacks.??? A new method for semi-supervised learning which uses a Manifold Mixup based consistency loss.

This method reduces error relative to Virtual Adversarial Training (VAT) (Miyato et al., 2018a) by 21.86% on CIFAR-10, and unlike VAT does not involve any additional significant computation.??? An analysis of Manifold Mixup and exact sufficient conditions for Manifold Mixup to achieve consistent interpolations.

Unlike Input Mixup, this doesn't require strong assumptions about the data distribution (see the failure case of Input Mixup in FIG0 ): only that the number of hidden units exceeds the number of classes, which is easily satisfied in many applications.

The Manifold Mixup algorithm consists of selecting a random layer (from a set of eligible layers including the input layer)

k. We then process the batch without any mixup until reaching that layer, and we perform mixup at that hidden layer, and then continue processing the network starting from the mixed hidden state, changing the target vector according to the mixup interpolation.

More formally, we can redefine our neural network function y = f (x) in terms of k: DISPLAYFORM0 Here g k is a function which runs a neural network from the input hidden state k to the output y, and h k is a function which computes the k-th hidden layer activation from the input x.

For the linear interpolation between factors, we define a variable ?? and we sample from p(??).

Following (Zhang et al., 2018) , we always use a beta distribution p(??) = Beta(??, ??).

With ?? = 1.0, this is equivalent to sampling from U (0, 1).We consider interpolation in the set of layers S k and minimize the expected Manifold Mixup loss.

DISPLAYFORM1 We backpropagate gradients through the entire computational graph, including to layers before the mixup process is applied (Section 5.1 and appendix Section B explore this issue directly).

In the case where k = 0 is the input layer and S k = 0, Manifold Mixup reduces to the mixup algorithm of Zhang et al. (2018) .

With ?? = 2.0, about 5% of the time ?? is within 5% of 0 or 1, which essentially means that an ordinary example is presented.

In the more general case, we can optimize the expectation in the Manifold Mixup objective by sampling a different layer to perform mixup in on each update.

We could also select a new random layer as well as a new lambda for each example in the minibatch.

In theory this should reduce the variance in the updates introduced by these random variables.

However in practice we found that this didn't have a significant effect on the results, so we decided to sample a single lambda and a randomly chosen layer per minibatch.

In comparison to Input Mixup, the results in the FIG3 demonstrate that Manifold Mixup reduces the loss calculated along hidden interpolations significantly better than Input Mixup, without significantly changing the loss calculated along visible space interpolations.

Our goal is to show that if one does mixup in a sufficiently deep hidden layer in a deep network, then a mixup loss of zero can be achieved so long the dimensionality of that hidden layer dim (H) is greater than the number of classes d. More specifically the resulting representations for that class must fall onto a subspace of dimension dim (H) ??? d.

Assume X and H to denote the input and representation spaces, respectively.

We denote the labelset by Y and let Z X ?? Y. Also, let us denote the set of all probability measures on Z by M (Z).

Assume G ??? H X to be the set of all possible functions that can be generated by the neural network mapping input to the representation space.

In this regard, each g ??? G represents a mapping from input to the representation units.

A similar definition can be made for F ??? Y H , as the space of all possible functions from the representation space to the output.

We are interested in the solution of the following problem, at least in some specific asymptotic regimes: DISPLAYFORM0 where We analyze the above-mentioned minimization when the probability measure P = P D is chosen as the empirical distribution over a finite dataset of size n, denoted by DISPLAYFORM1 DISPLAYFORM2 .

Let f * ??? F and g * ??? G be the minimizers in (2) with P = P D .In particular, we are interested in the case where G = H X , F = Y H , and H is a vector space; These conditions simply state that the two respective neural networks which map input into representation space, and representation space to the output are being extended asymptotically 1 .

In this regard, we show that the minimizer f * is a linear function from H to Y. This way, it is easy to show that the following equality holds: DISPLAYFORM3 where Proof.

With basic linear algebra, one can confirm that the following argument is true as long as DISPLAYFORM4 DISPLAYFORM5 where I d??d and 1 d are the d-dimensional identity matrix and all-one vector, respectively.

In fact, b1 T d is a rank-one matrix, while the rank of identity matrix is d. Therefore, A T H only needs to be rank d ??? 1.

DISPLAYFORM6 where h i here means the ith column of matrix H, and ?? i ??? {1, . . .

, d} is the class-index of the ith sample.

We show that such selections will make the objective in (2) equal to zero (which is the minimum possible value).

More precisely, the following relations hold: DISPLAYFORM7 1 Due to the consistency theorem that proves neural networks with nonlinear activation functions are dense in the function spaceThe final equality is a direct result of A T h ??i + b = y ??i for i = 1, . . .

, n.

Also, it can be shown that as long as dim (H) > d ??? 1, then data points in the representation space H have some degrees of freedom to move independently.

Corollary 1.

Consider the setting in Theorem 1, and assume dim (H) > d ??? 1.

Let g * ??? G to be the true minimizer of (2) for a given dataset D. Then, data-points in the representation space, i.e. DISPLAYFORM8 Proof.

In the proof of Theorem 1, we have DISPLAYFORM9 The r.h.s.

of FORMULA11 can become a rank-(d ??? 1) matrix as long as vector b is chosen properly.

Thus, A is free to have a null-space of dimension dim (H)???d+1.

This way, one can assign g * (X i ) = h ??i +e i , where h j and ?? i (for j = 1, . . .

, d and i = 1, . . .

, n) are defined in the same way as in Theorem 1, and e i s can are arbitrary vectors in the null-space of A, i.e. e i ??? ker (A) for all i.

This result implies that if the Manifold Mixup loss is minimized, then the representation for each class will lie on a subspace of dimension dim (H)???d+1.

In the most extreme case where dim (H) = d ??? 1, each hidden state from the same class will be driven to a single point, so the change in the hidden states following any direction on the class-conditional manifold will be zero.

In the more general case with a larger dim (H), a majority of directions in H-space will not change as we move along the class-conditional manifold.

Why are these properties desirable?

First, it can be seen as a flattening 2 . of the class-conditional manifold which encourages learning effective representations earlier in the network.

Second, it means that the region in hidden space occupied by data points from the true manifold has nearly zero measure.

So a randomly sampled hidden state within the convex hull spanned by the data is more likely to have a classification score that is not fully confident (non-zero entropy).

Thus it encourages the network to learn discriminative features in all layers of the network and to also assign low-confidence classification decisions to broad regions in the hidden space (this can be seen in FIG0 and Figure 6 ).

Regularization is a major area of research in machine learning.

Manifold Mixup closely builds on two threads of research.

The first is the idea of linearly interpolating between different randomly drawn examples and similarly interpolating the labels (Zhang et al., 2018; Tokozume et al., 2018) .

These methods encourage the output of the entire network to change linearly between two randomly drawn training samples, which can result in underfitting.

In contrast, for a particular layer at which mixing is done, Manifold Mixup allows lower layers to learn more concentrated features in such a way that it makes it easier for the output of the upper layers to change linearly between hidden states of two random samples, achieving better results (section 5.1 and Appendix B).Another line of research closely related to Manifold Mixup involves regularizing deep networks by perturbing the hidden states of the network.

These methods include dropout (Hinton et al., 2012) , batch normalization (Ioffe & Szegedy, 2015) , and the information bottleneck BID0 .

Notably Hinton et al. (2012) and Ioffe & Szegedy (2015) both demonstrated that regularizers already demonstrated to work well in the input space (salt and pepper noise and input normalization respectively) could also be adapted to improve results when applied to the hidden layers of a deep network.

We believe that the regularization effect of Manifold Mixup would be complementary to that of these algorithms.

Zhao & Cho (2018) explored improving adversarial robustness by classifying points using a function of the nearest neighbors in a fixed feature space.

This involved applying mixup between each set of nearest neighbor examples in that feature space.

The similarity between Zhao & Cho (2018) and Table 1 : Supervised Classification Results on CIFAR-10 (a) and CIFAR-100 (b).

We note significant improvement with Manifold Mixup especially in terms of Negative log-likelihood (NLL).

Please refer to Appendix C for details on the implementation of Manifold Mixup and Manifold Mixup All layers and results on SVHN.

??? and ??? refer to the results reported in (Zhang et al., 2018) and BID10 DISPLAYFORM0 Manifold Mixup is that both consider linear interpolations in hidden states with the same interpolation applied to the labels.

However an important difference is that Manifold Mixup backpropagates gradients through the earlier parts of the network (the layers before where mixup is applied) unlike Zhao & Cho (2018).

As discussed in Section 5.1 and Appendix B this was found to significantly change the learning process.

AdaMix BID8 is another related method which attempted to learn better mixing distributions to avoid overlap.

AdaMix reported 3.52% error on CIFAR-10 and 20.97% error on CIFAR-100.

We report 2.38% error on CIFAR-10 and 20.39% error on CIFAR-100.

AdaMix only interpolated in the input space, and they report that their method hurt results significantly when they tried to apply it to the hidden layers.

Thus this method likely works for different reasons from Manifold Mixup and might be complementary.

AgrLearn BID9 ) is a method which adds a new information bottleneck layer to the end of deep neural networks.

This achieved substantial improvements, and was used together with Input Mixup (Zhang et al., 2018) to achieve 2.45% test error on CIFAR-10.

As their method was complimentary with Input Mixup, it's possible that their method is also complimentary with Manifold Mixup, and this could be an interesting area for future work.

We present results on Manifold Mixup based regularization of networks using the PreActResNet architecture BID11 .

We closely followed the procedure of (Zhang et al., 2018) as a way of providing direct comparisons with the Input Mixup algorithm.

We used weight decay of 0.0001 and trained with SGD with momentum and multiplied the learning rate by 0.1 at regularly scheduled epochs.

These results for CIFAR-10 and CIFAR-100 are in Table 1a and 1b.

We also ran experiments where we took PreActResNet34 models trained on the normal CIFAR-100 data and evaluated them on test sets with artificial deformations (shearing, rotation, and zooming) and showed that Manifold Mixup demonstrated significant improvements (Appendix C Table 5), which suggests that Manifold Mixup performs better on the variations in the input space not seen during the training.

We also show that the number of epochs needed to reach good results is not significantly affected by using Manifold Mixup in FIG8 .To better understand why the method works, we performed an experiment where we trained with Manifold Mixup but blocked gradients immediately after the layer where we perform mixup.

On CIFAR-10 PreActResNet18, this caused us to achieve 4.86% test error when trained on 400 epochs and 4.33% test error when trained on 1200 epochs.

This is better than the baseline, but worse than Manifold Mixup or Input Mixup in both cases.

Because we randomly select the layer to mix, each layer of the network is still being trained, although not on every update.

This demonstrates that the Manifold Mixup method improves results by changing the layers both before and after the mixup operation is applied.

We also compared Manifold Mixup against other strong regularizers.

We selected the best performing hyperparameters for each of the following models using a validation set.

Using each model's best performing hyperparameters, test error averages and standard deviations for five trials (in %) for CIFAR-10 using PreResNet50 trained for 600 epochs are: vanilla PreResNet50 (4.96 ?? 0.19), Dropout (5.09 ?? 0.09), Cutout BID4 ) (4.77 ?? 0.38), Mixup (4.25 ?? 0.11) and Manifold Mixup (3.77 ?? 0.18).

This clearly shows that Manifold Mixup has strong regularizing effects. (Note that the results in Table 1 were run for 1200 epochs and thus these results are not directly comparable.)We also evaluate the quality of the representations learned by Manifold Mixup by applying K-Nearest Neighbour classifier on the feature extracted from the top layer of PreResNet18 for CIFAR-10.

We achieved test errors of 6.09% (Vanilla PreResNet18), 5.54% (Mixup) and 5.16% (Manifold Mixup).

It suggests that Manifold Mixup helps learning better representations.

Further analysis of how Manifold Mixup changes the representations is given in Appendix BThere are a couple of important questions to ask: how sensitive is the performance of Manifold Mixup with respect to the hyperparameter ?? and in which layers the mixing should be performed.

We found that Manifold Mixup works well for a wide range of ?? values.

Please refer to Appendix J for more details.

Furthermore, the results in Appendix K suggests that mixing should not be performed in the layers very close to the output layer.

Semi-supervised learning is concerned with building models which can take advantage of both labeled and unlabeled data.

It is particularly useful in domains where obtaining labels is challenging, but unlabeled data is plentiful.

The Manifold Mixup approach to semisupervised learning is closely related to the consistency regularization approach reviewed by Oliver et al. (2018) .

It involves minimizing loss on labelled samples as well as unlabeled samples by controlling the tradeoff between these two losses via a consistency coefficient.

In the Manifold Mixup approach for semi-supervised learning, the loss from labeled examples is computed as normal.

For computing loss from unlabelled samples, the model's predictions are evaluated on a random batch of unlabeled data points.

Then the normal manifold mixup procedure is used, but the targets to be mixed are the soft target outputs from the classifier.

The detailed algorithm for both Manifold Mixup and Input Mixup with semi-supervised learning are given in appendix D.Oliver et al. FORMULA2 performed a systematic study of semi-supervised algorithms using a fixed wide resnet architecture "WRN-28-2" (Zagoruyko & Komodakis, 2016) .

We evaluate Manifold Mixup using this same setup and achieve improvements for CIFAR-10 over the previously best performing algorithm, Virtual Adversarial Training (VAT) (Miyato et al., 2018a) and Mean-Teachers (Tarvainen & Valpola, 2017) .

For SVHN, Manifold Mixup is competitive with VAT and Mean-Teachers.

See TAB1 .

While VAT requires an additional calculation of the gradient and Mean-Teachers requires repeated model parameters averaging, Manifold Mixup requires no additional (non-trivial) computation.

In addition, we also explore the regularization ability of Manifold Mixup in a fully-supervised lowdata regime by training a PreResnet-152 model on 4000 labeled images from CIFAR-10.

We obtained 13.64 % test error which is comparable with the fully-supervised regularized baseline according to results reported in Oliver et al. (2018) .

Interestingly, we do not use a combination of two powerful regularizers ("Shake-Shake" and "Cut-out") and the more complex ResNext architecture as in Oliver et al. FORMULA2 and still achieve the same level of test accuracy, while doing much better than the fully supervised baseline not regularized with state-of-the-art regularizers (20.26% error).

Adversarial examples in some sense are the "worst case" scenario for models failing to perform well when evaluated with data off the manifold 3 .

Because Manifold Mixup only considers a subset of directions around data points (namely, those corresponding to interpolations), we would not expect the model to be robust to adversarial attacks which can consider any direction within an epsilon-ball of each example.

At the same time, Manifold Mixup expands the set of points seen during training, so an intriguing hypothesis is that these overlap somewhat with the set of possible adversarial examples, which would force adversarial attacks to consider a wider set of directions, and potentially be more computationally expensive.

To explore this we considered the Fast Gradient Sign Method (FGSM, BID6 which only requires a single gradient update and considers a relatively small subset of adversarial directions.

The resulting performance of Manifold Mixup against FGSM are given in TAB2 .

A challenge in evaluating adversarial examples comes from the gradient masking problem in which a defense succeeds solely due to reducing the quality of the gradient signal.

BID2 explored this issue in depth and proposed running an unbounded search for a large number of iterations to confirm the quality of the gradient signal.

Our Manifold Mixup passed this sanity check (see Appendix F).

While we found that Manifold Mixup greatly improved robustness to the FGSM attack, especially over Input Mixup (Zhang et al., 2018) , we found that Manifold Mixup did not significantly improve robustness against the stronger iterative projected gradient descent (PGD) attack (Madry et al., 2018) .

An important question is what kinds of feature combinations are being explored when we perform mixup in the hidden layers as opposed to linear interpolation in visible space.

To provide a qualita- tive study of this, we trained a small decoder convnet (with upsampling layers) to predict an image from the Manifold Mixup classifier's hidden representation (using a simple squared error loss in the visible space).

We then performed mixup on the hidden states between two random examples, and ran this interpolated hidden state through the convnet to get an estimate of what the point would look like in input space.

Similarly to earlier results on auto-encoders BID3 , we found that these interpolated h points corresponded to images with a blend of the features from the two images, as opposed to the less-semantic pixel-wise blending resulting from Input Mixup as shown in FIG4 and FIG5 .

Furthermore, this justifies the training objective for examples mixed-up in the hidden layers: (1) most of the interpolated points correspond to combinations of semantically meaningful factors, thus leading to the more training samples; and (2) none of the interpolated points between objects of two different categories A and B correspond to a third category C, thus justifying a training target which gives 0 probability on all the classes except A and B.

Deep neural networks often give incorrect yet extremely confident predictions on data points which are unlike those seen during training.

This problem is one of the most central challenges in deep learning both in theory and in practice.

We have investigated this from the perspective of the representations learned by deep networks.

In general, deep neural networks can learn representations such that real data points are widely distributed through the space and most of the area corresponds to high confidence classifications.

This has major downsides in that it may be too easy for the network to provide high confidence classification on points which are off of the data manifold and also that it may not provide enough incentive for the network to learn highly discriminative representations.

We have presented Manifold Mixup, a new algorithm which aims to improve the representations learned by deep networks by encouraging most of the hidden space to correspond to low confidence classifications while concentrating the hidden states for real examples onto a lower dimensional subspace.

We applied Manifold Mixup to several tasks and demonstrated improved test accuracy and dramatically improved test likelihood on classification, better robustness to adversarial examples from FGSM attack, and improved semi-supervised learning.

Manifold Mixup incurs virtually no additional computational cost, making it appealing for practitioners.

We conducted experiments using a generated synthetic dataset where each image is deterministically rendered from a set of independent factors.

The goal of this experiment is to study the impact of input mixup and an idealized version of Manifold Mixup where we know the true factors of variation in the data and we can do mixup in exactly the space of those factors.

This is not meant to be a fair evaluation or representation of how Manifold Mixup actually performs -rather it's meant to illustrate how generating relevant and semantically meaningful augmented data points can be much better than generating points which are far off the data manifold.

We considered three tasks.

In Task A, we train on images with angles uniformly sampled between (-70 ??? , -50 ??? ) (label 0) with 50% probability and uniformly between (50??? , 80 ??? ) (label 1) with 50% probability.

At test time we sampled uniformly between (-30 ??? , -10 ??? ) (label 0) with 50% probability and uniformly between (10 ??? , 30 ??? ) (label 1) with 50% probability.

Task B used the same setup as Task A for training, but the test instead used (-30 ??? , -20 ??? ) as label 0 and (-10 ??? , 30 ??? ) as label 1.

In Task C we made the label whether the digit was a "1" or a "7", and our training images were uniformly sampled between (-70 ??? , -50 ??? ) with 50% probability and uniformly between (50??? , 80??? ) with 50% probability.

The test data for Task C were uniformly sampled with angles from (-30??? , 30 DISPLAYFORM0 The examples of the data are in FIG6 and results are in table 4.

In all cases we found that Input Mixup gave some improvements in likelihood but limited improvements in accuracy -suggesting that the even generating nonsensical points can help a classifier trained with Input Mixup to be better calibrated.

Nonetheless the improvements were much smaller than those achieved with mixing in the ground truth attribute space.

Figure 6 : An experiment on a network trained on the 2D spiral dataset with a 2D bottleneck hidden state in the middle of the network (the same setup as 1).

Noise refers to gaussian noise in the bottleneck layer, dropout refers to dropout of 50% in all layers except the bottleneck itself (due to its low dimensionality), and batch normalization refers to batch normalization in all layers.

This shows that the effect of concentrating the hidden states for each class and providing a broad region of low confidence between the regions is not accomplished by the other regularizers.

We have found significant improvements from using Manifold Mixup, but a key question is whether the improvements come from changing the behavior of the layers before the mixup operation is applied or the layers after the mixup operation is applied.

This is a place where Manifold Mixup and Input Mixup are clearly differentiated, as Input Mixup has no "layers before the mixup operation" to change.

We conducted analytical experimented where the representations are low-dimensional enough to visualize.

More concretely, we trained a fully connected network on MNIST with two fully-connected leaky relu layers of 1024 units, followed by a 2-dimensional bottleneck layer, followed by two more fully-connected leaky-relu layers with 1024 units.

We then considered training with no mixup, training with mixup in the input space, and training only with mixup directly following the 2D bottleneck.

We consistently found that Manifold Mixup has the effect of making the representations much tighter, with the real data occupying more specific points, and with a more well separated margin between the classes, as shown in FIG7 C SUPERVISED REGULARIZATION For supervised regularization we considered architectures within the PreActResNet family: PreActResNet18, PreActResNet34, and PreActResNet152.

When using Manifold Mixup, we selected the layer to perform mixing uniformly at random from a set of eligible layers.

In our experiments on PreActResNets in TAB2 , for Manifold Mixup, our eligible layers for mixing were : the input layer, the output from the first resblock, and the output from the second resblock.

For PreActResNet18, the first resblock has four layers and the second resblock has four layers.

For PreActResNet34, the first resblock has six layers and the second resblock has eight layers.

For PreActResNet152, the first resblock has 9 layers and the second resblock has 24 layers.

Thus the mixing is often done fairly deep in the network, for example in PreActResNet152 the output of the second resblock is preceded by a total of 34 layers (including the initial convolution which is not in a resblock).

For Manifold Mixup All layers in Table 1a , our eligible layers for mixing were : the input layer, the output from the first resblock, and the output from the second resblock, and the output from the third resblock.

We trained all models for 1200 epochs and dropped the learning rates by a factor of 0.1 at 400 epochs and 800 epochs.

Table 6 presents results for SVHN dataset with PreActResNet18 architecture.

In Figure 9 and FIG0 , we present the training loss (Binary cross entropy) for Cifar10 and Cifar100 datasets respectively.

We observe that performing Manifold Mixup in higher layers allows the train loss to go down faster as compared against the Input Mixup.

This is consistent with the demonstration in FIG0 : Input mixup can suffer from underfitting since the interpolation between two examples can intersect with a real example.

In Manifold Mixup the hidden states in which the interpolation is performed, are learned, hence during the course of training they can evolve in such a way that the aforementioned intersection issue is avoided.

We present the procedure for Semi-supervised Manifold Mixup and Semi-supervised Input Mixup in Algorithms 1 and 3 respectively.

Algorithm 1 Semi-supervised Manifold Mixup.

f ?? : Neural Network; M anif oldM ixup: Manifold Mixup Algorithm 2; D L : set of labelled samples; D U L : set of unlabelled samples; ?? : consistency coefficient (weight of unlabeled loss, which is ramped up to increase from zero to its max value over the course of training); N : number of updates;??? i : Mixedup labels of labelled samples;?? i : predicted label of the labelled samples mixed at a hidden layer; y j : Psuedolabels for unlabelled samples;??? j : Mixedup Psuedolabels of unlabelled samples;?? j predicted label of the unlabelled samples mixed at a hidden layer DISPLAYFORM0 Cross Entropy loss 6: DISPLAYFORM1 g ??? ??? ?? L (Gradients of the minibatch Loss )12:?? ??? Update parameters using gradients g (e.g. SGD ) 13: end while DISPLAYFORM2 Sample labeled batch 5: DISPLAYFORM3 Compute Pseudolabels 9: DISPLAYFORM4 10: DISPLAYFORM5 g ??? ??? ?? L Gradients of the minibatch Loss 13:?? ??? Update parameters using gradients g (e.g. SGD ) 14: end while Table 5 : Models trained on the normal CIFAR-100 and evaluated on a test set with novel deformations.

Manifold Mixup (ours) consistently allows the model to be more robust to random shearing, rescaling, and rotation even though these deformations were not observed during training.

For the rotation experiment, each image is rotated with an angle uniformly sampled from the given range.

Likewise the shearing is performed with uniformly sampled angles.

Zooming-in refers to take a bounding box at the center of the image with k% of the length and k% of the width of the original image, and then expanding this image to fit the original size.

Likewise zooming-out refers to drawing a bounding box with k% of the height and k% of the width, and then taking this larger area and scaling it down to the original size of the image (the padding outside of the image is black).

2.37 Input Mixup (?? = 1.5) 2.41 Manifold Mixup (?? = 1.5) 1.92 Manifold Mixup (?? = 2.0) 1.90 Figure 9 : CIFAR-10 train set Binary Cross Entropy Loss (BCE) on Y-axis using PreActResNet18, with respect to training epochs (X-axis).

The numbers in {} refer to the resblock after which Manifold Mixup is performed.

The ordering of the losses is consistent over the course of training: Manifold Mixup with gradient blocked before the mixing layer has the highest training loss, followed by Input Mixup.

The lowest training loss is achieved by mixing in the deepest layer, which is highly consistent with Section 3 which suggests that having more hidden units can help to prevent underfitting.

FIG0 : CIFAR-100 train set Binary Cross Entropy Loss (BCE) on Y-axis using PreActResNet50, with respect to training epochs (X-axis).

The numbers in {} refer to the resblock after which Manifold Mixup is performed.

The lowest training loss is achieved by mixing in the deepest layer.

We use the WideResNet28-2 architecture used in (Oliver et al., 2018) and closely follow their experimental setup for fair comparison with other Semi-supervised learning algorithms.

We used SGD with momentum optimizer in our experiments.

For Cifar10, we run the experiments for 1000 epochs with initial learning rate is 0.1 and it is annealed by a factor of 0.1 at epoch 500, 750 and 875.

For SVHN, we run the experiments for 200 epochs with initial learning rate is 0.1 and it is annealed by a factor of 0.1 at epoch 100, 150 and 175.

The momentum parameter was set to 0.9.

We used L2 regularization coefficient 0.0005 and L1 regularization coefficient 0.001 in our experiments.

We use the batch-size of 100.The data pre-processing and augmentation in exactly the same as in (Oliver et al., 2018) .

For CIFAR-10, we use the standard train/validation split of 45,000 and 5000 images for training and validation respectively.

We use 4000 images out of 45,000 train images as labelled images for semi-supervised learning.

For SVHN, we use the standard train/validation split with 65932 and 7325 images for training and validation respectively.

We use 1000 images out of 65932 images as labelled images for semi-supervised learning.

We report the test accuracy of the model selected based on best validation accuracy.

For supervised loss, we used ?? (of ?? ??? Beta(??, ??)) from the set { 0.1, 0.2, 0.3... 1.0} and found 0.1 to be the best.

For unsupervised loss, we used ?? from the set {0.1, 0.5, 1.0, 1.5, 2.0.

3.0, 4.0} and found 2.0 to be the best.

The consistency coefficient is ramped up from its initial value 0.0 to its maximum value at 0.4 factor of total number of iterations using the same sigmoid schedule of (Tarvainen & Valpola, 2017) .

For CIFAR-10, we found max consistency coefficient = 1.0 to be the best.

For SVHN, we found max consistency coefficient = 2.0 to be the best.

When using Manifold Mixup, we selected the layer to perform mixing uniformly at random from a set of eligible layers.

In our experiments on WideResNet28-2 in TAB1 , our eligible layers for mixing were : the input layer, the output from the first resblock, and the output from the second resblock.

We ran the unbounded projected gradient descent (PGD) (Madry et al., 2018) sanity check suggested in BID2 .

We took our trained models for the input mixup baseline and manifold mixup and we ran PGD for 200 iterations with a step size of 0.01 which reduced the mixup model's accuracy to 1% and reduced the Manifold Mixup model's accuracy to 0%.

This is direct evidence that our defense did not improve results primarily as a result of gradient masking.

The Fast Gradient Sign Method (FGSM) BID6 is a simple one-step attack that produces x = x + ?? sgn(???xL(??, x, y)).

The recent literature has suggested that regularizing the discriminator is beneficial for training GANs (Salimans et al., 2016; BID7 Miyato et al., 2018b) .

In a similar vein, one could add mixup to the original GAN training objective such that the extra data augmentation acts as a beneficial regularization to the discriminator, which is what was proposed in Zhang et al. (2018) .

Mixup proposes the following objective 4 : max DISPLAYFORM0 where x1, x2 can be either real or fake samples, and ?? is sampled from a U nif orm(0, ??).

Note that we have used a function y(??; x1, x2) to denote the label since there are four possibilities depending on x1 and x2: DISPLAYFORM1 if x1 is real and x2 is fake 1 ??? ??, if x1 is fake and x2 is real 0, if both are fake 1, if both are realIn practice however, we find that it did not make sense to create mixes between real and real where the label is set to 1, (as shown in equation 9), since the mixup of two real examples in input space is not a real example.

So we only create mixes that are either real-fake, fake-real, or fake-fake.

Secondly, instead of using just the equation in 8, we optimize it in addition to the regular minimax GAN equations: DISPLAYFORM2 Using similar notation to earlier in the paper, we present the manifold mixup version of our GAN objective in which we mix in the hidden space of the discriminator: DISPLAYFORM3 where h k (??) is a function denoting the intermediate output of the discriminator at layer k, and d k (??) the output of the discriminator given input from layer k.

The layer k we choose the sample can be arbitrary combinations of the input layer (i.e., input mixup), or the first or second resblocks of the discriminator, all with equal probability of selection.

We run some experiments evaluating the quality of generated images on CIFAR10, using as a baseline JSGAN with spectral normalization (Miyato et al., 2018b ) (our configuration is almost identical to theirs).

Results are averaged over at least three runs 5 .

From these results, the best-performing mixup experiments (both input and Manifold Mixup) is with ?? = 0.5, with mixing in all layers (both resblocks and input) achieving an average Inception / FID of 8.04 ?? 0.08 / 21.2 ?? 0.47, input mixup achieving 8.03 ?? 0.08 / 21.4 ?? 0.56, for the baseline experiment 7.97 ?? 0.07 / 21.9 ?? 0.62.

This suggests that mixup acts as a useful regularization on the discriminator, which is even further improved by Manifold Mixup.

See FIG0 for the full set of experimental results

An essential motivation behind manifold mixup is that because the network learns the hidden states, it can do so in such a way that the interpolations between points are consistent.

Section 3 characterized this for hidden states with any number of dimensions and FIG0 showed how this can occur on the 2d spiral dataset.

Our goal here is to discuss concrete examples to illustrate what it means for the interpolations to be consistent.

If we consider any two points, the interpolated point between them is based on a sampled ?? and the soft-target for that interpolated point is the targets interpolated with the same ??.

So if we consider two points A,B which have the same label, it is apparent that every point on the line between A and B should have that same label with 100% confidence.

If we consider two points A,B with different labels, then the point which is halfway between them will be given the soft-label of 50% the label of A and 50% the label of B (and so on for other ?? values).It is clear that for many arrangements of data points, it is possible for a point in the space to be reached through distinct interpolations between different pairs of examples, and reached with different ?? values.

Because the learned model tries to capture the distribution p(y|h), it can only assign a single distribution over the label values to a single particular point (for example it could say that a point is 100% label A, or it could say that a point is 50% label A and 50% label B).

Intuitively, these inconsistent soft-labels at interpolated points can be avoided if the states for each class are more concentrated and classes vary along distinct dimensions in the hidden space.

The theory in Section 3 characterizes exactly what this concentration needs to be: that the representations for each class need to lie on a subspace of dimension equal to "number of hidden dimensions" -"number of classes" + 1.Figure 12: We consider a binary classification task with four data points represented in a 2D hidden space.

If we perform mixup in that hidden space, we can see that if the points are laid out in a certain way, two different interpolations can give inconsistent soft-labels (left and middle).

This leads to underfitting and high loss.

When training with manifold mixup, this can be explicitly avoided because the states are learned, so the model can learn to produce states for which all interpolations give consistent labels, an example of which is seen on the right side of the figure.

When we refer to flattening, we mean that the class-specific representations have reduced variability in some directions.

Our analysis in this section makes this more concrete.

We trained an MNIST classifier with a hidden state bottleneck in the middle with 12 units (intentionally selected to be just slightly greater than the number of classes).

We then took the representation for each class and computed a singular value decomposition ( FIG0 and FIG0 and we also computed an SVD over all of the representations together ( FIG0 ).

Our architecture contained three hidden layers with 1024 units and LeakyReLU activation, followed by a bottleneck representation layer (with either 12 or 30 hidden units), followed by an additional four hidden layers each with 1024 units and LeakyReLU activation.

When we performed Manifold Mixup for our analysis, we only performed mixing in the bottleneck layer, and used a beta distribution with an alpha of 2.0.

Additionally we performed another experiment FIG0 where we placed the bottleneck representation layer with 30 units immediately following the first hidden layer with 1024 units and LeakyReLU activation.

We found that Manifold Mixup had a striking effect on the singular values, with most of the singular values becoming much smaller.

Effectively, this means that the representations for each class have variance in fewer directions.

While our theory in Section 3 showed that this flattening must force each classes representations onto a lower-dimensional subspace (and hence an upper bound on the number of singular values) but this explores how this occurs empirically and does not require the number of hidden dimensions to be so small that it can be manually visualized.

In our experiments we tried using 12 hidden units in the bottleneck FIG0 as well as 30 hidden units FIG0 in the bottleneck.

Our results from this experiment are unequivocal: Manifold Mixup dramatically reduces the size of the smaller singular values for each classes representations.

This indicates a flattening of the class-specific representations.

At the same time, the singular values over all the representations are not changed in a clear way FIG0 ), which suggests that this flattening occurs in directions which are distinct from the directions occupied by representations from other classes, which is the same intuition behind our theory.

Moreover, FIG0 shows that when the mixing is performed earlier in the network, there is still a flattening effect, though it is weaker than in the later layers, and again Input Mixup has an inconsistent effect.

Figure 13: SVD on the class-specific representations in a bottleneck layer with 12 units following 3 hidden layers.

For the first singular value, the value (averaged across the plots) is 50.08 for the baseline, 37.17 for Input Mixup, and 43.44 for Manifold Mixup (these are the values at x=0 which are cutoff).

We can see that the class-specific SVD leads to singular values which are dramatically more concentrated when using Manifold Mixup with Input Mixup not having a consistent effect.

We compare the performance of Manifold Mixup using different values of hyper-parameter ?? by training a PreActResNet18 network on Cifar10 dataset, as shown in TAB6 .

Manifold Mixup outperformed Input Mixup for all alphas in the set (0.5, 1.0, 1.2, 1.5, 1.8, 2.0) -indeed the lowest result for Manifold Mixup is better than the worst result with Input Mixup.

Note that Input Mixup's results deteriorate when using an alpha that is too large, which is not seen with manifold mixup.

In this section, we discuss which layers are a good candidate for mixing in the Manifold Mixup algorithm.

We evaluated PreActResNet18 models on CIFAR-10 and considered mixing in a subset of the layers, we ran for fewer epochs than in the Section 5.1 (making the accuracies slightly lower across the board), and we decided to fix the alpha to 2.0 as we did in the the Section 5.1.

We considered different subsets of layers to mix in, with 0 referring to the input layer, 1/2/3 referring to the output of the 1st/2nd/3rd resblocks respectively.

For example 0,2 refers to mixing in the input layer and the output of the 2nd resblock.

{} refers to no mixing.

The results are presented in TAB7 Essentially, it helps to mix in more layers, except for the later layers which hurts the test accuracy to some extent -which is consistent with our theory in Section 3 : the theory in Section 3 assumes that the part of the network after mixing is a universal approximator, hence, there is a sensible case to be made for not mixing in the very last layers.

@highlight

A method for learning better representations, that acts as a regularizer and despite its no significant additional computation cost , achieves improvements over strong baselines on Supervised and Semi-supervised Learning tasks.