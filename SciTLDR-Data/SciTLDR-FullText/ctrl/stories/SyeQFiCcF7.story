Capsule Networks have shown encouraging results on defacto benchmark computer vision datasets such as MNIST, CIFAR and smallNORB.

Although, they are yet to be tested on tasks where (1) the entities detected inherently have more complex internal representations and (2) there are very few instances per class to learn from and (3) where point-wise classification is not suitable.

Hence, this paper carries out experiments on face verification in both controlled and uncontrolled settings that together address these points.

In doing so we introduce Siamese Capsule Networks, a new variant that can be used for pairwise learning tasks.

The model is trained using contrastive loss with l2-normalized capsule encoded pose features.

We find that Siamese Capsule Networks perform well against strong baselines on both pairwise learning datasets, yielding best results in the few-shot learning setting where image pairs in the test set contain unseen subjects.

Convolutional Neural networks (CNNs) have been a mainstay model for a wide variety of tasks in computer vision.

CNNs are effective at detecting local features in the receptive field, although the spatial relationship between features is lost when crude routing operations are performed to achieve translation invariance, as is the case with max and average pooling.

Essentially, pooling results in viewpoint invariance so that small perturbations in the input do not effect the output.

This leads to a significant loss of information about the internal properties of present entities (e.g location, orientation, shape and pose) in an image and relationships between them.

The issue is usually combated by having large amounts of annotated data from a wide variety of viewpoints, albeit redundant and less efficient in many cases.

As noted by hinton1985shape, from a psychology perspective of human shape perception, pooling does not account for the coordinate frames imposed on objects when performing mental rotation to identify handedness BID20 ; BID16 BID10 .

Hence, the scalar output activities from local kernel regions that summarize sets of local inputs are not sufficient for preserving reference frames that are used in human perception, since viewpoint information is discarded.

Spatial Transformer Networks (STN) BID11 have acknowledged the issue by using dynamic spatial transformations on feature mappings to enhance the geometric invariance of the model, although this approach addresses changes in viewpoint by learning to remove rotational and scale variance, as opposed to viewpoint variance being reflected in the model activations.

Instead of addressing translation invariance using pooling operations, BID6 have worked on achieving translation equivariance.

The recently proposed Capsule Networks BID21 ; BID5 have shown encouraging results to address these challenges.

Thus far, Capsule Networks have only been tested on datasets that have (1) a relatively sufficient number of instances per class to learn from and (2) utilized on tasks in the standard classification setup.

This paper extends Capsule Networks to the pairwise learning setting to learn relationships between whole entity encodings, while also demonstrating their ability to learn from little data that can perform few-shot learning where instances from new classes arise during testing (i.e zero-shot prediction).

The Siamese Capsule Network is trained using a contrastive loss with 2 -normalized encoded features and demonstrated on two face verification tasks.

BID6 first introduced the idea of using whole vectors to represent internal properties (referred to as instantiation parameters that include pose) of an entity with an associated activation probability where each capsule represents a single instance of an entity within in an image.

This differs from the single scalar outputs in conventional neural networks where pooling is used as a crude routing operation over filters.

Pooling performs sub-sampling so that neurons are invariant to viewpoint change, instead capsules look to preserve the information to achieve equivariance, akin to perceptual systems.

Hence, pooling is replaced with a dynamic routing scheme to send lowerlevel capsule (e.g nose, mouth, ears etc.) outputs as input to parent capsule (e.g face) that represent part-whole relationships to achieve translation equivariance and untangles the coordinate frame of an entity through linear transformations.

The idea has its roots in computer graphics where images are rendered given an internal hierarchical representation, for this reason the brain is hypothesized to solve an inverse graphics problem where given an image the cortex deconstructs it to its latent hierarchical properties.

The original paper by BID21 describes a dynamic routing scheme that represent these internal representations as vectors given a group of designated neurons called capsules, which consist of a pose vector u ∈ R d and activation α ∈ [0, 1].

The architecture consists of two convolutional layers that are used as the initial input representations for the first capsule layer that are then routed to a final class capsule layer.

The initial convolutional layers allow learned knowledge from local feature representations to be reused and replicated in other parts of the receptive field.

The capsule inputs are determined using a Iterative Dynamic Routing scheme.

A transformation W ij is made to output vector u i of capsule C L i .

The length of the vector u i represents the probability that this lower-level capsule detected a given object and the direction corresponds to the state of the object (e.g orientation, position or relationship to upper capsule).

The output vector u i is transformed into a prediction vectorû j|i , whereû j|i = W ij u i .

Then,û j|i is weighted by a coupling coefficient c ij to obtain s j = i c ijûj|i , where coupling coefficients for each capsule j c ij = 1 and c ij is got by log prior probabilities b ij from a sigmoid function, followed by the softmax, c ij = e bij / k e b ik .

Ifû L j|i has high scalar magnitude when multiplied by u L+1 j then the coupling coefficient c ij is increased and the remaining potential parent capsules coupling coefficients are decreased.

Routing By Agreement is then performed using coincidence filtering to find tight clusters of nearby predictions.

The entities output vector length is represented as the probability of an entity being present by using the nonlinear normalization shown in Equation 1 where vote v j is the output from total input s j , which is then used to compute the agreement a ij = v jûj|i that is added to the log prior b ij .

The capsule is assigned a high log-likelihood if densely connected clusters of predictions are found from a subset of s. The centroid of the dense cluster is output as the entities generalized pose.

This coincidence filtering step can also be achieved by traditional outlier detection methods such as Random sample consensus (RANSAC) BID3 BID3 classical Hough Transforms Ballard (1987) for finding subsets of the feature space with high agreement.

Although, the motivation for using the vector normalization of the instantiation parameters is to force the network to preserve orientation.

Lastly, a reconstruction loss on the images was used for regularization which constrains th capsules to learn properties that can better encode the entities.

In this paper, we do not use such regularization scheme by autoencoding pairs of input images, instead we use a variant of dropout.

BID5 recently describe matrix capsules that perform routing by agreement using the expectation maximization (EM) algorithm, motivated by computer graphics where pose matrices are used to define rotations and translations of objects to account for viewpoint changes.

Each parent capsule is considered a Gaussian and the pose matrix of each child capsule are considered data samples of the Gaussian.

A given layer L contains a set of capsules C L such that ∀C ∃ {M , α } ∈ C L where pose matrix M ∈ R n×n (n = 4) and activation α ∈ [0, 1] are the outputs.

A vote is made V ij = M i W ij for the pose matrix of C L+1 j where W ij ∈ R n×n is a learned viewpoint invariant transformation matrix from capsule

where the cost h j is the negative log-probability density weighted by the assignment probabilities r ij , −β u is the negative log probability density per pose matrix computed to describe C L+1 j .

If C L+1 j is activated −β a is the cost for describing (µ j , σ 2 j ) from lower-level pose data samples along with r ij and λ is the inverse temperature so as the assignment probability becomes higher the slope of the sigmoid curve becomes steeper (represents the presence of an entity instead of the nonlinear vector normalization seen in Equation 1).

The network uses 1 standard convolutional layer, a primary capsule layer, 2 intermediate capsule convolutional layer, followed by the final class capsule layer.

The matrix capsule network significantly outperformed CNNs on the SmallNORB dataset.

LaLonde & Bagci FORMULA7 introduce SegCaps which uses a locally connected dynamic routing scheme to reduce the number of parameters while using deconvolutional capsules to compensate for the loss of global information, showing best performance for segmenting pathological lungs from low dose CT scans.

The model obtained a 39% and 95% reduction in parameters over baseline architectures while outperforming both.

Bahadori FORMULA7 introduced Spectral Capsule Networks demonstrated on medical diagnosis.

The method shows faster convergence over the EM algorithm used with pose vectors.

Spatial coincidence filters align extracted features on a 1-d linear subspace.

The architecture consists of a 1d convolution followed by 3 residual layers with dilation.

Residual blocks R are used as nonlinear transformations for the pose and activation of the first primary capsule instead of the linear transformation that accounts for rotations in CV, since deformations made in healthcare imaging are not fully understood.

The weighted votes are obtained as s j,i = α i R j (u i ) ∀i where S j is a matrix of concatenated votes that are then decomposed using SVD, where the first singular value dimensions 1 is used to capture most of the variance between votes, thus the activation a j activation is computed as σ η(s DISPLAYFORM0 k is the ratio of all variance explained for all right singular vectors in V , b is optimized and η is decreased during training.

The model is trained by maximizing the log-likelihood showing better performance than the spread loss used with matrix capsules and mitigates the problem of capsules becoming dormant.

BID27 formalize the capsule routing strategy as an optimization of a clustering loss and a KL regularization term between the coupling coefficient distribution and its past states.

The proposed objective function follows as min C,S {L(C, S) := − i j c ij o j|i , s j + α i j c ij log c ij } where o j|i = T ij µ i /||T ij || F and ||T ij || F is the Frobenious norm of T ij .

This routing scheme shows significant benefit over the original routing scheme by BID21 as the number of routing iterations increase.

Evidently, there has been a surge of interest within the research community.

In contrast, the novelty presented in this paper is the pairwise learning capsule network scheme that proposes a different loss function, a change in architecture that compares images, aligns entities across images and describes a method for measuring similarity between final layer capsules such that inter-class variations are maximized and intra-class variations are minimized.

Before describing these points in detail, we briefly describe the current state of the art work (SoTA) in face verification that have utilized Siamese Networks.

Siamese Networks (SNs) are neural networks that learn relationships between encoded representations of instance pairs that lie on low dimensional manifold, where a chosen distance function d ω is used to find the similarity in output space.

Below we briefly describe state of the art convolutional SN's that have been used for face verification and face recognition.

BID24 presented a joint identification-verification approach for learning face verification with a contrastive loss and face recognition using cross-entropy loss.

To balance loss signals for both identification and verification, they investigate the effects of varying weights controlled by λ on the intra-personal and inter-personal variations, where λ = 0 leaves only the face recognition loss and λ → ∞ leaves the face verification loss.

Optimal results are found when λ = 0.05 intra personal variation is maximized while both class are distinguished.

BID28 propose a center loss function to improve discriminative feature learning in face recognition.

The center loss function proposed aims to improve the discriminability between feature representations by minimizing the intra-class variation while keeping features from different classes separable.

The center loss is given as DISPLAYFORM0 where z = W T j x i + b j .

The c yi is the centroid of feature representations pertaining to the i th class.

This penalizes the distance between class centers and minimizes the intra-class variation while the softmax keeps the inter-class features separable.

The centroids are computed during stochastic gradient descent as full batch updates would not be feasible for large networks.

BID15 proposed Sphereface, a hypersphere embedding that uses an angular softmax loss that constrains disrimination on a hypersphere manifold, motivated by the prior that faces lie on a manifold.

The model achieves 99.22 % on the LFW dataset, and competitive results on Youtube Face (YTF) and MegaFace.

BID22 proposed a triplet similarity embedding for face verification using a triple loss arg min W = α,p,n∈T max(0, α + α T W T W (n − p)) where for T triplet sets lies an anchor class α, positive class p and negative class n, a projection matrix W , (performed PCA to obtain W 0 ) is minimized with the constraint that W BID7 use deep metric learning for face verification with loss arg min DISPLAYFORM1 DISPLAYFORM2 F | where g(z) = log(1 + e βz )/β, β controls the slope steepness of the logistic function, ||A|| F is the frobenius norm of A and λ is a regularization parameter.

Hence, the loss function is made up of a logistic loss and regularization on parameters θ = [W, b] .

Best results are obtained using a combination of SIFT descriptors, dense SIFT and local binary patterns (LBP), obtaining 90.68% (+/-1.41) accuracy on the LFW dataset.

BID18 used an 2 -constraint on the softmax loss for face verification so that the encoded face features lie on the ambit of a hypersphere, showing good improvements in performance.

This work too uses an 2 -constraint on capsule encoded face embeddings.

FaceNet BID23 too uses a triplet network that combines the Inception network BID25 and a 8-layer convolutional model BID29 which learns to align face patches during training to perform face verification, recognition and clustering.

The method trains the network on triplets of increasing difficulty using a negative example mining technique.

Similarly, we consider a Siamese Inception Network for the tasks as one of a few comparisons to SCNs.

The most relevant and notable use of Siamese Networks for face verification is the DeepFace network, introduced by BID26 .

The performance obtained was on par with human level performance on the Faces in the Wild (LFW) dataset and significantly outperformed previous methods.

However, it is worth noting this model is trained on a large dataset from Facebook (SFC), therefore the model can be considered to be performing transfer learning before evaluation.

The model also carries out some manual steps for detecting, aligning and cropping faces from the images.

For detecting and aligning the face a 3D model is used.

The images are normalized to avoid any differences in illumination values, before creating a 3D model which is created by first identifying 6 fiducial points in the image using a Support Vector Regressor from a LBP histogram image descriptor.

Once the faces are cropped based on these points, a further 67 fiducial point are identified for 3D mesh model, followed by a piecewise affine transformation for each section of the image.

The cropped image is then passed to 3 CNN layers with an initial max-pooling layer followed two fully-connected layers.

Similar to Capsule Networks, the authors refrain from using max pooling at each layer due to information loss.

In contrast to this work, the only preprocessing steps for the proposed SCNs consist of pixel normalization and a reszing of the image.

The above work all achieve comparable state of the art results for face verification using either a single CNN or a combination of various CNNs, some of which are pretrained on large related datasets.

In contrast, this work looks to use a smaller Capsule Network that is more efficient, requires little preprocessing steps (i.e only a resizing of the image and normalization of input features, no aligning, cropping etc.) and can learn from relatively less data.

The Capsule Network for face verification is intended to identify enocded part-whole relationships of facial features and their pose that in turn leads to an improved similarity measure by aligning capsule features across paired images.

The architecture consists of a 5-hidden layer (includes 2 capsule layers) network with tied weights (since both inputs are from the same domain).

The 1 st layer is a convolutional filter with a stride of 3 and 256 channels with kernels κ 1 i ∈ R 9×9 ∀i over the image pairs x 1 , x 2 ∈ R 100×100 , resulting in 20, 992 parameters.

The 2 nd layer is the primary capsule layer that takes κ(1) and outputs κ (2) ∈ R 31×31 matrix for 32 capsules, leading to 5.309 × 10 6 parameters (663, 552 weights and 32 biases for each of 8 capsules).

The 3 rd layer is the face capsule layer, representing the routing of various properties of facial features, consisting of 5.90 × 10 6 parameters.

This layer is then passed to a single fully connected layer by concatenating DISPLAYFORM0 as input, while the sigmoid functions control the dropout rate for each capsule during training.

The nonlinear vector normalization shown in Equation 1 is replaced with a tanh function tanh(.) which we found in initial testing to produce better results.

Euclidean distance, Manhattan distance and cosine similarity are considered as measures between the capsule image encodings.

The aforementioned SCN architecture describes the setup for the AT&T dataset.

For the LFW dataset, 6 routing iterations are used and 4 for AT&T.Capsule Encoded Representations To encode paired images x 1 , x 2 into vector pairs h 1 , h 2 the pose vector of each capsule is vectorized and passed as input to a fully connected layer containing 20 activation units.

Hence, for each input there is a lower 20-dimensional representation of 32 capsule pose vectors resulting in 512 input features.

To ensure all capsules stay active the dropout probability rate is learned for each capsule.

The sigmoid function learns the dropout rate of the final capsule layer using Concrete Dropout BID4 , which builds on prior work Kingma et al. FORMULA7 ; BID17 by using a continuous relaxation that approximates the discrete Bernoulli distribution used for dropout, referred to as a concrete distribution.

Equation 2 shows the objective function for updating the concrete distribution.

For a given capsule probability p c in the last capsule layer, the sigmoid computes the relaxationz on the Bernoulli variable z, where u is drawn uniformly between [0,1] where t denotes the temperature values (t = 0.1 in our experiments) which forces probabilities at the extremum when small.

The pathwise derivative estimator is used to find a continuous estimation of the dropout mask.

The weight λ is used to prevent the activity vector lengths from deteriorating early in training if a class capsule is absent.

The overall loss is then simply the sum of the capsule losses c L c .

A spread loss BID5 has also been used to maximize the inter-class distance between the target class and the remaining classes for classifying on the smallNORB dataset.

This is given as DISPLAYFORM1 DISPLAYFORM2 where the margin m is increased linearly during training to ensure lower-level capsule stay active throughout training.

This work instead uses a contrastive margin loss BID2 where the aforementioned capsule encoding similarity function d ω outputs a predicted similarity score.

The contrastive loss L c ensures similar vectorized pose encodings are drawn together and dissimilar poses repulse.

Equation 3 shows a a pair of images that are passed to the SCN model where DISPLAYFORM3 2 computes the Euclidean distance between encodings and m is the margin.

When using Manhattan distance DISPLAYFORM4 A double margin loss that has been used in prior work by BID14 is also considered to affect matching pairs such that to account for positive pairs that can also have high variance in the distance measure.

It is worth noting this double margin is similar to the aforementioned margin loss used on class capsules, without the use of λ.

Equation 4 shows the double-margin contrastive loss where positive margin m p and negative margin m n are used to find better separation between matching and non-matching pairs.

This loss is only used for LFW, given the limited number of instances in AT&T we find the amount of overlap between pairs to be less severe in experimentation.

DISPLAYFORM5 The original reconstruction loss DISPLAYFORM6 i ) 2 used as regularization is not used in the pairwise learning setting, instead we rely on the dropout for regularization with exception of the SCN model that uses concrete dropout on the final layer.

Optimization Convergence can often be relatively slow for face verification tasks, where few informative batch updates (e.g a sample with significantly different pose for a given class) get large updates but soon after the effect is diminished through gradient exponential averaging (originally introduced to prevent α → 0).

Motivated by recent findings that improve adaptive learning rates we use AMSGrad BID19 .

AMSGrad improves over ADAM in some cases by replacing the exponential average of squared gradients with a maximum that mitigates the issue by keeping long-term memory of past gradients.

Thus, AMSGrad does not increase or decrease the learning rate based on gradient changes, avoiding divergent or vanishing step sizes over time.

Equation 5 presents the update rule, where diagonal of gradient g t is given as DISPLAYFORM7

A. AT&T dataset The AT&T face recognition and verification dataset consists of 40 different subjects with only 10 gray-pixel images per subject in a controlled setting.

This smaller dataset allows us to test how SCNs perform with little data.

For testing, we hold out 5 subjects so that we are testing on unseen subjects, as opposed to training on a given viewpoint of a subject and testing on another viewpoint of the same subject.

Hence, zero-shot pairwise prediction is performed during testing.

The LFW consists of 13,000 colored photographed faces from the web.

This dataset is significantly more complex not only because there 1680 subjects, with some subjects only consisting of two images, but also because of varied amount of aging, pose, gender, lighting and other such natural characteristics.

Each image is 250 × 250, in this work the image is resized to 100×100 and normalized.

From the original LFW dataset there has been 2 different versions of the dataset that align the images using funneling BID8 and deep funneling BID9 Baselines SCNs are compared against well-established architectures for image recognition and verification tasks, namely AlexNet, ResNet-34 and InceptionV3 with 6 inception layers instead of the original network that uses 8 layers which are used many of the aforementioned papers in Section 3.

Table 1 shows best test results obtained when using contrastive loss with Euclidean distance between encodings (i.e Mahalanobis distance) for both AT &T and LFW over 100 epochs.

The former uses m = 2.0 and the latter uses m = 0.2, while for the double margin contrastive loss m n = 0.2 matching margin and m p = 0.5 negative matching margin is selected.

These settings were chosen during 5-fold cross validation, grid searching over possible margin settings.

SCN outperforms baselines on the AT &T dataset after training for 100 epochs.

We find that because AT&T contains far fewer instances an adapted dropout rate leads to a slight increase in contrastive loss.

Additionally, adding a reconstruction loss with λ r = 1e −4 for both paired images led to a decrease in performance when compared to using dropout with a rate p = 0.2 on all layers except the final layer that encodes the pose vectors.

We find for the LFW dataset that the SCN and AlexNet have obtained the best results while SCN has 25% less parameters.

Additionally, the use of a double margin results in better results for the standard SCN but a slight drop in performance when used with concrete dropout on the final layer (i.e SDropCapNet).

Figure 2 illustrates the contrastive loss during training 2 -normalized features for each model tested with various distance measures on AT&T and LFW.

We find that SCN yields faster convergence on AT&T, particularly when using Manhattan distance.

However for Euclidean distance, we observe a loss variance reduction during training and the best overall performance.

Through experiments we find that batch normalized convolutional layers improves performance of the SCN.

In batch normalization,

provides a unit Gaussian batch that is shifted by γ (k) and scaled with DISPLAYFORM0 .

This allows the network to learn whether the input range should be more or less diffuse.

Batch normalization on the initial convolutional layers reduced variance in loss during training on both the AT &T and LF W datasets.

LFW test results show that the SCN model takes longer to converge particularly in the early stages of training, in comparison to AlexNet.

Figure 3 shows the probability density of the positive pair predictions for each model for all distances between encodings with contrastive loss for the LFW dataset.

We find the variance of predictions is lower in comparison to the remaining models, showing a higher precision in the predictions, particularly for Manhattan distance.

Additionally, varying distances for these matching images were close Finally, the SCN model has between 104-116 % less parameters than Alexnet, 24-27 % Resnet-34 and 127-135% less than the best standard baseline for both datasets.

However, even considering tied weights between models in the SCN, Capsule Networks are primarily limited in speed even with a reduction in parameters due to the routing iterations that are necessary during training.

This paper has introduced the Siamese Capsule Network, a novel architecture that extends Capsule Networks to the pairwise learning setting with a feature 2 -normalized contrastive loss that maximizes inter-class variance and minimizes intra-class variance.

The results indicate Capsule Networks perform better at learning from only few examples and converge faster when a contrastive loss is used that takes face embeddings in the form of encoded capsule pose vectors.

We find Siamese Capsule Networks to perform particularly well on the AT&T dataset in the few-shot learning setting, which is tested on unseen classes (i.e subjects) during testing, while competitive against baselines for the larger Labeled Faces In The Wild dataset.

<|TLDR|>

@highlight

A variant of capsule networks that can be used for pairwise learning tasks. Results shows that Siamese Capsule Networks work well in the few shot learning setting.