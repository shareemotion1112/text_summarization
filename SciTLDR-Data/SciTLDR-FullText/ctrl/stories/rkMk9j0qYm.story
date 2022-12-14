We introduce Explainable Adversarial Learning, ExL, an approach for training neural networks that are intrinsically robust to adversarial attacks.

We find that the implicit generative modeling of random noise with the same loss function used during posterior maximization, improves a model's understanding of the data manifold furthering adversarial robustness.

We prove our approach's efficacy and provide a simplistic visualization tool for understanding adversarial data, using Principal Component Analysis.

Our analysis reveals that adversarial robustness, in general, manifests in models with higher variance along the high-ranked principal components.

We show that models learnt with our approach perform remarkably well against a wide-range of attacks.

Furthermore, combining ExL with state-of-the-art adversarial training extends the robustness of a model, even beyond what it is adversarially trained for, in both white-box and black-box attack scenarios.

Despite surpassing human performance on several perception tasks, Machine Learning (ML) models remain vulnerable to adversarial examples: slightly perturbed inputs that are specifically designed to fool a model during test time BID2 BID29 BID6 BID23 .

Recent works have demonstrated the security danger adversarial attacks pose across several platforms with ML backend such as computer vision BID29 BID6 BID22 BID15 BID19 , malware detectors BID16 BID33 BID7 BID10 and gaming environments BID11 BID1 .

Even worse, adversarial inputs transfer across models: same inputs are misclassified by different models trained for the same task, thus enabling simple Black-Box (BB) 1 attacks against deployed ML systems .Several works BID14 BID24 BID3 demonstrating improved adversarial robustness have been shown to fail against stronger attacks BID0 .

The state-of-the-art approach for BB defense is ensemble adversarial training that augments the training dataset of the target model with adversarial examples transferred from other pre-trained models BID30 .

BID21 showed that models can even be made robust to White-Box (WB) 1 attacks by closely maximizing the model's loss with Projected Gradient Descent (PGD) based adversarial training.

Despite this progress, errors still appear for perturbations beyond what the model is adversarially trained for BID27 .There have been several hypotheses explaining the susceptibility of ML models to such attacks.

The most common one suggests that the overly linear behavior of deep neural models in a high dimensional input space causes adversarial examples BID6 BID20 .

Another hypothesis suggests that adversarial examples are off the data manifold BID28 BID18 .

Combining the two, we infer that excessive linearity causes models to extrapolate their behavior beyond the data manifold yielding pathological results for slightly perturbed inputs.

A question worth asking here is: Can we improve the viability of the model to generalize better on such out-of-sample data?In this paper, we propose Explainable Adversarial Learning (ExL), wherein we introduce multiplicative noise into the training inputs and optimize it with Stochastic Gradient Descent (SGD) while minimizing the overall cost function over the training data.

Essentially, the input noise (randomly initialized at the beginning) is gradually learnt during the training procedure.

As a result, the noise approximately models the input distribution to effectively maximize the likelihood of the class labels given the inputs.

FIG0 shows the input noise learnt during different stages of training by a simple convolutional network (ConvN et2 architecture discussed in Section 3 below), learning handwritten digits from MNIST dataset BID17 .

We observe that the noise gradually transforms and finally assumes a shape that highlights the most dominant features in the MNIST training data.

For instance, the MNIST images are centered digits on a black background.

Noise, in fact, learnt this centered characteristic.

This suggests that the model not only finds the right prediction but also the right explanation.

Noise inculcates this explainable behavior by discovering some knowledge about the input/output distribution during training.

FIG0 shows the noise learnt with ExL on colored CIFAR10 images BID13 ) (on ResNet18 architecture BID8 ), which reveals that noise template (also RGB) learns prominent color blobs on a greyish-black background, that de-emphasizes background pixels.

A recent theory BID4 suggests that adversarial examples (off manifold misclassified points) occur in close proximity to randomly chosen inputs on the data manifold that are, in fact, correctly classified.

With ExL, we hypothesize that the model learns to look in the vicinity of the onmanifold data points and thereby incorporate more out-of-sample data (without using any direct data augmentation) that, in turn, improves its generalization capability in the off-manifold input space.

We empirically evaluate this hypothesis by visualizing and studying the relationship between the adversarial and the clean inputs using Principal Component Analysis (PCA).

Examining the intermediate layer's output, we discover that models exhibiting adversarial robustness yield significantly lower distance between adversarial and clean inputs in the Principal Component (PC) subspace.

We further harness this result to establish that ExL noise modeling, indeed, acquires an improved realization of the input/output distribution characteristics that enables it to generalize better.

To further substantiate our hypothesis, we also show that ExL globally reduces the dimensionality of the space of adversarial examples BID31 .

We evaluate our approach on classification tasks such as MNIST, CIFAR10 and CIFAR100 and show that models trained with ExL are extensively more adversarially robust.

We also show that combining ExL with ensemble/PGD adversarial training significantly extends the robustness of a model, even beyond what it is adversarially trained for, in both BB/WB attack scenarios.

The basic idea of ExL is to inject random noise with the training data, continually minimizing the overall loss function by learning the parameters, as well as the noise at every step of training.

The noise, N , dimensionality is same as the input, X, that is, for a 32 ?? 32 ?? 3 sized image, the noise is 32 ?? 32 ?? 3.

In all our experiments, we use mini-batch SGD optimization.

Let's assume the size of the training minibatch is m and the number of images in the minibatch is k, then, total training images are m ?? k. Now, the total number of noisy templates are equal to the total number of inputs in each minibatch, k. Since, we want to learn the noise, we use the same k noise templates across all mini-batches 1, 2, ..., m. This ensures that the noise templates inherit characteristics from the entire training dataset.

Algorithm 1 shows the training procedure.

It is evident from Algorithm 1 that noise learning at every training step follows the overall loss (L, say cross-entropy) minimization that in turn enforces the maximum likelihood of the posterior.

DISPLAYFORM0 Forward Propagation:?? = f (X; ??) DISPLAYFORM1 Compute loss function: L(?? , Y )

Backward Propagation: DISPLAYFORM0 end for 10: until training converges Since adversarial attacks are created by adding perturbation to the clean input images, we were initially inclined toward using additive noise (X + N ) instead of multiplicative noise (X ?? N ) to perform ExL. However, we found that ExL training with X ?? N tends to learn improved noise characteristics by the end of training.

Fig. 2 (a) shows the performance results for different ExL training scenarios.

While ExL with X + N suffers a drastic ??? 10% accuracy loss with respect to standard SGD on clean data, X ?? N yields comparable accuracy.

Furthermore, we observe that using only negative gradients (i.e. ??? N L ??? 0) during backpropagation for ExL yields best accuracy (and closer to that of standard SGD trained model).

Visualizing a sample image with learnt noise after training, in Fig. 2 (b) , shows X + N disturbs the original image severely, while X ?? N has a faint effect, corroborating the accuracy results.

Since noise is modeled while conducting discriminative training, the multiplicative/additive nature of noise influences the overall optimization.

Thus, we observe that noise templates learnt with X ?? N and X + N are very different.

We also analyzed the adversarial robustness of the models when subjected to WB attacks created using the Fast Gradient Sign Method (FGSM) for different perturbation levels ( ) (Fig. 2 (a) ).

ExL, for both X ?? N /X + N scenarios, yields improved accuracy than standard SGD.

This establishes the effectiveness of the noise modeling technique during discriminative training towards improving a model's intrinsic adversarial resistance.

Still, X ?? N yields slightly better resistance than X + N .

Based upon these empirical studies, we chose to conform to multiplicative noise training in this paper.2 Note, WB attacks, in case of ExL, are crafted using the model's parameters as well as the learnt noise N .In all our experiments, we initialize the noise N from a random uniform distribution in the range [0.8, 1].

We select a high range in the beginning of training to limit the corruption induced on the training data due to the additional noise.

During evaluation/testing, we take the mean of the learnt noise across all the templates (( DISPLAYFORM1 , multiply the averaged noise with each test image and feed it to the network to obtain the final prediction.

Next, we present a general optimization perspective considering the maximum likelihood criterion for a classification task to explain adversarial robustness.

It is worth mentioning that while Algorithm 1 describes the backpropagation step simply by using gradient updates, we can use other techniques like regularization, momentum etc.

for improved optimization.

Given a data distribution D with inputs X ??? R d and corresponding labels Y , a classification/discriminative algorithm models the conditional distribution p(Y |X; ??) by learning the parameters ??.

Since X inherits only the on-manifold data points, a standard model thereby becomes susceptible to adversarial attacks.

For adversarial robustness, inclusion of the off-manifold data points while modeling the conditional probability is imperative.

An adversarially robust model should, thus, model p(Y |X, A; ??), where A represents the adversarial inputs.

Using Bayes rule, we can derive the prediction obtained from posterior modeling from a generative standpoint as: DISPLAYFORM0 2 Additional studies on other datasets comparing X +N vs. X ??N with different gradient update conditions can be found in Appendix A. See, experimental details and model description for Fig. 2 The methods employing adversarial training BID30 BID15 BID21 directly follow the left-hand side of Eqn.

1 wherein the training data is augmented with adversarial samples (A ??? A).

Such methods showcase adversarial robustness against a particular form of adversary (e.g. ??? -norm bounded) and hence remain vulnerable to stronger attack scenarios.

In an ideal case, A must encompass all set of adversarial examples (or the entire space of off-manifold data) for a concrete guarantee of robustness.

However, it is infeasible to anticipate all forms of adversarial attacks during training.

From a generative viewpoint (right-hand side of Eqn.

1), adversarial robustness requires modeling of the adversarial distribution while realizing the joint input/output distribution characteristics (p(X|Y ), p(Y )).

Yet, it remains a difficult engineering challenge to create rich generative models that can capture these distributions accurately.

Some recent works leveraging a generative model for robustness use a PixelCNN model BID28 to detect adversarial examples, or use Generative Adversarial Networks (GANs) to generate adversarial examples BID26 .

But, one might come across practical difficulties while implementing such methods due to the inherent training difficulty.

With Explainable Adversarial Learning, we partially address the above difficulty by modeling the noise based on the prediction loss of the posterior distribution.

First, let us assume that the noise (N) introduced with ExL spans a subspace of potential adversarial examples (N ??? A).

Based on Eqn.

1 the posterior optimization criterion with noise (N) becomes DISPLAYFORM1 The noise learning in ExL (Algorithm 1) indicates an implicit generative modeling behavior, that is constrained towards maximizing p(N|X, Y ) while increasing the likelihood of the posterior p(Y |X, N).

We believe that this partial and implicit generative modeling perspective with posterior maximization, during training, imparts an ExL model more knowledge about the data manifold, rendering it less susceptible toward adversarial attacks (See Appendix D for further intuition).

Next, we empirically demonstrate using PCA that, noise modeling indeed embraces some off-manifold data points.

PCA serves as a method to reduce a complex dataset to lower dimensions to reveal sometimes hidden, simplified structure that often underlie it.

Since the learned representations of a deep learning model lie in a high dimensional geometry of the data manifold, we opted to reduce the dimensionality of the feature space and visualize the relationship between the adversarial and clean inputs in this reduced PC subspace.

Essentially, we find the principal components (or eigen-vectors) of the activations of an intermediate layer of a trained model and project the learnt features onto the PC space.

To do this, we center the learned features about zero (F), factorize F using Singular Value Decomposition (SVD), i.e. F = U SV T and then transform the feature samples F onto the new subspace by computing FV = U S ??? F P C .

In FIG1 (a), we visualize the learnt representations of the Conv1 layer of a ResNet18 model trained on CIFAR-10 (with standard SGD) along different 2D-projections of the PC subspace in response to adversarial/clean input images.

Interestingly, we see that the model's perception of both the adversarial and clean inputs along high-rank PCs (say, PC1-PC10 that account for maximum variance in the data) is alike.

As we move toward lower-rank dimensions, the adversarial and clean image representations dissociate.

This implies that adversarial images place strong emphasis on PCs that account for little variance in the data.

While we note a similar trend with ExL ( FIG1 ), the dissociation occurs at latter PC dimensions compared to FIG1 (a).

A noteworthy observation here is that, adversarial examples lie in close vicinity of the clean inputs for both ExL/SGD scenarios ascertaining former theories of BID4 .To quantify the dissociation of the adversarial and clean projections in the PC subspace, we calculate the cosine distance ( DISPLAYFORM0 ) between them along different PC dimensions.

Here, N represents the total number of sample images used to perform PCA and F P C clean (F P C adv ) denote the transformed learnt representations corresponding to clean (adversarial) input, respectively.

The distance between the learnt representations (for the Conv1 layer of ResNet18 model from the above scenario) consistently increases for latter PCs as shown in FIG1 .

Interestingly, the cosine distance between adversarial and clean features measured for a model trained with ExL noise is significantly lesser than a standard SGD trained model.

This indicates that noise enables the model to look in the vicinity of the original data point and inculcate more adversarial data into its underlying representation.

Note, we consider projection across all former dimensions (say, PC0, PC1,...PC100) to calculate the distance at a later dimension (say, PC100) i.e., D P C 100 is calculated by taking the dot product between two 100-dimensional vectors: DISPLAYFORM1 To further understand the role of ExL noise in a model's behavior, we analyzed the variance captured in the Conv1 layer's activations of the ResNet18 model (in response to clean inputs) by different PCs, as illustrated Fig. 3 (d) .

If s i = {1, ..., M } are the singular values of the matrix S, the variance along a particular dimension P C k is defined as: DISPLAYFORM2 PCs provides a good measure of how much a particular dimension explains about the data.

We observe that ExL noise increases the explainability (or variance) along the high rank PCs, for instance, the net variance obtained from PC0-PC100 with ExL Noise (90%) is more than that of standard SGD (76%).

In fact, we observe a similar increase in variance in the leading PC dimensions for other intermediate blocks learnt activations of the ResNet18 model [See Appendix B] .

We can infer that the increase in variance along the high-rank PCs is a consequence of inclusion of more data points during the overall learning process.

Conversely, we can also interpret this as ExL noise embracing more off-manifold adversarial points into the overall data manifold that eventually determines the model's behavior.

It is worth mentioning that the variance analysis of the model's behavior in response to adversarial inputs yields nearly identical results as FIG1 DISPLAYFORM3 Interestingly, the authors in BID9 conducted PCA whitening of the raw image data for clean and adversarial inputs and demonstrated that adversarial image coefficients for later PCs have greater variance.

Our results from PC subspace analysis corroborates their experiments and further enables us to peek into the model's behavior for adversarial attacks.

Note, for all the PCA experiments above, we used 700 random images sampled from the CIFAR-10 test data, i.e. N = 700.

In addition, we used the Fast Gradient Sign Method (FGSM) method to create BB adversaries with a step size of 8/255, from a different source model (ResNet18 trained with SGD).

Given a test image X, an attack model perturbs the image to yield an adversarial image, X adv = X +???, such that a classifier f misclassifies X adv .

In this work, we consider ??? bounded adversaries studied in earlier works BID6 BID30 BID21 , wherein the perturbation ( ??? ??? ??? ) is regulated by some parameter .

Also, we study robustness against both BB/WB attacks to gauge the effectiveness of our approach.

For an exhaustive assessment, we consider the same attack methods deployed in BID30 ; BID21 : Fast Gradient Sign Method (FGSM): This single-step attack is a simple way to generate malicious perturbations in the direction of the loss gradient ??? X L(X, Y true ) as: DISPLAYFORM0 Step FGSM (R-FGSM): BID30 suggested to prepend single-step attacks with a small random step to escape the non-smooth vicinity of a data point that might degrade attacks based on single-step gradient computation.

For parameters , ?? (?? = /2), the attack is defined as: DISPLAYFORM1 This method iteratively applies FGSM k times with a step size of ?? ??? /k and projects each step perturbation to be bounded by .

Following BID30 , we use two-step iterative FGSM attacks.

Projected Gradient Descent (PGD): Similar to I-FGSM, this is a multi-step variant of FGSM: DISPLAYFORM2 .

BID21 show that this is a universal first-order adversary created by initializing the search for an adversary at a random point followed by several iterations of FGSM.

PGD attacks, till date, are one of the strongest BB/ WB adversaries.

We evaluated ExL on three datasets: MNIST, CIFAR10 and CIFAR100.

For each dataset, we report the accuracy of the models against BB/WB attacks (crafted from the test data) for 6 training scenarios: a) Standard SGD (without noise), b) ExL Noise, c) Ensemble Adversarial (EnsAdv) Training (SGD ens ), d) ExL Noise with EnsAdv Training (ExL ens ), e) PGD Adversarial (PGDAdv) Training (SGD P GD ), f) ExL Noise with PGDAdv Training (ExL P GD ).

Note, SGD ens and SGD P GD refer to the standard adversarial training employed in BID30 and BID21 , respectively.

Our results compare how the additional noise modeling improves over standard SGD in adversarial susceptibility.

Also, we integrate ExL with state-of-the-art PGD/Ensemble adversarial training techniques to analyze how noise modeling benefits them.

In case of EnsAdv training, we augmented the training dataset of the target model with adversarial examples (generated using R-FGSM), from an independently trained model, with same architecture as the target model.

In case of PGDAdv training, we augmented the training dataset of the target model with adversarial examples (generated using PGD) from the same target model.

Thus, as we see later, EnsAdv imparts robustness against BB attacks only, while, PGD makes a model robust to both BB/WB attacks.

In all experiments below, we report the WB/BB accuracy against strong adversaries created with PGD attack.

In additon, for BB, we also report the worst-case error over all small-step attacks FGSM, I-FGSM, R-FGSM, denoted as Min BB in TAB2 , 2.All networks were trained with mini-batch SGD using a batch size of 64 and momentum of 0.9 (0.5) for CIFAR (MNIST), respectively.

For CIFAR10, CIFAR100 we used additional weight decay regularization, ?? = 5e ??? 4.

Note, for noise modeling, we simply used the negative loss gradients (??? N L ??? 0) without additional optimization terms.

In general, ExL requires slightly more epochs of training to converge to similar accuracy as standard SGD, a result of the additional input noise modeling.

Also, ExL models, if not tuned with proper learning rate, have a tendency to overfit.

Hence, the learning rate for noise (?? noise ) was kept 1-2 orders of magnitude lesser than the overall network learning rate (??) throughout the training process.

All networks were implemented in PyTorch.

MNIST:

For MNIST, we consider a simple network with 2 Convolutional (C) layers with 32, 64 filters, each followed by 2??2 Max-pooling (M), and finally a Fully-Connected (FC) layer of size 1024, as the target model (ConvNet1: 32C-M-64C-M-1024FC).

We trained 6 ConvNet1 models independently corresponding to the different scenarios.

The EnsAdv (ExL ens , SGD ens ) models were trained with BB adversaries created from a separate SGD-trained ConvNet1 model using R-FGSM with = 0.1.

PGDAdv (ExL P GD , SGD P GD ) models were trained with WB adversaries created from the same target model using PGD with = 0.3, step-size = 0.01 over 40 steps.

4 .

ExL noise considerably improves the robustness of a model toward BB attacks.

An interesting observation here is that for = 0.1 (that was the perturbation size for EnsAdv training), both ExL ens /SGD ens yield nearly similar accuracy, ??? 98%.

However, for larger perturbation size = 0.2, 0.3, the network adversarially trained with ExL noise shows higher prediction capability (???> 5%) across the PGD attack methods.

We observe similar BB accuracy trend with PGDAdv methods (ExL P GD /SGD P GD ).

Columns 6-7 in TAB2 show the WB attack results.

All techniques except for the ones with PGDAdv training fail miserably against the strong WB PGD attacks.

Models trained with ExL noise, although yielding low accuracy, still perform better than SGD.

ExL P GD yields better accuracy than SGD P GD even beyond what the network is adversarially trained for ( > 0.3).

Note, for PGD attack in TAB2 , we used a step-size of 0.01 over 40/100 steps to create adversaries bounded by = 0.1/0.2/0.3.

We also evaluated the worst-case accuracy over all the BB attack methods when the source model is trained with ExL noise (not shown).

We found higher accuracies in this case, implying ExL models transfer attacks at lower rates.

As a result, in the remainder of the paper, we conduct BB attacks from models trained without noise modeling to evaluate the adversarial robustness 4 .

Clean Min BB PGD-40 PGD-100 PGD-40 PGD-100 CIFAR: For CIFAR10, we examined our approach on the ResNet18 architecture.

We used the ResNext29(2??64d) architecture BID32 with bottleneck width 64, cardinality 2 for CI-FAR100.

Similar to MNIST, we trained the target models separately corresponding to each scenario and crafted BB/WB attacks.

For EnsAdv training, we used BB adversaries created using R-FGSM ( = 8/255) from a separate SGD-trained network different from the BB source/target model.

For PGDAdv training, the target models were trained with WB adversaries created with PGD with = 8/255, step-size=2/255 over 7 steps.

Here, for PGD attacks, we use 7/20 steps of size 2/255 bounded by .

The results appear in TAB5 .

DISPLAYFORM0 For BB, we observe that ExL (81%/63.2% for CIFAR10/100) significantly boosts the robustness of a model as compared to SGD (50.3%/44.2% for CIFAR10/100).

Note, the improvement here is quite large in comparison to MNIST (that shows only 5% increase from SGD to ExL).

In fact, the accuracy obtained with ExL alone with BB attack, is almost comparable to that of an EnsAdv trained model without noise (SGD ens ).

The richness of the data manifold and feature representation space for larger models and complex datasets allows ExL to model better characteristics in the noise causing increased robustness.

As seen earlier, ExL noise (ExL ens , ExL P GD ) considerably improves the accuracy even for perturbations ( = (16, 32)/255) greater than what the network is adversarially trained for.

The increased susceptibility of SGD ens , SGD P GD for larger establishes that its capability is limited by the diversity of adversarial examples shown during training.

For WB attacks as well, ExL P GD show higher resistance.

Interestingly, while SGD, SGD ens yield infinitesimal performance (< 5%), ExL, ExL ens yield reasonably higher accuracy (> 25%) against WB attacks.

This further establishes the potential of noise modeling in enabling adversarial security.

It is worth mentioning that BB accuracy of SGD P GD , ExL P GD models in TAB2 , 2 are lower than SGD ens , ExL ens , since the former is attacked with stronger attacks crafted from models trained with PGDAdv 4 .

Attacking the former with similar adversaries as latter yields higher accuracy.

TAB5 .

FIG2 shows that variance across the leading PCs decreases as ExL P GD > SGD P GD > ExL ens > ExL > SGD ens > SGD.

Inclusion of adversarial data points with adversarial training or noise modeling informs a model more, leading to improved explainability.

We note that ExL ens and SGD P GD yield nearly similar variance ratio, although SGD P GD gives better accuracy than ExL ens for similar BB and WB attacks.

Since we are analyzing only the Conv1 layer, we get this result.

In FIG2 , we also plot the cosine distance between the adversarial (created from FGSM with specified ) and clean inputs in the PC subspace.

The distance across different scenarios along latter PCs increases as: ExL P GD < SGD P GD < ExL ens < ExL < SGD ens < SGD.

A noteworthy observation here is, PC distance follows the same order as decreasing variance and justifies the accuracy results in TAB5 .

The decreasing distance with ExL compared to SGD further signifies improved realization of the on-/off-manifold data.

Also, the fact that ExL P GD , ExL ens have lower distance for varying establishes that integrating noise modeling with adversarial training compounds adversarial robustness.

Interestingly, for both variance and PC distance, ExL has a better characteristic than SGD ens .

This proves that noise modeling enables implicit inclusion of adversarial data without direct data augmentation, as opposed to EnsAdv training (or SGD ens ) where the dataset is explicitly augmented.

This also explains the comparable BB accuracy between ExL, SGD ens in TAB5 .Adversarial Subspace Dimensionality : To further corroborate that ExL noise implicitly embraces adversarial points, we evaluated the adversarial subspace dimension using the Gradient-Aligned Adversarial Subspace (GAAS) method of BID31 .

We construct k orthogonal vectors r 1 , .., r k ??? {???1, 1} from a regular Hadamard matrix of order k ??? {2 2 , 2 3 , .., 2 7 }.

We then multiply each r i component-wise with the gradient, sign(??? X L(X, Y true )).

Hence, estimating the dimensionality reduces to finding a set of orthogonal perturbations, r i with r i ??? = in the vicinity of a data point that causes misclassification.

For each scenario of Table 2 (CIFAR10), we select 350 random test points, x, and plot the probability that we find at least k orthogonal vectors r i such that x + r i is misclassified.

FIG2 , (c) shows the results with varying for BB, WB instances.

We find that the size of the space of adversarial samples is much lower for a model trained with ExL noise than that of standard SGD.

For = 8/255, we find over 128/64 directions for ??? 25%/15% of the points in case of SGD/ExL. With EnsAdv training, the number of adversarial directions for SGD ens /ExL ens reduces to 64 that misclassifies ??? 17/15% of the points.

With PGDAdv training, the adversarial dimension significantly reduces in case of ExL P GD for both BB/WB.

As we increase the perturbation size ( = 32/255), we observe increasingly reduced number of misclassified points as well as adversarial dimensions for models trained with noise modeling.

The WB adversarial plot, in FIG2 (c), clearly shows the reduced space obtained with noise modeling with PGDAdv training (ExL P GD ) against plain PGDAdv (SGD P GD ) for = (8, 32)/255.Loss Surface Smoothening: By now, it is clear that while ExL alone can defend against BB attacks (as compared to SGD) reasonably well, it still remains vulnerable to WB attacks.

For WB defense and to further improve BB defense, we need to combine ExL noise modeling with adversarial training.

To further investigate this, we plotted the loss surface of MNIST models on examples x = x+ 1 ??g BB + 2 ??g W B in FIG3 , where g BB is the signed gradient, sign(??? X L(X, Y true ) source ), obtained from the source model (crafting the BB attacks) and g W B is the gradient obtained from the target model itself (crafting WB attacks), sign(??? X L(X, Y true ) target ).

We see that the loss surface in case of SGD is highly curved with steep slopes in the vicinity of the data point in both BB and WB direction.

The EnsAdv training, SGD ens , smoothens out the slope in the BB direction substantially, justifying their robustness against BB attacks.

Models trained with noise modeling, ExL (even without any data augmentation), yield a softer loss surface.

This is why ExL models transfer BB attacks at lower rates.

The surface in the WB direction along 2 with ExL, ExL ens still exhibits a sharper curvature (although slightly softer than SGD ens ) validating the lower accuracies against WB attacks (compared to BB attacks).

PGDAdv, on the other hand, smoothens out the loss surface substantially in both directions owing to the explicit inclusion of WB adversaries during training.

Note, ExL P GD yields a slightly softer surface than SGD P GD (not shown).

The smoothening effect of noise modeling further justifies the boosted robustness of ExL models for larger perturbations (outside -ball used during adversarial training).

It is worth mentioning that we get similar PCA/ Adversarial dimensionality/ loss surface results across all datasets.

We proposed Explainable Adversarial Learning, ExL, as a reliable method for improving adversarial robustness.

Specifically, our key findings are: 1) We show that noise modeling at the input during discriminative training improves a model's ability to generalize better for out-of-sample adversarial data (without explicit data augmentation).

2) Our PCA variance and cosine distance analysis provides a significant perspective to visualize and quantify a model's response to adversarial/clean data.

A crucial question one can ask is, How to break ExL defense?

The recent work BID0 shows that many defense methods cause 'gradient masking' that eventually fail.

We reiterate that, ExL alone does not give a strong BB/WB defense.

However, the smoothening effect of noise modeling on the loss FIG3 suggests that noise modeling decreases the magnitude of the gradient masking effect.

ExL does not change the classification model that makes it easy to be scaled to larger datasets while integrating with other adversarial defense techniques.

Coupled with other defense, ExL performs remarkably (even for larger values).

We combine ExL with EnsAdv & PGDAdv, which do not cause obfuscated gradients and hence can withstand strong attacks, however, upto a certain point.

For WB perturbations much greater than the training value, ExL+PGDAdv also breaks.

In fact, for adaptive BB adversaries BID30 or adversaries that query the model to yield full prediction confidence (not just the label), ExL+EnsAdv will be vulnerable.

Note, advantage with ExL is, being independent of the attack/defense method, ExL can be potentially combined with stronger attacks developed in future, to create stronger defenses.

While variance and principal subspace analysis help us understand a model's behavior, we cannot fully describe the structure of the manifold learnt by the linear subspace view.

However, PCA does provide a basic intuition about the generalization capability of complex image models.

In fact, our PC results establish the superiority of adversarial training methods (SGD ens ; SGD P GD : BID30 ; BID21 and can be used as a valid metric to gauge adversarial susceptibility in future proposals.

Finally, as our likelihood theory (Eqn.1) indicates, better noise modeling techniques with improved gradient penalties can further improve robustness and requires further investigation.

Also, performing noise modeling at intermediate layers to improve variance/explainability, and hence robustness, are other future work directions.

A APPENDIX A: JUSTIFICATION OF X + N VS X ?? N AND USE OF ???L N ??? 0 FOR NOISE MODELING FIG0 : For MNIST dataset, we show the noise template learnt when we use multiplicative/additive noise (N ) for Explainable Learning.

The final noise-integrated image (for a sample digit '9') that is fed to the network before and after training is also shown.

Additive noise disrupts the image drastically.

Multiplicative noise, on the other hand, enhances the relevant pixels while eliminating the background.

Accuracy corrsponding to each scenario is also shown and compared against standard SGD training scenario (without any noise).

Here, we train a simple convolutional architecture (ConvNet: 10C-M-20C-M-320FC) of 2 Convolutional (C) layers with 10, 20 filters, each followed by 2??2 Max-pooling (M) and a Fully-Connected (FC) layer of size 320.

We use mini-batch SGD with momentum of 0.5, learning rate (??=0.1) decayed by 0.1 every 15 epochs and batchsize 64 to learn the network parameters.

We trained 3 ConvNet models independently corresponding to each scenario for 30 epochs.

For the ExL scenarios, we conduct noise modelling with only negative loss gradients (???LN ??? 0) with noise learning rate, ??noise = 0.001, throughout the training process.

Note, the noise image shown is the average across all 64 noise templates.

Figure A2 : Here, we showcase the noise learnt by a simple convolutional network (ConvNet: 10C-M-20C-M-320FC), learning the CIFAR10 data with ExL (multiplicative noise) under different gradient update conditions.

As with MNIST ( FIG0 , we observe that the noise learnt enhances the region of interest while deemphasizing the background pixels.

Note, the noise in this case has RGB components as a result of which we see some prominent color blobs in the noise template after training.

The performance table shows that using only negative gradients (i.e. ???LN ??? 0) during backpropagation for noise modelling yields minimal loss in accuracy as compared to a standard SGD trained model.

We use mini-batch SGD with momentum of 0.9, weight decay 5e-4, learning rate (??=0.01) decayed by 0.2 every 10 epochs and batch-size 64 to learn the network parameters.

We trained 4 ConvNet models independently corresponding to each scenario for 30 epochs.

For the ExL scenarios, we conduct noise modelling by backpropagating the corresponding gradient with noise learning rate (??noise = 0.001) throughout the training process.

Note, the noise image shown is the average across all 64 noise templates.

We observe that ExL noise increases the explainability (or variance) along the high rank PCs.

Also, as we go deeper into the network, the absolute difference of the variance values between SGD/ExL decreases.

This is expected as the contribution of input noise on the overall representations decreases as we go deeper into the network.

Moreover, there is a generic-to-specific transition in the hierarchy of learnt features of a deep neural network.

Thus, the linear PC subspace analysis to quantify a model's knowledge of the data manifold is more applicable in the earlier layers, since they learn more general input-related characteristics.

Nonetheless, we see that ExL model yields widened explainability than SGD for each intermediate layer except the final Block4 that feeds into the output layer.

We use mini-batch SGD with momentum of 0.9, weight decay 5e-4, learning rate (??=0.1) decayed by 0.1 every 30 epochs and batch-size 64 to learn the network parameters.

We trained 2 ResNet-18 models independently corresponding to each scenario for 60 epochs.

For noise modelling, we use ??noise = 0.001 decayed by 0.1 every 30 epochs.

Note, we used a sample set of 700 test images to conduct the PCA.

FIG2 : Here, we show the variance captured in the leading Principal Component (PC) dimensions for the Conv1 and Block1 learnt activations in response to both clean and adversarial inputs for ResNet-18 models correponding to the scenarios discussed in FIG1 .

The model's variance for both clean and adversarial inputs are exactly same in case of ExL/SGD for Conv1 layers.

For Block1, the adversarial input variance is slighlty lower in case of SGD than that of clean input.

With ExL, the variance is still the same for Block1.

This indicates that PC variance statistics cannot differentiate between a model's knowledge of on-/off-manifold data.

It only tells us whether a model's underlying representation has acquired more knowledge about the data manifold.

To analyze a model's understanding of adversarial data, we need to look into the relationship between the clean and adversarial projection onto the PC subspace and measure the cosine distance.

Note, we used the Fast Gradient Sign Method (FGSM) method BID6 to create BB adversaries with a step size of 8/255, from another independently trained ResNet-18 model (source) with standard SGD.

The source attack model has the same hyperparameters as the SGD model in FIG1 and is trained for 40 epochs.

Before Training After Training FIG3 : Here, we show the noise templates learnt with noise modeling corresponding to different training scenarios of TAB2 , 2 in main paper: ExL (only noise modeling), ExL PGD (noise modeling with PGDAdv training ExLP GD ), ExL ens (noise modeling with EnsAdv training ExLens) for MNIST and CIFAR10 data.

A sample image (X ?? N ) before and after training with different scenarios is shown.

The fact that every training technique yields different noise template shows that noise influences the overall optimization.

Column 1 shows the noise template and correponding image (X ?? N ) before training, Coulmns 2-4 show the templates after training.

Note, noise shown is the mean across 64 templates.

The Pytorch implementation of ResNet-18 architecture for CIFAR10 and ResNext-29 architecture for CIFAR100 were taken from (Github).

For CIFAR10/CIFAR100, we use mini-batch SGD with momentum of 0.9, weight decay 5e-4 and batch size 64 for training the weight parameters of the models.

A detailed description of the learning rate and epochs for ResNet18 model (corresponding to Table 2 in main paper) is shown in TAB2 .

Similarly, TAB5 shows the parameters for ResNext-29 model.

The hyperparmeters corresponding to each scenario (of TAB2 , A2) are shown in Rows1-6 under Target type.

The hyperparameters for the source model used to attack the target models for BB scenarios is shown in Row 7/8 under Source type.

We use BB attacks from the SGD trained source model to attack SGD, ExL, ExL ens , P GD ens .

We use BB attacks from a model trained with PGD adversarial training ( = 8/255, step-size=2/255 over 7 steps) to craft strong BB attacks on SGD P GD , ExL P GD .

The model used to generate black box adversaries to augment the training dataset of the SGD ens , ExL ens target models is shown in Row 9 under EnsAdv type.

How to conduct Ensemble Adversarial Training?

Furthermore, in all our experiments, for EnsAdv training (SGD ens ), we use a slightly different approach than BID15 .

Instead of using a weighted loss function that controls the relative weight of adversarial/clean examples in the overall loss computation, we use a different learning rate ?? adv /?? (?? adv < ??) when training with adversarial/clean inputs, respectively, to learn the network parameters.

Accordingly, while performing adversarial training with explainable learning (ExL ens ), the noise modeling learning rate in addition to overall learning rate, ?? adv /??, for adversarial/clean inputs is also different, ?? noiseadv /?? noise (?? noiseadv < ?? noise ).How to conduct PGD Adversarial Training?

For PGD adversarial training (SGD P GD ), we used the techniques suggested in BID12 .

BID12 propose that training on a mixture of clean and adversarial examples (generated using PGD attack), instead of literally solving the min-max problem described by BID21 yields better performance.

In fact, this helps maintain good accuracy on both clean and adversarial examples.

Like EnsAdv training, here as well, we use a different learning rate ?? adv /?? (?? adv < ??) when training with adversarial/clean inputs, respectively, to learn the network parameters.

Accordingly, while performing PGD adversarial training with explainable learning (ExL P GD ), the noise modeling learning rate in addition to overall learning rate, ?? adv /??, for adversarial/clean inputs is also different, ?? noiseadv /?? noise (?? noiseadv < ?? noise ) Note, the adversarial inputs for EnsAdv training of a target model are created using BB adversaries generated by R-FGSM from a source (shown in Row 9 of TAB2 , A2), while PGDAdv training uses WB adversaries created with PGD attack from the same target model.

We also show the test accuracy (on clean data) for each model in TAB2 ,A2 for reference.

Note, the learning rate in each case decays by a factor of 0.1 every 20/30 epochs (Column 5 in TAB2 , A2).

For MNIST, we use 2 different architectures as source/ target models.

ConvNet1: 32C-M-64C-M-1024FC is the model used as target.

ConvNet2: 10C-M-20C-M-320FC is the model used as source.

Here, we use mini-batch SGD with momentum of 0.5, batch size 64, for training the weight parameters.

TAB9 shows the hyperparameters used to train the models in TAB2 of main paper.

The notations here are similar to that of TAB2 .

Note, the source model trained with PGDAdv training to craft BB attacks on ExL P GD , SGD P GD was trained with = 0.3, step-size=0.01 over 40 steps.

We use mini-batch SGD with momentum of 0.9, weight decay 5e-4 and batch size 64 for training the weight parameters of the models in Table A4 .

Intuitively, we can justify adversarial robustness inherited with noise modeling in two ways: First, by integrating noise during training, we allow a model to explore multiple directions within the vicinity of the data point (thereby incorporating more off-manifold data) and hence inculcate that knowledge in its underlying behavior.

Second, we note that noise learnt with ExL inherits the input data characteristics (i.e. N ??? X) and that the noise-modeling direction (??? N L) is aligned with the loss gradient, ??? X L (that is also used to calculate the adversarial inputs, X adv = X + sign(??? X L)).

This ensures that the exploration direction coincides with certain adversarial directions improving the model's generalization capability in such spaces.

Note, for fully guaranteed adversarial robustness as per Eqn.

1 in main paper, the joint input/output distribution (p(X|Y ), p(Y )) has to be realized in addition to the noise modeling and N should span the entire space of adversarial/off-manifold data.

<|TLDR|>

@highlight

Noise modeling at the input during discriminative training improves adversarial robustness. Propose PCA based evaluation metric for adversarial robustness

@highlight

This paper proposes, ExL, an adversarial training method using multiplicate noise that is shown to be helpful in defending against blackbox attacks on three datasets.

@highlight

This paper includes multiplicative noise N in training data to achieve adversarial robustness, when training on both model parameters theta and on the noise itself.