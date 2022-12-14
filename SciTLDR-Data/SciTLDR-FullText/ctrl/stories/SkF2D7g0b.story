Existing black-box attacks on deep neural networks (DNNs) so far have largely focused on transferability, where an adversarial instance generated for a locally trained model can “transfer” to attack other learning models.

In this paper, we propose novel Gradient Estimation black-box attacks for adversaries with query access to the target model’s class probabilities, which do not rely on transferability.

We also propose strategies to decouple the number of queries required to generate each adversarial sample from the dimensionality of the input.

An iterative variant of our attack achieves close to 100% adversarial success rates for both targeted and untargeted attacks on DNNs.

We carry out extensive experiments for a thorough comparative evaluation of black-box attacks and show that the proposed Gradient Estimation attacks outperform all transferability based black-box attacks we tested on both MNIST and CIFAR-10 datasets, achieving adversarial success rates similar to well known, state-of-the-art white-box attacks.

We also apply the Gradient Estimation attacks successfully against a real-world content moderation classiﬁer hosted by Clarifai.

Furthermore, we evaluate black-box attacks against state-of-the-art defenses.

We show that the Gradient Estimation attacks are very effective even against these defenses.

The ubiquity of machine learning provides adversaries with both opportunities and incentives to develop strategic approaches to fool learning systems and achieve their malicious goals.

Many attack strategies devised so far to generate adversarial examples to fool learning systems have been in the white-box setting, where adversaries are assumed to have access to the learning model BID18 ; BID0 ; BID1 ; BID6 ).

However, in many realistic settings, adversaries may only have black-box access to the model, i.e. they have no knowledge about the details of the learning system such as its parameters, but they may have query access to the model's predictions on input samples, including class probabilities.

For example, we find this to be the case in some popular commercial AI offerings, such as those from IBM, Google and Clarifai.

With access to query outputs such as class probabilities, the training loss of the target model can be found, but without access to the entire model, the adversary cannot access the gradients required to carry out white-box attacks.

Most existing black-box attacks on DNNs have focused on transferability based attacks BID12 ; BID7 ; BID13 ), where adversarial examples crafted for a local surrogate model can be used to attack the target model to which the adversary has no direct access.

The exploration of other black-box attack strategies is thus somewhat lacking so far in the literature.

In this paper, we design powerful new black-box attacks using limited query access to learning systems which achieve adversarial success rates close to that of white-box attacks.

These black-box attacks help us understand the extent of the threat posed to deployed systems by adversarial samples.

The code to reproduce our results can be found at https://github.com/ anonymous 1 .New black-box attacks.

We propose novel Gradient Estimation attacks on DNNs, where the adversary is only assumed to have query access to the target model.

These attacks do not need any access to a representative dataset or any knowledge of the target model architecture.

In the Gradient Estimation attacks, the adversary adds perturbations proportional to the estimated gradient, instead of the true gradient as in white-box attacks BID0 ; Kurakin et al. (2016) ).

Since the direct Gradient Estimation attack requires a number of queries on the order of the dimension of the input, we explore strategies for reducing the number of queries to the target model.

We also experimented with Simultaneous Perturbation Stochastic Approximation (SPSA) and Particle Swarm Optimization (PSO) as alternative methods to carry out query-based black-box attacks but found Gradient Estimation to work the best.

Query-reduction strategies We propose two strategies: random feature grouping and principal component analysis (PCA) based query reduction.

In our experiments with the Gradient Estimation attacks on state-of-the-art models on MNIST (784 dimensions) and CIFAR-10 (3072 dimensions) datasets, we find that they match white-box attack performance, achieving attack success rates up to 90% for single-step attacks in the untargeted case and up to 100% for iterative attacks in both targeted and untargeted cases.

We achieve this performance with just 200 to 800 queries per sample for single-step attacks and around 8,000 queries for iterative attacks.

This is much fewer than the closest related attack by .

While they achieve similar success rates as our attack, the running time of their attack is up to 160× longer for each adversarial sample (see Appendix I.6).A further advantage of the Gradient Estimation attack is that it does not require the adversary to train a local model, which could be an expensive and complex process for real-world datasets, in addition to the fact that training such a local model may require even more queries based on the training data.

Attacking real-world systems.

To demonstrate the effectiveness of our Gradient Estimation attacks in the real world, we also carry out a practical black-box attack using these methods against the Not Safe For Work (NSFW) classification and Content Moderation models developed by Clarifai, which we choose due to their socially relevant application.

These models have begun to be deployed for real-world moderation BID4 , which makes such black-box attacks especially pernicious.

We carry out these attacks with no knowledge of the training set.

We have demonstrated successful attacks ( FIG0 ) with just around 200 queries per image, taking around a minute per image.

In FIG0 , the target model classifies the adversarial image as 'safe' with high confidence, in spite of the content that had to be moderated still being clearly visible.

We note here that due to the nature of the images we experiment with, we only show one example here, as the others may be offensive to readers.

The full set of images is hosted anonymously at https://www.dropbox.com/s/ xsu31tjr0yq7rj7/clarifai-examples.zip?dl=0.Comparative evaluation of black-box attacks.

We carry out a thorough empirical comparison of various black-box attacks (given in TAB8 ) on both MNIST and CIFAR-10 datasets.

We study attacks that require zero queries to the learning model, including the addition of perturbations that are either random or proportional to the difference of means of the original and targeted classes, as well as various transferability based black-box attacks.

We show that the proposed Gradient Estimation attacks outperform other black-box attacks in terms of attack success rate and achieve results comparable with white-box attacks.

In addition, we also evaluate the effectiveness of these attacks on DNNs made more robust using adversarial training BID0 BID18 and its recent variants including ensemble adversarial training BID21 and iterative adversarial training BID9 .

We find that although standard and ensemble adversarial training confer some robustness against single-step attacks, they are vulnerable to iterative Gradient Estimation attacks, with adversar-ial success rates in excess of 70% for both targeted and untargeted attacks.

We find that our methods outperform other black-box attacks and achieve performance comparable to white-box attacks.

Related Work.

Existing black-box attacks that do not use a local model were first proposed for convex inducing two-class classifiers by BID11 .

For malware data, Xu et al. (2016) use genetic algorithms to craft adversarial samples, while Dang et al. (2017) use hill climbing algorithms.

These methods are prohibitively expensive for non-categorical and high-dimensional data such as images.

BID13 proposed using queries to a target model to train a local surrogate model, which was then used to to generate adversarial samples.

This attack relies on transferability.

To the best of our knowledge, the only previous literature on query-based black-box attacks in the deep learning setting is independent work by BID10 and .

BID10 propose a greedy local search to generate adversarial samples by perturbing randomly chosen pixels and using those which have a large impact on the output probabilities.

Their method uses 500 queries per iteration, and the greedy local search is run for around 150 iterations for each image, resulting in a total of 75,000 queries per image, which is much higher than any of our attacks.

Further, we find that our methods achieve higher targeted and untargeted attack success rates on both MNIST and CIFAR-10 as compared to their method.

propose a black-box attack method named ZOO, which also uses the method of finite differences to estimate the derivative of a function.

However, while we propose attacks that compute an adversarial perturbation, approximating FGSM and iterative FGS; ZOO approximates the Adam optimizer, while trying to perform coordinate descent on the loss function proposed by BID1 .

Neither of these works demonstrates the effectiveness of their attacks on real-world systems or on state-of-the-art defenses.

In this section, we will first introduce the notation we use throughout the paper and then describe the evaluation setup and metrics used in the remainder of the paper.

A classifier f (·; θ) : X → Y is a function mapping from the domain X to the set of classification outputs Y. (Y = {0, 1} in the case of binary classification, i.e. Y is the set of class labels.)

The number of possible classification outputs is then |Y|.

θ is the set of parameters associated with a classifier.

Throughout, the target classifier is denoted as f (·; θ), but the dependence on θ is dropped if it is clear from the context.

H denotes the constraint set which an adversarial sample must satisfy.

f (x, y) is used to represent the loss function for the classifier f with respect to inputs x ∈ X and their true labels y ∈ Y.Since the black-box attacks we analyze focus on neural networks in particular, we also define some notation specifically for neural networks.

The outputs of the penultimate layer of a neural network f , representing the output of the network computed sequentially over all preceding layers, are known as the logits.

We represent the logits as a vector φ f (x) ∈ R |Y| .

The final layer of a neural network f used for classification is usually a softmax layer represented as a vector of probabilities DISPLAYFORM0 .

The empirical evaluation carried out in Section 3 is on state-of-the-art neural networks on the MNIST (LeCun & Cortes, 1998) and CIFAR-10 (Krizhevsky & Hinton, 2009 ) datasets.

The details of the datasets are given in Appendix C.1, and the architecture and training details for all models are given in Appendix C.2.

Only results for untargeted attacks are given in the main body of the paper.

All results for targeted attacks are contained in Appendix E.

We use two different loss functions in our evaluation, the standard cross-entropy loss (abbreviated as xent) and the logit-based loss (ref.

Section 3.1.2, abbreviated as logit).

In all of these attacks, the adversary's perturbation is constrained using the L ∞ distance.

The details of baseline black-box attacks and results can be found in Appendix A.1.1.

Similarly, detailed descriptions and results for transferability-based attacks are in Appendix A.2.

The full set of attacks that was evaluated is given in TAB8 in Appendix G, which also provides a taxonomy for black-box attacks.

MNIST.

Each pixel of the MNIST image data is scaled to [0, 1] .

We trained four different models on the MNIST dataset, denoted Models A to D, which are used by BID21 and represent a good variety of architectures.

For the attacks constrained with the L ∞ distance, we vary the adversary's perturbation budget from 0 to 0.4, since at a perturbation budget of 0.5, any image can be made solid gray.

CIFAR-10.

Each pixel of the CIFAR-10 image data is in [0, 255] .

We choose three model architectures for this dataset, which we denote as Resnet-32, Resnet-28-10 (ResNet variants (He et al., 2016; Zagoruyko & Komodakis, 2016) ), and Std.-CNN (a standard CNN 2 from Tensorflow BID0 ).

For the attacks constrained with the L ∞ distance, we vary the adversary's perturbation budget from 0 to 28.

Throughout the paper, we use standard metrics to characterize the effectiveness of various attack strategies.

For MNIST, all metrics for single-step attacks are computed with respect to the test set consisting of 10,000 samples, while metrics for iterative attacks are computed with respect to the first 1,000 samples from the test set.

For the CIFAR-10 data, we choose 1,000 random samples from the test set for single-step attacks and a 100 random samples for iterative attacks.

In our evaluations of targeted attacks, we choose target T for each sample uniformly at random from the set of classification outputs, except the true class y of that sample.

Attack success rate.

The main metric, the attack success rate, is the fraction of samples that meets the adversary's goal: f (x adv ) = y for untargeted attacks and f (x adv ) = T for targeted attacks with target T BID18 BID21 .

Alternative evaluation metrics are discussed in Appendix C.3.Average distortion.

We also evaluate the average distortion for adversarial examples using average L 2 distance between the benign samples and the adversarial ones as suggested by Gu & Rigazio (2014) DISPLAYFORM0 where N is the number of samples.

This metric allows us to compare the average distortion for attacks which achieve similar attack success rates, and therefore infer which one is stealthier.

Number of queries.

Query based black-box attacks make queries to the target model, and this metric may affect the cost of mounting the attack.

This is an important consideration when attacking real-world systems which have costs associated with the number of queries made.

Deployed learning systems often provide feedback for input samples provided by the user.

Given query feedback, different adaptive, query-based algorithms can be applied by adversaries to understand the system and iteratively generate effective adversarial examples to attack it.

Formal definitions of query-based attacks are in Appendix D. We initially explored a number of methods of using query feedback to carry out black-box attacks including Particle Swarm Optimization (Kennedy, 2011) and Simultaneous Perturbation Stochastic Approximation BID16 .

However, these methods were not effective at finding adversarial examples for reasons detailed in Section 3.4, which also contains the results obtained.

Given the fact that many white-box attacks for generating adversarial examples are based on gradient information, we then tried directly estimating the gradient to carry out black-box attacks, and found it to be very effective in a range of conditions.

In other words, the adversary can approximate white-box Single-step and Iterative FGSM attacks BID0 Kurakin et al., 2016) using estimates of the losses that are needed to carry out those attacks.

We first propose a Gradient Estimation black-box attack based on the method of finite differences BID17 .

The drawback of a naive implementation of the finite difference method, however, is that it requires O(d) queries per input, where d is the dimension of the input.

This leads us to explore methods such as random grouping of features and feature combination using components obtained from Principal Component Analysis (PCA) to reduce the number of queries.

Threat model and justification.

We assume that the adversary can obtain the vector of output probabilities for any input x. The set of queries the adversary can make is then Q f = {p f (x), ∀x}. Note that an adversary with access to the softmax probabilities will be able to recover the logits up to an additive constant, by taking the logarithm of the softmax probabilities.

For untargeted attacks, the adversary only needs access to the output probabilities for the two most likely classes.

A compelling reason for assuming this threat model for the adversary is that many existing cloudbased ML services allow users to query trained models (Watson Visual Recognition, Clarifai, Google Vision API).

The results of these queries are confidence scores which can be used to carry out Gradient Estimation attacks.

These trained models are often deployed by the clients of these ML as a service (MLaaS) providers BID4 ).

Thus, an adversary can pose as a user for a MLaaS provider and create adversarial examples using our attack, which can then be used against any client of that provider.

In this section, we focus on the method of finite differences to carry out Gradient Estimation based attacks.

All the analysis and results are presented for untargeted attacks, but can be easily extended to targeted attacks (Appendix E).

Let the function whose gradient is being estimated be g(x).

The input to the function is a d-dimensional vector x, whose elements are represented as x i , where DISPLAYFORM0 The canonical basis vectors are represented as e i , where e i is 1 only in the i th component and 0 everywhere else.

Then, a two-sided estimation of the gradient of g with respect to x is given by DISPLAYFORM1 . . .

DISPLAYFORM2 δ is a free parameter that controls the accuracy of the estimation.

A one-sided approximation can also be used, but will be less accurate (Wright & Nocedal, 1999) .

If the gradient of the function g exists, then lim δ→0 FD x (g(x), δ) = ∇ x g(x).

The finite difference method is useful for a black-box adversary aiming to approximate a gradient based attack, since the gradient can be directly estimated with access to only the function values.

In the untargeted FGS method, the gradient is usually taken with respect to the cross-entropy loss between the true label of the input and the softmax probability vector.

The cross-entropy loss of a network f at an input DISPLAYFORM0 , where y is the index of the original class of the input.

The gradient of f (x, y) is DISPLAYFORM1 An adversary with query access to the softmax probabilities then just has to estimate the gradient of p f y (x) and plug it into Eq. 2 to get the estimated gradient of the loss.

The adversarial sample thus generated is DISPLAYFORM2 This method of generating adversarial samples is denoted as FD-xent.

We also use a loss function based on logits which was found to work well for white-box attacks by BID1 .

The loss function is given by DISPLAYFORM0 where y represents the ground truth label for the benign sample x and φ(·) are the logits.

κ is a confidence parameter that can be adjusted to control the strength of the adversarial perturbation.

If the confidence parameter κ is set to 0, the logit loss is max(φ(x + δ)

y − max{φ(x + δ) i : i = y}, 0).

For an input that is correctly classified, the first term is always greater than 0, and for an incorrectly classified input, an untargeted attack is not meaningful to carry out.

Thus, the loss term reduces to φ(x + δ) y − max{φ(x + δ) i : i = y} for relevant inputs.

An adversary can compute the logit values up to an additive constant by taking the logarithm of the softmax probabilities, which are assumed to be available in this threat model.

Since the loss function is equal to the difference of logits, the additive constant is canceled out.

Then, the finite differences method can be used to estimate the difference between the logit values for the original class y, and the second most likely class y , i.e., the one given by y = argmax i =y φ(x) i .

The untargeted adversarial sample generated for this loss in the white-box case is DISPLAYFORM1 Similarly, in the case of a black-box adversary with query-access to the softmax probabilities, the adversarial sample is DISPLAYFORM2 This attack is denoted as FD-logit.

Table 1 : Untargeted black-box attacks: Each entry has the attack success rate for the attack method given in that column on the model in each row.

The number in parentheses for each entry is ∆(X, X adv ), the average distortion over all samples used in the attack.

In each row, the entry in bold represents the black-box attack with the best performance on that model.

Gradient Estimation using Finite Differences is our method, which has performance matching white-box attacks.

Above:

The iterative variant of the gradient based attack described in Section A.1.2 is a powerful attack that often achieves much higher attack success rates in the white-box setting than the simple single-step gradient based attacks.

Thus, it stands to reason that a version of the iterative attack with estimated gradients will also perform better than the single-step attacks described until now.

An iterative attack with t + 1 iterations using the cross-entropy loss is: DISPLAYFORM0 where α is the step size and H is the constraint set for the adversarial sample.

This attack is denoted as IFD-xent.

If the logit loss is used instead, it is denoted as IFD-logit.

In this section, we summarize the results obtained using Gradient Estimation attacks with Finite Differences and describe the parameter choices made.

The y-axis for both figures gives the variation in adversarial success as is increased.

The most successful black-box attack strategy in both cases is the Gradient Estimation attack using Finite Differences with the logit loss (FD-logit), which coincides almost exactly with the white-box FGS attack with the logit loss (WB FGS-logit).

Also, the Gradient Estimation attack with query reduction using PCA (GE-QR (PCA-k, logit)) performs well for both datasets as well.

FD-logit and IFD-logit match white-box attack adversarial success rates: The Gradient Estimation attack with Finite Differences (FD-logit) is the most successful untargeted single-step black-box attack for MNIST and CIFAR-10 models.

It significantly outperforms transferability-based attacks (Table 1 ) and closely tracks white-box FGS with a logit loss (WB FGS-logit) on MNIST and CIFAR-10 ( FIG2 ).

For adversarial samples generated iteratively, the Iterative Gradient Estimation attack with Finite Differences (IFD-logit) achieves 100% adversarial success rate across all models on both datasets (Table 1) .

We used 0.3 for the value of for the MNIST dataset and 8 for the CIFAR-10 dataset.

The average distortion for both FD-logit and IFD-logit closely matches their white-box counterparts, FGS-logit and IFGS-logit as given in Table 8 .FD-T and IFD-T achieve the highest adversarial success rates in the targeted setting: For targeted black-box attacks, IFD-xent-T achieves 100% adversarial success rates on almost all models as shown by the results in Table 6 .

While FD-xent-T only achieves about 30% adversarial success rates, this matches the performance of single-step white-box attacks such as FGS-xent-T and FGS-logit-T ( TAB11 ).

The average distortion for samples generated using gradient estimation methods is similar with that of white-box attacks.

Parameter choices: We use δ = 1.0 for FD-xent and IFD-xent for both datasets, while using δ = 0.01 for FD-logit and IFD-logit.

We find that a larger value of δ is needed for xent loss based attacks to work.

The reason for this is that the probability values used in the xent loss are not as sensitive to changes as in the logit loss, and thus the gradient cannot be estimated since the function value does not change at all when a single pixel is perturbed.

For the Iterative Gradient Estimation attacks using Finite Differences, we use α = 0.01 and t = 40 for the MNIST results and α = 1.0 and t = 10 for CIFAR-10 throughout.

The same parameters are used for the white-box Iterative FGS attack results given in Appendix I.1.

This translates to 62720 queries for MNIST (40 steps of iteration) and 61440 queries (10 steps of iteration) for CIFAR-10 per sample.

We find these choices work well, and keep the running time of the Gradient Estimation attacks at a manageable level.

However, we find that we can achieve similar adversarial success rates with much fewer queries using query reduction methods which we describe in the next section.

The major drawback of the approximation based black-box attacks is that the number of queries needed per adversarial sample is large.

For an input with dimension d, the number of queries will be exactly 2d for a two-sided approximation.

This may be too large when the input is high-dimensional.

So we examine two techniques in order to reduce the number of queries the adversary has to make.

Both techniques involve estimating the gradient for groups of features, instead of estimating it one feature at a time.

The justification for the use of feature grouping comes from the relation between gradients and directional derivatives (Hildebrand, 1962) for differentiable functions.

The directional derivative of a function g is defined as DISPLAYFORM0 .

It is a generalization of a partial derivative.

For differentiable functions, ∇ v g(x) = ∇ x g(x) · v, which implies that the directional derivative is just the projection of the gradient along the direction v. Thus, estimating the gradient by grouping features is equivalent to estimating an approximation of the gradient constructed by projecting it along appropriately chosen directions.

The estimated gradient∇ x g(x) of any function g can be computed using the techniques below, and then plugged in to Equations 3 and 5 instead of the finite difference term to create an adversarial sample.

Next, we introduce the techniques applied to group the features for estimation.

Detailed algorithms for these techniques are given in Appendix F.

The simplest way to group features is to choose, without replacement, a random set of features.

The gradient can then be simultaneously estimated for all these features.

If the size of the set chosen is k, then the number of queries the adversary has to make is d k .

When k = 1, this reduces to the case where the partial derivative with respect to every feature is found, as in Section 3.1.

In each iteration of Algorithm 1, there is a set of indices S according to which v is determined, with v i = 1 if and only if i ∈ S. Thus, the directional derivative being estimated is i∈S ∂g(x) ∂xi , which is an average of partial derivatives.

Thus, the quantity being estimated is not the gradient itself, but an index-wise averaged version of it.

A more principled way to reduce the number of queries the adversary has to make to estimate the gradient is to compute directional derivatives along the principal components as determined by principal component analysis (PCA) BID15 , which requires the adversary to have access to a set of data which is represetative of the training data.

A more detailed description of PCA and the Gradient Estimation attack using PCA components for query reduction is given in Appendix F.2.

In Algorithm 2, U is the d × d matrix whose columns are the principal components u i , where DISPLAYFORM0 The quantity being estimated in Algorithm 2 in the Appendix is an approximation of the gradient in the PCA basis: DISPLAYFORM1 where the term on the left represents an approximation of the true gradient by the sum of its projection along the top k principal components.

In Algorithm 2, the weights of the representation in the PCA basis are approximated using the approximate directional derivatives along the principal components.

Performing an iterative attack with the gradient estimated using the finite difference method (Equation 1) could be expensive for an adversary, needing 2td queries to the target model, for t iterations with the two-sided finite difference estimation of the gradient.

To lower the number of queries needed, the adversary can use either of the query reduction techniques described above to reduce the number of queries to 2tk ( k < d).

These attacks using the cross-entropy loss are denoted as IGE-QR (RG-k, xent) for the random grouping technique and IGE-QR (PCA-k, xent) for the PCA-based technique.

In this section, we summarize the results obtained using Gradient Estimation attacks with query reduction.

Gradient estimation with query reduction maintains high attack success rates:

For both datasets, the Gradient Estimation attack with PCA based query reduction (GE-QR (PCA-k, logit)) is effective, with performance close to that of FD-logit with k = 100 for MNIST ( FIG2 ) and k = 400 for CIFAR-10 ( FIG2 ).

The Iterative Gradient Estimation attacks with both Random Grouping and PCA based query reduction (IGE-QR (RG-k, logit) and IGE-QR (PCA-k, logit)) achieve close to 100% success rates for untargeted attacks and above 80% for targeted attacks on Model A on MNIST Table 2 : Comparison of untargeted query-based black-box attack methods.

All results are for attacks using the first 1000 samples from the MNIST dataset on Model A and with an L ∞ constraint of 0.3.

The logit loss is used for all methods expect PSO, which uses the class probabilities.and Resnet-32 on CIFAR-10 ( FIG3 ).

FIG3 clearly shows the effectiveness of the gradient estimation attack across models, datasets, and adversarial goals.

While random grouping is not as effective as the PCA based method for Single-step attacks, it is as effective for iterative attacks.

Thus, powerful black-box attacks can be carried out purely using query access.

We experimented with Particle Swarm Optimization (PSO), 3 a commonly used evolutionary optimization strategy, to construct adversarial samples as was done by BID14 , but found it to be prohibitively slow for a large dataset, and it was unable to achieve high adversarial success rates even on the MNIST dataset.

We also tried to use the Simultaneous Perturbation Stochastic Approximation (SPSA) method, which is similar to the method of Finite Differences, but it estimates the gradient of the loss along a random direction r at each step, instead of along the canonical basis vectors.

While each step of SPSA only requires 2 queries to the target model, a large number of steps are nevertheless required to generate adversarial samples.

A single step of SPSA does not reliably produce adversarial samples.

The two main disadvantages of this method are that i) the convergence of SPSA is much more sensitive in practice to the choice of both δ (gradient estimation step size) and α (loss minimization step size), and ii) even with the same number of queries as the Gradient Estimation attacks, the attack success rate is lower even though the distortion is higher.

A comparative evaluation of all the query-based black-box attacks we experimented with for the MNIST dataset is given in Table 2 .

The PSO based attack uses class probabilities to define the loss function, as it was found to work better than the logit loss in our experiments.

The attack that achieves the best trade-off between speed and attack success is IGE-QR (RG-k, logit).Detailed evaluation results are contained in Appendix I. In particular, discussions of the results on baseline attacks (Appendix I.2), effect of dimension on query reduced Gradient Estimation attacks (Appendix I.4), Single-step attacks on defenses (Appendix I.5), and the efficiency of Gradient Estimation attacks (Appendix I.6) are provided.

Sample adversarial examples are shown in Appendix H.

In this section, we evaluate black-box attacks against different defenses based on adversarial training and its variants.

Details about the adversarially trained models can be found in Appendix B.

We focus on adversarial training based defenses as they aim to directly improve the robustness of DNNs, and are among the most effective defenses demonstrated so far in the literature.

We also conduct real-world attacks on models deployed by Clarifai, a MlaaS provider.

In the discussion of our results, we focus on the attack success rate obtained by Iterative Gradient Estimation attacks, since they perform much better than any single-step black-box attack.

Nevertheless, in Figure 6 and Appendix I.5, we show that with the addition of an initial random perturbation to overcome "gradient masking" BID21 , the Gradient Estimation attack with Finite Differences is the most effective single-step black-box attack on adversarially trained models on MNIST.

We train variants of Model A with the 3 adversarial training strategies described in Appendix B using adversarial samples based on an L ∞ constraint of 0.3.

Model A adv-0.3 is trained with FGS samples, while Model A adv-iter-0.3 is trained with iterative FGS samples using t = 40 and α = 0.01.

For the model with ensemble training, Model A adv-ens-0.3 is trained with pre-generated FGS samples for Models A, C, and D, as well as FGS samples.

The source of the samples is chosen randomly for each minibatch during training.

While single-step black-box attacks are less effective at lower than the one used for training, our experiments show that iterative black-box attacks continue to work well even against adversarially trained networks.

For example, the Iterative Gradient Estimation attack using Finite Differences with a logit loss (IFD-logit) achieves an adversarial success rate of 96.4% against Model A adv-ens-0.3 , while the best transferability attack has a success rate of 4.9%.

It is comparable to the white-box attack success rate of 93% from Table 10 .

However, Model A adv-iter-0.3 is quite robust even against iterative attacks, with the highest black-box attack success rate achieved being 14.5%.Further, in FIG3 , we can see that using just 4000 queries per sample, the Iterative Gradient Estimation attack using PCA for query reduction (IGE-QR (PCA-400, logit)) achieves 100% (untargeted) and 74.5% (targeted) adversarial success rates against Model A adv-0.3 .

Our methods far outperform the other black-box attacks, as shown in Table 10 .

We train variants of Resnet-32 using adversarial samples with an L ∞ constraint of 8.

Resnet-32 adv-8 is trained with FGS samples with the same constraint, and Resnet-32 ens-adv-8 is trained with pre-generated FGS samples from Resnet-32 and Std.-CNN as well as FGS samples.

Resnet-32 adv-iter-8 is trained with iterative FGS samples using t = 10 and α = 1.0.

Iterative black-box attacks perform well against adversarially trained models for CIFAR-10 as well.

IFD-logit achieves attack success rates of 100% against both Resnet-32 adv-8 and Resnet-32 adv-ens-8 (Table 3) , which reduces slightly to 97% when IFD-QR (PCA-400, logit) is used.

This matches the performance of white-box attacks as given in Table 10 .

IFD-QR (PCA-400, logit) also achieves a 72% success rate for targeted attacks at = 8 as shown in FIG3 .The iteratively trained model has poor performance on both benign as well as adversarial samples.

Resnet-32 adv-iter-8 has an accuracy of only 79.1% on benign data, as shown in TAB6 .

The Iterative Gradient Estimation attack using Finite Differences with cross-entropy loss (IFD-xent) achieves an untargeted attack success rate of 55% on this model, which is lower than on the other adversarially trained models, but still significant.

This is in line with the observation by BID9 Table 3 : Untargeted black-box attacks for models with adversarial training: adversarial success rates and average distortion ∆(X, X adv ) over the samples.

Above: MNIST, = 0.3.

Below: CIFAR-10, = 8.Summary.

Both single-step and iterative variants of the Gradient Estimation attacks outperform other black-box attacks on both the MNIST and CIFAR-10 datasets, achieving attack success rates close to those of white-box attacks even on adversarially trained models, as can be seen in Table 3 and FIG3 .

Since the only requirement for carrying out the Gradient Estimation based attacks is query-based access to the target model, a number of deployed public systems that provide classification as a service can be used to evaluate our methods.

We choose Clarifai, as it has a number of models trained to classify image datasets for a variety of practical applications, and it provides black-box access to its models and returns confidence scores upon querying.

In particular, Clarifai has models used for the detection of Not Safe For Work (NSFW) content, as well as for Content Moderation.

These are important applications where the presence of adversarial samples presents a real danger: an attacker, using query access to the model, could generate an adversarial sample which will no longer be classified as inappropriate.

For example, an adversary could upload violent images, adversarially modified, such that they are marked incorrectly as 'safe' by the Content Moderation model.

We evaluate our attack using the Gradient Estimation method on the Clarifai NSFW and Content Moderation models.

When we query the API with an image, it returns the confidence scores associated with each category, with the confidence scores summing to 1.

We use the random grouping technique in order to reduce the number of queries and take the logarithm of the confidence scores in order to use the logit loss.

A large number of successful attack images can be found at https: //www.dropbox.com/s/xsu31tjr0yq7rj7/clarifai-examples.zip?dl=0.

Due to their possibly offensive nature, they are not included in the paper.

An example of an attack on the Content Moderation API is given in FIG0 , where the original image on the left is clearly of some kind of drug on a table, with a spoon and a syringe.

It is classified as a drug by the Content Moderation model with a confidence score of 0.99.

The image on the right is an adversarial image generated with 192 queries to the Content Moderation API, with an L ∞ constraint on the perturbation of = 32.

While the image can still clearly be classified by a human as being of drugs on a table, the Content Moderation model now classifies it as 'safe' with a confidence score of 0.96.Remarks.

The proposed Gradient Estimation attacks can successfully generate adversarial examples that are misclassified by a real-world system hosted by Clarifai without prior knowledge of the training set or model.

Overall, in this paper, we conduct a systematic analysis of new and existing black-box attacks on state-of-the-art classifiers and defenses.

We propose Gradient Estimation attacks which achieve high attack success rates comparable with even white-box attacks and outperform other state-of-the-art black-box attacks.

We apply random grouping and PCA based methods to reduce the number of queries required to a small constant and demonstrate the effectiveness of the Gradient Estimation attack even in this setting.

We also apply our black-box attack against a real-world classifier and state-of-the-art defenses.

All of our results show that Gradient Estimation attacks are extremely effective in a variety of settings, making the development of better defenses against black-box attacks an urgent task.

Stephen

In this section, we describe existing methods for generating adversarial examples.

An adversary can generate adversarial example x adv from a benign sample x by adding an appropriate perturbation of small magnitude BID18 .

Such an adversarial example x adv will either cause the classifier to misclassify it into a targeted class (targeted attack), or any class other than the ground truth class (untargeted attack).

Now, we describe two baseline black-box attacks which can be carried out without any knowledge of or query access to the target model.

Random perturbations.

With no knowledge of f or the training set, the simplest manner in which an adversary may seek to carry out an attack is by adding a random perturbation to the input BID18 BID0 BID6 .

These perturbations can be generated by any distribution of the adversary's choice and constrained according to an appropriate norm.

If we let P be a distribution over X , and p is a random variable drawn according to P , then a noisy sample is just x noise = x + p. Since random noise is added, it is not possible to generate targeted adversarial samples in a principled manner.

This attack is denoted as Rand.

throughout.

A perturbation aligned with the difference of means of two classes is likely to be effective for an adversary hoping to cause misclassification for a broad range of classifiers BID22 .

While these perturbations are far from optimal for DNNs, they provide a useful baseline to compare against.

Adversaries with at least partial access to the training or test sets can carry out this attack.

An adversarial sample generated using this method, and with L ∞ constraints, is x adv = x + · sign(µ t − µ o ), where µ t is the mean of the target class and µ o is the mean of the original ground truth class.

For an untargeted attack, t = argmin i d(µ i − µ o ), where d(·, ·) is an appropriately chosen distance function.

In other words, the class whose mean is closest to the original class in terms of the Euclidean distance is chosen to be the target.

This attack is denoted as D. of M. throughout.

Now, we describe two white-box attack methods, used in transferability-based attacks, for which we constructed approximate, gradient-free versions in Section 3.

These attacks are based on either iterative or single-step gradient based minimization of appropriately defined loss functions of neural networks.

Since these methods all require the knowledge of the model's gradient, we assume the adversary has access to a local model f s .

Adversarial samples generated for f s can then be transferred to the target model f t to carry out a transferability-based attack BID12 BID7 ).

An ensemble of local models BID5 may also be used.

Transferability-based attacks are described in Appendix A.2.The single-step Fast Gradient method, first introduced by BID0 , utilizes a firstorder approximation of the loss function in order to construct adversarial samples for the adversary's surrogate local model f s .

The samples are constructed by performing a single step of gradient ascent for untargeted attacks.

Formally, the adversary generates samples x adv with L ∞ constraints (known as the Fast Gradient Sign (FGS) method) in the untargeted attack setting as DISPLAYFORM0 where f s (x, y) is the loss function with respect to which the gradient is taken.

The loss function typically used is the cross-entropy loss BID12 .Iterative Fast Gradient methods are simply multi-step variants of the Fast Gradient method described above (Kurakin et al., 2016) , where the gradient of the loss is added to the sample for t + 1 iterations, starting from the benign sample, and the updated sample is projected to satisfy the constraints H in every step: DISPLAYFORM1 with x 0 adv = x. Iterative fast gradient methods thus essentially carry out projected gradient descent (PGD) with the goal of maximizing the loss, as pointed out by BID9 .

Here we describe black-box attacks that assume the adversary has access to a representative set of training data in order to train a local model.

One of the earliest observations with regards to adversarial samples for neural networks was that they transfer; i.e, adversarial attack samples generated for one network are also adversarial for another network.

This observation directly led to the proposal of a black-box attack where an adversary would generate samples for a local network and transfer these to the target model, which is referred to as a Transferability based attack.

Transferability attack (single local model).

These attacks use a surrogate local model f s to craft adversarial samples, which are then submitted to f in order to cause misclassification.

Most existing black-box attacks are based on transferability from a single local model BID12 BID7 .

The different attack strategies to generate adversarial instances introduced in Section A.1 can be used here to generate adversarial instances against f s , so as to attack f .

s is best suited for generating adversarial samples that transfer well to the target model f , BID5 propose the generation of adversarial examples for an ensemble of local models.

This method modifies each of the existing transferability attacks by substituting a sum over the loss functions in place of the loss from a single local model.

Concretely, let the ensemble of m local models to be used to generate the local loss be {f s1 , . . .

, f sm }.

The ensemble loss is then computed as ens (x, y) = m i=1 α i f s i (x, y), where α i is the weight given to each model in the ensemble.

The FGS attack in the ensemble setting then becomes x adv = x + ·

sign(∇ x ens (x, y)).

The Iterative FGS attack is modified similarly.

BID5 show that the Transferability attack (local model ensemble) performs well even in the targeted attack case, while Transferability attack (single local model) is usually only effective for untargeted attacks.

The intuition is that while one model's gradient may not be adversarial for a target model, it is likely that at least one of the gradient directions from the ensemble represents a direction that is somewhat adversarial for the target model.

BID18 and BID0 introduced the concept of adversarial training, where the standard loss function for a neural network f is modified as follows:

where y is the true label of the sample x. The underlying objective of this modification is to make the neural networks more robust by penalizing it during training to count for adversarial samples.

During training, the adversarial samples are computed with respect to the current state of the network using an appropriate method such as FGSM.Ensemble adversarial training.

BID21 proposed an extension of the adversarial training paradigm which is called ensemble adversarial training.

As the name suggests, in ensemble adversarial training, the network is trained with adversarial samples from multiple networks.

Iterative adversarial training.

A further modification of the adversarial training paradigm proposes training with adversarial samples generated using iterative methods such as the iterative FGSM attack described earlier BID9 .

MNIST.

This is a dataset of images of handwritten digits (LeCun & Cortes, 1998) .

There are 60,000 training examples and 10,000 test examples.

Each image belongs to a single class from 0 to 9.

The images have a dimension d of 28 × 28 pixels (total of 784) and are grayscale.

Each pixel value lies in [0, 1].

The digits are size-normalized and centered.

This dataset is used commonly as a 'sanity-check' or first-level benchmark for state-of-the-art classifiers.

We use this dataset since it has been extensively studied from the attack perspective by previous work.

CIFAR-10.

This is a dataset of color images from 10 classes (Krizhevsky & Hinton, 2009

In this section, we present the architectures and training details for both the normally and adversarially trained variants of the models on both the MNIST and CIFAR-10 datasets.

The accuracy of each model on benign data is given in TAB6 .

Models A and C have both convolutional layers as well as fully connected layers.

They also have the same order of magnitude of parameters.

Model B, on the other hand, does not have fully connected layers and has an order of magnitude fewer parameters.

Similarly, Model D has no convolutional layers and has fewer parameters than all the other models.

Models A, B, and C all achieve greater than 99% classification accuracy on the test data.

Model D achieves 97.2% classification accuracy, due to the lack of convolutional layers.

For all adversarially trained models, each training batch contains 128 samples of which 64 are benign and 64 are adversarial samples (either FGSM or iterative FGSM).

This implies that the loss for each is weighted equally during training; i.e., in Eq. 9, α is set to 0.5.

For ensemble adversarial training, the source of the FGSM samples is chosen randomly for each training batch.

Networks using standard and ensemble adversarial training are trained for 12 epochs, while those using iterative adversarial training are trained for 64 epochs.

In particular, Resnet-32 is a standard 32 layer ResNet with no width expansion, and Resnet-28-10 is a wide ResNet with 28 layers with the width set to 10, based on the best performing ResNet from Zagoruyko & Komodakis (TensorFlow Authors, a).

The width indicates the multiplicative factor by which the number of filters in each residual layer is increased.

Std.-CNN is a CNN with two convolutional layers, each followed by a max-pooling and normalization layer and two fully connected layers, each of which has weight decay.

For each model architecture, we train 3 models, one on only the CIFAR-10 training data, one using standard adversarial training and one using ensemble adversarial training.

Resnet-32 is trained for 125,000 steps, Resnet-28-10 is trained for 167,000 steps and Std.-CNN is trained for 100,000 steps on the benign training data.

Models Resnet-32 and Resnet-28-10 are much more accurate than Std.-CNN.

The adversarial variants of Resnet-32 is trained for 80,000 steps.

All models were trained with a batch size of 128.The two ResNets achieve close to state-of-the-art accuracy ima on the CIFAR-10 test set, with Resnet-32 at 92.4% and Resnet-28-10 at 94.4%.

Std.-CNN, on the other hand, only achieves an accuracy of 81.4%, reflecting its simple architecture and the complexity of the task.

TAB6 :

Accuracy of models on the benign test data C.3 ALTERNATIVE ADVERSARIAL SUCCESS METRIC Note that the adversarial success rate can also be computed by considering only the fraction of inputs that meet the adversary's objective given that the original sample was correctly classified.

That is, one would count the fraction of correctly classified inputs (i.e. f (x) = y) for which f (x adv ) = y in the untargeted case, and f t (x adv ) = T in the targeted case.

In a sense, this fraction represents those samples which are truly adversarial, since they are misclassified solely due to the adversarial perturbation added and not due to the classifier's failure to generalize well.

In practice, both these methods of measuring the adversarial success rate lead to similar results for classifiers with high accuracy on the test data.

Here, we provide a unified framework assuming an adversary can make active queries to the model.

Existing attacks making zero queries are a special case in this framework.

Given an input instance x, the adversary makes a sequence of queries based on the adversarial constraint set H, and iteratively adds perturbations until the desired query results are obtained, using which the corresponding adversarial example x adv is generated.

We formally define the targeted and untargeted black-box attacks based on the framework as below.

Definition 1 (Untargeted black-box attack).

Given an input instance x and an iterative active query attack strategy A, a query sequence can be generated as DISPLAYFORM0 , where q i f denotes the ith corresponding query result on x i , and DISPLAYFORM1 , where k is the number of queries made.

Definition 2 (Targeted black-box attack).

Given an input instance x and an iterative active query attack strategy A, a query sequence can be generated as DISPLAYFORM2 , where q i f denotes the ith corresponding query result on x i , and we set x 1 = x. A black-box attack on f (·; θ) is targeted if the adversarial example x adv = x k satisfies f (x adv ; θ) = T , where T and k are the target class and the number of queries made, respectively.

The case where the adversary makes no queries to the target classifier is a special case we refer to as a zero-query attack.

In the literature, a number of these zero-query attacks have been carried out with varying degrees of success BID12 BID5 BID7 BID8 ).

The expressions for targeted white-box and Gradient Estimation attacks are given in this section.

Targeted transferability attacks are carried out using locally generated targeted white-box adversarial Table 6 : Targeted black-box attacks: adversarial success rates.

The number in parentheses () for each entry is ∆(X, X adv ), the average distortion over all samples used in the attack.

Above: MNIST, = 0.3.

Below: CIFAR-10, = 8.samples.

Adversarial samples generated using the targeted FGS attack are DISPLAYFORM0 where T is the target class.

Similarly, the adversarial samples generated using iterative FGS are DISPLAYFORM1 For the logit based loss, targeted adversarial samples are generated using the following loss term: DISPLAYFORM2 Targeted black-box adversarial samples generated using the Gradient Estimation method are then DISPLAYFORM3 Similarly, in the case of a black-box adversary with query-access to the logits, the adversarial sample is DISPLAYFORM4 F GRADIENT ESTIMATION WITH QUERY REDUCTION

This section contains the detailed algorithm for query reduction using random grouping.

Algorithm 1 Gradient estimation with query reduction using random features DISPLAYFORM0 Choose a set of random k indices Si out of [1, .

DISPLAYFORM1 , which is the two-sided approximation of the directional derivative along v 6: end for DISPLAYFORM2 Concretely, let the samples the adversary wants to misclassify be column vectors x i ∈ R d for i ∈ {1, . . .

, n} and let X be the d × n matrix of centered data samples (i.e. X = [x 1x2 . .

.x n ], wherẽ DISPLAYFORM3 .

The principal components of X are the normalized eigenvectors of its sample DISPLAYFORM4 Since C is a positive semidefinite matrix, there is a decomposition C = UΛU T where U is an orthogonal matrix, Λ = diag(λ 1 , . . .

, λ d ), and λ 1 ≥ . . .

≥ λ d ≥ 0.

Thus, U in Algorithm 2 is the d × d matrix whose columns are unit eigenvectors of C. The eigenvalue λ i is the variance of X along the i th component.

Further, PCA minimizes reconstruction error in terms of the L 2 norm; i.e., it provides a basis in which the Euclidean distance to the original sample from a sample reconstructed using a subset of the basis vectors is the smallest.

Algorithm 2 Gradient estimation with query reduction using PCA components Input: DISPLAYFORM5 Initialize v such that v = ui ui , where u i is the i th column of U 3: DISPLAYFORM6 which is the two-sided approximation of the directional derivative along v 4: DISPLAYFORM7 Taxonomy of black-box attacks: To deepen our understanding of the effectiveness of black-box attacks, in this work, we propose a taxonomy of black-box attacks, intuitively based on the number of queries on the target model used in the attack.

The details are provided in TAB8 .We evaluate the following attacks summarized in

In FIG9 , we show some examples of successful untargeted adversarial samples against Model A on MNIST and Resnet-32 on CIFAR-10.

These images were generated with an L ∞ constraint of = 0.3 for MNIST and = 8 for CIFAR-10.

Clearly, the amount of perturbation added by iterative attacks is much smaller, barely being visible in the images.

In this section, we present the white-box attack results for various cases in Tables 8-10 .

Where relevant, our results match previous work BID0 Kurakin et al., 2016) .

In the baseline attacks described in Appendix A.1.1, the choice of distribution for the random perturbation attack and the choice of distance function for the difference of means attack are not fixed.

Here, we describe the choices we make for both attacks.

The random perturbation p for each sample (for both MNIST and CIFAR-10) is chosen independently according to a multivariate normal distribution with mean 0, i.e. p ∼ N (0, I d ).

Then, depending on the norm constraint, either a signed and scaled version of the random perturbation (L ∞ ) or a scaled unit vector in the direction of the perturbation (L 2 ) is added.

For an untargeted attack utilizing perturbations aligned with the difference of means, for each sample, the mean of the class closest to the original class in the L 2 distance is determined.

All attacks use the logit loss.

Perturbations in the images generated using single-step attacks are far smaller than those for iterative attacks.

The '7' from MNIST is classified as a '3' by all single-step attacks and as a '9' by all iterative attacks.

The dog from CIFAR-10 is classified as a bird by the white-box FGS and Finite Difference attack, and as a frog by the Gradient Estimation attack with query reduction.

White-box Table 8 : Untargeted white-box attacks: adversarial success rates and average distortion ∆(X, X adv ) over the test set.

Above: MNIST, = 0.3.

Below: CIFAR-10, = 8.As expected, adversarial samples generated using Rand.

do not achieve high adversarial success rates in spite of having similar or larger average distortion than the other black-box attacks for both the MNIST and CIFAR-10 models.

However, the D. of M. method is quite effective at higher perturbation values for the MNIST dataset as can be seen in FIG2 .

Also, for Models B and D, the D. of M. attack is more effective than FD-xent.

The D. of M. method is less effective in the targeted attack case, but for Model D, it outperforms the transferability based attack considerably.

Its success rate is comparable to the targeted transferability based attack for Model A as well.

The relative effectiveness of the two baseline methods is reversed for the CIFAR-10 dataset, however, where Rand.

outperforms D. of M. considerably as is increased.

This indicates that the models trained on MNIST have normal vectors to decision boundaries which are more aligned with the vectors along the difference of means as compared to the models on CIFAR-10.

For the transferability experiments, we choose to transfer from Model B for MNIST dataset and from Resnet-28-10 for CIFAR-10 dataset, as these models are each similar to at least one of the Table 10 : Untargeted white-box attacks for models with adversarial training: adversarial success rates and average distortion ∆(X, X adv ) over the test set.

Above: MNIST, = 0.3.

Below: CIFAR-10, = 8.other models for their respective dataset and different from one of the others.

They are also fairly representative instances of DNNs used in practice.

Adversarial samples generated using single-step methods and transferred from Model B to the other models have higher success rates for untargeted attacks when they are generated using the logit loss as compared to the cross entropy loss as can be seen in Table 1 .

For iterative adversarial samples, however, the untargeted attack success rates are roughly the same for both loss functions.

As has been observed before, the adversarial success rate for targeted attacks with transferability is much lower than the untargeted case, even when iteratively generated samples are used.

For example, the highest targeted transferability rate in Table 6 is 54.5%, compared to 100.0% achieved by IFD-xent-T across models.

One attempt to improve the transferability rate is to use an ensemble of local models, instead of a single one.

The results for this on the MNIST data are presented in TAB7 .

In general, both untargeted and targeted transferability increase when an ensemble is used.

However, the increase is not monotonic in the number of models used in the ensemble, and we can see that the transferability rate for IFGS-xent samples falls sharply when Model D is added to the ensemble.

This may be due to it having a very different architecture as compared to the models, and thus also having very different gradient directions.

This highlights one of the pitfalls of transferability, where it is important to use a local surrogate model similar to the target model for achieving high attack success rates.

(c) Gradient Estimation attack with query reduction using PCA components and the logit loss (GE-QR (PCA-k, logit)) on Resnet-32 (CIFAR-10).

Relatively high success rates are maintained even for k = 400.

We consider the effectiveness of Gradient Estimation with random grouping based query reduction and the logit loss (GE-QR (RG-k, logit)) on Model A on MNIST data in FIG13 , where k is the number of indices chosen in each iteration of Algorithm 1.

Thus, as k increases and the number of groups decreases, we expect adversarial success to decrease as gradients over larger groups of features are averaged.

This is the effect we see in FIG13 , where the adversarial success rate drops from 93% to 63% at = 0.3 as k increases from 1 to 7.

Grouping with k = 7 translates to 112 queries per MNIST image, down from 784.

Thus, in order to achieve high adversarial success rates with the random grouping method, larger perturbation magnitudes are needed.

On the other hand, the PCA-based approach GE-QR (PCA-k, logit) is much more effective, as can be seen in FIG13 .

Using 100 principal components to estimate the gradient for Model A on MNIST as in Algorithm 2, the adversarial success rate at = 0.3 is 88.09%, as compared to 92.9% without any query reduction.

Similarly, using 400 principal components for Resnet-32 on CIFAR-10 FIG13 ), an adversarial success rate of 66.9% can be achieved at = 8.

At = 16, the adversarial success rate rises to 80.1%.

In this section, we analyse the effectiveness of single-step black-box attacks on adversarially trained models and show that the Gradient Estimation attacks using Finite Differences with the addition of random perturbations outperform other black-box attacks.

Evaluation of single-step attacks on model with basic adversarial training: In Figure 6a , we can see that both single-step black-box and white-box attacks have much lower adversarial success rates on Model A adv-0.3 as compared to Model A. The success rate of the Gradient Estimation attacks matches that of white-box attacks on these adversarially trained networks as well.

To overcome this, we add an initial random perturbation to samples before using the Gradient Estimation attack with Finite Differences and the logit loss (FD-logit).

These are then the most effective single step black-box attacks on Model A adv-0.3 at = 0.3 with an adversarial success rate of 32.2%, surpassing the Transferability attack (single local model) from B. In Figure 6b , we again see that the Gradient Estimation attacks using Finite Differences (FD-xent and FD-logit) and white-box FGS attacks (FGS-xent and FGS-logit) against Resnet-32.

As is increased, the attacks that perform the best are Random Perturbations (Rand.), Difference-ofmeans (D. of M.), and Transferability attack (single local model) from Resnet-28-10 with the latter performing slightly better than the baseline attacks.

This is due to the 'gradient masking' phenomenon and can be overcome by adding random perturbations as for MNIST.

An interesting effect is observed at = 4, where the adversarial success rate is higher than at = 8.

The likely explanation for this effect is that the model has overfitted to adversarial samples at = 8.

Our Gradient Estimation attack closely tracks the adversarial success rate of white-box attacks in this setting as well.

Increasing effectiveness of single-step attacks using initial random perturbation: Since the Gradient Estimation attack with Finite Differences (FD-xent and FD-logit) were not performing well due the masking of gradients at the benign sample x, we added an initial random perturbation to escape this low-gradient region as in the RAND-FGSM attack BID21 .

Figure 7 shows the effect of adding an initial L ∞ -constrained perturbation of magnitude 0.05.

With the addition of a random perturbation, FD-logit has a much improved adversarial success rate on Model A adv-0.3 , going up to 32.2% from 2.8% without the perturbation at a total perturbation value of 0.3.

It even outperforms the white-box FGS (FGS-logit) with the same random perturbation added.

This effect is also observed for Model A adv-ens-0.3 , but Model A adv-iter-0.3 appears to be resistant to single-step gradient based attacks.

Thus, our attacks work well for single-step attacks on DNNs with standard and ensemble adversarial training, and achieve performance levels close to that of white-box attacks.

In our evaluations, all models were run on a GPU with a batch size of 100.

On Model A on MNIST data, single-step attacks FD-xent and FD-logit take 6.2 × 10 −2 and 8.8 × 10 −2 seconds per sample respectively.

Thus, these attacks can be carried out on the entire MNIST test set of 10,000 images in about 10 minutes.

For iterative attacks with no query reduction, with 40 iterations per sample (α set to 0.01), both IFD-xent and IFD-xent-T taking about 2.4 seconds per sample.

Similarly, IFD-logit and IFD-logit-T take about 3.5 seconds per sample.

With query reduction, using IGE-QR (PCA-k, logit) with k = 100 and IGE-QR (RG-k, logit) with k = 8, the time taken is just 0.5 seconds per sample.

In contrast, the fastest attack from , the ZOO-ADAM attack, takes around 80 seconds per sample for MNIST, which is 24× slower than the Iterative Finite Difference attacks and around 160× slower than the Iterative Gradient Estimation attacks with query reduction.

For Resnet-32 on the CIFAR-10 dataset, FD-xent, FD-xent-T, FD-logit and FD-logit-T all take roughly 3s per sample.

The iterative variants of these attacks with 10 iterations (α set to 1.0) take roughly 30s per sample.

Using query reduction, both IGE-QR (PCA-k, logit) with k = 100 with 10 iterations takes just 5s per sample.

The time required per sample increases with the complexity of the network, which is observed even for white-box attacks.

For the CIFAR-10 dataset, the fastest attack from takes about 206 seconds per sample, which is 7× slower than the Iterative Finite Difference attacks and around 40× slower than the Iterative Gradient Estimation attacks with query reduction.

All the above numbers are for the case when queries are not made in parallel.

Our attack algorithm allows for queries to be made in parallel as well.

We find that a simple parallelization of the queries gives us a 2 − 4× speedup.

The limiting factor is the fact that the model is loaded on a single GPU, which implies that the current setup is not fully optimized to take advantage of the inherently parallel nature of our attack.

With further optimization, greater speedups can be achieved.

Remarks:

Overall, our attacks are very efficient and allow an adversary to generate a large number of adversarial samples in a short period of time.

<|TLDR|>

@highlight

Query-based black-box attacks on deep neural networks with adversarial success rates matching white-box attacks