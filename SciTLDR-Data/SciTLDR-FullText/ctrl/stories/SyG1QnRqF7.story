Learning deep networks which can resist large variations between training andtesting data is essential to build accurate and robust image classifiers.

Towardsthis end, a typical strategy is to apply data augmentation to enlarge the trainingset.

However,  standard  data  augmentation  is  essentially  a  brute-force  strategywhich is inefficient,  as it performs all the pre-defined transformations  to everytraining sample.

In this paper, we propose a principled approach to train networkswith  significantly  improved  resistance  to  large  variations  between  training  andtesting data.

This is achieved by embedding a learnable transformation moduleinto the introspective networks (Jin et al., 2017; Lazarow et al., 2017; Lee et al.,2018), which is a convolutional neural network (CNN) classifier empowered withgenerative capabilities.

Our approach alternatively synthesizes pseudo-negativesamples with learned transformations and enhances the classifier by retraining itwith synthesized samples.

Experimental results verify that our approach signif-icantly improves the ability of deep networks to resist large variations betweentraining and testing data and achieves classification accuracy improvements onseveral benchmark datasets, including MNIST, affNIST, SVHN and CIFAR-10.

Classification problems have rapidly progressed with advancements in convolutional neural networks (CNNs) BID17 BID15 BID25 BID6 BID7 .

CNNs are able to produce promising performance, given sufficient training data.

However, when the training data is limited and unable to cover all the data variations in the testing data (e.g., the training set is MNIST, while the testing set is affNIST), the trained networks generalize poorly on the testing data.

Consequently, how to learn deep networks which can resist large variations between training and testing data is a significant challenge for building accurate and robust image classifiers.

To address this issue, a typical strategy is to apply data augmentation to enlarging the training set, i.e., applying various transformations, including random translations, rotations and flips as well as Gaussian noise injection, to the existing training data.

This strategy is very effective in improving the performance, but it is essentially a brute-force strategy which is inefficient, as it exhaustively performs all these transformations to every training samples.

Neither is it theoretically formulated.

Alternatively, we realize that we can synthesize extra training samples with generative models.

But, the problem is how to generate synthetic samples which are able to improve the robustness of CNNs to large variations between training and testing data.

In this paper, we achieve this by embedding a learnable transformation module into introspective networks , a CNN classifier empowered with generative capabilities.

We name our approach introspective transformation network (ITN), which performs training by a reclassification-by-synthesis algorithm.

It alternatively synthesizes samples with learned transformations and enhances the classifier by retraining it with synthesized samples.

We use a min-max formulation to learn our ITN, where the transformation module transforms the synthesized pseudo-negative samples to maximize their variations to the original training samples and the CNN classifier is updated by minimizing the classification loss of the transformed synthesized pseudo-negative samples.

The transformation modules are learned jointly with the CNN classifier, which augments training data in an intelligent manner by narrowing down the search space for the variations.

Our approach can work with any models that have generative and discriminative abilities, such as generative adversarial networks (GANs) and introspective networks.

In this paper, we choose the introspective networks to generate extra training samples rather than GANs, because introspective networks have several advantages over GANs.

Introspective learning framework maintains one single CNN discriminator that itself is also a generator while GANs have separate discriminators and generators.

The generative and discriminative models are simultaneously refined over iterations.

Additionally, Introspective networks are easier to train than GANs with gradient descent algorithms by avoiding adversarial learning.

The main contribution of the paper is that we propose a principled approach that endows classifiers with the ability to resist larger variations between training and testing data in an intelligent and efficient manner.

Experimental results show that our approach achieves better performance than standard data augmentation on both classification and cross-dataset generalization.

Furthermore, we also show that our approach has great abilities in resisting different types of variations between training and testing data.

In recent years, a significant number of works have emerged focus on resisting large variations between training and testing data.

The most widely adopted approach is data augmentation that applies pre-defined transformations to the training data.

Nevertheless, this method lacks efficiency and stability since the users have to predict the types of transformations and manually applies them to the training set.

Better methods have been proposed by building connections between generative models and discriminative classifiers BID3 BID20 BID29 BID11 BID30 .

This type of methods capture the underlying generation process of the entire dataset.

The discrepancy between training and test data is reduced by generating more samples from the data distribution.

GANs BID4 have led a huge wave in exploring the generative adversarial structures.

Combining this structure with deep CNNs can produce models that have stronger generative abilities.

In GANs, generators and discriminators are trained simultaneously.

Generators try to generate fake images that fool the discriminators, while discriminators try to distinguish the real and fake images.

Many variations of GANs have emerged in the past three years, like DCGAN BID22 , WGAN and WGAN-GP BID5 .

These GANs variations show stronger learning ability that enables generating complex images.

Techniques have been proposed to improve adversarial learning for image generation BID24 BID5 BID1 as well as for training better image generative models BID22 BID9 .Introspective networks BID28 BID19 provide an alternative approach to generate samples.

Introspective networks are closely related to GANs since they both have generative and discriminative abilities but different in various ways.

Introspective networks maintain one single model that is both discriminative and generative at the same time while GANs have distinct generators and discriminators.

Introspective networks focus on introspective learning that synthesizes samples from its own classifier.

On the other hand, GANs emphasize adversarial learning that guides generators with separate discriminators.

The generators in GANs are mappings from the features to the images.

However, Introspective networks directly models the underlying statistics of an image with an efficient sampling/inference process.

We now describe the details of our approach in this section.

We first briefly review the introspective learning framework proposed by BID28 .

This is followed by our the detailed mathematical explanation of our approach.

In particular, we focus on explaining how our model generates unseen examples that complement the training dataset.

We only briefly review introspective learning for binary-class problems, since the same idea can be easily extended to multi-class problems.

Let us denote x ∈ R d as a data sample and y ∈ +1, −1 as the corresponding label of x. The goal of introspective learning is to model positive samples by learning the generative model p(x|y = +1).

Under Bayes rule, we have DISPLAYFORM0 where p(y|x) is a discriminative model.

For pedagogical simplicity, we assume p(y = 1) = p(y = −1) and this equation can be further simplified as: DISPLAYFORM1 The above equation suggests that a generative model for the positives p(x|y = +1) can be obtained from the discriminative model p(y|x) and a generative model p(x|y = −1) for the negatives.

However, to faithfully learn p(x|y = +1), we need to have a representative p(x|y = −1), which is very difficult to obtain.

A solution was provided in BID28 which learns p(x|y = −1) by using an iterative process starting from an initial reference distribution of the negatives p 0 (x|y = −1), e.g., p 0 (x|y = −1) = U (x), a Gaussian distribution on the entire space R d .

This is updated by DISPLAYFORM2 where q t (y|x) is a discriminative model learned on a given set of positives and a limited number of pseudo-negatives sampled from p t (x|y = −1) and Z t =qt FORMULA0 qt(y=−1|x) p t (x|y = −1)dx is the normalizing factor.

It has been proven that KL(p(x|y = +1)||p t+1 (x|y = −1)) ≤ KL(p(x|y = +1)||p t (x|y = −1))) (as long as each q t (y|x) makes a better-than-random prediction, the inequality holds) in BID28 , where KL(·||·) denotes the Kullback-Leibler divergences, which implies p t (x|y = −1) t=∞ → p(x|y = +1).

Therefore, gradually learning p t (x|y = −1) by following this iterative process of Eqn.(3), the samples drawn from x ∼ p t (x|y = −1) become indistinguishable from the given training samples.

Introspective Convolutional Networks (ICN) and Wasserstein Introspective Neural Networks (WINN) BID19 adopt the introspective learning framework and strengthen the classifiers by a reclassification-by-synthesis algorithm.

However, both of them fail to capture large data variations between the training and testing data, since most of the generated pseudo-negatives are very similar to the original samples.

But in practice, it is very common that the test data contain unseen variations that are not in training data, such as the same objects viewed from different angles and suffered from shape deformation.

To address this issue, we present our approach building upon the introspective learning framework to resist large data variations between training and test data.

Arguably, even large training sets cannot fully contains all the possible variations.

Our goal is to quickly generate extra training samples with beneficial unseen variations that is not covered by the training data to help classifiers become robust.

We assume that we can generates such training samples by applying a transformation function T (· ; σ) parametrized by learnable parameters σ to the original training samples.

Let us denote g(· ; ψ) as the function that maps the samples x to the transformation parameters σ, where ψ is the model parameter of the function g. The generated samples still belong to the same category of the original samples, since the transformation function T only changes the high-level geometric properties of the samples.

The outline of training procedures of ITN is presented in Algorithm 1.

We denote S + = {(x + i , +1), i = 1...|S + |} as the positive sample set, DISPLAYFORM0 ; σ t )} as the transformed positive sample set at t th iteration with transformation parameter σ t and S DISPLAYFORM1 as the set of pseudonegatives drawn from p t (x|y = −1).

We then will describe the detail of the training procedure.

Discriminative model We first demonstrate the approach of building robust classifiers with given σ t .

For a binary classification problem, at t th iteration, the discriminative model is represented as DISPLAYFORM2 Algorithm 1: Outline of ITN Training Algorithm 1: Input: Positive sample set S + , initial reference distribution p0(x|y = −1) and transformation function T 2: Output: Parameters θ, ω and ψ 3: Build S − 0 by sampling |S + | pseudo-negatives samples from p0(x|y = −1) 4: initialize parameters θ, ω and ψ, set t = 1 5: while not converge do 6:for each x DISPLAYFORM3 Compute transformation parameters σi = g(x DISPLAYFORM4 Choose i ∼ U (0, 1) and computexi = iT (x DISPLAYFORM5 9: end for 10:Compute θ, ω by Eqn.(6) 11:Compute ψ by Eqn.

FORMULA0 Sample pseudo-negatives samples Zt = {z DISPLAYFORM6 Update all samples in Zt by Eqn.

FORMULA0 Augment pseudo-negatives sample set S DISPLAYFORM7 .., |S + |} and t = t + 1 15: end while where θ t represents the model parameters at iteration t, and f t (x; θ t ) represents the model output at t th iteration.

Note that, q t (y|x; θ t ) is trained on S + , T (S + ; σ t ) and pseudo-negatives drawn from p t (x|y = −1).

In order to achieve stronger ability in resisting unseen variations, we want the distribution of T (S + ; σ t ) to be approximated by the distribution of pseudo negatives p t (x|y = −1), which can be achieved by minimizing the following Wasserstein distance BID5 : DISPLAYFORM8 where ω t is the extra parameter together with f t (·; θ t ) to compute the Wasserstein distance.

Eachx in the setX t is computed with the formulax DISPLAYFORM9 2 is the gradient penalty that stabilizes the training procedure of the Wasserstein loss function.

The goal of the discriminative model is to correctly classify any given x + , x T and x − .

Thus, the objective function of learning the discriminative model at iteration t is DISPLAYFORM10 The classifiers obtain the strong ability in resisting unseen variations by training on the extra samples while preserving the ability to correctly classify the original samples.

We discussed the binary classification case above.

When dealing with multi-class classification problems, it is needed to adapt the above reclassification-by-synthesis scheme to the multi-class case.

We can directly follow the strategies proposed in to extend ITN to deal with multi-class problems by learning a series of one-vs-all classifiers or a single CNN classifier.

Exploring variations.

The previous section describes how to learn the robust classifiers when the σ t is given.

However, σ t is unknown and there are huge number of possibilities to selecting σ t .

Now, the problem becomes how do we learn the σ t in a principled manner and apply it towards building robust classifiers?

We solve this issue by forming a min-max problem upon the Eqn.

FORMULA13 : DISPLAYFORM11 Here, we rewrite J(θ) and D(θ, ω) in Eqn.

FORMULA11 and Eqn.(6) as J(θ, σ) and D(θ, ω, σ), since σ is now an unknown variable.

We also subsequently drop the subscript t for notational simplicity.

This formulation gives us a unified perspective that encompasses some prior work on building robust classifiers.

The inner maximization part aims to find the transformation parameter σ that achieves the high loss values.

On the other hand, the goal of the outer minimization is expected to find the the model parameters θ that enables discriminators to correctly classify x T and ω allows the negative distribution to well approximate the distribution of T (S + ; σ t ) .

However, direclty solving Eqn.

7 is difficult.

Thus, we break this learning process and first find a σ * that satisfies DISPLAYFORM12 where θ and ω are fixed.

Then, θ and ω are learned with Eqn.(6) by keep σ = σ * .

Empirically, the first term in the Eqn.

8 dominates over other terms, therefore we can drop the second and third terms to focus on learning more robust classifiers.

The purpose of empirical approximation is to find the σ * that make x T hard to classify correctly.

Instead of enumerating all possible examples in the data augmentation, Eqn.(8) efficiently and precisely finds a proper σ that increase the robustness of the current classifiers.

We use g(· ; ψ) to learn σ, thus σ = g(x; ψ)+ζ, where ζ is random noise follows the standard normal distribution.

The function parameter ψ is learned by Eqn.(8) .

Notably, following the standard backpropagation procedure, we need to compute the derivative of the transformation function T in each step.

In other words, the transformation function T (·; σ) need to be differentiable with respect to the parameter ψ to allow the gradients to flow through the transformation function T when learning by backpropagation.

Generative model In the discriminative models, the updated discriminative model p(y|x) is learned by Eqn.(6).

The updated discriminative model is then used to compute the generative model by the Eqn.(3) in section 3.1.

The generative is learned by maximizing the likelihood function p(x).

However, directly learning the generative model is cumbersome since we only need samples from the latest generative model.

DISPLAYFORM13 where Z t indicates the normalizing factor at t th iteration.

The random samples x are updated by increasing maximize the log likelihood of p − n (x).

Note that maximizing log p Taking natural logarithm on both side of the equation above, we can get ln h t (x) = f t (x; θ t ).

Therefore, log p − n (x) can be rewritten as DISPLAYFORM14 where C is the constant computed with normalizing factors Z t .

This conversion allows us to maximize log p − n (x) by maximizing n−1 t=1 f t (x; θ t ).

By taking the derivative of log p − n (x), the update step ∇x is: DISPLAYFORM15 where η ∼ N (0, 1) is the random Gaussian noise and λ is the step size that is annealed in the sampling process.

In practice, we update from the samples generated from previous iterations to reduce time and memory complexity.

An update threshold is introduced to guarantee the generated negative images are above certain criteria, which ensures the quality of negative samples.

We modify the update threshold proposed in BID19 and keep track of the f t (x; θ t ) in every iteration.

In particular, we build a set D by recording E[f t (x; θ t )], where x ∈ S + in every iteration.

We form a normal distribution N (a, b) , where a and b represents mean and standard deviation computed from set D. The stop threshold is set to be a random number sampled from this normal distribution.

The reason behind this threshold is to make sure the generated negative images are close to the majority of transformed positive images in the feature space.

In this section, we demonstrate the ability of our algorithm in resisting the large variations between training and testing data through a series of experiments.

First, we show the outstanding classification performance of ITN on several benchmark datasets.

We also analyze the properties of the generated examples from different perspectives.

We then further explore the ability of our algorithm in resisting large variations with two challenging classification tasks and show the consistently better performance.

Finally, we illustrate the flexibility of our architecture in addressing different types of unseen variations.

Baselines We compare our method against CNNs, DCGAN (Radford et al., 2015) , WGAN-GP BID5 , ICN and WINN BID19 .

For generative models DCGAN and WGAN-GP, we adopt the evaluation metric proposed in .

The training phase becomes a two-step implementation.

We first generate negative samples with the original implementation.

Then, the generated negative images are used to augment the original training set.

We train a simple CNN that has the identical structure with our method on the augmented training set.

All results reported in this section are the average of multiple repetitions.

Experiment Setup All experiments are conducted with a simple CNN architecture BID19 that contains 4 convolutional layers, each having a 5 × 5 filter size with 64 channels and stride 2 in all layers.

Each convolutional layer is followed by a batch normalization layer BID8 and a swish activation function BID23 .

The last convolutional layer is followed by two consecutive fully connected layers to compute logits and Wasserstein distances.

The training epochs are 200 for both our method and all other baselines.

The optimizer used is the Adam optimizer BID13 with parameters β 1 = 0 and β 2 = 0.9.

Our method relies on the transformation function T (·) to convert the original samples to the unseen variations.

In the following experiments, we demonstrate the ability of ITN in resisting large variations with spatial transformers (STs) BID10 as our transformation function unless specified.

Theoretically, STs can represent all affine transformations, which endows more flexible ability in resisting unseen variations.

More importantly, STs are fully differentiable, which allows the learning procedure through standard backpropagation.

To demonstrate the effectiveness of ITN, we first evaluate our algorithm on 4 benchmark datasets, MNIST BID18 , affNIST (Tieleman, 2013), SVHN BID21 and CIFAR-10 ( BID14 ).

The MNIST dataset includes 55000, 5000 and 10000 handwritten digits in the training, validation and testing set, respectively.

The affNIST dataset is a variant from the MNIST dataset and it is built by applying various affine transformations to the samples in MNIST dataset.

To accord with the MNIST dataset and for the purpose of the following experiments, we reduce the size of training, validation and testing set to 55000, 5000 and 10000, respectively.

SVHN is a real-world dataset that contains house numbers images from Google Street View and CIFAR-10 contains 60000 images of ten different objects from the natural scenes.

The purpose of introducing these two datasets is to further verify the performance of ITN on real-world datasets.

The data augmentation we applied in the following experiments is the standard data augmentation that includes affine transformations, such as rotation, translation, scaling and shear.

Table 1 : testing errors of the classification experiments discussed in Section 4.1, where w/DA and w/o DA indicates whether data augmentation is applied.

As shown in Table 1 , our method achieves the best performance on all four datasets.

The overall improvements can be explained by the fact that our method generates novel and reliable negative images (shown in Figure 1 ) that effectively strengthen the classifiers.

The images we generate are different from the previous ones, but can still be recognized as the same class.

The boosted performance in value on MNIST dataset is marginal perhaps because the performance on the MNIST dataset is close to saturation.

The difference between training and testing split in MNIST dataset is also very small compared to other datasets.

Moreover, the amount of improvements increases as the dataset becomes complicated.

Based on the observation of the results, we conclude that our method has stronger ability in resisting unseen variations especially when the dataset is complicated.

On the other hand, we can clearly observe that our method outperforms the standard data augmentation on all datasets.

This result confirms the effectiveness and the advantages of our approach.

Additionally, ITN does not contradict with data augmentation since ITN shows even greater performance when integrating with data augmentation techniques.

The possible reason for this observation is that the explored space between ITN and data augmentation is not overlapped.

Therefore, the algorithm achieves greater performance when combining two methods together since more unseen variations are discovered in this case.

Figure 1: Images generated by our method on MNIST, affNIST, SVHN and CIFAR-10 dataset.

In each sector, the top row is the original images and the bottom row is our generated images.

We have shown the substantial performance improvements of ITN against other baselines on several benchmark datasets.

In this section, we want to further explore the ability of our method in resisting large variations.

We design a challenging cross dataset classification task between two significantly different datasets (cross dataset generalization).

The training set in this experiment is the MNIST dataset while the testing set is the affNIST dataset.

The difficulty of this classification tasks is clearly how to overcome such huge data discrepancy between training and testing set since the testing set includes much more variations.

Another reason why we pick these two datasets as training and testing set is that they share the same categories, which ensures the challenge is only about resisting large data variations.

As shown in TAB2 , ITN has clear improvements over CNN, WGAN-GP and WINN.

The amount of improvement is much larger than on the regular training and testing splits shown in Section 4.1.

More importantly, our performance in this challenging task is still better than CNN with data augmentation.

This encouraging result further verifies the efficiency and effectiveness of ITN compared with data augmentation.

It's not surprising that data augmentation improves the performance by a significant margin since the space of unseen variations is huge.

Data augmentation increases the classification performance by enumerating a large number of unseen samples, however, this bruteforce searching inevitably lacks efficiency and precision.

Another way to evaluate the ability of resisting variations is to reduce the amount of training samples.

Intuitively, the discrepancy between the training and testing sets increases when the number of samples in the training set shrinks.

The purpose of this experiment is to demonstrate the potential of ITN in resisting unseen variations from a different perspective.

We design the experiments where the training set is the MNIST dataset with only 0.1%, 1%, 10% and 25% of the whole training set while the testing set is the whole MNIST testing set.

Each sample is randomly selected from the pool while keeps the number of samples per class same.

Similarly, we repeat the same experiments on the CIFAR-10 dataset to further verify the results on a more complicated dataset.

As shown in Table 3 , our method has better results on all tasks.

This result is consistent with Section 4.2.1 and Section 4.1, which undoubtedly illustrate the strong ability of ITN in resisting unseen variations in the testing set.

The constant superior performance over data augmentation also proves the efficiency of ITN.

Table 3 : testing errors of the classification tasks described in Section 4.2.2, where M and C represents the experiments conducted on the MNIST dataset and CIFAR-10 dataset, respectively.

Even though we utilize STs to demonstrate our ability in resisting data variations, our method actually has the ability to generalize to other types of transformations.

Our algorithm can take other types of differentiable transformation functions and strengthen the discriminators in a similar manner.

Moreover, our algorithm can utilize multiple types of transformations at the same time and provide even stronger ability in resisting variations.

To verify this, we introduce another recently proposed work, Deep Diffeomorphic Transformer (DDT) Networks BID2 .

DDTs are similar to STs in a way that both of them can be optimized through standard backpropagation.

We replace the ST modules with the DDT modules and check whether our algorithm can resist such type of transformation.

Then, we include both STs and DDTs in our model and verify the performance again.

Let MNIST dataset be the training set of the experiments while the testing sets are the MNIST dataset with different types of transformation applied.

We introduce two types of testing sets in this section.

The first one is the normal testing set with random DDT transformation only.

The second one is similar to the first one but includes both random DDT and affine transformations.

The DDT transformation parameters are drawn from N (0, 0.7 × I d ) as suggest in BID2 , where I d represents the d dimensional identity matrix.

Then the transformed images are randomly placed in a 42 × 42 images.

We replicate the same experiment on the CIFAR-10 dataset.

We can make some interesting observations from the TAB5 .

First, ITN can integrate with flexibly with DDT or DDT + ST to resist the corresponding variations.

Second, ITN can resist partial unseen variations out of a mixture of transformations in the testing data.

More importantly, the performance of ITN won't degrade when the model has transformation functions that doesn't match the type of variations in the testing data, e.g. ITN(DDT + ST) on testing data with DDT only.

This observation allows us to apply multiple transformation functions in ITN without knowing the types of variations in the testing data and still maintain good performance.

We proposed a principled and smart approach that endows the classifiers with the ability to resist larger variations between training and testing data.

Our method, ITN strengthens the classifiers by generating unseen variations with various learned transformations.

Experimental results show consistent performance improvements not only on the classification tasks but also on the other challenging classification tasks, such as cross dataset generalization.

Moreover, ITN demonstrates its advantages in both effectiveness and efficiency over data augmentation.

Our future work includes applying our approach to large scale datasets and extending it to generate samples with more types of variations.

<|TLDR|>

@highlight

We propose a principled approach that endows classifiers with the ability to resist larger variations between training and testing data in an intelligent and efficient manner.

@highlight

Using introspective learning to handle data variations at test time

@highlight

This paper suggests the use of learned transformation networks, embedded within introspective networks to improve classification performance with synthesized examples.