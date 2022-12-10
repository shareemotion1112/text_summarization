Though deep neural networks have achieved the state of the art performance in visual classification, recent studies have shown that they are all vulnerable to the attack of adversarial examples.

To solve the problem, some regularization adversarial training methods, constraining the output label or logit, have been studied.

In this paper, we propose a novel regularized adversarial training framework ATLPA,namely Adversarial Tolerant Logit Pairing with Attention.

Instead of constraining a hard distribution (e.g., one-hot vectors or logit) in adversarial training, ATLPA uses Tolerant Logit which consists of confidence distribution on top-k classes and captures inter-class similarities at the image level.

Specifically, in addition to minimizing the empirical loss, ATLPA encourages attention map for pairs of examples to be similar.

When applied to clean examples and their adversarial counterparts, ATLPA improves accuracy on adversarial examples over adversarial training.

We evaluate ATLPA with the state of the art algorithms, the experiment results show that our method outperforms these baselines with higher accuracy.

Compared with previous work, our work is evaluated under highly challenging PGD attack: the maximum perturbation $\epsilon$ is 64 and 128 with 10 to 200 attack iterations.

In recent years, deep neural networks have been extensively deployed for computer vision tasks, particularly visual classification problems, where new algorithms reported to achieve or even surpass the human performance (Krizhevsky et al., 2012; He et al., 2015; Li et al., 2019a) .

Success of deep neural networks has led to an explosion in demand.

Recent studies (Szegedy et al., 2013; Goodfellow et al., 2014; Carlini & Wagner, 2016; Moosavi-Dezfooli et al., 2016; Bose & Aarabi, 2018) have shown that they are all vulnerable to the attack of adversarial examples.

Small and often imperceptible perturbations to the input images are sufficient to fool the most powerful deep neural networks.

In order to solve this problem, many defence methods have been proposed, among which adversarial training is considered to be the most effective one .Adversarial training (Goodfellow et al., 2014; Madry et al., 2017; Kannan et al., 2018; Tramèr et al., 2017; Pang et al., 2019) defends against adversarial perturbations by training networks on adversarial images that are generated on-the-fly during training.

Although aforementioned methods demonstrated the power of adversarial training in defence, we argue that we need to perform research on at least the following two aspects in order to further improve current defence methods.

Strictness vs. Tolerant.

Most existing defence methods only fit the outputs of adversarial examples to the one-hot vectors of clean examples counterparts.

Kannan et al. (2018) also fit confidence distribution on the all logits of clean examples counterparts, they call it as Logits Pair.

Despite its effectiveness, this is not necessarily the optimal target to fit, because except for maximizing the confidence score of the primary class (i.e., the ground-truth), allowing for some secondary classes (i.e., those visually similar ones to the ground-truth) to be preserved may help to alleviate the risk of over-fitting (Yang et al., 2018) .

We fit Tolerant Logit which consists of confidence distribution on top-k classes and captures inter-class similarities at the image level.

We believe that limited attention should be devoted to top-k classes of the confidence score, rather than strictly fitting the confidence distribution of all classes.

A More Tolerant Teacher Educates Better Students.

Process vs. Result.

In Fig. 1 , we visualize the spatial attention map of a flower and its corresponding adversarial image on ResNet-101 (He et al., 2015) pretrained on ImageNet (Russakovsky et al., 2015) .

The figure suggests that adversarial perturbations, while small in the pixel space, lead to very substantial noise in the attention map of the network.

Whereas the features for the clean image appear to focus primarily on semantically informative content in the image, the attention map for the adversarial image are activated across semantically irrelevant regions as well.

The state of the art adversarial training methods only encourage hard distribution of deep neural networks output (e.g., one-hot vectors (Madry et al., 2017; Tramèr et al., 2017) or logit (Kannan et al., 2018) ) for pairs of clean examples and adversarial counterparts to be similar.

In our opinion, it is not enough to align the difference between the clean examples and adversarial counterparts only at the output layer of the network, and we need to align the attention maps of middle layers of the whole network, e.g.,o uter layer outputs of conv2.x, conv3.x, conv4.x, conv5.x in ResNet-101.

We can't just focus on the result, but also on the process. (Russakovsky et al., 2015) .

(a) is original image and (b) is corresponding adversarial image.

For ResNet-101, which we use exclusively in this paper, we grouped filters into stages as described in (He et al., 2015) .

These stages are conv2.x, conv3.x, conv4.x, conv5.x.

The contributions of this paper are the following:

• We propose a novel regularized adversarial training framework ATLPA : a method that uses Tolerant Logit and encourages attention map for pairs of examples to be similar.

When applied to clean examples and their adversarial counterparts, ATLPA improves accuracy on adversarial examples over adversarial training.

Instead of constraining a hard distribution in adversarial training, Tolerant Logit consists of confidence distribution on top-k classes and captures inter-class similarities at the image level.

• We explain the reason why our ATLPA can improve the robustness of the model from three dimensions: average activations on discriminate parts, the diversity among learned features of different classes and trends of loss landscapes.

• We show that our ATLPA achieves the state of the art defense on a wide range of datasets against strong PGD gray-box and black-box attacks.

Compared with previous work, our work is evaluated under highly challenging PGD attack: the maximum perturbation ∈ {0.25, 0.5} i.e. L ∞ ∈ {0.25, 0.5} with 10 to 200 attack iterations.

To our knowledge, such a strong attack has not been previously explored on a wide range of datasets.

The rest of the paper is organized as follows: in Section 2 related works are summarized, in Section 3 definitions and threat models are introduced, in Section 4 our ATLPA is introduced, in Section 5 experimental results are presented and discussed, and finally in Section 6 the paper is concluded.

2 RELATED WORK evaluate the robustness of nine papers (Buckman et al., 2018; Ma et al., 2018; Guo et al., 2017; Dhillon et al., 2018; Xie et al., 2017; Song et al., 2017; Samangouei et al., 2018; Madry et al., 2017; Na et al., 2017) accepted to ICLR 2018 as non-certified white-box-secure defenses to adversarial examples.

They find that seven of the nine defenses use obfuscated gradients, a kind of gradient masking, as a phenomenon that leads to a false sense of security in defenses against adversarial examples.

Obfuscated gradients provide a limited increase in robustness and can be broken by improved attack techniques they develop.

The only defense they observe that significantly increases robustness to adversarial examples within the threat model proposed is adversarial training (Madry et al., 2017) .

Adversarial training (Goodfellow et al., 2014; Madry et al., 2017; Kannan et al., 2018; Tramèr et al., 2017; Pang et al., 2019) defends against adversarial perturbations by training networks on adversarial images that are generated on-the-fly during training.

For adversarial training, the most relevant work to our study is (Kannan et al., 2018) , which introduce a technique they call Adversarial Logit Pairing(ALP), a method that encourages logits for pairs of examples to be similar. (Engstrom et al., 2018; Mosbach et al., 2018 ) also put forward different opinions on the robustness of ALP.

Our ATLPA encourages attention map for pairs of examples to be similar.

When applied to clean examples and their adversarial counterparts, ATLPA improves accuracy on adversarial examples over adversarial training. (Araujo et al., 2019) adds random noise at training and inference time, adds denoising blocks to the model to increase adversarial robustness, neither of the above approaches focuses on the attention map.

Following (Pang et al., 2018; Yang et al., 2018; Pang et al., 2019) , we propose Tolerant Logit which consists of confidence distribution on top-k classes and captures inter-class similarities at the image level.

In terms of methodologies, our work is also related to deep transfer learning and knowledge distillation problems, the most relevant work to our study are (Zagoruyko & Komodakis, 2016; Li et al., 2019b) , which constrain the L 2 -norm of the difference between their behaviors (i.e., the feature maps of outer layer outputs in the source/target networks).

Our ATLPA constrains attention map for pairs of clean examples and their adversarial counterparts to be similar.

In this paper, we always assume the attacker is capable of forming untargeted attacks that consist of perturbations of limited L ∞ -norm.

This is a simplified task chosen because it is more amenable to benchmark evaluations.

We consider two different threat models characterizing amounts of information the adversary can have:

• Gray-box Attack We focus on defense against gray-box attacks in this paper.

In a grayback attack, the attacker knows both the original network and the defense algorithm.

Only the parameters of the defense model are hidden from the attacker.

This is also a standard setting assumed in many security systems and applications (Pfleeger & Pfleeger, 2004 ).

• Black-box Attack The attacker has no information about the models architecture or parameters, and no ability to send queries to the model to gather more information.

4.1 ARCHITECTURE Fig.2 represents architecture of ATLPA : a baseline model is adversarial trained so as, not only to make similar the output labels, but to also have similar Tolerant Logits and spatial attention maps to those of original images and adversarial images.

We use adversarial training with Projected Gradient Descent(PGD) (Madry et al., 2017) as the underlying basis for our methods:

wherep data is the underlying training data distribution, L(θ, x + δ, y) is a loss function at data point x which has true class y for a model with parameters θ, and the maximization with respect to δ is approximated using PGD.

In this paper, the loss is defined as:

Figure 2: Schematic representation of ATLPA: a baseline model is adversarial trained so as, not only to make similar the output labels, but to also have similar Tolerant Logits and spatial attention maps to those of original images and adversarial images.

Where L CE is cross entropy,α and β are hyper-parameters which balance Tolerant Logit Loss L T L and Attention Map Loss L AT .

When β=0, we call it ATLPA(w/o ATT), i.e., ATLPA without attention.

Instead of computing an extra loss over all classes just like ALP(Kannan et al., 2018), we pick up a few classes which have been assigned with the highest confidence scores, and assume that these classes are more likely to be semantically similar to the input image.

We use top-k classes of confidence distribution which capture inter-class similarities at the image level.

The logit of model is Z(x),f a k is short for the k-th largest element of Z(x).

Then we can define the following loss:

where w k is non-negative weight, used to adjust the influence of the k-th largest element of Z(x).

In the experiments we use K = 5.

We use Attention Map Loss to encourage the attention map from clean examples and their adversarial counterparts to be similar to each other.

Let also I denote the indices of all activation layer pairs for which we want to pay attention.

Then we can define the following total loss: are respectively the j-th pair of clean examples and their adversarial counterparts attention maps in vectorized form, and p refers to norm type (in the experiments we use p = 2).

F sums absolute values of attention maps raised to the power of p.

To evaluate the effectiveness of our defense strategy, we performed a series of image-classification experiments on 17 Flower Category Database (Nilsback & Zisserman, 2006) and BMW-10 Database.

Following Dubey et al., 2019) , we assume an adversary that uses the state of the art PGD adversarial attack method.

We consider untargeted attacks when evaluating under the gray and black-box settings; untargeted attacks are also used in our adversarial training.

We evaluate top-1 accuracy on validation images that are adversarially perturbed by the attacker.

In this paper, adversarial perturbation is considered under L ∞ norm (i.e., maximum perturbation for each pixel), with an allowed maximum value of .

The value of is relative to the pixel intensity scale of 256, we use = 64/256 = 0.25 and = 128/256 = 0.5.

PGD attacker with 10 to 200 attack iterations and step size α = 1.0/256 = 0.0039.

Our baselines are ResNet-101/152.

There are four groups of convolutional structures in the baseline model, which are described as conv2 x, conv3 x,conv4 x and conv5 x in (He et al., 2015)

We performed a series of image-classification experiments on a wide range of datasets.

Compared with data sets with very small image size e.g., MNIST is 28 * 28,CIFAR-10 is 32 * 32, the image size of our data sets is closer to the actual situation.

All the images are resized to 256 * 256 and normalized to zero mean for each channel, following with data augmentation operations of random mirror and random crop to 224 * 224.

• 17 Flower Category Database (Nilsback & Zisserman, 2006) contains images of flowers belonging to 17 different categories.

The images were acquired by searching the web and taking pictures.

There are 80 images for each category.

We use only classification labels during training.

While part location annotations are used in a quantitative evaluation of show cases, to explain the effect of our algorithm.

• BMW-10 dataset (Krause et al., 2013) contains 512 images of 10 BMW sedans.

The data is split into 360 training images and 152 testing images, where each class has been split roughly in a 70-30 split.

To perform image classification, we use ResNet-101/152 that were trained on our data sets.

We consider two different attack settings: (1) a gray-box attack setting in which the model used to generate the adversarial images is the same as the image-classification model, viz.

the ResNet-101; and (2) a black-box attack setting in which the adversarial images are generated using the ResNet-152 model; The backend prediction model of gray-box and black-box is ResNet-101 with different implementations of the state of the art defense methods,such as IGR (Ross & Doshi-Velez, 2017) , PAT (Madry et al., 2017) , RAT (Araujo et al., 2019) , Randomization (Xie et al., 2017) , ALP(Kannan et al., 2018) , and FD .

ALL the defence methods are all trained under the same adversarial training parameters: batch size is 16, iteration number is 6000, learning rate is 0.01, the ratio of original images and adversarial images is 1:1, under 2-iteration PGD attack, step size is 0.125.

Ensemble learning among different algorithms and models (Tramèr et al., 2017; Pang et al., 2019; Raff et al., 2019 ) is good idea,but here we only consider the use of one single algorithm and one single model.

The hyper-parameters settings of the above algorithms use the default values provided in their papers.

We will open source our code implementation if this paper is accepted.

Here, we first present results with ATLPA on 17 Flower Category Database.

Compared with previous work, (Kannan et al., 2018) was evaluated under 10-iteration PGD attack and = 0.0625,our work are evaluated under highly challenging PGD attack: the maximum perturbation ∈ {0.25, 0.5} i.e. L ∞ ∈ {0.25, 0.5} with 10 to 200 attack iterations.

The bigger the value of , the bigger the disturbance, the more significant the adversarial image effect is.

To our knowledge, such a strong attack has not been previously explored on a wide range of datasets.

As shown in Fig.3 that our ATLPA outperform the state-of-the-art in adversarial robustness against highly challenging gray-box and black-box PGD attacks.

Table 1 shows Main Result of our work:under strong 200-iteration PGD gray-box and blackbox attacks,our ATLPA outperform the state-of-the-art in adversarial robustness on all these databases.

For example, under strong 200-iteration PGD gray-box and black-box attacks on BMW-10 Database where prior art has 35% and 36% accuracy, our method achieves 61% and 62%.

The maximum perturbation is ∈ {0.25, 0.5}. Our ATLPA(purple line) outperform the state-of-theart in adversarial robustness against highly challenging gray-box and black-box PGD attacks.

Even our ATLPA(w/o ATT) does well, which is red line.

ATLPA(w/o ATT): ATLPA without Attention.

We visualized activation attention maps for defense against PGD attacks.

Baseline model is ResNet-101 (He et al., 2015) , which is pre-trained on ImageNet (Russakovsky et al., 2015) and fine-tuned on 17 Flower Category Database.

We found from APPENDIX (Fig. 5) that has a higher level of activation on the whole flower,compared with other defence methods.

To further understand the effect, we compared average activations on discriminate parts of 17 Flower Category Database for different defense methods.

17 Flower Category Database defined discriminative parts of flowers.

So for each image, we got several key regions which are very important to discriminate its category.

Using all testing examples of 17 Flower Category Database, we calculated normalized activations on these key regions of these different defense methods.

As shown in Table 2 , ATLPA got the highest average activations on those key regions, demonstrating that ATLPA

No Defence 0 0 15 10 IGR(Ross & Doshi-Velez, 2017) 10 3 17 10 PAT (Madry et al., 2017) 55 34 57 39 RAT (Araujo et al., 2019) 54 30 57 32 Randomization (Xie et al., 2017) focused on more discriminate features for flowers recognition.

In addition, the score of ATLPA is more bigger than ATLPA(w/o ATT), so it can be seen that the main factor is our Attention.

No Defense 0.10 0.10 0.15 0.15 ALP (Kannan et al., 2018) 0.15 0.15 0.14 0.14 IGR (Ross & Doshi-Velez, 2017) 0.14 0.14 0.13 0.13 PAT (Madry et al., 2017) 0.17 0.17 0.15 0.15 RAT (Araujo et al., 2019) 0

Previous work has shown that for a single network, promoting the diversity among learned features of different classes can improve adversarial robustness (Pang et al., 2018; .

As shown in AP-PENDIX (Fig. 7) ,the ATLPA and ATLPA(w/o ATT) training procedure conceals normal examples on low-dimensional manifolds in the final-layer hidden space.

Then the detector allowable regions can also be set low-dimensional as long as the regions contain all normal examples.

Therefore the white-box adversaries who intend to fool our detector have to generate adversarial examples with preciser calculations and larger noises.

To further understand the effect,we compute silhouette score (Rousseeuw, 1999) of the final hidden features of different defense after t-SNE (Laurens & Hinton, 2008) .

The range of silhouette score is [−1, 1].

The closer the samples of the same category are, the farther the samples of different categories are, the higher the score is.

We compute the silhouette score to quantify the quality of diversity among learned features of different classes.

As shown in Table 3 , ATLPA got the highest silhouette score, demonstrating that ATLPA promotes the diversity among learned features of different classes.

In addition, the scores of ATLPA and ATLPA(w/o ATT) are very close, so it can be seen that the main factor is our Tolerant Logit.

We generate loss plots by varying the input to the models, starting from an original input image chosen from the testing set of 17 Flower Category Database.

The z axis represents the loss.

If x is the original input, then we plot the loss varying along the space determined by two vectors: r1 = sign( x f (x)) and r2 ∼ Rademacher(0.5).

We thus plot the following function: z = loss(x · r1 + y · r2).

As shown in Fig. 4 ,the input varies in the same range and the landscape of our ATLPA varies in the smallest range, our ATLPA has better robustness.

In this paper, we propose a novel regularized adversarial training framework ATLPA a method that uses Tolerant Logit which consists of confidence distribution on top-k classes and captures inter-class similarities at the image level, and encourages attention map for pairs of examples to be similar.

We show that our ATLPA achieves the state of the art defense on a wide range of datasets against strong PGD gray-box and black-box attacks.

We explain the reason why our ATLPA can improve the robustness of the model from three dimensions: average activations on discriminate parts, the diversity among learned features of different classes and trends of loss landscapes.

The results of visualization and quantitative calculation show that our method is helpful to improve the robustness of the model.

17 Flower Category Database

@highlight

In this paper, we propose a novel regularized adversarial training framework ATLPA,namely Adversarial Tolerant Logit Pairing with Attention.