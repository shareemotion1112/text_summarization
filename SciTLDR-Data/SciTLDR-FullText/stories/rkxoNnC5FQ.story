Deep Learning for Computer Vision depends mainly on the source of supervision.

Photo-realistic simulators can generate large-scale automatically labeled synthetic data, but introduce a domain gap negatively impacting performance.

We propose a new unsupervised domain adaptation algorithm, called SPIGAN, relying on Simulator Privileged Information (PI) and Generative Adversarial Networks (GAN).

We use internal data from the simulator as PI during the training of a target task network.

We experimentally evaluate our approach on semantic segmentation.

We train the networks on real-world Cityscapes and Vistas datasets, using only unlabeled real-world images and synthetic labeled data with z-buffer (depth) PI from the SYNTHIA dataset.

Our method improves over no adaptation and state-of-the-art unsupervised domain adaptation techniques.

Learning from as little human supervision as possible is a major challenge in Machine Learning.

In Computer Vision, labeling images and videos is the main bottleneck towards achieving large scale learning and generalization.

Recently, training in simulation has shown continuous improvements in several tasks, such as optical flow BID32 , object detection BID31 BID52 BID47 BID36 , tracking BID10 , pose and viewpoint estimation BID44 BID34 BID46 , action recognition BID9 , and semantic segmentation BID15 BID39 BID38 .

However, large domain gaps between synthetic and real domains remain as the main handicap of this type of strategies.

This is often addressed by manually labeling some amount of real-world target data to train the model on mixed synthetic and real-world labeled data (supervised domain adaptation).

In contrast, several recent unsupervised domain adaptation algorithms have leveraged the potential of Generative Adversarial Networks (GANs) BID14 for pixel-level adaptation in this context BID1 BID45 .

These methods often use simulators as black-box generators of (x, y) input / output training samples for the desired task.

Our main observation is that simulators internally know a lot more about the world and how the scene is formed, which we call Privileged Information (PI).

This Privileged Information includes physical properties that might be useful for learning.

This additional information z is not available in the real-world and is, therefore, generally ignored during learning.

In this paper, we propose a novel adversarial learning algorithm, called SPIGAN, to leverage Simulator PI for GAN-based unsupervised learning of a target task network from unpaired unlabeled real-world data.

We jointly learn four different networks: (i) a generator G (to adapt the pixel-level distribution of synthetic images to be more like real ones), (ii) a discriminator D (to distinguish adapted and real images), (iii) a task network T (to predict the desired label y from image x), and (iv) a privileged network P trained on both synthetic images x and adapted ones G(x) to predict their associated privileged information z. Our main contribution is a new method to leverage PI from a simulator via the privileged network P , which acts as an auxiliary task and regularizer to the task network T , the main output of our SPIGAN learning algorithm.

We evaluate our approach on semantic segmentation in urban scenes, a challenging real-world task.

We use the standard Cityscapes BID6 and Vistas BID33 datasets as target real-world data (without using any of the training labels) and SYNTHIA BID39 as simulator output.

Although our method applies to any kind of PI that can be predicted via a deep network (optical flow, instance segmentation, object detection, material properties, forces, ...), we consider one of the most common and simple forms of PI available in any simulator: depth from its z-buffer.

We show that SPIGAN can successfully learn a semantic segmentation network T using no real-world labels, partially bridging the sim-to-real gap (see Figure 1 ).

SPIGAN also outperforms related state-of-the-art unsupervised domain adaptation methods.

The rest of the paper is organized as follows.

Section 2 presents a brief review of related works.

Section 3 presents our SPIGAN unsupervised domain adaptation algorithm using simulator privileged information.

We report our quantitative experiments on semantic segmentation in Section 4, and conclude in Section 5.

Domain adaptation (cf.

BID7 for a recent review) is generally approached either as domaininvariant learning BID19 BID17 BID11 or as a statistical alignment problem BID50 BID28 .

Our work focuses on unsupervised adaptation methods in the context of deep learning.

This problem consists in learning a model for a task in a target domain (e.g., semantic segmentation of real-world urban scenes) by combining unlabeled data from this domain with labeled data from a related but different source domain (e.g., synthetic data from simulation).

The main challenge is overcoming the domain gap, i.e. the differences between the source and target distributions, without any supervision from the target domain.

The Domain Adversarial Neural Network (DANN) BID50 BID11 BID12 ) is a popular approach that learns domain invariant features by maximizing domain confusion.

This approach has been successfully adopted and extended by many other researchers, e.g., BID37 ; .

Curriculum Domain Adaptation ) is a recent evolution for semantic segmentation that reduces the domain gap via a curriculum learning approach (solving simple tasks first, such as global label distribution in the target domain).Recently, adversarial domain adaptation based on GANs BID14 have shown encouraging results for unsupervised domain adaptation directly at the pixel level.

These techniques learn a generative model for source-to-target image translation, including from and to multiple domains BID48 BID45 BID24 .

In particular, CycleGAN leverages cycle consistency using a forward GAN and a backward GAN to improve the training stability and performance of image-to-image translation.

An alternative to GAN is Variational Auto-Encoders (VAEs), which have also been used for image translation .Several related works propose GAN-based unsupervised domain adaptation methods to address the specific domain gap between synthetic and real-world images.

SimGAN BID45 leverages simulation for the automatic generation of large annotated datasets with the goal of refining synthetic images to make them look more realistic.

Sadat BID40 effectively leverages synthetic data by treating foreground and background in different manners.

Similar to our approach, xs (synthetic) ys (label) zs (privileged info.)

DISPLAYFORM0 Figure 2: SPIGAN learning algorithm from unlabeled real-world images x r and the unpaired output of a simulator (synthetic images x s , their labels y s , e.g. semantic segmentation ground truth, and Privileged Information PI z s , e.g., depth from the z-buffer) modeled as random variables.

Four networks are learned jointly: (i) a generator G(x s ) ∼ x r , (ii) a discriminator D between G(x s ) = x f and x r , (iii) a perception task network T (x r ) ∼ y r , which is the main target output of SPIGAN (e.g., a semantic segmentation deep net), and (iv) a privileged network P to support the learning of T by predicting the simulator's PI z s .recent methods consider the final recognition task during the image translation process.

Closely related to our work, PixelDA BID1 is a pixel-level domain adaptation method that jointly trains a task classifier along with a GAN using simulation as its source domain but no privileged information.

These approaches focus on simple tasks and visual conditions that are easy to simulate, hence having a low domain gap to begin with.

On the other hand, BID21 are the first to study semantic segmentation as the task network in adversarial training. uses a curriculum learning style approach to reduce domain gap.

BID41 conducts domain adaptation by utilizing the task-specific decision boundaries with classifiers.

BID42 leverage the GAN framework by learning general representation shared between the generator and segmentation networks.

BID5 use a target guided distillation to encourage the task network to imitate a pretrained model.

BID57 propose to combine appearance and representation adaptation.

BID49 propose an adversarial learning method to adapt in the output (segmentation) space.

BID59 generates pseudo-labels based on confidence scores with balanced class distribution and propose an iterative self-training framework.

Our main novelty is the use of Privileged Information from a simulator in a generic way by considering a privileged network in our architecture (see Figure 2 ).

We show that for the challenging task of semantic segmentation of urban scenes, our approach significantly improves by augmenting the learning objective with our auxiliary privileged task, especially in the presence of a large sim-to-real domain gap, the main problem in challenging real-world conditions.

Our work is inspired by Learning Using Privileged Information (LUPI) BID51 , which is linked to distillation BID18 as shown by BID29 .

LUPI's goal is to leverage additional data only available at training time.

For unsupervised domain adaptation from a simulator, there is a lot of potentially useful information about the generation process that could inform the adaptation.

However, that information is only available at training time, as we do not have access to the internals of the real-world data generator.

Several works have used privileged information at training time for domain adaptation BID2 BID20 BID2 BID43 BID13 .

BID20 leverage RGBD information to help adapt an object detector at the feature level, while BID13 propose a similar concept of modality distillation for action recognition.

Inspired by this line of work, we exploit the privileged information from simulators for sim-to-real unsupervised domain adaptation.

Our goal is to design a procedure to learn a model (neural network) that solves a perception task (e.g., semantic segmentation) using raw sensory data coming from a target domain (e.g., videos of a car driving in urban environments) without using any ground truth data from the target domain.

We formalize this problem as unsupervised domain adaptation from a synthetic domain (source domain) to a real domain (target domain).

The source domain consists of labeled synthetic images together with Privileged Information (PI), obtained from the internal data structures of a simulator.

The target domain consists of unlabeled images.

The simulated source domain serves as an idealized representation of the world, offering full control of the environment (weather conditions, types of scene, sensor configurations, etc.) with automatic generation of raw sensory data and labels for the task of interest.

The main challenge we address in this work is how to overcome the gap between this synthetic source domain and the target domain to ensure generalization of the task network in the real-world without target supervision.

Our main hypothesis is that the PI provided by the simulator is a rich source of information to guide and constrain the training of the target task network.

The PI can be defined as any information internal to the simulator, such as depth, optical flow, or physical properties about scene components used during simulation (e.g., materials, forces, etc.).

We leverage the simulator's PI within a GAN framework, called SPIGAN.

Our approach is described in the next section.

DISPLAYFORM0 s ), i = 1 . . .

N s } be a set of N s simulated images x s with their labels y s and PI z s .

We describe our approach assuming a unified treatment of the PI, but our method trivially extends to multiple separate types of PI.

DISPLAYFORM1 , and (iv) a privileged network P (x; θ P ).

The generator G is a mapping function, transforming an image x s in X s (source domain) to x f in X f (adapted or fake domain).

SPIGAN aims to make the adapted domain statistically close to the target domain to maximize the accuracy of the task predictor T (x; θ T ) during testing.

The discriminator D is expected to tell the difference between x f and x r , playing an adversarial game with the generator until a termination criteria is met (refer to section 4.1) .

The target task network T is learned on the synthetic x s and adapted G(x s ; θ G ) images to predict the synthetic label y s , assuming the generator presents a reasonable degree of label (content) preservation.

This assumption is met for the regime of our experiments.

Similarly, the privileged network P is trained on the same input but to predict the PI z, which in turn assumes the generator G is also PI-preserving.

During testing only T (x; θ T ) is needed to do inference for the selected perception task.

The main learning goal is to train a model θ T that can correctly perform a perception task T in the target real-world domain.

All models are trained jointly in order to exploit all available information to constrain the solution space.

In this way, the PI provided by the privileged network P is used to constrain the learning of T and to encourage the generator to model the target domain while being label-and PI-preserving.

Our joint learning objective is described in the following section.

We design a consistent set of loss functions and domain-specific constraints related to the main prediction task T .

We optimize the following minimax objective: min DISPLAYFORM0 where α, β, γ, δ are the weights for adversarial loss, task prediction loss, PI regularization, and perceptual regularization respectively, further described below.

Adversarial loss L GAN .

Instead of using a standard adversarial loss, we use a least-squares based adversarial loss BID30 ; , which stabilizes the training process and generates better image results in our experiments: DISPLAYFORM1 where P r (resp.

P s ) denotes the real-world (resp.

synthetic) data distribution.

Task prediction loss L T .

We learn the task network by optimizing its loss over both synthetic images x s and their adapted version G(x s , θ G ).

This assumes the generator is label-preserving, i.e., that y s can be used as a label for both images.

Thanks to our joint objective, this assumption is directly encouraged during the learning of the generator through the joint estimation of θ P , which relates to scene properties captured by the PI.

Naturally, different tasks require different loss functions.

In our experiments, we consider the task of semantic segmentation and use the standard cross-entropy loss (Eq. 4) over images of size W × H and a probability distribution over C semantic categories.

The total combined loss in the special case of semantic segmentation is therefore: DISPLAYFORM2 DISPLAYFORM3 where 1 [a=b] is the indicator function.

PI regularization L P .

Similarly, the auxiliary task of predicting PI also requires different losses depending on the type of PI.

In our experiments, we use depth from the z-buffer and an 1 -norm: DISPLAYFORM4 Perceptual regularization L perc .

To maintain the semantics of the source images in the generated images, we additionally use the perceptual loss BID23 ; BID3 : DISPLAYFORM5 where φ is a mapping from image space to a pre-determined feature space Chen & Koltun (2017) (see 4.1 for more details).Optimization.

In practice, we follow the standard adversarial training strategy to optimize our joint learning objective (Eq. 1).

We alternate between updates to the parameters of the discriminator θ D , keeping all other parameters fixed, then fix θ D and optimize the parameters of the generator θ G , the privileged network θ P , and most importantly the task network θ T .

We discuss the details of our implementation, including hyper-parameters, in section 4.1.

We evaluate our unsupervised domain adaptation method on the task of semantic segmentation in a challenging real-world domain for which training labels are not available.

As our source synthetic domain, we select the public SYNTHIA dataset BID39 as synthetic source domain given the availability of automatic annotations and PI.

SYNTHIA is a dataset generated from an autonomous driving simulator of urban scenes.

These images were generated under different weathers and illumination conditions to maximize visual variability.

Pixel-wise segmentation and depth labels are provided for each image.

In our experiment, we use the sequence of SYNTHIA-RAND-CITYSCAPES, which contains semantic segmentation labels that are more compatible with Cityscapes.

For target real-world domains, we use the Cityscapes BID6 and Mapillary Vistas BID33 datasets.

Cityscapes is one of most widely used real-world urban scene image segmentation datasets with images collected around urban streets in Europe.

For this dataset, We use the standard split for training and validation with 2, 975 and 500 images respectively.

Mapillary Vistas is a larger dataset with a wider variety of scenes, cameras, locations, weathers, and illumination conditions.

We use 16, 000 images for training and 2, 000 images for evaluation.

During training, none of the labels from the real-world domains are used.

In our experiment, we first evaluate adaptation from SYNTHIA to Cityscapes on 16 classes, following the standard evaluation protocol used in Hoffman et al. of using PI by conducting ablation study with and without PI (depth) during adaptation from SYN-THIA to both Cityscapes and Vistas, on a common 7 categories ontology.

To be consistent with the semantic segmentation best practices, we use standard intersection-over-union (IoU) per category and mean intersection-over-union (mIoU) as our main validation metric.

We adapt the generator and discriminator model architectures from CycleGAN and BID23 .

For simplicity, we use a single sim-to-real generator (no cycle consistency) consisting of two down-sampling convolution layers, nine ResNet blocks BID16 and two fractionally-strided convolution layers.

Our discriminator is a PatchGAN network with 3 layers.

We use the standard FCN8s architecture BID28 for both the task predictor T and the privileged network P , given its ease of training and its acceptance in domain adaptation works BID21 .

For the perceptual loss L perc , we follow the implementation in BID3 .

The feature is constructed by the concatenation of the activations of a pre-trained VGG19 network BID53 of layers "conv1 2", "conv2 2", "conv3 2", "conv4 2", "conv5 2".

FORMULA3 , we set hyper-parameters using a coarse grid search on a small validation set different than the target set.

For Cityscapes, we use a subset of the validation set of Vistas, and vice-versa.

We found a set of values that are effective across datasets and experiments, which show they have a certain degree of robustness and generalization.

The weights in our joint adversarial loss (Eq. 1) are set to α = 1, β = 0.5, γ = 0.1, δ = 0.33, for the GAN, task, privileged, and perceptual objectives respectively.

This confirms that the two most important factors in the objective are the GAN and task losses (α = 1, β = 0.5).

This is intuitive, as the goal is to improve the generalization performance of the task network (the task loss being an empirical proxy) across a potentially large domain gap (addressed first and foremost by the GAN loss).

The regularization terms are secondary in the objective, stabilizing the training (perceptual loss) and constraining the adaptation process (privileged loss).

FIG1 show an example of our loss curves and the stability of our training.

Another critical hyper-parameter for unsupervised learning is the stopping criterion.

We observed that the stabilizing effects of the task and privileged losses (Eqs. 3,5) on the GAN objective (Eq. 2) made a simple rule effective for early stopping.

We stop training at the iteration when the discriminator loss is significantly and consistently better than the generator loss (iteration 90 in FIG2 ).

This is inspired by the semi-supervised results of BID8 , where effective discriminative adaptation of the task network might not always be linked to the best image generator.

We evaluate the methods with two resolutions: 320 × 640 and 512 × 1024, respectively.

Images are resized to the evaluated size during training and evaluation.

During training, we sample crops of size 320 × 320 (resp.

400 × 400) for lower (resp.

higher) resolution experiments.

In all adversarial learning cases, we do five steps of the generator for every step of the other networks.

The Adam optimizer BID25 ) is used to adjust all parameters with initial learning rate 0.0002 in our PyTorch implementation BID35 .

n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a 23.2 LSD n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a 34.

Table 2 : Semantic Segmentation results (per category and mean IoUs, higher is better) for SYN-THIA adapting to Cityscapes and Vistas.

The last column is the ratio of images in the validation set for which we observe negative transfer (lower is better).

In this section we present our evaluation of the SPIGAN algorithm in the context of adapting a semantic segmentation network from SYNTHIA to Cityscapes.

Depth maps from SYNTHIA are used as PI in the proposed algorithm.

We compare our results to several state-of-art domain adaptation algorithms, including FCNs in the wild (FCNs wild) BID21 , Curriculum DA (CDA) , Learning from synthetic data (LSD) BID42 , and Class-balanced Self-Training (CBST) BID59 .Quantitative results for these methods are shown in Table 1 for the semantic segmentation task on the target domain of Cityscapes (validation set).

As reference baselines, we include results training only on source images and non-adapted labels.

We also provide our algorithm performance without the PI for comparison (i.e., γ = 0 in Eq. 1, named "SPIGAN-no-PI").Results show that on Cityscapes SPIGAN achieves state-of-the-art semantic segmentation adaptation in terms of mean IoU. A finer analysis of the results attending to individual classes suggests that the use of PI helps to estimate layout-related classes such as road and sidewalk and object-related classes such as person, rider, car, bus and motorcycle.

SPIGAN achieves an improvement of 3% in 320 × 640, 1.0% in 512 × 1024, in mean IoU with respect to the non-PI method.

This improvement is thanks to the regularization provided by P (x; θ P ) during training, which decreases the number of artifacts as shown in Figure 5 .

This comparison, therefore, confirms our main contribution: a general approach to leveraging synthetic data and PI from the simulator to improve generalization performance across the sim-to-real domain gap.

To better understand the proposed algorithm, and the impact of PI, we conduct further experiments comparing SPIGAN (with PI), SPIGAN-no-PI (without PI), and SPIGAN-base (without both PI and perceptual regularization), the task network of SPIGAN trained only on the source domain (FCN source, lower bound, no adaptation), and on the target domain (FCN target, upper bound) , all at 320 × 640 resolution.

We also include results on the Vistas dataset, which presents a more challenging adaptation problem due to the higher diversity of its images.

For these experiments, we use a 7 semantic classes ontology to produce a balanced ontology common to the three datasets (SYNTHIA, Cityscapes and Vistas).

Adaptation results for both target domains are given in Table 2 .In addition to the conventional segmentation performance metrics, we also carried out a study to measure the amount of negative transfer, summarized in Table 2 .

A negative transfer case is defined as a real-world testing sample that has a mIoU lower than the FCN source prediction (no adaptation).As shown in Table 2 , SPIGAN-no-PI, including perceptual regularization, performs better than SPIGAN-base in both datasets.

The performance is generally improved in all categories, which implies that perceptual regularization effectively stabilizes the adaptation during training.

For Cityscapes, the quantitative results in Table 2 show that SPIGAN is able to provide dramatic adaptation as hypothesized.

SPIGAN improves the mean IoU by 17.1%, with the PI itself providing an improvement of 7.4%.

This is consistent with our observation in the previous experiment (Table 1).

We also notice that SPIGAN gets significant improvements on "nature", "construction", and "vehicle" categories.

In addition, SPIGAN is able to improve the IoU by +15% on the "human" category, a difficult class in semantic segmentation.

We provide examples of qualitative results for the adaptation from SYNTHIA to Cityscapes in Figure 5 and Figure 7 .On the Vistas dataset, SPIGAN is able to decrease the domain gap by +4.3% mean IoU.

In this case, using PI is crucial to improve generalization performance.

SPIGAN-no-PI indeed suffers from negative transfer, with its adapted network performing −13% worse than the FCN source without adaptation.

Table 2 shows that 80% of the evaluation images have a lower individual IoU after adaptation in the SPIGAN-no-PI case (vs. 42% in the SPIGAN case).The main difference between the Cityscapes and Vistas results is due to the difference in visual diversity between the datasets.

Cityscapes is indeed a more visually uniform benchmark than Vistas: it was recorded in a few German cities in nice weather, whereas Vistas contains crowdsourced data from all over the world with varying cameras, environments, and weathers.

This makes Cityscapes more amenable to image translation methods (including SPIGAN-no-PI), as can be seen in Figure 5 where a lot of the visual adaptation happens at the color and texture levels, whereas Figure 6 shows that SYNTHIA images adapted towards Vistas contain a lot more artifacts.

Furthermore, a larger domain gap is known to increase the risk of negative transfer (cf.

BID7 ).

This is indeed what we quantitatively measured in Table 2 and qualitatively confirmed in Figure 6 .

SPIGAN suffers from similar but less severe artifacts.

As shown in Figure 6 , they are more consistent with the depth of the scene, which helps addressing the domain gap and avoids the catastrophic failures visible in the SPIGAN-no-PI case.

This consistent improvement brought by PI in both of the experiments not only shows that PI imposes useful constraints that promote better task-oriented training, but also implies that PI more robustly guides the training to reduce domain shift.

By comparing the results on the two different datasets, we also found that all the unsupervised adaptation methods share some similarity in the performance of certain categories.

For instance, the "vehicle" category has seen the largest improvement for both Cityscapes and Vistas.

This trend is consistent with the well-known fact that "object" categories are easier to adapt than "stuff" BID52 .

However, the same improvement did not appear in the "human" category mainly because the SYNTHIA subset we used in our experiments contains very few humans.

This phenomenon has been recently studied in Sadat BID40 .

We present SPIGAN, a novel method for leveraging synthetic data and Privileged Information (PI) available in simulated environments to perform unsupervised domain adaptation of deep networks.

Our approach jointly learns a generative pixel-level adaptation network together with a target task network and privileged information models.

We showed that our approach is able to address large domain gaps between synthetic data and target real-world domains, including for challenging realworld tasks like semantic segmentation of urban scenes.

For future work, we plan to investigate SPIGAN applied to additional tasks, with different types of PI that can be obtained from simulation.

@highlight

An unsupervised sim-to-real domain adaptation method for semantic segmentation using privileged information from a simulator with GAN-based image translation.