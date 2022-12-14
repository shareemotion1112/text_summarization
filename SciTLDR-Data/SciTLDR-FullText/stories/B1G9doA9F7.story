Training a model to perform a task typically requires a large amount of data from the domains in which the task will be applied.

However, it is often the case that data are abundant in some domains but scarce in others.

Domain adaptation deals with the challenge of adapting a model trained from a data-rich source domain to perform well in a data-poor target domain.

In general, this requires learning plausible mappings between domains.

CycleGAN is a powerful framework that efficiently learns to map inputs from one domain to another using adversarial training and a cycle-consistency constraint.

However, the conventional approach of enforcing cycle-consistency via reconstruction may be overly restrictive in cases where one or more domains have limited training data.

In this paper, we propose an augmented cyclic adversarial learning model that enforces the cycle-consistency constraint via an external task specific model, which encourages the preservation of task-relevant content as opposed to exact reconstruction.

We explore digit classification in a low-resource setting in supervised, semi and unsupervised situation, as well as high resource unsupervised.

In low-resource supervised setting, the results show that our approach improves absolute performance by 14% and 4% when adapting SVHN to MNIST and vice versa, respectively, which outperforms unsupervised domain adaptation methods that require high-resource unlabeled target domain.

Moreover, using only few unsupervised target data, our approach can still outperforms many high-resource unsupervised models.

Our model also outperforms on USPS to MNIST and synthetic digit to SVHN for high resource unsupervised adaptation.

In speech domains, we similarly adopt a speech recognition model from each domain as the task specific model.

Our approach improves absolute performance of speech recognition by 2% for female speakers in the TIMIT dataset, where the majority of training samples are from male voices.

Domain adaptation BID18 BID36 BID1 aims to generalize a model from source domain to a target domain.

Typically, the source domain has a large amount of training data, whereas the data are scarce in the target domain.

This challenge is typically addressed by learning a mapping between domains, which allows data from the source domain to enrich the available data for training in the target domain.

One way of learning such mappings is through Generative Adversarial Networks (GANs BID8 with cycle-consistency constraint (CycleGAN Zhu et al., 2017) , which enforces that mapping of an example from the source to the target and then back to the source domain would result in the same example (and vice versa for a target example).

Due to this constraint, CycleGAN learns to preserve the 'content' 1 from the source domain while only transferring the 'style' to match the distribution of the target domain.

This is a powerful constraint, and various works BID37 BID12 have demonstrated its effectiveness in learning cross domain mappings.

Enforcing cycle-consistency is appealing as a technique for preserving semantic information of the data with respect to a task, but implementing it through reconstruction may be too restrictive when data are imbalanced across domains.

This is because the reconstruction error encourages exact match of samples from the reverse mapping, which may in turn encourage the forward-mapping to keep the sample close to the original domain.

Normally, the adversarial objectives would counter this effect; however, when data from the target domain are scarce, it is very difficult to learn a powerful discriminator that can capture meaningful properties of the target distribution.

Therefore, the resulting mappings learned is likely to be sub-optimal.

Importantly, for the learned mapping to be meaningful, it is not necessary to have the exact reconstruction.

As long as the 'semantic' information is preserved and the 'style' matches the corresponding distribution, it would be a valid mapping.

To address this issue, we propose an augmented cyclic adversarial learning model (ACAL) for domain adaptation.

In particular, we replace the reconstruction objective with a task specific model.

The model learns to preserve the 'semantic' information from the data samples in a particular domain by minimizing the loss of the mapped samples for the task specific model.

On the other hand, the task specific model also serves as an additional source of information for the corresponding domain and hence supplements the discriminator in that domain to facilitate better modeling of the distribution.

The task specific model can also be viewed as an implicit way of disentangling the information essential to the task from the 'style' information that relates to the data distribution of different domain.

We show that our approach improves the performance by 40% as compared to the baseline on digit domain adaptation.

We improve the phoneme error rate by ??? 5% on TIMIT dataset, when adapting the model trained on one speech from one gender to the other.

Our work is broadly related to domain adaptation using neural networks for both supervised and unsupervised domain adaptation.

Supervised Domain Adaptation When labels are available in the target domain, a common approach is to utilize the label information in target domain to minimize the discrepancy between source and target domain BID15 BID33 BID7 BID6 .

For example, BID15 applies the marginal Fisher analysis criteria and Maximum Mean Discrepancy (MMD) to minimize the distribution difference between source and target domain.

BID33 proposed to add a domain classifier that predicts domain label of the inputs, with a domain confusion loss.

BID7 leverages attributes by using attribute and class level classification loss with attribute consistent loss to fine-tune the target model.

Our method also employs models from both domains, however, our models are used to assist adversarial learning for better learning of the target domain distribution.

In addition, our final model for supervised domain adaptation is obtained by training on data from target domain as well as the transfered data from the source domain, rather than fine-tuning a source/target domain model.

More recently, various work have taken advantage of the substantial generation capabilities of the GAN framework and applied them to domain adaptation BID23 BID2 BID37 BID34 BID20 BID12 .

However, most of these works focus on high-resource unsupervised domain adaptation, which may be unsuitable for situations where the target domain data are limited.

BID2 uses a GAN to adapt data from the source to target domain while simultaneously training a classifier on both the source and adapted data.

Our method also employs task specific models; however, we use the models to augment the CycleGAN formulation.

We show that having cycles in both directions (i.e. from source to target and vice versa) is important in the case where the target domain has limited data (see sec. 4).

BID34 proposes adversarial discriminative domain adaptation (ADDA), where adversarial learning is employed to match the representation learned from the source and target domain.

Our method also utilizes pre-trained model from source domain, but we only implicitly match the representation distributions rather than explicitly enforcing representational similarity.

Cycle-consistent adversarial domain adaptation (CyCADA Hoffman et al., 2018) is perhaps the most similar work to our own.

This approach uses both 1 and semantic consistency to enforce cycle-consistency.

An important difference in our work is that we also include another cycle that starts from the target domain.

This is important because, if the target domain is of low resource, the adaptation from source to target may fail due to the difficulty in learning a good BID39 .

Middle: Relaxed cycle-consistent model (RCAL), where the cycle-consistency is enforced through task specific models in corresponding domain.

Right: Augmented cycle-consistent model (ACAL).

In addition to the relaxed model, the task specific model is also used to augment the discriminator of corresponding domain to facilitate learning.

In the diagrams x and L denote data and losses, respectively.

We point out that the ultimate goal of our approach is to use the mapped Source ??? Target samples (x S ???T ) to augment the limited data of the target domain (x T ).

discriminator in the target domain.

BID0 also suggests to improve CycleGAN by explicitly enforcing content consistency and style adaptation, by augmenting the cyclic adversarial learning to hidden representation of domains.

Our model is different from recent cyclic adversarial learning, due to implicit learning of content and style representation through an auxiliary task, which is more suitable for low resource domains.

Using classification to assist GAN training has also been explored previously BID31 BID32 BID21 .

BID31 proposed CatGAN, where the discriminator is converted to a multi-class classifier.

We extend this idea to any task specific model, including speech recognition task, and use this model to preserve task specific information regarding the data.

We also propose that the definition of task model can be extended to unsupervised tasks,such as language or speech modeling in domains, meaning augmented unsupervised domain adaptation.

To learn the true data distribution P data (X) in a nonparametric way, BID8 proposed the generative adversarial network (GAN).

In this framework, a discriminator network D(x) learns to discriminate between the data produced by a generator network G(z) and the data sampled from the true data distribution P data (X), whereas the generator models the true data distribution by learning to confuse the discriminator.

Under certain assumptions BID8 , the generator would learn the true data distribution when the game reaches equilibrium.

Training of GAN is in general done by alternately optimizing the following objective for D and G. min DISPLAYFORM0 2.2 CYCLEGAN CycleGAN BID39 extends this framework to multiple domains, P S (X) and P T (X), while learning to map samples back and forth between them.

Adversarial learning is applied such that the result mapping from G S ???T will match the target distribution P T (X), and similarly for the reverse mapping from G T ???S .

This is accomplished by the following adversarial objectives: DISPLAYFORM1 CycleGAN also introduces cycle-consistency, which enforces that each mapping is able to invert the other.

In the original work, this is achieved by including the following reconstruction objective: DISPLAYFORM2 Learning the CycleGAN model involves optimizing a weighted combination of the above objectives 2, 3 and 4.

Enforcing cycle-consistency using a reconstruction objective (e.g. eq. 4) may be too restrictive and potentially results in sub-optimal mapping functions.

This is because the learning dynamics of CycleGAN balance the two contrastive forces.

The adversarial objective encourages the mapping functions to generate samples that are close to the true distribution.

At the same time, the reconstruction objective encourages identity mapping.

Balancing these objectives may works well in the case where both domains have a relatively large number of training samples.

However, problems may arise in case of domain adaptation, where data within the target domain are relatively sparse.

Let P S (X) and P T (X) denote source and target domain distributions, respectively, and samples from P T (X) are limited.

In this case, it will be difficult for the discriminator D T to model the actual distribution P T (X).

A discriminator model with sufficient capacity will quickly overfit and the resulting D T will act like delta function on the sample points from P T (X).

Attempts to prevent this by limiting the capacity or using regularization may easily induce over-smoothing and under-fitting such that the probability outputs of D T are only weakly sensitive to the mapped samples.

In both cases, the influence of the reconstruction objective should begin to outweigh that of the adversarial objective, thereby encouraging an identity mapping.

More generally, even if we are are able to obtain a reasonable discriminator D T , the support of the distribution learned through it would likely to be small due to limited data.

Therefore, the learning signal G S ???T receive from D T would be limited.

To sum up, limited data within P T (X) would make it less likely that the discriminator will encourage meaningful cross domain mappings.

The root of the above issue in domain adaptation is two fold.

First, exact reconstruction is a too strong objective for enforcing cycle-consistency.

Second, learning a mapping function to a particular domain which solely depends on the discriminator for that domain is not sufficient.

To address these two problems, we propose to 1) use a task specific model to enforce the cycle-consistency constraint, and 2) use the same task specific model in addition to the discriminator to train more meaningful cross domain mappings.

In more detail, let M S and M T be the task specific models trained on domains P S (X, Y ) and P T (X, Y ), and L task denotes the task specific loss.

Our cycle-consistent objective is then: DISPLAYFORM0 Here, L task enforces cycle-consistency by requiring that the reverse mappings preserve the semantic information of the original sample.

Importantly, this constraint is less strict than when using reconstruction, because now as long as the content matches that of the original sample, the incurred loss will not increase.

(Some style consistency is implicitly enforced since each model M is trained on data within a particular domain.)

This is a much looser constraint than having consistency in the original data space, and thus we refer to this as the relaxed cycle-consistency objective.

To address the second issue, we augment the adversarial objective with corresponding objective: DISPLAYFORM1 Similar to adversarial training, we optimize the above objective by maximizing D S (D T ) and minimizing G T ???S (G S ???T ) and M S (M T ).

With the new terms, learning of the mapping functions G get assists from both the discriminator and the task specific model.

The task specific model learns to capture conditional probability distribution P S (Y |X) (P T (Y |X)), that also preserves information regarding P S (X) (P T (X)).

This conditional information is different than the information captured through the discriminator D S (D T ).

The difference is that the model is only required to preserve DISPLAYFORM2 end end useful information regarding X respect to predicting Y , for modeling the conditional distribution, which makes learning the conditional model a much easier problem.

In addition, the conditional model mediates the influence of data that the discriminator does not have access to (Y ), which should further assist learning of the mapping functions G T ???S (G S ???T ).In case of unsupervised domain adaptation, when there is no information of target conditional probability distribution P T (Y |X), we propose to use source model DISPLAYFORM3 .

Therefore, proposed model can be extended to unsupervised domain adaptation, with the corresponding modified objectives: DISPLAYFORM4 To further extend this approach to semi-supervised domain adaptation, both supervised and unsupervised objectives for labeled and unlabeled target samples are used interchangeably, as explained in Algorithm 1.

In this section, we evaluate our proposed model on domain adaptation for visual and speech recognition.

We continue the convention of referring to the data domains as 'source' and 'target', where target denotes the domain with either limited or unlabeled training data.

Visual domain adaptation is evaluated using the MNIST dataset (M) BID22 , Street View House Numbers (SVHN) datasets (S) BID26 , USPS (U) BID19 , MNISTM (MM) and Synthetic Digits (SD) BID4 .

Adaptation on speech is evaluated on the domain of gender within the TIMIT dataset BID5 , which contains broadband 16kHz recordings of 6300 utterances (5.4 hours) of phonetically-balanced speech.

The male/female ratio of speakers across train/validation/test sets is approximately 70% to 30%.

Therefore, we treat male speech as the source domain and female speech as the low resource target domain.

To get an idea of the contribution from each component of our model, in this section we perform a series of ablations and present the results in TAB0 .

We perform these ablations by treating SVHN as the source domain and MNIST as the target domain.

We down sample the MNIST training data so only 10 samples per class are available during training, denoted as MNIST-(10), which is only 0.17% of full training data.

The testing performance is calculated on the full MNIST test set.

We use a modified LeNet for all experiments in this ablation.

The Modified LeNet consists of two convolutional layers with 20 and 50 channels, followed by a dropout layer and two fully connected layers of 50 and 10 dimensionality.

There are various ways that one may utilize cycle-consistency or adversarial training to do domain adaptation from components of our model.

One way is to use adversarial training on the target domain to ensure matching of distribution of adapted data, and use the task specific model to ensure the 'content' of the data from the source domain is preserved.

This is the model described in BID2 , except their model is originally unsupervised.

This model is denoted as S ??? T in TAB0 .

It is also interesting to examine the importance of the double cycle, which is proposed in BID39 and adopted in our work.

Theoretically, one cycle would be sufficient to learn the mapping between domains; therefore, we also investigate the performance of one cycle only models, where one direction would be from source to target and then back, and similarly for the other direction.

These models are denoted as (S???T???S)-One Cycle and (T???S???T)-One Cycle in TAB0 , respectively.

To test the effectiveness of the relaxed cycle-consistency (eq. 5) and augmented adversarial loss (eq. 6 and 7), we also test one cycle models while progressively adding these two losses.

Interestingly, the one cycle relaxed and one cycle augmented models are similar to the model proposed in BID12 when their model performs mapping from source to target domain and then back.

The difference is that their model is unsupervised and includes more losses at different levels.

As can be seen from TAB0 , the simple conditional model performed surprisingly well as compared to more complicated cyclic counterparts.

This may be attributed to the reduced complexity, since it only needs to learn one set of mapping.

As expected, the single cycle performance is poor when the target domain is of limited data due to inefficient learning of discriminator in the target domain (see section 3).

When we change the cycle to the other direction, where there are abundant data in the target domain, the performance improves, but is still worse than the simple one without cycle.

This is because the adaptation mapping (i.e. G S ???T ) is only learned via the generated samples from G T ???S , which likely deviate from the real examples in practice.

This observation also suggests that it would be beneficial to have cycles in both directions when applying the cycle-consistency constraint, since then both mappings can be learned via real examples.

The trends get reversed when we are using relaxed implementation of cycle-consistency from the reconstruction error with the task specific losses.

This is because now the power of the task specific model is crucial to preserve the content of the data after the reverse mapping.

When the source domain dataset is sufficiently large, the cycle-consistency is preserved.

As such, the resulting learned mapping functions would preserve meaningful semantics of the data while transferring the styles to the target domain, and vice versa.

In addition, it is clear that augmenting the discriminator with task specific loss is helpful for learning adaptations.

Furthermore, the information added from the task specific model is clearly beneficial for improving the adaptation performance, without this none of the models outperform the baseline model, where no adaptation is performed.

Last but not least, it is also clear from the results that using task specific model improves the overall adaptation performance.

DISPLAYFORM0 To further evaluate the effectiveness of using task-specific loss with two cycles for low-resource unsupervised domain adaptation scenario, we comapre our model with CyCADA BID12 , and when no reconstruction loss is used in CyCADA, referred as "CyCADA (Relaxed)".

The latter resembles the (S ??? T ??? S)-ACAL in TAB0 , but with a different semantic loss.

As shown in FIG1 , CyCADA model and its relaxed variation fail to learn a good adaptation, where target domain contains few unlabaled samples per class.

Additionally, CyCADA models show high instability in low-resource situation.

As described in section 1.1, instability is an expected behvaiour of CyCADA when having limited target data, because the source to target cycle fails to preserve consistency, due to weak target domain discriminator.

However, ACAL model indicates stable and consistent performance, due to proper use of source classifier to enforce consistency, rather than relying on target and source discriminators.

In this section, we experiment on domain adaptation for the task of digit recognition.

In each experiment, we select one domain (MNIST, USPS, MNISTM, SVHN, Synthetic Digits) to be the target.

We conduct three types of domain adaptation, i.e. low-resource supervised, high-resource unsupervised, and low-resource semi-supervised adaptation.

The evaluation results are based on not using any data augmentation.

Low-resource supervised adaptation: In this setting, we sub-sample the target to contain only a few labeled samples per class, and using the other full dataset as the source domain.

In this setting, no unlabeled sample is used.

Comparison with recent low resource domain adaptation, FADA BID25 for MNIST, USPS, and SVHN adaptation is shown in FIG2 .

To provide more baselines, we also compared with model trained only on limited target data, and on combination of both labeled source and limited target domains.

As shown in FIG4 , ACAL outperforms FADA and two other baselines in all adaptations.

High-resource unsupervised adaptation; Here, we use the whole target domain with no label.

Evaluation results on all adaptation directions are presented in TAB1 (Appendix A).

It is evident that ACAL model performance is on par with the state of the art unsupervised approaches, and outperforms on MNIST???USPS and Syn-Digits???SVHN.

It is worth mentioning that Shu et al. (2018) improved their VADA adversarial model using natural gradient as teacher-student training, which is not directly comparable to adversarial approaches.

Moreover, the source-only baseline of BID30 ) is stronger than the reported unsupervised approaches, as well as our baseline.

DISPLAYFORM0 Low-resource semi-supervised adaptation: We also evaluate the performance of ACAL algorithm when there are limited labeled and unlabeled target samples in Table 6 (Appendix A).

In case of MNIST???USPS, our model outperforms many high-resource unsupervised domain adaptation in TAB1 by using < 1000 unlabeled samples only.

We also apply our proposed model to domain adaptation in speech recognition.

We use TIMIT dataset, where the male to female speaker ratio is about 7 : 3 and thus we choose the data subset from male speakers as the source and the subset from female speakers as the target domain.

We evaluate performance on the standard TIMIT test set and use phoneme error rate (PER) as the evaluation metric.

Spectrogram representation of audio is chosen for model evaluation.

As demonstrated by BID13 , multi-discriminator training significantly impacts adaptation performance.

Therefore, we used the multi-discriminator architecture as the discriminator for the adversarial loss in our evaluation.

Our task-specific model is a pre-trained speech recognition model within each domain in this set of experiments.

The result are shown in Table 3 .

We observe significant performance improvements over the baseline model as well as comparable or better performance as compared to previous methods.

It is interesting to note that the performance of the proposed model on the adapted male (M ??? F) almost matches the baseline model performance, where the model is trained on true female speech.

In addition, the performance gap in this case is significant as compared to other methods, which suggests the adapted distribution is indeed close to the true target distribution.

In addition, when combined with more data, our model further outperforms the baseline by a noticeable margin.

Table 3 : Speech domain adaptation results on TIMIT.

We treat Male (M) and Female (F) voices for the source and target domains, respectively, based on the intrinsic imbalance of speaker genders in the dataset (about 7 : 3 male/female ratio).

For the evaluation metric, lower is better.

In this paper, we propose to use augmented cycle-consistency adversarial learning for domain adaptation and introduce a task specific model to facilitate learning domain related mappings.

We enforce cycle-consistency using a task specific loss instead of the conventional reconstruction objective.

Additionally, we use the task specific model as an additional source of information for the discriminator in the corresponding domain.

We demonstrate the effectiveness of our proposed approach by evaluating on two domain adaptation tasks, and in both cases we achieve significant performance improvement as compared to the baseline.

By extending the definition of task-specific model to unsupervised learning, such as reconstruction loss using autoencoder, or self-supervision, our proposed method would work on all settings of domain adaptation.

Such unsupervised task can be speech modeling using wavenet BID35 , or language modeling using recurrent or transformer networks BID27 .

In this section, we evaluate domain adaptation for MNIST???SVHN for comparison with CycleGAN, as well as the relaxed version of the cycle-consistent objective (Relaxed-Cyc, see eq. 5 in section 3).

For the former, 1 reconstruction loss is replaced with the model loss in order to encouraging cycle-consistency.

We also experiment with two different task specific models M : specifically, DenseNet BID17 , representing a relatively complex architecture) and a modified LeNet (representing a relatively simple architecture, see section 4.1).

TAB2 and 5 show the results on augmenting the low resource MNIST and SVHN with the complementary high resource domain.

This approach improves test performance of the target classifier by a large margin, compared to when trained only using the target domain data.

We observe that training a more complicated deep model for the target domain weakens this effect.

As shown in TAB2 , using DenseNet as a classifier on MNIST (target) achieves ??? 24% lower test classification accuracy than using a variant of LeNet.

This difference likely reflects differences in the two architectures' degree of overfitting.

Overfitting will produce a false gradient signal during cycle adversarial learning (when classifying the adapted source examples).

Based on this observation, we use a comparatively simpler LeNet architecture with SVHN as the target domain (see TAB3 ).

Using our proposed approach, SVHN test performance improves by 27% over domain adaptation using CycleGAN.

We also include some qualitative results when performing domain adaptation from SVHN (source) to MNIST (target), as shown in Figure 5 .

We also compare the performance with different number of labeled target samples in FIG4 .

It indicates the improvement on generalization performance of target model using Augmented cyclic adaptation, with variable labeled target domain on MNIST and SVHN datasets.

Evaluation of semi supervised adaptation is presented in Table 6 .

74.61??0.43Figure 5: Qualitative comparison of domain adaptation for experimental models.

Each column illustrates the mapping performed by each of the models from the original SVHN image (source domain) to MNIST (target domain, 10 labeled samples per class in total).

It can be seen that the augmented cycle-consistent model is able to preserve most of the semantic information, while still approximately match the target distribution.

Table 6 : Low-resource semi and unsupervised domain adaptation on MNIST (M), USPS (U) and SVHN (S) datasets.

Note: n = 10 means 10 samples per class, and 10% denotes the percentage of target samples (per class) which have labels.

0% corresponds to low-resource unsupervised adaptation.

BID39 with modifications mentioned in BID13 .

Both generators in CycleGAN are based on U-net BID28 architecture with 4 layers of convolution of sizes (8,3,3,1,1), (16,3,3,1,1), (32,3,3,2,2) , (64,3,3,2,2), followed by corresponding deconvolution layers.

To increase stability of adversarial training, as proposed by BID13 , the discriminator output is modified to predict a single scalar as real/fake probability.

Discriminator has 4 convolution layers of sizes (8,4,4,2,2), (16,4,4,2,2), (32,4,4,2,2), (64,4,4,2,2), as default kernel and stride sizes in BID13 .

ASR model is implemented based on BID38 , which is trained only with maximum likelihood.

The model includes one convolutional layer of size (32, 41, 11, 2, 2) , and five residual convolution blocks of size (32,7,3,1,1), (32,5,3,1,1), (32,3,3,1,1), (64,3,3,2,1), (64,3,3,1,1) respectively.

Convolutional layers are followed by 4 layers of bidirectional GRU RNNs with 1024 hidden units per direction per layer.

Finally, a fully-connected hidden layer of size 1024 is used as the output layer.

DISPLAYFORM0

In this section we show some qualitative results on transcriptions produced from different models.

No Adaptation sil dh ah f aa sil p er z ih n ih n sil dh ih m z er v er r aa v iy ng aa n sil t ay m sil CycleGAN sil b er f aa sil p r ih th iy n m ih sil b ih ih m n sil f r eh m er r aw n iy ng er n sil t er m sil ACAL sil dh ih f aa l sil p r ih z ih n ih sil dh iy ih m f er m er r aa dh ih ng aa n sil t ah m sil True sil ch iy sil s sil t aa sil k ih ng z r ah n dh ih f er s sil t ay m dh eh r w aa r n sil No Adaptation sil ch iy sil ch s sil t aa sil k ih n ng z r ah m dh ah f er s sil t aa m dh eh w ah r n sil CycleGAN sil ch iy sil ch s sil t aa sil k ih ng z r ah n dh ih f er ih s sil t ay n dh eh r w aa r ng sil ACAL sil sh iy sil ch s sil t aa sil k ih ng z r ah m dh ah f er s sil t ay m dh eh r w aa r n sil No Adaptation sil k eh l s iy ih m ey sil k s sil b ow n z ih n sil t iy sil s sil t r aa l sil CycleGAN sil t aw s iy ih m n m ey sil k s sil b ow n z ih n sil t iy sil s sil t r aa ng sil ACAL sil k aw s iy ih m ey sil k s sil b ow n z ih n sil t iy sil s sil t r aa ng sil

@highlight

A new cyclic adversarial learning augmented with auxiliary task model which improves domain adaptation performance in low resource supervised and unsupervised situations 

@highlight

Proposes an extension of cycle-consistent adversatial adaptation methods in order to tackle domain adaptation where limited supervised target data is available.

@highlight

This paper introduces a domain adaptation approach based on the idea of Cyclic GAN and proposes two different algorithms.