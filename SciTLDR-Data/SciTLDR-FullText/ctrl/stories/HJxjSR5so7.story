Training a model to perform a task typically requires a large amount of data from the domains in which the task will be applied.

However, it is often the case that data are abundant in some domains but scarce in others.

Domain adaptation deals with the challenge of adapting a model trained from a data-rich source domain to perform well in a data-poor target domain.

In general, this requires learning plausible mappings between domains.

CycleGAN is a powerful framework that efficiently learns to map inputs from one domain to another using adversarial training and a cycle-consistency constraint.

However, the conventional approach of enforcing cycle-consistency via reconstruction may be overly restrictive in cases where one or more domains have limited training data.

In this paper, we propose an augmented cyclic adversarial learning model that enforces the cycle-consistency constraint via an external task specific model, which encourages the preservation of task-relevant content as opposed to exact reconstruction.

This task specific model both relaxes the cycle-consistency constraint and complements the role of the discriminator during training, serving as an augmented information source for learning the mapping.

We explore adaptation in speech and visual domains in low resource in supervised setting.

In speech domains, we adopt a speech recognition model from each domain as the task specific model.

Our approach improves absolute performance of speech recognition by 2% for female speakers in the TIMIT dataset, where the majority of training samples are from male voices.

In low-resource visual domain adaptation, the results show that our approach improves absolute performance by 14% and 4% when adapting SVHN to MNIST and vice versa, respectively, which outperforms unsupervised domain adaptation methods that require high-resource unlabeled target domain.

Domain adaptation BID14 BID31 BID1 aims to generalize a model from source domain to a target domain.

Typically, the source domain has a large amount of training data, whereas the data are scarce in the target domain.

This challenge is typically addressed by learning a mapping between domains, which allows data from the source domain to enrich the available data for training in the target domain.

One way of learning such mappings is through Generative Adversarial Networks (GANs BID7 with cycle-consistency constraint (CycleGAN Zhu et al., 2017) , which enforces that mapping of an example from the source to the target and then back to the source domain would result in the same example (and vice versa for a target example).

Due to this constraint, CycleGAN learns to preserve the 'content' 1 from the source domain while only transferring the 'style' to match the distribution of the target domain.

This is a powerful constraint, and various works BID32 BID20 BID10 have demonstrated its effectiveness in learning cross domain mappings.

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

We improve the phoneme error rate by ∼ 5% on TIMIT dataset, when adapting the model trained on one speech from one gender to the other.

Our work is broadly related to domain adaptation using neural networks for both supervised and unsupervised domain adaptation.

Supervised Domain Adaptation When labels are available in the target domain, a common approach is to utilize the label information in target domain to minimize the discrepancy between source and target domain BID13 BID28 BID6 BID5 .

For example, BID13 applies the marginal Fisher analysis criteria and Maximum Mean Discrepancy (MMD) to minimize the distribution difference between source and target domain.

BID28 proposed to add a domain classifier that predicts domain label of the inputs, with a domain confusion loss.

BID6 leverages attributes by using attribute and class level classification loss with attribute consistent loss to fine-tune the target model.

Our method also employs models from both domains, however, our models are used to assist adversarial learning for better learning of the target domain distribution.

In addition, our final model for supervised domain adaptation is obtained by training on data from target domain as well as the transfered data from the source domain, rather than fine-tuning a source/target domain model.

More recently, various work have taken advantage of the substantial generation capabilities of the GAN framework and applied them to domain adaptation BID19 BID2 BID32 BID29 BID16 BID10 .

However, most of these works focus on high-resource unsupervised domain adaptation, which may be unsuitable for situations where the target domain data are limited.

BID2 uses a GAN to adapt data from the source to target domain while simultaneously training a classifier on both the source and adapted data.

Our method also employs task specific models; however, we use the models to augment the CycleGAN formulation.

We show that having cycles in both directions (i.e. from source to target and vice versa) is important in the case where the target domain has limited data (see sec. 4).

BID29 proposes adversarial discriminative domain adaptation (ADDA), where adversarial learning is employed to match the representation learned from the source and target domain.

Our method also utilizes pre-trained model from source domain, but we only implicitly match the representation distributions rather than explicitly enforcing representational similarity.

Cycle-consistent adversarial domain adaptation (CyCADA Hoffman et al., 2018) is perhaps the most similar work to our own.

This approach uses both 1 and semantic Figure 1 : Illustration of proposed approach.

Left: CycleGAN BID34 .

Middle: Relaxed cycle-consistent model (RCAL), where the cycle-consistency is enforced through task specific models in corresponding domain.

Right: Augmented cycle-consistent model (ACAL).

In addition to the relaxed model, the task specific model is also used to augment the discriminator of corresponding domain to facilitate learning.

In the diagrams x and L denote data and losses, respectively.

We point out that the ultimate goal of our approach is to use the mapped Source → Target samples (x S →T ) to augment the limited data of the target domain (x T ).consistency to enforce cycle-consistency.

An important difference in our work is that we also include another cycle that starts from the target domain.

This is important because, if the target domain is of low resource, the adaptation from source to target may fail due to the difficulty in learning a good discriminator in the target domain.

BID0 also suggests to improve CycleGAN by explicitly enforcing content consistency and style adaptation, by augmenting the cyclic adversarial learning to hidden representation of domains.

Our model is different from recent cyclic adversarial learning, due to implicit learning of content and style representation through an auxiliary task, which is more suitable for low resource domains.

Using classification to assist GAN training has also been explored previously BID26 BID27 BID17 .

BID26 proposed CatGAN, where the discriminator is converted to a multi-class classifier.

We extend this idea to any task specific model, including speech recognition task, and use this model to preserve task specific information regarding the data.

We also propose that the definition of task model can be extended to unsupervised tasks,such as language or speech modeling in domains, meaning augmented unsupervised domain adaptation.

To learn the true data distribution P data (X) in a nonparametric way, BID7 proposed the generative adversarial network (GAN).

In this framework, a discriminator network D(x) learns to discriminate between the data produced by a generator network G(z) and the data sampled from the true data distribution P data (X), whereas the generator models the true data distribution by learning to confuse the discriminator.

Under certain assumptions BID7 , the generator would learn the true data distribution when the game reaches equilibrium.

Training of GAN is in general done by alternately optimizing the following objective for D and G. DISPLAYFORM0

CycleGAN BID34 extends this framework to multiple domains, P S (X) and P T (X), while learning to map samples back and forth between them.

Adversarial learning is applied such that the result mapping from G S →T will match the target distribution P T (X), and similarly for the reverse mapping from G T →S .

This is accomplished by the following adversarial objectives: DISPLAYFORM0 CycleGAN also introduces cycle-consistency, which enforces that each mapping is able to invert the other.

In the original work, this is achieved by including the following reconstruction objective: DISPLAYFORM1 Learning the CycleGAN model involves optimizing a weighted combination of the above objectives 2, 3 and 4.

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

Therefore, the learning signal G S →T gets from D T would be limited.

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

Input: source domain data P S (x, y), target domain data P T (x, y), pretrained source task model M S Output: target task model M T while not converged do Sample from (x s , y s ) from P S if y t in P T then %Supervised% Sample (x t , y t ) from P T Finetune source model M S on (x s , y s ) and (G T →S (x t ), y t ) samples (eq. 6) Train task model M T on (x t , y t ) and (G S →T (x s ), y s ) samples (eq. 7) DISPLAYFORM0 To address the second issue, we augment the adversarial objective with corresponding objective: DISPLAYFORM1 Similar to adversarial training, we optimize the above objective by maximizing D S (D T ) and minimizing G T →S (G S →T ) and M S (M T ).

With the new terms, the learning of mapping functions G get assists from both the discriminator and the task specific model.

The task specific model learns to capture conditional probability distribution P S (Y |X) (P T (Y |X)), that also preserves information regarding P S (X) (P T (X)).

This conditional information is different than the information captured through the discriminator D S (D T ).

The difference is that the model is only required to preserve useful information regarding X respect to predicting Y , for modeling the conditional distribution, which makes learning the conditional model a much easier problem.

In addition, the conditional model mediates the influence of data that the discriminator does not have access to (Y ), which should further assist learning of the mapping functions G T →S (G S →T ).In case of unsupervised domain adaptation, when there is no information of target conditional probability distribution P T (Y |X), we propose to use source model M S to estimate P T (Y |X) through adversarial learning, i.e. DISPLAYFORM2 .

Therefore, proposed model can be extended to unsupervised domain adaptation, with the corresponding modified objectives: To further extend this approach to semi-supervised domain adaptation, both supervised and unsupervised objectives for labeled and unlabeled target samples are used interchangeably, as explained in Algorithm 1.

DISPLAYFORM3

In this section, we evaluate our proposed model on domain adaptation for visual and speech recognition.

We continue the convention of referring to the data domains as 'source' and 'target', where target denotes the domain with either limited or unlabeled training data.

Visual domain adaptation is evaluated using the MNIST dataset (M) BID18 , Street View House Numbers (SVHN) datasets (S) BID23 , USPS (U) BID15 , MNISTM (MM) and Synthetic Digits (SD) BID3 .

Adaptation on speech is evaluated on the domain of gender within the TIMIT dataset BID4 , which contains broadband 16kHz recordings of 6300 utterances (5.4 hours) of phonetically-balanced speech.

The male/female ratio of speakers across train/validation/test sets is approximately 70% to 30%.

Therefore, we treat male speech as the source domain and female speech as the low resource target domain.

To get an idea of the contribution from each component of our model, in this section we perform a series of ablations and present the results in TAB0 .

We perform these ablations by treating SVHN as the source domain and MNIST as the target domain.

We down sample the MNIST training data so only 10 samples per class are available during training, which is only 0.17% of full training data.

The testing performance is calculated on the full MNIST test set.

We use a modified LeNet for all experiments in this ablation.

The Modified LeNet consists of two convolutional layers with 20 and 50 channels, followed by a dropout layer and two fully connected layers of 50 and 10 dimensionality.

There are various ways that one may utilize cycle-consistency or adversarial training to do domain adaptation from components of our model.

One way is to use adversarial training on the target domain to ensure matching of distribution of adapted data, and use the task specific model to ensure the 'content' of the data from the source domain is preserved.

This is the model described in BID2 , except their model is originally unsupervised.

This model is denoted as S → T in TAB0 .

It is also interesting to examine the importance of the double cycle, which is proposed in BID34 and adopted in our work.

Theoretically, one cycle would be sufficient to learn the mapping between domains; therefore, we also investigate the performance of one cycle only models, where one direction would be from source to target and then back, and similarly for the other direction.

These models are denoted as (S→T→S)-One Cycle and (T→S→T)-One Cycle in TAB0 , respectively.

To test the effectiveness of the relaxed cycle-consistency (eq. 5) and augmented adversarial loss (eq. 6 and 7), we also test one cycle models while progressively adding these two losses.

Interestingly, the one cycle relaxed and one cycle augmented models are similar to the model proposed in BID10 when their model performs mapping from source to target domain and then back.

The difference is that their model is unsupervised and includes more losses at different levels.

As can be seen from TAB0 , the simple conditional model performed surprisingly well as compared to more complicated cyclic counterparts.

This may be attributed to the reduced complexity, since it only needs to learn one set of mapping.

As expected, the single cycle performance is poor when the target domain is of limited data due to inefficient learning of discriminator in the target domain (see section 3).

When we change the cycle to the other direction, where there are abundant data in the target domain, the performance improves, but is still worse than the simple one without cycle.

This is because the adaptation mapping (i.e. G S →T ) is only learned via the generated samples from G T →S , which likely deviate from the real examples in practice.

This observation also suggests that it would be beneficial to have cycles in both directions when applying the cycle-consistency constraint, since then both mappings can be learned via real examples.

The trends get reversed when we are using relaxed implementation of cycle-consistency from the reconstruction error with the task specific losses.

This is because now the power of the task specific model is crucial to preserve the content of the data after the reverse mapping.

When the source domain dataset is sufficiently large, the cycle-consistency is preserved.

As such, the resulting learned mapping functions would preserve meaningful semantics of the data while transferring the styles to the target domain, and vice versa.

In addition, it is clear that augmenting the discriminator with task specific loss is helpful for learning adaptations.

Furthermore, the information added from the task specific model is clearly beneficial for improving the adaptation performance, without this none of the models outperform the baseline model, where no adaptation is performed.

Last but not least, it is also clear from the results that using task specific model improves the overall adaptation performance.

In this section, we experiment on domain adaptation for the task of digit recognition.

In each experiment, we select one domain (MNIST, USPS, MNISTM, SVHN, Synthetic Digits) to be the target.

We conduct two type of domain adaptation.

First, low-resource supervised adaptation where we sub-sample the target to contain only a few examples per class, using the other full dataset as the source domain.

Comparison with recent low resource domain adaptation, FADA (Motiian et al., 2017) for MNIST, USPS, and SVHN adaptation is shown in TAB1 .

We also apply our proposed model to domain adaptation in speech recognition.

We use the TIMIT dataset, where the male to female speaker ratio is about 7 : 3 and thus we choose the data subset from male speakers as the source and the subset from female speakers as the target domain.

We evaluate the FORMULA0 , multi-discriminator training significantly impacts adaptation performance.

Therefore, we used the multi-discriminator architecture as the discriminator for the adversarial loss in our evaluation.

Our task specific model is a pre-trained speech recognition model within each domain in this set of experiments.

The result are shown in TAB2 .

We observe significant performance improvements over the baseline model as well as comparable or better performance as compared to previous methods.

It is interesting to note that the performance of the proposed model on the adapted male (M → F) almost matches the baseline model performance, where the model is trained on true female speech.

In addition, the performance gap in this case is significant as compared to other methods, which suggests the adapted distribution is indeed close to the true target distribution.

In addition, when combined with more data, our model further out performs the baseline by a noticeable margin.

In this paper, we propose to use augmented cycle-consistency adversarial learning for domain adaptation and introduce a task specific model to facilitate learning domain related mappings.

We enforce cycle-consistency using a task specific loss instead of the conventional reconstruction objective.

Additionally, we use the task specific model as an additional source of information for the discriminator in the corresponding domain.

We demonstrate the effectiveness of our proposed approach by evaluating on two domain adaptation tasks, and in both cases we achieve significant performance improvement as compared to the baseline.

By extending the definition of task-specific model to unsupervised learning, such as reconstruction loss using autoencoder, or self-supervision, our proposed method would work on all settings of domain adaptation.

Such unsupervised task can be speech modeling using wavenet BID30 , or language modeling using recurrent or transformer networks BID24 .

<|TLDR|>

@highlight

A robust domain adaptation by employing a task specific loss in cyclic adversarial learning