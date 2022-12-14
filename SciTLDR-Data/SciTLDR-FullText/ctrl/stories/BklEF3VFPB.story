Domain adaptation tackles the problem of transferring knowledge from a label-rich source domain to an unlabeled or label-scarce target domain.

Recently domain-adversarial training (DAT) has shown promising capacity to learn a domain-invariant feature space by reversing the gradient propagation of a domain classifier.

However, DAT is still vulnerable in several aspects including (1) training instability due to the overwhelming discriminative ability of the domain classifier in adversarial training, (2) restrictive feature-level alignment, and (3) lack of interpretability or systematic explanation of the learned feature space.

In this paper, we propose a novel Max-margin Domain-Adversarial Training (MDAT) by designing an Adversarial Reconstruction Network (ARN).

The proposed MDAT stabilizes the gradient reversing in ARN by replacing the domain classifier with a reconstruction network, and in this manner ARN conducts both feature-level and pixel-level domain alignment without involving extra network structures.

Furthermore, ARN demonstrates strong robustness to a wide range of hyper-parameters settings, greatly alleviating the task of model selection.

Extensive empirical results validate that our approach outperforms other state-of-the-art domain alignment methods.

Additionally, the reconstructed target samples are visualized to interpret the domain-invariant feature space which conforms with our intuition.

Deep neural networks have gained great success on a wide range of tasks such as visual recognition and machine translation (LeCun et al., 2015) .

They usually require a large number of labeled data that can be prohibitively expensive to collect, and even with sufficient supervision their performance can still be poor when being generalized to a new environment.

The problem of discrepancy between the training and testing data distribution is commonly referred to as domain shift (Shimodaira, 2000) .

To alleviate the effect of such shift, domain adaptation sets out to obtain a model trained in a label-rich source domain to generalize well in an unlabeled target domain.

Domain adaptation has benefited various applications in many practical scenarios, including but not limited to object detection under challenging conditions (Chen et al., 2018) , cost-effective learning using only synthetic data to generalize to real-world imagery (Vazquez et al., 2013) , etc.

Prevailing methods for unsupervised domain adaptation (UDA) are mostly based on domain alignment which aims to learn domain-invariant features by reducing the distribution discrepancy between the source and target domain using some pre-defined metrics such as maximum mean discrepancy (Tzeng et al., 2014) .

Recently, Ganin & Lempitsky (2015) proposed to achieve domain alignment by domainadversarial training (DAT) that reverses the gradients of a domain classifier to maximize domain confusion.

Having yielded remarkable performance gain, DAT was employed in many subsequent UDA methods (Long et al., 2018; Shu et al., 2018) .

Even so, there still exist three critical issues of DAT that hinder its performance: (1) as the domain classifier has high-capacity to discriminate two domains, the unbalanced adversarial training cannot continuously provide effective gradients, which is usually overcome by manually adjusting the weights of adversarial training according to specific tasks; (2) DAT-based methods cannot deal with pixel-level domain shift (Hoffman et al., 2018) ; (3) the domain-invariant features learned by DAT are only based on intuition but difficult to interpret, which impedes the investigation of the underlying mechanism of adversarial domain adaptation.

To overcome the aforementioned difficulties, we propose an innovative DAT approach, namely Max-margin Domain-Adversarial Training (MDAT), to realize stable and comprehensive domain alignment.

To demonstrate its effectiveness, we develop an Adversarial Reconstruction Network (ARN) that only utilizes MDAT for UDA.

Specifically, ARN consists of a shared feature extractor, a label predictor, and a reconstruction network (i.e. decoder) that serves as a domain classifier.

Supervised learning is conducted on source domain, and MDAT helps learn domain-invariant features.

In MDAT, the decoder only focuses on reconstructing samples on source domain and pushing the target domain away from a margin, while the feature extractor aims to fool the decoder by learning to reconstruct samples on target domain.

In this way, three critical issues can be solved by MDAT: (1) the max-margin loss reduces the discriminative capacity of domain classifier, leading to balanced and thus stable adversarial training; (2) without involving new network structures, MDAT achieves both pixel-level and feature-level domain alignment; (3) visualizing the reconstructed samples reveals how the source and target domains are aligned.

We evaluate ARN with MDAT on five visual and non-visual UDA benchmarks.

It achieves significant improvement to DAT on all tasks with pixel-level or higher-level domain shift.

We also observe that it is insensitive to the choices of hyperparameters and as such is favorable for replication in practice.

In principle, our approach is generic and can be used to enhance any UDA methods that leverage domain alignment as an ingredient.

Domain adaptation aims to transfer knowledge from one domain to another.

Ben-David et al. (2010) provide an upper bound of the test error on the target domain in terms of the source error and the H H-distance.

As the source error is stationary for a fixed model, the goal of most UDA methods is to minimize the H H-distance by reducing some metrics such as Maximum Mean Discrepancy (MMD) (Tzeng et al., 2014; Long et al., 2015) and CORAL (Sun & Saenko, 2016) .

Inspired by Generative Adversarial Networks (GAN) (Goodfellow et al., 2014) , Ganin & Lempitsky (2015) proposed to learn domain-invariant features by adversarial training, which has inspired many UDA methods thereafter.

Adversarial Discriminative Domain Adaptation (ADDA) tried to fool the label classifier by adversarial training but not in an end-to-end manner.

CyCADA (Hoffman et al., 2018) and PixelDA (Bousmalis et al., 2017) leveraged GAN to conduct both feature-level and pixel-level domain adaptation, which yields significant improvement yet the network complexity is high.

Another line of approaches that are relevant to our method is the reconstruction network (i.e. the decoder network).

The success of image-to-image translation corroborates that it helps learn pixellevel features in an unsupervised manner.

In UDA, Ghifary et al. (2016) employed a decoder network for pixel-level adaptation, and Domain Separate Network (DSN) (Bousmalis et al., 2016) further leveraged multiple reconstruction networks to learn domain-specific features.

These approaches treat the decoder network as an independent component that is irrelevant to domain alignment (Glorot et al., 2011) .

In this paper, our approach proposes to utilize the decoder network as domain classifier in MDAT which enables both feature-level and pixel-level domain alignment in a stable and straightforward fashion.

In unsupervised domain adaptation, we assume that the model works with a labeled dataset X S and an unlabeled dataset X T .

Let X S = {(x s i , y s i )} i??? [Ns] denote the labeled dataset of N s samples from the source domain, and the certain label y s i belongs to the label space Y that is a finite set (Y = 1, 2, ..., K).

The other dataset X T = {x t i } i??? [Nt] has N t samples from the target domain but has no labels.

We further assume that two domains have different distributions, i.e. The proposed architecture is composed of a shared feature extractor G e for two domains, a label predictor G y and a reconstruction network G r .

In addition to the basic supervised learning in the source domain, our adversarial reconstruction training enables the extractor G e to learn domain-invariant features.

Specifically, the network G r aims to reconstruct the source samples x s and to impede the reconstruction of the target samples x t , while the extractor G e tries to fool the reconstruction network in order to reconstruct the target samples x t .

trained to determine whether the input sample belongs to the source or the target domain while the feature extractor learns to deceive the domain classifier, which is formulated as:

In DAT, we usually utilize CNN as the feature extractor and fully connected layers (FC) as the domain classifier.

DAT reduces the cross-domain discrepancy, achieving significant performance improvement for UDA.

Nevertheless, the training of DAT is rather unstable.

Without sophisticated tuning of the hyper-parameters, DAT cannot reach the convergence.

Through empirical experiments, we observe that such instability is due to the imbalanced minimax game.

The binary domain classifier D can easily achieve convergence with very high accuracy at an early training epoch, while it is much harder for the feature extractor F to fool the domain classifier and to simultaneously perform well on the source domain.

In this sense, the domain classifier dominates DAT, and the only solution is to palliate the training of D by tuning the hyper-parameters according to different tasks.

In our method, we restrict the capacity of the domain classifier so as to form a minimax game in a harmonious manner.

Inspired by the max-margin loss in Support Vector Machine (SVM) (Cristianini et al., 2000) (i.e. hinge loss), if we push the source domain and the target domain away from a margin rather than as far as possible, then the training task of F to fool D becomes easier.

For a binary domain classifier, we define the margin loss as

where y is the predicted domain label, [??] + := max(0, ??), m is a positive margin and t is the ground truth label for two domains (t = ???1 for the source domain and t = 1 for the target domain).

Then we introduce our MDAT scheme based on an innovative network architecture.

Besides the training instability issue, DAT also suffers from restrictive feature-level alignment -lack of pixel-level alignment.

To realize stable and comprehensive domain alignment together, we first propose an Adversarial Reconstruction Network (ARN) and then elaborate MDAT.

As depicted in Figure 1 , our model consists of three parts including a shared feature extractor G e for both domains, a label predictor G y and a reconstruction network G r .

Let the feature extractor G e (x; ?? e ) be a function parameterized by ?? e which maps an input sample x to a deep embedding z. Let the label predictor G y (z; ?? y ) be a task-specific function parameterized by ?? y which maps an embedding z to a task-specific prediction??.

The reconstruction network G r (z; ?? r ) is a decoding function parameterized by ?? r that maps an embedding z to its corresponding reconstructionx.

The first learning objective for the feature extractor G e and label predictor G y is to perform well in the source domain.

For a supervised K-way classification problem, it is simply achieved by minimizing the negative log-likelihood of the ground truth class for each sample:

where y s i is the one-hot encoding of the class label y s i and the logarithm operation is conducted on the softmax predictions of the model.

The second objective is to render the feature learning to be domain-invariant.

This is motivated by the covariate shift assumption (Shimodaira, 2000) that indicates if the feature distributions S(z) = {G e (x; ?? e )|x ??? D S } and T (z) = {G e (x; ?? e )|x ??? D T } are similar, the source label predictor G y can achieve a similar high accuracy in the target domain.

To this end, we design a decoder network G r that serves as a domain classifier, and then MDAT could be applied for stable training.

Different from the normal binary domain classifier, MDAT lets the decoder network G r only reconstruct the features in the source domain and push the features in the target domain away from a margin m. In this way, the decoder has the functionality of distinguishing the source domain from the target domain.

The objective of training G r is formulated as

where m is a positive margin and L r (??) is the mean squared error (MSE) term for the reconstruction loss that is defined as

where || ?? || 2 2 denotes the squared L 2 -norm.

Oppositely, to form a minimax game, the feature extractor G e learns to deceive G r such that the learned target features are indistinguishable to the source ones, which is formulated by:

Then the whole learning procedure of ARN with MDAT can be formulated by:

where L y denotes the negative log-likelihood of the ground truth class for labeled sample (x s i , y s i ) and ?? controls the interaction of the loss terms.

In the following section, we provide theoretical justifications on how MDAT reduces the distribution discrepancy, and discuss why it is superior to the classic DAT.

In this section, we provide the theoretical justifications on how the proposed method reduces the distribution discrepancy for UDA.

The rationale behind domain alignment is motivated from the learning theory of non-conservative domain adaptation problem by Ben-David et al. (Ben-David et al., 2010) : Theorem 3.1 Let H be the hypothesis space where h ??? H. Let (D S , s ) and (D T , t ) be the two domains and their corresponding generalization error functions.

The expected error for the target domain is upper bounded by

where

Theoretically, when we minimize the H H-distance, the upper bound of the expected error for the target domain is reduced accordingly.

As derived in DAT (Ganin & Lempitsky, 2015) , assuming a family of domain classifiers H d to be rich enough to contain the symmetric difference hypothesis set of H p , such that H p H p = {h|h = h 1 ??? h 2 , h 1 , h 2 ??? H p } where ??? is XOR-function, the empirical H p H p -distance has an upper bound with regard to the optimal domain classifier h:

whereD S andD T denote the distributions of the source and target feature space Z S and Z T , respectively.

Note that the MSE of G r plus a ceiling function is a form of domain classifier h(z), i.e. [m ??? L r (??)] + ??? 0.5 for m = 1.

It maps source samples to 0 and target samples to 1 which is exactly the upper bound in Eq.10.

Therefore, our reconstruction network G r maximizes the domain discrepancy with a margin and the feature extractor learns to minimize it oppositely.

Compared with the conventional DAT-based methods that are usually based on a binary logistic network (Ganin & Lempitsky, 2015) , the proposed ARN with MDAT is more attractive and incorporates new merits conceptually and theoretically:

(1) Stable training and insensitivity to hyper-parameters.

Using the decoder as domain classifier with a margin loss to restrain its overwhelming capacity in adversarial training, the minimax game can continuously provide effective gradients for training the feature extractor.

Moreover, through the experiments in Section 4, we discover that our method shows strong robustness to the hyperparameters, i.e. ?? and m, greatly alleviating the parameters tuning for model selection.

(2) Richer information for comprehensive domain alignment.

Rather than DAT that uses a bit of domain information, MDAT utilizes the reconstruction network as the domain classifier that could capture more domain-specific and pixel-level features during the unsupervised reconstruction (Bousmalis et al., 2016) .

Therefore, MDAT further helps address pixel-level domain shift apart from the feature-level shift, leading to comprehensive domain alignment in a straightforward manner.

(3) Feature visualization for method validation.

Another key merit of MDAT is that MDAT allows us to visualize the features directly by the reconstruction network.

It is crucial to understand to what extent the features are aligned since this helps to reveal the underlying mechanism of adversarial domain adaptation.

We will detail the interpretability of these adapted features in Section 4.3.

In this section, we evaluate the proposed ARN with MDAT on a number of visual and non-visual UDA tasks with varying degrees of domain shift.

We conduct ablation study to corroborate the effectiveness of MDAT and unsupervised reconstruction for UDA.

Then the sensitivity of the hyperparameters is investigated, and the adapted features are interpreted via the reconstruction network in ARN.

Setup.

We evaluate our method on four classic visual UDA datasets and a WiFi-based Gesture Recognition (WGR) dataset (Zou et al., 2019) .

The classic datasets have middle level of domain shift including MNIST (LeCun et al., 1998) , USPS (Hull, 1994) , Street View House Numbers (SVHN) (Netzer et al., 2011) and Synthetic Digits (SYN).

For a fair comparison, we follow the same CNN architecture as DANN (Ganin & Lempitsky, 2015) while using the inverse of G e as G r with pooling operation replaced by upsampling.

For the penalty term ??, we choose 0.02 by searching over the grid {10 ???2 , 1}. We also obtain the optimal margin m = 5 by a search over {10 ???1 , 10}. Then we use the same hyperparameter settings for all tasks to show the robustness.

For the optimization, we simply use Adam Optimizer (lr = 2 ?? 10 ???4 , ?? 1 = 0.5, ?? 2 = 0.999) and train all experiments for 50 epochs with batch size 128.

We implemented our model and conducted all the experiments using the PyTorch framework.

More implementation details are illustrated in the appendix.

Baselines.

We evaluate the efficacy of our approach by comparing it with existing UDA methods that perform three ways of domain alignment.

Specifically, MMD regularization (Long et al., 2015) and Correlation Alignment (Sun & Saenko, 2016) employ the statistical distribution matching.

DRCN (Ghifary et al., 2016) and DSN (Bousmalis et al., 2016) Table 1 : We compare with general, statistics-based (S), reconstruction-based (R) and adversarialbased (A) state-of-the-art approaches.

We repeated each experiment for 3 times and report the average and standard deviation (std) of the test accuracy in the target domain.

while many prevailing UDA methods adopt domain-adversarial training including DANN (Ganin & Lempitsky, 2015) , ADDA (Tzeng et al., 2017) , MECA (Morerio et al., 2018) , CyCADA (Hoffman et al., 2018) and CADA (Zou et al., 2019) .

For all transfer tasks, we follow the same protocol as DANN (Ganin & Lempitsky, 2015) that uses official training data split in both domains for training and evaluates the testing data split in the target domain.

Both datasets are composed of grey-scale handwritten images with diverse stroke weights, leading to low-level domain shift.

Since USPS has only 7291 training images, USPS???MNIST is more difficult.

As shown in Table 1 , our method achieves state-of-the-art accuracy of 98.6% on MNIST???USPS and 98.4% on USPS???MNIST, which demonstrates that ARN can tackle low-level domain shift by only using ART (rather than many adversarial UDA methods that adopt other loss terms to adjust classifier boundaries or conduct style transfer).

SVHN???MNIST and SYN???SVHN.

The SVHN dataset contains RGB digit images that introduce significant variations such as scale, background, embossing, rotation, slanting and even multiple digits.

The SYN data consists of 50k RGB images of varying color, background, blur and orientation.

These two tasks have tremendous pixel-level domain shfit.

The proposed method achieves a state-ofthe-art performance of 97.4% for SVHN???MNIST, far ahead of other DAT-based methods, significantly improving the classic DANN by 22.7%.

Similarly, ARN with MDAT also achieves a noticeable improvement of 5.3% compared with the source-only model, even outperforming the supervised SVHN accuracy 91.3%.

WiFi Gesture Recognition with Distant Domains.

To evaluate the proposed method on a non-visual UDA task, we applied our method to the WiFi gesture recognition dataset (Zou et al., 2019) .

The WiFi data of six gestures was collected in two rooms regarded as two domains.

The results in Table 2 demonstrate that our approach significantly improves classification accuracy against Source-Only and DANN by 32.9% and 23.1%, respectively.

Table 3 : The accuracy (%) with different hyperparameters on SVHN???MNIST.

The contribution of MDAT and image reconstruction in ARN.

We design an ablation study to verify the contribution of MDAT and unsupervised reconstruction in ARN.

To this end, we discard the term L r (x t ) in Eq.4, and evaluate the method, denoted as ARN w.o.

MDAT in Table 1 .

(1) Comparing ARN w.o.

MDAT with source-only model, we can infer the effect of unsupervised reconstruction for UDA.

It is observed that ARN w.o.

MDAT improves tasks with low-level domain shift such as MNIST???USPS, which conforms with our discussion that the unsupervised reconstruction is instrumental in learning low-level features.

(2) Comparing ARN w.o.

MDAT with the original ARN, we can infer the contribution of MDAT.

Table 1 shows that the MDAT achieves an impressive marginof-improvement.

For USPS???MNIST and SVHN???MNIST, the MDAT improves ARN w.o.

MDAT by around 30%.

It demonstrates that MDAT which helps learn domain-invariant representations is the main reason for the tremendous improvement.

Parameter sensitivity.

We investigate the effect of ?? and m on SVHN???MNIST.

The results in Table 3 show that ARN achieves good performance as ?? ??? [0.01, 0.1] and even with larger ?? ARN is able to achieve convergence.

In comparison, denoting ?? as the weight of adversarial loss, the DANN cannot converge when ?? > 0.2.

For the sensitivity of m, the accuracy of ARN exceeds 96.0% as m ??? 1.

These analyses validate that the training of ARN is not sensitive to the parameters and even in the worst cases ARN can achieve convergence.

Gradients and training procedure.

We draw the training procedure with regard to loss and target accuracy in Figure 2 (b) and Figure 2(a) , respectively.

In Figure 2 (b), ARN has smoother and more effective gradients (L r ) for all ??, while the loss of DAT domain classifier (L d ) gets extremely small at the beginning.

This observation conforms with our intuition, which demonstrates that by restricting the capacity of domain classifier MDAT provides more effective gradients for training feature extractor, leading to a more stable training procedure.

This could be further validated in Figure 2 (b) where the ARN accuracy is more stable than that of DAT across training epochs.

Table 4 : Visualizing the source image, target images and reconstructed target images (R-Target Images) for four digit adaptation tasks.

Interpreting MDAT features via reconstructed images.

One of the key advantages of ARN is that by visualizing the reconstructed target images we can infer how the features are domain-invariant.

We reconstruct the MDAT features of the test data and visualize them in Table 4 .

It is observed that the target features are reconstructed to source-like images by the decoder G r .

As discussed before, intuitively, MDAT forces the target features to mimic the source features, which conforms with our visualization.

Similar to image-to-image translation, this indicates that our method conducts implicit feature-to-feature translation that transfers the target features to source-like features, and hence the features become domain-invariant.

We analyze the performance of domain alignment for DANN (DAT) (Ganin & Lempitsky, 2015) and ARN (MDAT) by plotting T-SNE embeddings of the features z on the task SVHN???MNIST.

In Figure 3 (a), the source-only model obtains diverse embeddings for each category but the domains are not aligned.

In Figure 3 (b), the DANN aligns two domains but the decision boundaries of the classifier are vague.

In Figure 3 (c), the proposed ARN effectively aligns two domains for all categories and the classifier boundaries are much clearer.

We proposed a new domain alignment approach namely max-margin domain-adversarial training (MDAT) and a MDAT-based network for unsupervised domain adaptation.

The proposed method offers effective and stable gradients for the feature learning via an adversarial game between the feature extractor and the reconstruction network.

The theoretical analysis provides justifications on how it minimizes the distribution discrepancy.

Extensive experiments demonstrate the effectiveness of our method and we further interpret the features by visualization that conforms with our insight.

Potential evaluation on semi-supervised learning constitutes our future work.

Hyperparameter For all tasks, we simply use the same hyperparameters that are chosen from the sensitivity analysis.

We use ?? = 0.02 and m = 5.0, and we reckon that better results can be obtained by tuning the hyperparameters for specific tasks.

Network Architecture For a fair comparison, we follow the network in DANN (Ganin & Lempitsky, 2015) for digit adaptation and simply build the reconstruction network by the inverse network of the extractor.

Here we draw the network architectures in Table 5 .

For WiFi gesture recognition, we adopt the same architecture as CADA (Zou et al., 2019) that is a modified version of LeNet-5.

We have presented all the results of the sensitivity study in Section 4.2, and now we show their detailed training procedures in Figure 4 (a) and 4(b).

It is observed that the accuracy increases when ?? drops or the margin m increases.

The reason is very simple: (1) when ?? is too large, it affects the effect of supervised training on source domain; (2) when the margin m is small, the divergence between source and target domain (i.e. H H-distance) cannot be measured well.

Here we provide more visualization of the reconstructed images of target samples.

In Figure 5 , the target samples are shown in the left column while their corresponding reconstructed samples are shown in the right.

We can see that for low-level domain shift such as MNIST???USPS, the reconstructed target samples are very source-like while preserving their original shapes and skeletons.

However, for larger domain shift in Figure 5 (c) and 5(d), they are reconstructed to source-like same digits but simultaneously some noises are removed.

Specifically, in Figure 5 (d), we can see that one target sample (SVHN) may contain more than one digits that are noises for recognition.

After reconstruction, only the right digits are reconstructed.

Some target samples may suffer from terrible illumination conditions but their reconstructed digits are very clear, which is amazing.

<|TLDR|>

@highlight

A stable domain-adversarial training approach for robust and comprehensive domain adaptation