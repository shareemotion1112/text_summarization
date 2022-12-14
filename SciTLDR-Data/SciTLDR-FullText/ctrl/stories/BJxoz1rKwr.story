Self-supervised learning (SlfSL), aiming at learning feature representations through ingeniously designed pretext tasks without human annotation, has achieved compelling progress in the past few years.

Very recently, SlfSL has also been identified as a promising solution for semi-supervised learning (SemSL) since it offers a new paradigm to utilize unlabeled data.

This work further explores this direction by proposing a new framework to seamlessly couple SlfSL with SemSL.

Our insight is that the prediction target in SemSL can be modeled as the latent factor in the predictor for the SlfSL target.

Marginalizing over the latent factor naturally derives a new formulation which marries the prediction targets of these two learning processes.

By implementing this framework through a simple-but-effective SlfSL approach -- rotation angle prediction, we create a new SemSL approach called Conditional Rotation Angle Prediction (CRAP).

Specifically, CRAP is featured by adopting a module which predicts the image rotation angle \textbf{conditioned on the candidate image class}. Through experimental evaluation, we show that CRAP achieves superior performance over the other existing ways of combining SlfSL and SemSL.

Moreover, the proposed SemSL framework is highly extendable.

By augmenting CRAP with a simple SemSL technique and a modification of the rotation angle prediction task, our method has already achieved the state-of-the-art SemSL performance.

The recent success of deep learning is largely attributed to the availability of a large amount of labeled data.

However, acquiring high-quality labels can be very expensive and time-consuming.

Thus methods that can leverage easily accessible unlabeled data become extremely attractive.

Semisupervised learning (SemSL) and self-supervised learning (SlfSL) are two learning paradigms that can effectively utilize massive unlabeled data to bring improvement to predictive models.

SemSL assumes that a small portion of training data is provided with annotations and the research question is how to use the unlabeled training data to generate additional supervision signals for building a better predictive model.

In the past few years, various SemSL approaches have been developed in the context of deep learning.

The current state-of-the-art methods, e.g. MixMatch (Berthelot et al., 2019) , unsupervised data augmentation (Li et al., 2018) , converge to the strategy of combining multiple SemSL techniques, e.g. ??-Model (Laine & Aila, 2017) , Mean Teacher (Tarvainen & Valpola, 2017) , mixup (Zhang et al., 2018) , which have been proved successful in the past literature.

SlfSL aims for a more ambitious goal of learning representation without any human annotation.

The key assumption in SlfSL is that a properly designed pretext predictive task which can be effortlessly derived from data itself can provide sufficient supervision to train a good feature representation.

In the standard setting, the feature learning process is unaware of the downstream tasks, and it is expected that the learned feature can benefit various recognition tasks.

SlfSL also offers a new possibility for SemSL since it suggests a new paradigm of using unlabeled data, i.e., use them for feature training.

Recent work has shown great potential in this direction.

This work further advances this direction by proposing a new framework to seamlessly couple SlfSL with SemSL.

The key idea is that the prediction target in SemSL can serve as a latent factor in the course of predicting the pretext target in a SlfSL approach.

The connection between the predictive targets of those two learning processes can be established through marginalization over the latent factor, which also implies a new framework of SemSL.

The key component in this framework is a module that predicts the pretext target conditioned on the target of SemSL.

In this preliminary work, we implement this module by extending the rotation angle prediction method, a recently proposed SlfSL approach for image recognition.

Specifically, we make its prediction conditioned on each candidate image class, and we call our method Conditional Rotation Angle Prediction (CRAP).

The proposed framework is also highly extendable.

It is compatible with many SemSL and SlfSL approaches.

To demonstrate this, we further extend CRAP by using a simple SemSL technique and a modification to the rotation prediction task.

Through experimental evaluation, we show that the proposed CRAP achieves significantly better performance than the other SlfSL-based SemSL approaches, and the extended CRAP is on par with the state-of-the-art SemSL methods.

In summary, the main contributions of this paper are as follows:

??? We propose a new SemSL framework which seamlessly couples SlfSL and SemSL.

It points out a principal way of upgrading a SlfSL method to a SemSL approach.

??? Implementing this idea with a SlfSL approach, we create a new SemSL approach (CRAP) that can achieve superior performance than other SlfSL-based SemSL methods.

??? We further extend CRAP with a SemSL technique and an improvement over the SlfSL task.

The resulted new method achieves the state-of-the-art performance of SemSL.

Our work CRAP is closely related to both SemSL and SlfSL.

SemSL is a long-standing research topic which aims to learn a predictor from a few labeled examples along with abundant of unlabeled ones.

SemSL based on different principals are developed in the past decades, e.g., "transductive" models (Gammerman et al., 1998; Joachims, 2003) , multi-view style approaches (Blum & Mitchell, 1998; Zhou & Li, 2005) and generative model-based methods (Kingma et al., 2014; Springenberg, 2016) , etc.

Recently, the consistency regularization based methods have become quite influential due to their promising performance in the context of deep learning.

Specifically, ??-Model (Laine & Aila, 2017 ) requires model's predictions to be invariant when various perturbations are added to the input data.

Mean Teacher (Tarvainen & Valpola, 2017 ) enforces a student model producing similar output as a teacher model whose weights are calculated through the moving average over the weight of student model.

Virtual Adversarial Training (Miyato et al., 2018) encourages the predictions for input data and its adversarially perturbed version to be consistent.

More recently, mixup (Zhang et al., 2018; Verma et al., 2019) has emerged as a powerful SemSL regularization method which requires the output of mixed data to be close to the output mixing of original images.

In order to achieve good performance, most state-of-the-art approaches adopt the strategy of combining several existing techniques together.

For example, Interpolation Consistency Training (Verma et al., 2019) incorporates Mean Teacher into the mixup regularization, MixMatch (Berthelot et al., 2019 ) adopts a technique that uses fused predictions as pseudo prediction target as well as the mixup regularization.

Unsupervised data augmentation (Li et al., 2018) upgrades ??-Model with advanced data augmentation methods.

SlfSL is another powerful paradigm which learns feature representations through training on pretext tasks whose labels are not human annotated .

Various pretext tasks are designed in different approaches.

For example, image inpainting (Pathak et al., 2016) trains model to reproduce an arbitrary masked region of the input image.

Image colorization (Zhang et al., 2016) encourages model to perform colorization of an input grayscale image.

Rotation angle prediction (Gidaris et al., 2018) forces model to recognize the angle of a rotated input image.

After training with the pretext task defined in a SlfSL method, the network is used as a pretrained model and can be fine-tuned for a downstream task on task-specific data.

Generally speaking, it is still challenging for SlfSL method to achieve competitive performance to fully-supervised approaches.

However, SlfSL provides many new insights into the use of unlabeled data and may have a profound impact to other learning paradigms, such as semi-supervised learning.

SlfSL based SemSL is an emerging approach which incorporates SlfSL into SemSL.

The most straightforward approach is to first perform SlfSL on all available data and then fine-tune the learned model on labeled samples.

S 4 L ) is a newly proposed method which jointly train the downstream task and pretext task in a multi-task fashion without breaking them into stages.

In this paper, we further advance this direction through proposing a novel architecture which explicitly links these two tasks together and ensure that solving one task is beneficial to the other.

In SemSL, we are given a set of training samples {x 1 , x 2 , ?? ?? ?? , x n } ??? X with only a few of them X l = {x 1 , x 2 , ?? ?? ?? , x l } ??? X annotated with labels {y 1 , y 2 , ?? ?? ?? , y l } ??? Y l (usually l << n and y is considered as discrete class label here).

The goal of a SemSL algorithm is to learn a better posterior probability estimator over y, i.e., p(y|x, ??) with ?? denoting model parameters, from both labeled and unlabeled training samples.

SlfSL aims to learn feature representations via a pretext task.

The task usually defines a target z, which can be derived from the training data itself, e.g., rotation angle of the input image.

Once z is defined, SlfSL is equivalent to training a predictor to model p(z|x; ??).

There are two existing schemes to leverage SlfSL for SemSL.

The first is to use SlfSL to learn the feature from the whole training set and then fine-tuning the network on the labeled part.

The other is jointly optimizing the tasks of predicting y and z, as in the recently proposed S 4 L method.

As shown in Figure 1 (a) , S 4 L constructs a network with two branches and a shared feature extractor.

One branch for modeling p(y|x; ??) and another branch for modeling p(z|x; ??).

However, in both methods the pretext target z predictor p(z|x; ??) is implicitly related to the task of predicting y.

Our framework is different in that we explicitly incorporate y into the predictor for z. Specifically, we treat y as the latent factor in p(z|x; ??) and factorize p(z|x; ??) through marginalization:

Eq. 1 suggests that the pretext target predictor p(z|x; ??) can be implemented as two parts: a model to estimate p(y|x; ??) and a model to estimate z conditioned on both x and y, i.e., p(z|x, y; ??).

For the labeled samples, the ground-truth y is observed and can be used for training p(y|x; ??).

For unlabeled samples, the estimation from p(y|x; ??) and p(z|x, y; ??) will be combined together to make the final prediction about z. Consequently, optimizing the loss for p(z|x; ??) will also provide gradient to back-propagate through p(y|x; ??).

This is in contrast to the case of S 4 L, where the gradient generated from the unlabeled data will not flow through p(y|x; ??).

Theoretically, p(z|x; ??) and p(y|x; ??) can be two networks, but in practise we model them as two branches connecting to a shared feature extractor.

p(z|x; ??) suggested by Eq. 1 is essentially a pretext target predictor with a special structure and partial observations on its latent variable, i.e. y. The benefits of using such a predictor can be understood from three perspectives: (1) p(y|x; ??) in Eq. 1 acts as a soft selector to select p(z|x, y; ??) for predicting z. If the estimation of p(y|x; ??) is accurate, it will select p(z|x, y =??(x); ??) for prediction and update, where??(x) is the true class of x. This selective updating will make p(z|x, y; ??) give more accurate prediction over z if y matches??(x).

After such an update, p(z|x, y; ??) will in turn encourage p(y|x; ??) to attain higher value for y =??(x) since the prediction from p(z|x, y =??(x); ??) is more likely to be accurate.

Thus, the terms p(y|x; ??) and p(z|x, y; ??) will reinforce each other during training.

(2) even if p(y|x; ??) is not accurate (this may happen at the beginning of the training process), p(z|x, y; ??) can still perform the pretext target prediction and act as an unsupervised feature learner.

Thus, the features will be gradually improved in the course of training.

With a better feature representation, the estimation of p(y|x; ??) will also be improved.

(3) Finally, to predict z in Eq. 1, p(z|x, y; ??) needs to be evaluated for each candidate y.

This in effect is similar to creating an ensemble of diversified pretext target predictors and with the combination weight given by p(y|x; ??) according to the marginalization rule.

Thus, training features with Eq. 1 may enjoy the benefit from ensemble learning.

Again, this will lead to better features and thus benefit the modelling of p(y|x; ??) and p(z|x, y; ??).

The above framework provides a guideline for turning a SlfSL method into a SemSL algorithm: (1) modifying a SlfSL predictor p(z|x; ??) by p(z|x, y; ??) and introducing a branch for p(y|x; ??) (2) optimizing the prediction of z on the SemSL dataset and update the branches p(z|x, y; ??), p(y|x; ??) and their shared feature extractor.

(3) using p(y|x; ??) as downstream task predictor or adding an additional branch for training p(y|x; ??) only with the labeled data as in S 4 L. More details about the additional branch will be explained in Section 4.

< l a t e x i t s h a 1 _ b a s e 6 4 = " t 2 W 6 x u / Y c G q z 7 e 6 w r U f 4 A 7 6 z e 2 A = " < l a t e x i t s h a 1 _ b a s e 6 4 = " t 2 W 6 x u / Y c G q z 7 e 6 w r U f 4 A 7 6 z e 2 A = "

In the following part, we will describe an implementation of this framework, which is realized by upgrading the rotation-angle prediction-based SlfSL to its conditional version.

Rotation angle prediction is a recently proposed SlfSL approach for image recognition.

It randomly rotates the input image by one of the four possible rotation angles ({0

??? , 90

??? }) and requires the network to give a correct prediction of the rotation angle.

Despite being extremely simple, this method works surprisingly well in practice.

The underlying logic is that to correctly predict the rotation angle, the network needs to recognize the canonical view of objects from each class and thus enforces the network to learn informative patterns of each image category.

Following the proposed framework, we upgrade rotation angle prediction to conditional rotation angle prediction (CRAP) for semi-supervised learning.

In this case, z in Eq. 1 is the rotation angle and y is the class label of input image x. We realize p(z|x, y; ??) by allocating a rotation angle prediction branch for each class.

The prediction from each branch is then aggregated with the aid of p(y|x; ??) for the final prediction of z as shown in Eq. 1.

A more detailed schematic illustration of the CRAP method is shown in Figure 1 (b) .

As seen, our method adopts a network with multiple branches and a shared feature extractor.

Specifically, branches within the dashed box are called auxiliary branches since they are only used for training and will be discarded at the test stage.

It contains C rotation predictors which corresponds to p(z|x, y; ??) and a semantic classifier which generates p(y|x; ??).

The auxiliary branches and feature extractor are trained by using the procedure described in Section 3.

Note that in CRAP, we do not directly use the semantic classifier from the auxiliary branches as the final classifier.

Instead, we introduce an additional semantic classifier and learn it only via the loss incurred from the labeled data.

This treatment is similar to S 4 L and we find this strategy work slightly better in practice.

We postulate the reason is that the p(y|x; ??) branch in auxiliary branches is mainly trained by the supervision generated from the optimization of p(z|x; ??).

Such supervision is noisy comparing with the loss generated from the ground-truth y.

It is better to use such a branch just for feature training since the latter is more tolerant to noisy supervision.

Remark: (1) One potential obstacle of our model is that the quantity of parameters in the auxiliary branches would increase significantly with a large C. To tackle this, we propose to perform dimension reduction for the features feeding into the rotation predictor.

Results in Section 5.3 show that this scheme is effective as our performance will not drop even when the dimension is reduced from 2048 to 16.

(2) The CRAP method is also highly expendable.

In the following, we will extend CRAP from two perspectives: improving p(y|x; ??) and improving p(z|x, y; ??).

As discussed in Section 3, our method essentially introduces a network module with a special structure and partial observations on the latent variable y. Besides using labeled data to provide supervision for y, we can also use existing SemSL techniques to provide extra loss for modeling p(y|x; ??).

To implement such an extension, we employ a simple SemSL loss as follows: we rotate each image in four angles within one batch (the prediction of the rotated image can be obtained as the byproduct of CRAP) and obtain the arithmetic averagep of the predicted distributions across these four rotated samples.

Then we perform a sharpening operation overp as in MixMatch :

where C is the number of classes and T ??? (0, 1] is a temperature hyper-parameter.

Then we use the cross entropy betweenp i and p(y|x; ??) (in auxiliary branches) as an additional loss.

Note that other (more powerful) SemSL can also apply here.

We choose the above SemSL technique is simply because its operation, i.e. image rotation, has already been employed in the CRAP algorithm and thus could be reused to generate the additional SemSL loss without increasing the complexity of the algorithm.

We also make another extension over CRAP by introducing an improved version of the conditional rotation prediction task.

Specifically, we require the rotation prediction branch to predict rotation angle for a mixed version of the rotated image, that is, we randomly mix the input image x i with another randomly sampled rotated image x j via x mix = ??x i + (1 ??? ??)x j , with ?? sampled from [0.5, 1].

Meanwhile, the class prediction p(y|x i ; ??) is calculated from the unmixed version of the input x i .

In such a design, the network needs to recognize the rotation angle of the target object with the noisy distraction from another image, and we call this scheme denoising rotation prediction.

The purpose of introducing this modified task is to make the SlfSL task more challenging and more dependent on the correct prediction from p(y|x; ??).

To see this point, let's consider the following example.

Letter 'A' is rotated with 270

??? and is mixed with letter 'B' with rotation 90

??? .

Directly predicting the rotation angle for this mixed image encounters an ambiguity: whose rotation angle, A's or B's, is the right answer?

In other words, the network cannot know which image class is the class-of-interest.

This ambiguity can only be resolved from the output of p(y|x; ??) since its input is unmixed target image.

Therefore, this improved rotation prediction task relies more on the correct prediction from the semantic classifier and training through CRAP is expected to give stronger supervision signal to p(y|x; ??).

Note that although the denoising rotation prediction also uses mix operation, it is completely different from mixup.

The latter constructs a loss to require the output of the mixed image to be mixed version of the outputs of original images.

This loss is not applied in our method.

For more algorithm details about CRAP and the extended CRAP, please refer to the Appendix A.1.

In this section, we conduct experiments to evaluate the proposed CRAP method 1 .

The purpose of our experiments is threefolds: (1) to validate if CRAP is better than other SlfSL-based SemSL algorithms.

(2) to compare CRAP and extended CRAP (denoted as CRAP+ hereafter) against the state-of-the-art SemSL methods.

(3) to understand the contribution of various components in CRAP.

To make a fair comparison to recent works, different experimental protocols are adopted for different datasets.

Specifically, for CIFAR-10 and CIFAR-100 (Krizhevsky et al., 2009 ) and SVHN (Netzer et al., 2011) , we directly follow the settings in (Berthelot et al., 2019 ).

For ILSVRC-2012 (Russakovsky et al., 2015 , our settings are identical to except for data pre-processing operations for which we only use the inception crop augmentation and horizontal mirroring.

We ensure that all the baselines are compared under the same setting.

Followed the standard settings of SemSL, the performance with different amount of labeled samples are tested.

For CIFAR-10 and SVHN, sample size of labeled images is ranged in five levels: {250, 500, 1000, 2000, 4000}. For CIFAR-100, 10000 labeled data is used for training.

For ILSVRC-2012, 10% and 1% of images are labeled among the whole dataset.

In each experiment, three independent trials are conducted for all datasets except for ILSVRC-2012.

See more details in Table 8 in Appendix.

Firstly, we compare CRAP to other SlfSL-based SemSL algorithms on five datasets: CIFAR-10, CIFAR-100, SVHN, SVHN+Extra and ILSVRC-2012.

Two SlfSL-based SemSL baseline approaches are considered: 1) Fine-tune: taking the model pretrained on the pretext task as an initialization and fine-tuning with a set of labeled data.

We term this method Fine-tune in the following sections.

2) S 4 L: S 4 L method proposed in .

Note that we do not include any methods which combine other SemSL techniques.

For this reason, we only use our basic CRAP algorithm in the comparison in this subsection.

As a reference, we also report the performance obtained by only using the labeled part of the dataset for training, denoting as Labeled-only.

The experimental results are as follows:

The results are presented in Table 1 .

We find that the "Fine-tune" strategy leads to a mixed amount of improvement over the "Labeled-only" case.

It is observed that a large improvement can be obtained when the amount of labeled samples is ranged from 500 to 2000 but not on 250 and 4000's settings.

It might be because on one hand too few labeled samples are not sufficient to perform an effective fine-tuning while on the other hand the significant improvement diminishes after the sample size increase.

In comparison, S 4 L achieves much better accuracy for the case of using few samples.

This is largely benefited from its down-stream-task awareness design -the labeled training samples exerts impact at the feature learning stage.

Our CRAP method achieves significantly better performance than those two ways of incorporating SlfSL for SemSL and always halves the test error of S 4 L in most cases.

Table 2 shows the results of each method.

Somehow surprisingly, we find that the Fine-tune and S 4 L do not necessarily outperform the Labeled-only baseline.

They actually performs worse than Labeled-only on SVHN.

With more training data in SVHN + Extra, S 4 L tends to bring benefits for enhancing performance when the size of labeled samples are small e.g., with 250 samples.

In comparison, the proposed CRAP still manages to produce significant improvement over Labeled-only in all those settings.

This result clearly demonstrates that the simple combination of SlfSL and SemSL may not necessarily bring improvement and a properly-designed strategy of incorporating SlfSL with SemSL is crucial.

CIFAR-100 As shown in Table 3 , it is obvious that all SlfSL-based SemSL methods can have better accuracy than that of Labeled-only.

S 4 L leads to a marginal improvement over Fine-tune although its performance is a little bit unstable on different partitions as shown by its higher variance.

Again, the proposed CRAP achieves significant improvement over those baselines.

Table 4 presents the results of each method.

The top block of Table 4 shows the reported results in the original S 4 L paper and we also re-implement S 4 L based on the code of .

Due to the difference of data pre-processing, results in the upper block cannot be directly compared to those below.

Again, we have observed that CRAP is consistently superior to S 4 L in all settings.

As mentioned in Section 4, for saving the computational cost, we propose to reduce the dimensionality of features fed into the rotation angle predictor when there is a large number of classes.

In Table 5 , we demonstrates the effect of this scheme.

As seen, the test performance stays the same when the feature dimensions is gradually reduced from 2048 to only 16 dimensions.

This clearly validates the effectiveness of the proposed scheme.

In the following section, we proceed to demonstrate the performance of CRAP+, that is, the extended CRAP method by incorporating the two extensions discussed in Section 4.1 and 4.2.

We compare its performance against the current state-of-the-art methods in SemSL.

Similar to (Berthelot et al., 2019) , several SemSL baselines are considered: Pseudo-Label, ??-Model, Mean Teacher, Virtual Adversarial Training (VAT), MixUp and MixMatch 2 .

Since a fair and comprehensive comparison has been done in (Berthelot et al., 2019) and we strictly follow the same experimental setting, we directly compare CRAP+ to the numbers reported in (Berthelot et al., 2019) .

The experimental results are shown in Figure 2 , Figure 3 and Table 6 .

As seen from those Figures and Table, the proposed CRAP+ is on-par with the best performed approaches, e.g., Mixmatch, in those datasets.

This clearly demonstrates the power of the proposed method.

Note that the current state-ofthe-art in SemSL is achieved by carefully combining multiple existing successful ideas in SemSL.

In contrast, our CRAP+ achieves excellent performance via an innovative framework of marrying SlfSL with SemSL.

Conceptually, the latter enjoys greater potential.

In fact, CRAP might be further extended by using more successful techniques in SemSL, such as MixUp.

Since the focus of this paper is to study how SlfSL can benefit SemSL, we do not pursue this direction here.

Since there are several components in CRAP and CRAP+, we study the effect of adding or removing some components in order to provide additional insight into the role of each part.

Specifically, we measure the effect of (1) only adding extension 1 to CRAP, i.e., incorporating an additional SemSL loss through sharpening operations on the semantic classifier in auxiliary branches (2) further adding extension 2 to CRAP.

The resulted model is identical to CRAP+ (3) removing semantic classifier of main branch from CRAP.

This is equivalent to using the semantic classifier in auxiliary branches for testing (4) removing rotation angle prediction branch from auxiliary branches and adding extension 1 to CRAP.

The resulted structure can be seen as a variant of only using the SemSL technique in Extension 1 (but also with the classifier in main branch) (5) removing whole auxiliary branches from CRAP, i.e., pure supervised method with data rotated.

We conduct ablation studies on CIFAR-10 with 250 and 4000 labels with results presented in Table 7.

The main observations are: (1) The two extensions in CRAP+ will bring varying degrees of improvement.

Extension 1 in Section 4.1, i.e., a stronger p(y|x; ??) modeling, perhaps leads to greater improvement.

(2) Using an additional semantic classifier leads to a slight performance improvement over the strategy of directly utilizing p(y|x; ??) in the auxiliary branches for testing (method in third line from the bottom).

(3) Using the sharpening strategy as in our extension 1 and training a SemSL method alone does not produce good performance.

This indicates the superior performance of CRAP+ is not simply coming from a strong SemSL method but its incorporation with the CRAP framework.

(4) Applying rotation as a data augmentation for labeled data (the last method in Table 7) will not lead to improved performance over the labeled-only baseline as by cross referring the results in Table 9 .

This shows that the advantage of CRAP is not coming from the rotation data augmentation.

In this work, we introduce a framework for effectively coupling SemSL with SlfSL.

The proposed CRAP method is an implementation of this framework and it shows compelling performance on several benchmark datasets compared to other SlfSL-based SemSL methods.

Furthermore, two extensions are incorporated into CRAP to create an improved method which achieves comparable performance to the state-of-the-art SemSL methods.

<|TLDR|>

@highlight

Coupling semi-supervised learning with self-supervised learning and explicitly modeling the self-supervised task conditioned on the semi-supervised one