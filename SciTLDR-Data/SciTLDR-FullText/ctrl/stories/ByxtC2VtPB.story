It has been widely recognized that adversarial examples can be easily crafted to fool deep networks, which mainly root from the locally non-linear behavior nearby input examples.

Applying mixup in training provides an effective mechanism to improve generalization performance and model robustness against adversarial perturbations, which introduces the globally linear behavior in-between training examples.

However, in previous work, the mixup-trained models only passively defend adversarial attacks in inference by directly classifying the inputs, where the induced global linearity is not well exploited.

Namely, since the locality of the adversarial perturbations, it would be more efficient to actively break the locality via the globality of the model predictions.

Inspired by simple geometric intuition, we develop an inference principle, named mixup inference (MI), for mixup-trained models.

MI mixups the input with other random clean samples, which can shrink and transfer the equivalent perturbation if the input is adversarial.

Our experiments on CIFAR-10 and CIFAR-100 demonstrate that MI can further improve the adversarial robustness for the models trained by mixup and its variants.

Deep neural networks (DNNs) have achieved state-of-the-art performance on various tasks (Goodfellow et al., 2016) .

However, counter-intuitive adversarial examples generally exist in different domains, including computer vision (Szegedy et al., 2014) , natural language processing (Jin et al., 2019) , reinforcement learning (Huang et al., 2017) , speech and graph data (Dai et al., 2018) .

As DNNs are being widely deployed, it is imperative to improve model robustness and defend adversarial attacks, especially in safety-critical cases.

Previous work shows that adversarial examples mainly root from the locally unstable behavior of classifiers on the data manifolds (Goodfellow et al., 2015; Fawzi et al., 2016; 2018; Pang et al., 2018b) , where a small adversarial perturbation in the input space can lead to an unreasonable shift in the feature space.

On the one hand, many previous methods try to solve this problem in the inference phase, by introducing transformations on the input images.

These attempts include performing local linear transformation like adding Gaussian noise (Tabacof & Valle, 2016) , where the processed inputs are kept nearby the original ones, such that the classifiers can maintain high performance on the clean inputs.

However, as shown in Fig. 1(a) , the equivalent perturbation, i.e., the crafted adversarial perturbation, is still ?? and this strategy is easy to be adaptively evaded since the randomness of x 0 w.r.t x 0 is local (Athalye et al., 2018) .

Another category of these attempts is to apply various non-linear transformations, e.g., different operations of image processing (Guo et al., 2018; Raff et al., 2019) .

They are usually off-the-shelf for different classifiers, and generally aim to disturb the adversarial perturbations, as shown in Fig. 1(b ).

Yet these methods are not quite reliable since there is no illustration or guarantee on to what extent they can work.

On the other hand, many efforts have been devoted to improving adversarial robustness in the training phase.

For examples, the adversarial training (AT) methods (Madry et al., 2018; Shafahi et al., 2019) induce locally stable behavior via data augmentation on adversarial examples.

However, AT methods are usually computationally expensive, and will often degenerate model performance on the clean inputs or under general-purpose transformations like rotation (Engstrom et al., 2019) .

In contrast, the mixup training method introduces globally linear behavior in-between the data manifolds, which can also improve adversarial robustness (Zhang

Virtual inputs The processed inputs fed into classifiers (a) (b) (c) Figure 1 : Intuitive mechanisms in the input space of different input-processing based defenses.

x is the crafted adversarial example, x0 is the original clean example, which is virtual and unknown for the classifiers.

?? is the adversarial perturbation.

et al., 2018; Verma et al., 2019a) .

Although this improvement is usually less significant than it resulted by AT methods, mixup-trained models can keep state-of-the-art performance on the clean inputs; meanwhile, the mixup training is computationally more efficient than AT.

The interpolated AT method also shows that the mixup mechanism can further benefit the AT methods.

However, most of the previous work only focuses on embedding the mixup mechanism in the training phase, while the induced global linearity of the model predictions is not well exploited in the inference phase.

Compared to passive defense by directly classifying the inputs , it would be more effective to actively defend adversarial attacks by breaking their locality via the globally linear behavior of the mixup-trained models.

In this paper, we develop an inference principle for mixup-trained models, named mixup inference (MI).

In each execution, MI performs a global linear transformation on the inputs, which mixups the input x with a sampled clean example x s , i.e.,x = ??x + (1 ??? ??)x s (detailed in Alg.

1), and feedx into the classifier as the processed input.

There are two basic mechanisms for robustness improving under the MI operation (detailed in Sec. 3.2.1), which can be illustrated by simple geometric intuition in Fig. 1(c) .

One is perturbation shrinkage: if the input is adversarial, i.e., x = x 0 + ??, the perturbation ?? will shrink by a factor ?? after performing MI, which is exactly the mixup ratio of MI according to the similarity between triangles.

Another one is input transfer: after the MI operation, the reduced perturbation ???? acts on random x 0 .

Comparing to the spatially or semantically local randomness introduced by Gaussian noise or image processing,x 0 introduces spatially global and semantically diverse randomness w.r.t x 0 .

This makes it less effective to perform adaptive attacks against MI (Athalye et al., 2018) .

Furthermore, the global linearity of the mixup-trained models ensures that the information of x 0 remained inx 0 is proportional to ??, such that the identity of x 0 can be recovered from the statistics ofx 0 .

In experiments, we evaluate MI on CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton, 2009 ) under the oblivious attacks (Carlini & Wagner, 2017) and the adaptive attacks (Athalye et al., 2018 ).

The results demonstrate that our MI method is efficient in defending adversarial attacks in inference, and is also compatible with other variants of mixup, e.g., the interpolated AT method .

Note that Shimada et al. (2019) also propose to mixup the input points in the test phase, but they do not consider their method from the aspect of adversarial robustness.

In this section, we first introduce the notations applied in this paper, then we provide the formula of mixup in training.

We introduce the adversarial attacks and threat models in Appendix A.1.

Given an input-label pair (x, y), a classifier F returns the softmax prediction vector F (x) and the

, where L is the number of classes and [L] = {1, ?? ?? ?? , L}.

The classifier F makes a correct prediction on x if y =??.

In the adversarial setting, we augment the data pair (x, y) to a triplet (x, y, z) with an extra binary variable z, i.e.,

The variable z is usually considered as hidden in the inference phase, so an input x (either clean or adversarially corrupted) can be generally denoted as x = x 0 + ?? ?? 1 z=1 .

Here x 0 is a clean sample from the data manifold p(x) with label y 0 , 1 z=1 is the indicator function, and ?? is a potential perturbation crafted by adversaries.

It is worthy to note that the perturbation ?? should not change the true label of the input, i.e., y = y 0 .

For p -norm adversarial attacks (Kurakin et al., 2017; Madry et al., 2018) , we have ?? p ??? , where is a preset threshold.

Based on the assumption that adversarial examples are off the data manifolds, we formally have x 0 + ?? / ??? supp(p(x)) (Pang et al., 2018a) .

In supervised learning, the most commonly used training mechanism is the empirical risk minimization (ERM) principle (Vapnik, 2013) , which minimizes

with the loss function L. While computationally efficient, ERM could lead to memorization of data (Zhang et al., 2017) and weak adversarial robustness (Szegedy et al., 2014) .

As an alternative, introduce the mixup training mechanism, which minimizes 1 m m j=1 L(F (x j ),??? j ).

Herex j = ??x j0 + (1 ??? ??)x j1 ;??? j = ??y j0 + (1 ??? ??)y j1 , the input-label pairs (x j0 , y j0 ) and (x j1 , y j1 ) are randomly sampled from the training dataset, ?? ??? Beta(??, ??) and ?? is a hyperparameter.

Training by mixup will induce globally linear behavior of models in-between data manifolds, which can empirically improve generalization performance and adversarial robustness Tokozume et al., 2018a; b; Verma et al., 2019a; b) .

Compared to the adversarial training (AT) methods (Goodfellow et al., 2015; Madry et al., 2018) , trained by mixup requires much less computation and can keep state-of-the-art performance on the clean inputs.

Although the mixup mechanism has been widely shown to be effective in different domains (Berthelot et al., 2019; Beckham et al., 2019; Verma et al., 2019a; b) , most of the previous work only focuses on embedding the mixup mechanism in the training phase, while in the inference phase the global linearity of the trained model is not well exploited.

Compared to passively defending adversarial examples by directly classifying them, it would be more effective to actively utilize the globality of mixup-trained models in the inference phase to break the locality of adversarial perturbations.

The above insight inspires us to propose the mixup inference (MI) method, which is a specialized inference principle for the mixup-trained models.

In the following, we apply colored y,?? and y s to visually distinguish different notations.

Consider an input triplet (x, y, z), where z is unknown in advance.

When directly feeding x into the classifier F , we can obtain the predicted label??.

In the adversarial setting, we are only interested in the cases where x is correctly classified by F if it is clean, or wrongly classified if it is adversarial (Kurakin et al., 2018) .

This can be formally denoted as

The general mechanism of MI works as follows.

Every time we execute MI, we first sample a label y s ??? p s (y), then we sample x s from p s (x|y s ) and mixup it with x asx = ??x + (1 ??? ??)x s .

p s (x, y) denotes the sample distribution, which is constrained to be on the data manifold, i.e., supp(p s (x)) ??? supp(p(x)).

In practice, we execute MI for N times and average the output predictions to obtain F MI (x), as described in Alg.

1.

Here we fix the mixup ratio ?? in MI as a hyperparameter, while similar properties hold if ?? comes from certain distribution.

Theoretically, with unlimited capability and sufficient clean samples, a well mixup-trained model F can be denoted as a linear function H on the convex combinations of clean examples (Hornik et al., 1989; Guo et al., 2019) , i.e., ???x i , x j ??? p(x) and ?? ??? [0, 1], there is

Specially, we consider the case where the training objective L is the cross-entropy loss, then H(x i ) should predict the one-hot vector of label y i , i.e., H y (

Input: The mixup-trained classifier F ; the input x. Hyperparameters: The sample distribution p s ; the mixup ratio ??; the number of execution N .

adversarial, then there should be an extra non-linear part G(??; x 0 ) of F , since x is off the data manifolds.

Thus for any input x, the prediction vector can be compactly denoted as

According to Eq. (3) and Eq. (4), the output ofx in MI is given by:

where ??? ?????? represents the limitation when the execution times N ??? ???. Now we separately investigate the y-th and??-th (could be the same one) components of F (x) according to Eq. (5), and see how these two components differ from those of F (x).

These two components are critical because they decide whether we can correctly classify or detect adversarial examples (Goodfellow et al., 2016) .

Note that there is H y (x 0 ) = 1 and H ys (x s ) = 1, thus we have the y-th components as

Furthermore, according to Eq. (2), there is 1 y=?? = 1 z=0 .

We can represent the??-th components as

From the above formulas we can find that, except for the hidden variable z, the sampling label y s is another variable which controls the MI output F (x) for each execution.

Different distributions of sampling y s result in different versions of MI.

Here we consider two easy-to-implement cases:

MI with predicted label (MI-PL): In this case, the sampling label y s is the same as the predicted label??, i.e., p s (y) = 1 y=?? is a Dirac distribution on??.

In this case, the label y s is uniformly sampled from the labels other than??, i.e., p s (y) = U??(y) is a discrete uniform distribution on the set {y ??? [L]|y =??}.

We list the simplified formulas of Eq. (7) and Eq. (8) under different cases in Table 1 for clear representation.

With the above formulas, we can evaluate how the model performance changes with and without MI by focusing on the formula of

Specifically, in the general-purpose setting where we aim to correctly classify adversarial examples (Madry et al., 2018) , we claim that the MI method improves the robustness if the prediction value on the true label y increases while it on the adversarial label?? decreases after performing MI when the input is adversarial (z = 1).

This can be formally denoted as

We refer to this condition in Eq. (10) as robustness improving condition (RIC).

Further, in the detection-purpose setting where we want to detect the hidden variable z and filter out adversarial inputs, we can take the gap of the??-th component of predictions before and after the MI operation, i.e., ???F??(x; p s ) as the detection metric (Pang et al., 2018a) .

To formally measure the detection ability on z, we use the detection gap (DG), denoted as

A higher value of DG indicates that ???F??(x; p s ) is better as a detection metric.

In the following sections, we specifically analyze the properties of different versions of MI according to Table 1 , and we will see that the MI methods can be used and benefit in different defense strategies.

In the MI-PL case, when the input is clean (i.e., z = 0), there is F (x) = F (x), which means ideally the MI-PL operation does not influence the predictions on the clean inputs.

When the input is adversarial (i.e., z = 1), MI-PL can be applied as a general-purpose defense or a detection-purpose defense, as we separately introduce below:

General-purpose defense: If MI-PL can improve the general-purpose robustness, it should satisfy RIC in Eq. (10).

By simple derivation and the results of Table 1 , this means that

Since an adversarial perturbation usually suppress the predicted confidence on the true label and promote it on the target label (Goodfellow et al., 2015) , there should be G??(??;x 0 ) > 0 and G y (??;x 0 ) < 0.

Note that the left part of Eq. (12) can be decomposed into

Here Eq. (13) indicates the two basic mechanisms of the MI operations defending adversarial attacks, as shown in Fig. 1(c) .

The first mechanism is input transfer, i.e., the clean input that the adversarial perturbation acts on transfers from the deterministic x 0 to stochasticx 0 .

Compared to the Gaussian noise or different image processing methods which introduce spatially or semantically local randomness, the stochasticx 0 induces spatially global and semantically diverse randomness.

This will make it harder to perform an adaptive attack in the white-box setting (Athalye et al., 2018) .

The second mechanism is perturbation shrinkage, where the original perturbation ?? shrinks by a factor ??.

This equivalently shrinks the perturbation threshold since ???? p = ?? ?? p ??? ?? , which means that MI generally imposes a tighter upper bound on the potential attack ability for a crafted perturbation.

Besides, empirical results in previous work also show that a smaller perturbation threshold largely weakens the effect of attacks (Kurakin et al., 2018) .

Therefore, if an adversarial attack defended by these two mechanisms leads to a prediction degradation as in Eq. (12), then applying MI-PL would improve the robustness against this adversarial attack.

Similar properties also hold for MI-OL as described in Sec. 3.2.2.

In Fig. 2 , we empirically demonstrate that most of the existing adversarial attacks, e.g., the PGD attack (Madry et al., 2018) satisfies these properties.

Detection-purpose defense: According to Eq. (11), the formula of DG for MI-PL is

By comparing Eq. (12) and Eq. (14), we can find that they are consistent with each other, which means that for a given adversarial attack, if MI-PL can better defend it in general-purpose, then ideally MI-PL can also better detect the crafted adversarial examples.

As to MI-OL, when the input is clean (z = 0), there would be a degeneration on the optimal clean prediction as F y (x) = F??(x) = ??, since the sampled x s does not come from the true label y. As compensation, MI-OL can better improve robustness compared to MI-PL when the input is adversarial (z = 1), since the sampled x s also does not come from the adversarial label?? in this case.

General-purpose defense: Note that in the MI-OL formulas of Table 1 , there is a term of 1 y=ys .

Since we uniformly select y s from the set [L] \ {??}, there is E(1 y=ys ) = 1 L???1 .

According to the RIC, MI-OL can improve robustness against the adversarial attacks if there satisfies

Note that the conditions in Eq. (15) is strictly looser than Eq. (12), which means MI-OL can defend broader range of attacks than MI-PL, as verified in Fig. 2 .

Detection-purpose defense: According to Eq. (11) and Table 1 , the DG for MI-OL is

It is interesting to note that DG MI-PL = DG MI-OL , thus the two variants of MI have the same theoretical performance in the detection-purpose defenses.

However, in practice we find that MI-PL performs better than MI-OL in detection, since empirically mixup-trained models cannot induce ideal global linearity (cf.

Fig. 2 in ).

Besides, according to Eq. (6), to statistically make sure that the clean inputs will be correctly classified after MI-OL, there should be ???k ??? [L] \ {y},

In this section, we provide the experimental results on CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton, 2009 ) to demonstrate the effectiveness of our MI methods on defending adversarial attacks.

Our code is available at an anonymous link: http://bit.ly/2kpUZVR.

In training, we use ResNet-50 (He et al., 2016) and apply the momentum SGD optimizer (Qian, 1999) on both CIFAR-10 and CIFAR-100.

We run the training for 200 epochs with the batch size of 64.

The initial learning rate is 0.01 for ERM, mixup and AT; 0.1 for interpolated AT .

The learning rate decays with a factor of 0.1 at 100 and 150 epochs.

The attack method for AT and interpolated AT is untargeted PGD-10 with = 8/255 and step size 2/255 (Madry et al., 2018) , and the ratio of the clean examples and the adversarial ones in each mini-batch is 1 : 1 .

The hyperparameter ?? for mixup and interpolated AT is 1.0 .

All defenses with randomness are executed 30 times to obtain the averaged predictions .

To verify and illustrate our theoretical analyses in Sec. 3, we provide the empirical relationship between the output predictions of MI and the hyperparameter ?? in Fig. 2 .

The notations and formulas annotated in Fig. 2 correspond to those introduced in Sec. 3.

We can see that the results follow our theoretical conclusions under the assumption of ideal global linearity.

Besides, both MI-PL and MI-OL empirically satisfy RIC in this case, which indicates that they can improve robustness under the untargeted PGD-10 attack on CIFAR-10, as quantitatively demonstrated in the following sections.

In this subsection, we evaluate the performance of our method under the oblivious-box attacks (Carlini & Wagner, 2017) .

The oblivious threat model assumes that the adversary is not aware of the existence of the defense mechanism, e.g., MI, and generate adversarial examples based on the unsecured classification model.

We separately apply the model trained by mixup and interpolated AT as the classification model.

The AUC scores for the detection-purpose defense are given in Fig. 3(a) .

The results show that applying MI-PL in inference can better detect adversarial attacks, while directly detecting by the returned confidence without MI-PL performs even worse than a random guess.

Table 3 : Classification accuracy (%) on the oblivious adversarial examples crafted on 1,000 randomly sampled test points of CIFAR-100.

Perturbation = 8/255 with step size 2/255.

The subscripts indicate the number of iteration steps when performing attacks.

The notation ??? 1 represents accuracy less than 1%.

The parameter settings for each method can be found in Table 5 .

Methods Cle.

PGD10 PGD50 PGD200 PGD10 PGD50 PGD200 We also compare MI with previous general-purpose defenses applied in the inference phase, e.g., adding Gaussian noise or random rotation (Tabacof & Valle, 2016) ; performing random padding or resizing after random cropping (Guo et al., 2018; .

The performance of our method and baselines on CIFAR-10 and CIFAR-100 are reported in Table 2 and Table 3 , respectively.

Since for each defense method, there is a trade-off between the accuracy on clean samples and adversarial samples depending on the hyperparameters, e.g., the standard deviation for Gaussian noise, we carefully select the hyperparameters to ensure both our method and baselines keep a similar performance on clean data for fair comparisons.

The hyperparameters used in our method and baselines are reported in Table 4 and Table 5 .

In Fig. 3(b) , we further explore this trade-off by grid searching the hyperparameter space for each defense to demonstrate the superiority of our method.

As shown in these results, our MI method can significantly improve the robustness for the trained models with induced global linearity, and is compatible with training-phase defenses like the interpolated AT method.

As a practical strategy, we also evaluate a variant of MI, called MI-Combined, which applies MI-OL if the input is detected as adversarial by MI-PL with a default detection threshold; otherwise returns the prediction on the original input.

We also perform ablation studies of ERM / AT + MI-OL in Table 2 , where no global linearity is induced.

The results verify that our MI methods indeed exploit the global linearity of the mixup-trained models, rather than simply introduce randomness.

Following Athalye et al. (2018) , we test our method under the white-box adaptive attacks (detailed in Appendix B.2).

Since we mainly adopt the PGD attack framework, which synthesizes adversarial examples iteratively, the adversarial noise will be clipped to make the input image stay within the valid range.

It results in the fact that with mixup on different training examples, the adversarial perturbation will be clipped differently.

To address this issue, we average the generated perturbations over the adaptive samples as the final perturbation.

The results of the adversarial accuracy w.r.t the number of adaptive samples are shown in Fig. 4 .

We can see that even under a strong adaptive attack, equipped with MI can still improve the robustness for the classification models.

In this section, we provide more backgrounds which are related to our work in the main text.

Adversarial attacks.

Although deep learning methods have achieved substantial success in different domains (Goodfellow et al., 2016) , human imperceptible adversarial perturbations can be easily crafted to fool high-performance models, e.g., deep neural networks (DNNs) (Nguyen et al., 2015) .

One of the most commonly studied adversarial attack is the projected gradient descent (PGD) method (Madry et al., 2018) .

Let r be the number of iteration steps, x 0 be the original clean example, then PGD iteratively crafts the adversarial example as

where clip x, (??) is the clipping function.

Here x * 0 is a randomly perturbed image in the neighborhood of x 0 , i.e.,?? (x 0 , ), and the finally returned adversarial example is x = x * r = x 0 + ??, following our notations in the main text.

Threat models.

Here we introduce different threat models in the adversarial setting.

As suggested in , a threat model includes a set of assumptions about the adversarys goals, capabilities, and knowledge.

Adversary's goals could be simply fooling the classifiers to misclassify, which is referred to as untargeted mode.

Alternatively, the goals can be more specific to make the model misclassify certain examples from a source class into a target class, which is referred to as targeted mode.

In our experiments, we evaluate under both modes, as shown in Table 2 and Table 3.

Adversary's capabilities describe the constraints imposed on the attackers.

Adversarial examples require the perturbation ?? to be bounded by a small threshold under p -norm, i.e., ?? p ??? .

For example, in the PGD attack, we consider under the ??? -norm.

Adversary's knowledge describes what knowledge the adversary is assumed to have.

Typically, there are three settings when evaluating a defense method:

??? Oblivious adversaries are not aware of the existence of the defense D and generate adversarial examples based on the unsecured classification model F (Carlini & Wagner, 2017 ).

??? White-box adversaries know the scheme and parameters of D, and can design adaptive methods to attack both the model F and the defense D simultaneously (Athalye et al., 2018 ).

??? Black-box adversaries have no access to the parameters of the defense D or the model F with varying degrees of black-box access .

In our experiments, we mainly test under the oblivious setting (Sec. 4.3) and white-box setting (Sec. 4.4), since previous work has already demonstrated that randomness itself is efficient on defending black-box attacks (Guo et al., 2018; .

To date, the most widely applied framework for adversarial training (AT) methods is the saddle point framework introduced in Madry et al. (2018) :

Here ?? represents the trainable parameters in the classifier F , and S is a set of allowed perturbations.

In implementation, the inner maximization problem for each input-label pair (x, y) is approximately solved by, e.g., the PGD method with different random initialization (Madry et al., 2018) .

As a variant of the AT method, propose the interpolated AT method, which combines AT with mixup.

Interpolated AT trains on interpolations of adversarial examples along with interpolations of unperturbed examples (cf.

Alg.

1 in .

Previous empirical results demonstrate that interpolated AT can obtain higher accuracy on the clean inputs compared to the AT method without mixup, while keeping the similar performance of robustness.

AT + MI-OL (ablation study) The ??OL = 0.8

The ??OL = 0.6

We provide more technical details about our method and the implementation of the experiments.

Generality.

According to Sec. 3, except for the mixup-trained models, the MI method is generally compatible with any trained model with induced global linearity.

These models could be trained by other methods, e.g., manifold mixup (Verma et al., 2019a; Inoue, 2018; .

Besides, to better defend white-box adaptive attacks, the mixup ratio ?? in MI could also be sampled from certain distribution to put in additional randomness.

Empirical gap.

As demonstrated in Fig. 2 , there is a gap between the empirical results and the theoretical formulas in Table 1 .

This is because that the mixup mechanism mainly acts as a regularization in training, which means the induced global linearity may not satisfy the expected behaviors.

To improve the performance of MI, a stronger regularization can be imposed, e.g., training with mixup for more epochs, or applying matched ?? both in training and inference.

Following Athalye et al. (2018) , we design the adaptive attacks for our MI method.

Specifically, according to Eq. (6), the expected model prediction returned by MI is:

Note that generally the ?? in MI comes from certain distribution.

For simplicity, we fix ?? as a hyperparameter in our implementation.

Therefore, the gradients of the prediction w.r.t.

the input x is:

= E ps ???F (u) ???u u=??x+(1?????)xs ??

?????x + (1 ??? ??)x s ???x

= ??E ps ???F (u) ???u | u=??x+(1?????)xs .

Table 3 .

The number of execution for each random method is 30.

In the implementation of adaptive PGD attacks, we first sample a series of examples {x s,k } N A k=1 , where N A is the number of adaptive samples in Fig. 3 .

Then according to Eq. (18), the sign of gradients used in adaptive PGD can be approximated by

sign ???F MI (x) ???x ??? sign N A k=1 ???F (u) ???u u=??x+(1?????)x s,k .(24)

The hyperparameter settings of the experiments shown in Table 2 and Table 3 are provided in Table 4 and Table 5 , respectively.

Since the original methods in and Guo et al. (2018) are both designed for the models on ImageNet, we adapt them for CIFAR-10 and CIFAR-100.

Most of our experiments are conducted on the NVIDIA DGX-1 server with eight Tesla P100 GPUs.

<|TLDR|>

@highlight

We exploit the global linearity of the mixup-trained models in inference to break the locality of the adversarial perturbations.