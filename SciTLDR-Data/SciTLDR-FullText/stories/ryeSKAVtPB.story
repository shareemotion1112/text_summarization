Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, especially white-box targeted attacks.

This paper studies the problem of how aggressive white-box targeted attacks can be to go beyond widely used Top-1 attacks.

We propose to learn ordered Top-k attacks (k>=1), which enforce the Top-k predicted labels of an adversarial example to be the k (randomly) selected and ordered labels (the ground-truth label is exclusive).

Two methods are presented.

First, we extend the vanilla Carlini-Wagner (C&W) method and use it as a strong baseline.

Second, we present an adversarial distillation framework consisting of two components: (i) Computing an adversarial probability distribution for any given ordered Top-$k$ targeted labels. (ii) Learning adversarial examples by minimizing the Kullback-Leibler (KL) divergence between the adversarial distribution and the predicted distribution, together with the perturbation energy penalty.

In computing adversarial distributions, we explore how to leverage label semantic similarities, leading to knowledge-oriented attacks.

In experiments, we test Top-k (k=1,2,5,10) attacks in the ImageNet-1000 val dataset using two popular DNNs trained with the clean ImageNet-1000  train dataset, ResNet-50 and DenseNet-121.

Overall, the adversarial distillation approach obtains the best results, especially by large margin when computation budget is limited..

It reduces the perturbation energy consistently with the same attack success rate on all the four k's, and improve the attack success rate by large margin against the modified C&W method for k=10.

Despite the recent dramatic progress, deep neural networks (DNNs) (LeCun et al., 1998; Krizhevsky et al., 2012; He et al., 2016; Szegedy et al., 2016) trained for visual recognition tasks (e.g., image classification) can be easily fooled by so-called adversarial attacks which utilize visually imperceptible, carefully-crafted perturbations to cause networks to misclassify inputs in arbitrarily chosen ways in the close set of labels used in training (Nguyen et al., 2015; Szegedy et al., 2014; Athalye & Sutskever, 2017; Carlini & Wagner, 2016) , even with one-pixel attacks (Su et al., 2017) .

The existence of adversarial attacks hinders the deployment of DNNs-based visual recognition systems in a wide range of applications such as autonomous driving and smart medical diagnosis in the long-run.

In this paper, we are interested in learning visually-imperceptible targeted attacks under the whitebox setting in image classification tasks.

In the literature, most methods address targeted attacks in the Top-1 manner, in which an adversarial attack is said to be successful if a randomly selected label (not the ground-truth label) is predicted as the Top-1 label with the added perturbation satisfying to be visually-imperceptible.

One question arises,

• The "robustness" of an attack method itself : How far is the attack method able to push the underlying ground-truth label in the prediction of the learned adversarial examples?

Table 1 shows the evaluation results of the "robustness" of different attack methods.

The widely used C&W method (Carlini & Wagner, 2016) does not push the GT labels very far, especially when smaller perturbation energy is aimed using larger search range (e.g., the average rank of the GT label is 2.6 for C&W 9×1000 ).

Consider Top-5, if the ground-truth labels of adversarial examples still largely appear in the Top-5 of the prediction, we may be over-confident about the 100% ASR, (He et al., 2016) .

Please see Sec. 4 for detail of experimental settings.

Method ASR Proportion of GT Labels in Top-k (smaller is better) Average Rank of GT Labels (larger is better)

Top-3 Top-5 Top-10 Top-50 Top-100 C&W9×30 (Carlini & Wagner, 2016) 99.9 36.9 50.5 66.3 90.0 95.1 20.4 C&W9×1000 (Carlini & Wagner, 2016) 100 71.9 87.0 96.1 99.9 100 2.6 FGSM (Goodfellow et al., 2015) 80.7 25.5 37.8 52.8 81.2 89.2 44.2 PGD10 (Madry et al., 2018) 100 3.3 6.7 12 34.7 43.9 306.5 MIFGSM10 (Dong et al., 2018) 99.9 0.7 1.9 6.0 22.5 32.3 404.4

especially when some downstream modules may rely on Top-5 predictions in their decision making.

But, the three untargeted attack approaches are much better in terms of pushing the GT labels since they are usually move against the GT label explicitly in the optimization, but their perturbation energies are usually much larger.

As we shall show, more "robust" attack methods can be developed by harnessing the advantages of the two types of attack methods.

In addition, the targeted Top-1 attack setting could limit the flexibility of attacks, and may lead to less rich perturbations.

To facilitate explicit control of targeted attacks and enable more "robust" attack methods, one natural solution, which is the focus of this paper, is to develop ordered Top-k targeted attacks which enforce the Top-k predicted labels of an adversarial example to be the k (randomly) selected and ordered labels (k ≥ 1, the GT label is exclusive).

In this paper, we present two methods of learning ordered Top-k attacks.

The basic idea is to design proper adversarial objective functions that result in imperceptible perturbations for any test image through iterative gradient-based back-propagation.

First, we extend the vanilla Carlini-Wagner (C&W) method (Carlini & Wagner, 2016) and use it as a strong baseline.

Second, we present an adversarial distillation (AD) framework consisting of two components: (i) Computing an adversarial probability distribution for any given ordered Top-k targeted labels.

(ii) Learning adversarial examples by minimizing the Kullback-Leibler (KL) divergence between the adversarial distribution and the predicted distribution, together with the perturbation energy penalty.

The proposed AD framework can be viewed as applying the network distillation frameworks (Hinton et al., 2015; Bucila et al., 2006; Papernot et al., 2016) for "the bad" induced by target adversarial distributions.

To compute a proper adversarial distribution for any given ordered Top-k targeted labels, the AD framework is motivated by two aspects: (i) The difference between the objective functions used by the C&W method and the three untargeted attack methods (Table 1) respectively.

The former maximizes the margin of the logits between the target and the runner-up (either GT or ResNet-50.

AD is better than the modified C&W method (CW * ).

The thickness represents the 2 energy (thinner is better).

Please see Sec. 4 for detail of experimental settings.

not), while the latter maximizes the cross-entropy between the prediction probabilities (softmax of logits) and the one-hot distribution of the ground-truth. (ii) The label smoothing methods Pereyra et al., 2017) , which are often used to improve the performance of DNNs by addressing the over-confidence issue in the one-hot vector encoding of labels.

More specifically, we explore how to leverage label semantic similarities in computing "smoothed" adversarial distributions, leading to knowledge-oriented attacks.

We measure label semantic similarities using the cosine distance between some off-the-shelf word2vec embedding of labels such as the pretrained Glove embedding (Pennington et al., 2014) .

Along this direction, another question of interest is further investigated: Are all Top-k targets equally challenging for an attack approach?

In experiments, we test Top-k (k = 1, 2, 5, 10) in the ImageNet-1000 (Russakovsky et al., 2015) val dataset using two popular DNNs trained with clean ImageNet-1000 train dataset, ResNet-50 (He et al., 2016) and DenseNet-121 (Huang et al., 2017) respectively.

Overall, the adversarial distillation approach obtains the best results.

It reduces the perturbation energy consistently with the same attack success rate on all the four k's, and improve the attack success rate by large margin against the modified C&W method for k = 10 (see Fig. 1 ).

We observe that Top-k targets that are distant from the GT label in terms of either label semantic distance or prediction scores of clean images are actually more difficulty to attack.

In summary, not only can ordered Top-k attacks improve the "robustness" of attacks, but also they provide insights on how aggressive adversarial attacks can be (under affordable optimization budgets).

Our Contributions.

This paper makes three main contributions to the field of learning adversarial attacks: (i) The problem in study is novel.

Learning ordered Top-k adversarial attacks is an important problem that reflects the robustness of attacks themselves, but has not been addressed in the literature. (ii) The proposed adversarial distillation framework is effective, especially when k is large (such as k = 5, 10). (iii) The proposed knowledge-oriented adversarial distillation is novel.

It worth exploring the existing distillation framework for a novel problem (ordered Top-k adversarial attacks) with some novel modifications (knowledge-oriented target distributions as "teachers").

The growing ubiquity of DNNs in advanced machine learning and AI systems dramatically increases their capabilities, but also increases the potential for new vulnerabilities to attacks.

This situation has become critical as many powerful approaches have been developed where imperceptible perturbations to DNN inputs could deceive a well-trained DNN, significantly altering its prediction.

Assuming full access to DNNs pretrained with clean images, white-box targeted attacks are powerful ways of investigating the brittleness of DNNs and their sensitivity to non-robust yet well-generalizing features in the data.

Distillation.

The central idea of our proposed AD method is built on distillation.

Network distillation (Bucila et al., 2006; Hinton et al., 2015 ) is a powerful training scheme proposed to train a new, usually lightweight model (a.k.a., the student) to mimic another already trained model (a.k.a.

the teacher).

It takes a functional viewpoint of the knowledge learned by the teacher as the conditional distribution it produces over outputs given an input.

It teaches the student to keep up or emulate by adding some regularization terms to the loss in order to encourage the two models to be similar directly based on the distilled knowledge, replacing the training labels.

Label smoothing can be treated as a simple hand-crafted knowledge to help improve model performance.

Distillation has been exploited to develop defense models (Papernot et al., 2016) to improve model robustness.

Our proposed adversarial distillation method utilizes the distillation idea in an opposite direction, leveraging label semantic driven knowledge for learning ordered Top-k attacks and improving attack robustness.

Adversarial Attack.

For image classification tasks using DNNs, the discovery of the existence of visually-imperceptible adversarial attacks (Szegedy et al., 2014 ) was a big shock in developing DNNs.

White-box attacks provide a powerful way of evaluating model brittleness.

In a plain and loose explanation, DNNs are universal function approximator (Hornik et al., 1989) and capable of even fitting random labels in large scale classification tasks as ImageNet-1000 (Russakovsky et al., 2015) .

Thus, adversarial attacks are generally learnable provided proper objective functions are given, especially when DNNs are trained with fully differentible back-propagation.

Many white-box attack methods focus on norm-ball constrained objective functions (Szegedy et al., 2014; Kurakin et al., 2017; Carlini & Wagner, 2016; Dong et al., 2018) .

The C&W method investigates 7 different loss functions.

The best performing loss function found by the C&W method has been applied in many attack methods and achieved strong results (Chen et al., 2017; Madry et al., 2018; .

By introducing momentum in the MIFGSM method (Dong et al., 2018) and the p gradient projection in the PGD method (Madry et al., 2018) , they usually achieve better performance in generating adversarial examples.

In the meanwhile, some other attack methods such as the StrAttack (Xu et al., 2018 ) also investigate different loss functions for better interpretability of attacks.

Our proposed method leverages label semantic knowledge in the loss function design for the first time.

In this section, we first briefly introduce the white-box attack setting and the widely used C&W method (Carlini & Wagner, 2016) under the Top-1 protocol, to be self-contained.

Then we define the ordered Top-k attack formulation.

To learn ordered Top-k attacks, we present detail of a modified C&W method as a strong baseline and the proposed AD framework.

We focus on classification tasks using DNNs.

Denote by (x, y) a pair of a clean input x ∈ X and its ground-truth label y ∈ Y. For example, in the ImageNet-1000 classification task, x represents a RGB image defined in the lattice of 224×224 and we have X R 3×224×224 .

y is the category label and we have Y {1, · · · , 1000}. Let f (·; Θ) be a DNN pretrained on clean training data where Θ collects all estimated parameters and is fixed in learning adversarial examples.

For notation simplicity, we denote by f (·) a pretrained DNN.

The prediction for an input x from f (·) is usually defined using softmax function by,

where P ∈ R |Y| represents the estimated confidence/probability vector (P c ≥ 0 and c P c = 1) and z(x) is the logit vector.

The predicted label is then inferred byŷ = arg max c∈[1,|Y|] P c .

The traditional Top-1 protocol of learning targeted attacks.

For an input (x, y), given a target label t = y, we seek to compute some visually-imperceptible perturbation δ(x, t, f ) using the pretrained and fixed DNN f (·) under the white-box setting.

White-box attacks assume the complete knowledge of the pretrained DNN f , including its parameter values, architecture, training method, etc.

The perturbed example is defined by,

which is called an adversarial example of x if t =ŷ = arg max c f (x ) c and the perturbation δ(x, t, f ) is sufficiently small according to some energy metric.

The C&W Method (Carlini & Wagner, 2016) .

Learning δ(x, t, f ) under the Top-1 protocol is posed as a constrained optimization problem (Athalye & Sutskever, 2017; Carlini & Wagner, 2016) ,

n , where E(·) is defined by a p norm (e.g., the 2 norm) and n the size of the input domain (e.g., the number of pixels).

To overcome the difficulty (non-linear and non-convex constraints) of directly solving Eqn.

3, the C&W method expresses it in a different form by designing some loss functions L(x ) = L(x + δ) such that the first constraint t = arg max c f (x ) c is satisfied if and only if L(x ) ≤ 0.

The best loss function proposed by the C&W method is defined by the hinge loss,

which induces penalties when the logit of the target label is not the maximum among all labels.

Then, the learning problem is formulated by,

subject to x + δ ∈ [0, 1] n , which can be solved via back-propagation with the constraint satisfied via introducing a tanh layer.

For the trade-off parameter λ, a binary search will be performed during the learning (e.g., 9 × 1000).

It is straightforward to extend Eqn.

3 for learning ordered Top-k attacks (k ≥ 1).

Denote by (t 1 , · · · , t k ) the ordered Top-k targets (t i = y).

We have, minimize

subject to t i = arg max

Directly solving Eqn.

6 is a difficulty task and proper loss functions are entailed, similar in spirit to the approximation approaches used in the Top-1 protocol, to ensure the first constraint can be satisfied once the optimization is converged (note that the optimization may fail, i.e., attacks fail).

3.3 LEARNING ORDERED TOP-k ATTACKS 3.3.1 A MODIFIED C&W METHOD We can modify the loss function (Eqn.

4) of the C&W method accordingly to solve Eqn.

6.

We have,

which covers the vanilla C&W loss (Eqn.

4), i.e., when k = 1,

CW (x ).

The C&W loss function does not care where the underlying GT label will be as long as it is not in the Top-k.

On the one hand, it is powerful in terms of attack success rate.

On the other hand, the GT label may be very close to the Top-k, leading to over-confident attacks (see Tabel.

1).

In addition, it is generic for any given Top-k targets.

As we will show, they are less effective if we select the Top-k targets from the sub-set of labels which are least like the ground-truth label in terms of label semantics.

To overcome the shortcomings of the C&W loss function and In our adversarial distillation framework, we adopt the view of point proposed in the network distillation method (Hinton et al., 2015) that the full confidence/probability distribution summarizes the knowledge of a trained DNN.

We hypothesize that we can leverage the network distillation framework to learn the ordered Top-k attacks by designing a proper adversarial probability distribution across the entire set of labels that satisfies the specification of the given ordered Top-k targets, and facilitates explicit control of placing the GT label, as well as top-down integration of label semantics.

Consider a given set of Top-k targets, {t 1 , · · · , t k }, denoted by P AD the adversarial probability distribution in which P AD ti > P AD tj (∀i < j) and P AD ti

The space of candidate distributions are huge.

We present a simple knowledge-oriented approach to define the adversarial distribution.

We first specify the logit distribution and then compute the probability distribution using softmax.

Denote by Z the maximum logit (e.g., Z = 10 in our experiments).

We define the adversarial logits for the ordered Top-k targets by, z

where γ is an empirically chosen decreasing factor (e.g., γ = 0.3 in our experiments).

For the remaining categories l / ∈ {t 1 , · · · , t k }, we define the adversarial logit by,

where 0 ≤ α < z AD t k is the maximum logit that can be assigned to any j, s(a, b) is the semantic similarity between the label a and label b, and is a small position for numerical consideration (e.g., = 1e-5).

We compute s(a, b) using the cosine distance between the Glove (Pennington et al., 2014) embedding vectors of category names and −1 ≤ s(a, b) ≤ 1.

Here, when α = 0, we discard the semantic knowledge and treat all the remaining categories equally.

Note that our design of P AD is similar in spirit to the label smoothing technique and its variants Pereyra et al., 2017 ) except that we target attack labels and exploit label semantic knowledge.

The design choice is still preliminary, although we observe its effectiveness in experiments.

We hope this can encourage more sophisticated work to be explored.

With the adversarial probability distribution P AD defined above as the target, we use the KL divergence as the loss function in our adversarial distillation framework as done in network distillation (Hinton et al., 2015) and we have, L (k)

and then we follow the same optimization scheme as done in the C&W method (Eqn.

5).

In this section, we evaluate ordered Top-k attacks with k = 1, 2, 5, 10 in the ImageNet-1000 benchmark (Russakovsky et al., 2015) using two pretrained DNNs, ResNet-50 (He et al., 2016) and DenseNet-121 (Huang et al., 2017 ) from the PyTorch model zoo 1 .

We implement our method using the AdverTorch toolkit 2 .

Our source code will be released.

Data.

In ImageNet-1000 (Russakovsky et al., 2015) , there are 50, 000 images for validation.

To study attacks, we utilize the subset of images for which the predictions of both the ResNet-50 and DenseNet-121 are correct.

To reduce the computational demand, we randomly sample a smaller subset, as commonly done in the literature.

We first randomly select 500 categories to enlarge the coverage of categories, and then randomly chose 2 images per selected categories, resulting in 1000 test images in total.

Settings.

We follow the protocol used in the C&W method.

We only test 2 norm as the energy penalty for perturbations in learning (Eqn.

5).

But, we evaluate learned adversarial examples in terms of three norms ( 1 , 2 and ∞ ).

We test two search schema for the trade-off parameter λ in optimization: both use 9 steps of binary search, and 30 and 1000 iterations of optimization are performed for each trial of λ.

In practice, computation budget is an important factor and less computationally expensive ones are usually preferred.

Only α = 1 is used in Eqn.

9 in experiments for simplicity due to computational demand.

We compare the results under three scenarios proposed in the C&W method (Carlini & Wagner, 2016) : The Best Case settings test the attack against all incorrect classes, and report the target class(es) that was least difficult to attack.

The Worst Case settings test the attack against all incorrect classes, and report the target class(es) that was most difficult to attack.

The Average Case settings select the target class(es) uniformly at random among the labels that are not the GT.

We first test ordered Top-k attacks using ResNet-50 for the four selected k's.

Table.

2 summarizes the quantitative results and comparisons.

For Top-10 attacks, the proposed AD method obtains significantly better results in terms of both ASR and the 2 energy of the added perturbation.

For example, the proposed AD method has relative 362.3% ASR improvement over the strong C&W baseline for the worst case setting.

For Top-5 attacks, the AD method obtains significantly better results when the search budget is relatively low (i.e., 9 × 30).

For Top-k (k = 1, 2) attacks, both the C&W method and the AD method can achieve 100% ASR, but the AD method has consistently lower energies of the added perturbation, i.e., finding more effective attacks and richer perturbations.

Fig. 2 shows some learned adversarial examples of ordered Top-10 and Top-5 attacks.

Table 2 : Results and comparisons under the ordered Top-k targeted attack protocol using randomly selected and ordered 10 targets (GT exclusive) in ImageNet using ResNet-50.

For Top-1 attacks, we also compare with three state-of-the-art untargeted attack methods, FGSM (Goodfellow et al., 2015) , PGD (Madry et al., 2018) and MIFGSM (Dong et al., 2018) .

10 iterations are used for both PGD and MIFGSM.

Intuitively, we understand that they should not be equally difficult.

We conduct some experiments to test this hypothesis.

In particular, we test whether the label semantic knowledge can help identify the weak spots of different attack methods, and whether the proposed AD method can gain more in those weak spots.

We test Top-5 using ResNet-50 3 .

Table.

3 summarizes the results.

We observe that for the 9 × 30 search budget, attacks are more challenging if the Top-5 targets are selected from the least-like set in terms of the label semantic similarity (see Eqn.

9), or from the lowest-score set in terms of prediction scores on clean images.

To investigate if the observations from ResNets hold for other DNNs, we also test DenseNet-121 (Huang et al., 2017) in ImageNet-1000.

We test two settings: k = 1, 5

4 .

Overall, we obtain similar results. (He et al., 2016) .

The proposed AD method has smaller perturbation energies and "cleaner" (lower-entropy) prediction distributions.

Note that for Top-10 attacks, the 9 × 30 search scheme does not work (see Table.

2).

Table 3 : Results of ordered Top-5 targeted attacks with targets being selected based on (Top) label similarity, which uses 5 most-like labels and 5 least-like labels as targets respectively, and (Bottom) prediction score of clean image, which uses 5 highest-score labels and 5-lowest score labels.

In both cases, GT labels are exclusive.

Top-1 targeted attack protocol using randomly selected and ordered 5 targets (GT exclusive).

For Top-1 attacks, we also compare with three state-of-the-art untargeted attack methods, FGSM (Goodfellow et al., 2015) , PGD (Madry et al., 2018) and MIFGSM (Dong et al., 2018) .

10 iterations are used for both PGD and MIFGSM.

This paper proposes to extend the traditional Top-1 targeted attack setting to the ordered Top-k setting (k ≥ 1) under the white-box attack protocol.

The ordered Top-k targeted attacks can improve the robustness of attacks themselves.

To our knowledge, it is the first work studying this ordered Top-k attacks.

To learn the ordered Top-k attacks, we present a conceptually simple yet effective adversarial distillation framework motivated by network distillation.

We also develop a modified C&W method as the strong baseline for the ordered Top-k targeted attacks.

In experiments, the proposed method is tested in ImageNet-1000 using two popular DNNs, ResNet-50 and DenseNet-121, with consistently better results obtained.

We investigate the effectiveness of label semantic knowledge in designing the adversarial distribution for distilling the ordered Top-k targeted attacks.

Discussions.

We have shown that the proposed AD method is generally applicable to learn ordered Top-k attacks.

But, we note that the two components of the AD framework are in their simplest forms in this paper, and need to be more thoroughly studied: designing more informative adversarial distributions to guide the optimization to learn adversarial examples better and faster, and investigating loss functions other than KL divergence such as the Jensen-Shannon (JS) divergence or the Earth-Mover distance.

On the other hand, we observed that the proposed AD method is more effective when computation budget is limited (e.g., using the 9 × 30 search scheme).

This leads to the theoretically and computationally interesting question whether different attack methods all will work comparably well if the computation budget is not limited.

Of course, in practice, we prefer more powerful ones when only limited computation budget is allowed.

Furthermore, we observed that both the modified C&W method and the AD method largely do not work in learning Top-k (k ≥ 20) attacks with the two search schema (9 × 30 and 9 × 1000).

We are working on addressing the aforementioned issues to test the Top-k (k ≥ 20) cases, thus providing a thorough empirical answer to the question: how aggressive can adversarial attacks be?

@highlight

ordered Top-k adversarial attacks