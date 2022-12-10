The information bottleneck method provides an information-theoretic method for representation learning, by training an encoder to retain all information which is relevant for predicting the label, while minimizing the amount of other, superfluous information in the representation.

The original formulation, however, requires labeled data in order to identify which information is superfluous.

In this work, we extend this ability to the multi-view unsupervised setting, in which two views of the same underlying entity are provided but the label is unknown.

This enables us to identify superfluous information as that which is not shared by both views.

A theoretical analysis leads to the definition of a new multi-view model that produces state-of-the-art results on the Sketchy dataset and on label-limited versions of the MIR-Flickr dataset.

We also extend our theory to the single-view setting by taking advantage of standard data augmentation techniques, empirically showing better generalization capabilities when compared to traditional unsupervised approaches for representation learning.

The goal of deep representation learning (LeCun et al., 2015) is to transform a raw observational input, x, into a representation, z, to extract useful information.

Significant progress has been made in deep learning via supervised representation learning, where the labels, y, for the downstream task are known while p(y|x) is learned directly (Sutskever et al., 2012; Hinton et al., 2012) .

Due to the cost of acquiring large labeled datasets, a recently renewed focus on unsupervised representation learning seeks to generate representations, z, that allow learning of (a priori unknown) target supervised tasks more efficiently, i.e. with fewer labels (Devlin et al., 2018; Radford et al., 2019) .

Our work is based on the information bottleneck principle (Tishby et al., 2000) stating that whenever a data representation discards information from the input which is not useful for a given task, it becomes less affected by nuisances, resulting in increased robustness for downstream tasks.

In the supervised setting, one can directly apply the information bottleneck method by minimizing the mutual information between z and x while simultaneously maximizing the mutual information between z and y (Alemi et al., 2017) .

In the unsupervised setting, discarding only superfluous information is more challenging as without labels one cannot directly identify the relevant information.

Recent literature (Devon Hjelm et al., 2019; van den Oord et al., 2018) has instead focused on the InfoMax objective maximizing the mutual information between x and z, I(x, z), instead of minimizing it.

In this paper, we extend the information bottleneck method to the unsupervised multi-view setting.

To do this, we rely on a basic assumption of the multi-view literature -that each view provides the same task relevant information (Zhao et al., 2017) .

Hence, one can improve generalization by discarding from the representation all information which is not shared by both views.

We do this through an objective which maximizes the mutual information between the representations of the two views (Multi-View InfoMax) while at the same time reducing the mutual information between each view and its corresponding representation (as with the information bottleneck).

The resulting representation contains only the information shared by both views, eliminating the effect of independent factors of variations.

Our contributions are three-fold: (1) We extend the information bottleneck principle to the unsupervised multi-view setting and provide a rigorous theoretical analysis of its application.

(2) We define a new model that empirically leads to state-of-the-art results in the low-data setting on two standard multi-view datasets, Sketchy and MIR-Flickr.

(3) By exploiting standard data augmentation techniques, we empirically show that the representations obtained with our model in single-view settings are more robust than other popular unsupervised approaches for representation learning, connecting our theory to the choice of augmentation function class.

The challenge of representation learning can be formulated as finding a distribution p(z|x) that maps data observations x ∈ X into a code space z ∈ Z. Whenever the end goal involves predicting a label y, we consider only the z that are discriminative enough to identify the label.

This requirement can be quantified by considering the amount of target information that remains accessible after encoding the data, and is known in literature as sufficiency of z for y (Achille & Soatto, 2018): Definition 1.

Sufficiency: A representation z of x is sufficient for y if and only if I(x; y|z) = 0

Any model that has access to a sufficient representation z must be able to predict y at least as accurately as if it has access to the original data x instead.

In fact, z is sufficient for y if and only if the amount of information regarding the task is unchanged by the encoding procedure (see Proposition B.1 in the Appendix):

I(x; y|z) = 0 ⇐⇒ I(x; y) = I(y; z).

(1)

Among sufficient representations, the ones that result in better generalization for unlabeled data instances are particularly appealing.

When x has higher information content than y, some of the information in x must be irrelevant for the prediction task.

This can be better understood by subdividing I(x; z) into three components by using the chain rule of mutual information (see Appendix A):

I(x; z) = I(x; z|y)

Conditional mutual information I(x; z|y) represents the information in z that is not predictive of y, i.e. superfluous information.

While I(x; y) is a constant determined by how much label information is accessible from the raw observations; the last term I(x; y|z) represents the amount of information regarding y that is lost by encoding x into z. Note that this last term is zero whenever z is sufficient for x. Since the amount of predictive information I(x; y) is fixed, Proposition 2.1.

A sufficient representation z of x for y is minimal whenever I(x; z|y) is minimal.

Minimizing the amount of superfluous information can be done directly only in supervised settings.

In fact, reducing I(x; z) without violating the sufficiency constraint necessarily requires making some additional assumptions on the predictive task (see Theorem B.1 in the Appendix).

In Section 3 we describe a strategy to safely reduce the information content of a representation even when the label y is not observed, by exploiting redundant information in the form of an additional view on the data.

Let v 1 and v 2 be two images of the same object from different view-points and y be its label.

Assuming that the object is clearly distinguishable from both v 1 and v 2 , any representation z with all the information that is accessible from both views would also contain the necessary label information.

Furthermore, if z captures only the details that are visible from both pictures, it would reduce the total information content, discarding the view-specific details and reducing the sensitivity of the representation to view-changes.

The theory to support this intuition is described in the following where v 1 and v 2 are jointly observed and referred to as data-views.

In this section we extend our analysis of sufficiency and robustness to the multi-view setting.

Intuitively, we can guarantee that z is sufficient for predicting y even without knowing y by ensuring that z maintains all information which is shared by v 1 and v 2 .

This intuition relies on a basic assumption of the multi-view environment -that each view provides the same task relevant information.

To formalize this we define redundancy.

Definition 2.

Redundancy: v 1 is redundant with respect to v 2 for y if and only if I(y; v 1 |v 2 ) = 0

Intuitively, a view v 1 is redundant for a task whenever it is irrelevant for the prediction of y when v 2 is already observed.

Whenever v 1 and v 2 are mutually redundant (v 1 is redundant with respect to v 2 for y, and vice-versa), we can show the following: Corollary 1.

Let v 1 and v 2 be two mutually redundant views for a target y and let z 1 be a representation of v 1 .

If z 1 is sufficient for v 2 (I(v 1 ; v 2 |z 1 ) = 0) then z 1 is as predictive for y as the joint observation of the two views (I(v 1 v 2 ; y) = I(y; z 1 )).

In other words, whenever it is possible to assume mutual redundancy, any representation which contains all the information shared by both views (the redundant information) is as useful as their joint observation for predicting the label y.

By factorizing the mutual information between v 1 and z 1 analogously to Equation 2, we can identify 3 components:

.

Since I(v 1 ; v 2 ) is a constant that depends only on the two views and I(v 1 ; v 2 |z 1 ) must be zero if we want the representation to be sufficient for the label, we conclude that I(v 1 ; z 1 ) can be reduced by minimizing I(v 1 ; z 1 |v 2 ).

This term intuitively represents the information z 1 contains which is unique to v 1 and not shared by v 2 .

Since we assumed mutual redundancy between the two views, this information must be irrelevant for the predictive task and, therefore, it can be safely discarded.

The proofs and formal assertions for the above statements and Corollary 1 can be found in Appendix B.

The less the two views have in common, the more I(z 1 ; v 1 ) can be reduced without violating sufficiency for the label, the more robust the resulting representation.

At the extreme, v 1 and v 2 share only label information, in which case we can show that z 1 is minimal for y and our method is identical to the supervised information bottleneck method without needing to access the labels.

Conversely, if v 1 and v 2 are identical, then our method degenerates to the InfoMax principle since no information can be safely discarded (see Appendix E).

Given v 1 and v 2 that satisfy the mutual redundancy condition for a label y, we would like to define an objective function for the representation z 1 of v 1 that discards as much information as possible without losing any label information.

In Section 3.1 we showed that we can maintain sufficiency for y by ensuring that I(v 1 ; v 2 |z 1 ) = 0, and that decreasing I(z 1 ; v 1 |v 2 ) will increase the robustness of the representation by discarding irrelevant information.

So if we combine these two terms using a relaxed Lagrangian objective, then we obtain:

where θ denotes the dependency on the parameters of the encoder p θ (z 1 |v 1 ), and λ 1 represents the Lagrangian multiplier introduced by the constrained optimization.

Symmetrically, we define a

Algorithm 1:

Figure 1: Visualizing our Multi-View Information Bottleneck model for both multi-view and singleview settings.

Whenever p(v 1 ) and p(v 2 ) have the same distribution, the two encoders can share their parameters.

loss L 2 to optimize the parameters ψ of a conditional distribution p ψ (z 2 |v 2 ) that defines a robust sufficient representation z 2 of the second view v 2 :

Although L 1 and L 2 can not be computed directly, by defining z 1 and z 2 on the same domain Z and re-parametrizing the Lagrangian multipliers, their sum can be upper bounded as follows:

sufficiency of z1 and z2 for predicting y

where D SKL is the symmetrized KL divergence obtained by averaging D KL (p θ (z 1 |v 1 )||p ψ (z 2 |v 2 )) and D KL (p ψ (z 2 |v 2 )||p θ (z 1 |v 1 )), while the coefficient β defines the trade-off between sufficiency and robustness of the representation, which is a hyper-parameter in this work.

The resulting MultiView Infomation Bottleneck (MIB) model (Equation 5 ) is visualized in Figure 1 , while the batchbased computation of the loss function is summarized in Algorithm 1.

The symmetrized KL divergence D SKL (p θ (z 1 |v 1 )||p ψ (z 2 |v 2 )) can be computed directly whenever p θ (z 1 |v 1 ) and p ψ (z 2 |v 2 ) have a known density, while the mutual information between the two representations I θψ (z 1 ; z 2 ) can be maximized by using any sample-based differentiable mutual information lower bound.

Both the Jensen-Shannon I JS (Devon Hjelm et al., 2019; Poole et al., 2019) and the InfoNCE I NCE (van den Oord et al., 2018) estimators used in this work require introducing an auxiliary parameteric model C ξ (z 1 , z 2 ), which is jointly optimized during the training procedure.

The full derivation for the MIB loss function can be found in Appendix F.

In this section, we introduce a methodology to build mutually redundant views starting from single observations x with domain X by exploiting known symmetries of the task.

By picking a class T of functions t : X → W that do not affect label information, it is possible to artificially build views that satisfy mutual redundancy for y with a procedure similar to dataaugmentation.

Let t 1 and t 2 be two random variables over T, then v 1 := t 1 (x) and v 2 := t 2 (x) must be mutually redundant for y. Since no function in T affects label information (I(v 1 ; y) = I(v 2 ; y) = I(x; y)), a representation z 1 of v 1 that is sufficient for v 2 must contain same amount of predictive information as x. Formal proofs can be found in Appendix B.4.

Whenever the two transformations for the same observations are independent (I(t 1 ; t 2 |x) = 0), they introduce uncorrelated variations in the two views.

As an example, if T represents a set of small translations, the two views will differ by a small shift.

Since this information is not shared, z 1 that contains only common information between v 1 and v 2 will discard fine-grained details regarding the position.

For single-view datasets, we generate the two views v 1 and v 2 by independently sampling two functions from the same function class T with uniform probability.

Since the resulting t 1 and t 2 have the same distribution, the two generated views will also have the same marginals.

For this reason, the two conditional distributions p θ (z 1 |v 1 ) and p ψ (z 2 |v 2 ) can share their parameters and only one encoder can be used.

Full (or partial) parameter sharing can be also applied in the multiview settings whenever the two views have the same (or similar) marginal distributions.

The space of all the possible representations z of x for a predictive task y can be represented as a region in the Information Plane (Tishby et al., 2000) .

Each representation is characterised by the amount of information regarding the raw observation I(x; z) and the corresponding measure of accessible predictive information I(y; z) (x and y axis respectively on Figure 2 ).

Ideally, a good representation would be maximally informative about the label while retaining a minimal amount of information from the observations (top left corner of the parallelogram).

Further details on the Information Plane and the bounds visualized in Figure 2 are described in Appendix C. Thanks to recent progress in mutual information estimation (Nguyen et al., 2008; Ishmael Belghazi et al., 2018; Poole et al., 2019) , the InfoMax principle (Linsker, 1988) has gained attention for unsupervised representation learning (Devon Hjelm et al., 2019; van den Oord et al., 2018) .

Since the InfoMax objective involves maximizing I(x; z), the resulting representation aims to preserve all the information regarding the raw observations (top right corner in Figure 2) .

Despite their success, Tschannen et al. (2019) has shown that the effectiveness of the InfoMax models is due to inductive biases introduced by the architecture and estimators rather than the training objective itself, since the InfoMax objective can be trivially maximized by using invertible encoders.

On the other hand, Variational Autoencoders (VAEs) (Kingma & Welling, 2014 ) define a training objective that balances compression and reconstruction error (Alemi et al., 2018) through an hyper-parameter β.

Whenever β is close to 0, the VAE objective aims for a lossless representation, approaching the same region of the Information Plane as the one targeted by InfoMax (Barber & Agakov, 2003) .

When β approaches large values, the representation becomes more compressed, showing increased generalization and disentanglement (Higgins et al., 2017; Burgess et al., 2018) , and, as β approaches infinity, I(z; x) goes to zero.

During this transition from low to high β, however, there are no guarantees that VAEs will retain label information (Theorem B.1 in the Appendix).

The path between the two regimes depends on how well the label information aligns with the inductive bias introduced by encoder (Jimenez Rezende & Mohamed, 2015; Kingma et al., 2016 ), prior (Tomczak & Welling, 2018 and decoder architectures (Gulrajani et al., 2017; Chen et al., 2017) .

Concurrent work applies the InfoMax principle in Multi-View settings (Ji et al., 2019; Hénaff et al., 2019; Tian et al., 2019; Bachman et al., 2019) , aiming to maximize mutual information between the representation z of a first data-view x and a second one v 2 .

The target representation for the MultiView InfoMax (MV-InfoMax) models should contain at least the amount of information in x that is predictive for v 2 , targeting the region I(z; x) ≥ I(x; v 2 ) on the Information Plane.

Whenever x is redundant with respect to v 2 for y, the representation must be also sufficient for y (Corollary 1).

Since z has no incentive in discarding any information regarding x, a representation that is opti-

Method mAP@all Prec@200 SaN (Yu et al., 2017) 0.208 0.292 GN Triplet (Sangkloy et al., 2016) 0.529 0.716 Siamese CNN (Qi et al., 2016) 0.481 0.612 Siamese-AlexNet (Liu et al., 2017) Table 1 : Examples of the two views and class label from the Sketchy dataset (on the left) and comparison between MIB and other popular models in literature on the sketch-based image retrieval task (on the right).

* denotes models that use a 64-bits binary representation.

The results for MIB corresponds to β = 1.

mal according to the InfoMax principle is also optimal for MV-InfoMax.

Our model with β = 0 (Equation 5) belong to this family of objectives since the minimality term is discarded.

In contrast to all of the above, our work is the first to explicitly identify and discard superfluous information from the representation in the unsupervised multi-view setting.

The idea of discarding irrelevant information was introduced in Tishby et al. (2000) and identified as one of the possible reasons behind the generalization capabilities of deep neural networks by Tishby & Zaslavsky (2015) and Achille & Soatto (2018) .

The direct removal of superfluous information has, so far, been done only in supervised settings (Alemi et al., 2017) .

Conversely, β-VAE models remove information indiscriminately without identifying which part is superfluous, and the InfoMax and Multi-View InfoMax methods do not explicitly try to remove superfluous information at all.

In fact, among the representations that are optimal according to Multi-View InfoMax (purple dotted line in Figure 2 ), the MIB objective results in the representation with the least superfluous information, i.e. the most robust.

In this section we demonstrate the effectiveness of our model against state-of-the-art baselines in both the multi-view and single-view setting.

In the single-view setting, we also estimate the coordinates on the Information Plane for each of the baseline methods as well as our method to validate the theory in Section 3.

The results reported in the following sections are obtained using the Jensen-Shannon I JS (Devon Hjelm et al., 2019; Poole et al., 2019) estimator, which resulted in better performance for MIB and the other InfoMax-based models (Table 2 in the supplementary material).

In order to facilitate the comparison between the effect of the different loss functions, the same estimator is used across the different models.

We compare MIB on the sketch-based image retrieval (Sangkloy et al., 2016) and Flickr multiclass image classification (Huiskes & Lew, 2008) tasks with domain specific and prior multi-view learning methods.

Sketchy The Sketchy dataset (Sangkloy et al., 2016) consists of 12,500 images and 75,471 handdrawn sketches of objects from 125 classes.

As in Liu et al. (2017) , we also include another 60,502 images from the ImageNet (Deng et al., 2009 ) from the same classes, which results in total 73,002 natural object images.

As per the experimental protocol of Zhang et al. (2018) , a total of 6,250 sketches (50 sketches per category) are randomly selected and removed from the training set for testing purpose, which leaves 69,221 sketches for training the model.

The sketch-based image retrieval task is a ranking of 73,002 natural images according to the unseen test (query) sketch.

Retrieval is done for our model by generating representations for the query sketch as well as all natural images, (Eitz et al., 2012) .

The resulting flattened 4096-dimensional feature vectors are fed to our image and sketch encoders to produce a 64-dimensional representation.

Both encoders consist of neural networks with hidden layers of 2048 and 1024 units respectively.

Size of the representation and regularization strength β are tuned on a validation sub-split.

We evaluate MIB on five different train/test splits 1 and report mean and standard deviation in Table 5 .1.

Further details on our training procedure and architecture are in Appendix G. Table 5 .1 shows that the our model achieves strong performance for both mean average precision (mAP@all) and precision at 200 (Prec@200), suggesting that the representation is able to capture the common class information between the paired pictures and sketches.

The effectiveness of MIB on the retrieval task can be mostly imputed to the regularization introduced with the symmetrized KL divergence between the two encoded views.

Other than discarding view-private information, this term actively aligns the representations of v 1 and v 2 , making the MIB model especially suitable for retrieval tasks MIR-Flickr The MIR-Flickr dataset (Huiskes & Lew, 2008) consists of 1M images annotated with 800K distinct user tags.

Each image is represented by a vector of 3,857 hand-crafted image features (v 1 ), while the 2,000 most frequent tags are used to produce a 2000-dimensional multihot encoding (v 2 ) for each picture.

The dataset is divided into labeled and unlabeled sets that respectively contain 975K and 25K images, where the labeled set also contains 38 distinct topic classes together with the user tags.

Training images with less than two tags are removed, which reduces the total number of training samples to 749,647 pairs (Sohn et al., 2014; Wang et al., 2016) .

The labeled set contains 5 different splits of train, validation and test sets of size 10K/5K/10K respectively.

Following a standard procedure in literature (Srivastava & Salakhutdinov, 2014; Wang et al., 2016) , we train our model on the unlabeled pairs of images and tags.

Then a multi-label logistic classifier is trained from the representation of 10K labeled train images to the corresponding macro-categories.

The quality of the 1 Processed dataset and splits will be publicly released on paper acceptance 2 These results are included only for completeness, as the Multi-View InfoMax objective does not produce consistent representations for the two views so there is no straight-forward way to use it for ranking.

representation is assessed based on the performance of the trained logistic classifier on the labeled test set.

Each encoder consists of a multi-layer perceptron of 4 hidden layers with ReLU activations learning two 1024-dimensional representations z 1 and z 2 for images v 1 and tags v 2 respectively.

Examples of the two views, labels, and further details on the training procedure are in Appendix G.

Our MIB model is compared with other popular multi-view learning models in Figure 3 for β = 0 (Multi-View InfoMax), β = 1 and β = 10 −3 (best on validation set).

Although the tuned MIB performs similarly to Multi-View InfoMax with a large number of labels, it outperforms it when fewer labels are available.

Furthermore, by choosing a larger β the accuracy of our model drastically increases in scarce label regimes, while slightly reducing the accuracy when all the labels are observed (see right side of Figure 3 ).

This effect is likely due to a violation of the mutual redundancy constraint (see Figure 6 in the supplementary material) which can be compensated with smaller values of β for less aggressive compression.

A possible reason for the effectiveness of MIB against some of the other baselines may be our ability to use mutual information estimators that do not require reconstruction.

Both Multi-View VAE (MVAE) and Deep Variational CCA (VCCA) rely on a reconstruction term to capture crossmodal information, which can introduce bias that decreases performance.

In this section, we compare the performance of different unsupervised learning models by measuring their data efficiency and empirically estimating the coordinates of their representation on the Information Plane.

Since accurate estimation of mutual information is extremely expensive (McAllester & Stratos, 2018) , we focus on relatively small experiments that aim to uncover the difference between popular approaches for representation learning.

The dataset is generated from MNIST by creating the two views, v 1 and v 2 , via the application of data augmentation consisting of small affine transformations and independent pixel corruption to each image.

These are kept small enough to ensure that label information is not effected.

Each pair of views is generated from the same underlying image, so no label information is used in this process (details in Appendix G).

To evaluate, we train the encoders using the unlabeled multi-view dataset just described, and then fix the representation model.

A logistic regression model is trained using the resulting representations along with a subset of labels for the training set, and we report the accuracy of this model on a disjoint test set as is standard for the unsupervised representation learning literature (Tschannen et al., 2019; Tian et al., 2019; van den Oord et al., 2018) .

We estimate I(x; z) and I(y; z) using mutual information estimation networks trained from scratch on the final representations using batches of joint samples {(

∼ p(x, y)p θ (z|x).

All models are trained using the same encoder architecture consisting of 2 layers of 1024 hidden units with ReLU activations, resulting in 64-dimensional representations.

The same data augmentation procedure was also applied for single-view architectures and models were trained for 1 million iterations with batch size B = 64.

Figure 4 summarizes the results.

The empirical measurements of mutual information reported on the Information Plane are consistent with the theoretical analysis reported in Section 4: models that retain less information about the data while maintaining the maximal amount of predictive information, result in better classification performance at low-label regimes, confirming the hypothesis that discarding irrelevant information yields robustness and more data-efficient representations.

Notably, the MIB model with β = 1 retains almost exclusively label information, hardly decreasing the classification performance when only one label is used for each data point.

In this work, we introduce Multi-View Information Bottleneck, a novel method that relies on multiple data-views to produce robust representation for downstream tasks.

Most of the multi-view literature operates under the assumption that each view is individually sufficient for determining the label (Zhao et al., 2017) , while our method only requires the weaker mutual redundancy condition outlined in Section 3, enabling it to be applied to any traditional multi-view task.

In our experiments, we compared MIB empirically against other approaches in the literature on three such tasks: sketch-based image retrieval, multi-view and unsupervised representation learning.

The strong performance obtained in the different areas show that Multi-View Information Bottleneck can be practically applied to various tasks for which the paired observations are either available or are artificially produced.

Furthermore, the positive results on the MIR-Flickr dataset show that our model can work well in practice even when mutual redundancy holds only approximately.

There are multiple extensions that we would like to explore in future work.

One interesting direction would be considering more than two views.

In Appendix D we discuss why the mutual redundancy condition cannot be trivially extended to more than two views, but we still believe such an extension is possible.

Secondly, we believe that exploring the role played by different choices of data augmentation could bridge the gap between the Information Bottleneck principle and with the literature on invariant neural networks (Bloem-Reddy & Whye Teh, 2019), which are able to exploit known symmetries and structure of the data to remove superfluous information.

In this section we enumerate some of the properties of mutual information that are used to prove the theorems reported in this work.

For any random variables w, x, y and z:

(P 1 ) Positivity:

I(x; y) ≥ 0, I(x; y|z) ≥ 0 (P 2 ) Chain rule:

= I(x; y) − I(y; z)

Since both I(x; y) and I(y; z) are non-negative (P 1 ), I(x; y|z) = 0 ⇐⇒ I(y; z) = I(x; y)

Theorem B.1.

Let x, z and y be random variables with joint distribution p(x, y, z).

Let z be a representation of x that satisfies I(x; z) > I(x; z ), then it is always possible to find a label y for which z is not predictive for y while z is.

(H 1 ) I(y; z |x) = 0

(T 1 ) I(x; z ) < I(x; z) =⇒ ∃y.

I(y; z) > I(y; z ) = 0

Proof.

By construction.

1.

We first factorize x as a function of two independent random variables (Proposition 2.1 Achille & Soatto (2018)) by picking y such that:

for some deterministic function f .

Note that such y always exists.

2.

Since x is a function of y and z :

(C 4 ) I(x; z|yz ) = 0

Considering I(y; z):

= I(y; z|x) + I(x; y; z) Whenever I(x; z) > I(x; z ), I(y; z) must be strictly positive, while I(y; z ) = 0 by construction.

Therefore such y exists.

Corollary B.1.1.

Let z be a representation of x that discards observational information.

There is always a label y for which a z is not predictive, while the original observations are.

Hypothesis:

Thesis:

(T 1 ) ∃y.

I(y; x) > I(y; z ) = 0

Proof.

By construction using Theorem B.1.

1.

Set z = x:

Since the hypothesis are met, we conclude that there exist y such that I(y; x) > I(y; z ) = 0

Hypothesis:

(H 1 ) I(y; z 1 |v 2 v 1 ) = 0

Thesis:

Proof.

Since z 1 is a representation of v 1 :

Therefore:

Proposition B.3.

Let v 1 be a redundant view with respect to v 2 for y. Any representation z 1 of v 1 that is sufficient for v 2 is also sufficient for y.

(H 1 ) I(y; z 1 |v 2 v 1 ) = 0 (H 2 ) I(y; v 1 |v 2 ) = 0

Thesis:

Proof.

Using the results from Theorem B.2: (H 1 ) I(y; z 1 |v 1 v 2 ) = 0

Thesis:

Proof.

I(y; z 1 )

Corollary B.2.1.

Let v 1 and v 2 be mutually redundant views for y. Let z 1 be a representation of v 1 that is sufficient for v 2 .

Then:

Hypothesis:

Thesis:

Proof.

Using Theorem B.2

Since I(y; z 1 ) ≤ I(y; v 1 v 2 ) is a consequence of the data processing inequality, we conclude that I(y; z 1 ) = I(y; v 1 v 2 )

Let x and y be random variables with domain X and Y respectively.

Let T be a class of functions t : X → W and let t 1 and t 2 be a random variables over T that depends only on x. For the theorems and corollaries discussed in this section, we are going to consider the independence assumption that can be derived from the graphical model G reported in Figure 5 .

Figure 5: Visualization of the graphical model G that relates the observations x, label y, functions used for augmentation t 1 , t 2 and the representation z 1 .

Proposition B.4.

Whenever I(t 1 (x); y) = I(t 2 (x); y) = I(x; y) the two views t 1 (x) and t 2 (x) must be mutually redundant for y.

(H 1 ) Independence relations determined by G Thesis:

(T 1 ) I(t 1 (x); y) = I(t 2 (x); y) = I(x; y) =⇒ I(t 1 (x); y|t 2 (x)) + I(t 2 (x); y|t 1 (x)) = 0 Proof.

(C 1 ) I(t 1 (x); y|xt 2 (x)) = 0 (C 2 ) I(y; t 2 (x)|x) = 0 2.

Since t 2 (x) is uniquely determined by x and t 2 :

(C 3 ) I(t 2 (x); y|xt 2 ) = 0 3.

Consider I(t 1 (x); y|t 2 (x)) I(t 1 (x); y|t 2 (x)) (P3) = I(t 1 (x); y|xt 2 (x)) + I(t 1 (x); y; x|t 2 (x)) (C1) = I(t 1 (x); y; x|t 2 (x)) (P3) = I(y; x|t 2 (x)) − I(y; x|t 1 (x)t 2 (x)) (P1) ≤ I(y; x|t 2 (x)) (P3) = I(y; x) − I(y; x; t 2 (x)) (P3) = I(y; x) − I(y; t 2 (x)) + I(y; t 2 (x)|x) (P3) = I(y; x) − I(y; t 2 (x)) + I(y; t 2 (x)|t 2 x) + I(y; t 2 (x); t 2 |x) (C3) = I(y; x) − I(y; t 2 (x)) + I(y; t 2 (x); t 2 |x) (P3) = I(y; x) − I(y; t 2 (x)) + I(y; t 2 (x)|x) − I(y; t 2 (x)|t 2 x) (P1) ≥ I(y; x) − I(y; t 2 (x)) + I(y; t 2 (x)|x) (C2) ≥ I(y; x) − I(y; t 2 (x)) Therefore I(y; x) = I(y; t 2 (x)) =⇒ I(t 1 (x); y|t 2 (x)) = 0

The proof for I(y; x) = I(y; t 1 (x)) =⇒ I(t 2 (x); y|t 1 (x)) = 0 is symmetric, therefore we conclude I(t 1 (x); y) = I(t 2 (x); y) = I(x; y) =⇒ I(t 1 (x); y|t 2 (x)) + I(t 2 (x); y|t 1 (x)) = 0 Theorem B.3.

Let I(t 1 (x); y) = I(t 2 (x); y) = I(x; y).

Let z 1 be a representation of t 1 (x) .

If z 1 is sufficient for t 2 (x) then I(x; y) = I(y; z 1 ).

(H 1 ) Independence relations determined by G (H 2 ) I(t 1 (x); y) = I(t 2 (x); y) = I(x; y) Thesis:

(T 1 ) I(t 1 (x); t 2 (x)|z 1 ) = 0 =⇒ I(x; y) = I(y; z 1 ) Proof.

Since t 1 (x) is redundant for t 2 (x) (Proposition B.4) any representation z 1 of t 1 (x) that is sufficient for t 2 (x) must also be sufficient for y (Theorem B.2).

Using Proposition B.1 we have I(y; z 1 ) = I(y; t 1 (x)).

Since I(y; t 1 (x)) = I(y; x) by hypothesis, we conclude I(x; y) = I(y; z 1 )

Every representation z of x must satisfy the following constraints:

• 0 ≤ I(y; z) ≤ I(x; y): The amount of label information ranges from 0 to the total predictive information accessible from the raw observations I(x; y).

• I(y; z) ≤ I(x; z) ≤ I(y; z) + H(x|y): The representation must contain more information about the observations than about the label.

When x is discrete, the amount of discarded label information I(x; y) − I(y; z) must be smaller than the amount of discarded observational information H(x) − I(x; z), which implies I(x; z) ≤ I(y; z) + H(x|y).

Proof.

Since z is a representation of x:

Considering the four bounds separately:

= H(x|y) + I(y; z)

Note that (H 2 ) is needed only to prove bound 4.

For continuous x bounds 1, 2 and 3 still hold.

The mutual redundancy condition between two views v 1 and v 2 for a label y can not be trivially extended to an arbitrary number of views, as the relation is not transitive because of some higher order interaction between the different views and the label.

This can be shown with a simple example.

Given three views v 1 , v 2 and v 3 and a task y such that:

• v 1 and v 2 are mutually redundant for y • v 2 and v 3 are mutually redundant for y Then, v 1 is not necessarily mutually redundant with respect to v 3 for y.

We can show this with a simple example, Let v 1 , v 2 and v 3 be fair and independent binary random variables.

Defining y as the exclusive or of v 1 and v 3 ( y := v 1 XOR v 3 ), we have that I(v 1 ; y) = I(v 3 ; y) = 0.

In this settings, v 1 and v 2 are mutually redundant for y:

Analogously, v 2 and v 3 are also mutually redundant for y as the three random variables are not predictive for each other.

Nevertheless, v 1 and v 3 and not mutually redundant for y:

Where H(v 1 |v 3 y) = H(v 3 |v 1 y) = 0 follows from v 1 = v 3 XOR y and v 3 = v 1 XOR y, while H(v 1 ) = H(v 3 ) = 1 holds by construction.

This counter-intuitive higher order interaction between multiple views makes our theory non-trivial to generalize to more than two views, requiring an extension of our theory to ensure sufficiency for the label.

Different objectives in literature can be seen as a special case of the Multi-View Information Bottleneck principle.

In this section we show that the supervised version of Information Bottleneck is equivalent to the corresponding Multi-View version whenever the two redundant views have only label information in common.

A second subsection show equivalence between InfoMax and MultiView Information Bottleneck whenever the two views are identical.

Whenever the two mutually redundant views v 1 and v 2 have only label information in common (or when one of the two views is the label itself) the Multi-View Information Bottleneck objective is equivalent to the respective supervised version.

This can be shown by proving that I(v 1 ; z 1 |v 2 ) = I(v 1 ; z 1 |y), i.e. a representation z 1 of v 1 that is sufficient and minimal for v 2 is also sufficient and minimal for y.

Proposition E.1.

Let v 1 and v 2 be mutually redundant views for a label y that share only label information.

Then a sufficient representation z 1 of v 1 for v 2 that is minimal for v 2 is also a minimal representation for y.

Hypothesis:

Thesis:

Proof.

1.

Consider I(v 1 ; z):

= I(v 1 ; z 1 |v 2 ) + I(v 1 ; y)

2.

Using Corollary 1, from (H 2 ) and (H 3 ) follows I(v 1 ; y|z 1 ) = 0 3.

I(v 1 ; z) can be alternatively expressed as:

= I(v 1 ; z 1 |y) + I(v 1 ; y)

Equating 1 and 3, we conclude I(v 1 ; z 1 |v 2 ) = I(v 1 ; z 1 |y).

Whenever v 1 = v 2 , a representation z 1 of v 1 that is sufficient for v 1 must contain all the original information.

Furthermore since I(v 1 ; z 1 |v 1 ) = 0 for every representation, no superfluous information can be identified and removed.

As a consequence, a minimal sufficient representation z 1 of v 1 for v 1 is any representation for which mutual information is maximal, hence InfoMax.

Starting from Equation 3, we consider the sum of the losses L 1 (θ; λ 1 ) and L 2 (ψ; λ 2 ) that aim to create the minimal sufficient representations z 1 and z 2 respectively:

Considering z 1 and z 2 on the same domain Z, I θ (v 1 ; z 1 |v 2 ) can be expressed as:

Note that the bound is tight whenever p ψ (z 2 |v 2 ) coincides with p θ (z 1 |v 2 ).

This happens whenever z 1 and z 2 produce a consistent encoding.

Analogously I ψ (v 2 ; z 2 |v 1 ) is upper bounded by D KL (p ψ (z 2 |v 2 )||p θ (z 1 |v 1 )).

I θ (v 1 ; v 2 |z 1 ) can be rephrased as:

* follows from z 2 representation of v 2 .

The bound reported in this equation is tight whenever z 2 is sufficient for z 1 .

This happens whenever z 2 contains all the information regarding z 1 (and therefore v 1 ).

Once again, the same bound can symmetrically be used to define I θ (v 1 ; v 2 |z 2 ) ≤ I(v 1 ; v 2 ) − I θψ (z 1 ; z 2 ).

Since I(v 1 ; v 2 ) is constant in θ and ψ, the loss function in Equation 6 can be upper-bounded with;

Where:

Lastly, multiplying both terms with β := 2 λ1+λ2 and re-parametrizing the objective, we obtain:

G EXPERIMENTAL PROCEDURE AND DETAILS

The two stochastic encoders p θ (z 1 |v 1 ) and p ψ (z 2 |v 2 ) are modeled by Normal distributions parametrized with neural networks (µ θ , σ 2 θ ) and (µ ψ , σ 2 ψ ) respectively:

Since the density of the two encoders can be evaluated, the symmetrized KL-divergence in equation 4 can be directly computed.

On the other hand, I θψ (z 1 ; z 2 ) requires the use of a mutual information estimator.

To facilitate the optimization, the hyper-parameter β is slowly increased during training, starting from a small value ≈ 10 −4 to its final value with an exponential schedule.

This is because the mutual information estimator is trained together with the other architectures and, since it starts from a random initialization, it requires an initial warm-up.

Starting with bigger β results in the encoder collapsing into a fixed representation.

The update policy for the hyper-parameter during training has not shown strong influence on the representation, as long as the mutual information estimator network has reached full capacity.

All the experiments have been performed using the Adam optimizer with a learning rate of 10

• Input:

The two views for the sketch-based classification task consist of 4096 dimensional sketch and image features extracted from two distinct VGG-16 network models which were pre-trained on images and sketches from the TU-Berlin dataset Eitz et al. (2012) for endto-end classification.

The feature extractors are frozen during the training procedure of for the two representations.

Each training iteration used batches of size B = 128.

• Encoder and Critic architectures: Both sketch and image encoders consist of multi-layer perceptrons of 2 hidden ReLU units of size 2,048 and 1,024 respectively with an output of size 2x64 that parametrizes mean and variance for the two Gaussian posteriors.

The critic architecture also consists of a multi layer perceptron of 2 hidden ReLU units of size 512.

• β update policy: The initial value of β is set to 10 −4 .

Starting from the 10,000 th training iteration, the value of β is exponentially increased up to 1.0 during the following 250,000 training iterations.

The value of β is then kept fixed to one until the end of the training procedure (500,000 iterations).

• Evaluation: All natural images are used as both training sets and retrieval galleries.

The 64 dimensional real outputs of sketch and image representation are compared using Euclidean distance.

For having a fair comparison other methods that rely on binary hashing (Liu et al., 2017; Zhang et al., 2018) , we used Hamming distance on a binarized representation (obtained by applying iterative quantization Gong et al. (2013) on our real valued representation).

We report the mean average precision (mAP@all) and precision at toprank 200 (Prec@200) Su et al. (2015) on both the real and binary representation to evaluate our method and compare it with prior works.

• Input: Whitening is applied to the handcrafted image features.

Batches of size B = 128 are used for each update step.

• Encoders and Critic architectures: The two encoders consists of a multi layer perceptron of 4 hidden ReLU units of size 1,024, which exactly resemble the architecture used in Figure 6 : Examples of pictures v 1 , tags v 2 and category labels y for the MIR-Flickr dataset (Srivastava & Salakhutdinov, 2014) .

As visualized is the second row, the tags are not always predictive of the label.

For this reason, the mutual redundancy assumption holds only approximately.

"watermelon", "hilarious", "chihuahua", "dog"

"animals", "dog", "food"

"colors", "cores", "centro", "comercial", "building"

"clouds", "sky", "structures" Wang et al. (2016) .

Both representations z 1 and z 2 have a size of 1,024, therefore the two architecture output a total of 2x1,024 parameters that define mean and variance of the respective factorized Gaussian posterior.

Similarly to the Sketchy experiments, the critic is consists of a multi-layer perceptron of 2 hidden ReLU units of size 512.

• β update policy: The initial value of β is set to 10 −8 .

Starting from 150000 th iteration, β is set to exponentially increase up to 1.0 (and 10 −3 ) during the following 150,000 iterations.

• Evaluation: Once the models are trained on the unlabeled set, the representation of the 25,000 labeled images is computed.

The resulting vectors are used for training and evaluating a multi-label logistic regression classifier on the respective splits.

The optimal parameters (such as β) for our model are chosen based on the performance on the validation set.

In Table 3 , we report the aggregated mean of the 5 test splits as the final value mean average precision value.

• • Encoders, Decoders and Critic architectures: All the encoders used for the MNIST experiments consist of neural networks with two hidden layers of 1,024 units and ReLU activations, producing a 2x64-dimensional parameter vector that is used to parameterize mean and variance for the Gaussian posteriors.

The decoders used for the VAE experiments also consist of the networks of the same size.

Similarly, the critic architecture used for mutual information estimation consists of two hidden layers of 1,204 units each and ReLU activations.

• β update policy: The initial value of β is set to 10 −3 , which is increased with an exponential schedule starting from the 50,000 th until 1the 50,000 th iteration.

The value of β is then kept constant until the 1,000,000 th iteration.

The same annealing policy is used to trained the different β-VAEs reported in this work.

• Evaluation: The trained representation are evaluated following the well-known protocol described in Tschannen et al. ,000 iterations.

The Jensen-Shannon mutual information lower bound is maximized during training, while the numerical estimation are computed using an energy-based bound (Poole et al., 2019; Devon Hjelm et al., 2019) .

The final values for I(x; z) and I(y; z) are computed by averaging the mutual information estimation on the whole dataset.

In order to reduce the variance of the estimator, the lowest and highest 5% are removed before averaging.

This practical detail makes the estimation more consistent and less susceptible to numerical instabilities.

In this section we include additional quantitative results and visualizations which refer to the singleview MNIST experiments reported in section 5.2.

Table 2 : Comparison of the amount of input information I(x; z), label information I(z; y), and accuracy of a linear classifier trained with different amount of labeled Examples (Ex) for the models reported in Figure 4 .

Both the results obtained using the Jensen-Shannon I JSD (Devon Hjelm et al., 2019; Poole et al., 2019) and the InfoNCE I NCE (van den Oord et al., 2018) estimators are reported.

Figure 7 reports the linear projection of the embedding obtained using the MIB model.

The latent space appears to roughly consists of ten clusters which corresponds to the different digits.

This observation is consistent with the empirical measurement of input and label information I(x; z) ≈ I(z; y) ≈ log 10, and the performance of the linear classifier in scarce label regimes.

As the cluster are distinct and concentrated around the respective centroids, 10 labeled examples are sufficient to align the centroid coordinates with the digit labels.

H ABLATION STUDIES H.1 DIFFERENT RANGES OF DATA AUGMENTATION Figure 8 visualizes the effect of different ranges of corruption probabily as data augmentation strategy to produce the two views v 1 and v 2 .

The MV-InfoMax Model does not seem to get any advantage from the use increasing amount of corruption, and it representation remains approximately in the same region of the information plane.

On the other hand, the models trained with the MIB objective are able to take advantage of the augmentation to remove irrelevant data information and the representation transitions from the top right corner of the Information Plane (no-augmentation) to the top-left.

When the amount of corruption approaches 100%, the mutual redundancy assumption is clearly violated, and the performances of MIB deteriorate.

In the initial part of the transitions between the two regimes (which corresponds to extremely low probability of corruption) the MIB models drops some label information that is quickly re-gained when pixel corruption becomes more frequent.

We hypothesize that this behavior is due to a problem with the optimization procedure, since the corruption are extremely unlikely, the Monte-Carlo estimation for the symmetrized Kullback-Leibler divergence is more biased.

Using more examples of views produced from the same data-point within the same batch could mitigate this issue.

The hyper-parameter β (Equation 5) determines the trade-off between sufficiency and minimality of the representation for the second data view.

When β is zero, the training objective of MIB is equivalent to the Multi-View InfoMax target, since the representation has no incentive to discard any information.

When 0 < β ≤ 1 the sufficiency constrain is enforced, while the superfluous information is gradually removed from the representation.

Values of β > 1 can result in representations that violate the sufficiency constraint, since the minimization of I(x; z|v 2 ) is prioritized.

The trade-off resulting from the choice of different β is visualized in Figure 9 and compared against β-VAE.

Note that in each point of the pareto-front the MIB model results in a better trade-off between I(x; z) and I(y; z) when compared to β-VAE.

The effectiveness of the Multi-View Information Bottleneck model is also justified by the corresponding values of predictive accuracy.

Published as a conference paper at ICLR 2020

<|TLDR|>

@highlight

We extend the information bottleneck method to the unsupervised multiview setting and show state of the art results on standard datasets