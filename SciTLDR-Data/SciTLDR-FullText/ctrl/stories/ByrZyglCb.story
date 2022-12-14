Deep networks have recently been shown to be vulnerable to universal perturbations: there exist very small image-agnostic perturbations that cause most natural images to be misclassified by such classifiers.

In this paper, we provide a quantitative analysis of the robustness of classifiers to universal perturbations, and draw a formal link between the robustness to universal perturbations, and the geometry of the decision boundary.

Specifically, we establish theoretical bounds on the robustness of classifiers under two decision boundary models (flat and curved models).

We show in particular that the robustness of deep networks to universal perturbations is driven by a key property of their curvature: there exist shared directions along which the decision boundary of deep networks is systematically positively curved.

Under such conditions, we prove the existence of small universal perturbations.

Our analysis further provides a novel geometric method for computing universal perturbations, in addition to explaining their properties.

Despite the success of deep neural networks in solving complex visual tasks BID11 ; BID15 , these classifiers have recently been shown to be highly vulnerable to perturbations in the input space.

In BID24 , state-of-the-art classifiers are empirically shown to be vulnerable to universal perturbations: there exist very small imageagnostic perturbations that cause most natural images to be misclassified.

The existence of universal perturbation is further shown in Hendrik BID12 to extend to other visual tasks, such as semantic segmentation.

Universal perturbations fundamentally differ from the random noise regime, and exploit essential properties of deep networks to misclassify most natural images with perturbations of very small magnitude.

Why are state-of-the-art classifiers highly vulnerable to these specific directions in the input space?

What do these directions represent?

To answer these questions, we follow a theoretical approach and find the causes of this vulnerability in the geometry of the decision boundaries induced by deep neural networks.

For deep networks, we show that the key to answering these questions lies in the existence of shared directions (across different datapoints) along which the decision boundary is highly curved.

This establishes fundamental connections between geometry and robustness to universal perturbations, and thereby reveals new properties of the decision boundaries induced by deep networks.

Our aim here is to derive an analysis of the vulnerability to universal perturbations in terms of the geometric properties of the boundary.

To this end, we introduce two decision boundary models: 1) the locally flat model assumes that the first order linear approximation of the decision boundary holds locally in the vicinity of the natural images, and 2) the locally curved model provides a second order local description of the decision boundary, and takes into account the curvature information.

We summarize our contributions as follows:??? Under the locally flat decision boundary model, we show that classifiers are vulnerable to universal directions as long as the normals to the decision boundaries in the vicinity of natural images are correlated (i.e., they approximately span a low dimensional space).

This result formalizes and proves some of the empirical observations made in BID24 .???

Under the locally curved decision boundary model, the robustness to universal perturbations is instead driven by the curvature of the decision boundary; we show that the existence of shared directions along which the decision boundary is positively 1 curved implies the existence of very small universal perturbations.??? We show that state-of-the-art deep nets remarkably satisfy the assumption of our theorem derived for the locally curved model: there actually exist shared directions along which the decision boundary of deep neural networks are positively curved.

Our theoretical result consequently captures the large vulnerability of state-of-the-art deep networks to universal perturbations.??? We finally show that the developed theoretical framework provides a novel (geometric) method for computing universal perturbations, and further explains some of the properties observed in BID24 (e.g., diversity, transferability) regarding the robustness to universal perturbations.

Consider an L-class classifier f : DISPLAYFORM0 is the kth component of f (x) that corresponds to the k th class.

We define by ?? a distribution over natural images in R d .

The main focus of this paper is to analyze the robustness of classifiers to universal (image-agnostic) noise.

Specifically, we define v to be a universal noise vector ifk(x + v) =k(x) for "most" x ??? ??. Formally, a perturbation v is (??, ??)-universal, if the following two constraints are satisfied: DISPLAYFORM1 This perturbation image v is coined "universal", as it represents a fixed image-agnostic perturbation that causes label change for a large fraction of images sampled from the data distribution ??. In BID24 , state-of-the-art classifiers have been shown to be surprisingly vulnerable to this simple perturbation regime.

It should be noted that universal perturbations are different from adversarial perturbations BID29 ; BID1 , which are datapoint-specific perturbations that are sought to fool a specific image.

An adversarial perturbation is a solution to the following optimization problem DISPLAYFORM2 which corresponds to the smallest additive perturbation that is necessary to change the label of the classifierk for x. From a geometric perspective, r(x) quantifies the distance from x to the decision boundary (see FIG1 ).

In addition, due to the optimality conditions of Eq. (1), r(x) is orthogonal to the decision boundary at x + r(x), as illustrated in FIG1 .In the remainder of the paper, we analyze the robustness of classifiers to universal noise, with respect to the geometry of the decision boundary of the classifier f .

Formally, the pairwise decision boundary, when restricting the classifier to class i and j is defined by B = {z ??? R d : f i (z) ??? f j (z) = 0} (we omit the dependence of B on i, j for simplicity).

The decision boundary of the classifier hence corresponds to points in the input space that are equally likely to be classified as i or j.

In the following sections, we introduce two models on the decision boundary, and quantify in each case the robustness of such classifiers to universal perturbations.

We then show that the locally curved model better explains the vulnerability of deep networks to such perturbations.

We start here our analysis by assuming a locally flat decision boundary model, and analyze the robustness of classifiers to universal perturbations under this decision boundary model.

We specifically study the existence of a universal direction v, such that DISPLAYFORM0 where v is a vector of sufficiently small norm.

It should be noted that a universal direction (as opposed to a universal vector) is sought in Eq. (2), as this definition is more adapted to the analysis of classifiers with locally flat decision boundaries.

For example, while a binary linear classifier has a universal direction that fools all the data points, only half of the data points can be fooled with a universal vector (provided the classes are balanced) (see FIG1 ).

We therefore consider this slightly modified definition in the remainder of this section.

We start our analysis by introducing our local decision boundary model.

For x ??? R d , note that x + r(x) belongs to the decision boundary and r(x) is normal to the decision boundary at x + r(x) (see FIG1 .

A linear approximation of the decision boundary of the classifier at x + r(x) is therefore given by x + {v : r(x)T v = r(x) 2 2 }.

Under this approximation, the vector r(x) hence captures the local geometry of the decision boundary in the vicinity of datapoint x. We assume a local decision boundary model in the vicinity of datapoints x ??? ??, where the local classification region of x occurs in the halfspace r(x)T v ??? r(x) 2 2 .

Equivalently, we assume that outside of this half-space, the classifier outputs a different label thank(x).

However, since we are analyzing the robustness to universal directions (and not vectors), we consider the following condition, given by DISPLAYFORM1 (3) where B(??) is a ball of radius ?? centered at 0.

An illustration of this decision boundary model is provided in Fig. 2a .

It should be noted that linear classifiers satisfy this decision boundary model, as their decision boundaries are globally flat.

This local decision boundary model is however more general, as we do not assume that the decision boundary is linear, but rather that the classification region in the vicinity of x is included in x + {v : |r(x)T v| ??? r(x) 2 2 }.

Moreover, it should be noted that the model being assumed here is on the decision boundary of the classifier, and not an assumption on the classification function f .2 Fig. 2a provides an example of nonlinear decision boundary that satisfies this model.

In all the theoretical results of this paper, we assume that r(x) 2 = 1, for all x ??? ??, for simplicity of the exposition.

The results can be extended in a straightforward way to the case where r(x) 2 takes different values for points sampled from ??. The following result shows that classifiers following the locally flat decision boundary model are not robust to small universal perturbations, provided the normals to the decision boundary (in the vicinity of datapoints) approximately belong to a low dimensional subspace of dimension m d.

Theorem 1.

Let ?? ??? 0, ?? ??? 0.

Let S be an m dimensional subspace such that P S r(x) 2 ??? 1 ??? ?? for almost all x ??? ??,, where P S is the projection operator on the subspace.

Assume moreover that L s (x, ??) holds for almost all x ??? ??, with ?? = ??? em ??(1?????) .

Then, there exists a universal noise vector v, such that v 2 ??? ?? and DISPLAYFORM2 The proof can be found in supplementary material, and relies on the construction of a universal perturbation through randomly sampling from S. The vulnerability of classifiers to universal perturbations can be attributed to the shared geometric properties of the classifier's decision boundary in the vicinity of different data points.

In the above theorem, this shared geometric property across different data points is expressed in terms of the normal vectors r(x).

The main assumption of the above theorem is specifically that normal vectors r(x) to the decision boundary in the neighborhood of data points approximately live in a subspace S of low dimension m < d. Under this assumption, the above result shows the existence of universal perturbations of 2 norm of order ??? m. When m d, Theorem 1 hence shows that very small (compared to random noise, which scales as FORMULA2 ) universal perturbations misclassifying most data points can be found.

DISPLAYFORM3 Remark 1.

Theorem 1 can be readily applied to assess the robustness of multiclass linear classifiers to universal perturbations.

In fact, when DISPLAYFORM4 These normal vectors exactly span a subspace of dimension L ??? 1.

Hence, by applying the result with ?? = 0, and m = L ??? 1, we obtain that linear classifiers are vulnerable to universal noise, with magnitude proportional to ??? L ??? 1.

In typical problems, we have L d, which leads to very small universal directions.

Remark 2.

Theorem 1 provides a partial expalanation to the vulnerability of deep networks, provided a locally flat decision boundary is assumed.

Evidence in favor of this assumption was given through visualization of randomly chosen cross-sections in Warde-Farley et al. FORMULA2 ; .

In addition, normal vectors to the decision boundary of deep nets (near data points) have been observed to approximately span a subspace S of sufficiently small dimension in BID24 .

However, unlike linear classifiers, the dimensionality of this subspace m is typically larger than the the number of classes L, leading to large upper bounds on the norm of the universal noise, under the flat decision boundary model.

This simplified model of the decision boundary hence fails to exhaustively explain the large vulnerability of state-of-the-art deep neural networks to universal perturbations.

We show in the next section that the second order information of the decision boundary contains crucial information (curvature) that captures the high vulnerability to universal perturbations.

We now consider a model of the decision boundary in the vicinity of the data points that allows to leverage the curvature of nonlinear classifiers.

Under this decision boundary model, we study the existence of universal perturbations satisfyingk(x + v) =k(x) for most x ??? ??. the decision boundary is positively curved for many data points.

In the remaining of this section, we formally prove the existence of universal perturbations, when there exists common positively curved directions of the decision boundary.

Recalling the definitions of Sec. 2, a quadratic approximation of the decision boundary at z = x + r(x) gives x + {v : DISPLAYFORM0 , where H z denotes the Hessian of F at z, and ?? x = ???F (z) 2 r(x) 2 , with F = f i ??? f j .

In this model, the second order information (encoded in the Hessian matrix H z ) captures the curvature of the decision boundary.

We assume a local decision boundary model in the vicinity of datapoints x ??? ??, where the local classification region of x is bounded by a quadratic form.

Formally, we assume that there exists ?? > 0 where the following condition holds for almost all x ??? ??: DISPLAYFORM1 An illustration of this quadratic decision boundary model is shown in Fig. 2b .

The following result shows the existence of universal perturbations, provided a subspace S exists where the decision boundary has positive curvature along most directions of S: Theorem 2.

Let ?? > 0, ?? > 0 and m ??? N. Assume that the quadratic decision boundary model DISPLAYFORM2 where H r(x),v z = ?? T H z ?? with ?? an orthonormal basis of span(r(x), v), and S denotes the unit sphere in S. Then, there is a universal perturbation vector v such that v 2 ??? ?? and DISPLAYFORM3 The above theorem quantifies the robustness of classifiers to universal perturbations in terms of the curvature ?? of the decision boundary, along normal sections spanned by r(x), and vectors v ??? S (see Normal section U of the decision boundary, along the plane spanned by the normal vector r(x) and v. Right: Geometric interpretation of the assumption in Theorem 2.

Theorem 2 assumes that the decision boundary along normal sections (r(x), v) is locally (in a ?? neighborhood) located inside a disk of radius 1/??.

Note the difference with respect to traditional notions of curvature, which express the curvature in terms of the osculating circle at x + r(x).

The assumption we use here is more "global".

Remark 1.

We stress that Theorem 2 does not assume that the decision boundary is curved in the direction of all vectors in R d , but we rather assume the existence of a subspace S where the decision boundary is positively curved (in the vicinity of natural images x) along most directions in S. Moreover, it should be noted that, unlike Theorem 1, where the normals to the decision boundary are assumed to belong to a low dimensional subspace, no assumption is imposed on the normal vectors.

Instead, we assume the existence of a subspace S leading to positive curvature, for points on the decision boundary in the vicinity of natural images.

Remark 2.

Theorem 2 does not only predict the vulnerability of classifiers, but it also provides a constructive way to find such universal perturbations.

In fact, random vectors sampled from the subspace S are predicted to be universal perturbations (see supp.

material for more details).

In Section 5, we will show that this new construction works remarkably well for deep networks, as predicted by our analysis.

We first evaluate the validity of the assumption of Theorem 2 for deep neural networks, that is the existence of a low dimensional subspace where the decision boundary is positively curved along most directions sampled from the subspace.

To construct the subspace, we find the directions that lead to large positive curvature in the vicinity of a given set of training points {x 1 , . . .

, x n }.

We recall that principal directions v 1 , . . .

, v d???1 at a point z on the decision boundary correspond to the eigenvectors (with nonzero eigenvalue) of the matrix H t z , given by H t z = P H z P , where P denotes the projection operator on the tangent to the decision boundary at z, and H z denotes the Hessian of the decision boundary function evaluated at z Lee (2009).

Common directions with large average curvature at z i = x i + r(x i ) (where r(x i ) is the minimal perturbation defined in Eq. (1)) hence correspond to the eigenvectors of the average Hessian matrix DISPLAYFORM0 We therefore set our subspace, S c , to be the span of the first m eigenvectors of H, and show that the subspace constructed in this way satisfies the assumption of Theorem 2.

To determine whether the decision boundary is positively curved in most directions of S c (for unseen datapoints from the validation set), we compute the average curvature across random directions in S c for points on the decision boundary, i.e. z = x + r(x); the average curvature is formally given by DISPLAYFORM1 where S denotes the unit sphere in S c .

In Fig. 7 (a) , the average of ?? S (x) across points sampled from the validation set is shown (as well as the standard deviation) in function of the subspace dimension m, for a LeNet architecture BID17 trained on the CIFAR-10 dataset.

5 Observe that when the dimension of the subspace is sufficiently small, the average curvature is strongly oriented towards positive curvature, which empirically shows the existence of this subspace S c where the decision boundary is positively curved for most data points in the validation set.

This empirical evidence hence suggests that the assumption of Theorem 2 is satisfied, and that universal perturbations hence represent random vectors sampled from this subspace S c .To show this strong link between the vulnerability of universal perturbations and the positive curvature of the decision boundary, we now visualize normal sections of the decision boundary of deep networks trained on ImageNet (CaffeNet BID13 and ResNet-152 BID11 ) and CIFAR-10 (LeNet (LeCun et al., 1998) and ResNet-18 BID11 ) in the direction of their respective universal perturbations.6 Specifically, we visualize normal sections of the decision boundary in the plane (r(x), v), where v is a universal perturbation computed using the universal perturbations algorithm of BID24 .

The visualizations are shown in FIG6 and 6.

Interestingly, the universal perturbations belong to highly positively curved directions of the decision boundary, despite the absence of any geometric constraint in the algorithm to compute universal perturbations.

To fool most data points, universal perturbations hence naturally seek common directions of the embedding space, where the decision boundary is positively curved.

These directions lead to very small universal perturbations, as highlighted by our analysis in Theorem 2.

It should be noted that such highly curved directions of the decision boundary are rare, as random normal sections are comparatively flat (see FIG6 and 6, second row).

This is due to the fact that most principal curvatures are approximately zero, for points sampled on the decision boundary in the vicinity of data points.

Recall that Theorem 2 suggests a novel procedure to generate universal perturbations; in fact, random perturbations from S c are predicted to be universal perturbations.

To assess the validity of this result, Fig. 7 (b) illustrates the fooling rate of the universal perturbations (for the LeNet network on CIFAR-10) sampled uniformly at random from the unit sphere in subspace S c , and scaled to have a fixed norm (1/5th of the norm of the random noise required to fool most data points).

We assess the quality of such perturbation by further indicating in Fig. 7 (b) the fooling rate of the universal 5 The LeNet architecture we used has two convolutional layers (filters of size 5) followed by three fully connected layers.

We used SGD for training, with a step size 0.01 and a momentum term of 0.9 and weight decay of 10 ???4 .

The accuracy of the network on the test set is 78.4%.

6 For the networks on ImageNet, we used the Caffe pre-trained models https://github.com/BVLC/ caffe/wiki/Model-Zoo.

The ResNet-18 architecture was trained on the CIFAR-10 task with stochastic gradient descent with momentum and weight decay regularization.

It achieves an accuracy on the test of 94.18%.

Fooling rate of universal perturbations (on an unseen validation set) computed using random perturbations in 1) S c : the subspace of positively curved directions, and 2) S f : the subspace collecting normal vectors r(x).

The dotted line corresponds to the fooling rate using the algorithm in BID24 .

S f corresponds to the largest singular vectors corresponding to the matrix gathering the normal vectors r(x) in the training set (similar to the approach in BID24 ).perturbation computed using the original algorithm in BID24 .

Observe that random perturbations sampled from S c (with m small) provide very powerful universal perturbations, fooling nearly 85% of data points from the validation set.

This rate is comparable to that of the algorithm in BID24 , while using much less training points (only n = 100, while at least 2, 000 training points are required by Moosavi-Dezfooli et al. FORMULA2 ).

The very large fooling rates achieved with such a simple procedure (random generation in S c ) confirms that the curvature is the governing factor that controls the robustness of classifiers to universal perturbations, as analyzed in Section 4.

In fact, such high fooling rates cannot be achieved by only using the model of Section 3 (neglecting the curvature information), as illustrated in Fig. 7 (b) .

Specifically, by generating random perturbations from the subspace S f collecting normal vectors r(x) (which is the procedure that is suggested by Theorem 1 to compute universal perturbations, without taking into account second order information), the best universal perturbation achieves a fooling rate of 65%, which is significantly worse than if the curvature is used to craft the perturbation.

We further perform in Appendix C the same experiment on other architectures to verify the consistency of the results across networks.

It can be seen that, similarly to Fig. 7 (b) , the proposed approach of generating universal perturbations through random sampling from the subspace S c achieves high fooling rates (comparable to the algorithm in BID24 , and significantly higher than by using S f ).Fig 8 illustrates a universal perturbation for ImageNet, corresponding to the maximally curved shared direction (or in other words, the maximum eigenvalue of H computed using n = 200 random samples).

7 The CaffeNet architecture is used, and FIG8 also represents sample perturbed images that fool the classifier.

Just like the universal perturbation in BID24 , the perturbations are not very perceptible, and lead to misclassification of most unseen images in the validation set.

For this example on ImageNet, the fooling rate of this perturbation is 67.2% on the validation set.

This is significantly larger than the fooling rate of the perturbation computed using S f only (38%), but lower than that of the original algorithm (85.4%) proposed in BID24 .

We hypothesize that this gap for ImageNet is partially due to the small number of samples, which was made due to computational restrictions.

The existence of this subspace S c (and that universal perturbations are random vectors in S c ) further explains the high diversity of universal perturbations.

Fig. 9 illustrates different universal perturbations for CIFAR-10 computed by sampling random directions from S c .

The diversity of such perturbations justifies why re-training with perturbed images (as in Moosavi-Dezfooli et al. FORMULA2 ) does not significantly improve the robustness of such networks, as other directions in S c can still lead to universal perturbations, even if the network becomes robust to some directions.

Finally, it is interesting to note that this subspace S c is likely to be shared not only across datapoints, but also different networks (to some extent).

To support this claim, FIG1 shows the cosine of the principal angles between subspaces S

In this paper, we analyzed the robustness of classifiers to universal perturbations, under two decision boundary models: Locally flat and curved.

We showed that the first are not robust to universal directions, provided the normal vectors in the vicinity of natural images are correlated.

While this model explains the vulnerability for e.g., linear classifiers, this model discards the curvature information, which is essential to fully analyze the robustness of deep nets to universal perturbations.

The second, classifiers with curved decision boundaries, are instead not robust to universal perturbations, provided the existence of a shared subspace along which the decision boundary is positively curved (for most 7 We used m = 1 in this experiment as the matrix H is prohibitively large for ImageNet.

Figure 9 : Diversity of universal perturbations randomly sampled from the subspace S c .

The normalized inner product between two perturbations is less than 0.1. directions).

We empirically verify this assumption for deep nets.

Our analysis hence explains the existence of universal perturbations, and further provides a purely geometric approach for computing such perturbations, in addition to explaining properties of perturbations, such as their diversity.

Other authors have focused on the analysis of the robustness properties of SVM classifiers (e.g., BID33 ) and new approaches for constructing robust classifiers (based on robust optimization) Caramanis et al. FORMULA2 ; BID16 .

More recently, some have assessed the robustness of deep neural networks to different regimes such as adversarial perturbations BID29 ; BID1 , random noise , and occlusions BID28 BID5 .

The robustness of classifiers to adversarial perturbations has been specifically studied in BID29 ; BID8 ; ; BID2 ; BID0 , followed by works to improve the robustness BID20 ; BID10 BID26 ; BID3 , and attempts at explaining the phenomenon in BID8 ; BID6 ; BID30 ; BID31 .

This paper however differs from these previous works as we study universal (imageagnostic) perturbations that can fool every image in a dataset, as opposed to image-specific adversarial perturbations that are not universal across datapoints (as shown in BID24 ).

Moreover, explanations that hinge on the output of a deep network being well approximated by a linear function of the inputs f (x) = W x + b are inconclusive, as the assumption is violated even for relatively small networks.

We show here that it is precisely the large curvature of the decision boundary that causes vulnerability to universal perturbations.

Our bounds indeed show an increasing vulnerability with respect to the curvature of the decision boundary, and represent up to our knowledge the first quantitative result showing tight links between robustness and curvature.

In addition, we show empirically that the first-order approximation of the decision boundary is not sufficient to explain the high vulnerability to universal perturbations ( Fig. 7 (b) ).

Recent works have further proposed new methods for computing universal perturbations Mopuri et al. FORMULA2 ; BID14 ; instead, we focus here on an analysis of the phenomenon of vulnerability to universal perturbations, while also providing a constructive approach to compute universal perturbations leveraging our curvature analysis.

Finally, it should be noted that recent works have studied properties of deep networks from a geometric perspective (such as their expressivity BID27 ; BID22 ); our focus is different in this paper as we analyze the robustness with the geometry of the decision boundary.

Our analysis hence shows that to construct classifiers that are robust to universal perturbations, it is key to suppress this subspace of shared positive directions, which can possibly be done through regularization of the objective function.

This will be the subject of future works.

We first start by recalling a result from , which is based on BID4 .

Lemma 1.

Let v be a random vector uniformly drawn from the unit sphere S d???1 , and P m be the projection matrix onto the first m coordinates.

Then, DISPLAYFORM0 with ?? 1 (??, m) = max((1/e)?? 2/m , 1 ??? 2(1 ??? ?? 2/m ), and ?? 2 (??, m) = 1 + 2 DISPLAYFORM1 We use the above lemma to prove our result, which we recall as follows: DISPLAYFORM2 Let S be an m dimensional subspace such that P S r(x) 2 ??? 1 ??? ?? for almost all x ??? ??,, where P S is the projection operator on the subspace.

Assume moreover that L s (x, ??) holds for almost all x ??? ??, with ?? = ??? em ??(1?????) .

Then, there exists a universal noise vector v, such that v 2 ??? ?? and DISPLAYFORM3 Proof.

Define S to be the unit sphere centered at 0 in the subspace S. Let ?? = ??? em ??(1?????) , and denote by ??S the sphere scaled by ??.

We have DISPLAYFORM4 where P S orth denotes the projection operator on the orthogonal of S. Observe that (P S orth r(x)) T v = 0.

Note moreover that r(x) 2 2 = 1 by assumption.

Hence, the above expression simplifies to DISPLAYFORM5 where we have used the assumption of the projection of r(x) on the subspace S. Hence, it follows from Lemma 1 that DISPLAYFORM6 Hence, there exists a universal vector v of 2 norm ?? such that DISPLAYFORM7

Theorem 2.

Let ?? > 0, ?? > 0 and m ??? N. Assume that the quadratic decision boundary model DISPLAYFORM0 where H r(x),v z = ?? T H z ?? with ?? an orthonormal basis of span(r(x), v), and S denotes the unit sphere in S. Then, there is a universal perturbation vector v such that v 2 ??? ?? and DISPLAYFORM1 Proof.

Let x ??? ??. We have To bound the above probability by ??, we set = C ?? ??? m , where C = 2 log(2/??).

We therefore choose ?? such that DISPLAYFORM2 The solution of this second order equation gives DISPLAYFORM3 Hence, for this choice of ??, we have by construction DISPLAYFORM4 We therefore conclude that E v?????S P x????? k (x + v) =k(x) ??? 1 ??? ?? ??? ??.

This shows the existence of a universal noise vector v ??? ??S such thatk(x + v) =k(x) with probability larger than 1 ??? ?? ??? ??.

We perform here similar experiment to Fig. 7 (b) on the VGG-16 and ResNet-18 architectures.

It can be seen that, similarly to Fig. 7 (b) , the proposed approach of generating universal perturbations through random sampling from the subspace S c achieves high fooling rates (comparable to the algorithm in BID24 , and significantly higher than by using S f ).C.2 TRANSFERABILITY OF UNIVERSAL PERTURBATIONS FIG1 shows examples of normal cross-sections of the decision boundary across a fixed direction in S c , for the VGG-16 architecture (but where S c is computed for CaffeNet).

Note that the decision boundary across this fixed direction is positively curved for both networks, albeit computing this subspace for a distinct network.

The sharing of S c across different nets explains the transferability of universal perturbations observed in BID24 .

Constantine Caramanis, Shie Mannor, and Huan Xu.

Robust optimization in machine learning.

In Suvrit Sra, Sebastian Nowozin, and Stephen J Wright (eds.), Optimization for machine learning, chapter 14.

Mit Press, 2012.

<|TLDR|>

@highlight

Analysis of vulnerability of classifiers to universal perturbations and relation to the curvature of the decision boundary.

@highlight

The paper provides an interesting analysis linking the geometry of classifier decision boundaries to small universal adversarial perturbations.

@highlight

This paper discusses universal perturbations - perturbations that can mislead a trained classifier if added to most of input data points.

@highlight

The paper develops models which attempt to explain the existence of universal perturbations which fool neural networks