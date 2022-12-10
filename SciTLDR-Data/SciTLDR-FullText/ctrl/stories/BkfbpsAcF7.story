Despite their impressive performance, deep neural networks exhibit striking failures on out-of-distribution inputs.

One core idea of adversarial example research is to reveal neural network errors under such distribution shifts.

We decompose these errors into two complementary sources: sensitivity and invariance.

We show deep networks are not only too sensitive to task-irrelevant changes of their input, as is well-known from epsilon-adversarial examples, but are also too invariant to a wide range of task-relevant changes, thus making vast regions in input space vulnerable to adversarial attacks.

We show such excessive invariance occurs across various tasks and architecture types.

On MNIST and ImageNet one can manipulate the class-specific content of almost any image without changing the hidden activations.

We identify an insufficiency of the standard cross-entropy loss as a reason for these failures.

Further, we extend this objective based on an information-theoretic analysis so it encourages the model to consider all task-dependent features in its decision.

This provides the first approach tailored explicitly to overcome excessive invariance and resulting vulnerabilities.

Figure 1: All images shown cause a competitive ImageNet-trained network to output the exact same probabilities over all 1000 classes (logits shown above each image).

The leftmost image is from the ImageNet validation set; all other images are constructed such that they match the non-class related information of images taken from other classes (for details see section 2.1).

The excessive invariance revealed by this set of adversarial examples demonstrates that the logits contain only a small fraction of the information perceptually relevant to humans for discrimination between the classes.

Adversarial vulnerability is one of the most iconic failure cases of modern machine learning models BID45 ) and a prime example of their weakness in out-of-distribution generalization.

It is particularly striking that under i.i.d.

settings deep networks show superhuman performance on many tasks BID33 , while tiny targeted shifts of the input distribution can cause them to make unintuitive mistakes.

The reason for these failures and how they may be avoided or at least mitigated is an active research area BID41 BID20 BID11 .So far, the study of adversarial examples has mostly been concerned with the setting of small perturbation, or -adversaries BID23 BID35 BID38 .Perturbation-based adversarial examples are appealing because they allow to quantitatively measure notions of adversarial robustness BID9 .

However, recent work argued that the perturbation-based approach is unrealistically restrictive and called for the need of generalizing the concept of adversarial examples to the unrestricted case, including any input crafted to be misinterpreted by the learned model BID44 BID10 ).

Yet, settings beyond -robustness are hard to formalize BID19 .We argue here for an alternative, complementary viewpoint on the problem of adversarial examples.

Instead of focusing on transformations erroneously crossing the decision-boundary of classifiers, we focus on excessive invariance as a major cause for adversarial vulnerability.

To this end, we introduce the concept of invariance-based adversarial examples and show that class-specific content of almost any input can be changed arbitrarily without changing activations of the network, as illustrated in figure 1 for ImageNet.

This viewpoint opens up new directions to analyze and control crucial aspects underlying vulnerability to unrestricted adversarial examples.

The invariance perspective suggests that adversarial vulnerability is a consequence of narrow learning, yielding classifiers that rely only on few highly predictive features in their decisions.

This has also been supported by the observation that deep networks strongly rely on spectral statistical regularities BID29 , or stationary statistics BID17 to make their decisions, rather than more abstract features like shape and appearance.

We hypothesize that a major reason for this excessive invariance can be understood from an information-theoretic viewpoint of crossentropy, which maximizes a bound on the mutual information between labels and representation, giving no incentive to explain all class-dependent aspects of the input.

This may be desirable in some cases, but to achieve truly general understanding of a scene or an object, machine learning models have to learn to successfully separate essence from nuisance and subsequently generalize even under shifted input distributions.

• We identify excessive invariance underlying striking failures in deep networks and formalize the connection to adversarial examples.•

We show invariance-based adversarial examples can be observed across various tasks and types of deep network architectures.•

We propose an invertible network architecture that gives explicit access to its decision space, enabling class-specific manipulations to images while leaving all dimensions of the representation seen by the final classifier invariant.• From an information-theoretic viewpoint, we identify the cross-entropy objective as a major reason for the observed failures.

Leveraging invertible networks, we propose an alternative objective that provably reduces excessive invariance and works well in practice.

In this section, we define pre-images and establish a link to adversarial examples.

DISPLAYFORM0 with layers f i and let F i denote the network up to layer i. Further, let D : R d → {1, . . .

, C} be a classifier with D = arg max k=1,...,C sof tmax(F (x)) k .

Then, for input x ∈ R d , we define the following pre-images (i) i-th Layer pre-image: DISPLAYFORM1 DISPLAYFORM2 Moreover, the (sub-)network is invariant to perturbations ∆x which satisfy x * = x + ∆x.→ Invariance-based: DISPLAYFORM3 Figure 2: Connection between (1) invariance-based (long pink arrow) and (2) perturbation-based adversarial examples (short orange arrow).

Class distributions are shown in green and blue; dashed line is the decision-boundary of a classifier.

All adversarial examples can be reached either by crossing the decision-boundary of the classifier via perturbations, or by moving within the pre-image of the classifier to mis-classified regions.

The two viewpoints are complementary to one another and highlight that adversarial vulnerability is not only caused by excessive sensitivity to semantically meaningless perturbations, but also by excessive insensitivity to semantically meaningful transformations.

Non-trivial pre-images (pre-images containing more elements than input x) after the i-th layer occur if the chain f i • · · · • f 1 is not injective, for instance due to subsampling or non-injective activation functions like ReLU BID6 .

This accumulated invariance can become problematic if not controlled properly, as we will show in the following.

We define perturbation-based adversarial examples by introducing the notion of an oracle (e.g., a human decision-maker or the unknown input-output function considered in learning theory): DISPLAYFORM4 . .

, C} is the classifier and o : R d → {1, . . .

, C} is the oracle.(ii) Created by adversary: DISPLAYFORM5 Further, -bounded adversarial ex.

x * of x fulfill x − x * < , · a norm on R d and > 0.Usually, such examples are constructed as -bounded adversarial examples BID23 .

However, as our goal is to characterize general invariances of the network, we do not restrict ourselves to bounded perturbations.

Definition 3 (Invariance-based Adversarial Examples).

Let G denote the i-th layer, logits or the classifier (Definition 1) and let x * = x be in the G pre-image of x and and o an oracle (Definition 2).

Then, an invariance-based adversarial example DISPLAYFORM6 Intuitively, adversarial perturbations cause the output of the classifier to change while the oracle would still consider the new input x * as being from the original class.

Hence in the context ofbounded perturbations, the classifier is too sensitive to task-irrelevant changes.

On the other hand, movements in the pre-image leave the classifier invariant.

If those movements induce a change in class as judged by the oracle, we call these invariance-based adversarial examples.

In this case, however, the classifier is too insensitive to task-relevant changes.

In conclusion, these two modes are complementary to each other, whereas both constitute failure modes of the learned classifier.

When not restricting to -perturbations, perturbation-based and invariance-based adversarial examples yield the same input x * via DISPLAYFORM7 with different reference points x 1 and x 2 , see Figure 2 .

Hence, the key difference is the change of reference, which allows us to approach these failure modes from different directions.

To connect these failure modes with an intuitive understanding of variations in the data, we now introduce the notion of invariance to nuisance and semantic variations, see also BID1 .Definition 4 (Semantic/ Nuisance perturbation of an input).

Let o be an oracle (Definition 2) and DISPLAYFORM8 For example, such a nuisance perturbation could be a translation or occlusion in image classification.

Further in Appendix A, we discuss the synthetic example called Adversarial Spheres from BID20 , where nuisance and semantics can be explicitly formalized as rotation and norm scaling.

Figure 3: The fully invertible RevNet, a hybrid of Glow and iRevNet with simple readout structure.

z s represents the logits and z n the nuisance.

As invariance-based adversarial examples manifest themselves in changes which do not affect the output of the network F , we need a generic approach that gives us access to the discarded nuisance variability.

While feature nuisances are intractable to access for general architectures (see comment after Definition 1), invertible classifiers only remove nuisance variability in their final projection BID28 ).

For C < d, we denote the classifier as D : R d → {1, ..., C}. Our contributions in this section are: (1) Introduce an invertible architecture with a simplified readout structure, allowing to exactly visualize manipulations in the hidden-space, (2) Propose an analytic attack based on this architecture allowing to analyze its decision-making, (3) Reveal striking invariance-based vulnerability in competitive classifiers.

Bijective classifiers with simplified readout.

We build deep networks that give access to their decision space by removing the final linear mapping onto the class probes in invertible RevNet-classifiers and call these networks fully invertible RevNets.

The fully invertible RevNet classifier can be written as D θ = arg max k=1,...,C sof tmax(F θ (x) k ), where F θ represents the bijective network.

We denote z = F θ (x), z s = z 1,...,C as the logits (semantic variables) and z n = z C+1,...,d as the nuisance variables (z n is not used for classification).

In practice we choose the first C indices of the final z tensor or apply a more sophiscticated DCT scheme (see appendix D) to set the subspace z s , but other choices work as well.

The architecture of the network is similar to iRevNets BID28 with some additional Glow components like actnorm BID31 , squeezing, dimension splitting and affine block structure BID15 , see Figure 3 for a graphical description.

As all components are common in the bijective network literature, we refer the reader to Appendix D for exact training and architecture details.

Due to its simple readout structure, the resulting invertible network allows to qualitatively and quantitatively investigate the task-specific content in nuisance and logit variables.

Despite this restriction, we achieve performance on par with commonly-used baselines on MNIST and ImageNet, see Table 1 BID43 and two ResNet BID25 variants, as well as an iRevNet BID28 ) with a non-invertible final projection onto the logits.

Our proposed fully invertible RevNet performs roughly on par with others.

Analytic attack.

To analyze the trained models, we can sample elements from the logit pre-image by computing x met = F −1 (z s ,z n ), where z s andz n are taken from two different inputs.

We term this heuristic metameric sampling.

The samples would be from the true data distribution if the subspaces would be factorized as P (z s , z n ) = P (z s )P (z n ).

Experimentally we find that logit metamers are revealing adversarial subspaces and are visually close to natural images on ImageNet.

Thus, metameric sampling gives us an analytic tool to inspect dependencies between semantic and nuisance variables without the need for expensive and approximate optimization procedures.

Attack on adversarial spheres.

First, we evaluate our analytic attack on the synthetic spheres dataset, where the task is to classify samples as belonging to one out of two spheres with different radii.

We choose the sphere dimensionality to be d = 100 and the radii: R 1 = 1, R 2 = 10.

By training a fully-connected fully invertible RevNet, we obtain 100% accuracy.

After training we visualize the decision-boundaries of the original classifier D and a posthoc trained classifier on z n (nuisance classifier), see FIG1 .

We densely sample points in a 2D subspace, following BID20 , to visualize two cases: 1) the decision-boundary on a 2D plane spanned by two randomly chosen data points, 2) the decision-boundary spanned by metameric sample x met and reference point x. In the metameric sample subspace we identify excessive invariance of the classifier.

Here, it is possible to move any point from the inner sphere to the outer sphere without changing the classifiers predictions.

However, this is not possible for the classifier trained on z n .

Most notably, the visualized failure is not due to a lack of data seen during training, but rather due to excessive invariance of the original classifier D on z s .

Thus, the nuisance classifier on z n does not exhibit the same adversarial vulnerability in its subspace.

Figure 5: Each column shows three images belonging together.

Top row are source images from which we sample the logits, middle row are logit metamers and bottom row images from which we sample the nuisances.

Top row and middle row have the same (approximately for ResNets, exactly for fully invertible RevNets) logit activations.

Thus, it is possible to change the image content completely without changing the 10-and 1000-dimensional logit vectors respectively.

This highlights a striking failure of classifiers to capture all task-dependent variability.

Attack on MNIST and ImageNet.

After validating its potential to uncover adversarial subspaces, we apply metameric sampling to fully invertible RevNets trained on MNIST and Imagenet, see Figure 5 .

The result is striking, as the nuisance variables z n are dominating the visual appearance of the logit metamers, making it possible to attach any semantic content to any logit activation pattern.

Note that the entire 1000-dimensional feature vector containing probabilities over all ImageNet classses remains unchanged by any of the transformations we apply.

To show our findings are not a particular property of bijective networks, we attack an ImageNet trained ResNet152 with a gradientbased version of our metameric attack, also known as feature adversaries BID39 .

The attack minimizes the mean squared error between a given set of logits from one image to another image (see appendix B for details).

The attack shows the same failures for non-bijective models.

This result highlights the general relevance of our finding and poses the question of the origin of this excessive invariance, which we will analyze in the following section.

In this section we identify why the cross-entropy objective does not necessarily encourage to explain all task-dependent variations of the data and propose a way to fix this.

As shown in FIG1 , the nuisance classifier on z n uses task-relevant information not captured by the logit classifier D θ on z s (evident by its superior performance in the adversarial subspace).We leverage the simple readout-structure of our invertible network and turn this observation into a formal explanation framework using information theory: Let (x, y) ∼ D with labels y ∈ {0, 1} C .

Then the goal of a classifier can be stated as maximizing the mutual information (Cover & BID14 between semantic features z s (logits) extracted by network F θ and labels y, denoted by I(y; z s ).Adversarial distribution shift.

As the previously discussed failures required to modify input data from distribution D, we introduce the concept of an adversarial distribution shift D Adv = D to formalize these modifications.

Our first assumptions for D Adv is I D Adv (z n ; y) ≤ I D (z n ; y).

Intuitively, the nuisance variables z n of our network do not become more informative about y. Thus, the distribution shift may reduce the predictiveness of features encoded in z s , but does not introduce or increase the predictive value of variations captured in z n .

Second, we assume I D Adv (y; z s |z n ) ≤ I D Adv (y; z s ), which corresponds to positive or zero interaction information, see e.g. BID18 .

While the information in z s and z n can be redundant in this assumption, synergetic effects where conditioning on z n increase the mutual information between y and z s are excluded.

Bijective networks F θ capture all variations by design which translates to information preservation I(y; x) = I(y; F θ (x)), see BID32 .

Consider the reformulation I(y; x) = I(y; F θ (x)) = I(y; z s , z n ) = I(y; z s ) + I(y; z n |z s ) = I(y; z n ) + I(y; z s |z n )by the chain rule of mutual information (Cover & BID14 , where I(y; z n |z s ) denotes the conditional mutual information.

Most strikingly, equation 5 offers two ways forward:1.

Direct increase of I(y; z s )2.

Indirect increase of I(y; z s |z n ) via decreasing I(y; z n ).Usually in a classification task, only I(y; z s ) is increased actively via training a classifier.

While this approach is sufficient in most cases, expressed via high accuracies on training and test data, it may fail under D Adv .

This highlights why cross-entropy training may not be sufficient to overcome excessive semantic invariance.

However, by leveraging the bijection F θ we can minimize the unused information I(y; z n ) using the intuition of a nuisance classifier.

Definition 5 (Independence cross-entropy loss).

Let DISPLAYFORM0 be the nuisance classifier with θ nc ∈ R p2 .

Then, the independence cross-entropy loss is defined as: DISPLAYFORM1 .The underlying principles of the nuisance classification loss L nCE can be understood using a variational lower bound on mutual information from BID5 .

In summary, the minimization is with respect to a lower bound on I D (y; z n ), while the maximization aims to tighten the bound (see Lemma 10 in Appendix C).

By using these results, we now state the main result under the assumed distribution shift and successful minimization (proof in Appendix C.1):Theorem 6 (Information I DAdv (y; z s ) maximal after distribution shift).

Let D Adv denote the adversarial distribution and D the training distribution.

Assume I D (y; z n ) = 0 by minimizing L iCE and the distribution shift satisfies I D Adv (z n ; y) ≤ I D (z n ; y) and Under distribution D, the iCE-loss minimizes I(y; z n ) (Lemma 10, Appendix C), but has no effect as the CE-loss already maximizes I(y; z s ).

However under the shift to D Adv , the information I(y; z s ) decreases when training only under the CE-loss (orange arrow), while the iCE-loss induces I(y; z n ) = 0 and thus leaves I(y; z s ) unchanged (Theorem 6).

DISPLAYFORM2

Thus, incorporating the nuisance classifier allows for the discussed indirect increase of I D Adv (y; z s ) under an adversarial distribution shift, visualized in Figure 6 .To aid stability and further encourage factorization of z s and z n in practice, we add a maximum likelihood term to our independence cross-entropy objective as DISPLAYFORM0 where det(J x θ ) denotes the determinant of the Jacobian of F θ (x) and p k ∼ N (β k , γ k ) with β k , γ k learned parameter.

The log-determinant can be computed exactly in our model with negligible additional cost.

Note, that optimizing L M LEn on the nuisance variables together with L sCE amounts to maximum-likelihood under a factorial prior (see Lemma 11 in Appendix C).Just as in GANs the quality of the result relies on a tight bound provided by the nuisance classifier and convergence of the MLE term.

Thus, it is important to analyze the success of the objective after training.

We do this by applying our metameric sampling attack, but there are also other ways like evaluating a more powerful nuisance classifier after training.

In this section, we show that our proposed independence cross-entropy loss is effective in reducing invariance-based vulnerability in practice by comparing it to vanilla cross-entropy training in four aspects: (1) error on train and test set, (2) effect under distribution shift, perturbing nuisances via metameric sampling, (3) evaluate accuracy of a classifier on the nuisance variables to quantify the class-specific information in them and (4) on our newly introduced shiftMNIST, an augmented version of MNIST to benchmark adversarial distribution shifts according to Theorem 6.For all experiments we use the same network architecture and settings, the only difference being the two additional loss terms as explained in Definition 5 and equation 6.

In terms of test error of the logit classifier, both losses perform approximately on par, whereas the gap between train and test error vanishes for our proposed loss function, indicating less overfitting.

For classification errors see TAB3 in appendix D.

To analyze if our proposed loss indeed leads to independence between z n and labels y, we attack it with our metameric sampling procedure.

As we are only looking on data samples and not on samples from the model (factorized gaussian on nuisances), this attack should reveal if the network learned to trick the objective.

In FIG2 we show interpolations between original images and logit metamers in CE-and iCE-trained fully invertible RevNets.

In particular, we are holding the activations z s constant, while linearly interpolating nuisances z n down the column.

The CE-trained network allows us to transform any image into any class without changing the logits.

However, when training with our proposed iCE, the picture changes fundamentally and interpolations in the pre-image only change the style of a digit, but not its semantic content.

This shows our loss has the ability to overcome excessive task-related invariance and encourages the model to explain and separate all task-related variability of the input from the nuisances of the task.

−1 (z s ,z n ) with logit activations z s taken from original image andz n obtained by linearly interpolating from the original nuisance z n (first row) to the nuisance of a target example z * n (last row upper block).

The used target example is shown at the bottom.

When training with cross-entropy, virtually any image can be turned into any class without changing the logits z s , illustrating strong vulnerability to invariance-based adversaries.

Yet, training with independence cross-entropy solves the problem and interpolations between nuisances z n and z * n preserve the semantic content of the image.

A classifier trained on the nuisance variables of the cross-entropy trained model performs even better than the logit classifier.

Yet, a classifier on the nuisances of the independence cross-entropy trained model is performing poorly (Table 2 in appendix D).

This indicates little class-specific information in the nuisances z n , as intended by our objective function.

Note also that this inability of the nuisance classifier to decode class-specific information is not due to it being hard to read out from z n , as this would be revealed by the metameric sampling attack (see FIG2 .

At test time, the binary code is not present and the network can not rely on it anymore.

(b) Textured shiftMNIST introduces textured backgrounds for each digit category which are patches sampled from the describable texture dataset BID13 .

DISPLAYFORM0 At train time the same type of texture is underlayed each digit of the same category, while texture types across categories differ.

At test time, the relationship is broken and texture backgrounds are paired with digits randomly, again minimizing the mutual information between background and label in a targeted manner.

See Figure 8 for examples 1 .It turns out that this task is indeed very hard for standard classifiers and their tendency to become excessively invariant to semantically meaningful features, as predicted by our theoretical analysis.

When trained with cross-entropy, ResNets and fi-RevNets make zero errors on the train set, while having error rates of up to 87% on the shifted test set.

This is striking, given that e.g. in binary shiftMNIST, only one single pixel is removed under D Adv , leaving the whole image almost unchanged.

When applying our independence cross-entropy, the picture changes again.

The errors made by the network improve by up to almost 38% on binary shiftMNIST and around 28% on textured shiftMNIST.

This highlights the effectiveness of our proposed loss function and its ability to minimize catastrophic failure under severe distribution shifts exploiting excessive invariance.

Adversarial examples.

Adversarial examples often include -norm restrictions BID45 , while BID19 argue for a broader definition to fully capture the implications for security.

The -adversarial examples have also been extended to -feature adversaries BID39 , which are equivalent to our approximate metameric sampling attack.

Some works BID44 BID16 consider unrestricted adversarial examples, which are closely related to invariance-based adversarial vulnerability.

The difference to human perception revealed by adversarial examples fundamentally questions which statistics deep networks use to base their decisions BID29 BID47 .Relationship between standard and bijective networks.

We leverage recent advances in reversible BID21 and bijective networks BID28 BID4 BID31 for our analysis.

It has been shown that ResNets and iRevNets behave similarly on various levels of their representation on challenging tasks BID28 and that iRevNets as well as Glow-type networks are related to ResNets by the choice of dimension splitting applied in their residual blocks BID24 .

Perhaps unsurprisingly, given so many similarities, ResNets themselves have been shown to be provably bijective under mild conditions BID7 .

Further, excessive invariance of the type we discuss here has been shown to occur in non residual-type architectures as well BID20 BID6 .

For instance, it has been observed that up to 60% of semantically meaningful input dimensions on the adversarial spheres problem are learned to be ignored, while retaining virtually perfect performance BID20 .

In summary, there is ample evidence that RevNet-type networks are closely related to ResNets, while providing a principled framework to study widely observed issues related to excessive invariance in deep learning in general and adversarial robustness in particular.

Information theory.

The information-theoretic view has gained recent interest in machine learning due to the information bottleneck BID46 BID42 BID2 and usage in generative modelling BID26 .

As a consequence, the estimation of mutual information BID5 BID3 BID1 BID8 has attracted growing attention.

The concept of group-wise independence between latent variables goes back to classical independent subspace analysis BID27 and received attention in learning unbiased representations, e.g. see the Fair Variational Autoencoder BID34 .

Furthermore, extended cross-entropy losses via entropy terms BID37 or minimizing predictability of variables BID40 has been introduced for other applications.

Our proposed loss also shows similarity to the GAN loss BID22 .

However, in our case there is no notion of real or fake samples, but exploring similarities in the optimization are a promising avenue for future work.

Failures of deep networks under distribution shift and their difficulty in out-of-distribution generalization are prime examples of the limitations in current machine learning models.

The field of adversarial example research aims to close this gap from a robustness point of view.

While a lot of work has studied -adversarial examples, recent trends extend the efforts towards the unrestricted case.

However, adversarial examples with no restriction are hard to formalize beyond testing error.

We introduce a reverse view on the problem to: (1) show that a major cause for adversarial vulnerability is excessive invariance to semantically meaningful variations, (2) demonstrate that this issue persists across tasks and architectures; and (3) make the control of invariance tractable via fully-invertible networks.

In summary, we demonstrated how a bijective network architecture enables us to identify large adversarial subspaces on multiple datasets like the adversarial spheres, MNIST and ImageNet.

Afterwards, we formalized the distribution shifts causing such undesirable behavior via information theory.

Using this framework, we find one of the major reasons is the insufficiency of the vanilla cross-entropy loss to learn semantic representations that capture all task-dependent variations in the input.

We extend the loss function by components that explicitly encourage a split between semantically meaningful and nuisance features.

Finally, we empirically show that this split can remove unwanted invariances by performing a set of targeted invariance-based distribution shift experiments.

Example 7 (Semantic and nuisance on Adversarial Spheres BID20 ).

Consider classifying inputs x from two classes given by radii R 1 or R 2 .

Further, let (r, φ) denote the spherical coordinates of x. Then, any perturbation ∆x, x * = x + ∆x with r * = r is semantic.

On the other hand, if r * = r the perturbation is a nuisance with respect to the task of discriminating two spheres.

In this example, the max-margin classifier D(x) = sign x − R1+R2 2 is invariant to any nuisance perturbation, while being only sensitive to semantic perturbations.

In summary, the transform to spherical coordinates allows to linearize semantic and nuisance perturbations.

Using this notion, invariance-based adversarial examples can be attributed to perturbations of x * = x + ∆x with following two properties 1.

Perturbed sample x * stays in the pre-image {x DISPLAYFORM0 Thus, the failure of the classifier D can be thought of a mis-alignment between its invariance (expressed through the pre-image) and the semantics of the data and task (expressed by the oracle).Example 8 (Mis-aligned classifier on Adversarial Spheres).

Consider the classifier DISPLAYFORM1 which computes the norm of x from its first d − 1 cartesian-coordinates.

Then, D is invariant to a semantic perturbation with ∆r = R 2 − R 1 if only changes in the last coordinate x d are made.

We empirically evaluate the classifier in equation 7 on the spheres problem (10M/2M samples setting BID20 ) and validate that it can reach perfect classification accuracy.

However, by construction, perturbing the invariant dimension x * d = x d + ∆x d allows us to move all samples from the inner sphere to the outer sphere.

Thus, the accuracy of the classifier drops to chance level when evaluating its performance under such a distributional shift.

To conclude, this underlines how classifiers with optimal performance on finite samples can exhibit non-intuitive failure modes due to excessive invariance with respect to semantic variations.

We use a standard Imagenet pre-trained Resnet-154 as provided by the torchvision package BID36 and choose a logit percept y = G(x) that can be based on any seed image.

Then we optimize various imagesx to be metameric to x by simply minimizing a mean squared error loss of the form: DISPLAYFORM0 in the 1000-dimensional semantic logit space via stochastic gradient descent.

We optimize with Adam in Pytorch default settings and a learning rate of 0.01 for 3000 iterations.

The optimization thus takes the form of an adversarial attack targeting all logit entries and with no norm restriction on the input distance.

Note that our metameric sampling attack in bijective networks is the analytic reverse equivalent of this attack.

It leads to the exact solution at the cost of one inverse pass instead of an approximate solution here at the cost of thousands of gradient steps.

Figure 9: Here we show a batch of randomly sampled metamers from our ImageNet-trained fully invertible RevNet-48.

The quality is generally similar, sometimes colored artifacts appear.

Computing mutual information is often intractable as it requires the joint probability p(x, y), see BID14 for an extensive treatment of information theory.

However, following variational lower bound can be used for approximation, see BID5 .

Lemma 9 (Variational lower bound on mutual information).

Let X, Y be random variables with conditional density p(y|x).

Further, let q θ (y|x) be a variational density depending on parameter θ.

Then, the lower bound DISPLAYFORM0 holds with equality if p(y|x) = q θ (y|x).While above lower bound removes the need for the computation of p(y|x), estimating the expectation E Y |X still requires sampling from it.

Using this bound, we can now state the effect of the nuisance classifiation loss.

Lemma 10 (Effect of nuisance classifier).

Define semantics as z s = F θ (x) 1,...,C and nuisances as z n = F θ (x) C+1,...,d , where (x, y) ∼ D. Then, the nuisance classification loss yields DISPLAYFORM1 (ii) Maximization to tighten bound on I D (y; z n ): Under a perfect model of the conditional density, DISPLAYFORM2 Proof.

To proof above result, we need to draw the connection to the variational lower bound on mutual information from Lemma 9.

Let the nuisance classifier D θnc (z n ) model the variational posterior q θnc (y|z n ).

Then we have the lower bound DISPLAYFORM3 From Lemma 9 follows, that if D θnc (z n ) = p(y|z n ), it holds I(y; z n ) = I θnc (y; z n ).

Hence, the nuisance classifier needs to model the conditional density perfectly.

Estimating this bound via Monte Carlo simulation requires sampling from the conditional density p(y|z n ).

Following BID2 , we have the Markov property y ↔ x ↔ z n as labels y interact with inputs x and representation z n interacts with inputs x. Hence, DISPLAYFORM4 Including above and assuming F θ (x) = z n to be a deterministic function, we have DISPLAYFORM5 Lemma 11 (Effect of MLE-term).

Define semantics as z s = F θ (x) 1,...,C and nuisances as z n = F θ (x) C+1,...,d , where (x, y) ∼ D. Then, the MLE-term in equation 6 together with cross-entropy on the semantics DISPLAYFORM6 minimizes the mutual information I(z s ; z n ).Proof.

Letz s = sof tmax(z s ).

Then minimizing the loss terms L sCE and L M LEn is a maximum likelihood estimation under the factorial prior DISPLAYFORM7 where Cat is a categorical distribution.

As sof tmax is shift-invariant, sof tmax(x + c) = sof tmax(x), above factorial prior forz s and z n yields independence between logits z s and z n up to a constant c. Finally note, the log term and summation in L M LEn and L CE is re-formulation for computational ease but does not change its minimizer as the logarithm is monotone.

From the assumptions follows I D Adv (y; z n ) = 0.

Furthermore, we have the assumption DISPLAYFORM0 excluding synergetic effects in the interaction information BID18 .

By information preservation under homeomorphisms BID32 and the chain rule of mutual information (Cover & BID14 , we have DISPLAYFORM1 ..,C is obtained by the deterministic transform F , by the data processing inequality (Cover & BID14 we have the inequality I D Adv (y; x) ≥ I D Adv (y; z s ).

Thus, the claimed equality must hold.

Remark 12.

Since our goal is to maximize the mutual information I(y; z s ) while minimizing I(y; z n ), we need to ensure that this objective is well defined as mutual information can be unbounded from above for continuous random variables.

However, due to the data processing inequality (Cover & BID14 we have I(y; z n ) = I(y; F θ (x)) ≤ I(y; x).

Hence, we have a fixed upper bound given by our data (x, y).

Compared to BID8 there is thus no need for gradient clipping or a switch to the bounded Jensen-Shannon divergence as in BID26 is not necessary.

All experiments were based on a fully invertible RevNet model with different hyperparameters for each dataset.

For the spheres experiment we used Pytorch BID36 and for MNIST, as well as Imagenet Tensorflow BID0 .

The network is a fully connected fully invertible RevNet.

It has 4 RevNet-type ReLU bottleneck blocks with additive couplings and uses no batchnorm.

We train it via cross-entropy and use the Adam optimizer BID30 ) with a learning rate of 0.0001 and otherwise default Pytorch settings.

The nuisance classifier is a 3 layer ReLU network with 1000 hidden units per layer.

We choose the spheres to be 100-dimensional, with R 1 = 1 and R 2 = 10, train on 500k samples for 10 epochs and then validate on another 100k holdout set.

We achieve 100% train and validation accuracy for logit and nuisance classifier.

We use a convolutional fully invertible RevNet with additional actnorm and invertible 1x1 convolutions between each layer as introduced in BID31 .

The network has 3 stages, after which half of the variables are factored out and an invertible downsampling, or squeezing BID15 BID28 ) is applied.

The network has 16 RevNet blocks with batch norm per stage and 128 filters per layer.

We also dequantize the inputs as is typically done in flow-based generative models.

The network is trained via Adamax (Kingma & Ba, 2014) with a base learning rate of 0.001 for 100 epochs and we multiply the it with a factor of 0.2 every 30 epochs and use a batch size of 64 and l2 weight decay of 1e-4.

For training we compare vanilla cross-entropy training with our proposed independence cross-entropy loss.

To have a more balanced loss signal, we normalize L nCE by the number of input dimensions it receives for the maximization step.

The nuisance classifier is a fullyconnected 3 layer ReLU network with 512 units.

As data-augmentation we use random shifts of 3 pixels.

For classification errors of the different architectures we compare, see TAB3 : Results comparing cross-entropy training (CE) with independence cross-entropy training (iCE) from Definition 5 and two architectures from the literature.

The accuracy of the logit classifiers is on par for the CE and iCE networks, but the train error is higher for CE compared to test error, indicating less overfitting for iCE.

Further, a classifier independently trained on the nuisance variables is able to reach even smaller error than on the logits for CE, but just 27.70% error for iCE, indicating that we have successfully removed most of the information of the label from the nuisance variables and fixed the problem of excessive invariance to semantically meaningful variability with no cost in test error.network.

The first three stages consist of additive and the last of affine coupling layers.

After the final layer we apply an orthogonal 2D DCT type-II to all feature maps and read out the classes in the low-pass components of the transformation.

This effectively gives us an invertible global average pooling and makes our network even more similar to ResNets, that always apply global average pooling on their final feature maps.

We train the network with momentum SGD for 128 epochs, a batch size of 480 (distributed to 6 GPUs), a base learning rate of 0.1, which is reduced by a factor of 0.1 every 32 epochs.

We apply momentum of 0.9 and l2 weight decay of 1e-4.

<|TLDR|>

@highlight

We show deep networks are not only too sensitive to task-irrelevant changes of their input, but also too invariant to a wide range of task-relevant changes, thus making vast regions in input space vulnerable to adversarial attacks.