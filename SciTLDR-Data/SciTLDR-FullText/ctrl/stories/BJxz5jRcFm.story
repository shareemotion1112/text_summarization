The ever-increasing size of modern datasets combined with the difficulty of obtaining label information has made semi-supervised learning of significant practical importance in modern machine learning applications.

In comparison to supervised learning, the key difficulty in semi-supervised learning is how to make full use of the unlabeled data.

In order to utilize manifold information provided by unlabeled data, we propose a novel regularization called the tangent-normal adversarial regularization, which is composed by two parts.

The two parts complement with each other and jointly enforce the smoothness along two different directions that are crucial for semi-supervised learning.

One is applied along the tangent space of the data manifold, aiming to enforce local invariance of the classifier on the manifold, while the other is performed on the normal space orthogonal to the tangent space, intending to impose robustness on the classifier against the noise causing the observed data deviating from the underlying data manifold.

Both of the two regularizers are achieved by the strategy of virtual adversarial training.

Our method has achieved state-of-the-art performance on semi-supervised learning tasks on both artificial dataset and practical datasets.

The recent success of supervised learning (SL) models, like deep convolutional neural networks, highly relies on the huge amount of labeled data.

However, though obtaining data itself might be relatively effortless in various circumstances, to acquire the annotated labels is still costly, limiting the further applications of SL methods in practical problems.

Semi-supervised learning (SSL) models, which requires only a small part of data to be labeled, does not suffer from such restrictions.

The advantage that SSL depends less on well-annotated datasets makes it of crucial practical importance and draws lots of research interests.

The common setting in SSL is that we have access to a relatively small amount of labeled data and much larger amount of unlabeled data.

And we need to train a classifier utilizing those data.

Comparing to SL, the main challenge of SSL is how to make full use of the huge amount of unlabeled data, i.e., how to utilize the marginalized input distribution p(x) to improve the prediction model i.e., the conditional distribution of supervised target p(y|x).

To solve this problem, there are mainly three streams of research.

The first approach, based on probabilistic models, recognizes the SSL problem as a specialized missing data imputation task for classification problem.

The common scheme of this method is to establish a hidden variable model capturing the relationship between the input and label, and then applies Bayesian inference techniques to optimize the model BID10 Zhu et al., 2003; BID21 .

Suffering from the estimation of posterior being either inaccurate or computationally inefficient, this approach performs less well especially in high-dimensional dataset BID10 .The second line tries to construct proper regularization using the unlabeled data, to impose the desired smoothness on the classifier.

One kind of useful regularization is achieved by adversarial training BID8 , or virtual adversarial training (VAT) when applied to unlabeled data BID15 .

Such regularization leads to robustness of classifier to adversarial examples, thus inducing smoothness of classifier in input space where the observed data is presented.

The input space being high dimensional, though, the data itself is concentrated on a underlying manifold of much lower dimensionality BID2 BID17 Chapelle et al., 2009; BID22 .

Thus directly performing VAT in input space might overly regularize and does potential harm to the classifier.

Another kind of regularization called manifold regularization aims to encourage invariance of classifier on manifold BID25 BID0 BID18 BID11 BID22 , rather than in input space as VAT has done.

Such manifold regularization is implemented by tangent propagation BID25 BID11 or manifold Laplacian norm BID0 BID13 , requiring evaluating the Jacobian of classifier (with respect to manifold representation of data) and thus being highly computationally inefficient.

The third way is related to generative adversarial network (GAN) BID7 .

Most GAN based approaches modify the discriminator to include a classifier, by splitting the real class of original discriminator into K subclasses, where K denotes the number of classes of labeled data BID24 BID19 BID5 BID20 .

The features extracted for distinguishing the example being real or fake, which can be viewed as a kind of coarse label, have implicit benefits for supervised classification task.

Besides that, there are also works jointly training a classifier, a discriminator and a generator BID14 .Our work mainly follows the second line.

We firstly sort out three important assumptions that motivate our idea:The manifold assumption The observed data presented in high dimensional space is with high probability concentrated in the vicinity of some underlying manifold of much lower dimensionality BID2 BID17 Chapelle et al., 2009; BID22 .

We denote the underlying manifold as M. We further assume that the classification task concerned relies and only relies on M BID22 .

The noisy observation assumption The observed data x can be decomposed into two parts as x = x 0 + n, where x 0 is exactly supported on the underlying manifold M and n is some noise independent of x 0 BID1 BID21 .

With the assumption that the classifier only depends on the underlying manifold M, the noise part might have undesired influences on the learning of the classifier.

The semi-supervised learning assumption If two points x 1 , x 2 ∈ M are close in manifold distance, then the conditional probability p(y|x 1 ) and p(y|x 2 ) are similar BID0 BID22 BID18 .

In other words, the true classifier, or the true condition distribution p(y|X) varies smoothly along the underlying manifold M. DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 c 0 w R T 9 e + O m m f W V l n i K i e 7 2 8 f e R P y f 1 y 8 p 3 R 7 U U h c l o R Y P g 9 J S A e U w S R G G 0 q A g V T n C h Z F u V x A j b r g g l 7 X v Q g g f n / y U H G 9 0 w 6 A b f t 1 s 7 3 2 a x b H A V t l 7 t s Z C t s X 2 2 G d 2 x H p M s B / s i t 2 w W + / S u / b u v P u H 0 j l v 1 v O O / Q P v 9 x 8 6 h a h y < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " f j E y Y 9 J 5 C G f U l R v p 4 t P 7 5 Z Z a A t s = " > A DISPLAYFORM4 c 0 w R T 9 e + O m m f W V l n i K i e 7 2 8 f e R P y f 1 y 8 p 3 R 7 U U h c l o R Y P g 9 J S A e U w S R G G 0 q A g V T n C h Z F u V x A j b r g g l 7 X v Q g g f n / y U H G 9 0 w 6 A b f t 1 s 7 3 2 a x b H A V t l 7 t s Z C t s X 2 2 G d 2 x H p M s B / s i t 2 w W + / S u / b u v P u H 0 j l v 1 v O O / Q P v 9 x 8 6 h a h y < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " f j E y Y 9 J 5 C G f U l R v p 4 t P 7 5 Z Z a A t s = " > A DISPLAYFORM5 c 0 w R T 9 e + O m m f W V l n i K i e 7 2 8 f e R P y f 1 y 8 p 3 R 7 U U h c l o R Y P g 9 J S A e U w S R G G 0 q A g V T n C h Z F u V x A j b r g g l 7 X v Q g g f n / y U H G 9 0 w 6 A b f t 1 s 7 3 2 a x b H A V t l 7 t s Z C t s X 2 2 G d 2 x H p M s B / s i t 2 w W + / S u / b u v P u H 0 j l v 1 v O O / Q P v 9 x 8 6 h a h y < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " f j E y Y 9 J 5 C G f U l R v p 4 t P 7 5 Z Z a A t s = " > A DISPLAYFORM6 c 0 w R T 9 e + O m m f W V l n i K i e 7 2 8 f e R P y f 1 y 8 p 3 R 7 U U h c l o R Y P g 9 J S A e U w S R G G 0 q A g V T n C h Z F u V x A j b r g g l 7 X v Q g g f n / y U H G 9 0 w 6 A b f t 1 s 7 3 2 a x b H A V t l 7 t s Z C t s X 2 2 G d 2 x H p M s B / s i t 2 w W + / S u / b u v P u H 0 j l v 1 v O O / Q P v 9 x 8 6 h a h y < / l a t e x i t > x 0 < l a t e x i t s h a 1 _ b a s e 6 4 = " U v o A x a M + o b w r T 4 7 t H V I T M R l 7 n k U = " > A A A C P n i c b V B N a 9 t A E F 2 l T e o q X 2 5 y 7 DISPLAYFORM7 4 2 c d e G 4 N 8 / + S E Y n n Z 9 r + t / f t 3 q n 6 3 j a J C X 5 B X p E J + 8 I X 3 y g Z y T A e H k K / l O f p J f z r X z w / n t / L m 1 b j j r m U N y p 5 y / / w C 0 + a o E < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " U v o A x a M + o b w r T 4 7 t H V I T M R l 7 n k U = " > A A A C P n i c b V B N a 9 t A E F 2 l T e o q X 2 5 y 7 DISPLAYFORM8 4 2 c d e G 4 N 8 / + S E Y n n Z 9 r + t / f t 3 q n 6 3 j a J C X 5 B X p E J + 8 I X 3 y g Z y T A e H k K / l O f p J f z r X z w / n t / L m 1 b j j r m U N y p 5 y / / w C 0 + a o E < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " U v o A x a M + o b w r T 4 7 t H V I T M R l 7 n k U = " > A A A C P n i c b V B N a 9 t A E F 2 l T e o q X 2 5 y 7 DISPLAYFORM9

n n Z 9 r + t / f t 3 q n 6 3 j a J C X 5 B X p E J + 8 I X 3 y g Z y T A e H k K / l O f p J f z r X z w / n t / L m 1 b j j r m U N y p 5 y / / w C 0 + a o E < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " U v o A x a M + o b w r T 4 7 t H V I T M R l 7 n k U = " > A A A C P n i c b V B N a 9 t A E F 2 l T e o q X 2 5 y 7 G W p M T g k G C k U W i g G k 1 5 6 a U m h / g D L E a v 1 K F 6 8 W o n d U b A Q + m W 9 5 D f 0 l m M v P b S U X n v s y v G h + R h Y e P P e G 2 b n R Z k U B j 3 v x t l 4 8 n R z 6 1 n j u b u 9 s 7 u 3 3 3

n n Z 9 r + t / f t 3 q n 6 3 j a J C X 5 B X p E J + 8 I X 3 y g Z y T A e H k K / l O f p J f z r X z w / n t / L m 1 b j j r m U N y p 5 y / / w C 0 + a o E < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " U v o A x a M + o b w r T 4 7 t H V I T M R l 7 n k U = " > A A A C P n i c b V B N a 9 t A E F 2 l T e o q X 2 5 y 7 G W p M T g k G C k U W i g G k 1 5 6 a U m h / g D L E a v 1 K F 6 8 W o n d U b A Q + m W 9 5 D f 0 l m M v P b S U X n v s y v G h + R h Y e P P e G 2 b n R Z k U B j 3 v x t l 4 8 n R z 6 1 n j u b u 9 s 7 u 3 3 3

n n Z 9 r + t / f t 3 q n 6 3 j a J C X 5 B X p E J + 8 I X 3 y g Z y T A e H k K / l O f p J f z r X z w / n t / L m 1 b j j r m U N y p 5 y / / w C 0 + a o E < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " U v o A x a M + o b w r T 4 7 t H V I T M R l 7 n k U = " > A A A C P n i c b V B N a 9 t A E F 2 l T e o q X 2 5 y 7 G W p M T g k G C k U W i g G k 1 5 6 a U m h / g D L E a v 1 K F 6 8 W o n d U b A Q + m W 9 5 D f 0 l m M v P b S U X n v s y v G h + R h Y e P P e G 2 b n R Z k U B j 3 v x t l 4 8 n R z 6 1 n j u b u 9 s 7 u 3 3 3 tangent-normal adversarial regularization.

x = x0 + n is the observed data, where x0 is exactly supported on the underlying manifold M and n is the noise independent of x0.

r is the adversarial perturbation along the tangent space to induce invariance of the classifier on manifold; r ⊥ is the adversarial perturbation along the normal space to impose robustness on the classifier against noise n.

Inspired by the three assumptions, we introduce a novel regularization called the tangent-normal adversarial regularization (TNAR), which is composed by two parts.

The tangent adversarial regularization (TAR) induces the smoothness of the classifier along the tangent space of the underlying manifold, to enforce the invariance of the classifier along manifold.

And the normal adversarial regularization (NAR) penalizes the deviation of the classifier along directions orthogonal to the tangent space, to impose robustness on the classifier against the noise carried in observed data.

The two regularization terms enforce different aspects of the classifier's smoothness and jointly improve the generalization performance, as demonstrated in Section 4.To realize our idea, we have two challenges to conquer: how to estimate the underlying manifold and how to efficiently perform TNAR.For the first issue, we take advantage of the generative models equipped with an extra encoder, to characterize coordinate chart of manifold BID11 BID13 BID20 .

More specifically, in this work we choose variational autoendoer (VAE) BID9 and localized GAN (Qi et al., 2018) to estimate the underlying manifold from data.

For the second problem, we develop an adversarial regularization approach based on virtual adversarial training (VAT) BID16 .

Different from VAT, we perform virtual adversarial training in tangent space and normal space separately as illustrated in FIG8 , which leads to a number of new technical difficulties and we will elaborate the corresponding solutions later.

Compared with the traditional manifold regularization methods based on tangent propagation BID25 BID11 or manifold Laplacian norm BID0 BID13 , our realization does not require explicitly evaluating the Jacobian of classifier.

All we need is to calculate the derivative of matrix vector product, which only costs a few times of back or forward propagation of network.

We denote the labeled and unlabeled dataset as D l = {(x l , y l )} and D ul = {x ul } respectively, thus D := D l ∪ D ul is the full dataset.

The output of classification model is written as p(y|x, θ), where θ is the model parameters to be trained.

We use (·, ·) to represent supervised loss function.

And the regularization term is denoted as R with specific subscript for distinction.

The observed space of x is written as R D .

And the underlying manifold of the observed data x is written as DISPLAYFORM0 We use z for the manifold representation of data x.

We denote the decoder, or the generator, as x = g(z) and the encoder as z = h(x), which form the coordinate chart of manifold together.

If not stated otherwise, we always assume x and z correspond to the coordinate of the same data point in observed space R D and on manifold M, i.e., g(z) = x and h(x) = z. The tangent space of M at point DISPLAYFORM1 where J z g is the Jacobian of g at point z. The tangent space T x M is also the span of the columns of J z g. For convenience, we define J := J z g.

The perturbation in the observed space R D is denoted as r ∈ R D , while the perturbation on the manifold representation is denoted as η ∈ R d .

Hence the perturbation on manifold is g(z DISPLAYFORM2 When the perturbation η is small enough for the holding of the first order Taylor's expansion, the perturbation on manifold is approximately equal to the perturbation on its tangent space, g(z + η) − g(z) ≈ J · η ∈ T x M.

Therefore we say a perturbation r ∈ R D is actually on manifold, if there is a perturbation η ∈ R d , such that r = J · η.

VAT BID16 ) is an effective regularization method for SSL.

The virtual adversarial loss introduced in VAT is defined by the robustness of the classifier against local perturbation in the input space R D .

Hence VAT imposes a kind of smoothness condition on the classifier.

Mathematically, the virtual adversarial loss in VAT for SSL is DISPLAYFORM0 , where the VAT regularization R vat is defined as R vat (x; θ) := max r 2 ≤ dist(p(y|x, θ), p(y|x + r, θ)), where dist(·, ·) is some distribution distance measure and controls the magnitude of the adversarial example.

For simplicity, define F (x, r, θ) := dist(p(y|x, θ), p(y + r, θ)).Then R vat = max r 2 ≤ F (x, r, θ).

The so called virtual adversarial example is r * := arg max r ≤ F (x, r, θ).

Once we have r * , the VAT loss can be optimized with the objective as DISPLAYFORM1 To obtain the virtual adversarial example r * , BID16 suggested to apply second order Taylor's expansion to F (x, r, θ) around r = 0 as DISPLAYFORM2 where H := ∇ 2 r F (x, r, θ)| r=0 denotes the Hessian of F with respect to r. The vanishing of the first two terms in Taylor's expansion occurs because that dist(·, ·) is a distance measure with minimum zero and r = 0 is the corresponding optimal value, indicating that at r = 0, both the value and the gradient of F (x, r, θ) are zero.

Therefore for small enough , r * ≈ arg max r 2 ≤ 1 2 r T Hr, which is an eigenvalue problem and the direction of r * can be solved by power iteration.

We take advantage of generative model with both encoder h and decoder g to estimate the underlying data manifold M and its tangent space T x M. As assumed by previous works BID11 BID13 , perfect generative models with both decoder and encoder can describe the data manifold, where the decoder g(z) and the encoder h(x) together serve as the coordinate chart of manifold M. Note that the encoder is indispensable for it helps to identify the manifold coordinate z = h(x) for point x ∈ M. With the trained generative model, the tangent space is given by DISPLAYFORM0 , or the span of the columns of J = J z g.

In this work, we adopt VAE BID9 and localized GAN (Qi et al., 2018) to learn the targeted underlying data manifold M as summarized below.

VAE VAE BID9 ) is a well known generative model consisting of both encoder and decoder.

The training of VAE is by optimizing the variational lower bound of log likelihood, DISPLAYFORM1 Here p(z) is the prior of hidden variable z, and q(z|x, θ), p(x|z, θ) models the encoder and decoder in VAE, respectively.

The derivation of the lower bound with respect to θ is well defined thanks to the reparameterization trick, thus it could be optimized by gradient based method.

The lower bound could also be interpreted as a reconstruction term plus a regularization term BID9 .

With a trained VAE, the encoder and decoder are given as h(x) = arg max z q(z|x) and g(z) = arg max x q(x|z) accordingly.

Localized GAN Localized GAN BID20 suggests to use a localized generator G(x, z) to replace the global generator g(z) in vanilla GAN Goodfellow et al. (2014a) .

The key difference between localized GAN and previous generative model for manifold is that, localized GAN learns a distinguishing local coordinate chart for each point x ∈ M, which is given by G(x, z), rather than one global coordinate chart.

To model the local coordinate chart in data manifold, localized GAN requires the localized generator to satisfy two more regularity conditions: 1) locality: G(x, 0) = x, so that G(x, z) is localized around x; 2) orthogonmality: DISPLAYFORM2 is non-degenerated.

The two conditions are achieved by the following penalty during training of localized GAN: DISPLAYFORM3 Since G(x, z) defines a local coordinate chart for each x separately, in which the latent encode of x is z = 0, there is no need for the extra encoder to provide the manifold representation of x.

In this section we elaborate our proposed tangent-normal adversarial regularization (TNAR) strategy.

The TNAR loss to be minimized for SSL is DISPLAYFORM0 The first term in Eq. (4) is a common used supervised loss.

R tangent and R normal is the so called tangent adversarial regularization (TAR) and normal adversarial regularization (NAR) accordingly, jointly forming the proposed TNAR.

We assume that we already have a well trained generative model for the underlying data manifold M, with encoder h and decoder g, which can be obtained as described in Section 2.3.

Vanilla VAT penalizes the variety of the classifier against local perturbation in the input space R D BID16 , which might overly regularize the classifier, since the semi-supervised learning assumption only indicates that the true conditional distribution varies smoothly along the underlying manifold M, but not the whole input space R D BID0 BID22 BID18 .

To avoid this shortcoming of vanilla VAT, we propose the tangent adversarial regularization (TAR), which restricts virtual adversarial training to the tangent space of the underlying manifold T x M, to enforce manifold invariance property of the classifier.

DISPLAYFORM0 where F (x, r, θ) is defined as in Eq. (1).

To optimize Eq. (5), we first apply Taylor's expansion to F (x, r, θ) so that R tangent (x; θ) ≈ max r 2 ≤ ,r∈TxM=Jzg(R d ) 1 2 r T Hr, where the notations and the derivation are as in Eq. (2).

We further reformulate R tangent as DISPLAYFORM1 demand r being orthogonal to only one specific tangent direction, i.e., the tangent space adversarial perturbation r .

Thus the constraint J T · r = 0 is relaxed to (r ) T · r = 0.

And we further replace the constraint by a regularization term, DISPLAYFORM2 where λ is a hyperparameter introduced to control the orthogonality of r.

Since Eq. FORMULA15 is again an eigenvalue problem, and we can apply power iteration to solve it.

Note that a small identity matrix λ r I is needed to be added to keep 1 2 H − λr r T + λ r I semipositive definite, which does not change the optimal solution of the eigenvalue problem.

The power iteration is as DISPLAYFORM3 And the evaluation of Hr is by Hr = ∇ r ∇ r F (x, 0, θ) · r , which could be computed efficiently.

After finding the optimal solution of Eq. FORMULA15 as r ⊥ , the NAR becomes R normal (x, θ) = F (x, r ⊥ , θ).Finally, as in BID16 , we add entropy regularization to our loss function.

It ensures neural networks to output a more determinate prediction and has implicit benefits for performing virtual adversarial training, R entropy (x, θ) := − y p(y|x, θ) log p(y|x, θ).

Our final loss for SSL is DISPLAYFORM4 The TAR inherits the computational efficiency from VAT and the manifold invariance property from traditional manifold regularization.

The NAR causes the classifier for SSL being robust against the off manifold noise contained in the observed data.

These advantages make our proposed TNAR, the combination of TAR and NAR, a reasonable regularization method for SSL, the superiority of which will be shown in the experiment part in Section 4.

We also conduct experiments on FashionMNIST dataset 1 .

There are three sets of experiments with the number of labeled data being 100, 200 and 1, 000, respectively.

The details about the networks are in Appendix.

The corresponding results are shown in TAB1 , from which we observe at least two phenomena.

The first is that our proposed TANR methods (TNAR-VAE, TNAR-LGAN) achieve lower classification errors than VAT in all circumstances with different number of labeled data.

The second is that the performance of our method depends on the estimation of the underlying manifold of the observed data.

In this case, TNAR-VAE brings larger improvement than TNAR-LGAN, since VAE produces better diverse examples according to our observation.

As the development of generative model capturing more accurate underlying manifold, it is expected that our proposed regularization strategy benefits more for SSL.

We conduct ablation study on FashionMNIST datasets to demonstrate that both of the two regularization terms in TNAR are crucial for SSL.

The results are reported in TAB1 .

Removing either tangent adversarial regularization (NAR) or normal adversarial regularization (TAR) will harm the final performance, since they fail to enforce the manifold invariance or the robustness against the off-manifold noise.

Furthermore, the adversarial perturbations and adversarial examples are shown in FIG10 .

We can easily observe that the tangent adversarial perturbation focuses on the edges of foreground objects, while the normal space perturbation mostly appears as certain noise over the whole image.

This is consistent with our understanding on the role of perturbation along the two directions that capture the different aspects of smoothness.

There are two classes of experiments for demonstrating the effectiveness of TNAR in SSL, SVHN with 1, 000 labeled data, and CIFAR-10 with 4, 000 labeled data.

The experiment setups are identical with BID16 .

We test two kinds of convolutional neural networks as classifier (denoted as "small" and "large") as in BID16 .

Since it is difficult to obtain satisfying VAE on CIFAR-10, we only conduct the proposed TNAR with the underlying manifold identified by Localized GAN (TNAR-LGAN) for CIFAR-10.

Note that in BID16 , the authors applied ZCA as pre-processing procedure, while other compared methods do not use this trick.

For fair comparison, we only report the performance of VAT without ZCA.

More detailed experimental settings are included in Appendix.

BID6 7.41 17.99 Improved GAN BID24 8.11 18.63 Tripple GAN BID14 5.77 16.99 FM GAN BID11 4.39 16.20 LGAN BID20 4 In TAB2 we report the experiments results on CIFAR-10 and SVHN, showing that our proposed TNAR outperforms other state-of-the-art SSL methods on both SVHN and CIFAR-10, demonstrating the superiority of our proposed TNAR.

We present the tangent-normal adversarial regularization, a novel regularization strategy for semisupervised learning, composing of regularization on the tangent and normal space separately.

The tangent adversarial regularization enforces manifold invariance of the classifier, while the normal adversarial regularization imposes robustness of the classifier against the noise contained in the observed data.

Experiments on artificial dataset and multiple practical datasets demonstrate that our approach outperforms other state-of-the-art methods for semi-supervised learning.

The performance of our method relies on the quality of the estimation of the underlying manifold, hence the breakthroughs on modeling data manifold could also benefit our strategy for semi-supervised learning, which we leave as future work.

represent two different classes.

The observed data is sampled as x = x 0 + n, where x 0 is uniformly sampled from M and n ∼ N (0, 2 −2 ).

We sample 6 labeled training data, 3 for each class, and 3, 000 unlabeled training data, as shown in FIG9 .

In FashionMNIST 2 experiments, we preserver 1, 00 data for validation from the original training dataset.

That is, we use 100/200/1, 000 labeled data for training and the other 100 labeled data for validation.

For pre-processing, we scale images into 0 ∼ 1.

The classification neural network is as following. (a, b) means the convolution filter is with a × a shape and b channels.

The max pooling layer is with stride 2.

And we apply local response normalization (LRN) BID23 .

The number of hidden nodes in the first fully connected layer is 512.Conv(3, 32) → ReLU → Conv(3, 32) → ReLU → MaxPooling → LRN → Conv(3, 64) → ReLU → Conv(3, 64) → ReLU → MaxPooling → LRN → FC1 → ReLU → FC2 For the labeled data, the batch size is 32, and for the unlabeled data, the batch size is 128.

All networks are trained for 12, 000 updates.

The optimizer is ADAM with initial learning rate 0.001, and linearly decay over the last 4, 000 updates.

The hyperparameters tuned is the magnitude of the tangent adversarial perturbation ( 1 ), the magnitude of the normal adversarial perturbation ( 2 ) and the hyperparameter λ in Eq. (11).

Other hyperparameters are all set to 1.

We tune λ from {1, 0.1, 0.01, 0.001}, and 1 , 2 randomly from [0.05, 20].

BID24 ; BID12 .

All the convolutional layers and fully connected layers are followed by batch normalization except the fully connected layer on CIFAR-10.

The slopes of all lReLU functions in the networks are 0.1.

The encoder of the VAE for identify the underlying manifold is a LeNet-like one, with two convolutional layers and one fully connected layer.

And the decoder is symmetric with the encoder, except using deconvolutional layers to replace convolutional layer.

The latent dimensionality is 128.

The localized GAN for identify the underlying manifold is similar as stated in BID20 .

And the implementation is modified from https://github.com/z331565360/Localized-GAN.

We change the latent dimensionality into 128.

We tried both joint training the LGAN with the classifier, and training them separately, observing no difference.

In SVHN 3 and CIFAR-10 4 experiments, we preserve 1, 000 data for validation from the original training set.

That is, we use 1, 000/4, 000 labeled data for training and the other 1, 000 labeled data for validation.

The only pre-processing on data is to scale the pixels value into 0 ∼ 1.

We do not use data augmentation.

The structure of classification neural network is shown in TAB4 , which is identical as in BID16 .For the labeled data, the batch size is 32, and for the unlabeled data, the batch size is 128.

For SVHN, all networks are trained for 48, 000 updates.

And for CIFAR-10, all networks are trained for 200, 000 updates.

The optimizer is ADAM with initial learning rate 0.001, and linearly decay over the last 16, 000 updates.

The hyperparameters tuned is the magnitude of the tangent adversarial perturbation ( 1 ), the magnitude of the normal adversarial perturbation ( 2 ) and the hyperparameter λ in Eq. (11).

Other hyperparameters are all set to 1.

We tune λ from {1, 0.1, 0.01, 0.001}, and 1 , 2 randomly from [0.

05, 20] .The VAE for identify the underlying manifold for SVHN is implemented as in https:// github.com/axium/VAE-SVHN.

The only modification is we change the coefficient of the regularization term from 0.01 to 1.

The localized GAN for identify the underlying manifold for SVHN and CIFAR-10 is similar as stated in BID20 .

And the implementation is modified from https://github.com/z331565360/Localized-GAN.

We change the latent dimensionality into 512 for both SVHN and CIFAR-10.

More adversarial perturbations and adversarial examples in tangent space and normal space are shown in FIG12 and FIG13 .

Note that the perturbations is actually too small to distinguish easily, thus we show the scaled perturbations.

From left to right: original example, tangent adversarial perturbation, normal adversarial perturbation, tangent adversarial example, normal adversarial example.

<|TLDR|>

@highlight

We propose a novel manifold regularization strategy based on adversarial training, which can significantly improve the performance of semi-supervised learning.