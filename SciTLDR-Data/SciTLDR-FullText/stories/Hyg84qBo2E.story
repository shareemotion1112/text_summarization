Unsupervised domain adaptation aims to generalize the hypothesis trained in a source domain to an unlabeled target domain.

One popular approach to this problem is to learn a domain-invariant representation for both domains.

In this work, we study, theoretically and empirically, the explicit effect of the embedding on generalization to the target domain.

In particular, the complexity of the class of embeddings affects an upper bound on the target domain's risk.

This is reflected in our experiments, too.

Domain adaptation is critical in many applications where collecting large-scale supervised data is prohibitively expensive or intractable, or conditions at prediction time can change.

For instance, self-driving cars must be robust to various conditions such as different weather, change of landscape and traffic.

In such cases, the model learned from limited source data should ideally generalize to different target domains.

Specifically, unsupervised domain adaptation aims to transfer knowledge learned from a labeled source domain to similar but completely unlabeled target domains.

One popular approach to unsupervised domain adaptation is to learn domain-invariant representations BID7 BID5 , by minimizing a divergence between the representations of source and target domains.

The prediction function is learned on the latent space, with the aim of making it domain-independent.

A series of theoretical works justifies this idea BID9 BID1 BID3 .Despite the empirical success of domain-invariant representations, exactly matching the representations of source and target distribution can sometimes fail to achieve domain adaptation.

For example, BID13 show that exact matching may increase target error if label distributions are different between source and target domain, and propose a new divergence metric to overcome this limitation.

BID14 establish lower and upper bounds on the risk when label distributions between source and target domains differ.

BID6 point out the information lost in non-invertible embeddings, and propose different generalization bounds based on the overlap of the supports of source and target distribution.

In contrast to previous analyses that focus on changes in the label distributions or on joint support, we here study the effect of the complexity of the joint representation.

In particular, we show a general bound on the target risk that reflects a tradeoff between the embedding complexity and the divergence of source and target in the latent representation space.

In particular, a too powerful class of embedding functions can result in overfitting the source data and the distribution matching, leading to arbitrarily high target risk.

Hence, a restriction (taking into account assumptions about correspondences and invariances) is needed.

Our experiments reflect these trends empirically, too.

For simplicity, we consider binary classification with input space X ✓ R n and output space Y = {0, 1}. Define H to be the hypothesis class from X to Y. The learning algorithm obtains two datasets: labeled source data X S with distribution p S , and unlabeled target data X T with distribution p T .

We will use p S and p T to denote the joint distribution on data and labels X, Y and the marginals, i.e., p S (X) and p S (Y ).

Unsupervised domain adaptation seeks a hypothesis h 2 H that minimizes the risk in the target domain measured by a loss function`(here, zero-one loss): DISPLAYFORM0 We will not assume common support in source and target domain, in line with standard benchmarks for domain adaptation such as adapting from MNIST to M-MNIST.

A common approach to domain adaptation is to learn a joint embedding of source and target data BID5 BID12 .

The idea is that aligning source and target distributions in this latent space Z results in a domaininvariant representation, and hence a subsequent classifier f from the embedding to Y will generalize from source to target.

Formally, this results in the following objective function on the hypothesis h = fg := f g, where G is the class of embedding functions to Z, and we minimize a divergence d between the distributions p S (Z g ) = p S (g(X)), p T (Z g ) of source and target after mapping to Z: DISPLAYFORM0 The divergence d could be, e.g., the Jensen-Shannon BID5 or Wasserstein distance BID11 . introduced the H H-divergence to bound the worst-case loss from extrapolating between

] be the expected disagreement between two hypotheses, then the H H-divergence is defined as follows.

Definition 1. (H H-divergence) Given two domain distributions p S and p T over X , and a hypothesis class H, the DISPLAYFORM0 This divergence allows to bound the risk on the target domain: Theorem 1. (Ben-David et al., 2010) For all hypotheses h 2 H, the target risk is bounded as DISPLAYFORM1 where H is the best joint risk DISPLAYFORM2 Similar results have been obtained for continuous labels BID3 BID9 ).Theorem 1 is an influential theoretical result in unsupervised domain adaptation, and motivated work on domain invariant representations.

For example, recent work BID5 ; BID6 ) applied Theorem 1 to the hypothesis space F that maps the representation space Z induced by an encoder g to the output space: DISPLAYFORM3 where F (g) is the best hypothesis risk with fixed g, i.e., DISPLAYFORM4 The F F divergence implicitly depends on the fixed g and can be small if g provides a suitable representation.

However, if g induces a wrong alignment, then the best hypothesis risk F (g) is large with any function class F. The following example will illustrate such a situation, motivating to explicitly take the class of embeddings into account when bounding the target risk.

We begin with an illustrative toy example.

FIG1 shows a binary classification problem in 2D with disjoint support and a slight shift in the label distributions from source to target: p S (y = 1) = p T (y = 1) + 2✏.

Assume the representation space is one dimensional, so the embedding g is a function from 2D to 1D.

If we allow arbitrary, nonlinear embeddings, then, for instance, the embedding shown in FIG1 (b), together with an optimal predictor, achieves zero source loss and a zero divergence, and is hence optimal according to the objective (2).

However, the target risk of this combination of embedding and predictor is maximal: R T (fg) = 1.If we restrict the class G of embeddings to linear maps g(x) = Wx where W 2 IR 1⇥2 , then the embeddings that are optimal with respect to the objective (2) are of the form W = ⇥ a, 0 ⇤ .

Together with an optimal source classifier f , they achieve a non-zero value of 2✏ for objective (2) due to the shift in class distributions.

However, these embeddings retain label correspondences, and can lead to a zero target risk.

This example illustrates that a too rich class of embeddings can "overfit" the alignment, and hence lead to arbitrarily bad solutions.

Hence, the complexity of the encoder class plays an important role in learning domain invariant representation too.

Motivated by the above example, we next expose how the bound on the target risk depends on the complexity of the embedding class.

To do so, we apply Theorem 1 to the hypothesis h = fg: DISPLAYFORM0 Comparing the bound (4) to the previous bound (3), we notice two differences: the best in-class joint risk now minimizes over both F and G, i.e., DISPLAYFORM1 which is smaller than Fg and reflects the fact that we are learning both f and g. In return, the divergence term d FG FG (p S , p T ) becomes larger than the one in bound (3).To better understand these tradeoffs, we derive a more interpetable form of the bound on the target risk.

Before presenting the bound, we define an extended version of DISPLAYFORM2 For two domain distributions p S and p T over X , an encoder class G, and predictor class F, the F G G -divergence between p S and p T is DISPLAYFORM3 Note that the F G G -divergence is strictly smaller than the FG FG-divergence, since the two hypotheses in the supremum, fg and fg 0 , share the same predictor f .

We are ready to state the following result.

Theorem 2.

For all f 2 F and g 2 G, DISPLAYFORM4 where FG (g) is the best in-class joint risk defined as DISPLAYFORM5 A detailed proof of the theorem may be found in the Appendix.

The first term of the bound is the source risk.

The second term i is the F F-divergence between the distributions p S (Z g ) and p T (Z g ) in the representation space; this also appears in the previous bound (3).

The first term in ii measure the F G G -divergence between source and target distribution, which may decrease as the complexity of the encoder decreases.

However, a less complex encoder class G can lead to increasing the best hypothesis risk FG (g).

Therefore, ii makes a trade-off explicit between the divergence and the model complexity.

Note that, as opposed to different FG , FG (g) also measures the correctness of the encoder in the source domain.

If the encoder fails to provide informative representations in the source domain, then first term in FG (g) can be large.

The last two terms in Theorem 1 express a similar complexity trade-off as ii , but this time with repect to the hypothesis class H, which here combines the encoder and predictor.

Hence, directly applying Theorem 1 to the composition (Equation (4)) treats both jointly and does not make the role of the embedding as explicit as Theorem 2.

For example, Theorem 2 shows that we can also make the bound tighter by minimizing the divergence between the corresponding distributions in the embedding space, as long as the encoder provides useful representations in the source domain.

If i is sufficiently small, the FG FG-divergence reduces to the F G G -divergence, which is strictly smaller than the FG FG-divergence.

Comparing to the previous bound in Equation FORMULA6 , which assumes a fixed g, we do not assume a known encoder and instead quantify the effect of the encoder family.

Moreover, the term F (g) in bound (3) involves the source and target risk, whereas in FG (g) the encoder g only affects the source risk, which can be estimated empirically.

Importantly, without restricting the complexity of the encoder or embedding, the F G G -divergence can be large, indicating that the target risk may be large too.

This suggests that restricting the model complexity of the embedding is crucial for domain invariant representation learning.

To reduce the worst case divergence i , we need to restrict the encoder family to those that can approximately minimize i , in coordination with the predictor class F. Practically, we can optimize the original objective of domain invariant representations in Equation 2 to align the latent distributions.

Term ii implies that we should choose the minimal complexity encoder class G that is is still expressive enough to encode the data from both domains.

Practically, this can be done by regularizing the encoder, e.g., restricting Lipschitz constants or norms of weight matrices.

More explicitly, one may limit the number of layers of a neural network, or apply inductive biases via selecting network architectures.

For instance, comparing to fully connected networks (FCs), convolutional neural networks (CNNs) restrict the output to be spatially consistent with respect to the input.

Next, we empirically test Theorem 2 via one example of domain-invariant representations: Domain-Adversarial Neural Networks (DANN) BID5 , which measure the latent divergence via a domain discriminator (JensenShannon divergence).

We use the standard benchmark MNIST !

MNIST-M (Ganin & Lempitsky FORMULA0 , where the task is to classify unlabeled handwritten digits overlayed with random photographs (MNIST-M) based on labeled images of digits alone (MNIST).

We consider two categories of complexity: number of layers and inductive bias (CNN).

To analyze the effect of the encoder's complexity, we augment the original two-layer CNN encoders with 1 to 5 additional CNN layers, leaving other settings unchanged.

We retrain each model for 5 times and plot the mean and standard deviation of target error with respect to the number of layers in FIG2 (a): Initially, the target error decreases, and then increases when more layers are added.

This corroborates our theory: the CNN encoder without additional layers does not have enough expressive power.

As a consequence, the best hypothesis risk term FG is larger.

However, when more layers are added, the complexity increases and subsequently makes the disagreements larger.

To investigate the importance of inductive bias in domain invariant representations, we replace the CNN encoder with an MLP encoder.

The experimental results are shown in 2(b).

Comparing the target error between (a) and (b) in FIG2 , we can see that the target error with an MLP encoder is significantly higher than with a CNN encoder.

Comparing to CNNs, which encode invariance via pooling and learned filters, MLPs do not have any inductive bias and lead to worse performance.

In fact, the target error with MLP-based domain adaptation is higher than just training on the source, suggesting that, without an appropriate inductive bias, learning domain invariant representations can even worsen the performance.

To gain deeper insight, we use t-SNE BID8 to visualize source and target embedding distributions in FIG2 (c),(d).

With the inductive bias of CNNs, the representations of the target domain aligns well with those of source domain.

In contrast, the MLP encoder results in a strong label mismatch.

The experiments show that the complexity of the encoder can have a direct effect on the target error.

A more complex encoder class leads to larger theoretical bound on the target error, and, indeed, aligned with the theory, we see a significant performance drop in target domain.

Moreover, the experiments suggest that inductive bias is important too.

With a suitable inductive bias such as CNNs, DANN achieves higher performance than the with the MLP encoder, even if the CNN encoder has twice the number of layers.

CNNs are standard for many vision tasks, such as digit recognition.

However, explicit supervision may be required to identify the encoder class when we have less prior knowledge about the task BID10 BID2 .

In this work, we study the role of embedding complexity for domain-invariant representations.

We theoretically and empirically show that restricting the encoder is necessary for successful adaptation, a fact that has mostly been overlooked by previous work.

In fact, without carefully selecting the encoder class, learning domain invariant representations might even harm the performance.

Our observations motivate future research on identifying eappropriate encoder classes for various tasks.

Theorem 2.

For all f 2 F and g 2 G, DISPLAYFORM0 where FG (g) is the best in-class joint risk defined as DISPLAYFORM1 Proof.

We first define the optimal composition hypothesis f ?

g ? with respect to an encoder g to be the hypothesis which minimizes the following error DISPLAYFORM2 By the triangle inequality for classification error ), DISPLAYFORM3 The second term in the R.H.S of Eq. 4 can be bounded as DISPLAYFORM4 The third term in the R.H.S of Eq. 4 can be bounded as DISPLAYFORM5 Combine the above bounds, we have DISPLAYFORM6 where DISPLAYFORM7

@highlight

A general upper bound on the target domain's risk that reflects the role of embedding-complexity.