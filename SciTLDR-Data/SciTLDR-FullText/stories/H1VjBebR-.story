We discuss the feasibility of the following learning problem: given unmatched samples from two domains and nothing else, learn a mapping between the two, which preserves semantics.

Due to the lack of paired samples and without any definition of the semantic information, the problem might seem ill-posed.

Specifically, in typical cases, it seems possible to build infinitely many alternative mappings  from every target mapping.

This apparent ambiguity stands in sharp contrast to the recent empirical success in solving this problem.



We identify the abstract notion of aligning two domains in a semantic way with concrete terms of minimal relative complexity.

A theoretical framework for measuring the complexity of compositions of functions is developed in order to show that it is reasonable to expect the minimal complexity mapping to be unique.

The measured complexity used is directly related to the depth of the neural networks being learned and a semantically aligned mapping could then be captured simply by learning using architectures that are not much bigger than the minimal architecture.



Various predictions are made based on the hypothesis that semantic alignment can be captured by the minimal mapping.

These are verified extensively.

In addition, a new mapping algorithm is proposed and shown to lead to better mapping results.

Multiple recent reports (Xia et al., 2016; BID13 BID12 Yi et al., 2017) convincingly demonstrated that one can learn to map between two domains that are each specified merely by a set of unlabeled examples.

For example, given a set of unlabeled images of horses, and a set of unlabeled images of zebras, CycleGAN (Zhu et al., 2017) creates the analog zebra image for a new image of a horse and vice versa.

These recent methods employ two types of constraints.

First, when mapping from one domain to another, the output has to be indistinguishable from the samples of the new domain.

This is enforced using GANs BID9 and is applied at the distribution level: the mapping of horse images to the zebra domain should create images that are indistinguishable from the training images of zebras and vice versa.

The second type of constraint enforces that for every single sample, transforming it to the other domain and back (by a composition of the mappings in the two directions) results in the original sample.

This is enforced for each training sample from either domain: every training image of a horse (zebra), which is mapped to a zebra (horse) image and then back to the source domain, should be as similar as possible to the original input image.

In another example, taken from DiscoGAN BID13 , a function is learned to map a handbag to a shoe of a similar style.

One may wonder why striped bags are not mapped, for example, to shoes with a checkerboard pattern.

If every striped pattern in either domain is mapped to a checkerboard pattern in the other and vice-versa, then both the distribution constraints and the circularity constraints might hold.

The former could hold since both striped and checkerboard patterned objects would be generated.

Circularity could hold since, for example, a striped object would be mapped to a checkerboard object in the other domain and then back to the original striped object.

One may claim that the distribution of striped bags is similar to those of striped shoes and that the distribution of checkerboard patterns is also the same in both domains.

In this case, the alignment follows from fitting the shapes of the distributions.

This explanation is unlikely, since no effort is being made to create handbags and shoes that have the same distributions of these properties, as well as many other properties.

Our work is dedicated to the alternative hypothesis that the target mapping is implicitly defined by being approximated by the lowest-complexity mapping that has a low discrepancy between the mapped samples and the target distribution, i.e., the property that even a good discriminator cannot distinguish between the generated samples and the target ones.

In Sec. 2 we explore the inherent ambiguity of cross domain mapping.

In Sec. 3, we present the hypothesis and two verifiable predictions, as well as a new unsupervised mapping algorithm.

In Sec. 4, we show that the number of minimal complexity mappings is expected to be small.

Sec. 5 verifies the various predictions.

Some context to our work, including classical ideas such as Occam's Razor, MDL, and Kolmogorov complexity are discussed in Sec. 6.

The learning algorithm is provided with only two unlabeled datasets: one includes i.i.d samples from the first distribution and the second includes i.i.d samples from the other distribution (all notations are listed in Appendix B, Tab.

5).

To semantically tie the two distributions together, a generative view can be taken.

This view is well aligned with the success of GAN-based image generation, e.g., (Radford et al., 2015) , in mapping random input vectors into realistic-looking images.

Let z ∈ X be a random vector that is distributed according to the distribution D Z and which we employ to denote the semantic essence of samples in X A and X B .

We denote D A = y A • D Z and D B = y B • D Z , where the functions y A : X → X A and y B : X → X B (see FIG1 ), and f • D denotes the distribution of f (x), where x ∼ D. Following the circularity-based methods (Xia et al., 2016; BID13 BID12 Yi et al., 2017) , we assume that both y A and y B are invertible.

The assumption of invertibility is further justified by the recent success of supervised pre-image computation methods BID4 .

In unsupervised learning, given training samples, one may be expected to be able to recover the underlying properties of the generated samples, even with very weak supervision BID3 .

However, if the target function between domains A and B is not invertible, because for each member of A there are a few possible members of B (or vice versa), we can add a stochastic component to A that is responsible for choosing which member in B to take, given a member of A. For example, if A is a space of handbag images and B is a space of shoes, such that for every handbag, there are a few analogous shoes, then a stochastic variable can be added such that given a handbag, one shoe is selected among the different analog shoes.

We denote by y AB = y B • y

A , the function that maps the first domain to the second domain.

It is semantic in the sense that it goes through the shared semantic space X .

The goal of the learner is to fit a function h ∈ H, for some hypothesis class H that is closest to y AB , DISPLAYFORM0 where R D [f 1 , f 2 ] = E x∼D (f 1 (x), f 2 (x)), for a loss function : R × R → R and a distribution D. It is not clear that such fitting is possible without further information.

Assume, for example, that there is a natural order on the samples in X B .

A mapping that transforms an input sample x ∈ X A to the sample that is next in order to y AB (x), could be just as feasible.

More generally, one can permute the samples in X A by some function Π that replaces each sample with another sample that has a similar likelihood (see Def.

1 below) and learn h that satisfies h = y AB • Π. We call this difficulty "the alignment problem" and our work is dedicated to understanding the plausibility of learning despite this problem.

In multiple recent contributions (Xia et al., 2016; BID13 BID12 Yi et al., 2017) circularity is employed.

Circularity requires the recovery of both y AB and y BA = y A • y DISPLAYFORM1 simultaneously.

Namely, functions h and h are learned jointly by minimizing the risk: DISPLAYFORM2 where disc C (D 1 , D 2 ) = sup c1,c2∈C |R D1 [c 1 , c 2 ] − R D2 [c 1 , c 2 ]| denotes the discrepancy between distributions D 1 and D 2 that is implemented with a GAN BID8 .The first term in Eq. 3 ensures that the samples generated by mapping domain A to domain B follow the distribution of samples in domain B. The second term is the analog term for the mapping in the other direction.

The last two terms ensure that mapping a sample from one domain to the second and back, results in the original sample.

While the circularity constraints, expressed as the last two terms in Eq. 3, are elegant and do not require additional supervision, for every invertible permutation Π of the samples in domain B (not to be confused with a permutation of the vector elements of the representation of samples in B)

we have DISPLAYFORM3 DISPLAYFORM4 Therefore, every circularity preserving pair h and h gives rise to many possible solutions of the form h = h • Π andh = Π −1 • h .

If Π happens to satisfy D B (x) ≈ D B (Π(x)), then the discrepancy terms in Eq. 3 also remain largely unchanged.

Circularity by itself cannot, therefore, explain the recent success of unsupervised mapping.

2 .

There are infinitely many mappings that preserve the uniform distribution on the two segments.

However, only two stand out as "semantic".

These are exactly the two mappings that can be captured by a neural network with only two hidden neurons and Leaky ReLU activations, i.e., by a function h(x) = σ a (W x + b), for a weight matrix W and the bias vector b.

In order to illustrate our hypothesis, we present a very simple toy example, depicted in FIG2 .

Consider the domain A of uniformly distributed points (x 1 , x 2 ) ∈ R 2 , where 0 ≤ x 1 < 1 and x 2 = 0.5.

Let B be a similar domain, except x 2 = 2.

We are interested in learning the mapping y 2D AB ((x 1 , 0.5) ) = (x 1 , 2) .

We note that there are infinitely many mappings from domain A to B that satisfy the constraints of Eq. 3.However, when we learn the mapping using a neural network with one hidden layer of size 2, and Leaky ReLU activations 1 (Maas et al., 2013) , y

AB is one of only two options.

In this case h(x) = σ a (W x + b), for W ∈ R 2×2 ,b ∈ R 2 and where σ a is applied per coordinate.

The only admissible solutions are of the form DISPLAYFORM0 which are identical, for every b, to y 2D AB or to an alternative y 2D AB ((x 1 , 0.5) ) = (1 − x 1 , 2) .

Exactly the same situation holds for any pair of line segments in R d + .

Therefore, by restricting the hypothesis space of h, we eliminate all alternative solutions, except two.

These two are exactly the two mappings that would commonly be considered "more semantic" than any other mapping, and can be expressed as the simplest possible mapping through a shared one dimensional space.

While this is an extreme example, we believe that the principle is general since limiting the complexity of the admissible solutions eliminates the solutions that are derived from y AB by permuting the samples in the space X A , because such mixing requires added complexity.

In this work, we focus on functions of the form DISPLAYFORM0 here, W 1 , ..., W n+1 are invertible linear transformations from R M to itself.

In addition, σ is a non-linear element-wise activation function.

We will mainly focus on σ that is Leaky ReLU with parameter 0 < a = 1.

In addition, for any function f , we define the complexity of f , denoted by C(f ) as the minimal number n such that there are invertible linear transformations DISPLAYFORM1 Our function complexity framework, therefore, measures the complexity of a function as the depth of a neural network which implements it, or the shallowest network, if there are multiple such networks.

In other words, we use the number of layers of a network as a proxy for the Kolmogorov complexity of functions, using layers in lieu of the primitives of the universal Turing machines, which is natural for studying functions that can be computed by feedforward neural networks.

Note that capacity is typically controlled by means of norm regularization, which is optimized during training.

Here, the architecture is bounded to a certain number of layers.

This measure of complexity is intuitive and provides a clear and stable stratification of functions.

Norm capacity (for norms larger than zero) are not effective in comparing functions of different architectures.

In Sec. 5, we demonstrate that the L1 and L2 norms of the desired mapping are within the range of norms that are obtained when employing bigger or smaller architectures.

Other ways to define the complexity of functions, such as the VC-dimension (Vapnik & Chervonenkis, 1971b) and Rademacher complexity BID2 , are not suitable for measuring the complexity of individual functions, since their natural application is in measuring the capacity of classes of functions.

The simplicity hypothesis leads to concrete predictions, which are verified in Sec. 5.

The first one states that in contrast to the current common wisdom, one can learn a semantically aligned mapping between two spaces without any matching samples and even without circularity.

Prediction 1.

When learning with a small enough network in an unsupervised way a mapping between domains that share common characteristics, the GAN constraint in the target domain is sufficient to obtain a semantically aligned mapping.

The strongest clue that helps identify the alignment of the semantic mapping from the other mappings is the suitable complexity of the network that is learned.

A network with a complexity that is too low cannot replicate the target distribution, when taking inputs in the source domain (high discrepancy).

A network that has a complexity that is too high, would not learn the minimal complexity mapping, since it could be distracted by other alignment solutions.

We believe that the success of the recent methods results from selecting the architecture used in an appropriate way.

For example, DiscoGAN BID13 employs either eight or ten layers, depending on the dataset.

We make the following prediction:Prediction 2.

When learning in an unsupervised way a mapping between domains, the complexity of the network needs to be carefully adjusted.

This prediction is also surprising, since in supervised learning, extra depth is not as detrimental, if at all.

As far as we know, this is the first time that this clear distinction between supervised and unsupervised learning is made 2 .

If the simplicity hypothesis is correct, then in order to capture the target alignment, one would need to learn with the minimal complexity architecture that supports a small discrepancy.

However, deeper architectures can lead to even smaller discrepancies and to better outcomes.

In order to enjoy both the alignment provided by our hypothesis and the improved output quality, we propose to find a function h of a non-minimal complexity k 2 that minimizes the following objective function DISPLAYFORM0 where k 1 is the minimal complexity for mapping with low discrepancy between domain A and domain B. In other words, we suggest to find a function h that is both a high complexity mapping from domain A to B and is close to a function of low complexity that has low discrepancy.

There are alternative ways to implement an algorithm that minimizes the objective function presented in Eq. 6.

Assuming, based on this equation, that for h that minimizes the objective function, the corresponding g * = arg inf DISPLAYFORM1 has a (relatively) small discrepancy, leads to a two-step algorithm.

The algorithm first finds a function g that has small complexity and small discrepancy and then finds h of a larger complexity that is close to g. This is implemented in Alg.

1. Note that in the first step, k 1 is being estimated, for example, by gradually increasing its value, until g with a discrepancy lower than a threshold 0 is found.

We suggest to use a liberal threshold, since the goal of the network g is to provide alignment and not the lowest possible discrepancy.

Require: Unlabeled training sets S A DISPLAYFORM0 , a desired complexity k 2 , and a trade-off parameter λ 1: Identify a complexity k 1 , which leads to a small discrepancy min DISPLAYFORM1

Recall, from Sec. 2, that disc is the discrepancy distance, which is based on the optimal discriminator.

Also discussed were the functions Π, that switches between members in the domain B that have similar probabilities.

These are defined using the discrepancy distance as follows (simplified version; the definitions and results of this section are stated more broadly in Appendix A): Definition 1 (Density preserving mapping).

Let X = (X , D X ) be a domain.

A 0 -density preserving mapping over X (or an 0 -DPM for short) is a function Π such that DISPLAYFORM0 We denote the set of all 0 -DPMs of complexity k by DPM 0 (X; k) DISPLAYFORM1 Below, we define a similarity relation between functions that reflects whether the two are similar.

In this way, we are able to bound the number of different (non-similar) minimal complexity mappings by the number of different DPMs.

DISPLAYFORM2 Put differently, two functions of the same complexity have this relation, if for every step of their processing, the activations of the matching functions are similar.

The defined relation is reflexive and symmetric, but not transitive.

Therefore, there are many different ways to partition the space of functions into disjoint subsets such that in each subset, any two functions have the closeness property.

We count the number of functions as the minimal number of subsets required in order to cover the entire space.

This quantity is denoted by N(U, ∼ U ) where U is the set and ∼ U is the closeness relation.

The formal presentation is in Def.

9, which slightly generalizes the notion of covering numbers (Anthony & Bartlett, 2009 ).Informally, the following theorem states that the number of minimal low-discrepancy mappings is upper bounded by both the number of DPMs of a certain size over D A and over D B .

This result is useful, since DPMs are expected to be rare in real-world domains.

When imagining mapping a space to itself, in a way that preserves the distribution, one first considers symmetries.

Near-perfect symmetries are rare in natural domains, and when these occur, e.g., BID13 , they form wellunderstood ambiguities.

Another option that can be considered is that of replacing specific samples in domain B with other samples of the same probability.

However, these very local discontinuous mappings are of very high complexity, since this complexity is required for reducing the modeling error for discontinuous functions.

One can also consider replacing larger sub-domains with other sub-domains such that the distribution is preserved.

This could be possible, for example, if the distribution within the sub-domains is almost uniform (unlikely), or if it is estimated inaccurately due to the limitations of the training set.

We, therefore, make the following prediction.

Prediction 3.

The number of DPMs of low complexity is small.

Given two domains A and B, there is a certain complexity C 0 A,B , which is the minimal complexity of the networks needed in order to achieve discrepancy smaller than 0 for mapping the distribution D A to the distribution D B .

The set of minimal complexity mappings, i.e., mappings of complexity C 0 A,B that achieve 0 discrepancy is denoted by DISPLAYFORM3 The following theorem shows that the covering number of this set is similar to the covering number of the DPMs.

Therefore, if prediction 3 above holds, the number of minimal low-discrepancy mappings is small.

Theorem 1 (Informal).

Let σ be a Leaky ReLU with parameter 0 < a = 1 and assume identifiability.

Let 0 , 1 and 2 < 1 be three positive constants and A = (X A , D A ) and B = (X B , D B ) are two domains.

Then, DISPLAYFORM4 The theorem assumes identifiability.

In the context of neural networks, the general question of uniqueness up to invariants, also known as identifiability, is an open question.

Several authors have made progress in this area for different neural network architectures.

The most notable work has been done by BID6 that proves identifiability for σ = tanh.

Furthermore, the representation is unique up to some invariants.

Other works (Williamson & Helmke, 1995; BID5 BID14 Sussmann, 1992) prove such uniqueness for neural networks with only one hidden layer and various activation functions.

Similarly, in Lem.

3 in the Appendix, we show that identifiability holds for Leaky ReLU networks with one hidden layer.

The first group of experiments is dedicated to test the validity of the three predictions made, in order to give further support to the simplicity hypothesis.

Next, we evaluate the success of the proposed algorithm in comparison to the DiscoGAN method of BID13 .We chose to experiment with the DiscoGAN architecture since it focuses on semantic tasks that contain a lesser component of texture or style transfer.

The CycleGAN architecture of BID12 inherits much from the style transfer architecture of Pix2Pix BID12 , and the discrepancy term is based on a patch-based analysis, which introduces local constraints that could mask the added freedom introduced by adding layers.

In addition, the U-net architecture of Ronneberger et al. (2015) used by BID12 deviates from the connectivity pattern of our model.

Experiments in this architecture and with the architecture of DualGAN (Yi et al., 2017) , which focuses on tasks similar to CycleGAN, and shares many of the architectural choices, including U-nets and the use of patches, are left for future work.

Prediction 1 states that since the unsupervised mapping methods are aimed at learning minimal complexity low discrepancy functions, GANs are sufficient.

In the literature BID12 BID13 , learning a mapping h : X A → X B , based only on the GAN constraint on B, is presented as a failing baseline.

In (Yi et al., 2017) , among many non-semantic mappings obtained by the GAN baseline, one can find images of GANs that are successful.

However, this goes unnoticed.

In order to validate the prediction that a purely GAN based solution is viable, we conducted a series of experiments using the DiscoGAN architecture and GAN loss only.

We consider image domains A and B, where X A = X B = R 3×64×64 .In DiscoGAN, the generator is built of: (i) an encoder consisting of convolutional layers with 4 × 4 filters followed by Leaky ReLU activation units and (ii) a decoder consisting of deconvolutional layers with 4 × 4 filters followed by a ReLU activation units.

Sigmoid is used for the output layer.

Between four to five convolutional/deconvolutional layers are used, depending on the domains used in A and B (we match the published code architecture per dataset).

The discriminator is similar to the encoder, but has an additional convolutional layer as the first layer and a sigmoid output unit.

The first set of experiments considers the CelebA face dataset.

Transformations are learned between the subset of images labeled as male and those labeled as female, as well as from blond to black hair and eyeglasses to no eyeglasses.

The results are shown in FIG10 , 4, and 5, (resp.).

It is evident that the output image is highly related to the input images.

In the case of mapping handbags to shoes, as seen in Fig. 6 , the GAN does not provide a meaningful solution.

However, in the case of edges to shoes and vice versa FIG6 , the GAN solution is successful.

In Prediction 2, we predict that the selection of the right number of layers is crucial in unsupervised learning.

Using fewer layers than needed, will not support the modeling of the target alignment between the domains.

In contrast, adding superfluous layers would mean that more and more alternative mappings obscure the target transformation.

In BID13 , 8 or 10 layers are employed (counting both convolution and deconvolution) depending on the experiment.

In our experiment, we vary the number of layers and inspect the influence on the results.

The experiments are also repeated for the Wasserstein GAN loss (using the same architecture) in Appendix E.These experiments were done on the CelebA gender conversion task, where 8 layers are employed in the experiments of BID13 .

Using the public implementation and adding and removing layers, we obtain the results in FIG1 .

Note that since the encoder and the decoder parts of the learned network are symmetrical, the number of layers is always even.

As can be seen, changing the number of layers has a dramatic effect on the results.

The best results are obtained at 6 or 8 layers with 6 having the best alignment and 8 having better discrepancy.

The results degrade quickly, as one deviates from the optimal value.

Using fewer layers, the GAN fails to produce images of the desired class.

Adding layers, the semantic alignment is lost, just as expected.

Note that BID13 have preferred low discrepancy over alignment in their choice.

In other words, the selected architecture of size k = 8 presents acceptable images at the price of lower alignment compared to an architecture of size k − 2.

This is probably a result of ambiguity that is already present at the size k architecture.

On the other hand, the smaller architecture of size k − 2 does not produce images of extremely low discrepancy, and there is no architecture that benefits both, an extremely low discrepancy and high alignment.

This is observed for example in Fig. 8 where females are translated to males.

For 4 layers the discrepancy is too low and the mapping fails to produce images of males.

For 6 layers, the discrepancy is relatively low and the alignment is at its highest.

For 8 layers, the discrepancy is at its lowest value, nevertheless, the alignment is worse.

While our discrete notion of complexity seems to be highly related to the quality of the results, the norm of the weights do not seem to point to a clear architecture, as shown in Tab.

2(a).

Since the table compares the norms of architectures of different sizes, we also approximated the functions using networks of a fixed depth k = 18 and then measured the norm.

These results are presented in Tab.

2(b).

In both cases, the optimal depth, which is 6 or 8, does not appear to have a be an optimum in any of the measurements.

Prediction 3 states that there are only a handful of DPMs, except for the identity function.

In order to verify it, we trained a DiscoGAN from a distribution A to itself with an added loss of the form − x∈A |x − h(x)|.

In our experiment, testing network complexities from 2 to 12, we could not find a DPM, see FIG1 and Tab.

3.

For lower complexities, the identity was learned despite the added loss.

For higher complexities, the network learned the identity while changing the background color.

For even higher complexities, other mapping emerged.

However, these mappings did not satisfy the circularity constraint, and are unlikely to be DPMs.

The goal of Alg.

1 is to find a well-aligned solution with higher complexity than the minimal solution and potentially smaller discrepancy.

It has two stages.

In the first one, k 1 , which is the minimal complexity that leads to a low discrepancy, is identified.

This follows a set of experiments that are similar to the one that is captured, for example, by FIG2 .

To demonstrate robustness, we select a single value of k 1 across all experiments.

Specifically, we use k 1 = 6, which, as discussed above, typically leads to a low (but not very low) discrepancy, while the alignment is still unambiguous.

Once g is trained, we proceed to the next step of optimizing a second network of complexity k 2 .

Note that while the first function (g) uses the complete DiscoGAN architecture, the second network (h) only employs a one-directional mapping, since alignment is obtained by g. Figs. 21-29 depict the obtained results, for a varying number of layers.

First, the result obtained by the DiscoGAN method with k 1 is displayed.

The results of applying Alg.

1 are then displayed for a varying k 2 .As can be seen, our algorithm leads to more sophisticated mappings.

BID13 have noted that their solutions are, at many times, related to texture or style transfer and, for example, geometric transformations are not well captured.

The new method is able to better capture such complex transformations.

Consider the case of mapping male to female in FIG2 , first row.

A man with a beard is mapped to a female image.

While for g the beard is still somewhat present, it is not so for h with k 2 > k 1 .

On the female to male mappings in FIG1 it is evident in most mappings that g produces a more blurred image, while h is more coherent for k 2 > k 1 .

Another example is in the blond to black hair mapping in FIG2 .

In the 5th row, the style transfer nature of g is evident, since it maps a red object behind the head together with the whole blond hair, producing an unrealistic black hair.

h of complexity k 2 = 8 is able to separate that object from the hair, and in k 2 > 8 it produces realistic looking black hair.

This kind of transformation requires more than a simple style transfer.

On the edges to shoes and edges to handbags mappings of FIG2 and FIG2 , while the general structure is clearly present, it is significantly sharpened by mapping h with k 2 > k 1 .For the face datasets, we also employ face descriptors in order to learn whether the mapping is semantic.

Namely, we can check if the identity is preserved post mapping by comparing the VGG face descriptors of Parkhi et al. (2015) .

One can assume that two images that match will have many similar features and so the VGG representation will be similar.

The cosine similarities are used, as is commonly done.

In addition, we train a linear classifier in the space of the VGG face descriptors in order to distinguish between Male/Female, Eyeglasses/No-eyeglasses, and Blond/Black.

This way, we can check, beyond discrepancy, that the mapping indeed transforms between the domains.

The training samples in domains A and B are used to train this classifier, which is then applied to a set of test images before and after mapping, measuring the accuracy.

The higher the accuracy, the better the separation.

Tab.

4 presents the results for both the k 1 layers network g, alternative networks g of higher complexity (shown as baseline only), and the network h trained using Alg.

1.

We expect the alignment of g to be best at complexity k 1 , and worse due to the loss of discrepancy for alternative network g with complexity k > k 1 .

We expect this loss of alignment to be resolved for networks h trained with Alg.

1.In the experiments of black to blond hair and blond to black hair mappings, we note that h with k 2 = 8 has the best descriptor similarity, and very good separation accuracy and discrepancy.

Higher values of k 2 are best in terms of separation accuracy and discrepancy, but lose somewhat in descriptor similarity.

A similar situation occurs for male to female and female to male mappings and in eyeglasses to non-eyeglasses, where k 2 = 8 results in the best similarity score and higher values of k 2 result in better separation accuracy and discrepancy.

It is interesting to note, that the distance between g and h is also minimal for k 2 = 8.

Perhaps, with more effective optimization, higher complexities could also maintain similarity, while delivering lower discrepancies.

Our stratified complexity model is related to structural risk minimization (SRM) by Vapnik & Chervonenkis (1971a) , which employs a hierarchy of nested subsets of hypothesis classes in order of increasing complexity.

In our stratification, which is based on the number of layers, the complexity classes are not necessarily nested.

A major emphasis in SRM is the dependence on the number of samples: the algorithm selects the hypothesis from one of the nested hypothesis classes depending on the amount of training data.

In our case, one can expect higher values of k 2 to be beneficial as the number of training samples grows.

However, the exact characterization of this relation is left for future work.

Alg.

1 can be seen as a form of distillation.

The first step of the algorithm finds the minimal complexity for mapping between the two domains and obtains the first generator.

Then, a second generator, with a large complexity, is trained while being encouraged to output images which are close to the output of the first generator.

This resembles the distillation methods proposed by BID11 and later analyzed by BID10 .Since the method depicted in Alg.

1 optimizes, among other things, the architecture of the network, our method is somewhat related to work that learn the network's structure during training, e.g., (Saxena & Verbeek, 2016; Wen et al., 2016; Liu et al., 2015; BID7 BID15 .

This body of work, which deals exclusively with supervised learning, optimizes the networks loss by modifying both the parameters and the hyperparameters.

For GAN based loss, this would not work, since with more capacity, one can reduce the discrepancy but quickly lose the alignment.

Indeed, we point to a key difference between supervised learning and unsupervised learning.

While in the former, deeper networks, which can learn even random labels, work well (Zhang et al., 2017) , unsupervised learning requires a careful control of the network capacity.

This realization, which echoes the application of MDL for model selection in unsupervised learning (Zemel, 1994) , was overshadowed by the overgeneralized belief that deeper networks lead to higher accuracy.

The limitations of unsupervised based learning that are due to symmetry, are also a part of our model.

For example, the mapping of cars in one pose to cars in the mirrored pose that sometimes happens in BID13 , is similar in nature to the mapping of x to 1 − x in the simple example given in Sec. 3.1.

Such symmetries occur when we can divide y AB into two functions y AB = y 2 • y 1 such that a function W is a linear mapping and also a DPM of y 1 • D A and, therefore, DISPLAYFORM0 While we focus on unsupervised learning, the emergence of semantics when learning with a restricted capacity is widely applicable, such as with autoencoders, transfer learning, semi-supervised learning and elsewhere.

As an extreme example, Sutskever et al. FORMULA0 present empirical evidence that a meaningful mapper can be learned, even from very few examples, if the network trained is kept small.

The recent success in mapping between two domains in an unsupervised way and without any existing knowledge, other than network hyperparameters, is nothing less than extraordinary and has far reaching consequences.

As far as we know, nothing in the existing machine learning or cognitive science literature suggests that this would be possible.

We provide an intuitive definition of function complexity and employ it in order to identify minimal complexity mappings, which we conjecture play a pivotal role in this success.

If our hypothesis is correct, simply by training networks that are not too complex, the target mapping stands out from all other alternative mappings.

Our analysis leads directly to a new unsupervised cross domain mapping algorithm that is able to avoid the ambiguity of such mapping, yet enjoy the expressiveness of deep neural networks.

The experiments demonstrate that the analogies become richer in details and more complex, while maintaining the alignment.

We show that the number of low-discrepancy mappings that are of low-complexity is expected to be small.

Our main proof is based on the assumption of identifiability, which constitutes an open question.

We hope that there would be a renewed interest in this problem, which has been open for decades for networks with more than a single hidden layer and is unexplored for modern activation functions.

FIG1 : Results for mapping Males to itself (B=A) using a DiscoGAN architecture and enforcing that the mapping is not the identity mapping.

The odd rows present the learned mapping h, and the even rows present the full cycle h • h.

--------Number of layers: --------Input op 4 6 8 10 12 14 FIG1 : Results for mapping the Females to itself (B=A) using a DiscoGAN architecture and enforcing that the mapping is not the identity mapping.

The odd rows present the learned mapping h, and the even rows present the full cycle h • h. DISPLAYFORM0

Published as a conference paper at ICLR 2018 FIG1 : Results for mapping shoe edges to itself (B=A) using a DiscoGAN architecture and enforcing that the mapping is not the identity mapping.

The odd rows present the learned mapping h, and the even rows present the full cycle h • h. DISPLAYFORM0 --------Number of layers: --------Input op 4 6 8 10 12 14 DISPLAYFORM1 Figure 17: Results for mapping handbag edges to itself (B=A), using a DiscoGAN architecture and enforcing that the mapping is not the identity mapping.

The odd rows present the learned mapping h, and the even rows present the full cycle h • h.--------Number of layers: --------Input op 4 6 8 10 12 14 FIG1 : Results for mapping handbags to itself (B=A), using a DiscoGAN architecture and enforcing that the mapping is not the identity mapping.

The odd rows present the learned mapping h, and the even rows present the full cycle h • h. DISPLAYFORM2

Published as a conference paper at ICLR 2018 FIG1 : Results for mapping shoes to itself (B=A) using a DiscoGAN architecture and enforcing that the mapping is not the identity mapping.

The odd rows present the learned mapping h, and the even rows present the full cycle h • h. DISPLAYFORM0 DISPLAYFORM1 Figure 20: Results for Alg.

1 on Male2Female dataset for mapping male to female.

Shown is a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM2 Figure 21: Results for Alg.

1 on Male2Female dataset for mapping female to male.

Shown is a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM3 Figure 22: Results for Alg.

1 on celebA dataset for mapping blond to black.

Shown is a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM4 Figure 23: Results for Alg.

1 on celebA dataset for mapping black to blond.

Shown is a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM5 Figure 24: Results for Alg.

1 on Eyeglasses dataset for mapping eyeglasses to no eyeglasses.

Shown is a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM6 Figure 25: Results for Alg.

1 on Eyeglasses dataset for mapping no eyeglasses to eyeglasses.

Shown is a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM7 Figure 26: Results for Alg.

1 on Edges2Handbags dataset for mapping edges to handbags.

Shown is a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM8 Figure 27: Results for Alg.

1 on Edges2Handbags dataset for mapping handbags to edges.

Shown are a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM9 Figure 28: Results for Alg.

1 on Edges2Shoes dataset for mapping edges to shoes.

Shown are a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

DISPLAYFORM10 Figure 29: Results for Alg.

1 on Edges2Shoes dataset for mapping shoes to edges.

Shown are a minimal complexity mapping g that has low discrepancy, and various mappings h obtained by the method.

For brevity, we have not presented our results in the most general way.

For example, in Def.

1, we did not bound the complexity of the discriminators.

For the same reason, some of our terms were described and not yet formally defined.

In order to model the composition of neural networks, we define a complexity measurement that assigns a value based on the number of simple functions that make up a complex function.

Definition 3 (Stratified complexity model (SCM)).

A stratified complexity model N := SCM[C] is a hypothesis class of functions p : R M → R M specified by a set of functions C. Every function p in N has an appropriate decomposition: DISPLAYFORM0 .., p n ∈ C} and C 0 = {Id}).• Every function in C is invertible.

A SCM partitions a set of invertible functions into disjoint complexity classes, DISPLAYFORM1 When considering simple functions p i that are layers in a neural network, each complexity class contains the functions that are implemented by networks of n hidden layers.

In addition, we denote the complexity of a function p: DISPLAYFORM2 If the complexity of a function p equals n, then any appropriate decomposition p = p n • ...

• p 1 will be called a minimal decomposition of p. According to this measurement, the complexity of a function p is determined by the minimal number of primitive functions required in order to represent it.

In this work, we focus our attention on SCMs that represent the architectures of fully connected neural networks with layers of a fixed size, i.e.,

A NN-SCM is a SCM N = SCM[C] that satisfies the following conditions: DISPLAYFORM0 both linear transformations and the associated matrix forms.• σ is a non-linear element-wise activation function.

For brevity, we denote N := SCM[σ] to refer to a NN-SCM with the activation function σ.

The NN-SCM with the Leaky ReLU activation function is of a particular interest, since BID13 BID12 employ it as the main activation function (plain ReLUs and tanh are also used).

In the NN-SCM framework, to specify the function obtained by a decomposition DISPLAYFORM1 • σ • W 1 we simply write: DISPLAYFORM2 It is useful to characterize the effect of inversion on the complexity of functions, since, for example, we consider both h = Π • h and h = Π −1 • h .

The following lemma states that, in the case of NN-SCM with σ that is the Leaky ReLU, the complexity of the inverse function is the same as that of the original function.

DISPLAYFORM3 be a NN-SCM with σ that is the Leaky ReLU with parameter 0 < a = 1.

Then, for any u ∈ N , C(u −1 ) = C(u).Proof.

First, we denote C (p) the minimal number n such that there are invertible linear mappings DISPLAYFORM4 .

This complexity measure is similar to the complexity measure C. For a function p such that C(p) = 0, we have, C(p) = C (p).

Nevertheless, for p such that C(p) = 0, it is not necessarily true that C (p) = 0.

For example, if p = Id is an invertible linear mapping, we have, C(p) = 0 and DISPLAYFORM5 be any function such that C(p) = 1.

We consider that: DISPLAYFORM6 Therefore, DISPLAYFORM7 In particular, DISPLAYFORM8 2 /a and, therefore, σ is a linear mapping -in contradiction.

Thus, C (p −1 ) = 1.Next, we would like to show that for any u ∈ N , C (u DISPLAYFORM9 In particular, DISPLAYFORM10 or, DISPLAYFORM11 Therefore, by Lem.

7, DISPLAYFORM12 On the other hand, if DISPLAYFORM13 .

Finally, we would like to show that for every u ∈ N , we have: C(u −1 ) = C(u).

If C(u) = 0, then, by Lem.

11, C(u −1 ) = 0.

On the other hand, if C(u) = 0, then, by Lem.

11, C(u −1 ) = 0 and by the above: C(u) = C (u) = C (u −1 ) = C(u).

Based on our simplicity hypothesis, we present a definition of a minimal complexity mapping that is both intuitive and well-defined in concrete complexity terms.

Given two distributions D A and D B , a minimal complexity mapping f : X A → X B between domains A and B is a mapping that has minimal complexity among the functions h : DISPLAYFORM0 Consider, again, the example of a line segment in R M (Sec. 3.1) and the semantic space of the interval, [0, 1] ⊂ R. The two linear mappings, which map either segment ends to 0 and the other to 1 are minimal, when using f that are ReLU based neural networks.

Other mappings to this segment are possible, simply by permuting points on the segment in R M .

However, these alternative mappings have higher complexity, since the two mappings above are the only ones with the minimal possible complexity.

In order to measure the distance between h • D A and D B , we use the discrepancy distance, disc D .

In this work, we focus on classes of discriminators D of the form D m := {u|C(u) ≤ m} for some m ∈ N. In addition, for simplicity, we will write disc m := disc Dm .Definition 5 (Minimal complexity mappings).

Let N = SCM [C] .

Let A = (X A , D A ) and B = (X B , D B ) be two domains.

We define the (m, 0 )-minimal complexity between A and B as: DISPLAYFORM1 The set of (m, 0 )-minimal complexity mappings between A and B is: DISPLAYFORM2 We note that for any fixed 0 > 0, the sequence {C DISPLAYFORM3 is monotonically increasing as m tends to infinity.

In addition, we assume that for every two distributions of interest, D I and D J , and an error rate 0 > 0, there is a function h of finite complexity such that disc ∞ (h • D I , D J ) ≤ 0 .

Therefore, the sequence {C A,B .

For simplicity, sometimes we will assume that m = ∞. In this case, we will write H 0 (A, B) := H 0 (A, B; ∞).

Every neural network implementation gives rise to many alternative implementations by performing simple operations, such as permuting the units of any hidden layer, and then permuting back as part of the linear mapping in the next layer.

Therefore, it is first required to identify and address the set of transformations that could be inconsequential to the function which the network computes.

Definition 6 (Invariant set).

Let N = SCM[σ] be a NN-SCM.

The invariant set Invariant(N ) is the set of all τ : R M → R M that satisfy the following conditions: DISPLAYFORM0 • DISPLAYFORM1

For example, for neural networks with the tanh activation function, the set of invariant functions contains the linear transformations that take vectors, permute them and multiply each coordinate by DISPLAYFORM0 ] where e i is the i'th standard basis vector, π is a permutation over [M ] and i ∈ {±1} BID6 ).In the following lemma, we characterize the set of all invariant functions for σ that is Leaky ReLU with parameter 0 < a = 1.Lemma 2.

Let N = SCM[σ] with σ be Leaky ReLU with parameter 0 < a = 1.

Then, DISPLAYFORM1 Here, e i denotes the i'th standard basis vector in R M and Sym M is the set of permutations of [M ].Proof.

Let τ be an invertible linear mapping satisfying σ • τ = τ • σ.

We consider that for all i ∈ [M ] and vector x; σ( τ i , x ) = τ i , σ(x) , where τ i is the i'th row of τ and τ i,j is the (i, j) entry of τ .

For x = e j , we have: DISPLAYFORM2 For x = −e j , we have: DISPLAYFORM3 If τ i,j < 0, then the first equation leads to contradiction.

Otherwise, the equations are both satisfied.

Finally, for x = e j − e k , we have: DISPLAYFORM4 If τ i,j −τ i,k = 0, then, τ i,j −aτ i,k = 0 and since a = 1, 0, we have, DISPLAYFORM5 there is at most one entry τ i,j that is not 0.

If for all j ∈ [M ], τ i,j = 0, then the mapping τ is not invertible, in contradiction.

Therefore, for each i ∈ [M ] there is exactly one entry τ i,j > 0 (it is non-negative as shown above).

Finally, if there are i 1 = i 2 such that τ i1,j , τ i2,j = 0 then the matrix is invertible.

Therefore, τ is a member of the set defined in Eq. 20.

In addition, it is easy to see that every member of the noted set satisfies the conditions of the invariant set.

Thus, we obtain the desired equation.

Our analysis is made much simpler, if every function has one invariant representation up to a sequence of manipulations using invariant functions that do not change the essence of the processing at each layer.

Assumption 1 (Identifiability).

Let N = SCM[σ] with σ that is Leaky ReLU with parameter 0 < a = 1.

Then, every function p ∈ N is identifiable (with respect to Invariant(N )), i.e., for any two minimal decompositions, DISPLAYFORM6 , there are invariants τ 1 , ..., τ n ∈ Invariant(N ) such that: DISPLAYFORM7 Uniqueness up to invariants, also known as identifiability, forms an open question.

BID6 proved identifiability for the tanh activation function.

Other works (Williamson & Helmke, 1995; BID5 BID14 Sussmann, 1992) prove such uniqueness for neural networks with only one hidden layer and various classical activation functions.

In the following lemma, we show that identifiability holds for Leaky ReLU networks with only one hidden layer.

DISPLAYFORM8 with σ that is Leaky ReLU with parameter 0 < a = 1.

Any function DISPLAYFORM9 Proof.

An alternative representation of the equation is: DISPLAYFORM10 We would like to prove that if σ • U = V • σ then V = U .

We have: DISPLAYFORM11 In particular, if v i is the i'th row of V (similarly u i ) and x = e j : DISPLAYFORM12 where v i,j is the (i, j) entry of V (similarly u i,j ).

Similarly, for x = −e j : DISPLAYFORM13 If u i,j is negative, we obtain: au i,j = v i,j (the first equation) and −u i,j = −av i,j (the second equation) that yields a = 1 in contradiction.

Therefore, u i,j ≥ 0 and u i,j = v i,j (the second equation).We conclude that DISPLAYFORM14 As far as we know, there are no other results continuing the identifiability line of work for activation functions such as Leaky ReLU.

Uniqueness, which is stronger than identifiability, since it means that even multiple representations with different number of layers do not exist, does not hold for these activation functions.

To see this, note that for every M × M invertible linear mapping W , the following holds: DISPLAYFORM15 where σ is the Leaky ReLU activation function with parameter a. We conjecture that for networks with Leaky ReLU activations identifiability holds, or at least for networks with a fixed number of neurons per layer.

In addition to identifiability, we make the following assumption, which states that almost all mappings are non-degenerate.

DISPLAYFORM16 with σ that is Leaky ReLU with parameter 0 < a = 1.

Assume that the set of (W 1 , ..., DISPLAYFORM17

In the unsupervised alignment problem, the algorithms are provided with only two unmatched datasets of samples from the domains A and B and the task is to learn a well-aligned function between them.

Since we hypothesize that the alignment of the target mapping is typically captured by the lowest complexity low-discrepancy mapping, we develop the machinery needed in order to show that such mappings are rare.

Recall that disc m is the discrepancy distance for discriminators of complexity up to m. In Sec. 2, we have discussed the functions Π which replaces between members in the domain B that have similar probabilities.

Formally, these are defined using the discrepancy distance.

Definition 7 (Density preserving mapping).

DISPLAYFORM0 We denote the set of all (m, 0 )-DPMs of complexity k by DPM 0 (X; m, DISPLAYFORM1 We would like to bound the number of mappings that are both low-discrepancy and low-complexity by the number of DPMs.

We consider that there are infinitely many DPMs.

For example, if we slightly perturb the weights of a minimal representation of a DPM, Π, we obtain a new DPM.

Therefore, we define a similarity relation between functions that reflects whether the two are similar.

In this way, we are able to bound the number of different (non-similar) minimal-complexity mappings by the number of different DPMs.

• We denote DISPLAYFORM0 • We denote f DISPLAYFORM1 n and there are minimal decompositions: f = DISPLAYFORM2 The defined relation is reflexive and symmetric, but not transitive.

Therefore, there are many different ways to partition the space of functions into disjoint subsets such that in each subset, any two functions are similar.

We count the number of functions up to the similarity as the minimal number of subsets required in order to cover the entire space.

This idea is presented in Def.

9, which slightly generalizes the notion of covering numbers (Anthony & Bartlett, 2009 ).Definition 9 (Covering number).

Let (U, ∼ U ) be a set and a reflexive and symmetric relation.

A covering of (U, ∼ U ), is a tuple (U, ≡ U ) such that: ≡ U is an equivalence relation and DISPLAYFORM3 min U/ ≡ U s.t: the minimum is taken over (U, ≡ U ) that is a covering of (U, ∼ U )Here, U/ ≡ U is the quotient set of U by ≡ U .Thm.

1 below states that the number of low discrepancy mappings of complexity C DISPLAYFORM4 Proof.

See Sec. D.

Tab.

5 lists the symbols used in our work.

Functions from the feature space to the domains, y A : X → X A and DISPLAYFORM0 The risk function DISPLAYFORM1 where is a loss function and DISPLAYFORM2 The discrepancy between two distributions D 1 and DISPLAYFORM3 A SCM specified by a class of functions C (see Def.

3) DISPLAYFORM4 A NN-SCM specified by the activation function σ (see Def.

4) DISPLAYFORM5 The complexity of a function p (see Eqs. 9, 10) Invariant(N )The invariant The set of ( 0 , m)-minimal complexity mappings between A and B (see Def.

5) DISPLAYFORM6 The covering number of U with respect to relation ∼ U on U (see Def.9) X :← x x is assigned to X

In this section, we prove various lemmas that are used in the proof of Thm.

1.

In Sec. C.1 we present the assumptions taken in various lemmas in the appendix.

In Sec. C.2 we prove useful inequalities involving the discrepancy distance.

Sec. C.3 provides lemmas concerning the defined complexity measure and invariant functions.

The lemmas in Sec. C.4 concern the properties of inverse functions.

We list the assumptions employed in our proofs.

Assumptions 1 and 2 were already presented and are heavily used.

Assumptions 3 and its relaxation 4 are mild assumptions that were taken for convenience.

Assumption 1 (Identifiability).

Let N = SCM[σ] with σ that is Leaky ReLU with parameter 0 < a = 1.

Then, every function p ∈ N is identifiable (with respect to Invariant(N )), i.e., for any two minimal decompositions, DISPLAYFORM0 .., τ n ∈ Invariant(N ) such that: DISPLAYFORM1 with σ that is Leaky ReLU with parameter 0 < a = 1.

Assume that the set of (W 1 , ..., DISPLAYFORM2 with σ that is Leaky ReLU with parameter 0 < a = 1.

For every m > 0 (possibly ∞) and n > 0, the function disc DISPLAYFORM3 In the case that the norm of the discriminator is bounded, Lem 19, it follows from the following assumption, which is well-justified, (cf.

Shalev-Shwartz & Ben-David (2014), page 162, Eq.14.13).

Lemma 4.

Let D 1 and D 2 be two classes of functions and D 1 , D 2 two distributions.

Assume that DISPLAYFORM0 Proof.

By the definition of discrepancy: DISPLAYFORM1 Since D 1 • {p} ⊂ D 2 we have: DISPLAYFORM2 The second inequality is a special case for DISPLAYFORM3 Lemma 5.

Let A = (X 1 , D 1 ) and B = (X 2 , D 2 ) be two domains and DISPLAYFORM4 2.

Let y 1 , y 2 and y = y 2 • y −1 1 be three functions and DISPLAYFORM5 3.

Let h be any function and m ≥ k + C(h −1 ).

Then, DISPLAYFORM6 Proof.

1.

Follows from Lem.

4, since m ≥ k + C(p), we have: DISPLAYFORM7 Therefore, by the triangle inequality, DISPLAYFORM8 2.

We use Lem.

4 with p :← y 2 , D 1 :← D k , and DISPLAYFORM9 Therefore, by the triangle inequality, DISPLAYFORM10 3.

Follows immediately from Lem.

4 for p :← h −1 and DISPLAYFORM11

Lemma 6.

Let N = SCM [C] .

In addition, let u, v be any two functions.

Then, DISPLAYFORM0 Proof.

We begin with the case C(v) = 0.

In this case, DISPLAYFORM1 The case C(u) = 0 is analogous.

Next, we assume that DISPLAYFORM2 • v 1 be minimal decompositions of u and v (resp.).

Therefore, we can represent, DISPLAYFORM3 The lower bound follows immediately from the upper bound: DISPLAYFORM4 By similar considerations, we also have: DISPLAYFORM5 For a given function u ∈ N = SCM[C], we define, DISPLAYFORM6 .

In addition, let u, v be any two functions.

Then, DISPLAYFORM7 Proof.

We begin by proving the upper bound.

We assume C (u) = n and DISPLAYFORM8 • v 1 be minimal decompositions of u and v (resp.).

Therefore, we can represent, DISPLAYFORM9 The lower bound follows immediately from the upper bound: DISPLAYFORM10 By similar considerations, DISPLAYFORM11 Lemma 8.

Invariant(N ) is closed under inverse and composition, i.e, DISPLAYFORM12 And, DISPLAYFORM13 Proof.

Inverse: Let τ ∈ Invariant(N ).

Then, by definition, τ is an invertible linear mapping and τ • σ = σ • τ .

In particular, τ −1 is also an invertible linear mapping and τ DISPLAYFORM14 Composition: Let τ 1 , τ 2 ∈ Invariant(N ).

Then, τ i is an invertible linear mapping and τ i • σ = σ • τ i for i = 1, 2.

In particular, τ 1 • τ 2 is also an invertible linear mapping and DISPLAYFORM15 with σ that is Leaky ReLU with 0 < a = 1.

Assume that p obeys identifiability, i.e., that Assumption 1 holds.

Then, for any two minimal decompositions DISPLAYFORM16 we have: DISPLAYFORM17 and DISPLAYFORM18 Proof.

We prove that DISPLAYFORM19 Otherwise, by minimal identifiability, DISPLAYFORM20 In addition, DISPLAYFORM21 Since each for all k ∈ [i], τ k commutes with σ, we have, DISPLAYFORM22 and DISPLAYFORM23 By similar considerations, DISPLAYFORM24 Lemma 10.

Let N = SCM[σ] with σ that is Leaky ReLU with parameter 0 < a = 1.

Then, every invertible linear mapping W is a member of C 0 .

DISPLAYFORM25 Lemma 11.

C 0 is closed under inverse and composition, i.e, DISPLAYFORM26 and, DISPLAYFORM27 Proof.

Inverse: By definition, u ∈ C 0 iff for all n ∈ N and q ∈ C n , we have: DISPLAYFORM28 Published as a conference paper at ICLR 2018 DISPLAYFORM29 where σ is the Leaky ReLU activation function, with parameter 0 < a = 1.

Let f = F [W n+1 , ..., W 1 ] be a minimal decomposition.

Then, for all i ∈ [n], we have: DISPLAYFORM30 Proof.

We prove this statement by induction on i from i = n backwards to i = 1.

DISPLAYFORM31 Induction hypothesis: We assume that: DISPLAYFORM32 Case i − 1: We consider that by the induction hypothesis: DISPLAYFORM33 (61) Finally, we conclude that: DISPLAYFORM34 Lemma 13.

Let N = SCM[σ] with σ that is Leaky ReLU with parameter 0 < a = 1.

Then, for all DISPLAYFORM35 By the third part of Lem.

5, for h :← y, we have: DISPLAYFORM36 In particular, C Definition 10 (Set embedding).

Let (U, ∼ U ) and (V, ∼ V ) be two tuples of sets and symmetric and reflexive relations on them (resp.).

A function G : U → V is an embedding of (U, ∼ U ) in (V, ∼ V ) and we denote (U, ∼ U ) (V, ∼ V ) if: DISPLAYFORM37 Lemma 14.

Let (U, ∼ U ) and (V, ∼ V ) be two tuples of sets and reflexive and symmetric relations on them (resp.).

DISPLAYFORM38 Then, by definition, there is an embedding function G : U → V such that: DISPLAYFORM39 Let (V, ≡ V ) be a covering of (V, ∼ V ).

We define a covering (U, ≡ U ) of (U, ∼ U ) as follows: DISPLAYFORM40 Part 1: We would like to prove that (U, ≡ U ) is a covering of (U, ∼ U ).

It is easy to see that ≡ U is an equivalence relation since ≡ V is an equivalence relation.

Next, we would like to prove that DISPLAYFORM41 By the definition of ≡ U : DISPLAYFORM42 In addition, since (V, ≡ V ) is a covering of (V, ∼ V ): DISPLAYFORM43 Finally, since G is an embedding: DISPLAYFORM44 We conclude: DISPLAYFORM45 Therefore, (U, ≡ U ) is indeed a covering of (U, ∼ U ).Part 2: We would like to prove that DISPLAYFORM46 Then, by definition of ≡ U we have: DISPLAYFORM47 Therefore, the covering number of (U, ∼ U ) is at most the covering number of (V, ∼ V ).Lemma 15.

Let (U, ≡ 1 ) and (U, ≡ 2 ) be two coverings of (U, ∼ U ).

Then, (U 2 , ≡ 1 × ≡ 2 ) is a covering of (U 2 , ∼ 2 U ).

Where U 2 = U × U and the relation ∼ 2 U is defined as follows: DISPLAYFORM48 and ≡ 1 × ≡ 2 is defined as: DISPLAYFORM49 Proof.

We have to prove that ≡ 1 × ≡ 2 is an equivalence relation and that DISPLAYFORM50 The RHS is true since ≡ 1 and ≡ 2 are reflexive relations.

Symmetry: DISPLAYFORM51 Since ≡ 1 and ≡ 2 are symmetric, we have: DISPLAYFORM52 In addition, DISPLAYFORM53 Therefore, DISPLAYFORM54 Transitivity: follows from similar arguments.

Covering: DISPLAYFORM55 Since (U, ≡ i ) is a covering of (U, ∼ U ), for i = 1, 2, we have: DISPLAYFORM56 By the definition of ∼ 2 U we have: DISPLAYFORM57 Therefore, DISPLAYFORM58 Lemma 16.

Let (U, ∼ U ) be a tuple of a set and a reflexive and symmetric relation on it (resp.).

Then, DISPLAYFORM59 Proof.

Let ≡ U be an equivalence relation such that (U, ≡ U ) is a covering of (U, ∼ U ).

By Lem.

15, DISPLAYFORM60 Thus, for every covering DISPLAYFORM61 Lemma 17.

Let (U, ∼ U ) be a tuple of a set and a reflexive and symmetric relation on it (resp.).

Then, DISPLAYFORM62 Proof.

We define an embedding from (U, u, u) .

This is an embedding, because, DISPLAYFORM63 DISPLAYFORM64 Lemma 18.

Let (U, ∼ U ) and (V, ∼ V ) be two tuples of sets and reflexive and symmetric relations on them (resp.).

Assume that U ⊂ V and DISPLAYFORM65 Proof.

Let (V, ≡ V ) be a covering of (V, ∼ V ).

Then, it is easy to see that (U, ≡ U ) is a covering of (U, ∼ U ), where ≡ U := (≡ V ) U .

In addition, we have: |U/ ≡ U | ≤ |V/ ≡ V |.

Thus, for every covering of (V, ∼ V ), we can find a smaller covering for (U, ∼ U ).

In particular, N(U, DISPLAYFORM66

Thm.

1 employs assumption 3.

In Lem.

19 we prove that this assumption holds for the case of a continuous risk (assumption 4) if the discriminators have bounded weights.

DISPLAYFORM0 Then, for all m > 0, n > 0 and E > 0, the function disc m,E (F [W n , ..., DISPLAYFORM1 DISPLAYFORM2 therefore, DISPLAYFORM3 In particular, there is some > 0 and an increasing sequence {k j } ∞ j=1 ⊂ N such that Q kj > for all j ∈ N. With no loss of generality, we can assume that k j = j (otherwise, we replace the original sequence with the new one).

Since (V DISPLAYFORM4 DISPLAYFORM5 Therefore, by the triangle inequality, DISPLAYFORM6 in contradiction.

Thus, we conclude that: DISPLAYFORM7 DISPLAYFORM8 • DISPLAYFORM9 • DISPLAYFORM10 we have: DISPLAYFORM11 • DISPLAYFORM12 DISPLAYFORM13 By Assumption 1, DISPLAYFORM14 n .

Therefore, we define a minimal decomposition for g as follows: DISPLAYFORM15 n .

This is a minimal decomposition of g, since each invariant function is an invertible linear mapping and commutes with σ.

By Lem.

9 we have: DISPLAYFORM16 Therefore, by Lem.

4, since C(τ i ) = 0, we have: DISPLAYFORM17 with σ that is Leaky ReLU with parameter 0 < a = 1.

We have: DISPLAYFORM18 Proof.

Assume by contradiction thatf DISPLAYFORM19 By Lem.

21, sincef DISPLAYFORM20 Since f DISPLAYFORM21 Therefore, by the triangle inequality, we arrive to a contradiction: DISPLAYFORM22 with σ that is a Leaky ReLU with parameter 0 < a = 1.

Let A = (X A , D A ) and B = (X B , D B ) are two domains.

We have: DISPLAYFORM23 Proof.

We would like to show that the function G(h) = h −1 is an embedding of DISPLAYFORM24 and by Lem.

1 and Lem.

13, A) .

Next, we would like to prove that for all h 1 , h 2 ∈ H 0 (A, B): DISPLAYFORM25 DISPLAYFORM26 We consider that by Lem.

1, DISPLAYFORM27 n+1 /a] are minimal decompositions.

In addition, by Lem.

12, we have: DISPLAYFORM28 By the first item of Lem.

5, for DISPLAYFORM29 Similarly (by the first item of Lem.

5), we have: DISPLAYFORM30 Therefore, we conclude that h 1 DISPLAYFORM31 with σ that is a Leaky ReLU with parameter 0 < a = 1.

Assume Assumptions 1, 2 and 3.

Let 0 , 1 and 2 such that 0 < 1 /2 and 2 < 1 − 2 0 be three positive constants and A = (X A , D A ) and B = (X B , D B ) are two domains.

Assume that m ≥ k+2C DISPLAYFORM32 Proof.

Let be any positive constant such that: < min{( 1 − 2 0 − 2 )/4, 2 /2}. For such , we have 2 0 ≤ 1 − 4 and 2 ≤ 1 − 2 0 − 4 .

In addition, let t := k + C 0 A,B + 1.

We would like to find an embedding mapping: DISPLAYFORM33 Part 1: In this part, we show how to construct G. Let (f, g) ∈ (H 0 (A, B; m)) 2 .

We denote: f = F [W n+1 , ..., W 1 ] and g = F [V n+1 , ..., V 1 ] minimal decompositions of f and g (resp.).

By Lem.

20, there are functionsf = F [W n+1 , ...,W 1 ] andḡ = F [V n+1 , ...,V 1 ] such that: DISPLAYFORM34 • DISPLAYFORM35 • DISPLAYFORM36 Part 2: In this part, we show that: DISPLAYFORM37 By Part 1, C(f •ḡ −1 ) = 2n = 2C 0 A,B .

In addition, by the first item of Lem.

5, for DISPLAYFORM38 Since f ∈ H 0 (A, B; m): DISPLAYFORM39 In addition, by the third item of Lem.

5, for h :←ḡ and m ≥ t + C 0 A,B ≥ t + C(ḡ −1 ), we have: DISPLAYFORM40 ≤ 2 0 + 2 and we conclude that: DISPLAYFORM41 Part 3: In this part, we show that G is an embedding.

It requires showing that DISPLAYFORM42 Assume by contradiction that G(f, g) DISPLAYFORM43 We denote G(f, g) =f •ḡ −1 and G(f , g ) =f • (ḡ ) −1 (see Part 1).Assume that f DISPLAYFORM44 f .

In particular, for every two decompositions: DISPLAYFORM45 there is an index i ∈ [n + 1] such that: DISPLAYFORM46 The option i = n + 1 is not a possibility, since: DISPLAYFORM47 By the first item of Lem.

5, for DISPLAYFORM48 Again, by the first item of Lem.

5, for DISPLAYFORM49 Alternatively, for any minimal decompositionsf DISPLAYFORM50 −1 such that: DISPLAYFORM51 in contradiction to F (f, g) DISPLAYFORM52 Assume that g DISPLAYFORM53 2 /a, ...,V DISPLAYFORM54 The case i = n + 1 is not a possibility, similarly to Eq. 126.

Therefore, there is i ∈ [n] such that Eq. 133 holds.

In addition, By Lem.

4, for p :

← −1/a · σ of complexity 1 we have: DISPLAYFORM55 In addition, by Lem.

5, for DISPLAYFORM56 such that Eq. 139 holds, in contradiction to F (f, g) DISPLAYFORM57 F (f , g ).

Alternatively, for all 0 , 1 , 2 , such that < min{( 1 − 2 0 − 2 )/4, 2 /2}, DISPLAYFORM58

Published as a conference paper at ICLR 2018In particular, we can replace with /2 in the inequality.

By Lem.

18, the function q = N DPM 2 0+ B; k, 2C

It is interesting to check whether the predictions made are valid for other forms of discrepancy such as the one used in the Wasserstein GAN Arjovsky et al. (2017) (WGAN) .

This is done below for Prediction 2, which predicts that the selection of the right number of layers is crucial in unsupervised learning.

In the WGAN experiment, we employ the architecture of BID13 and vary the number of layers and inspect the influence on the results.

For the generator, the architecture is identical while for WGAN's critic, the last sigmoid layer is removed.

These experiments were done on the CelebA dataset, obtaining the results in FIG9 .Note that since the encoder and the decoder parts of the learned network are symmetrical, the number of layers is always even.

As can be seen, changing the number of layers has a dramatic effect on the results.

The best overall results are obtained at 6 layers.

Using fewer layers, WGAN often fails to produce images of the desired class.

Adding layers, the semantic alignment is lost, as expected.

While most of our experiments have focused on the DiscoGAN architecture of BID13 , an additional experiment was conducted in order to verify that these extend to the CycleGAN architecture of BID12 .The results are shown in FIG10 .

As can be seen running an experiment on the Aerial images to Maps dataset, we found that 8 layers produces an aligned solution.

Using 10 layers produces unaligned map images with low discrepancy.

For fewer than 8 layer, the discrepancy is high and the images are not very detailed.

@highlight

Our hypothesis is that given two domains, the lowest complexity mapping that has a low discrepancy approximates the target mapping.

@highlight

The paper addresses the problem of learning mappings between different domains without any supervision, stating three conjectures.

@highlight

Demonstrates that in unsupervised learning on unaligned data it is possible to learn the between domains mapping using GAN only without a reconstruction loss.