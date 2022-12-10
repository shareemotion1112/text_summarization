Generative networks are promising models for specifying visual transformations.

Unfortunately, certification of generative models is challenging as one needs to capture sufficient non-convexity so to produce precise bounds on the output.

Existing verification methods either fail to scale to generative networks or do not capture enough non-convexity.

In this work, we present a new verifier, called ApproxLine, that can certify non-trivial properties of generative networks.

ApproxLine performs both deterministic and probabilistic abstract interpretation and captures infinite sets of outputs of generative networks.

We show that ApproxLine can verify interesting interpolations in the network's latent space.

Neural networks are becoming increasingly used across a wide range of applications, including facial recognition and autonomous driving.

So far, certification of their behavior has remained predominantly focused on uniform classification of norm-bounded balls Katz et al., 2017; Wong et al., 2018; Gowal et al., 2018; Singh et al., 2018; Raghunathan et al., 2018; Tjeng et al., 2017; Dvijotham et al., 2018b; Salman et al., 2019; Dvijotham et al., 2018c; , which aim to capture invisible perturbations.

However, a system's safety can also depend on its behavior on visible transformations.

For these reasons, investigation of techniques to certify more complex specifications has started to take place (Liu et al., 2019; Dvijotham et al., 2018a; Singh et al., 2019) .

Of particular interest is the work of Sotoudeh & Thakur (2019) which shows that if the inputs of a network are restricted to a line segment, the verification problem can sometimes be efficiently solved exactly.

The resulting method has been used to certify non-norm-bounded properties of ACAS Xu networks (Julian et al., 2018) and improve Integrated Gradients (Sundararajan et al., 2017) .

This work We extend this technique in two key ways: (i) we demonstrate how to soundly approximate EXACTLINE, handling significantly larger networks faster than even methods based on sampling can (a form of deterministic abstract interpretation), and (ii) we use this approximation to provide guaranteed bounds on the probabilities of outputs given a distribution over the inputs (a form of probabilistic abstract interpretation).

We believe this is the first time probabilistic abstract interpretation has been applied in the context of neural networks.

Based on these techniques, we also provide the first system capable of certifying interesting properties of generative networks.

• A verification system APPROXLINE, capable of flexibly capturing the needed non-convexity.

• A method to compute tight deterministic bounds on probabilities with APPROXLINE, which is to our knowledge the first time that probabilistic abstract interpretation has been applied to neural networks.

• An evaluation on autoencoders for CelebA, where we prove for the first time the consistency of image attributes through interpolations.

• The first demonstration of deterministic verification of certain visible, highly non-convex specifications, such as that a classifier for "is bald" is robust to different amounts of "moustache," or that a classifier n A is robust to different head rotations, as shown in Figure 1 .

Figure 1: Using APPROXLINE to find probability bounds for a generative specification over flipped images.

Green polygonal chains represent activation distributions at each layer exactly.

Blue boxes are relaxations of segments highlighted by yellow boxes.

We label regions with their probabilities.

Here, we introduce the terminology of robustness verification and provide an overview of our verification technique.

Let N : R m → R n be a neural network with m inputs and n output classes which classifies an input x ∈ R m to class arg max i N (x) i .

Specification A robustness specification is a pair (X, Y) where X ⊆ R m is a set of input activations and Y ⊆ R n is a set of permissible outputs for those inputs.

Deterministic robustness Given a specification (X, Y), a neural network N is said to be (X, Y)-robust if for all x ∈ X, we have N (x) ∈ Y. In the adversarial robustness literature, the set X is usually an l 2 -or l ∞ -ball, and Y is a set of outputs that correspond to a specific classification.

In our case, X shall be a line segment connecting two encodings.

The deterministic verification problem is to prove (ideally with 100% confidence) that a network is deterministically robust for a single specification.

As deciding robustness is NP-hard (Katz et al., 2017) , the problem is frequently relaxed to permit false negatives (but not false positives) and solved by sound overapproximation.

Probabilistic robustness Even if N is not completely robust, it may still be useful to quantify its lack of robustness.

Given a distribution µ over X, we are interested in finding provable bounds on the robustness probability Pr x∼µ [N (x) ∈ Y], which we call probabilistic [robustness] bounds.

It is well known that certain generative models appear to produce interpretable transformations between outputs for interpolations of encodings in the latent space (Dumoulin et al., 2016; Mathieu et al., 2016; Bowman et al., 2015; Radford et al., 2015; Mescheder et al., 2017; Ha & Eck, 2017; Dinh et al., 2016; Larsen et al., 2015; Van den Oord et al., 2016; Lu et al., 2018; He et al., 2019) .

I.e., as we move from one latent vector to another, there are interpretable attributes of the outputs that gradually appear or disappear.

This leads to the following verification question: given encodings of two outputs with a number of shared attributes, what fraction of the line segment between the encodings generates outputs sharing those same attributes?

To anwer this question, we can verify a generator using a trusted attribute detector, or we verify an attribute detector based on a trusted generator.

For both tasks, we have to analyze the outputs of neural networks restricted to line segments.

Sotoudeh & Thakur (2019) computes succinct representations of a piecewise-linear neural networks restricted to such line segments AB ⊂ R m .

It is visualized in the top row of Figure 1 , where a line segment e 1 e 2 ⊂ R m between encodings produced by an encoder neural network n E is passed through a decoder n D and an attribute detector n A .

In more detail, the primitive P(N | AB ) computes a polygonal chain (P 1 , . . .

, P k ) in R m representing the line segment AB, such that the neural network N is affine on the segment P i P i+1 for all 0 ≤ i < k. As a consequence, the polygonal chain (N (P 1 ), . . .

, N (P k )) represents the image of AB under N .

To compute this, one can incrementally compute normalized distances on the input segment AB for the output of each layer i of the network, N i .

Specifically, we find 0 (A − B) ), and keep track of the nodes N i (A + t i,j (A − B) ).

In the case of affine operations such as matrix multiplication or convolution, one can simply apply that operation to each node and leave the distances unchanged.

The case of ReLU more segments may be introduced.

To compute ReLU, one can apply it per-dimension, d, and check for each 0 ≤ j < k i whether

This is done analogously in the case where the segment is decreasing in dimension d instead of increasing.

We extend EXACTLINE to perform exact probabilistic inference by associating with each segment P j in the chain a probability distribution µ j over that segment, and a probability p j .

In the case of the uniform distribution, as in Figure 1 , every µ j is also a uniform distribution, and p j = t j+1 − t j .

APPROXLINE Unfortunately, EXACTLINE sometimes scales poorly for tasks using generative models, because too many line segments are generated.

We improve scaling by introducing a sound overapproximation, APPROXLINE.

Instead of maintaining a single polygonal chain, APPROXLINE maintains a list of polygonal chains and a list of interval constraints 1 , such that the neural network's activations are guaranteed to either lie on one of the polygonal chains or to satisfy one of the interval constraints.

We introduce a novel relaxation heuristic, which chooses subsets of line segments in polygonal chains and replaces them with interval constraints that subsume them.

Our relaxation heuristic attempts to combine line segments that are adjacent and short.

To perform probabilistic inference, each element also carries its probability.

For a feed-forward network, each operation acts on the elements individually, without modifying the probability associated with it.

This is shown in the bottom row of Figure 1 .

Here, the probabilities of the segments that get approximated are shown by the yellow regions in the top row.

One can observe that they remain unchanged when converted to intervals (in blue) in the bottom row.

One can also observe that probabilities associated with intervals do not change, even when the intervals change substantially.

Python pseudocode for propogation of APPROXLINE is shown in Appendix A.

To understand the interaction between probabilistic inference and the relaxation heuristic, it is best to work through a contrived example.

Suppose we have an EXACTLINE in two dimensions with the nodes (1, 1), (2, 2), (3, 2), (5, 1), (7, 5) with weights 0.1, 0.1, 0.5, 0.3 on its segments.

The weights describe the probabilities of the output being on each segment respectively, where distributions on the segments themselves are uniform.

Our relaxation heuristic might combine the line segments with nodes (1, 1), (2, 2) and (2, 2), (3, 2) and approximate them by a single interval constraint with lower bound (1, 1) and upper bound (3, 2).

(I.e., the first component is between 1 and 3, and the second component is between 1 and 2.)

In this case, we can understand the output of approximation as an APPROXLINE: An interval constraint with center (2, 1.5), radius (1, 0.5) and weight 0.1 + 0.1 = 0.2, as well as a polygonal chain formed of the segment (3, 2)(5, 1) with weight 0.5 and the segment (5, 1)(7, 5) with weight 0.3.

To compute a lower bound on the robustness probability, we would sum the probabilities of each element where it can be proven there is no violation.

For example, assume that only the point (1, 2) is disallowed by the specification.

The inferred robustness probability lower bound would be 0.8.

To compute an upper bound on the robustness probability, we sum the probabilities of each element where it can be proven that at least one point is safe.

Here, we obtain 1.

Dvijotham et al. (2018a) verify probabilistic properties universally over sets of inputs by bounding the probability that a dual approach verifies the property.

In contrast, our system verifies properties that are either universally quantified or probabilistic.

However, the networks we verify are multiple orders of magnitude larger.

While they only provide upper bounds on the probability that a specification has been violated, we provide extremely tight bounds on such probabilities from both sides.

PROVEN (Weng et al., 2018) uses sampling to find high confidence bounds (confidence intervals) on the probability of misclassification.

While PROVEN only provides high confidence bounds (99.99%), APPROXLINE provides bounds with 100% confidence.

Nevertheless, our method is much faster and produces better results than a similar sampling-based technique for finding confidence intervals using Clopper & Pearson (1934) (used by smoothing methods).

Another line of work is smoothing, which provides a defense with high confidence statistical robustness guarantees (Cohen et al., 2019; Lecuyer et al., 2018; Liu et al., 2018; Li et al., 2018; Cao & Gong, 2017) .

In contrast, APPROXLINE provides deterministic guarantees, and is not a defense.

We briefly review important concepts, closely following their presentations given in previous work where applicable.

In our work, we assume that we can decompose the neural network as a sequence of l piecewise-linear layers:

An abstract domain (Cousot & Cousot, 1977 ) is a set of symbolic representations of sets of program states.

We write A n to denote an abstract domain whose elements each represent an element of P(R n ), in our case a set of vectors of n neural network activations.

The concretization function γ n : A n → P(R n ) maps a symbolic representation a ∈ A n to its concrete interpretation as a set X ∈ P(R n ) of neural network activation vectors.

The concrete transformer

Using this notation, the (X, Y)-robustness property of a neural network N can be written as

An abstract transformer T # f : A m → A n transforms symbolic representations to symbolic representations overapproximating the effect of the function f :

g .

We will follow this recipe for the neural network N , abstracting it as

Abstract interpretation provides a sound, typically incomplete method to certify neural network robustness.

Namely, to show that a neural network N :

n }.

Abstract interpretation with the box domain B is equivalent to bounds propagation with standard interval arithmetic.

Powerset domain Given an abstract domain A, elements of its powerset domain P(A) n are (finite) sets of elements of A n .

The concretization function is given by γ n (a) = a ∈a γ n (a ) (using the concretization function of the underlying domain A).

We can lift any abstract transformer for A to an abstract transformer for P(A) by applying the transformer to each of the elements.

Union domain Given abstract domains A and A , an element of their union domain is a tuple (a, a ) with a ∈ A n and a ∈ A n .

The concretization function is γ n (a, a ) = γ n (a) ∪ γ n (a ).

We can apply abstract transformers of the same function for A and A to the tuple elements independently.

We denote as D n the set of probability measures over R n .

Probabilistic abstract interpretation is an instantiation of abstract interpretation where deterministic points from R n are replaced by measures from D n .

I.e., a probabilistic abstract domain (Cousot & Monerau, 2012 ) is a set of symbolic representations of sets of measures over program states.

We again use subscript notation to determine the number of activations: a probabilistic abstract domain A n has elements that each represent an element of P(D n ).

The probabilistic concretization function γ n : A n → P(D n ) maps each abstract element to the set of measures it represents.

For a measurable function f : R m → R n , the corresponding probabilistic concrete transformer

where Y ranges over measurable subsets of R n .

A probabilistic abstract transformer T # f : A m → A n abstracts the probabilistic concrete transformer in the standard way: it satisfies ∀a ∈ A m .

T f (γ m (a)) ⊆

γ n (T # f (a)), as in the deterministic setting.

Probabilistic abstract interpretation provides a sound method to compute bounds on robustness probabilities.

Namely, to show that

Domain lifting Any deterministic abstract domain can be directly interpreted as a probabilistic abstract domain, where the concretization of an element is given as the set of probability measures whose support is a subset of the deterministic concretization.

The original deterministic abstract transformers can still be used.

Convex combinations Given two probabilistic abstract domains A and A , we can form their convex combination domain, whose elements are tuples (a, a , p) with a ∈ A n , a ∈ A n and p ∈ [0, 1].

The concretization function is given by γ n (a, a , p)

We can apply abstract transformers of the same function for A and A to the respective elements of the tuple independently, leaving p intact.

Similarly, given a single probabilistic abstract domain A, elements of its convex combination domain are tuples (a, λ) where

. .

, k}}. We can apply abstract transformers for A independently to each entry of a, leaving λ intact.

Here we define APPROXLINE, its non-convex relaxations, and its usage for probabilistic inference.

First, note that we can use EXACTLINE to create an abstract domain E. The elements of E n are polygonal chains (P 1 , . . .

, P k ) in R n for some k.

The concretization function γ n maps a polygonal chain (P 1 , . . .

, P k ) in R n to the set of points in R n that lie on it.

For a piecewise-linear function

m to a new polygonal chain in R n by concatenating the results of the EXACTLINE primitive on consecutive line segments P i P i+1 , eliminating adjacent duplicate points and applying the function f to all points.

The resulting abstract transformers are exact, i.e., they satisfy the subset relation in

) with equality.

Our abstract domain is the union of the powersets of the EXACTLINE and box domains.

Therefore, an abstract element is a tuple of a set of polygonal paths and a set of boxes, whose interpretation is that the activations of the neural network in a given layer are on one of the polygonal paths or within one of the boxes.

For x 1 , x 2 ∈ R n , we write S(x 1 , x 2 ) = ({(x 1 , x 2 )}, {}) to denote the abstract element that represents a single line segment connecting x 1 and x 2 .

Like EXACTLINE, we focus on the case where the abstract element describing the input activations captures such a line segment.

Note that if we use the standard lifting of abstract transformers T # Li for the EXACTLINE and box domains into our union of powersets domain, propagating a segment S(

is equivalent to using only the EXACTLINE domain: As the standard lifting applies the abstract transformers to all elements of both sets independently, we will simply obtain an abstract element ({(P 1 , . . .

, P k ), {}), where (P 1 , . . .

, P k ) is a polygonal path exactly describing the image of x 1 x 2 under N .

Relaxation Therefore, our abstract transformers may, before applying a lifted abstract transformer, apply relaxation operators that turn an abstract element a into another abstract element a such that γ n (a) ⊆ γ n (a ).

We use two kinds of relaxation operators: bounding box operators remove a single line segment, splitting the polygonal chain into at most two new polygonal chains (at most one on each side of the removed line segment).

The removed line segment is then replaced by its bounding box.

Merge operators replace multiple boxes by their common bounding box.

Carefully applying the relaxation operators, we can explore a rich tradeoff between the EXACTLINE domain and the box domain.

Our analysis generalizes both: if we never apply any relaxation operators, the analysis reduces to EXACTLINE, and will be exact but potentially slow.

If we relax the initial line segment into its bounding box, the analysis reduces to box and be will be imprecise but fast.

Relaxation heuristic For our evaluation, we use the following relaxation heuristic, applied before each convolutional layer of the neural network.

The heuristic is parameterized by a relaxation percentage p ∈ [0, 1] and a clustering parameter k ∈ N. Each chain with t > 1000 nodes is traversed from one end to the other, and each line segment is turned into its bounding box, until the chain ends, the total number of nodes visited exceeds t/k or we find a line segment whose length is strictly above the p-th percentile, computed over all segment lengths in the chain prior to applying the heuristic.

All bounding boxes generated in one such step (from adjacent line segments) are then merged, the next segment (if any) is skipped, and the traversal is restarted on the remaining segments of the chain.

This way, each polygonal chain is split into some new polygonal chains and a number of new boxes.

The EXACTLINE domain can be extended such that it captures a single probability distribution on a polygonal chain.

For each line segment (P i , P i+1 ) on the polygonal chain (P 1 , . . .

, P k ) in R n , we additionally store a symbolic representation of a measure µ i on [0, 1], such that

where X ranges over measurable subsets of R n .

I.e., we have γ n (a) = {ν}. Whenever an abstract transformer splits a line segment, it additionally splits the corresponding measure, appropriately applying affine transformations, such that the new measures each range over [0, 1] again.

Note that if measures are uniform, it suffices to store µ i ([0, 1]) as the symbolic representation of µ i .

Our probabilistic abstract domain is the convex combination of the convex combination domains of this probabilistic EXACTLINE domain and the standard lifting of the box domain as a probabilistic abstract domain.

In practice, it is convenient to store an abstract element a with p probabilistic polygonal chains and q probabilistic boxes as

Its concretization is then given as

where

Our input always captures a uniform distribution on a line segment.

Relaxation and heuristic Our deterministic relaxations can be extended to work in the probabilistic setting.

When we replace a line segment by its bounding box, we use the total weight in its measure as the new entry in the weight vector λ corresponding to the box.

When we merge multiple boxes, their weights are added to give the weight for the the resulting box.

We then use the same relaxation heuristic as we described previously also in the probabilistic setting.

Computing bounds Given a probabilistic abstract element a as above, describing the output distribution of the neural network, we want to compute optimal bounds on the robustness probabilities P = {ν(Y) | ν ∈ γ n (a)}. The part of the distribution tracked by the probabilistic EXACTLINE domain has all its probability mass in perfectly determined locations, while the probability mass in each box can be located anywhere inside it.

We can compute bounds (l, u) = (min P, max P) = e + j∈L λ j , e + j∈U λ j , where

Here, we used the deterministic box concretization γ n .

We write APPROXLINE p k to denote our analysis (deterministic and probabilistic versions) where the relaxation heuristic uses relaxation percentage p and clustering parameter k. We implement APPROXLINE as in the DiffAI framework, taking advantage of the GPU parallelization provided by PyTorch (Paszke et al., 2017) .

Additionally, we use our implementation of APPROXLINE to compute exact results without approximation.

To get exact results, it suffices to set the relaxation percentage p to 0, in which case the clustering parameter k can be ignored.

Verification using APPROXLINE 0 k is equivalent to EXACTLINE up to floating point error.

To distinguish our GPU implementation from the original CPU implementation, we call our method EXACT instead of EXACTLINE.

EXACT is additionally capable of doing exact probabilistic inference.

We run on a machine with a GeForce GTX 1080 with 12 GB of GPU memory, and four processors with a total of 64 GB of RAM.

For generative specifications, we use decoders from autoencoders with either 32 or 64 latent dimensions trained in two different ways: VAE and CycleAE, described below.

We train them to reconstruct CelebA with image sizes 64 × 64.

We always use Adam Kingma & Ba (2014) with a learning rate of 0.0001 and a batch size of 100.

The specific network architectures are described in Appendix B. Our decoder always has 74128 neurons and the attribute detector has 24676 neurons.

VAE l is a variational autoencoder (Kingma & Welling, 2013) with l latent dimensions.

CycleAE l is a repurposed CycleGAN (Zhu et al., 2017) with l latent dimensions.

While these were originally designed for unsupervised style transfer between two data distributions, P and Q, we use it to build an autoencoder such that the generator behaves like a GAN and the encodings are distributed evenly among the latent space.

Specifically, we use a normal distribution in l dimensions for the embedding/latent space P with a small feed forward network D P as the latent space discriminator.

The distribution Q is the image distribution, and for its discriminator D Q we use the BEGAN method (Berthelot et al., 2017) , which determines an example's realism based on an autoencoder (also with l latent dimensions), which is trained to reproduce the ground-truth distribution Q and adaptively to fail to reproduce the GAN generator's distribution.

Attribute Detector is trained to recognize the 40 attributes provided by CelebA. Specifically, the attribute detector has a linear output.

We consider the attribute i to be detected as present in the input image if and only if the i-th component of the output of the attribute detector is strictly greater than 0.5.

The attribute detector is trained using Adam, minimizing the L1 loss between either 1 and the attribute (if it is present) or 0 and the attribute (if it is absent).

We train it for 300 epochs.

Given a generative model capable of producing interpolations between inputs which remain on the data manifold, there are many different verification goals one might pursue: E.g., check whether the generative model is correct with respect to a trusted classifier or whether a classifier is robust to interpretable interpolations between data points generated from a trusted generative model.

Even trusting neither the generator nor the classifier, we might want to verify that they are consistent.

We address all of these goals by efficiently computing the attribute consistency of a generative model with respect to an attribute detector: For a point picked uniformly at random between the encodings e 1 and e 2 of two ground truth inputs with matching attributes, we would like to determine the probability that its decoding will have the same attribute i.

We define the attribute consistency as where t is the ground truth for attribute i.

We will frequently omit the attribute detector n A and the decoder n D from C if it is clear from context which networks are being evaluated.

In this section, we demonstrate that probabilistic APPROXLINE is precise and efficient enough to provide useful bounds on the attribute consistency for interesting generative models and specifications on a reasonable dataset.

To this end we compare APPROXLINE to a variety of other methods which are also capable of providing probabilistic bounds.

We do this for CycleAE 32 trained for 200 epochs.

Specifically, suppose P is a set of unordered pairs {a, b} from the data set with a A,i > 0.5 ⇐⇒ b A,i > 0.5 for each of the k attributes i, where a A are ground truth attribute labels of a. Using each method, we find bounds on the true value of average attribute consistency asĈ

where n E is the encoding network.

Each method finds a probabilistic bound, [l, u] , such that l ≤Ĉ ≤ u. We call u − l its width.

We compare probabilistic APPROXLINE against two other probabilistic abstract domains, EXACT (=APPROXLINE 0 k ), and HZono lifted probabilistically.

Furthermore, we also compare against sampling with binomial confidence intervals on C using the ClopperPearson interval.

For probabilistic sampling, we take samples and recalculate the ClopperPearson interval with a confidence of 99.99% until the interval width is below 0.002 (chosen to be the same as our best result with APPROXLINE).

To avoid an incorrect calculation, we discard this interval and prior samples, and resample using the estimated number of samples.

Importantly, the probabilistic bound returned by the abstract domains is guaranteed to be correct 100% of the time, while for sampling it is only guaranteed to be correct 99.99% of the time.

For all methods, we set a timeout of 60s, and report the largest possible probabilistic bound if a timeout or out-of-memory error occurs.

For APPROXLINE, if an out-of-memory error occurs, we refine the hyperparameters using schedule A in Appendix C and restart (without resetting the timeout clock).

Figure 2 shows the results of running these on |P | = 100 pairs of matching celebrities with matching attribute labels, chosen uniformly at random from CelebA (each method uses the same P ).

The graph shows that while HZono is the fastest domain, it is unable to prove any specifications.

Sampling and EXACT do not appear to be significantly slower than APPROXLINE, but it can be observed that the average width of the probabilistic bounds they produce is large.

This is because Sampling frequently times out, and EXACT frequently exhausts GPU memory.

On the other hand, APPROXLINE provides an average probabilistic bound width of less than 0.002 in under 30s with perfect confidence (compared with the lower confidence provided by sampling).

Here, we demonstrate how to use our domain to check the attribute consistency of a model against an attribute detector.

We do this for two possible generative specifications: (i) generating rotated heads using flipped images, and (ii) adding previously absent attributes to faces.

For the results in this section, we use schedule B described in Appendix C.

Comparing models with turning heads It is known that VAEs are capable of generating images with intermediate poses of the subject from flipped images of the subject.

An example of this transformation is shown in Figure 3b .

Here, we show how one can use APPROXLINE to compare the effectiveness of different autoencoding models in performing this task.

To do this, we trained all 4 architectures described above for 20 epochs.

We then create a line specification over the encodings

, where a and Flipped(a) are the images shown in Figure 3b .

The width of the largest probabilistic bound was smaller than 3 × 10 −6 , so only the lower bounds are shown.

Less than 50 seconds were necessary to compute each bound, and the fastest computation was for CycleAE 64 at 30 seconds.

Lower Bound on Correctness of the flipped images shown in Figure 3b .

For a human face that is turned in one direction, ideally the different reconstructions will correspond to images of different orientations of the same face in 3D space.

As none of the CelebA attributes correspond to pose, the attribute detector should recognize the same set of attributes for all interpolations.

We used deterministic APPROXLINE

to demonstrate which attributes provably remain the correct for every possible interpolation (as visualized in Appendix E).

While we are able to show in the worst case, 32 out of 40 attributes are entirely robust to flipping, some attributes are not robust across interpolation.

Figure 4 demonstrates the results of using probabilistic APPROXLINE to find the average lower bound on the fraction of the input interpolation encodings which do result in the correct attribute appearing in the output image.

Verifying attribute independence Here, we demonstrate using APPROXLINE that attribute detection for one feature is invariant to a transformation in an independent feature.

Specifically, we verify for a single image the effect of adding a mustache.

This transformation is shown in Figure 3c .

To do this, we find the attribute vector m for "mustache" (i = 22 in CelebA) using the 80k training-set images in the manner described by Larsen et al. (2015) , and compute probabilistic bounds for C j (n E (o), n E (o) + 2m, o A,j ) for j = 22 and the image o. Using APPROXLINE we are able to prove that 30 out of the 40 attributes are entirely robust through the addition of a mustache.

Among the attributes which can be proven to be robust are i = 4 for "bald" and i = 39 for "young".

We are able to find that the attribute i = 24 for "NoBeard" is not entirely robust to the addition of the mustache vector.

We find a lower bound on the robustness probability for that attribute of 0.83522 and an upper bound of 0.83528.

In this paper we presented a highly scalable non-convex relaxation to verify neural network properties where inputs are restricted to a line segment.

Our results show that our method is faster and more precise than previous methods for the same networks, including sampling.

This speed and precision permitted us to verify properties based on interesting visual transformations induced by generative networks for the first time, including probabilistic properties.

For both models, we use the same encoders and decoders (even in the autoencoder descriminator from BEGAN), and always use the same attribute detectors.

Here we use Conv s C × W × H to denote a convolution which produces C channels, with a kernel width of W pixels and height of H, with a stride of s and padding of 1.

FC n is a fully connected layer which outputs n neurons.

ConvT s,p C × W × H is a transposed convolutional layer (Dumoulin & Visin, 2016 ) with a kernel width and height of W and H respectively and a stride of s and padding of 1 and out-padding of p, which produces C output channels.

• Latent Descriminator is a fully connected feed forward network with 5 hidden layers each of 100 dimensions.

• Encoder is a standard convolutional neural network: x → Conv 1 32 × 3 × 3 → ReLU → Conv 2 32 × 4 × 4 → ReLU → Conv 1 64 × 3 × 3 → ReLU → Conv 2 64 × 4 × 4 → ReLU → FC 512 → ReLU → FC 512 →

l.

• Decoder is a transposed convolutional network which has 74128 neurons: l → FC 400 → ReLU → FC 2048 → ReLU → ConvT 2,1 16 × 3 × 3 → ReLU →

ConvT 1,0 3 × 3 × 3 → x

• Attribute Detector has 24676 neurons: x →

Conv 2 16 × 4 × 4 → ReLU → Conv 2 32 × 4 × 4 → ReLU → FC 100 → 40.

While many refinement schemes start with an imprecise approximation and progressively tighten it, we observe that being only occasionally memory limited and rarely time limited, it conserves more time to start with the most precise approximation we have determined usually works, and progressively try less precise approximations as we determine that more precise ones can not fit into GPU memory.

Thus, we start searching for a probabilistic robustness bound with APPROXLINE Here we demonstrate how modifying the approximation parameters, p and N of APPROXLINE p N effect its speed and precision.

Figure 5 shows the result of varying these on x-axis.

The bottom number, N is the number of clusters that will be ideally made, and the top number p is the percentage of nodes which are permitted to be clustered.

Figure 6 : Blue means that the interpolative specification visualized in Figure 3b has been deterministically and entirely verified for the attribute (horizontal) using APPROXLINE

@highlight

We verify deterministic and probabilistic properties of neural networks using non-convex relaxations over visible transformations specified by generative models