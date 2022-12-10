The resemblance between the methods used in studying quantum-many body physics and in machine learning has drawn considerable attention.

In particular, tensor networks (TNs) and deep learning architectures bear striking similarities to the extent that TNs can be used for machine learning.

Previous results used one-dimensional TNs in image recognition, showing limited scalability and a request of high bond dimension.

In this work, we train two-dimensional hierarchical TNs to solve image recognition problems, using a training algorithm derived from the multipartite entanglement renormalization ansatz (MERA).

This approach overcomes scalability issues and implies novel mathematical connections among quantum many-body physics, quantum information theory, and machine learning.

While keeping the TN unitary in the training phase, TN states can be defined, which optimally encodes each class of the images into a quantum many-body state.

We study the quantum features of the TN states, including quantum entanglement and fidelity.

We suggest these quantities could be novel properties that characterize the image classes, as well as the machine learning tasks.

Our work could be further applied to identifying possible quantum properties of certain artificial intelligence methods.

Over the past years, we have witnessed a booming progress in applying quantum theories and technologies to realistic problems.

Paradigmatic examples include quantum simulators BID31 and quantum computers (Steane, 1998; BID16 BID2 aimed at tackling challenging problems that are beyond the capability of classical digital computations.

The power of these methods stems from the properties quantum many-body systems.

Tensor networks (TNs) belong to the most powerful numerical tools for studying quantum manybody systems BID22 BID13 BID26 .

The main challenge lies in the exponential growth of the Hilbert space with the system size, making exact descriptions of such quantum states impossible even for systems as small as O(10 2 ) electrons.

To break the "exponential wall", TNs were suggested as an efficient ansatz that lowers the computational cost to a polynomial dependence on the system size.

Astonishing achievements have been made in studying, e.g. spins, bosons, fermions, anyons, gauge fields, and so on Cirac & Verstraete, 2009; BID23 BID26 BID26 .

TNs are also exploited to predict interactions that are used to design quantum simulators BID25 .As TNs allowed the numerical treatment of difficult physical systems by providing layers of abstraction, deep learning achieved similar striking advances in automated feature extraction and pattern recognition BID19 .

The resemblance between the two approaches is beyond superficial.

At theoretical level, there is a mapping between deep learning and the renormalization group BID1 , which in turn connects holography and deep learning BID37 BID10 , and also allows studying network design from the perspective of quantum entanglement BID20 .

In turn, neural networks can represent quantum states BID3 BID4 BID15 BID11 .Most recently, TNs have been applied to solve machine learning problems such as dimensionality reduction BID5 , handwriting recognition BID30 BID12 .

Through a feature mapping, an image described as classical information is transferred into a product state defined in a Hilbert space.

Then these states are acted onto a TN, giving an output vector that determines the classification of the images into a predefined number of classes.

Going further with this clue, it can be seen that when using a vector space for solving image recognition problems, one faces a similar "exponential wall" as in quantum many-body systems.

For recognizing an object in the real world, there exist infinite possibilities since the shapes and colors change, in principle, continuously.

An image or a gray-scale photo provides an approximation, where the total number of possibilities is lowered to 256 N per channel, with N describing the number of pixels, and it is assumed to be fixed for simplicity.

Similar to the applications in quantum physics, TNs show a promising way to lower such an exponentially large space to a polynomial one.

This work contributes in two aspects.

Firstly, we derive an efficient quantum-inspired learning algorithm based on a hierarchical representation that is known as tree TN (TTN) (see, e.g., BID21 ).

Compared with Refs.

BID30 BID12 where a onedimensional (1D) TN (called matrix product state (MPS) (Östlund & Rommer, 1995) ) is used, TTN suits more the two-dimensional (2D) nature of images.

The algorithm is inspired by the multipartite entanglement renormalization ansatz (MERA) approach BID35 BID36 BID7 BID9 , where the tensors in the TN are kept to be unitary during the training.

We test the algorithm on both the MNIST (handwriting recognition with binary images) and CIFAR (recognition of color images) databases and obtain accuracies comparable to the performance of convolutional neural networks.

More importantly, the TN states can then be defined that optimally encodes each class of images as a quantum many-body state, which is akin to the study of a duality between probabilistic graphical models and TNs BID27 .

We contrast the bond dimension and model complexity, with results indicating that a growing bond dimension overfits the data.

we study the representation in the different layers in the hierarchical TN with t-SNE ( BID32 , and find that the level of abstraction changes the same way as in a deep convolutional neural network BID18 or a deep belief network BID14 , and the highest level of the hierarchy allows for a clear separation of the classes.

Finally, we show that the fidelities between each two TN states from the two different image classes are low, and we calculate the entanglement entropy of each TN state, which gives an indication of the difficulty of each class.

A TN is defined as a group of tensors whose indexes are shared and contracted in a specific way.

TN can represent the partition function of a classical system, and also of a quantum many-body state which is mathematically a higher-dimensional vector.

For the latter, one famous example is the MPS that is written as DISPLAYFORM0 s N α N −1 .

An MPS can simply be understood as a d N -dimensional vector, with d the dimension of s i .

Though the space increases exponentially with N , the cost of an MPS increases only polynomially as N dD 2 (with D dimension of α n ).

When using it to describe an N − site physical state, the un-contracted open indexes {s n } are called physical bonds that represent the physical Hilbert space 1 , and contracted dummy indexes {α m } are called virtual bonds that carry the quantum entanglement.

MPS is essentially a 1D state representation.

When applied to 2D systems, MPS suffers severe restrictions since one has to choose a snake-like 1D path that covers the 2D manifold.

This issue is known in physics as the area law of entanglement entropy BID33 BID13 BID28 .A TTN FIG1 ) provides a natural expression for 2D states, which we can write as a hierarchical structure of K layers: DISPLAYFORM1 where N k is the number of tensors in the k-th layer.

To avoid the disaster brought by an extremely large number of indexes in a TN, we use the following symbolic and graphic conventions.

A tensor is denoted by a bold letter without indexes, e.g., T, whose elements are denoted by T α1α2··· .

Note a vector and a matrix are first-and second-order tensors with one and two indexes, respectively.

When two tensors are multiplied together, the common indexes are to be contracted.

One example is the inner product of two vectors, where DISPLAYFORM2 We take the transpose of v because we always assume the vectors to be column vectors.

Another example is the matrix product, where DISPLAYFORM3 αb2 is simplified to DISPLAYFORM4 .

α is an dummy index, and b 1 and b 2 are two open indexes.

In the graphic representation, a tensor is a block connecting to several bonds.

Each bond represents an index belonging to this tensor.

The dummy indexes are represented by the shared bonds that connect to two different blocks.

Following this convention, Eq. (1) can be simplified to DISPLAYFORM5 Similar to the MPS, a TTN also provides a representation of a d N -dimensional vector.

The cost is also polynomial to N .

One advantage is that the TTN bears a hierarchical structure and can be naturally built for 2D systems.

In a TTN, each local tensor is chosen to have one upward index and four downward indexes.

For representing a pure state, the tensor on the top only has four downward indexes.

All the indexes except the downward ones of the tensors in the first layer are dummy and will be contracted.

In our work, the TTN is slightly different from the pure state representation, by adding an upward index to the top tensor ( FIG1 ).

This added index corresponds to the labels in the supervised machine learning.

Before training, we need to prepare the data with a feature function that maps N scalars (N is the dimension of the images) to the tensor product of N normalized vectors.

The choice of the feature function is diversified: we chose the one used in Ref.

BID30 , where the dimension of each vector (d) can be controlled.

Then, the space is transformed from that of N scalars to a d N -dimensional vector (Hilbert) space.

After "vectorizing" the j-th image in the dataset, the output for classification is ad-dimensional vector obtained by contracting this huge vector with the TTN, which reads as DISPLAYFORM6 where {v [j,n] } denotes the n-th vector given by the j-th sample.

One can see thatd is the dimension of the upward index of the top tensor, and should equal to the number of the classes.

We use the convention that the position of the maximum value gives the classification of the image predicted by the TTN, akin to a softmax layer in a deep learning network.

One choice of the cost function to be minimized is the square error, which is defined as DISPLAYFORM7 where J is the number of training samples.

L [j] is ad-dimensional vector corresponding to the j-th label.

For example, if the j-th sample belongs to the p-th class, L [j] is defined as DISPLAYFORM8 3 MERA-INSPIRED TRAINING ALGORITHM Inspired by MERA BID35 , we derive a highly efficient training algorithm.

To proceed, let us rewrite the cost function in the following form DISPLAYFORM9 The third term comes from the normalization of L [j] , and we assume the second term is always real.

The dominant cost comes from the first term.

We borrow the idea from the MERA approach to reduce this cost.

Mathematically speaking, the central idea is to impose that Ψ is orthogonal, i.e., ΨΨ † = I. Then Ψ is optimized with Ψ † Ψ = I satisfied in the valid subspace that optimizes the classification.

By satisfying in the subspace, we do not require an identity from Ψ † Ψ, but mean DISPLAYFORM10 In MERA, a stronger constraint is used.

With the TTN, each tensor has one upward and four downward indexes, which gives a non-square orthogonal matrix by grouping the downward indexes into a large one.

Such tensors are called isometries and satisfy TT † = I after contracting all downwards indexes with its conjugate.

When all the tensors are isometries, the TTN gives a unitary transformation that satisfies ΨΨ † = I; it compresses a d N -dimensional space to ad-dimensional one.

In this way, the first terms becomes a constant, and we only need to deal with the second term.

The cost function becomes DISPLAYFORM11 Each term in f is simply the contraction of one TN, which can be efficiently computed.

We stress that independent of Eq. (3), Eq. (6) can be directly used as the cost function.

This will lead to a more interesting picture connected to the condensed matter physics and quantum information theory.

From the physical point of view, the central idea of MERA is the renormalization group (RG) of the entanglement BID35 .

The RG flows are implemented by the isometries that satisfy TT † = I. On one hand, the orthogonality makes the state remain normalized, a basic requirement of quantum states.

On the other hand, the renormalization group flows can be considered as the compressions of the Hilbert space (from the downward to upward indexes).

The orthogonality ensure that such compressions are unbiased with T † T I in the subspace.

The difference from the identity characterizes the errors caused by the compressions.

More discussions are given in Sec. 5.The tensors in the TTN are updated alternatively to minimize Eq. (6).

To update T [k,n] for instance, we assume other tensors are fixed and define the environment tensor E [k,n] , which is calculated by contracting everything in Eq. (6) after taking out T [k,n] FIG1 ) BID9 .

Then the cost function becomes f = −Tr(T [k,n] E [k,n] ).

Under the constraint that T [k,n] is an isometry, the solution of the optimal point is given by T [k,n] = VU † where V and U are calculated from the singular value decomposition E [k,n] = UΛV † .

At this point, we have f = − a Λ a .Then, the update of one tensor becomes the calculation of the environment tensor and its singular value decomposition.

In the alternating process for updating all the tensors, some tricks are used to accelerate the computations.

The idea is to save some intermediate results to avoid repetitive calculations by taking advantage of the tree structure.

Another important detail is to normalize the vector obtained each time by contracting four vectors with a tensor.

The strategy for building a multi-class classifier is the one-against-all classification scheme in machine learning.

For each class, we train one TTN so that it recognizes whether an image belongs to this class.

The output of Eq. (2) is a two-dimensional vector.

We fix the label for a yes answer as L yes = [1, 0].

For P classes, we will accordingly have P TTNs, denoted by {Ψ (p) }.

Then for recognizing an image (vectorized to {v[n] }), we define a P -dimensional vector F as DISPLAYFORM12 The position of its maximal element gives which class the image belongs to.

Algorithm 1 One-against-All Require: data : data points, n : the number of data points 1: for i = 0 → 9 do 2:Train binary classifier classif ier k corresponding to each handwritten digit 3: end for 4: for j = 1 → n do 5: DISPLAYFORM13

Our approach to classify image data begins by mapping each pixel x j to a d-component vector φ sj (x j ).

This feature map was introduced by BID30 ) and defined as DISPLAYFORM0 , where s j runs from 1 to d. By using a larger d, the TTN has the potential to approximate a richer class of functions.

DISPLAYFORM1 Figure 3: Embedding of data instances of CIFAR-10 by t-SNE corresponding to each layer in the TTN: (a) original data distribution and (b) the 1st, (c) 2nd, (d) 3rd, (e) 4th, and (f) 5th layer.

To verify the representation power of TTNs, we used the CIFAR-10 dataset BID17 ).

The dataset consists of 60,000 32 × 32 RGB images in 10 classes, with 6,000 instances per class.

There are 50,000 training images and 10,000 test images.

Each RGB image was originally 32 × 32 pixels: we transformed them to grayscale.

Working with gray-scale images reduced the complexity of training, with the trade-off being that less information was available for learning.

We built a TTN with five layers and used the MERA-like algorithm (Section 3) to train the model.

Specifically, we built a binary classification model to investigate key machine learning and quantum features, instead of constructing a complex multiclass model.

We found both the input bond (physical indexes) and the virtual bond (geometrical indexes) had a great impact on the representation power of TTNs, as showed in FIG3 .

This indicates that the limitation of representation power (learnability) of the TTNs is related to the input bond.

The same way, the virtual bond determine how accurately the TTNs approximate this limitation.

From the perspective of tensor algebra, the representation power of TTNs depends on the tensor contracted from the entire TTN.

Thus the limitation of this relies on the input bond.

Furthermore, the TTNs could be considered as a decomposition of this complete contraction, and the virtual bond determine how well the TTNs approximate this.

Moreover, this phenomenon could be interpreted from the perspective of quantum many-body theory: the higher entanglement in a quantum manybody system, the more representation power this quantum system has.

The sequence of convolutional and pooling layers in the feature extraction part of a deep learning network is known to arrive at higher and higher levels of abstractions that helps separating the classes in a discriminative learner BID19 .

This is often visualized by embedding the representation in two dimensions by t-SNE ( BID32 , and by coloring the instances according to their classes.

If the classes clearly separate in this embedding, the subsequent classifier will have an easy task performing classification at a high accuracy.

We plotted this embedding for each layer in the TN in Fig. 3 .

We observe the same pattern as in deep learning, having a clear separation in the highest level of abstraction.

To test the generalization of TTNs on a benchmark dataset, we used the MNIST collection, which is widely used in handwritten digit recognition.

The training set consists of 60,000 examples, and the Similar to the last experiment, we built a binary model to show the performance of generalization.

With the increase of bond dimension (both of the input bond and virtual bond), we found an apparent rise of training accuracy, which is consistent with the results in FIG3 .

At the same time, we observed the decline of testing accuracy.

The increase of bond dimension leads to a sharp increase of the number of parameters and, as a result, it will give rise to overfitting and lower the performance of generalization.

Therefore, one must pay attention to finding the optimal bond dimension -we can think of this as a hyperparameter controlling model complexity.

We choose the one-against-all strategy to build a 10-class model, which classify an input image by choosing the label for which the output is largest.

Considering the efficiency and avoiding overfitting, we use the minimal values of d TAB0 to reach the training accuracy around 95%.

Taking one trained TTN Ψ where the index for the labels is assumed to be P -dimensional, we can define P normalized TTN vector (state) as DISPLAYFORM0 In Φ [p] , the upward index of the top tensor is contracted with the label (L [p] ), giving a TN state that represents a normalized d N -dimensional vector (pure quantum state).The quantum state representations allow us to use quantum theories to study images and the related issues.

Let us begin with the cost function.

In Section 3, we started from a frequently used cost function in Eq. FORMULA7 , and derived a cost function in Eq. (6).

In the following, we show that Eq. (6) can be understood by the notion of fidelity.

With Eq. (8), the cost function in Eq. (6) can be rewritten as DISPLAYFORM1 The fidelity between two states (normalized vectors) is defined as their inner product, thus each term in the summation is simply the fidelity (Steane, 1998; BID0 between a vectorized image and the corresponding TTN state Φ [p] .

Considering that the fidelity measures the distance between two states, {Φ [p] } are the P states that minimize the distance between each Φ [p] and the p-th vectorized images.

In other words, the cost function is in fact the total fidelity, and Φ [p] is the quantum state (normalized vector) that optimally encodes the p-th class of images.

Note that due to the orthogonality, such P states are orthogonal to each other, i.e., Φ[p ] † Φ [p] = I p p .

This might trap us to a bad local minimum.

For this reason, we propose the one-against-all strategy (see Algorithm 3).

For each class, we have two TN states labeled yes and no, respectively, and in total 2P TN states.

{Φ [p] } are then defined by taking the P yes-labeled TN states.

The elements of F in Eq. FORMULA12 are defined by the summation of the fidelity between Φ[p] and the class of vectorized images.

In this scenario, the classification is decided by finding the Φ [p] that gives the maximal fidelity with the input image, while the orthogonal conditions among {Φ [p] } no longer exist.

Besides the algorithmic interpretation, fidelity may imply more intrinsic information.

Without the orthogonality of {Φ [p] }, the fidelity FIG1 ) describes the differences between the quantum states that encode different classes of images.

As shown in FIG4 , F p p remains quite small in most cases, indicating that the orthogonality still approximately holds.

Still, some results are still relatively large, e.g., F 4,9 = 0.1353.

We speculate this is closely related to the ways how the data are fed and processed in the TN.

In our case, two image classes that have similar shapes will result in a larger fidelity, because the TTN essentially provides a real-space renormalization flow.

In other words, the input vectors are still initially arranged and renormalized layer by layer according to their spatial locations in the image; each tensor renormalizes four nearest-neighboring vectors into one vector.

Fidelity can be potentially applied to building a network, where the nodes are classes of images and the weights of the connections are given by the F p p .

This might provide a mathematical model on how different classes of images are associated to each other.

We leave these questions for future investigations.

DISPLAYFORM2 Another important concept of quantum mechanics is (bipartite) entanglement, a quantum version of correlations BID0 .

It is one of the key characters that distinguishes the quantum states from classical ones.

Entanglement is usually given by a normalized positivedefined vector called entanglement spectrum (denoted as Λ), and is measured by the entanglement entropy S = − a Λ 2 a ln Λ 2 a .

Having two subsystems, entanglement entropy measures the amount of information of one subsystem that can be gained by measuring the other subsystem.

In the framework of TN, entanglement entropy determines the minimal dimensions of the dummy indexes needed for reaching a certain precision.

In our image recognition, entanglement entropy characterizes how much information of one part of the image we can gain by knowing the rest part of the image.

In other words, if we only know a part of an image and want to predict the rest according to the trained TTN (the quantum state that encodes the corresponding class), the entanglement entropy measures how accurately this can be done.

Here, an important analog is between knowing a part of the image and measuring the corresponding subsystem of the quantum state.

Thus, the trained TTN might be used on image processing, e.g., to recover an image from a damaged or compressed lower-resolution version.

FIG4 shows the entanglement entropy for each class in the MNIST dataset.

We computed two kinds of entanglement entropy marked by up-down and left-right.

The first one denotes the entanglement between Upper part of the images with the lower part one.

The later one denotes the entanglement between left part with the right part.

With the TTN, the entanglement spectrum is simply the singular values of the matrix M = L † T [K,1] with L the label and T [K,1] the top tensor ( FIG1 ).

This is because the all the tensors in the TTN are orthogonal.

Note that M has four indexes, of which each represents the effective space renormalized from one quarter of the vectorized image.

Thus, the bipartition of the entanglement determines how the four indexes of M are grouped into two bigger indexes before calculating the SVD.

We compute two kinds of entanglement entropy by cutting the system in the middle along the x or y direction.

Our results suggest that the images of "0" and "4" are the easiest and hardest, respectively, to predict one part of the image by knowing the other part.

We continued the forays into using tensor networks for machine learning, focusing on hierarchical, two-dimensional tree tensor networks that we found a natural fit for image recognition problems.

This proved a scalable approach that had a high precision, and we can conclude the following observations:• The limitation of representation power (learnability) of the TTNs model strongly depends on the input bond (physical indexes).

And, the virtual bond (geometrical indexes) determine how well the TTNs approximate this limitation.• A hierarchical tensor network exhibits the same increase level of abstraction as a deep convolutional neural network or a deep belief network.• Fidelity can give us an insight how difficult it is to tell two classes apart.• Entanglement entropy has potential to characterize the difficulty of representing a class of problems.

In future work, we plan to use fidelity-based training in an unsupervised setting and applying the trained TTN to recover damaged or compressed images and using entanglement entropy to characterize the accuracy.

@highlight

This approach overcomes scalability issues and implies novel mathematical connections among quantum many-body physics, quantum information theory, and machine learning.