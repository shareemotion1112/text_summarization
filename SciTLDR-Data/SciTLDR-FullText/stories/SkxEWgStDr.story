We present a simple proof for the benefit of depth in multi-layer feedforward network with rectifed activation (``"depth separation").

Specifically we present a sequence of classification problems f_i such that (a) for any fixed depth rectified network we can find an index m such that problems with index > m require exponential network width to fully represent the function f_m; and (b) for any problem f_m in the family, we present a concrete neural network with linear depth and bounded width that fully represents it.



While there are several previous work showing similar results, our proof uses substantially simpler tools and techniques, and should be accessible to undergraduate students in computer science and people with similar backgrounds.

We present a simple, geometric proof of the benefit of depth in deep neural networks.

We prove that there exist a set of functions indexed by m, each of which can be efficiently represented by a depth m rectified MLP network requiring O(m) parameters.

However, for any bounded depth rectified MLP network, there is a function f m in this set that representing it will require an exponential number of parameters in m.

More formally, let G d be the set of multi-layer perceptron (MLP) networks with rectified activation and d hidden layers, and let g Θ be such an MLP with parameters Θ. We will prove the following theorem: Theorem 1 (Depth Separation).

There exists a set of functions f 1 , f 2 , ..., f i : R 2 → {−1, 1} such that: While this is not a novel result, a main characteristic of our proof is its simplicity.

In contrast to previous work, our proof uses only basic algebra, geometry and simple combinatorial arguments.

As such, it can be easily read and understood by newcomers and practitioners, or taught in an undergraduate class, without requiring extensive background.

Tailoring to these crowds, our presentation style is more verbose then is usual in papers of this kind, attempting to spell out all steps explicitly.

We also opted to trade generality for proof simplicity, remaining in input space R 2 rather than the more general R n , thus allowing us to work with lines rather than hyperplanes.

Beyond being easy to visualize, it also results in simple proofs of the different lemmas.

The expressive power gained by depth in multi-layer perceptron (MLP) networks is relatively well studied, with multiple works showing that deep MLPs can represent functions that cannot be represented by similar but shallower networks, unless those have a significantly larger number of units (Delalleau & Bengio, 2011; Pascanu et al., 2013; Bianchini & Scarselli, 2014) .

Telgarsky (2015; show that network depth facilitate fast oscillations in the network response function.

Oscillations enabled by a linear growth in depth are shown to require exponential growth in the number of units when approximated well by a shallower network.

Eldan & Shamir (2016) study approximation to the unit sphere in a wide family of activation function.

In their construction they show that a 3-layer MLP could first compute the polynomial x 2 for each of the dimensions and use the last layer to threshold the sum of them to model the unit sphere indicator.

They analytically show that the same approximation with 2-layer network requires exponentially growth in width with precision.

Yarotsky (2017) ; Safran & Shamir (2016) show that depth is useful for approximating polynomials by ReLU MLPs.

Specifically, that f (x) = x 2 could be efficiently approximated with network depth.

While results similar to ours could be derived by a combination of the construction in Eldan & Shamir (2016) and the polynomial approximation of Yarotsky (2017) , we present a different (and to our taste, simpler) proof, using a geometrical interpretation and the number of response regions of ReLU networks, without explicitly modeling the x 2 polynomial.

The ReLU MLP decision space was studied by Pascanu et al. (2013) .

They show that the input space is sequentially refined by the ReLU and linear operations of the network to form separated convex polytopes in the input space.

They call these regions response regions.

They also establish a lower bound on the maximal number of regions, a bound which is tightened by Montufar et al. (2014); Raghu et al. (2017) ; Arora et al. (2016) ; Serra et al. (2017) .

We rely on the notion of response region in our proof, while attempting to provide an accessible explanation of it.

Some of the lemmas we present are simplified versions of results presented in these previous works.

3 BACKGROUND A 2d region is convex iff, for any two points in the region, all points on the line connecting the two points is also within the region.

A polygon with all internal angles < 180 o is a convex region.

A ReLU MLP with L layers parameterized by Θ is a multivariate function defined as the composition:

Where h A i s are parameterized affine transformations; Θ the set of parameters in them; and σ is the ReLU activation function: a non linear element-wise activation function defined by σ(x) = max{0, x}. We consider ReLU MLPs where all hidden layers have the same width w.

1 Without loss of generality we define the last layer of network, h out , as a weighted sum over its inputs where a sum strictly greater than zero is mapped to the 1 class, and otherwise to the −1 class.

The combination of linear operations and the ReLU function result in a piecewise linear function of the input X.

Piecewise linear activation functions such as ReLU split the input space into convex regions of linear activation.

This is asserted formally and visualized in Hanin & Rolnick (2019) .

The ReLU function has two regions ("pieces") of linearity x > 0, x ≤ 0.

Within each of these, linearity is maintained.

The sequential composition of affine transformations and the ReLU operations created by the MLP layers, divides the the input space into convex polytopes (in 2d these are convex polygons).

Within each such polytope, the function behaves linearly.

We call these polytopes linear response regions.

The number of these linear response regions, and specifically the effect of MLP depth on the maximal number of regions, was studied in multiple works Montufar et al. (2014) ; Raghu et al. (2017) ; Arora et al. (2016) ; Serra et al. (2017) .

We focus on the simpler case of 2-class classification ReLU MLP on the Euclidean plane and denote the maximal number of response regions of a network of d layers each with w units as r(w, d).

Our presentation of the proof of lemma 4 gives more insight into response regions.

Montufar et al. (2014) present the concept of folding transformation and their implementation with ReLUs.

Looking at one or more layers as a function f : R 2 → R 2 , a folding transformation maps a part of the input space to coincide with another.

Subsequent operations on the resulting space will apply to both parts, indifferently to their origin in their initial position.

As a simple example, consider a ReLU MLP of input dimension 1.

A simple folding two-layer transformation could easily model the function f (x) = |x|, mapping the negative input values to their positive counterparts.

Then, any composed operation in subsequent layers will apply to both the negative values and positive values.

This simple mechanism of "code reuse" is key to our constructed deep network and its unit-efficiency.

Let P m be a regular polygon with 2 m+1 edges (Figure 1 ).

Without loss of generality, P m is centered around the origin, bounded by the unit circle, and has a vertex at (0, 1).

2 The set of polygons P 1 , P 2 , ... approaches the unit circle as m → ∞. Let f m be the function with decision boundary

Points within polygon P m are of class 1, while other points are of class −1.

We begin with proving (a) of Theorem 1.

We will use the following lemmas, with proofs provided later.

Proof: A linear layer followed by a rectifier is piecewise-linear.

A composition of piecewise linear functions is itself piecewise linear.

Lemma 3.

Modeling f m as a piecewise linear function requires at least 2 m response regions.

Lemma 4.

Rectified MLP with input in R 2 , with d hidden layers and where each layer has width of at most w, has at most 2 2d log 2 w = 2 2d · 2 log 2 w response regions.

From lemma 4, it is clear that in order to achieve 2 m−2d = O(2 m ) response regions for a network with bounded depth d, we must grow w. To accommodate for the log factor, the growth in w must be exponential.

As function f m requires 2 m response regions (lemma 3), and m can be arbitrarily large, this proves (a).

We now turn to prove (b).

While lemma 4 tells us a network with 2 m response regions requires depth O(m), it does not guarantee such a network exists.

To prove (b), we construct such a network, with bounded width and linear depth.

The construction is based on folding transformations.

We manually construct the regular polygon decision boundary for polygon P m through exploitation of symmetry.

Intuitively, our construction resembles children paper-cutting, where a sheet of paper is folded multiple times, then cut with scissors.

Unfolding the paper reveals a complex pattern with distinctive symmetries.

Tracing and cutting the same pattern without any paper folding would require much more effort.

Analogously, we'll show how deep networks could implement "folds" through their layers and how ReLU operations, like scissor cuts, are mirrored through the symmetries induced by these folds.

Conversely, shallow networks, unable to "fold the paper", must make many more cuts -i.e.

must have much more units in order to create the very same pattern.

Formally, our deep network operates as follows: first, it folds across both the X and Y axes, mapping the input space into the first quadrant (x, y) → (|x|, |y|).

It now has to deal only with the positive part of the decision boundary.

It then proceeds in steps, in which it first rotates the space around the origin until the remaining decision boundary is symmetric around the X axis, and then folds around the X axis, resulting in half the previous decision boundary, in the first quadrant.

This process continues until the decision boundary is a single line, which can be trivially separated.

The first step cuts the number of edges in the decision boundary by a factor of four, while each subsequent rotate + fold sequence further cuts the number of polygon edges in half.

This process is depicted in figure 2 More formally, we require four types of transformations:

• f oldXY ( x 0 x 1 ) : R 2 → R 2 -initial mapping of input to the first quadrant.

• rotate Θ ( x 0 x 1 ) : R 2 → R 2 -clockwise rotation around the origin by an angle of Θ.

• f oldX( x 0 x 1 ) : R 2 → R 2 -folding across the X axis.

• top( x 0 x 1 ) : R 2 → R 1 -the final activation layer.

These operations are realized in the network layers, using a combination of linear matrix operations and ReLU activations.

The rotate operation is simply a rotation matrix.

Rotating by an angle of Θ is realized as:

The initial folding across both X and Y axes first transforms the input (x, y) to (x, −x, y, −y) using a linear transformation.

It then trims the negative values using a ReLU, and sums the first two and last two coordinates using another linear operation, resulting in:

Where σ is the elementwise ReLU activation function.

Folding across the X axes is similar, but as all x values are guaranteed to be positive, we do not need to consider −x.

Finally, the final classification layer is:

Composing these operations, the constructed network for problem f m has the form:

Note that the angle of rotation is decreased by a factor of 2 in every subsequent rotate.

The rotate and f oldX transformations pair, folds input space along a symmetry axis and effectively reduce the problem by half.

This results in a f oldXY operation followed by a sequence of m rotate • f oldX operations, followed by top.

Marking a f old operation as FσC and a rotate operation as R, where F, C, R being matrices, the MLP takes the form: FσCRFσCRFσCRF . . .

where a sequence CRF of matrix operations can be collapsed into a single matrix M .

This brings us to the familiar MLP form that alternates matrix multiplications and ReLU activations.

Overall, the network has m + 1 non-linear activations (from m f oldX operations and 1 f oldXY operation), resulting in m + 1 layers.

The response regions produced by the constructed MLP and by a shallow network are depicted in Figure 3 .

5.1 LEMMA 3

Modeling P m as a piecewise linear function requires at least 2 m response regions.

Proof: consider the polygon P m , and let M LP m be a ReLU MLP (piecewise-linear function) correctly classifying the problem.

Let V even be the set of every second vertex along a complete traversal of P m .

For each vertex take an step away from the origin to create V even (see Figure 4a for an illustration).

Each of the points in V even are strictly outside P m and therefore should be classified as class −1.

The response regions produced by M LP m are both convex and linear.

Let p i , p j by two arbitrary points in V even , p i = p j .

We will show that p i , p j belong in different response regions.

Assume by contradiction that p i , p j are in the same response region.

By convexity all points in a straight line between p i and p j are also in the same response region.

Also, by linearity these points have an activation value between p j and p j and therefore should also be classified as class −1.

From the problem construction we know that lines between the even vertices of P m cross the class boundary as demonstrated in Figure 4b .

Therefore, p i and p j must lay in different response regions.

Since p i and p j are arbitrary, M LP m 's number of response regions is at least |V even | = 2 m .

Rectified MLP with input in R 2 , with d hidden layers and where each layer has width of at most w, has at most 2 2d log 2 w response regions.

Proof: Raghu et al. (2017) prove a version of this lemma for input space R n , which have at most O(w nd ) = O(2 nd log 2 w ) response region.

We prove the more restricted case of inputs in R 2 , in a similar fashion.

We first consider the bound for 1 hidden-layer networks, then extend to d layers.

The first part of the proof follows classic and basic results in computational geometry.

The argument in the second part (move from 1 to d layers) is essentially the same one of Raghu et al. (2017) .

m then moving them slightly such that they are strictly outside P m .

Right: a chord a in green connecting any two vertices of V even , must cross P m .

Had both of the chord vertices been in the same response region, by convexity so do all points on a. By linearity, the final network activation of a's points will interpolate the activation of a's endpoints.

Number of regions in a line-arrangement of n lines We start by showing that the maximal number of regions in R 2 created by a line arrangement of n lines, denoted r(n), is r(n) ≤ n 2 .

This is based on classic result from computational geometry (Zaslavsky, 1975) .

Initially, the entire space is a region.

A single line divides the space in two, adding one additional region.

What happens as we add additional lines?

The second line intersects 3 with the first, and splits each of the previous regions in two, adding 2 more regions.

The third line intersects with both lines, dividing the line into three sections.

Each section splits a region, adding 3 more regions.

Continuing this way, the ith line intersects i − 1 lines, resulting in i sections, each intersecting a region and thus adding a region.

Figure 5 shows this for the 4th line.

We get: r(n) = 1 + 1 + 2 + 3 + 4 + . . .

+ n = 1 +

A 1 hidden-layer ReLU network is a line arrangement Consider a network of the form y = v(Ax + b) where the matrix A projects the input x to w dimensions, and the vector v combines them into a weighted sum.

The entire input space is linear under this network: the output is linear in the input.

4 When setting an ReLU activation function after the first layer: y = vσ(Ax + b) we get a 1-hidden layer ReLU network.

For a network with a width w hidden layer (A ∈ R w×2 ), we get w linear equations, A (i) x + b ( i) corresponding to w piecewise linear functions: each function has a section where it behaves according to its corresponding equation (the "active" section), and a section where it is 0 (the "rectified" section).

The input transitions between the active and the rectified sections of function i at the boundary given by

Thus, each ReLU neuron corresponds to a line that splits the input space into two: one input region where the neuron is active, and one where it is rectified.

Within each region, the behavior of the neuron is linear.

For a width w network, we have w such lines -a line arrangement of w lines.

The arrangement splits the space into at most r(w) < w 2 convex cells, where each cell corresponds to a set of active neurons.

Within each cell, the behavior of the input is linear.

Such a cell is called a linear region.

Additional Layers (Raghu et al., 2017; Pascanu et al., 2013) Additional layers further split the linear regions.

Consider the network after d − 1 layers, and a given linear region R. Within R, the : By iteratively introducing lines we can count the maximal number of regions created by k lines.

In general positions, the 4th introduced line (d. in greed) will intersect its 3 predecessor in 3 different points.

These will create 4 sections, each splitting a region into two (red-blue) hence adding 4 regions to the total count.

set of active neurons in layers < d − 1 is constant, and so within the region the next layer computes a linear function of the input.

As above, the ReLU activation then again gives w line equations, but this time these equations are only valid within R. The next layer than splits R into at most r(w) regions.

Raghu et al. (2017) Consider a network with two hidden layers of width w.

The first layer introduced at most r(w) ≤ w 2 convex regions.

As we saw above, for the second layer each region can be split again into at most r(w) regions, resulting in at most w 2 · w 2 = (w 2 ) 2 regions.

Applying this recursively, we get that the maximal number of regions in a depth d width w ReLU MLP network is r(w, d) = w 2d .

By writing w as 2 log2w we get the bound asserted in the lemma, concluding its proof.

@highlight

ReLU MLP depth seperation proof with gemoteric arguments

@highlight

A proof that deeper networks need less units than shallower ones for a family of problems. 