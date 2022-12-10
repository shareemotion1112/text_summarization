Deep Convolution Neural Networks (CNNs), rooted by the pioneer work of  \cite{Hinton1986,LeCun1985,Alex2012}, and summarized in \cite{LeCunBengioHinton2015},   have been shown to be very useful in a variety of fields.

The  state-of-the art CNN machines such as image rest net \cite{He_2016_CVPR} are described by real value inputs and kernel convolutions followed by the local and  non-linear rectified linear outputs.

Understanding the role of these layers, the accuracy and limitations of them,  as well as making them more efficient (fewer parameters)  are all ongoing research questions.

Inspired in quantum theory, we propose the use of complex value kernel functions, followed by the local non-linear  absolute (modulus) operator square.

We argue that an advantage of quantum inspired complex kernels is robustness to realistic unpredictable scenarios (such as clutter noise,  data deformations).

We study a concrete problem of shape detection and show that when multiple overlapping shapes are deformed and/or clutter noise is added, a convolution layer with quantum inspired complex kernels outperforms the statistical/classical kernel counterpart and a "Bayesian shape estimator" .

The superior performance is due to the quantum phenomena of interference, not present in classical CNNs.

The convolution process in machine learning maybe summarized as follows.

Given an input f L−1 (x) ≥ 0 to a convolution layer L, it produces an output DISPLAYFORM0 From g L (y) a local and non-linear function is applied, f L (y) = f (g L (y)), e.g., f = ReLu (rectified linear units) or f = |.|, the magnitude operator.

This output is then the input to the next convolution layer (L+1) or simply the output of the whole process.

We can also write a discrete form of these convolutions, as it is implemented in computers.

We write g DISPLAYFORM1 , where the continuous variables y, x becomes the integers i, j respectively, the kernel function K(y − x) → w ij becomes the weights of the CNN and the integral over dx becomes the sum over j.

These kernels are learned from data so that an error (or optimization criteria) is minimized.

The kernels used today a real value functions.

We show how our understanding of the optimization criteria "dictate" the construction of the quantum inspired complex value kernel.

In order to concentrate and study our proposal of quantum inspired kernels, we simplify the problem as much as possible hoping to identify the crux of the limitation of current use of real value kernels.

We place known shapes in an image, at any location, and in the presence of deformation and clutter noise.

These shapes may have been learned by a CNN.

Our main focus is on the feedforward performance, when new inputs are presented.

Due to this focus, we are able to construct a Bayesian a posteriori probability model to the problem, which is based on real value prior and likelihood models, and compare it to the quantum inspired kernel method.

The main advantage of the quantum inspired method over existing methods is its high resistance to deviations from the model, such as data deformation, multiple objects (shapes) overlapping, clutter noise.

The main new factor is the quantum interference phenomenon BID1 BID0 , and we argue it is a desired phenomena for building convolution networks.

It can be carried out by developing complex value kernels driven by classic data driven optimization criteria.

Here we demonstrate its strength on a shape detection problem where we can compare it to state of the art classical convolution techniques.

We also can compare to the MAP estimator of the Bayesian model for the shape detection problem.

To be clear, we do not provide (yet) a recipe on how to build kernels for the full CNN framework for machine learning, and so the title of this paper reflects that.

Here, we plant a seed on the topic of building complex value kernels inspired in quantum theory, by demonstrating that for a given one layer problem of shape detection (where the classic data optimization criteria is well defined), we can build such complex value kernel and demonstrate the relevance of the interference phenomena.

To our knowledge such a demonstration is a new contribution to the field.

We also speculate on how this process can be generalized.

We are given an image I with some known objects to be detected.

The data is a set of N feature points, DISPLAYFORM0 Here we focus on 2-dimensional data, so D = 2.

An image I may be described by the set of feature points, as shown for example in figure 1.

Or, the feature points can be extracted from I, using for example, SIFT features, HOG features, or maximum of wavelet responses (which are convolutions with complex value kernels).

It maybe be the first two or so layers of a CNN trained in image recognition tasks.

The problem of object detection has been well addressed for example by the SSD machine BID7 , using convolution networks.

Here as we will demonstrate, given the points, we can construct a ONE layer CNN that solves the problem of shape detection and so we focus on this formulation.

It allows us to study in depth its performance (including an analytical study not just empirical).

8, 8) , radius 3, and with 100 points, is deformed as follows: for each point x i a random value drawn from a uniform distribution with range (−η i , η i ), η i = 0.05, is added along the radius.

(b) 1000 points of clutter are added by sampling from a uniform distribution inside a box of size 9 × 9, with corners at points: (4,4), (4,13), (13,4), (13,13).

We organize the paper as follows.

Section 2 presents a general description of shapes, which is easily adapted to any optimization method.

Section 3 presents the Bayesian method and the Hough transform method (and a convolution implementation) to the shape detection problem.

Section 4 lays out our main proposal of using quantum theory to address the shape detection problem.

The theory also leads naturally to a classical statistical method behaving like a voting scheme, and we establish a connection to Hough transforms.

Section 5 presents a theoretical and empirical analysis of the quantum method for shape detection and a comparison with the classical statistical method.

We demonstrate that for large deformations or clutter noise scenarios the quantum method outperforms the classical statistical method.

Section 6 concludes the paper.

A shape S may be defined by the set of points x satisfying S Θ (x) = 0, where Θ is a set of parameters describing S. Let µ be a shape's center (in our setting µ = (µ x , µ y )).

The choice of µ is in general arbitrary, though frequently there is a natural choice for µ for a given shape, such as its "center of mass": the average position of its coordinates.

We consider all the translations of S Θ (x) to represent the same shape, so that a shape is translation invariant.

It is then convenient to describe the shapes as S Θ (x − µ), with the parameters Θ not including the parameters of µ. Thus we describe a shape by the set of points X such that DISPLAYFORM0 The more complex a shape is, the larger is the set of parameters required to describe it.

For example, to describe a circle, we use three parameters {µ x , µ y , r} representing the center and the radius of the circle, i.e., DISPLAYFORM1 (see figure 1 a.) An ellipse can be described by DISPLAYFORM2 , where Θ = {Σ} is the covariance matrix, specified by three independent parameters.

We also require that a shape representation be such that if all the values of the parameters in Θ are 0, then the the set of points that belong to the shape "collapses" to just X = {µ}. This is the case for the parameterizations of the two examples above: the circle and the ellipse.

Energy Model:

Given a shape model we can create an energy model per data point x as DISPLAYFORM3 where the parameter p ≥ 0 defines the L p norm (after the sum over the points is taken and the 1/p root is applied).

The smaller E S Θ (x − µ), the more it is likely that the data point x belongs to the shape S Θ with center µ. In this paper, we set p = 1 because of its simplicity and robust properties.

To address realistic scenarios we must study the detection of shapes under deformations.

When deformations are present, the energy (1) is no longer zero for deformed points associated to the shape S Θ (x − µ).

Let each ideal shape data point x S i be deformed by adding η i to its coordinates, so DISPLAYFORM0 Deformations of a shape are only observed in the directions perpendicular to the shape tangents, i.e., along the direction of ∇ x S Θ (x − µ) xi , where ∇ x is the gradient operator.

For example, for a (deformed) circle shape, Θ = {r} and S r (x−µ) = 1− DISPLAYFORM1 , and so DISPLAYFORM2 r 2 ∝r i , wherer i is a unit vector pointing outwards in the radius direction at point DISPLAYFORM3

Given a set of data points X = {x 1 , x 2 , ..., x N } in R D originated from a shape S Θ (x i − µ) = 0.

We assume that each data point is independently deformed by η i (a random variable, since the direction is along the shape gradient), conditional on being a shape point.

Based on the energy model (1), for p = 1 (for simplicity and robust properties), we can write the likelihood model as DISPLAYFORM0 where C is a normalization constant and λ a constant that scale the errors/energies.

The product over all points is a consequence of the conditional independence of the deformations given the shape parameter (Θ, µ).

Assuming a prior distributions on the parameters to be uniform, we conclude that the a posteriori distribution is simply the likelihood model up to a normalization, i.e., DISPLAYFORM1 where Z is a normalization constant (does not depend on the parameters).

The parameters that maximize the likelihood L(Θ, µ) = log P(Θ, DISPLAYFORM2

A Hough transform cast binary votes from each data point.

The votes are for the shape parameter values that are consistent with the data point.

More precisely, each vote is given by DISPLAYFORM0 where u(x) is the Heaviside step function, u(x) = 1 if x ≥ 0 and zero otherwise, i.e., u = 1 if DISPLAYFORM1 α and u = 0 otherwise.

The parameter α clearly defines the error tolerance for a data point x i to belong to the shape S Θ (x − µ), the larger is α the smaller is the tolerance.

One can carry out this Hough transform for center detection as a convolution process.

More precisely, create a kernel, DISPLAYFORM2 | for x in a rectangular (or square) shape that includes all x for which u 1 α − |S Θ (x)| = 1.

The Hough transform for center detection is then the convolution of the kernel with the input image.

The result of the convolution at each location is the Hough vote for that location to be the center.

(b) The Bayesian method (showing the probability value).

The radius is fed to the method.

The method mixes all data yielding the highest probability in the wrong place.

increasing the parameter p can only improve a little as all data participate in the final estimation.

(c) The Hough method with α = 2.769 estimated to include all circle points that have been deformed.

The method is resistant to clutter noise.

When we have one circle with deformations (e.g., see FIG0 ), the Bayesian approach is just the "perfect" model.

Even if noise distributed uniformly across the image is added (e.g., see figure 1b), the Bayesian method will work very well.

However, as one adds clutter noise to the data (noise that is not uniform and "may correspond to clutter" in an image) as shown in figure 2, the Bayesian method mix all the data, has no mechanism to discard any data, and the Hough method outperforms the Bayesian one.

Even applying robust measures, decreasing p in the energy model, will have limited effect compared to the Hough method that can discard completely the data.

Consider another scenario of two overlapping and deformed circles, shown in FIG2 .

Again, the Bayesian approach does not capture the complexity of the data, two circles and not just one, and end up yielding the best one circle fit in the "middle", while the Hough method cope with this data by widening the center detection probabilities (bluring the center probabilities) and thus, including both true centers.

Still, the Hough method is not able to suggest that there are two circles/two peaks.

In summary, the Bayesian model is always the best one, as long as the data follows the exact model generation.

However, it is weak at dealing with *real world uncertainty* on the data (clutter data, multiple figures), other scenarios that occur often.

The Hough method, modeled after the same true positive event (shape detection) is more robust to these data variations and for the center detection problem can be carried out as a convolution.

The radius is fed to the method.

The method mixes all data yielding the highest probability approximately in the "middle" of both centers and no suggestion of two peaks/circles/centers exists.

(c) The Hough method with α = 2.769 estimated to include all circle points that have been deformed.

The method yields a probability that is more diluted and includes the correct centers, but does not suggest two peaks.

Quantum theory was developed for system of particles that evolve over time.

For us to utilize here the benefits of such a theory for the shape detection problem we invoke a hidden time parameter.

We refer to this time parameter as hidden since the input is only one static picture of a shape.

A hidden shape dynamics is not a new idea in computer vision, for example, scale space was proposed to describe shape evolution and allows for better shapes comparisons BID11 BID6 .

Hidden shape dynamics was also employed to describe a time evolution equation to produce shape-skeletons BID9 .

Since our optimization criteria per point is given by "the energy" of (1), we refer to classic concept of action, the one that is optimized to produce the optimal path, as DISPLAYFORM0 | where we are adopting for simplicity p = 1.

The idea is that a shapes evolve from the center µ = x(t = 0) to the shape point x = x(t = T ) in a time interval T .

During this evolutions all other parameters also evolve from Θ(t = 0) = 0 to Θ(t = T ) = Θ. The evolution is reversible, so we may say equivalently, the shape point x contracts to the center µ in the interval of time T .Following the path integral point of view of quantum theory BID1 , we consider the wave propagation to evolve by the integral over all path DISPLAYFORM1 where ψ Θ(t) (x(t)) is the probability amplitude that characterize the state of the shape, P T 0 is a path of shape contraction, from an initial state (x(0), Θ(0)) = (x, Θ) to a final state (x(T ), Θ(T )) = (µ, 0).

The integral is over all possible paths that initialize in (x(0), Θ(0)) = (x, Θ) and end in (x(T ), Θ(T )) = (µ, 0).

The Kernel K is of the form DISPLAYFORM2 where a new parameter, , is introduced.

It has the notation used in quantum mechanics for the reduced Planck's constant, but here it will have its own interpretation and value (see section 5.1.3).We now address the given image described by X = {x 1 , x 2 , ..., x N } ⊂ R 2 (e.g., see FIG0 ).

We consider an empirical estimation of ψ Θ (x) to be given by a set of impulses at the empirical data set DISPLAYFORM3 , where δ(x) is the Dirac delta function.

The normalization ensure the probability 1 when integrated everywhere.

Note that ψ Θ (x) is a pure state, a superposition of impulses.

Then, substituting this state into equation FORMULA18 , with the kernel provided by (4), yields the evolution of the probability amplitude DISPLAYFORM4 where C = e i T |S Θ (x−µ)| dµ. Thus shape points with deformations, x i , are interpreted as evidence of different quantum paths, not just the optimal classical path (which has no deformation).

Equation 5 is a convolution of the kernel K(x) = e i T |S Θ (x)| throughout the center candidates, except it is discretized at the locations where data is available.

According to quantum theory, the probability associated with this probability amplitude (a pure state) is given by P(Θ) = |ψ Θ (µ)| 2 , i.e., DISPLAYFORM5 which can also be expanded as DISPLAYFORM6 It is convenient to define the phase DISPLAYFORM7

Note the interference phenomenon arising from the cosine terms in the probability (6).

More precisely, a pair of data points that belongs to the shape will have a small magnitude difference, |φ ij | 1, and will produce a large cosine term, cos φ ij ≈ 1.

Two different data points that belong to the clutter will likely produce different phases, scaled inversely according to , so that small values of will create larger phase difference.

Pairs of clutter data points, not belonging to the shape, with large and varying phase differences, will produce quite different cosine terms, positive and/or negative ones.

If an image contains a large amount of clutter, the clutter points will end up canceling each other.

If an image contains little clutter, the clutter points will not contribute much.

This effect can be described by the following property for large numbers: if N 1 then DISPLAYFORM0 N , when each k is a random variable.

Figure 4 shows the performance of the quantum method on the same data as shown in FIG1 and FIG2 .

The accuracy of the detection of the centers and the identification of two centers shows how the quantum inspired method outperforms the classical counterparts.

In figure 4a , due to interference, clutter noise cancels out (negative terms on the probability equation 6 balance positive ones), and the center is peaked.

We do see effects of the noise inducing some fluctuation.

In figure 4b the two circle center peaks outperform both classical methods results as depicted in FIG2 .

A more thorough analysis is carried out in the next section to better understand and compare the performance of these different methods.

Note that even though the probability reflects a pair-wise computation as seen in FORMULA23 , we evaluate it by taking the magnitude square of the probability amplitude (given by equation FORMULA21 ), which is computed as a sum of N complex numbers.

Thus, the complexity of the computations is linear in the data set size.

After all, it is a convolution process.(a) Quantum Probability on figure 2a (b) Quantum Probability on figure 3aFigure 4: Quantum Probability depicted for input shown in FIG1 , respectively.

The parameters used where T = 1, = 0.12.

The quantum method outperform the classical methods, as the center detection shown in (a) is more peaked than in the Hough method and in (b) the two peaks emerge.

These are results of the interference phenomena, as cancellation of probabilities (negative terms on the probability equation 6) contribute to better resolve the center detection problem.

we derive a classical probability from the quantum probability amplitude via the Wick rotation Wick (1954) .

It is a mathematical technique frequently employed in physics, which transforms quantum physical systems into statistical physical systems and vice-versa.

It consists in replacing the term i T by a real parameter α in the probability amplitude.

Considering the probability amplitude equation FORMULA21 , the Wick rotation yields DISPLAYFORM0 We can interpret this as follows.

Each data point x i produces a vote v(Θ, µ|x i ) = e − α |S Θ (xi−µ)| , with values between 0 and 1.

The parameter α controls the weight decay of the vote.

Interestingly, this probability model resembles a Hough transform with each vote FORMULA26 being approximated by the binary vote described by (2)

We analyze the quantum method described by the probability (6), derived from the amplitude (5), and compare it with the classical statistical method described by (7) and its approximation (2).

This analysis and experiments is carried for a simple example, the circle.

We consider the circle shape, S r * (x − µ) = 1 − (x−µ) 2 (r * ) 2 of radius r * and its evaluation not only at the true center µ * but also at small displacements from it µ = µ * + δµ where δµ r * < 1 with δµ = |δµ|.

The points of an original circle are deformed to create the final "deformed" circle shape.

Each point is moved by a random vector η i pointing along the radius, i.e., η i = η ir * i withr * i being the unit vector along the radius.

Thus, we may write each point as DISPLAYFORM0 The deformation is assumed to vary independently and uniformly point wise.

Thus, η i ∈ (−η, η) and P(η i ) = 1 2η .

Plugging in the deformations and center displacement into the shape representation, S i = S r * (x i − µ), we get DISPLAYFORM1 DISPLAYFORM2 For the special case of the evaluation of the shape at the true center, δµ = 0 we obtain DISPLAYFORM3 The action for each path is given by |S Θ (x i − µ)| and we have multiple paths samples from the data.

Note that when we apply the quantum method, we interpret data derived from shape deformation as evidence of quantum trajectories (paths) that are not optimal while the classical interpretation for such data is a statistical error/deformation.

Both are different probabilistic interpretations that lead to different methods of evaluations of the optimal parameters as we analyze next.

We interpret the probability amplitude in equation (5), the sum over i = 1, . . .

, N , as a sum over many independent samples.

In general given a function f (|S|), then the sum over all points, DISPLAYFORM0 , can be interpreted as N C times the statistical average of the function f over the random variable S i .

In this case, the random variable S i (η i , δµ i ) represent two independent and uniform random variables, (η i , δµ i ), or (a i , b i ).Inserting shape equation FORMULA28 into the quantum probability amplitude of equation FORMULA21 DISPLAYFORM1 where I ab (e) = 1 4ab DISPLAYFORM2 2 −2aibi)| and at the true center we get DISPLAYFORM3 The ratio of the probabilities (magnitude square of the probability amplitudes) for the circle is then given by DISPLAYFORM4 These integrals can be evaluated numerically (or via integration of a Taylor series expansions, and then a numerical evaluation of the expansion).

Inserting shape equation FORMULA28 into the vote for the Hough transform giving by equation FORMULA14 result in DISPLAYFORM0 and interpreting the Hough total vote, V DISPLAYFORM1 as an average over a function of the random variable |S i | multiplied by the number of votes, we get DISPLAYFORM2 where DISPLAYFORM3 and at the true cen- DISPLAYFORM4 The ratio of the votes at the true center and the displaced center is then given by DISPLAYFORM5 We now address the choice of hyper-parameters of the models, namely for the quantum model and α for the classical counterpart so that the detection of the true center is as accurate as possible.

In this section, without loss of generality, we set T = 1.

In this way we can concentrate on understanding role (and not T ).

The amplitude probability given by equation FORMULA21 has a parameter where the inverse of scales up the magnitude of the shape values.

The smaller is , the more the phase ϕ i = 1 |S Θ (x i − µ)| reaches any point in the unit circle.

A large can make the phase ϕ i very small and a small can send each shape point to any point in the unit circle .The parameter large can help in aligning shape points to similar phases.

That suggests as large as possible.

At the same time, should help in misaligning pair of points where at least one of them does not belong to the shape.

That suggests small values of .

Similarly, if one is evaluating a shape with the "wrong" set of parameters, we woulud want the shape points to cancel each other.

In our example of the circle, we would like that shape points evaluated at a center displacement from the true center to yield some cancellation.

That suggests small.

One can explore the parameter that maximize the ratio Q C (a, b, ) given by equation (9).

We can also attempt to analytically balance both requests (high amplitude at the true center and low amplitude at the displaced center) by choosing values of such that ϕ i = 1 |S r * (x i − µ * )| ≤ π ∀i = 1, . . .

, N C .

More precisely, by choosing DISPLAYFORM0 Figure 5 suggests this choice of gives high ratios for Q(a, b, ).Now we discuss the estimation of α.

we note that for DISPLAYFORM1 2 all shape points will vote for the true center.

Thus, choosing α so that largest value of its inverse is DISPLAYFORM2 will guarantee all votes and give a lower vote for shape points evaluated from the displaced center.

One could search for higher values of α, smaller inverse values, so that reducing votes at the true center and expecting to reduce them further at the displaced center, i.e., to maximize H(a, b, α) in equation (10).

FIG4 suggests such changes do not improve the Hough transform performance.

FIG4 demonstrates that the quantum method outperforms the classical Hough transform on accuracy detection.

We can also perform a similar analysis adding noise, to obtain similar results.

This will require another two pages (a page of analysis and a page of graphs), and if the conference permits, we will be happy to add.

Deep Convolution Neural Networks (CNNs), rooted on the pioneer work of BID8 ; BID4 ; BID3 , and summarized in BID5 , have been shown to be very useful in a variety of fields.

Inspired in quantum theory, we investigated the use of complex value kernel functions, followed by the local non-linear absolute (modulus) operator square.

We studied a concrete problem of .

For each of the figures 5a, 5b,5c we vary we vary b = 1 2 a, a, 2a (or center displacements δµ = 0.25, 0.5, 1), respectively.

These figures depict ratios Q(a, b, ) × (blue) for ∈ (0.047, 0.2802) and H(a, b, α) × ← − α (red) for ← − α ∈ (22.727, 2.769) (The reverse arrow implies the x-axis start at the maximum value and decreases thereafter).

All plots have 200 points, with uniform steps in their respective range.

Note that our proposed parameter value is = 0.1401, the solution to equation FORMULA42 , and indeed gives a high ratio.

Also, α = 2.769 is the smallest value to yield all Hough votes in the center.

Clearly the quantum ratio outperforms the best classical Hough method, which does not vary much across α values.

As the center displacement increases, the quantum method probability, for = 0.1401, decreases much faster than the Hough method probability.

Final figure 5d display values of |ψ| 2 (µ * ) × (at the true center) in blue, for ∈ (0.047, 0.2802), with 200 uniform steps.

In red, V (µ * ) × ← − α for ← − α ∈ (22.727, 2.769), with 200 uniform steps.

DISPLAYFORM0 shape detection and showed that when multiple overlapping shapes are deformed and/or clutter noise is added, a convolution layer with quantum inspired complex kernels outperforms the statistical/classical kernel counterpart and a "Bayesian shape estimator".

It is worth to mention that the Bayesian shape estimator is the best method as long as the data satisfy the model assumptions.

Once we add multiple shapes, or add clutter noise (not uniform noise), the Bayesian method breaks down rather easily, but not the quantum method nor the statistical version of it (the Hough method being an approximation to it).

An analysis comparing the Quantum method to the Hough method was carried out to demonstrate the superior accuracy performance of the quantum method, due to the quantum phenomena of interference, not present in the classical CNN.We have not focused on the problem of learning the shapes here.

Given the proposed quantum kernel method, the standard techniques of gradient descent method should also work to learn the kernels, since complex value kernels are also continuous and differentiable.

Each layer of the networks carries twice as many parameters, since complex numbers are a compact notation for two numbers, but the trust of the work is to suggest that they may perform better and reduce the size of the entire network.

These are just speculations and more investigation of the details that entice such a construction are needed.

Note that many articles in the past have mentioned "quantum" and "neural networks" together.

Several of them use Schrödinger equation, a quantum physics modeling of the world.

Here in no point we visited a concept in physics (forces, energies), as Schrödinger equation would imply, the only model is the one of shapes (computer vision model).

Quantum theory is here used as an alternative statistical method, a purely mathematical construction that can be applied to different models and fields, as long as it brings benefits.

Also, in our search, we did not find an article that explores the phenomena of interference and demonstrate its advantage in neural networks.

The task of brining quantum ideas to this field must require demonstrations of its utility, and we think we did that here.

<|TLDR|>

@highlight

A quantum inspired kernel for convolution network, exhibiting interference phenomena,  can be very useful (and compared it with real value  counterpart).