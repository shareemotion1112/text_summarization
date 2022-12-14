We present the first verification that a neural network for perception tasks produces a correct output within a specified tolerance for every input of interest.

We define correctness relative to a specification which identifies 1) a state space consisting of all relevant states of the world and 2) an observation process that produces neural network inputs from the states of the world.

Tiling the state and input spaces with a finite number of tiles, obtaining ground truth bounds from the state tiles and network output bounds from the input tiles, then comparing the ground truth and network output bounds delivers an upper bound on the network output error for any input of interest.

Results from two case studies highlight the ability of our technique to deliver tight error bounds for all inputs of interest and show how the error bounds vary over the state and input spaces.

Neural networks are now recognized as powerful function approximators with impressive performance across a wide range of applications, especially perception tasks (e.g. vision, speech recognition).

Current techniques, however, provide no correctness guarantees on such neural perception systemsthere is currently no way to verify that a neural network provides correct outputs (within a specified tolerance) for all inputs of interest.

The closest the field has come is robustness verification, which aims to verify if the network prediction is stable for all inputs in some neighborhood around a selected input point.

But robustness verification does not verify for all inputs of interest -it only verifies around local regions.

Besides, it does not guarantee that the output, even if stable, is actually correct -there is no specification that defines the correct output for any input except for the manually-labeled center point of each region.

We present the first correctness verification of neural networks for perception -the first verification that a neural network produces a correct output within a specified tolerance for every input of interest.

Neural networks are often used to predict some property of the world given an observation such as an image or audio recording.

We therefore define correctness relative to a specification which identifies 1) a state space consisting of all relevant states of the world and 2) an observation process that produces neural network inputs from the states of the world.

Then the inputs of interest are all inputs that can be observed from the state space via the observation process.

We define the set of inputs of interest as the feasible input space.

Because the quantity of interest that the network predicts is some property of the state of the world, the state defines the ground truth output (and therefore defines the correct output for each input to the neural network).

We present Tiler, the algorithm for correctness verification of neural networks.

Evaluating the correctness of the network on a single state is straightforward -use the observation process to obtain the possible inputs for that state, use the neural network to obtain the possible outputs, then compare the outputs to the ground truth from the state.

To do correctness verification, we generalize this idea to work with tiled state and input spaces.

We cover the state and input spaces with a finite number of tiles: each state tile comprises a set of states; each input tile is the image of the corresponding state tile under the observation process.

The state tiles provide ground truth bounds for the corresponding input tiles.

We use recently developed techniques from the robustness verification literature to obtain network output bounds for each input tile (Xiang et al., 2018; Gehr et al., 2018; Weng et al., 2018; Bastani et al., 2016; Lomuscio and Maganti, 2017; Tjeng et al., 2019) .

A comparison of the ground truth and output bounds delivers an error upper bound for that region of the state space.

The error bounds for all the tiles jointly provide the correctness verification result.

We present two case studies.

The first involves a world with a (idealized) fixed road and a camera that can vary its horizontal offset and viewing angle with respect to the centerline of the road (Section 5).

The state of the world is therefore characterized by the offset ?? and the viewing angle ??.

A neural network takes the camera image as input and predicts the offset and the viewing angle.

The state space includes the ?? and ?? of interest.

The observation process is the camera imaging process, which maps camera positions to images.

This state space and the camera imaging process provide the specification.

The feasible input space is the set of camera images that can be observed from all camera positions of interest.

For each image, the camera positions of all the states that can produce the image give the possible ground truths.

We tile the state space using a grid on (??, ??).

Each state tile gives a bound on the ground truth of ?? and ??.

We then apply the observation process to project each state tile into the image space.

We compute a bounding box for each input tile and apply techniques from robustness verification (Tjeng et al., 2019) to obtain neural network output bounds for each input tile.

Comparing the ground truth bounds and the network output bounds gives upper bounds on network prediction error for each tile.

We verify that our trained neural network provides good accuracy across the majority of the state space of interest and bound the maximum error the network will ever produce on any feasible input.

The second case study verifies a neural network that classifies a LiDAR measurement of a sign in an (idealized) scene into one of three shapes (Section 6).

The state space includes the position of the LiDAR sensor and the shape of the sign.

We tile the state space, project each tile into the input space via the LiDAR observation process, and again apply techniques from robustness verification to verify the network, including identifying regions of the input space where the network may deliver an incorrect classification.

Specification: We show how to use state spaces and observation processes to specify the global correctness of neural networks for perception (the space of all inputs of interest, and the correct output for each input).

This is the first systematic approach (to our knowledge) to give global correctness specification for perception neural networks.

We present an algorithm, Tiler, for correctness verification.

With state spaces and observation processes providing specification, this is the first algorithm (to our knowledge) for verifying that a neural network produces the correct output (up to a specified tolerance) for every input of interest.

The algorithm can also compute tighter correctness bounds for focused regions of the state and input spaces.

Case Study: We apply this framework to a problem of predicting camera position from image and a problem of classifying shape of the sign from LiDAR measurement.

We obtain the first correctness verification of neural networks for perception tasks.

Motivated by the vulnerability of neural networks to adversarial attacks (Papernot et al., 2016; Szegedy et al., 2014) , researchers have developed a range of techniques for verifying robustnessthey aim to verify if the neural network prediction is stable in some neighborhood around a selected input point.

Salman et al. (2019) ; Liu et al. (2019) provide an overview of the field.

A range of approaches have been explored, including layer-by-layer reachability analysis (Xiang et al., 2017; with abstract interpretation (Gehr et al., 2018) or bounding the local Lipschitz constant (Weng et al., 2018) , formulating the network as constraints and solving the resulting optimization problem (Bastani et al., 2016; Lomuscio and Maganti, 2017; Cheng et al., 2017; Tjeng et al., 2019) , solving the dual problem (Dvijotham et al., 2018; Raghunathan et al., 2018) , and formulating and solving using SMT/SAT solvers (Katz et al., 2017; Ehlers, 2017; Huang et al., 2017) .

When the adversarial region is large, several techniques divide the domain into smaller subdomains and verify each of them (Singh et al., 2019; Bunel et al., 2017) .

Unlike the research presented in this paper, none of this prior research formalizes or attempts to verify that a neural network for perception computes correct (instead of stable) outputs within a specified tolerance for all inputs of interest (instead of local regions around labelled points).

Prior work on neural network testing focuses on constructing better test cases to expose problematic network behaviors.

Researchers have developed approaches to build test cases that improve coverage on possible states of the neural network, for example neuron coverage (Pei et al., 2017; Tian et al., 2018) and generalizations to multi-granular coverage (Ma et al., 2018) and MC/DC (Kelly J. et al., 2001 ) inspired coverage .

Odena and Goodfellow (2018) presents coverage-guided fuzzing methods for testing neural networks using the above coverage criteria.

Tian et al. (2018) generates realistic test cases by applying natural transformations (e.g. brightness change, rotation, add rain) to seed images.

O 'Kelly et al. (2018) uses simulation to test autonomous driving systems with deep learning based perception.

Unlike this prior research, which tests the neural network on only a set of input points, the research presented in this paper verifies correctness for all inputs of interest.

Consider the general perception problem of taking an input observation x and trying to predict some quantity of interest y.

It can be a regression problem (continuous y) or a classification problem (discrete y).

Some neural network model is trained for this task.

We denote its function by f : X ??? Y, where X is the space of all possible inputs to the neural network and Y is the space of all possible outputs.

Behind the input observation x there is some state of the world s. Denote S as the space of all states of the world that the network is expected to work in.

For each state of the world, a set of possible inputs can be observed.

We denote this observation process using a mapping g : S ??? P(X ), where g(s) is the set of inputs that can be observed from s. Here P(??) is the power set, andX ??? X is the feasible input space, the part of input space that may be observed from the state space S. Concretely,X = {x|???s ??? S, x ??? g(s)}.

The quantity of interest y is some attribute of the state of the world.

We denote the ground truth of y using a function ?? : S ??? Y. This specifies the ground truth for each input, which we denote as a mappingf :X ??? P(Y).f (x) is the set of possible ground truth values of y for a given x:

f (x) = {y|???s ??? S, y = ??(s), x ??? g(s)}.

(1)

The feasible input spaceX and the ground truth mappingf together form a specification.

In general, we cannot compute and representX andf directly -indeed, the purpose of the neural network is to compute an approximation to this ground truthf which is otherwise not available given only the input x.

X andf are instead determined implicitly by S, g, and ??.

The error of the neural network is then characterized by the difference between f andf .

Concretely, the maximum possible error at a given input x ???X is:

where d(??, ??) is some measurement on the size of the error between two values of the quantity of interest.

For regression, we consider the absolute value of the difference d(y 1 , y 2 ) = |y 1 ??? y 2 |.

1 For classification, we consider a binary error measurement d(y 1 , y 2 ) = 1 y1 =y2 (indicator function), i.e.

the error is 0 if the prediction is correct, 1 if the prediction is incorrect.

The goal of correctness verification is to compute upper bounds on network prediction errors with respect to the specification.

We formulate the problem of correctness verification formally here:

Problem formulation of correctness verification: Given a trained neural network f and a specification (X ,f ) determined implicitly by S, g, and ??, compute upper bounds on error e(x) for any feasible input x ???X .

We next present Tiler, an algorithm for correctness verification of neural networks.

We present here the algorithm for regression settings, with sufficient conditions for the resulting error bounds to be sound.

The algorithm for classification settings is similar (see Appendix B).

Step 1: Divide the state space S into state tiles

The image of each S i under g gives an input tile (a tile in the input space):

The resulting tiles {X i } satisfy the following condition:

Step 2: For each S i , compute the ground truth bound as an interval

The bounds computed this way satisfy the following condition, which (intuitively) states that the possible ground truth values for an input point must be covered jointly by the ground truth bounds of all the input tiles that contain this point:

Previous research has produced a variety of methods that bound the neural network output over given input region.

Examples include layer-by-layer reachability analysis (Xiang et al., 2018; Gehr et al., 2018; Weng et al., 2018) and formulating constrained optimization problems (Bastani et al., 2016; Lomuscio and Maganti, 2017; Tjeng et al., 2019) .

Each method typically works for certain classes of networks (e.g. piece-wise linear networks) and certain classes of input regions (e.g. polytopes).

For each input tile X i , we therefore introduce a bounding box B i that 1) includes X i and 2) is supported by the solving method:

Step 3: Using S i and g, compute a bounding box B i for each tile

The bounding boxes B i 's must satisfy the following condition:

Step 4: Given f and bounding boxes {B i }, use an appropriate solver to solve for the network output ranges

The neural network has a single output entry for each quantity of interest.

Denote the value of the output entry as o(x), f (x) = o(x).

The network output bounds (l i , u i ) returned by the solver must satisfy the following condition:

Step 5:

For each tile, use the ground truth bound (l i , u i ) and network output bound (l i , u i ) to compute the error bound e i :

e i gives the upper bound on prediction error when the state of the world s is in S i .

This is because (l i , u i ) covers the ground truth values in S i , and (l i , u i ) covers the possible network outputs for all inputs that can be generated from S i .

From these error bounds {e i }, we compute a global error bound:

We can also compute a local error bound for any feasible input x ???X :

Note that max {i|x???Xi} e i provides a tighter local error bound.

But since it is generally much easier to check containment of x in B i 's than in X i 's, we adopt the current formulation.

Algorithm 1 formally presents the Tiler algorithm (for regression).

The implementations of DI-VIDESTATESPACE, GETGROUNDTRUTHBOUND, and GETBOUNDINGBOX are problem dependent.

The choice of SOLVER needs to be compatible with B i and f .

Conditions 4.1 to 4.4 specify the sufficient conditions for the returned results from these four methods such that the guarantees obtained are sound.

Algorithm 1 Tiler (for regression)

Step 1 3:

for each S i do 4:

Step 2 5:

Step 3 6:

Step 4 7:

Step 5 8:

end for

e global ??? max({e i })

Step 5 10:

return e global , {e i }, {B i } {

e i }, {B i } can be used later to compute e local (x) 11: end procedure

The complexity of this algorithm is determined by the number of tiles, which scales with the dimension of the state space S. Because the computations for each tile are independent, our Tiler implementation executes these computations in parallel.

Our formulation also applies to the case of noisy observations.

Notice that the observation process g maps from a state to a set of possible inputs, so noise can be incorporated here.

The above version of Tiler produces hard guarantees, i.e. the error bounds computed are valid for all cases.

This works for observations with bounded noise.

For cases where noise is unbounded (e.g. Gaussian noise), Tiler can be adjusted to provide probabilistic guarantees: we compute bounding boxes B i such that P (x ??? B i |x ??? g(s), s ??? S i ) > 1 ??? for some small .

Here we also need the probability measure associated with the observation process -g(s) now gives the probability distribution of input x given state s.

This will give an error bound that holds with probability at least 1 ??? for any state in this tile.

We demonstrate how this is achieved in practice in the second case study (Section 6).

Tiler provides a way to verify the correctness of the neural network over the whole feasible input spaceX .

To make the system complete, we need a method to detect whether a new observed input is withinX (the network is designed to work for it, and we have guaranteed correctness) or not (the network is not designed for it, so we don't have guarantees).

In general, checking containment directly withX is hard, since there is no explicit representation of it.

Instead, we use the bounding boxes {B i } returned by Tiler as a proxy forX : for a new input x * , we check if x * is contained in any of the B i '

s. Since the network output ranges computed in Step 4 covers the inputs in each B i , and the error bounds incorporate the network output ranges, we know that the network output will not have unexpected drastic changes in B i '

s.

This makes B i 's a good proxy for the space of legal inputs.

Searching through all the B i 's can introduce a large overhead.

We propose a way to speed up the search by utilizing the network prediction and the verified error bounds.

Given the network prediction y * = f (x * ) and the global error bound e global , we can prune the search space by discarding tiles that do not overlap with [y * ??? e global , y * + e global ] in the ground truth attribute.

The idea is that we only need to search the local region in the state space that has ground truth attribute close to the prediction, since we have verified the bound for the maximum prediction error.

We demonstrate this detection method and the prediction-guided search in the first case study (Section 5).

Problem set-up:

Consider a world containing a road with a centerline, two side lines, and a camera taking images of the road.

The camera is positioned at a fixed height above the road, but can vary its horizontal offset and viewing angle with respect to the centerline of the road.

Figure 1a presents a schematic of the scene.

The state of the world s is characterized by the offset ?? and angle ?? of the camera position.

We therefore label the states as s ??,?? .

We consider the camera position between the range ?? ??? [???40, 40] (length unit of the scene, the road width from the centerline to the side lines is 50 units) and ?? ??? [???60

The input x to the neural network is the image taken by the camera.

The observation process g is the camera imaging process.

For each pixel, we shoot a ray from the center of that pixel through the camera focal point and compute the intersection of the ray with objects in the scene.

The intensity of that intersection point is taken as the intensity of the pixel.

The resulting x's are 32??32 gray scale images with intensities in [0, 255] (see Appendix C.1 and C.2 for detailed descriptions of the scene and the camera imaging process).

Figure 1b presents an example image.

The feasible input spaceX is the set of all images that can be taken with ?? ??? [???40, 40] and ?? ??? [???60

The quantity of interest y is the camera position (??, ??).

The ground truth function ?? is simply ??(s ??,?? ) = (??, ??).

For the neural network, we use the same ConvNet architecture as CNN A in Tjeng et al. (2019) and the small network in .

It has 2 convolutional layers (size 4??4, stride 2) with 16 and 32 filters respectively, followed by a fully connected layer with 100 units.

All the activation functions are ReLUs.

The output layer is a linear layer with 2 output nodes, corresponding to the predictions of ?? and ??.

The network is trained on 130k images and validated on 1000 images generated from our imaging process.

The camera positions of the training and validation images are sampled uniformly from the range ?? ??? [???50, 50] and ?? ??? [???70

The network is trained with an l 1 -loss function, using Adam (Kingma and Ba, 2014) (see Appendix E for more training details).

For error analysis, we treat the predictions of ?? and ?? separately.

The goal is to find upper bounds on the prediction errors e ?? (x) and e ?? (x) for any feasible input x ???X .

Tiler Figure 1c presents a schematic of how we apply Tiler to this problem.

Tiles are constructed by dividing S on (??, ??) into a grid of equal-sized rectangles with length a and width b. Each cell in the grid is then We next encapsulate each tile X i with an l ??? -norm ball B i by computing, for each pixel, the range of possible values it can take within the tile.

As the camera position varies in a cell S i , the intersection point between the ray from the pixel and the scene sweeps over a region in the scene.

The range of intensity values in that region determines the range of values for that pixel.

We compute this region for each pixel, then find the pixel value range (see Appendix C.3).

The resulting B i is an l ??? -norm ball in the image space covering X i , represented by 32??32 pixel-wise ranges.

To solve the range of outputs of the ConvNet for inputs in the l ??? -norm ball, we adopt the approach from (Tjeng et al., 2019) .

They formulate the robustness verification problem as mixed integer linear program (MILP).

Presolving on ReLU stability and progressive bound tightening are used to improve efficiency.

We adopt the same formulation but change the MILP objectives.

For each l ??? -norm ball, we solve 4 optimization problems: maximizing and minimizing the output entry for ??, and another two for ??.

Denote the objectives solved as ?? Experimental results We run Tiler with a cell size of 0.1 (the side length of each cell in the (??, ??) grid is a = b = 0.1).

The step that takes the majority of time is the optimization solver.

With parallelism, the optimization step takes about 15 hours running on 40 CPUs@3.00 GHz, solving 960000 ?? 4 MILP problems.

We compute global error bounds by taking the maximum of e i ?? and e i ?? over all tiles.

The global error bound for ?? is 12.66, which is 15.8% of the measurement range (80 length units for ??); for ?? is 7.13

??? (5.94% of the 120 ??? measurement range).

We therefore successfully verify the correctness of the network with these tolerances for all feasible inputs.

We present the visualizations of the error bound landscape by plotting the error bounds of each tile as heatmaps over the (??, ??) space.

Figures 2a and 2d present the resulting heatmaps for e i ?? and e i ?? , respectively.

To further inspect the distribution of the error bounds, we compute the percentage of the state space S (measured on the (??, ??) grid) that has error bounds below some threshold value.

The percentage varying with threshold value can be viewed as a cumulative distribution.

Figures 2c and 2f present the cumulative distributions of the error bounds.

It can be seen that most of the state space can be guaranteed with much lower error bounds, with only a small percentage of the regions having larger guarantees.

This is especially the case for the offset measurement: 99% of the state space is guaranteed to have error less than 2.65 (3.3% of the measurement range), while the global error bound is 12.66 (15.8%).

A key question is how well the error bounds reflect the actual maximum error made by the neural network.

To study the tightness of the error bounds, we compute empirical estimates of the maximum errors for each S i , denoted as?? i ?? and?? i ?? .

We sample multiple (??, ??) within each cell S i , generate input images for each (??, ??), then take the maximum over the errors of these points as the empirical estimate of the maximum error for S i .

The sample points are drawn on a sub-grid within each cell, with sampling spacing 0.05.

This estimate is a lower bound on the maximum error for S i , providing a reference for evaluating the tightness of the error upper bounds we get from Tiler.

We take the maximum of?? i ?? 's and?? i ?? 's to get a lower bound estimate of the global maximum error.

The lower bound estimate of the global maximum error for ?? is 9.12 (11.4% of the measurement range); for ?? is 4.08

??? (3.4% of the measurement range).

We can see that the error bounds that Tiler delivers are close to the lower bound estimates derived from the observed errors that the network exhibits for specific inputs.

Having visualized the heatmaps for the bounds e Figures 2b and 2e .

We can see that most of the regions that have large error bounds are due to the fact that the network itself has large errors there.

By computing the cumulative distributions of these gaps between bounds and estimates, we found that for angle measurement, 99% of the state space has error gap below 1.9

??? (1.6% of measurement range); and for offset measurement, 99% of the state space has error gap below 1.41 length units (1.8%).

The gaps indicate the maximum possible improvements on the error bounds.

The first factor is that we use interval arithmetic to compute the error bound in Tiler: Tiler takes the maximum distance between the range of possible ground truths and the range of possible network outputs as the bound.

The second factor is the extra space included in the box B i that is not in the tile X i .

This results in a larger range on network output being used for calculating error bound, which in turn makes the error bound itself larger.

Effect of tile size: Both of the factors described above are affected by the tile size.

We run Tiler with a sequence of cell sizes (0.05, 0.1, 0.2, 0.4, 0.8) for the (??, ??) grid.

Figure 3a shows how the 99 percentiles of the error upper bounds and the gap between error bounds and estimates vary with cell size.

As tile size gets finer, Tiler provides better error bounds, and the tightness of bounds improves.

These results show that we can get better error bounds with finer tile sizes.

But this improvement might be at the cost of time: reducing tile sizes also increases the total number of tiles and the number of optimization problems to solve.

Figure 3b shows how the total solving time varies with cell size.

For cell sizes smaller than 0.2, the trend can be explained by the above argument.

For cell sizes larger than 0.2, total solving time increases with cell size instead.

The reason is each optimization problem becomes harder to solve as the tile becomes large.

Specifically, the approach we adopt (Tjeng et al., 2019) relies on the presolving on ReLU stability to improve speed.

The number of unstable ReLUs will increase drastically as the cell size becomes large, which makes the solving slower.

We implement the input detector by checking if the new input x * is contained in any of the bounding boxes B i .

We test the detector with 3 types of inputs: 1) legal inputs, generated from the state space through the imaging process; 2) corrupted inputs, obtained by applying i.i.d uniformly distributed per-pixel perturbation to legal inputs; 3) inputs from a new scene, where the road is wider and there is a double centerline.

Figure 3c to 3e show some example images for each type.

We randomly generated 500 images for each type.

Our detector is able to flag all inputs from type 1 as legal, and all inputs from type 2 and 3 as illegal.

On average, naive search (over all B i ) takes 1.04s per input, while prediction-guided search takes 0.04s per input.

So the prediction-guided search gives a 26?? speedup without any compromise in functionality.

Problem set-up The world in this case contains a planar sign standing on the ground.

There are 3 types of signs with different shapes: square, triangle, and circle (Figure 4b) .

A LiDAR sensor takes measurement of the scene, which is used as input to a neural network to classify the shape of the sign.

The sensor can vary its distance d and angle ?? with respect to the sign, but its height is fixed, and it is always facing towards the sign.

Figure 4a shows the schematic of the set-up.

Assume the working zone for the LiDAR sensor is with position d ??? [30, 60] and ?? ??? [???45

??? , 45 ??? ].

Then the state space S has 3 dimensions: two continuous (d and ??), and one discrete (sign shape c).

The LiDAR sensor emits an array of 32??32 laser beams in fixed directions.

The measurement from each beam is the distance to the first object hit in that direction.

We consider a LiDAR measurement model where the maximum distance that can be measured is MAX_RANGE=300, and the measurement has a Gaussian noise with zero mean and a standard deviation of 0.1% of MAX_RANGE.

This gives the observation process g. Appendix D.1 and D.2 provides more details on the scene and LiDAR measurement model.

We use a CNN with 2 convolutional layers (size 4??4) with 16 filters, followed by a fully connected layer with 100 units.

The distance measurements are preprocessed before feeding into the network: first dividing them by MAX_RANGE to scale to [0,1], then using 1 minus the scaled distances as inputs.

This helps the network training.

We train the network using 50k points from each class, and validating using 500 points from each class.

The training settings are the same as the previous case study (Appendix E).

Tiler The state tiles are constructed in each of the three shape subspaces.

We divide the ?? dimension uniformly into 90 intervals and the d dimension uniformly in the inverse scale into 60 intervals to obtain a grid with 5400 cells per shape.

To compute the bounding box B i for a given tile S i , we first find a lower bound and an upper bound on the distance of object for each beam as the sensor position varies within that tile S i (Appendix D.3).

We extend this lower and upper bound by 5??, where ?? is the standard deviation of the Gaussian measurement noise.

This way we have P (x ??? B i |x ??? g(s), s ??? S i ) >= (P (|a| <= 5??|a ??? N (0, ?? 2 ))) N > 0.999, where N = 32 ?? 32 is the input dimension.

The factor 5 can be changed, depending on the required probabilistic guarantee.

Same as the previous case study, we adopt the MILP method (Tjeng et al., 2019) to solve the network output, which is used to decide whether the tile is verified to be correct or not.

We plot the verification results as heatmaps over the state space in Figure 5 (top row).

We are able to verify the correctness of the network over the majority of the state space.

In particular, we verify that the network is always correct when the shape of the sign is triangle.

Besides the tiling described above, we also run a finer tiling with half the cell sizes in both d and ??.

Figure 5 (bottom row) shows the verification results.

By reducing the tile sizes, we can verify more regions in the state space.

For the tiles that we are unable to verify correctness (red squares in the heatmaps), there are inputs within those bounding boxes on which the network will predict a different class.

We inspect several of such tiles to see the inputs that cause problems.

Figure 4c shows a few examples.

In some of the cases (top two in figure) , the 'misclassified' inputs actually do not look like coming from the ground truth class.

This is because the extra space included in B i is too large, so that it includes inputs that are reasonably different from feasible inputs.

Such cases will be reduced as the tile size becomes smaller, since the extra spaces in B i 's will be shrunk.

We have indeed observed such phenomenon, as we can verify more regions when the tile size becomes smaller.

In some other cases (bottom example), however, the misclassified inputs are perceptually similar to the ground truth class.

Yet the network predicts a different class.

This reveals that the network is potentially not very robust on inputs around these points.

In this sense, our framework provides a way to systematically find regions of the input space of interests where the network is potentially vulnerable.

The techniques presented in this paper work with specifications provided by the combination of a state space of the world and an observation process that converts states into neural network inputs.

Results from the case studies highlight how well the approach works for a state space characterized by several attributes and a camera imaging or LiDAR measurement observation process.

We anticipate that the technique will also work well for other problems that have a low dimensional state space (but potentially a high dimensional input space).

For higher dimensional state spaces, the framework makes it possible to systematically target specific regions of the input space to verify.

Potential applications include targeted verification, directed testing, and the identification of illegal inputs for which the network is not expected to work on.

A PROOFS FOR THEOREM 1 AND 2 Theorem 1 (Local error bound for regression).

Given that Condition 4.1, 4.2(a), 4.3, and 4.4(a) are satisfied, then ???x ???X , e(x) ??? e local (x), where e(x) is defined in Equation 2 and e local (x) is computed from Equation 3 and 5.

Proof.

For any x ???X , we have

Condition 4.1 and 4.2(a) guarantees that for any x ???X and y ???f (x), we can find a tile X i such that x ??? X i and l i ??? y ??? u i .

Let t(y, x) be a function that gives such a tile X t(y,x) for a given x and y ???f (x).

Then

Since x ??? X t(y,x) and X t(y,x) ??? B t(y,x) (Condition 4.3), x ??? B t(y,x) .

By Condition 4.4(a),

This gives

Since x ??? B t(y,x) for all y ???f (x), we have {t(y, x)|y ???f (x)} ??? {i|x ??? B i }, which gives

Theorem 2 (Global error bound for regression).

Given that Condition 4.1, 4.2(a), 4.3, and 4.4(a) are satisfied, then ???x ???X , e(x) ??? e global , where e(x) is defined in Equation 2 and e global is computed from Equation 3 and 4.

Proof.

By Theorem 1, we have ???x ???X ,

We present here the algorithm of Tiler for classification settings.

Step 1 (tiling the space) is the same as regression.

Step 2: For each S i , compute the ground truth bound as a set C i ??? Y, such that ???s ??? S i , ??(s) ??? C i .

The bounds computed this way satisfy the following condition:

Condition 4.2(b).

For any x ???X , ???y ???f (x), ???X i such that x ??? X i and y ??? C i .

The idea behind Condition 4.2(b) is the same as that of Condition 4.2(a), but formulated for discrete y.

For tiles with C i containing more than 1 class, we cannot verify correctness since there is more than 1 possible ground truth class for that tile.

Therefore, we should try to make the state tiles containing only 1 ground truth class each when tiling the space in step 1.

For tiles with |C i | = 1, we proceed to the following steps.

Step 3 (compute bounding box for each input tile) is the same as regression.

The next step is to solve the network output range.

Suppose the quantity of interest has K possible classes.

Then the output layer of the neural network is typically a softmax layer with K output nodes.

Denote the k-th output score before softmax as o k (x).

We use the solver to solve the minimum difference between the output score for the ground truth class and each of the other classes:

Step 4:

Given f , B i , and the ground truth class c i (the only element in C i ), use appropriate solver to solve lower bounds on the difference between the output score for class c i and each of the other classes: l

for k ??? {1, . . .

, K} \ {c i }.

The bounds need to satisfy:

Step 5:

For each tile, compute an error bound e i , with e i = 0 meaning the network is guaranteed to be correct for this state tile, and e i = 1 meaning no guarantee:

Otherwise, e i = 1.

We can then compute the global and local error bounds using Equation 4 and 5, same as in the regression case.

Theorem 3 (Local error bound for classification).

Given that Condition 4. 1, 4.2(b), 4.3, and 4.4(b) are satisfied, then ???x ???X , e(x) ??? e local (x), where e(x) is defined in Equation 2 and e local (x) is computed from Equation 6 and 5.

Equivalently, when e local (x) = 0, the network prediction is guaranteed to be correct at x. which also meansf (x) ??? {i|x???Xi} C i .

But {i|x ??? X i } ??? {i|x ??? B i } (Condition 4.3), which give??

Sincef (x) is not empty, we havef (x) = {c p } = f (x).

Theorem 4 (Global error bound for classification).

Given that Condition 4. 1, 4.2(b), 4.3, and 4.4(b) are satisfied, then if e global = 0, the network prediction is guaranteed to be correct for all x ???X .

e global is computed from Equation 6 and 4.

Proof.

We aim to prove that if e global = 0, then ???x ???X , f (x) =f (x).

According to Equation 4, e global = 0 means e i = 0 for all i.

Then ???x ???X , e local (x) = 0, which by Theorem 3 indicates that f (x) =f (x).

Algorithm 2 formally presents the Tiler algorithm for classification.

Input: S, g, ??, f Output:

e global , {e i }, {B i } 1: procedure TILER(S, g, ??, f ) 2:

Step 1 3:

for each S i do 4:

Step 2 5:

Step 3 6:

Step 4 7:

Step 5, Equation 6 8:

end for

e global ??? max({e i })

Step 5 10:

return e global , {e i }, {B i } {

e i }, {B i } can be used later to compute e local (x) 11: end procedure C POSITION MEASUREMENT FROM ROAD SCENE C.1 SCENE For clarity, we describe the scene in a Cartesian coordinate system.

Treating 1 length unit as roughly 5 centimeters will give a realistic scale.

The scene contains a road in the xy-plane, extending along the y-axis.

A schematic view of the road down along the z-axis is shown in Figure 6a The schematic of the camera is shown in Figure 6b .

The camera's height above the road is fixed at z c = 20.0.

The focal length f = 1.0.

The image plane is divided into 32 ?? 32 pixels, with pixel side length d = 0.16.

The camera imaging process we use can be viewed as an one-step ray tracing: the intensity value of each pixel is determined by shooting a ray from the center of that pixel through the focal point, and take the intensity of the intersection point between the ray and the scene.

In this example scene, the intersection points for the top half of the pixels are in the sky (intensity 0.0).

The intersection points for the lower half of the pixels are on the xy-plane.

The position of the intersection point (in the

P P C is the transformation from pixel coordinate to camera coordinate.

Camera coordinate has the origin at the focal point, and axes aligned with the orientation of the camera.

We define the focal coordinate to have its origin also at the focal point, but with axes aligned with the world coordinate (the coordinate system used in Appendix C.1).

R CF is the rotation matrix that transforms from camera coordinate to focal coordinate.

P p represents the projection to the road plane through the focal point, in the focal coordinate.

Finally, T F W is the translation matrix that transforms from the focal coordinate to the world coordinate.

The transformation matrices are given below:

The variables are defined as in Table 1 , with ?? and ?? being the offset and angle of the camera.

After the intensity values of the pixels are determined, they are scaled and quantized to the range [0, 255] , which are used as the final image taken.

In the road scene example, we need to encapsulate each tile in the input space with a l ??? -norm ball for the MILP solver to solve.

This requires a method to compute a range for each pixel that covers all values this pixel can take for images in this tile.

This section presents the method used in this paper.

A tile in this example corresponds to images taken with camera position in a local range

For pixels in the upper half of the image, their values will always be the intensity of the sky.

For each pixel in the lower half of the image, if we trace the intersection point between the projection ray from this pixel and the road plane, it will sweep over a closed region as the camera position varies in the ??-?? cell.

The range of possible values for that pixel is then determined by the range of intensities in that region.

In this example, there is an efficient way of computing the range of intensities in the region of sweep.

Since the intensities on the road plane only varies with x, it suffices to find the span on x for the region.

The extrema on x can only be achieved at: 1) the four corners of the ??-?? cell; 2) the points on the two edges ?? = ?? 1 and ?? = ?? 2 where ?? gives the ray of that pixel perpendicular to the y axis (if that ?? is contained in [?? 1 , ?? 2 ]).

Therefore, by computing the location of these critical points, we can obtain the range of x. We can then obtain the range of intensities covered in the region of sweep, which will give the range of pixel values.

The sign is a planer object standing on the ground.

It consists of a holding stick and a sign head on top of the stick.

The stick is of height 40.0 and width 2.0 (all in terms of the length unit of the scene).

The sign head has three possible shapes: square (with side length 10.0), equilateral triangle (with side length 10.0), and circle (with diameter 10.0).

The center of the sign head coincides with the middle point of the top of the stick.

The LiDAR sensor can vary its position within the working zone (see Figure 4a ).

Its height is fixed at 40.0 above the ground.

The center direction of the LiDAR is always pointing parallel to the ground plane towards the centerline of the sign.

The LiDAR sensor emits an array of 32??32 laser beams and measures the distance to the first object along each beam.

The directions of the beams are arranged as follows.

At distance f = 4.0 away from the center of the sensor, there is a (imaginary) 32??32 grid with cell size 0.1.

There is a beam shooting from the center of the sensor through the center of each cell in the grid.

The LiDAR model we consider has a maximum measurement range of MAX_RANGE=300.0.

If the distance of the object is larger than MAX_RANGE, the reflected signal is too weak to be detected.

Therefore, all the distances larger than MAX_RANGE will be measured as MAX_RANGE.

The distance measurement contains a Guassian noise n ??? N (0, ?? 2 ), where ?? = 0.001 ?? MAX_RANGE.

So the measured distance for a beam is given by

where d 0 is the actual distance and d is the measured distance.

To compute the bounding boxes in Tiler, we need to compute a lower bound and an upper bound on the measured distance for each beam as the sensor position varies within a tile.

We first determine whether the intersection point p between the beam and the scene is 1) always on the sign, 2) always on the background (ground/sky), or 3) covers both cases, when the sensor position varies within the In most situations, the distances of p at the list of critical positions (we refer it as the list of critical distances) contain the maximum and minimum distances of p in the tile.

There is one exception: at d = d 1 , as ?? varies in [?? 1 , ?? 2 ], if the intersection point shifts from sign to background (or vice versa), then the minimum distance of p can occur at (d 1 , ?? ) where ?? is not equal to ?? 1 or ?? 2 .

To handle this case, if at the previous step we find that p do occur on the sign plane in the tile, then we add the distance of the intersection point between the beam and the sign plane at position (d 1 , ?? 1 ) (or (d 1 , ?? 2 ), whichever gives a smaller distance) to the list of critical distances.

Notice that this intersection point is not necessarily on the sign.

In this way, the min and max among the critical distances are guaranteed to bound the distance range for the beam as the sensor position varies within the tile.

After this, the bounds can be extended according to the noise scale to get the bounding boxes.

This section presents the additional details of the training of the neural networks in the case studies.

We use Adam optimizer with learning rate 0.01.

We use early stopping based on the loss on the validation set: we terminate training if the validation performance does not improve in 5 consecutive epochs.

We take the model from the epoch that has the lowest loss on the validation set.

<|TLDR|>

@highlight

We present the first verification that a neural network for perception tasks produces a correct output within a specified tolerance for every input of interest. 