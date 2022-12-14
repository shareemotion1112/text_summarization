In this paper we approach two relevant deep learning topics: i) tackling of graph structured input data and ii) a better understanding and analysis of deep networks and related learning algorithms.

With this in mind we focus on the topological classification of reachability in a particular subset of planar graphs (Mazes).

Doing so, we are able to model the topology of data while staying in Euclidean space, thus allowing its processing with standard CNN architectures.

We suggest a suitable architecture for this problem and show that it can express a perfect solution to the classification task.

The shape of the cost function around this solution is also derived and, remarkably, does not depend on the size of the maze in the large maze limit.

Responsible for this behavior are rare events in the dataset which strongly regulate the shape of the cost function near this global minimum.

We further identify an obstacle to learning in the form of poorly performing local minima in which the network chooses to ignore some of the inputs.

We further support our claims with training experiments and numerical analysis of the cost function on networks with up to $128$ layers.

Deep convolutional networks have achieved great success in the last years by presenting human and super-human performance on many machine learning problems such as image classification, speech recognition and natural language processing ).

Importantly, the data in these common tasks presents particular statistical properties and it normally rests on regular lattices (e.g. images) in Euclidean space BID3 ).

Recently, more attention has been given to other highly relevant problems in which the input data belongs to non-Euclidean spaces.

Such kind of data may present a graph structure when it represents, for instance, social networks, knowledge bases, brain activity, protein-interaction, 3D shapes and human body poses.

Although some works found in the literature propose methods and network architectures specifically tailored to tackle graph-like input data BID3 ; BID4 ; BID15 ; BID22 ; BID23 b) ), in comparison with other topics in the field this one is still not vastly investigated.

Another recent focus of interest of the machine learning community is in the detailed analysis of the functioning of deep networks and related algorithms BID8 ; BID12 ).

The minimization of high dimensional non-convex loss function by means of stochastic gradient descent techniques is theoretically unlikely, however the successful practical achievements suggest the contrary.

The hypothesis that very deep neural nets do not suffer from local minima BID9 ) is not completely proven BID36 ).

The already classical adversarial examples BID27 ), as well as new doubts about supposedly well understood questions, such as generalization BID43 ), bring even more relevance to a better understanding of the methods.

In the present work we aim to advance simultaneously in the two directions described above.

To accomplish this goal we focus on the topological classification of graphs BID29 ; BID30 ).

However, we restrict our attention to a particular subset of planar graphs constrained by a regular lattice.

The reason for that is threefold: i) doing so we still touch upon the issue of real world graph structured data, such as the 2D pose of a human body BID1 ; BID16 ) or road networks BID25 ; BID39 ); ii) we maintain the data in Euclidean space, allowing its processing with standard CNN architectures; iii) this particular class of graphs has various non-trivial statistical properties derived from percolation theory and conformal field theories BID5 ; BID20 ; BID34 ), allowing us to analytically compute various properties of a deep CNN proposed by the authors to tackle the problem.

Specifically, we introduce Maze-testing, a specialized version of the reachability problem in graphs BID42 ).

In Maze-testing, random mazes, defined as L by L binary images, are classified as solvable or unsolvable according to the existence of a path between given starting and ending points in the maze (vertices in the planar graph).

Other recent works approach maze problems without framing them as graphs BID37 ; BID28 ; BID33 ).

However, to do so with mazes (and maps) is a common practice in graph theory BID2 ; BID32 ) and in applied areas, such as robotics BID11 ; BID7 ).

Our Mazetesting problem enjoys a high degree of analytical tractability, thereby allowing us to gain important theoretical insights regarding the learning process.

We propose a deep network to tackle the problem that consists of O(L 2 ) layers, alternating convolutional, sigmoid, and skip operations, followed at the end by a logistic regression function.

We prove that such a network can express an exact solution to this problem which we call the optimal-BFS (breadth-first search) minimum.

We derive the shape of the cost function around this minimum.

Quite surprisingly, we find that gradients around the minimum do not scale with L. This peculiar effect is attributed to rare events in the data.

In addition, we shed light on a type of sub-optimal local minima in the cost function which we dub "neglect minima".

Such minima occur when the network discards some important features of the data samples, and instead develops a sub-optimal strategy based on the remaining features.

Minima similar in nature to the above optimal-BFS and neglect minima are shown to occur in numerical training and dominate the training dynamics.

Despite the fact the Maze-testing is a toy problem, we believe that its fundamental properties can be observed in real problems, as is frequently the case in natural phenomena BID31 ), making the presented analytical analysis of broader relevance.

Additionally important, our framework also relates to neural network architectures with augmented memory, such as Neural Turing Machines BID13 ) and memory networks BID40 ; BID35 ).

The hot-spot images FIG9 , used to track the state of our graph search algorithm, may be seen as an external memory.

Therefore, to observe how activations spread from the starting to the ending point in the hot-spot images, and to analyze errors and the landscape of the cost function (Sec. 5) , is analogous to analyze how errors occur in the memory of the aforementioned architectures.

This connection gets even stronger when such memory architectures are employed over graph structured data, to perform task such as natural language reasoning and graph search ; BID17 ; BID14 ).

In these cases, it can be considered that their memories in fact encode graphs, as it happens in our framework.

Thus, the present analysis may eventually help towards a better understanding of the cost functions of memory architectures, potentially leading to improvements of their weight initialization and optimization algorithms thereby facilitating training BID26 ).The paper is organized as follows: Sec. 2 describes in detail the Maze-testing problem.

In Sec. 3 we suggest an appropriate architecture for the problem.

In Sec. 4 we describe an optimal set of weights for the proposed architecture and prove that it solves the problem exactly.

In Sec. 5 we report on training experiments and describe the observed training phenomena.

In Sec. 6 we provide an analytical understanding of the observed training phenomena.

Finally, we conclude with a discussion and an outlook.

Let us introduce the Maze-testing classification problem.

Mazes are constructed as a random two dimensional, L ?? L, black and white array of cells (I) where the probability (??) of having a black cell is given by ?? c = 0.59274(6), while the other cells are white.

An additional image (H 0 ), called the initial hot-spot image, is provided.

It defines the starting point by being zero (Off) everywhere except on a 2 ?? 2 square of cells having the value 1 (On) chosen at a random position (see FIG9 .

A Maze-testing sample (i.e. a maze and a hot-spot image) is labelled Solvable if the ending point, defined as a 2 ?? 2 square at the center of the maze, is reachable from the starting point (defined by the hot-spot image) by moving horizontally or vertically along black cells.

The sample is labelled Unsolvable otherwise.

A maze-testing sample consists of a maze (I) and an initial hot-spot image (H0).

The proposed architecture processes H0 by generating a series of hot-spot images (Hi>0) which are of the same dimension as H0 however their pixels are not binary but rather take on values between 0 (Off, pale-orange) and 1 (On, red).

This architecture can represent an optimal solution, wherein the red region in H0 spreads on the black cluster in I to which it belongs.

Once the spreading has exhausted, the Solvable/Unsolvable label is determined by the values of Hn at center (ending point) of the maze.

In the above example, the maze in question is Unsolvable, therefore the On cells do not reach the ending point at the center of Hn.

A maze in a Maze-testing sample has various non-trivial statistical properties which can be derived analytically based on results from percolation theory and conformal field theory BID5 ; BID20 ; BID34 ).

Throughout this work we directly employ such statistical properties, however we refer the reader to the aforementioned references for further details and mathematical derivations.

At the particular value chosen for ??, the problem is at the percolation-threshold which marks the phase transition between the two different connectivity properties of the maze: below ?? c the chance of having a solvable maze decays exponentially with r (the geometrical distance between the ending and starting points).

Above ?? c it tends to a constant at large r. Exactly at ?? c the chance of having a solvable maze decays as a power law (1/r ?? , ?? = 5/24).

We note in passing that although Mazetesting can be defined for any ??, only the choice ?? = ?? c leads to a computational problem whose typical complexity increases with L.Maze-testing datasets can be produced very easily by generating random arrays and then analyzing their connectivity properties using breadth-first search (BFS), whose worse case complexity is O(L 2 ).

Notably, as the system size grows larger, the chance of producing solvable mazes decay as 1/L ?? , and so, for very large L, the labels will be biased towards unsolvable.

There are several ways to de-bias the dataset.

One is to select an unbiased subset of it.

Alternatively, one can gradually increase the size of the starting-point to a starting-square whose length scales as L ?? .

Unless stated otherwise, we simply leave the dataset biased but define a normalized test error (err), which is proportional to the average mislabeling rate of the dataset divided by the average probability of being solvable.

Here we introduce an image classification architecture to tackle the Maze-testing problem.

We frame maze samples as a subclass of planar graphs, defined as regular lattices in the Euclidean space, which can be handle by regular CNNs.

Our architecture can express an exact solution to the problem and, at least for small Mazes (L ??? 16), it can find quite good solutions during training.

Although applicable to general cases, graph oriented architectures find it difficult to handle large sparse graphs due to regularization issues BID15 ; BID22 ), whereas we show that our architecture can perform reasonably well in the planar subclass.

Our network, shown in FIG9 , is a deep feedforward network with skip layers, followed by a logistic regression module.

The deep part of the network consists of n alternating convolutional and sigmoid layers.

Each such layer (i) receives two L ?? L images, one corresponding to the original maze (I) and the other is the output of the previous layer (H i???1 ).

It performs the operation DISPLAYFORM0 , where * denotes a 2D convolution, the K convolutional kernel is 1??1, the K hot kernel is 3??3, b is a bias, and ??(x) = (1+e ???x ) ???1 is the sigmoid function.

The logistic regression layer consists of two perceptrons (j = 0, 1), acting on DISPLAYFORM1 where H n is the rasterized/flattened version of H n , W j is a 2 ?? L 2 matrix, and b reg is a vector of dimension 2.

The logistic regression module outputs the label Solvable if p 1 ??? p 0 and Unsolvable otherwise.

The cost function we used during training was the standard negative log-likelihood.

As we next show, the architecture above can provide an exact solution to the problem by effectively forming a cellular automaton executing a breadth-first search (BFS).

A choice of parameters which achieves this is ?? ??? ?? c = 9.727??0.001, DISPLAYFORM0 T , where q center is the index of H n which corresponds to the center of the maze.

Let us explain how the above neural network processes the image (see also FIG9 ).

Initially H 0 is On only at the starting-point.

Passing through the first convolutional-sigmoid layer it outputs H 1 which will be On (i.e. have values close to one) on all black cells which neighbor the On cells as well as on the original starting point.

Thus On regions spread on the black cluster which contains the original starting-point, while white clusters and black clusters which do not contain the starting-point remain Off (close to zero in H i ).

The final logistic regression layer simply checks whether one of the 2 ?? 2 cells at the center of the maze are On and outputs the labels accordingly.

To formalize the above we start by defining two activation thresholds, v l and v h , and refer to activations which are below v l as being Off and to those above v h as being On.

The quantity v l is defined as the smallest of the three real solutions of the equation v l = ??(5v l ??? 0.5??).

Notably we previously chose ?? > ?? c as this is the critical value above which three real solutions to v l (rather than one) exist.

For v h we choose 0.9.Next, we go case by case and show that the action of the convolutional-sigmoid layer switches activations between Off and On just as a BFS would.

This amounts to bounding the expression ??(K hot * H i???1 + K * I + b) for all possibly 3 ?? 3 sub-arrays of H i???1 and 1 ?? 1 sub-arrays of I. There are thus 2 10 possibilities to be examined.

FIG1 shows the desired action of the layer on three important cases (A-C).

Each case depicts the maze shape around some arbitrary point x, the hot-spot image around x before the action of the layer (H i???1 ), and the desired action of the layer (H i ).

Case A. Having a white cell at x implies I[x] = 0 and therefore the argument of the above sigmoid is smaller than ???0.5?? c this regardless of H i???1 at and around x. Thus H i [x] < v l and so it is Off.

As the 9 activations of H i???1 played no role, case A covers in fact 2 9 different cases.

Case B. Consider a black cell at x, with H i???1 in its vicinity all being Off (vicinity here refers to x and its 4 neighbors).

Here the argument is smaller or equal to 5v l ??? 0.5?? c , and so the activation remains Off as desired.

Case B covers 2 4 cases as the values of H i???1 on the 4 corners were irrelevant.

Case C. Consider a black cell at x with one or more On activations of H i???1 in its vicinity.

Here the argument is larger than v h ?? c ??? 0.5?? c = 0.4?? c .

The sigmoid is then larger than 0.97 implying it is On.

Case C covers 2 4 (2 5 ??? 1) different cases.

Since 2 9 + 2 4 + 2 4 (2 5 ??? 1) = 2 10 we exhausted all possible cases.

Lastly it can be easily verified that given an On (Off) activation at the center of the full maze the logistic regression layer will output the label Solvable (Unsolvable).Let us now determine the required depth for this specific architecture.

The previous analysis tells us that at depth d unsolvable mazes would always be labelled correctly however solvable mazes would be label correctly only if the shortest-path between the starting-point and the center is d or less.

The worse case scenario thus occurs when the center of the maze and the starting-point are connected by a one dimensional curve twisting its way along O(L 2 ) sites.

Therefore, for perfect performance the network depth would have to scale as the number of sites namely n = O(L 2 ).

A tighter but probabilistic bound on the minimal depth can be established by borrowing various results from percolation theory.

It is known, from BID44 , that the typical length of the shortest path (l) for critical percolation scales as r dmin , where r is the geometrical distance and d min = 1.1(3).

Moreover, it is known that the probability distribution P (l|r) has a tail which falls as l ???2 for l ???> r dmin BID10 ).

Consequently, the chance that at distance r the shortest path is longer than r dmin r a , where a is some small positive number, decays to zero and so, d should scale as L with a power slightly larger than d min (say n = L 1.2 ).

We have performed several training experiments with our architecture on L = 16 and L = 32 mazes with depth n = 16 and n = 32 respectively, datasets of sizes M = 1000, M = 10000, and M = 50000.

Unless stated otherwise we used a batch size of 20 and a learning rate of 0.02.

In the following, we split the experiments into two different groups corresponding to the related phenomena observed during training, which will the analyzed in detail in the next section.

Optimal-BFS like minima.

For L = 16, M = 10000 mazes and a positive random initialization for K hot and K in [0, 6/8] the network found a solution with ??? 9% normalized test error performance in 3 out of the 7 different initializations (baseline test error was 50%).

In all three successful cases the minima was a variant of the Optimal-BFS minima which we refer to as the checkerboard-BFS minima.

It is similar to the optimal-BFS but spreads the On activations from the starting-point using a checkerboard pattern rather than a uniform one, as shown in FIG2 .

The fact that it reaches ??? 9% test error rather than zero is attributed to this checkerboard behavior which can occasionally miss out the exit point from the maze.

Neglect minima.

Again for L = 16 but allowing for negative entries in K and K hot test error following 14 attempts and 500 epochs did not improve below 44%.

Analyzing the weights of the network, the 6% improvement over the baseline error (50%) came solely from identifying the inverse correlation between many white cells near the center of the maze and the chance of being solvable.

Notably, this heuristic approach completely neglects information regarding the starting-point of the maze.

For L = 32 mazes, despite trying several random initialization strategies including positive entries, dataset sizes, and learning rates, the network always settled into such a partial neglect minimum.

In an unsuccessful attempt to push the weights away from such partial neglect behavior, we performed further training experiments with a biased dataset in which the maze itself was uncorrelated with the label.

More accurately, marginalizing over the starting-point there is an equal chance for both labels given any particular maze.

To achieve this, a maze shape was chosen randomly and then many random locations were tried-out for the starting-point using that same maze.

From these, we picked 5 that resulted in a Solvable label and 5 that resulted in an Unsolvable label.

Maze shapes which were always Unsolvable were discarded.

Both the L = 16 and L = 32 mazes trained on this biased dataset performed poorly and yielded 50% test error.

Interestingly they improved their cost function by settling into weights in which b ??? ???10 is large compared to [K hot ] ij <??? 1 while W and b were close to zero (order of 0.01).

We have verified that such weights imply that activations in the last layer have a negligible dependence on the starting-point and a weak dependence on the maze shape.

We thus refer to this minimum as a "total neglect minimum".

Here we seek an analytical understanding of the aforementioned training phenomena through the analysis of the cost function around solutions similar or equal to those the network settled into during training.

Specifically we shall first study the cost function landscape around the optimal-BFS minimum.

As would become clearer at the end of that analysis, the optimal BFS shares many similarities with the checkerboard-BFS minimum obtained during training and one thus expects a similar cost function landscape around both of these.

The second phenomena analyzed below is the total neglect minimum obtained during training on the biased dataset.

The total neglect minimum can be thought of as an extreme version of the partial neglect minima found for L = 32 in the original dataset.

Our analysis of the cost function near the optimal-BFS minimum will be based on two separate models capturing the short and long scale behavior of the network near this miminum.

In the first model we approximate the network by linearizing its action around weak activations.

This model would enable us to identify the density of what we call "bugs" in the network.

In the second model we discretize the activation levels of the neural network into binary variables and study how the resulting cellular automaton behaves when such bugs are introduced.

FIG3 shows a numerical example of the dynamics we wish to analyze.

Up to layer 19 (H19) the On activations spread according to BFS however at H20 a very faint localized unwanted On activation begins to develop (a bug) and quickly saturates (H23).

Past this point BFS dynamics continues normally but spreads both the original and the unwanted On activations.

While not shown explicitly, On activations still appear only on black maze cells.

Notably the bug developed in rather large black region as can be deduced from the large red region in its origin.

See also a short movie showing the occurrence of this bug at https://youtu.be/2I436BVAVdM and more bugs at https://youtu.be/kh-AfOo4TkU.At https://youtu.be/t-_TDkt3ER4 a similar behavior is shown for the checkerboard-BFS.

Unlike an algorithm, a neural network is an analog entity and so a-priori there are no sharp distinctions between a functioning and a dis-functioning neural network.

An algorithm can be debugged and the bug can be identified as happening at a particular time step.

However it is unclear if one can generally pin-point a layer and a region within where a deep neural network clearly malfunctioned.

Interestingly we show that in our toy problem such pin-pointing of errors can be done in a sharp fashion by identifying fast and local processes which cause an unwanted switching been Off and On activations in H i (see FIG3 ).

We call these events bugs, as they are local, harmful, and have a sharp meaning in the algorithmic context.

Below we obtain asymptotic expressions for the chance of generating such bugs as the network weights are perturbed away from the optimal-BFS minimum.

The main result of this subsection, derived below, is that the density of bugs (or chance of bug per cell) scales as DISPLAYFORM0 for (?? ??? ?? c ) <??? 0 and zero for ?? ??? ?? c >= 0 where C ??? 1.7.

Following the analysis below, we expect the same dependence to hold for generic small perturbations only with different C and ?? c .

We have tested this claim numerically on several other types of perturbations (including ones that break the ??/2 rotational symmetry of the weights) and found that it holds.

To derive Eq.(1), we first recall the analysis in Sec. 4, initially as it is decreased ?? has no effect but to shift v l (the Off activation threshold) to a higher value.

However, at the critical value (?? = ?? c , v l = v l,c ) the solution corresponding to v l vanishes (becomes complex) and the correspondence with the BFS algorithm no longer holds in general.

This must not mean that all Off activations are no longer stable.

Indeed, recall that in Sec. 4 the argument that a black Off cell in the vicinity of Off cells remains Off FIG1 , Case B) assumed a worse case scenario in which all the cells in its vicinity where both Off, black, and had the maximal Off activation allowed (v l ).

However, if some cells in its vicinity are white, their Off activations levels are mainly determined by the absence of the large K term in the sigmoid argument and orders of magnitude smaller than v l .

We come to the conclusion that black Off cells in the vicinity of many white cells are less prone to be spontaneously turned On than black Off cells which are part of a large cluster of black cells (see also the bug in FIG3 ).

In fact using the same arguments one can show that infinitesimally below ?? c only uniform black mazes will cause the network to malfunction.

To further quantify this, consider a maze of size l ?? l where the hot-spot image is initially all zero and thus Off.

Intuitively this hot-spot image should be thought of as a sub-area of a larger maze located far away from the starting-point.

In this case a functioning network must leave all activation levels below v l .

To assess the chance of bugs we thus study the probability that the output of the final convolutional-sigmoid layer will have one or more On cells.

To this end, we find it useful to linearize the system around low activation yielding (see the Appendix for a complete derivation) DISPLAYFORM1 where r b denotes black cells (I(r b ) = 1), the sum is over the nearest neighboring black cells to r b , DISPLAYFORM2 For a given maze (I), Eq. (2), defines a linear Hermitian operator (L I ) with random off-diagonal matrix elements dictated by I via the restriction of the off-diagonal terms to black cells.

Stability of Off activations is ensured if this linear operator is contracting or equivalently if all its eigenvalues are smaller than 1 in magnitude.

Hermitian operators with local noisy entries have been studied extensively in physics, in the context of disordered systems and Anderson localization BID19 ).

Let us describe the main relevant results.

For almost all I's the spectrum of L consists of localized eigenfunctions (?? m ).

Any such function is centered around a random site (x m ) and decays exponentially away from that site with a decay length of ?? which in our case would be a several cells long.

Thus given ?? m with an eigenvalue |E m | > 1, t repeated actions of the convolutional-sigmoid layer will make ?? n [x] in a ?? vicinity of x m grow in size as e Emt .

Thus (|E m | ??? 1) ???1 gives the characteristic time it takes these localized eigenvalue to grow into an unwanted localized region with an On activation which we define as a bug.

Our original question of determining the chance of bugs now translates into a linear algebra task: finding, N??, the number of eigenvalues in L I which are larger than 1 in magnitude, averaged over I, for a given ??.

Since?? simply scales all eigenvalues one finds that N?? is the number of eigenvalues larger than?? ???1 in L I with?? = 1.

Analyzing this latter operator, it is easy to show that the maximal eigenvalues occurs when ?? n (r) has a uniform pattern on a large uniform region where the I is black.

Indeed if I contains a black uniform true box of dimension l u ?? l u , the maximal eigenvalue is easily shown to be E lu = 5 ??? 2?? 2 /(l u ) 2 .

However the chance that such a uniform region exists goes as (l/l u ) 2 e log(??c)l 2 u and so P (???E) ??? l 2 e log(??c )2?? 2 (???E), where ???E = 5 ??? E. This reasoning is rigorous as far as lower bounds on N?? are concerned, however it turns out to capture the functional behavior of P (???E) near ???E = 0 accurately BID18 ) which is given by ???E) , where the unknown constant C captures the dependence on various microscopic details.

In the Appendix we find numerically that C ??? 0.7.

Following this we find DISPLAYFORM3 DISPLAYFORM4 The range of integration is chosen to includes all eigenvalues which, following a multiplication by??, would be larger than 1.To conclude we found the number of isolated unwanted On activations which develop on l ?? l Off regions.

Dividing this number by l 2 we obtain the density of bugs (?? bug ) near ?? ??? ?? c .

The last technical step is thus to express ?? bug in terms of ??.

Focusing on the small ?? bug region or ???E ??? 0 + , we find that ???E = 0 occurs when d?? dx (?? ???1 (?? ??? (??))) = 1/(5??),?? = 1/5, and ?? = ?? c = 9.72(7).Expanding around ?? = ?? c we find DISPLAYFORM5 2 ).

Approximating the integral over P (x) and taking the leading scale dependence, we arrive at Eq. FORMULA3 with C = C 10??c 49?????c .

In this subsection we wish to understand the large scale effect of ?? bug namely, its effect on the test error and the cost function.

Our key results here are that DISPLAYFORM0 DISPLAYFORM1 despite its appearance it can be verified that the above right hand side is smaller than L ???5/48 within its domain of applicability.

To derive Eqs. FORMULA9 and FORMULA10 , we note that as a bug is created in a large maze, it quickly switches On the cells within the black "room" in which it was created.

From this region it spreads according to BFS and turns On the entire cluster connected to the buggy room (see FIG3 ).

To asses the effect this bug has on performance first note that solvable mazes would be labelled Solvable regardless of bugs however unsolvable mazes might appear solvable if a bug occurs on a cell which is connected to the center of the maze.

Assuming we have an unsolvable maze, we thus ask what is the chance of it being classified as solvable.

Given a particular unsolvable maze instance (I), the chance of classifying it as solvable is given by p err (I) = 1 DISPLAYFORM2 where s counts the number of sites in the cluster connected to the central site (central cluster).

The probability distribution of s for percolation is known and given by p(s) = Bs 1????? , ?? = 187/91 BID6 ), with B being an order of one constant which depends on the underlying lattice.

Since clusters have a fractional dimension, the maximal cluster size is L d f .

Consequently, p err (I) averaged over all I instances is given by DISPLAYFORM3 ds, which can be easily expressed in terms of Gamma functions (??(x), ??(a, x)) (see BID0 ).

In the limit of ?? bug <??? L ???d f , where its derivatives with respect to ?? bug are maximal, it simplifies to DISPLAYFORM4 whereas for ?? bug > L ???d f , its behavior changes to p err = (???B??(2 ??? ?? ))?? DISPLAYFORM5 bug .

Notably once ?? bug becomes of order one, several of the approximation we took break down.

Let us relate p err to the test error (err).

In Sec. (2) the cost function was defined as the mislabeling chance over the average chance of being solvable (p solvable ).

Following the above discussion the mislabelling chance is p err p solvable and consequently err = p err .

Combining Eqs. 1 and 5 we obtain our key results, Eqs. (3, 4)As a side, one should appreciate a potential training obstacle that has been avoided related to the fact that err ??? ?? 5/91 big .

Considering L ??? ???, if ?? bug was simply proportional to (?? c ??? ??), err will have a sharp singularity near zero.

For instance, as one reduces err by a factor of 1/e, the gradients increase by e 86/5 ??? 3E + 7.

These effects are in accordance with ones intuition that a few bugs in a long algorithm will typically have a devastating effect on performance.

Interestingly however, the essential singularity in ?? bug (??), derived in the previous section, completely flattens the gradients near ?? c .Thus the essentially singularity which comes directly from rare events in the dataset strongly regulates the test error and in a related way the cost function.

However it also has a negative side-effect concerning the robustness of generalization.

Given a finite dataset the rarity of events is bounded and so having ?? < ?? c may still provide perfect performance.

However when encountering a larger dataset some samples with rarer events (i.e. larger black region) would appear and the network will fail sharply on these (i.e. the wrong prediction would get a high probability).

Further implications of this dependence on rare events on training and generalization errors will be studied in future work.

To provide an explanation for this phenomena let us divide the activations of the upper layer to its starting-point dependent and independent parts.

Let H n denote the activations at the top layer.

We expand them as a sum of two functions DISPLAYFORM0 where the function A and B are normalized such that their variance on the data (?? and ??, respectively) is 1.

Notably near the reported total neglect minima we found that ?? ?? ??? e ???10 .

Also note that for the biased dataset the maze itself is uncorrelated with the labels and thus ?? can be thought of as noise.

Clearly any solution to the Maze testing problem requires the starting-point dependent part (??) to become larger than the independent part (??).

We argue however that in the process of increasing ?? the activations will have to go through an intermediate "noisy" region.

In this noisy region ?? grows in magnitude however much less than ?? and in particular obeys ?? < ?? 2 .

As shown in the Appendix the negative log-likelihood, a commonly used cost function, is proportional to ?? 2 ??? ?? for ??, ?? 1.

Thus it penalizes random false predictions and, within a region obeying ?? < ?? 2 it has a minimum (global with respect to that region) when ?? = ?? = 0.

The later being the definition of a total neglect minima.

Establishing the above ?? ?? 2 conjecture analytically requires several pathological cases to be examined and is left for future work.

In this work we provide an argument for its typical correctness along with supporting numerics in the Appendix.

A deep convolution network with a finite kernel has a notion of distance and locality.

For many parameters ranges it exhibits a typical correlation length (??).

That is a scale beyond which two activations are statistically independent.

Clearly to solve the current problem ?? has to grow to an order of L such that information from the input reaches the output.

However as ?? gradually grows, relevant and irrelevant information is being mixed and propagated onto the final layer.

While ?? depends on information which is locally accessible at each layer (i.e. the maze shape), ?? requires information to travel from the first layer to the last.

Consequently ?? and ?? are expected to scale differently, as e ???L/?? and e ???1/?? resp. (for ?? << L).

Given this one finds that ?? ?? 2 as claimed.

Further numerical support of this conjecture is shown in the Appendix where an upper bound on the ratio ??/?? 2 is studied on 100 different paths leading from the total neglect miminum found during training to the checkerboard-BFS minimum.

In all cases there is a large region around the total neglect minimum in which ?? ?? 2 .

Despite their black-box reputation, in this work we were able to shed some light on how a particular deep CNN architecture learns to classify topological properties of graph structured data.

Instead of focusing our attention on general graphs, which would correspond to data in non-Euclidean spaces, we restricted ourselves to planar graphs over regular lattices, which are still capable of modelling real world problems while being suitable to CNN architectures.

We described a toy problem of this type (Maze-testing) and showed that a simple CNN architecture can express an exact solution to this problem.

Our main contribution was an asymptotic analysis of the cost function landscape near two types of minima which the network typically settles into: BFS type minima which effectively executes a breadth-first search algorithm and poorly performing minima in which important features of the input are neglected.

Quite surprisingly, we found that near the BFS type minima gradients do not scale with L, the maze size.

This implies that global optimization approaches can find such minima in an average time that does not increase with L. Such very moderate gradients are the result of an essential singularity in the cost function around the exact solution.

This singularity in turn arises from rare statistical events in the data which act as early precursors to failure of the neural network thereby preventing a sharp and abrupt increase in the cost function.

In addition we identified an obstacle to learning whose severity scales with L which we called neglect minima.

These are poorly performing minima in which the network neglects some important features relevant for predicting the label.

We conjectured that these occur since the gradual incorporation of these important features in the prediction requires some period in the training process in which predictions become more noisy.

A "wall of noise" then keeps the network in a poorly performing state.

It would be interesting to study how well the results and lessons learned here generalize to other tasks which require very deep architectures.

These include the importance of rare-events, the essential singularities in the cost function, the localized nature of malfunctions (bugs), and neglect minima stabilized by walls of noise.

These conjectures potentially could be tested analytically, using other toy models as well as on real world problems, such as basic graph algorithms (e.g. shortest-path) BID14 ); textual reasoning on the bAbI dataset ), which can be modelled as a graph; and primitive operations in "memory" architectures (e.g. copy and sorting) BID13 ).

More specifically the importance of rare-events can be analyzed by studying the statistics of errors on the dataset as it is perturbed away from a numerically obtained minimum.

Technically one should test whether the perturbation induces an typical small deviation of the prediction on most samples in the dataset or rather a strong deviation on just a few samples.

Bugs can be similarly identified by comparing the activations of the network on the numerically obtained minimum and on some small perturbation to that minimum while again looking at typical versus extreme deviations.

Such an analysis can potentially lead to safer and more robust designs were the network fails typically and mildly rather than rarely and strongly.

Turning to partial neglect minima these can be identified provided one has some prior knowledge on the relevant features in the dataset.

The correlations or mutual information between these features and the activations at the final layer can then be studied to detect any sign of neglect.

If problems involving neglect are discovered it may be beneficial to add extra terms to the cost function which encourage more mutual information between these neglected features and the labels thereby overcoming the noise barrier and pushing the training dynamics away from such neglect minimum.

We have implemented the architecture described in the main text using Theano BID38 ) and tested how cost changes as a function of ?? = ?? c ??? ?? (?? c = 9.727..) for mazes of sizes L = 24, 36 and depth (number of layers) 128.

These depths are enough to keep the error rate negligible at ?? = 0.

A slight change made compared to Maze-testing as described in the main text, is that the hot-spot was fixed at a distance L/2 for all mazes.

The size of the datasets was between 1E + 5 and 1E + 6.

We numerically obtained the normalized performance (cost L (??)) as a function of L and ??.

As it follows from Eq. (4) in the main text the curve, log(L ???2+5/24 cost L (??)), for the L = 24 and L = 36 results should collapse on each other for ?? bug < L ???d f .

FIG4 ) of the main-test depicts three such curves, two for L = 36, to give an impression of statistical error, and one for L = 24 curve (green), along with the fit to the theory (dashed line).

The fit, which involves two parameters (the proportionally constant in Eq. (4) of the main text and C) captures well the behavior over three orders of magnitude.

As our results are only asymptotic, both in the sense of large L and ?? ??? ?? c , minor discrepancies are expected.

To prepare the action of sigmoid-convolutional for linearization we find it useful to introduce the following variables on locations (r b , r w ) with black (b) and white (w) cells DISPLAYFORM0 a(r w ) = e ???5.5?? .homogeneous.

Consequently destabilization occurs exactly at?? = 1/5 and is not blurred by the inhomogeneous terms.

Recall that ?? c is defined as the value at which the two lower solutions of .

It is now easy to verify that even within the linear approximation destabilization occurs exactly at ?? c .

The source of this agreement is the fact that d vanishes for a uniform black maze.

The qualitative lesson here is thus the following: The eigenvectors of S with large s, are associated with large black regions in the maze.

It is only on the boundaries of such regions that d is non-zero.

Consequently near ?? ??? ?? c the d term projected on the largest eigenvalues can, to a good accuracy, be ignored and stability analysis can be carried on the homogeneous equation ?? = S ?? where s n < 1 means stability and s n > 1 implies a bug.

Consider an abstract classification tasks where data point x ??? X are classified into two categories l ??? {0, 1} using a deterministic function f : X ??? {0, 1} and further assume for simplicity that the chance of f (x) = a is equal to f (x) = b. Phrased as a conditional probability distribution P f (l|x) is given by P f (f (a)|x) = 1 while P f (!f (a)|x) = 0.

Next we wish to compare the following family of DISPLAYFORM0 where g|X ??? {0, 1} is a random function, uncorrelated with f (x), outputting the labels {0, 1} with equal probability.

Notably at ?? = 1/2, ?? = 0 it yields P f while at ?? , ?? = 0 it is simply the maximum entropy distribution.

Let us measure the log-likelihood of P ?? ,?? under P f for ?? , ?? 1 L(?? , ?? ) = We thus find that ?? reduces the log-likelihood in what can be viewed as a penalty to false confidence or noise.

Assuming, as argued in the main text, that ?? is constrained to be smaller than ?? 2 near ?? ??? 0, it is preferable to take both ?? and ?? to zero and reach the maximal entropy distribution.

We note by passing that the same arguments could be easily generalized to f (x), g(x) taking real values leading again to an O(??) ??? O(?? 2 ) dependence in the cost function.

Let us relate the above notations to the ones in the main text.

Clearly x = ({I}, H 0 ) and {0, 1} = {U nsolvable, Solvable}. Next we recall that in the main text ?? and ?? multiplied the vectors function representing the H 0 -depended and H 0 -independent parts of H n .

The probability estimated by the logistic regression module was given by P (Solvable|x) = e K Solvable ?? Hn e ??? K Solvable ?? Hn + e ??? K U nsolvable ?? Hn (17) P (U nsolvable|x) = e K U nsolvable ??

Hn e ??? K Solvable ?? Hn + e ??? K U nsolvable ?? Hn which yields, to leading order in ?? and ?? P ??,?? (l|x) = 1/2 + ??(2l + 1) DISPLAYFORM1 where K ??? = ( K Solvable ??? K U nsolvable )/2 and (2l + 1) understood as the taking the values ??1.Consequently (2f ??? 1) and (2g ??? 1) are naturally identified with K Solvable ?? A/N A and K Solvable ?? B/N B respectively with N A and N B being normalization constants ensuring a variance of 1.

While (?? , ?? ) = (N A ??, N B ??).

Recall also that by construction of the dataset, the g we thus obtain is uncorrelated with f .

2 CONJECTUREHere we provide numerical evidence showing that ?? ?? 2 in a large region around the total neglect minima found during the training of our architecture on the biased dataset (i.e. the one where marginalizing over the starting-point yields a 50/50 chance of being solvable regardless of the maze shape).For a given set of K h ot, K and b parameters we fix the maze shape and study the variance of the top layer activations given O(100) different starting points.

We pick the maximal of these and then average this maximal variance over O(100) different mazes.

This yields our estimate of ??.

In fact it is an upper bound on ?? as this averaged-max-variance may reflect wrong prediction provided that they depend on H 0 .We then obtain an estimate of ?? by again calculating the average-max-variance of the top layer however now with H 0 = 0 for all maze shapes.

Next we chose a 100 random paths parametrized by ?? leading from the total neglect minima (?? = 0) for the total neglect through a random point at ?? = 15, and then to the checkerboard-BFS minima at ?? = 30.

The random point was placed within a hyper-cube of length 4 having the total neglect minima at its center.

The path was a simple quadratic interpolation between the three point.

The graph below shows the statistics of ??/?? 2 on these 100 different paths.

Notably no path even had ?? > e ???30 ?? 2 within the hyper-cube.

We have tried three different other lengths for the hyper cube (12 and 1) and arrived at the same conclusions.

The natural logarithm of an upper bound to ??/?? 2 as a function of a paramterization (??) of a path leading from the numerically obtained total neglect minima to the checkerboard BFS minima through a random point.

The three different curves show the max,mean, and median based on a 100 different paths.

Notably no path violated the ?? ?? 2 constrain in the vicinity of the total neglect minima.

@highlight

A toy dataset based on critical percolation in a planar graph provides an analytical window to the training dynamics of deep neural networks  