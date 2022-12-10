This paper presents the ballistic graph neural network.

Ballistic graph neural network tackles the weight distribution from a transportation perspective and has many different properties comparing to the traditional graph neural network pipeline.

The ballistic graph neural network does not require to calculate any eigenvalue.

The filters propagate exponentially faster($\sigma^2 \sim T^2$) comparing to traditional graph neural network($\sigma^2 \sim T$).

We use a perturbed coin operator to perturb and optimize the diffusion rate.

Our results show that by selecting the diffusion speed, the network can reach a similar accuracy with fewer parameters.

We also show the perturbed filters act as better representations comparing to pure ballistic ones.

We provide a new perspective of training graph neural network, by adjusting the diffusion rate, the neural network's performance can be improved.

How to collect the nodes' correlation on graphs fast and precisely?

Inspired by convolutional neural networks(CNNs), graph convolutional networks(GCNs) can be applied to many graph-based structures like images, chemical molecules and learning systems.

Kipf & Welling (2016) Similar to neural networks, GCNs rely on random walk diffusion based feature engineering to extract and exploit the useful features of the input data.

Recent works show random walk based methods can represent graph-structured data on the spatial vertex domain.

For example, Li et al. (2017) use bidirectional random walks on the graph to capture the spatial dependency and Perozzi et al. (2014) present a scalable learning algorithm for latent representations of vertices in a network using random walks.

Except for the spatial domain, many researchers focus on approximating filters using spectral graph theory method, for example, Bruna et al. (2013) construct a convolutional architecture based on the spectrum of the graph Laplacian; Defferrard et al. (2016) use high order polynomials of Laplacian matrix to learn the graphs in a NN structure model.

Consider a undirected graph G(V, E), for random walk start from vertex v 2 V (G), let p t (u) denotes the probability on vertex u at time t, we have P u p t (u) = 1.

At time = t + 1, the probability at vertex v will be:

where d(u) is degree on vertex u. The normalized walk matrix is defined as D 1/2 AD 1/2 .

where E(G) denotes the edges on G. Matrix notation as follows:

Consider a lazy random walk with 1/2 probability staying on current nodes.

AD 1 becomes 1 2 (AD 1 + I) and the lazy normalized lazy walk is (I + D 1/2 AD 1/2 )/2, where A is the adjancy matrix and D is the degree matrix.

For regular graph, D 1/2 AD 1/2 = AD 1 = D 1 A.

Graph convolutional networks(GCN) are powerful tools for learning graph representationKipf & Welling (2016) .

For traditional GCN, the structure is shown in Figure 1 .

Akin to neural networks(NNs), promising improvements have been achieved by defining the random walk diffusionbased filters and using them in a multi-layer NN.

However, as the depth of layers grows, over- smoothing appears a common issue faced by GCNs.

Li et al. (2018) .

The over-smooth can be attributed to the stacking of random walk diffusion-based feature extraction, resulting in the similarty of node representations.

In Kipf & Welling (2016) , the convolution isL = I + D 1/2 AD 1/2 .

In practice, the first part can be regarded as adding self-loops to the node and then the latter part can be regarded as a walk based diffusion.

For a k step lazy random no biased walk, the final probability distribution of the random walk will converge to applyingL k on the initial state.

The distance from the start point of a simple random walk will converge to C p k, where C, k is the constant and the number of total steps respectively.

1 .

The random walk based method can be regarded as a low pass filter on the graph.

The normalized graph Laplacian is L normalized = I L where L = D 1/2 AD 1/2 .

It is easy to prove that L normalized has an eigenvalue 0 with a eigenvector d 1/2 , where d is the degree vector.

L normalized has n eigenvalues: 0  i  n 1  2 with normalized orthonormal eigenvectors

Fourier transformation for a signal ⇡ t on graph with basis i is f (⇡ t ) i = ⇡ t · i , t denotes the time steps.

0 and 0 corresponds to the lowest frequency part and the larger i correponds to higher frequency components.

Consider the operator L rw = I + AD 1 , L rw has n eigenvectors The normalized distribution after t steps is:

, the Fourier transform reads:

Since rw 0 = 1, thus the diffusion operator (L rw ) t preserves the zero-frequency component rw 0 and suppress the high frequency part.

In this case, as the depth of GCN increase, the high-frequency information is lost, resulting in the over-smooth problem.

In this section, we analyze the long-time random walk behaviour from the spatial domain.

Figure 2 (a) and (b) show the short time and long-time behaviour respectively, for short time random walk, 1 there are many discussions about random walk asymptotic behaviour, for example, please see: https: //www.mit.edu/~kardar/teaching/projects/chemotaxis(AndreaSchmidt)/more_ random.htm, http://mathworld.wolfram.com/RandomWalk2-Dimensional.html and http://www.math.caltech.edu/~2016-17/2term/ma003/Notes/Lecture16.pdf 2 Lrw has similar matrices, share properties of represented linear operatorL neighbourhood information is captured and learned; as the time steps become larger, the probability distribution on the nodes becomes indistinguishable and increases the error in the classification task.

The distribution of ballistic walk proposed in this paper is shown in Figure 2 (c), being different from the random walk method, the distribution puts more weight on the more distant nodes as time steps increase.

Relationship between distribution and distance The definition of distance is:

probablity on node⇥least number of hoppings between start point and node (4) Figure 2 (b) and (c) show the distribution at the same number of steps start from the same point.

As the ballistic walk puts more weight on the farther nodes: the indistinguishable problem is circumvented, and the distance is increased.

The shape of the ballistic distribution is more oscillated, thus higher frequency information is preserved.

Since the distance within the same time steps is larger, we regard the ballistic walk is a faster transportation method comparing to the random walk (Figure 2(d) ).

Note the faster term does not mean the ballistic walk goes to farther nodes, both the random walk and ballistic walk reach the k-th hopping nodes in time step k, the ballistic walk has different weight distribution on more distant nodes.

In traditional GCN, The random walk/Laplacian matrix collects the correlated information over a graph.

Here we consider a two-dimensional condition, taking the start point as (0,0) and the correlated point is (i,j), the distance is denoted as d ij .

As discussed, the distance walk travels is C p k, in this case, though a d ij step walker can reach (i,j), the probability distribution on the correlated point is relatively low since the walk's average distance is C p k. In order to fully capture the correlation between the two vertices(in other words, increase the weight between (0,0) and (i,j)), two main methods are used:

1. take steps d ij 2 steps, in Defferrard et al. (2016) , in analogy to a 5⇥5 filter, the authors use Laplacian polynomial order up to 25 to achieve similar accuracy, the number of steps is far larger than the filter's size in CNNs.

()*+ ,-+.

Figure 3: Classical diffusion 2.

Pooling: the distance of two vertices is shortened after pooling operation.

d ij will reduce to dd ij /2e/ after a 2 ⇥2 pooling.

e.g. Henaff et al. (2015) and Bruna et al. (2013) use max pooling, Defferrard et al. (2016) use efficient pooling and Tran et al. (2018) use sort pooling.

Figure 3 shows the relation between the distance and number of steps for the random walk/Laplacian matrix.

As the steps increases, the walk diffuses with the distance ⇠ C p k, where k is the number of steps, resulting in the inefficiency in collecting information.

Suggested by Hammond et al. (2011) , the filter on most common graph convolutional network is:

comprises orthonormal eigenvectors and ⇤ = diag( 1 , ..., n ) is a diagonal matrix of eigenvalues, which is approximated by k-th order polynomials ofL. Wu et al. (2019) Since the number of steps corresponds to the polynomial order of the Laplacian, as the polynomial order grows, the distance walker travelled changes slower and slower(the distribution becomes smooth), resulting in low efficiency and duplicated filters.

In the next section, we will introduce the ballistic walk method that, instead of walking at classical diffusion speed, the walker is able to reach an average distance ⇠ Ck in k steps walking.

Different from classical diffusive transportation, this method enables us to collect correlation faster.

Contributions We summarize our contributions as fourfold.

(1) We discuss the over-smooth of traditional GCN and propose the ballistic graph neural network.

(2) We show the ballistic walk is a faster transportation comparing to classical random walk based methods.

(3) We use the ballistic walk as feature extraction, and ballistic graph neural network achieves promising performance using fewer parameters comparing to random walk based feature extraction.

(4) We introduce noise to the coin space during ballistic transportation.

The perturbed ballistic walk transports slower and is able to collect correlation within a reasonable distance region.

Thus the perturbed ballistic walk is a better representation comparing to pure ballistic filters.

In the following, we will focus on the regular graph to demonstrate the ballistic graph neural network, where image, video and speech data are represented.

The ballistic walk algorithm consists of two parts, a walker in the position space H spatial and a coin in the coin space H c .

Thus the walker is described using states in Hibert space H spatial ⌦ H c .

Let the walker initially be at the state | i 0 = |i, ji p ⌦ s 0 , where s 0 is normally symmetric state in H c .

In analogy to the classical random walk, the next state of the walker can be expressed by In this paper, we consider the ballistic walk on a regular two-dimensional graph.

The coin space H c consists of four states: |#i, |"i, | i, |!i, represents move up, down, left and right for the next step.

The spatial space H spatial consists N states representing the walker's position, where N is the number of nodes.

The notation |ni denotes an orthonormal basis for H spatial and hn| is the Hermitian conjugate of the state.

For a finite-dimensional vector space, the inner product hn 0 |ni is nn 0 and the outer product |n 0 i hn| equals to a matrix in R N ⇥N .

The probability stay on the node |i, ji is

Pseudo-code of our method is given in Algorithm 1.

Algorithm 1: Ballistic walk on 2D regular graph Result: The walker's state after K steps start from (i, j)

Methods In the last section, we introduce the ballistic walk on 2D regular graph.

Next, we use ballistic walk as feature extraction layer and learn graph representations.

The experiment is constructed as follows.

Taking MNIST classification as an example.

First, we take the non-zero pixels as the start points of the ballistic walk.

The ballistic distributions at different time steps of digital 0 are shown in Figure 5 .

The stacked ballistic feature layer is then fully connected to a set of hidden units with relu activation.

The final layer's width is the number of classes(for MNIST is 10) with softmax activation for class prediction(shown in Figure 4 ).

The Hadamard matrix H is 1 p 2  1 1 1 1 and the initial state is 0 = 1j/2 |"i + 1/2 |#i 1j/2 | i 1/2 |!i.

Figure 6 and 7 show the difference in diffusion between the random walk based diffusion and the ballistic diffusion on a 28 ⇥ 28 grid starting from the center.

Comparing to the classical random walk, the ballistic walk shows cohesive behaviour and transports faster.

The comparison between the speed is shown in Figure 8 .

The diffusive classical walk's distances at time = 15 and time = 20 center around the same range.

The ballistic walk's differences are more significant, which means collecting different information.

Comparing to classical random walk, the ballistic walk has a speed of ⇠ C. This linear transportation behaviour enables the filters to collect correlation on the graph more efficiently.

As shown in Figure 8 , the distance for a diffusive walk at time = 25 is around taking an 8-step ballistic walk.

Defferrard et al. (2016) considers 25 steps diffusive filters to approximate a 5 ⇥ 5 kernel with 10 feature maps(10 hidden units).

For comparison, we take an 8-step-ballistic kernel with the same number of feature maps.

The feature maps are then fully connected to 10/32 units and then connected to 10 units for classification.

The notations are denoted as Ball10 and Ball32.

Table 1 : Results on MNIST dataset using Ballistic filters with K = 8 compared with traditional diffusion-based graph convolutional network with K = 25 in Defferrard et al. (2016) .

Baselines We compare our approaches with the following baselines: For iterating over all the vertices of the graph, the authors generate a random walk |W vi = t| for every node, and then use it to update representations.

• Graph Attention Networks (GAT) : Veličković et al. (2017) assigns different weights to different nodes in a neighbourhood.

The graph attentional layer changes the weight distribution on the neighbourhood nodes.

• Manifold Regularization (ManiReg): Belkin et al. (2006) brings together ideas from the theory of regularization in reproducing kernel Hilbert spaces, manifold learning and spectral methods.

In the paper, their propose data-dependent geometric regularization method based on graph Laplacian.

• Graph Convolutional Network (GCN): Kipf & Welling (2016) conducts the following layer-wise propagation in hidden layers using random walk based method(

The final perceptron layer for classication is defines as:

• Graph Learning-Convolutional Networks(GLCN): Jiang et al. (2019) contains one graph learning layer, several graph convolution layers and one final perceptron layer.

The layer-wise propagation rule is:

In the last section, we introduce the ballistic walk, which transports faster than the diffusive classical walk.

By selecting the ballistic filters up to K = 8, we reach 97% and use 1/3 parameters comparing to spline method using classical diffusive filters.

This suggests ballistic filters are able to collect correlation more efficiently comparing to random walk based Laplacian filters.

Figure 10 shows the transportation behaviour of different kinds of filters.

There exists two phases: 'trapped to diffusive' phase and 'diffusive to ballistic' phase.

The laplacian-based filters can be regarded as the up-bound filters of the trapped to diffusive phase(the orange points) and obey the p steps law.

As the steps grow, the filters are inefficient.

As shown in the Figure 10 , the filters are repeatedly sampling the region with distance< 10 as the steps grow up to 70 steps, this means the filtered information can be very similar, leading to invalid feature layer.

The ballistic filters lie at the up-bound of the 'diffusive to ballistic' phase(the blue points), the linear propagation ensures gathering the long correlation information in a relatively small number of steps.

However, linear transportation also brings drawbacks:

• Sparse sampling at the mid-distance region: as shown in Figure 10 (figure in the figure) , for a 35-steps walk, the points from distance = 6 to distance = 13 enables the ballistic filters better interpret the long correlation.

However, the distance intervals between the ballistic filters are relatively sparse, and this can result in the missing of correlation.

•

Beyond the boundary: the linear ballistic transportation makes the walker go beyond the boundary (for our case the distance is 14).

With the same number of the steps, the ballistic walk travels to the boundary line(shown in Figure 11 ).

Is there a way to generate filters that can collect the correlation within distance < 14 area while circumventing cumbersome classical diffusion?

In other words, we are interested in generating filters with a transportation speed between ballistic and classical diffusion.

By controlling the speed of the filters, we circumvent going beyond the boundary and make all our filters localized between the regions with restricted distance(denoted as the diffusive to ballistic phase in Figure 10 ).

In the ballistic diffusion, we use Hadamard transformation on the coin space, The Hadamard operator (SU (2)) helps spilt the state in the coin space and finally leads to linear ballistic transportation.

However, as mentioned in the last section, we are interested in generating filters lines between ballistic and classical phase so that we can circumvent the boundary and slow-transportation problem.

In this section, we introduce the de-coherence scheme to perturb ballistic transportation by adding a noise term to the Hadamard operation at every step.

This noisy perturbation results in the de-coherence of ballistic filters and thus slows down the transportation.

We want our filters have a diffusion distance in a reasonable region(a < Distance < b).

However, the ballistic filters' distances increase with steps.

The filters are not capable to dense sampling some specific regions.

By selecting different randomness and steps, we can generate filters localized in a bounded area.

The noisy Hadamard can be written as Table 2 : Diffusion rate with different randomness Table 3 shows the accuracy with different perturbed filters(↵ = 0, 0.05, 0.10, 0.15, 0.20).

= 2 ⇥ R ⇥ ⇡↵ denotes the randomness in the coin space, R is a random number between 0 and 1.

The corresponding transportation speed is shown in table 2 and Figure 12 .

As the ↵ increases to 0.20, the speed drops to 0.323.

↵ is a controller of the diffusion speed, as ↵ becomes larger, the ballistic tranportation will finally evolve to the classical diffusive couterpart.

After taking randomized operations, the accuracy can be improved.

In other words, by using filters from the perturbed ballistic walk, we are now able to dense sample the 'meaningful regions' and avoid the shallow sampling and slow transportation problem by selecting the step and the randomness of the ballistic walk.

The 'meaningful regions' are denoted as blue and yellow in Figure 10 figure) .

In our model, we fix the first eight filters as the pure ballistic filters without perturbation.

We then select different filters from perturbed filters.

The model architecture is shown in Figure 13 .

The input signals are first passed to 25 different feature maps using the selected filters.

We then apply the convolutional operation and average pooling on the feature maps.

After a fully connected layer with 512 hidden units, the network is connected to 10 units for the classification task.

We select our filters up to the distance<14 regions(⇠ 25 steps for ballistic duffusion), and this ensures the filters gather the correlation information within reasonable regions.

1-8 3,6,9,12 3,6,9,12 3,6,9,12 3,6,9,12,15 99.39±0.09

1 -8 3,6,9,12 3,6,9,12 9,12,15,18 12,15,18,21,24 the classification accuracy is around 99.11%, when we keep the first eight ballistic filters and use different filters with different randomness, the accuracy increases to 99.39%.

Our results show that the classification accuracy can be improved using a mixture of perturbed filters.

The ballistic walk filters can also be generalized to different coin operators.

Except using Hadamard and noisy Hadamard coin operator, we can also use a discrete Fourier operator(DFO) or Grover operator, the discrete Fourier operator is written:

... !

where ! = e 2⇡i/d is the dth root of unity, d is the degree of regular graph, and the Grover operator is: Table 4 .

Note that we usually select unitary operation in the coin space to keep the probability as a constant.

However, not every unitary operator results in ballistic transportation, the Grover operator will localize near the start point as the steps grow(shown in Figure 14) , however, they all have a speed-up effect comparing to the classical diffusive filters.

The DFO and Grover operator have a transportation speed between the ballistic Structure DFO10 DFO32 Grover10 Grover32 Accuracy 97.32 97.58 97.26 97.39 Table 4 : Performance using Grover and DFO filters using NN structure in Figure 4 .

filters and diffusive filters, thus can be regarded as special forms of randomized filters.

The general Hadamard coin is balanced.

DFO retains the balanced properties on a general graph.

Every state in coin space is obtained with equal probability.

The Grover operator helps retain the symmetry of the signal and is permutation symmetric.

The Grover operator is not a balanced coin because the probability(weight in our case) does not change its propagating directions(p = (1 2/d) 2 ).

In this paper, we introduced a generalization of graph neural network: ballistic graph neural network.

We started from the speed problem of the traditional diffusive kernel and tackle this problem from the perspective of transportation behaviour.

We showed the linear transportation behaviour of ballistic filters and introduced the de-coherence scheme to adjust the filters' speed.

Compared with diffusive filters, the ballistic filters achieve similar accuracy using fewer of the parameters.

Besides, we showed that the efficiency of the ballistic filters could be improved by controlling transportation behaviour.

Compared to the random walk method, we used two operators: the coin operator and shift operator to control the walker, and thus controlled the information the walker gathers.

Our pipeline provides a new perspective for efficient extracting the graphical information using diffusion-related models.

Future work can investigate these two directions:

The Network Structure.

In this paper, we use simplified architecture to demonstrate the concept of the ballistic walk, the layers are limited to 5 layers, and we use traditional average pooling.

More layers can be added to improve particular accuracy, and more sophisticated pooling methods can be introducedDefferrard et al. (2016) .

Other techniques like dropout can also be employed to improve accuracy.

The Ballistic Filter.

De-coherence can also be introduced into the shift operator.

In other words, we can use perturbed shifted operator, and thus we introduce randomness in the spatial domain.

We can also try different unitary operators in the coin space or change the initial state of the walker.

The extension to general graphs can be generalized by adding self-loops to the nodes and thus make the graph regular.

The ballistic filters are inspired by two-dimensional quantum walk.

The quantum coherence effect guarantees fast ballistic transportation.

The different states in the coin space can be regarded as the independent state from spatial behaviour, for example, the spin of fermions or the polarization of light.

More information about the quantum walk can be found at Childs et al. (2003) .

Why introducing ballistic filters results in better performance?

We here offer a conjecture from the perspective of signal processing using one-dimensional condition.

The classical diffusion in the one-dimensional case has the shape of:

and the frequency part can be written as:ĝ

The g(x) can be regarded as a gaussian low pass filter.

For a gaussian high-pass filter, the spatial distribution is: Makandar & Halalli (2015) hg

The long time probability distribution of ballistic walk is: Luo & Xue (2015) P (x) = P 0 + ae

(12) Figure 15 shows the distribution of gaussian high pass filter and the cumulative distribution of 24th and 25step of ballistic diffusion.

These two distributions have a similar shape while the ballistic distribution has steeper edges resulted from fast transportation.

The ballistic filters' capability to collect the long-time probability means it can act as a high-pass filter with different sizes.

The size of the filters depends on the walking steps.

Figure 16 shows the ballistic diffusion with a pulse signal from t = ⌧ to t = ⌧ .

The orange dashed line is an approximated shape of ballistic transportation of the leftmost signal(t = ⌧ ), and the blue dashed line corresponds to t = ⌧ .

The width of the approximated shape is related to the walking steps.

For random walk based diffusive transportation after certain steps of diffusion, the region from t = ⌧ to t = ⌧ have a gaussian shape since it is sum of gaussian distribution with centers range from t = ⌧ to t = ⌧ .

The classical diffusion acts like a blur filter(low pass filter).

For ballistic diffusion, the shape of the pulse signal from t = ⌧ to t = ⌧ evolves to a 'valley' shape and thus, the ballistic diffusion is similar to a high pass filter.

@highlight

A new perspective on how to collect the correlation between nodes based on diffusion properties.

@highlight

A new diffusion operation for graph neural networks that does not require eigenvalue calculation and can propagate exponentially faster compared to traditional graph neural networks.

@highlight

The paper proposes to cope with the speed of diffusion problem by introducing ballistic walk.