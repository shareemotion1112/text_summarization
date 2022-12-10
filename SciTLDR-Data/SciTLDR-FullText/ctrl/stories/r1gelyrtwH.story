Sparsely available data points cause a numerical error on finite differences which hinder to modeling the dynamics of physical systems.

The discretization error becomes even larger when the sparse data are irregularly distributed so that the data defined on an unstructured grid, making it hard to build deep learning models to handle physics-governing observations on the unstructured grid.

In this paper, we propose a novel architecture named Physics-aware Difference Graph Networks (PA-DGN) that exploits neighboring information to learn finite differences inspired by physics equations.

PA-DGN further leverages data-driven end-to-end learning to discover underlying dynamical relations between the spatial and temporal differences in given observations.

We demonstrate the superiority of PA-DGN in the approximation of directional derivatives and the prediction of graph signals on the synthetic data and the real-world climate observations from weather stations.

Modeling real world phenomena, such as climate observations, traffic flow, physics and chemistry simulation (Li et al., 2018; Geng et al., 2019; Long et al., 2018; de Bezenac et al., 2018; SanchezGonzalez et al., 2018; Gilmer et al., 2017) , is important but extremely challenging.

While deep learning has achieved remarkable successes in prediction tasks by learning latent representations from data-rich applications such as image recognition (Krizhevsky et al., 2012 ), text understanding (Wu et al., 2016) , and speech recognition , we are confronted with many challenging scenarios in modeling natural phenomena by deep neural networks when a limited number of observations are only available.

Particularly, the sparsely available data points cause substantial numerical error and the limitation requires a more principled way to redesign deep learning models.

Although many works have been proposed to model physics-simulated observations using deep learning, many of them are designed under the assumption that input is on a continuous domain.

For example, Raissi et al. (2017a; proposed Physics-informed neural networks (PINNs) to learn nonlinear relations between input (spatial-and temporal-coordinates (x, t)) and output simulated by a given PDE.

Since Raissi et al. (2017a; use the coordinates as input and compute derivatives based on the coordinates to represent a given PDE, the setting is only valid when the data are continuously observed over spatial and temporal space.

Under the similar direction of PINNs, Chen et al. (2015) proposed a method to leverage the nonlinear diffusion process for image restoration.

de Bezenac et al. (2018) incorporated the transport physics (advection-diffusion equation) with deep neural networks for forecasting sea surface temperature by extracting the motion field.

Lutter et al. (2019) introduced Deep Lagrangian Networks specialized to learn Lagrangian mechanics with learnable parameters.

Seo & Liu (2019) proposed a physicsinformed regularizer to impose data-specific physics equations.

In common, the methods in Chen et al. (2015) ; de Bezenac et al. (2018) ; Lutter et al. (2019) are not efficiently applicable to sparsely discretized input as only a small number of data points are available and continuous properties on given space are not easily recovered.

It is inappropriate to directly use continuous differential operators to provide local behaviors because it is hard to approximate the continuous derivatives precisely with the sparse points (Amenta & Kil, 2004; Luo et al., 2009; Shewchuk, 2002) .

Furthermore, they are only applicable when the specific physics equations are explicitly given and still hard to be generalized to incorporate other types of equations.

As another direction to modeling physics-simulated data, Long et al. (2018) proposed PDE-Net which uncovers the underlying hidden PDEs and predicts the dynamics of complex systems.

Ruthotto & Haber (2018) derived new CNNs: parabolic and hyperbolic CNNs based on ResNet (He et al., 2016) architecture motivated by PDE theory.

While Long et al. (2018) ; Ruthotto & Haber (2018) are flexible to uncover hidden physics from the constrained kernels, it is still restrictive to a regular grid where the proposed constraints on the learnable filters are easily defined.

Reasoning physical dynamics of discrete objects has been actively studied Battaglia et al., 2016; Chang et al., 2016) as the appearance of graph-based neural networks (Kipf & Welling, 2017; Santoro et al., 2017; Gilmer et al., 2017) .

Although these models can handle sparsely located data points without explicitly given physics equations, they are purely data-driven so that the physics-inspired inductive bias, exploiting finite differences, is not considered at all.

In contrast, our method consists of physics-aware modules allowing efficiently leveraging the inductive bias to learn spatiotemporal data from the physics system.

In this paper, we propose Physics-aware Difference Graph Networks (PA-DGN) whose architecture is motivated to leverage differences of sparsely available data from the physical systems.

The differences are particularly important since most of the physics-related dynamic equations (e.g., Navier-Stokes equations) handle differences of physical quantities in spatial and temporal space instead of using the quantities directly.

Inspired by the property, we first propose Spatial Difference Layer (SDL) to efficiently learn the local representations by aggregating neighboring information in the sparse data points.

The layer is based on Graph Networks (GN) as it easily leverages structural features to learn the localized representations and the parameters for computing the localized features are shared.

Then, the layer is connected with Recurrent Graph Networks (RGN) to be combined with temporal difference which is another core component of physics-related dynamic equations.

PA-DGN is applicable to various tasks and we provide two representative tasks; the approximation of directional derivatives and the prediction of graph signals.

• We tackle a limitation of the sparsely discretized data which cause numerical error to model the physical system by proposing Spatial Difference Layer (SDL) for efficiently exploiting neighboring information under the limitation of sparsely observable points.

• We combine SDL with Recurrent Graph Networks to build PA-DGN which automatically learns the underlying spatiotemporal dynamics in graph signals.

• We verify that PA-DGN is effective in approximating directional derivatives and predicting graph signals in synthetic data.

Then, we conduct exhaustive experiments to predict climate observations from land-based weather stations and demonstrate that PA-DGN outperforms other baselines.

In this section, we introduce the building module used to learn spatial differences of graph signals and describe how the module is used to predict signals in the physics system.

As approximations of derivatives in continuous domain, difference operators have been used as a core role to compute numerical solutions of (continuous) differential equations.

Since it is hard to derive closed-form expressions of derivatives in real-world data, the difference operators have been considered as alternative tools to describe and solve PDEs in practice.

The operators are especially important for physics-related data (e.g., meteorological observations) because the governing rules behind the observations are mostly differential equations.

Graph signal Given a graph G = (V, E) where V is a set of vertices V = {1, . . . , N v } and E a set of edges E ⊆ {(i, j)|i, j ∈ V} (|E| = N e ), graph signals on all nodes at time t are f (t) ∈ R Nv where f : V → R. In addition, graph signals on edges can be defined similarly, F (t) ∈ R Ne where F : E → R. Note that both signals can be multidimensional.

Gradient on graph The gradient (∇) of a function on nodes of a graph is represented by finite difference

where L 2 (V) and L 2 (E) denote vector spaces of node/edge functions, respectively.

The gradients on a graph provide finite differences of graph signals and they become corresponding edge (i, j) features.

Laplace-Beltrami operator Laplace-Beltrami operator (or Laplacian, ∆) in graph domain is defined as

This operator is usually regarded as a matrix form in other literature, L = D − A where A is an adjacency matrix and D = diag( j:j =i A ij ) is a degree matrix.

According to Crane (2018) , the gradient and Laplacian operator on the triangulated mesh can be discretized by incorporating the coordinates of nodes.

To obtain the gradient operator, the per-face gradient of each triangular face is calculated first.

Then, the gradient on each node is the area-weighted average of all its neighboring faces, and the gradient on edge (i, j) is defined as the dot product between the per-node gradient value and the direction vector e ij .

The Laplacian operator can be discretized with Finite Element Method (FEM):

where node j belongs to node i's immediate neighbors (j ∈ N i ) and (α j , β j ) are two opposing angles of the edge (i, j).

While the difference operators are generalized in Riemannian manifolds (Lai et al., 2013; Lim, 2015) , there are numerical error compared to those in continuous space and it can be worse when the nodes are spatially far from neighboring nodes because the connected nodes (j ∈ N i ) of i's node fail to represent local features around the i-th node.

Furthermore, the error is even larger if available data points are sparsely distributed (e.g., sensor-based observations).

In other words, since the difference operators are highly limited to immediate neighboring information only, they are unlikely to discover meaningful spatial variations behind the sparse observations.

To mitigate the limitation, we propose Spatial Difference Layer (SDL) which consists of a set of parameters to define learnable difference operators as a form of gradient and Laplacian to fully utilize neighboring information:

where w ij are the parameters tuning the difference operators along with the corresponding edge direction e ij .

Note that the two forms (Eq 1) are associated with edge and node features, respectively.

The subscript in ∇ w and ∆ w denotes that the difference operators are functions of the learnable parameters w. w (g) ij and w (l) ij are obtained by integrating local information as follow:

(2) While the standard difference operators consider two connected nodes only (i and j) for each edge (i, j), Eq 2 uses a larger view (h-hop) to represent the differences between i and j nodes.

Since Graph Networks (GN) are efficient networks to aggregate neighboring information, we use GN for g(·) function and w ij are edge features from the output of GN.

Note that Eq 2 can be viewed as a higher-order difference equation because nodes/edges which are multi-hop apart are considered.

w ij has a similar role of parameters in convolution kernels of CNNs.

For example, while the standard gradient operator can be regarded as an example of simple edge-detecting filters, the operator can be a sharpening filter if w (g1) ij = 1 and w (g2) ij = |Ni|+1 |Ni| for i node and the operators over each edge are summed.

In other words, by modulating w ij , it is readily extended to conventional kernels including edge detection or sharpening filters and even further complicated kernels.

On top of w ij , the difference forms in Eq 1 make an optimizing process for learnable parameters based on the differences instead of values themselves intentionally.

Thus, Eq 1 naturally provides the physics-inspired inductive bias which is particularly effective for modeling physics-related observations.

Furthermore, it is possible to increase the number of channels for w (g) ij and w (l) ij to be more expressive.

Figure 1 illustrates how the exemplary filters convolve the given graph signals.

Difference graph Once the modulated spatial differences (∇ w f (t), ∆ w f (t)) are obtained, they will be concatenated with the current signals f (t) to construct node-wise (z i ) and edge-wise (z ij ) features and the graph is called a difference graph.

Note that the difference graph includes all information to describe spatial variations.

Recurrent graph networks Given a snapshot (f (t), F (t)) of a sequence of graph signals, one difference graph is obtained and it is used to predict next graph signals.

While a non-linear layer can be used to combine the learned spatial differences to predict the next signals, it is limited to discover spatial relations only among the features in the difference graph.

Since many equations describing physics-related phenomena are non-static (e.g., Navier-Stokes equations), we adopt Recurrent Graph Networks (RGN) with a graph state G h as input to combine the spatial differences with temporal variations.

RGN returns a graph state (G * h = (h * (v) , h * (e) )) and next graph signal z * i and z * ij .

The update rule is described as follow:

) for all i ∈ V. z i is an aggregated edge attribute related to the node i.

where φ e , φ v are edge and node update functions, respectively, and they can be a recurrent unit (e.g., GRU cell).

Finally, the prediction is made through a decoder by feeding the graph signal, z * i and z * ij .

Figure 4: Gradients and graph structure of sampled points.

Left: the synthetic function is f 1 (x, y) = 0.1x 2 + 0.5y 2 .

Right: the synthetic function is f 2 (x, y) = sin(x) + cos(y).

Learning objective Letf andF denote predictions of the target node/edge signals.

PA-DGN is trained by minimizing the following objective:

For multistep predictions, L is summed as many as the number of predicting steps.

If only one type (node or edge) of signal is given, the corresponding term in Eq 3 is used to optimize the parameters in SDL and RGN simultaneously.

To investigate if the proposed spatial difference forms (Eq 1) can be beneficial to learning physicsrelated patterns, we use SDL to two different tasks: (1) approximate directional derivatives and (2) predict synthetic graph signals.

Figure 3: Directional derivative on graph

As we claimed in Section 2.3, the standard difference forms (gradient and Laplacian) on a graph can become easily inaccurate because they are susceptible to a distance of two points and variations of a given function.

To evaluate the applicability of the proposed SDL, we train SDL to approximate directional derivatives on a graph.

First, we define a synthetic function and its gradients on 2D space and sample 200 points (x i , y i ).

Then, we construct a graph on the sampled points by using k-NN algorithm (k = 4).

With the known gradient ∇f = ( ∂f ∂x , ∂f ∂y ) at each point (a node in the graph), we can compute directional derivatives by projecting ∇f to a connected edge e ij (See Figure 3) .

We compare against four baselines: (1) the finite gradient (FinGrad) (2) Multilayer Perceptron Layer (MLP) (3) Graph Networks (GN) (4) a different form of Eq 1 (One-w).

For the finite gradient ((f j − f i )/||x j − x i ||), there is no learnable parameter and it only uses two points.

For MLP, we feed (f i , f j , x i , x j ) as input to see whether learnable parameters can benefit the approximation or not.

For GN, we use distances of two connected points as edge features and function values on the points as node features.

The edge feature output of GN is used as a prediction for the directional derivative on the edge.

Finally, we modify the proposed form as (∇ w f ) ij = w ij * f j − f i .

GN and the modified form are used to verify the effectiveness of Eq 1.

Note that we define two synthetic functions (Figure 4 ) which have different property; (1) monotonically increasing from a center and (2) periodically varying.

Approximation accuracy As shown in Table 1 , the proposed spatial difference layer outperforms others by a large margin.

As expected, FinGrad provides the largest error since it only considers two points without learnable parameters.

It is found that the learnable parameters can significantly benefit to approximate the directional derivatives even if the input is the same (FinGrad vs. MLP).

Note that utilizing neighboring information is generally helpful to learn spatial variations properly.

However, simply training parameters in GN is not sufficient and explicitly defining difference, which is important to understand spatial variations, provides more robust inductive bias.

One important thing we found is that One-w is not effective as much as GN and it can be even worse than FinGrad.

It is because of its limited degree of freedom.

As implied in the form (∇ w f ) ij = w ij * f j − f i , only one w ij adjusts the relative difference between f i and f j , and this is not enough to learn whole possible linear combinations of f i and f j .

The unstable performance supports that the form of SDL is not ad-hoc but more effectively designed.

We evaluate PA-DGN on the synthetic data sampled from the simulation of specific convectiondiffusion equations, to provide if the proposed model can predict next signals of the simulated dynamics from observations on discrete nodes only.

For the simulated dynamics, we use an equation similar to the one in Long et al. (2018) .

where the index i is for pointing the i-th node whose coordinate is

Then, we uniformly sample 250 points in the above 2D space.

The task is to predict signal values of all points in the future M steps given observed values of first N steps.

For our experiments, we choose N = 5 and M = 15.

Since there is no a priori graph structure on sampled points, we construct a graph with k-NN algorithm (k = 4) using the Euclidean distance.

Figure 5 shows the dynamics and the graph structure of sampled points.

To evaluate the effect of the proposed SDL on the above prediction task, we cascade SDL and a linear regression model as our prediction model since the dynamics follows a linear partial differential equation.

We compare its performance with four baselines: (1) Vector Auto-Regressor (VAR); (2) Multi-Layer Perceptron (MLP); (3) StandardOP: the standard approximation of differential operators in Section 2.1 followed by a linear regressor; (4) MeshOP: similar to StandardOP but use the discretization on triangulated mesh in Section 2.2 for differential operators.

Prediction Performance Table 2 shows the prediction performance of different models measured with mean absolute error.

The prediction model with our proposed spatial differential layer outperforms other baselines.

All models incorporating any form of spatial differential operators (StandardOP, MeshOP and SDL) outperform those without spatial differential operators (VAR and MLP), showing that introducing spatial differences information inspired by the intrinsic dynamics helps prediction.

However, in cases where points with observable signal are sparse in the space, spatial differential operators approximated with fixed rules can be inaccurate and sub-optimal for prediction since the locally linear assumption which they are based on no longer holds.

Our proposed spatial differential layer, to the contrary, is capable of bridging the gap between approximated difference operators and accurate ones by introducing learnable coefficients utilizing neighboring information, and thus improves the prediction performance of the model.

We evaluate the proposed model on the task of predicting climate observations (Temperature) from the land-based weather stations located in the United States.

Data and task We sample the weather stations located in the United States from the Online Climate Data Directory of the National Oceanic and Atmospheric Administration (NOAA) and choose the stations which have actively measured meteorological observations during 2015.

We choose two geographically close but meteorologically diverse groups of stations: the Western and Southeastern states.

We use k-Nearest Neighbor (NN) algorithm (k = 4) to generate graph structures and the final adjacency matrix is A = (A k + A k )/2 to make it symmetric where A k is the output adjacency matrix from k-NN algorithm.

Our main task is to predict the next graph signals based on the current and past graph signals.

All methods we evaluate are trained through the objective (Eq 3) with the Adam optimizer and we use scheduled sampling (Bengio et al., 2015) for the models with recurrent modules.

We evaluate PA-DGN and other baselines on two prediction tasks, (1) 1-step and (2) multistep-ahead predictions.

Furthermore, we demonstrate the ablation study that provides how much the spatial derivatives are important signals to predict the graph dynamics.

We compare against the widely used baselines (VAR, MLP, and GRU) for 1-step and multistep prediction.

Then, we use Recurrent Graph Neural Networks (RGN) to examine how much the graph structure is beneficial.

Finally, we evaluate PA-DGN to verify if the proposed architecture (Eq 1) is able to improve the prediction quality.

Experiment results for the prediction task are summarized in Table 3 .

Overall, RGN and PA-DGN are better than other baselines and it implies that the graph structure provides useful inductive bias for the task.

It is intuitive as the meteorological observations are continuously changing over the space and time and thus, the observations of the closer stations from the i-th station are strongly related to observations at the i-th station.

PA-DGN outperforms RGN and the discrepancy comes from the fact that the spatial derivatives (Eq 1) we feed in PA-DGN are beneficial and this finding is expected because the meteorological signals at a certain point are a function of not only its previous signal but also the relative differences between neighbor signals and itself.

Knowing the relative differences among local observations is particularly essential to understand physics-related dynamics.

For example, Diffusion equation, which describes how physical quantities (e.g., heat) are transported through space over time, is also a function of relative differences of the quantities ( df dt = D∆f ) rather than values of the neighbor signals.

In other words, spatial differences are physics-aware features and it is desired to leverage the features as input to learn dynamics related to physical phenomena.

We further investigate if the modulated spatial derivatives (Eq 1) are effectively advantageous compared to the spatial derivatives defined in Riemannian manifolds.

First, RGN without any spatial derivatives is assessed for the prediction tasks on Western and Southeastern states graph signals.

Note that this model does not use any extra features but the graph signal, f (t).

Secondly, we add (1) StandardOP, the discrete spatial differences (Gradient and Laplacian) in Section 2.1 and (2) MeshOP, the triangular mesh approximation of differential operators in Section 2.2 separately as additional signals to RGN.

Finally, we incorporate with RGN our proposed Spatial Difference Layer.

Table 3 shows the contribution of each component.

As expected, PA-DGN provides much higher drops in MAE (3.56%,5.50%,8.51% and 8.73%,8.32%,5.49% on two datasets, respectively) compared to RGN without derivatives and the results demonstrate that the derivatives, namely, relative differences from neighbor signals are effectively useful.

However, neither RGN with StandardOP nor with MeshOP can consistently outperform RGN.

We also found that PA-DGN consistently shows positive effects on the prediction error compared to the fixed derivatives.

This finding is a piece of evidence to support that the parameters modulating spatial derivatives in our proposed Spacial Difference Layer are properly inferred to optimize the networks.

In this paper, we introduce a novel architecture (PA-DGN) that approximates spatial derivatives to use them to represent PDEs which have a prominent role for physics-aware modeling.

PA-DGN effectively learns the modulated derivatives for predictions and the derivatives can be used to discover hidden physics describing interactions between temporal and spatial derivatives.

A.1 SIMULATED DATA For the simulated dynamics, we discretize the following partial differential equation similar to the one in Long et al. (2018) to simulate the corresponding linear variable-coefficient convection-diffusion equation on graphs.

In a continuous space, we define the linear variable-coefficient convection-diffusion equation as:

, with

We follow the setting of initialization in Long et al. (2018) :

, where N = 9, λ k,l , γ k,l ∼ N 0, 1 50 , and k and l are chosen randomly.

We use spatial difference operators to approximate spatial derivatives:

, where s is the spatial grid size for discretization.

Then we rewrite (5) with difference operators defined on graphs:

, where

Then we replace the gradient w.r.t time in (8) with temporal discretization:

, where ∆t is the time step in temporal discretization.

Equation (12) is used for simulating the dynamics described by the equation (5).

Then, we uniformly sample 250 points in the above 2D space and choose their corresponding time series of u as the dataset used in our synthetic experiments.

We generate 1000 sessions on a 50 × 50 regular mesh with time step size ∆t = 0.01.

700 sessions are used for training, 150 for validation and 150 for test.

Here we provide additional details for models we used in this work, including model architecture settings and hyper-parameter settings.

Unless mentioned otherwise, all models use a hidden dimension of size 64.

• VAR: A vector autoregression model with 2 lags.

Input is the concatenated features of previous 2 frames.

The weights are shared among all nodes in the graph.

• MLP: A multilayer perceptron model with 2 hidden layers.

Input is the concatenated features of previous 2 frames.

The weights are shared among all nodes in the graph.

• GRU: A Gated Recurrent Unit network with 2 hidden layers.

Input is the concatenated features of previous 2 frames.

The weights are shared among all nodes in the graph.

• RGN: A recurrent graph neural network model with 2 GN blocks.

Each GN block has an edge update block and a node update block, both of which use a 2-layer GRU cell as the update function.

We set its hidden dimension to 73 so that it has the same number of learnable parameters as our proposed model PA-DGN.

• RGN(StandardOP): Similar to RGN, but use the output of difference operators in Section 2.1 as extra input features.

We set its hidden dimension to 73.

• RGN(MeshOP): Similar to RGN(StandardOP), but the extra input features are calculated using opeartors in Section 2.2.

We set its hidden dimension to 73.

• PA-DGN: Our proposed model.

The spatial derivative layer uses a message passing neural network (MPNN) with 2 GN blocks using 2-layer MLPs as update functions.

The forward network part uses a recurrent graph neural network with 2 recurrent GN blocks using 2-layer GRU cells as update functions.

The numbers of learnable parameters of all models are listed as follows:

The number of evaluation runs We performed 3 times for every experiment in this paper to report the mean and standard deviations.

Length of prediction For experiments on synthetic data, all models take first 5 frames as input and predict the following 15 frames.

For experiments on NOAA datasets, all models take first 12 frames as input and predict the following 12 frames.

Training hyper-parameters We use Adam optimizer with learning rate 1e-3, batch size 8, and weight decay of 5e-4.

All experiments are trained for a maximum of 2000 epochs with early stopping.

All experiments are trained using inverse sigmoid scheduled sampling with the coefficient k = 107.

Environments All experiments are implemented with Python3.6 and PyTorch 1.1.0, and are run with NVIDIA GTX 1080 Ti GPUs.

In this section, we evaluate the effect of 2 different graph structures on baselines and our models:

(1) k-NN: a graph constructed with k-NN algorithm (k = 4); (2) TriMesh: a graph generated with Delaunay Triangulation.

All graphs use the Euclidean distance.

Table 5 and Table 6 show the effect of different graph structures on the synthetic dataset used in Section 3.2 and the real-world dataset in Section 4.2 separately.

We find that for different models the effect of graph structures is not homogeneous.

For RGN and PA-DGN, k-NN graph is more beneficial to the prediction performance than TriMesh graph, because these two models rely more on neighboring information and a k-NN graph incorporates it better than a Delaunay Triangulation graph.

However, switching from TriMesh graph to k-NN graph is harmful to the prediction accuracy of RGN(MeshOP) since Delaunay Triangulation is a well-defined method for generating triangulated mesh in contrast to k-NN graphs.

Given the various effect of graph structures on different models, our proposed PA-DGN under k-NN graphs always outperforms other baselines using any graph structure.

Figure 7 provides the distribution of MAEs across the nodes of PA-DGN applied to the graph signal prediction task of the west coast region of the real-world dataset in Section 4.2.

As shown in the figure, nodes with the highest prediction error for short-term prediction are gathered in the inner part where the observable nodes are sparse, while for long-term prediction nodes in the area with a limited number of observable points no longer have the largest MAE.

This implies that PA-DGN can utilize neighboring information efficiently even under the limitation of sparsely observable points.

<|TLDR|>

@highlight

We propose physics-aware difference graph networks designed to effectively learn spatial differences to modeling sparsely-observed dynamics.