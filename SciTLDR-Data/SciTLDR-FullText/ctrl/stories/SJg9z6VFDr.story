Recently various neural networks have been proposed for irregularly structured data such as graphs and manifolds.

To our knowledge, all existing graph networks have discrete depth.

Inspired by neural ordinary differential equation (NODE) for data in the Euclidean domain, we extend the idea of continuous-depth models to graph data, and propose graph ordinary differential equation (GODE).

The derivative of hidden node states are parameterized with a graph neural network, and the output states are the solution to this ordinary differential equation.

We demonstrate two end-to-end methods for efficient training of GODE: (1) indirect back-propagation with the adjoint method; (2) direct back-propagation through the ODE solver, which accurately computes the gradient.

We demonstrate that direct backprop outperforms the adjoint method in experiments.

We then introduce a family of bijective blocks, which enables $\mathcal{O}(1)$ memory consumption.

We demonstrate that GODE can be easily adapted to different existing graph neural networks and improve accuracy.

We validate the performance of GODE in both semi-supervised node classification tasks and graph classification tasks.

Our GODE model achieves a continuous model in time, memory efficiency, accurate gradient estimation, and generalizability with different graph networks.

Convolutional neural networks (CNN) have achieved great success in various tasks, such as image classification (He et al., 2016) and segmentation (Long et al., 2015) , video processing (Deng et al., 2014) and machine translation (Sutskever et al., 2014) .

However, CNNs are limited to data that can be represented by a grid in the Euclidean domain, such as images (2D grid) and text (1D grid), which hinders their application in irregularly structured datasets.

A graph data structure represents objects as nodes and relations between objects as edges.

Graphs are widely used to model irregularly structured data, such as social networks (Kipf & Welling, 2016) , protein interaction networks (Fout et al., 2017) , citation and knowledge graphs (Hamaguchi et al., 2017) .

Early works use traditional methods such as random walk (Lovász et al., 1993) , independent component analysis (ICA) (Hyvärinen & Oja, 2000) and graph embedding (Yan et al., 2006) to model graphs, however their performance is inferior due to the low expressive capacity.

Recently a new class of models called graph neural networks (GNN) (Scarselli et al., 2008) were proposed.

Inspired by the success of CNNs, researchers generalize convolution operations to graphs to capture the local information.

There are mainly two types of methods to perform convolution on a graph: spectral methods and non-spectral methods.

Spectral methods typically first compute the graph Laplacian, then perform filtering in the spectral domain (Bruna et al., 2013) .

Other methods aim to approximate the filters without computing the graph Laplacian for faster speed (Defferrard et al., 2016) .

For non-spectral methods, the convolution operation is directly performed in the graph domain, aggregating information only from the neighbors of a node (Duvenaud et al., 2015; Atwood & Towsley, 2016) .

The recently proposed GraphSAGE (Hamilton et al., 2017 ) learns a convolution kernel in an inductive manner.

To our knowledge, all existing GNN models mentioned above have a structure of discrete layers.

The discrete structure makes it hard for the GNN to model continuous diffusion processes (Freidlin & Wentzell, 1993; Kondor & Lafferty, 2002) in graphs.

The recently proposed neural ordinary differential equation (NODE) ) views a neural network as an ordinary differential equation (ODE), whose derivative is parameterized by the network, and the output is the solution to this ODE.

We extend NODE from the Euclidean domain to graphs and propose graph ordinary differential equations (GODE), where the message propagation on a graph is modeled as an ODE.

NODEs are typically trained with adjoint method.

NODEs have the advantages of adaptive evaluation, accuracy-speed control by changing error tolerance, and are free-form continuous invertible models Grathwohl et al., 2018) .

However, to our knowledge, in benchmark image classification tasks, NODEs are significantly inferior to state-of-the-art discrete-layer models (error rate: 19% for NODE vs 7% for ResNet18 on CIFAR10) (Dupont et al., 2019; Gholami et al., 2019) .

In this work, we show this is caused by error in gradient estimation during training of NODE, and propose a memory-efficient framework for accurate gradient estimation.

We demonstrate our framework for free-form ODEs generalizes to various model structures, and achieves high accuracy for both NODE and GODE in benchmark tasks.

Our contribution can be summarized as follows:

1.

We propose a framework for free-form NODEs to accurately estimate the gradient, which is fundamental to deep-learning models.

Our method significantly improves the performance on benchmark classification (reduces test error from 19% to 5% on CIFAR10).

2.

Our framework is memory-efficient for free-form ODEs.

When applied to restricted-form invertible blocks, the model achieves constant memory usage.

3.

We generalize ODE to graph data and propose GODE models.

4.

We demonstrate improved performance on different graph models and various datasets.

There have been efforts to view neural networks as differential equations.

Lu (2017) viewed a residual network as a discretization of a differential equation and proposed several new architectures based on numerical methods in ODE solver.

Haber & Ruthotto (2017) proposed a stable architecture based on analysis of the ODE.

proposed neural ordinary differential equation (NODE), which treats the neural network as a continuous ODE.

NODE was later used in a continuous normalizing flow for generative models (Grathwohl et al., 2018 ).

There have been many studies on the training of NODE.

The adjoint method has long been widely used in optimal control (Stapor et al., 2018 ) and geophysical problems (Plessix, 2006) , and recently applied to ODE .

Dupont et al. (2019) proposed augmented neural ODEs to improve the expressive capacity of NODEs.

However, to our knowledge, none of the methods above discusses the inaccurate gradient estimation issue; empirical performances of NODE in benchmark classification tasks are significantly inferior to state-of-the-art discrete-layer models.

GNNs can be divided into two categories: spectral methods and non-spectral methods.

Spectral GNNs perform filtering in the Fourier domain of a graph, thus need information of the whole graph to determine the graph Laplacian.

In contrast, non-spectral GNNs only consider message aggregation around neighbor nodes, therefore are localized and generally require less computation (Zhou et al., 2018) .

We first briefly introduce several spectral methods.

Bruna et al. (2013) first introduced graph convolution in the Fourier domain based on the graph Laplacian, however the computation burden is heavy because of non-localized filters.

Henaff et al. (2015) incorporated a graph estimation procedure in spectral networks and parameterized spectral filters into a localized version with smooth coefficients.

Defferrard et al. (2016) used Chebyshev expansion to approximate the filters without the need to compute the graph Laplacian and its eigenvectors, therefore significantly accelerated the running speed.

Kipf & Welling (2016) proposed to use a localized first-order approximation of graph convolution on graph data and achieved superior performance in semi-supervised tasks for node classification.

Defferrard et al. (2016) proposed fast localized spectral filtering on graphs.

Non-spectral methods typically define convolution operations on a graph, only considering neighbors of a certain node.

MoNet (Monti, 2017) uses a mixture of CNNs to generalize convolution to graphs.

GraphSAGE (Hamilton et al., 2017 ) samples a fixed size of neighbors for each node for fast localized inference.

Graph attention networks (Veličković et al., 2017) learn different weights for different neighbors of a node.

The graph isomorphism network (GIN) (Xu et al., 2018) has a structure as expressive as the Weisfeiler-Lehman graph isomorphism test.

Invertible blocks are a family of neural network blocks whose forward function is a bijective mapping.

Therefore, the input to a bijective block can be accurately reconstructed from its outputs.

Invertible blocks have been used in normalizing flow (Rezende & Mohamed, 2015; Dinh, 2016; Kingma & Dhariwal, 2018; Dinh et al., 2014; Kingma et al., 2016) , where the model is required to be invertible in order to calculate the log-density of data distribution.

Later on, Jacobsen et al. (2018) used bijective blocks to build invertible networks.

Gomez et al. (2017) proposed to use invertible blocks to perform back propagation without storing activation, which achieves a memory-efficient network structure.

They were able to discard activation of middle layers, because each layer's activation can be reconstructed from the next layer with invertible blocks.

We first consider discrete-layer models with residual connection (He et al., 2016) , which can be represented as:

where x k is the states in the kth layer; f k (·) is any differentiable function whose output has the same shape as its input.

When we add more layers with shared weights, and let the stepsize in Eq. 1 go to infinitesimal, the difference equation turns into a neural ordinary differential equation (NODE) :

We use z(t) in the continuous case and x k in the discrete case to represent hidden states.

f (·) is the derivative parameterized by a network.

Note that a key difference between Eq. 1 and 2 is the form of f : in the discrete case, different layers (different k values) have their own function f k ; while in the continuous case, f is shared across all time t.

The forward pass of model with discrete layers can be written as:

where K is the total number of layers.

Then an output layer (e.g. fully-connected layer for classification) is applied on x K .

The forward pass of a NODE is:

where z(0) = input and T is the integration time, corresponding to number of layers K in the discrete case.

The transformation of states z is modeled as the solution to the NODE.

Then an output layer is applied on z(T ).

Integration in the forward pass can be performed with any ODE solver, such as the Euler Method, Runge-Kutta Method, VODE solver and Dopris Solver (Milne & Milne, 1953; Brown et al., 1989; Ascher et al., 1997) .

The adjoint method is widely used in optimal process control and functional analysis (Stapor et al., 2018; Pontryagin, 2018) .

We follow the method by .

Denote model parameters as θ, which is independent of time.

Define the adjoint as:

Figure 1: Comparison of two methods for back-propagation on NODE.

As in figure (a) , the ODE solver is discretized at points {t0, t1, ..., tN } during forward pass.

Black dashed curve shows hidden state solved in forward-time, denoted as z(t).

Figure (b) shows the adjoint method, red solid line shows the hidden state solved in reverse-time, denoted as h(t).

Ideally z(t) = h(t) and dashed curve overlaps with solid curve; however, the reverse-time solution could be numerically unstable, and causes z(t) = h(t), thus causes error in gradient.

Figure (c) shows the direct back-propagation through ODE solver.

In direct back-propagation, we save evaluation time points {t0, t1, ...tN } during forward pass; during backward pass, we re-build the computation graph by directly evaluating at the same time points.

In this way, z(ti) = h(ti).

Since the hidden state can be accurately reconstructed, the gradient can be accurately evaluated.

where L is the loss function.

Then we have

with detailed proof from optimization perspective in appendix F. Then we can perform gradient descent to optimize θ to minimize L. Eq. 6 is a reverse-time integration, which can be solved with any ODE solver .

To evaluate

, we need to determine z(t) by solving Eq. 2 reverse-time (Directly storing z(t) during forward pass requires a large memory consumption, because the continuous model is equivalent to an infinite-layer model).

To summarize, in the forward pass we solve Eq. 2 forward in time; in the backward pass, we solve Eq. 2 and 6 reverse in time, with initial condition determined from Eq. 5 at time T .

We give an intuition why the reverse-time ODE solver causes inaccurate gradient in adjoint methods.

The backward pass (Eq. 6) requires determining f (z(t), t, θ) and

, which requires determining z(t) by solving Eq. 2 reverse-time.

As shown in Fig. 1 (a,b) , the hidden state solved forward-time (z(t i )) and the hidden state solved reverse-time (h(t i )) may not be equal; this could be caused by the instability of reverse-time ODE, and is represented by the mismatch between z(t) (dashed curve) and h(t) (solid curve).

Error h(t) − z(t) will cause error in gradient dL dθ .

Proposition 1 For an ODE in the form

, denote the Jacobian of f as J f .

If this ODE is stable both in forward-time and reverse-time, then Re(λ i (J f )) = 0 ∀i, where λ i (J f ) is the ith eigenvalue of J f , and Re(λ) is the real part of λ.

Detailed proof is in appendix C. Proposition 1 indicates that if the Jacobian of the original system Eq. 2 has eigenvalues whose real-part are not 0, then either the reverse-time or forward-time ODE is unstable.

When |Re(λ)| is large, either forward-time or reverse-time ODE is sensitive to numerical errors.

This phenomenon is also addressed in Chang et al. (2018) .

This instability affects the accuracy of solution to Eq. 2 and 6, thus affects the accuracy of the computed gradient.

The adjoint method might be sensitive to numerical errors when solving the ODE in reverse-time.

To resolve this, we propose to directly back-propagate through the ODE solver.

As in Fig. 1(a) , the ODE solver uses discretization for numerical integration, evaluated at time points {t 0 , t 1 , ...t N }.

Fig. 1(c) demonstrates the direct back-propagation with accurate hidden states h(t i ), which can be achieved with two methods: (1) the activation z(t i ) can be saved in cache for back-prop, but requires huge memory; or (2) we can accurately reconstruct z(t i ) by re-building the computation graph directly at evaluated time points {t i }.

Since the model is evaluated at the same time points t i in forward-time, it's guaranteed that z(t i ) = h(t i ).

Therefore direct back-prop is accurate, regardless of the stability of Eq. 2.

Similar to the continuous case, we can define the adjoint with discrete time.

Then we have:

where a i is the adjoint for the ith step in discrete forward-time ODE solution.

Eq. 7 can be viewed as a numerical discretization of Eq. 6.

We show Eq. 6 can be derived from an optimization perspective.

Detailed derivations of Eq. 6-7 are in appendix E and F.

Algorithm 1: Algorithm for accurate gradient estimation in ODE solver for free-form functions Define model

, where f is a free-form function.

Denote integration time as T .

Forward (f, T, z 0 , tolerance):

Select initial step size h = h 0 (adaptively with adaptive step-size solver).

time points = empty list() While t < T : state = f.state dict(), accept step = F alse While Not accept step: f.load state dict(state) with grad disabled: z new, error estimate = step(f, z, t, h) If error estimate < tolerance: accept step = T rue z = z new, t = t + h, time points.append(t) else:

reduce stepsize h according to error estimate delete z new, error estimate and related local computation graph cache.save(time points) return z, cache Backward (f, T, z 0 , tolerance, cache):

Details of our method are summarized in Algorithm 1.

We discuss its properties below:

Summary of the algorithm (1) During forward pass, the solver performs a numerical integration, with the stepsize adaptively varying with error estimation.

(2) During forward pass, the solver outputs the integrated value, and the evaluation time points {t i }.

All middle activations are deleted to save memory.

(3) During backward pass, the solver re-builds the computation graph, by directly evaluating at saved time points, without adaptive searching.

(4) During backward pass, the solver performs a numerical version (Eq. 7) of reverse-time integration (Eq. 6).

Support for free-form continuous dynamics There's no constraint on the form of f .

Therefore, our algorithm is a generic method.

Memory consumption analysis (1) Suppose f has N f layers, the number of forward evaluation step is N t on average, and the evaluations to adaptively search for an optimal stepsize is K. A naive solver will take O(N f × N t × K), while our method consumes O(N f × N t ) because all middle activations are deleted during forward pass, and we don't need to search for optimal stepsize in backward pass.

(2) If we perform step-wise checkpoint method, where we only store z(t i ) for all t i , and compute the gradient ∂z(ti+1) ∂z(ti) for one t i at a time, then the memory consumption can be reduced to O(N f +N t ).

(3) Since the solver can handle free-form functions, it can also handle restricted form invertible block (see below).

In this case, we don't need to store z(t i ), and the memory consumption can reduce to O(N f ).

More memory-efficient solver with invertible blocks Restricting the form of f to invertible blocks (Gomez et al., 2017) allows for O(N f ) memory consumption.

For invertible blocks, input x is split into two parts (x 1 , x 2 ) of the same size (e.g. x has shape N × C, where N is batch size, C is channel number; we can split x into x 1 and x 2 with shape N × C 2 ).

The forward and inverse of a bijective block can be denoted as:

where the output of a bijective block is denoted as (y 1 , y 2 ) with the same size as (x 1 , x 2 ).

F and G are any differentiable neural networks, whose output has the same shape as the input.

ψ(α, β) is a differentiable bijective function w.r.t α when β is given; ψ −1 (α, β) is the inverse function of ψ.

Theorem 1 If ψ(α, β) is a bijective function w.r.t α when β is given, then the block defined by Eq. 8 is a bijective mapping.

Proof of Theorem 1 is given in appendix D. Based on this, we can apply different ψ functions for different tasks.

Since x can be accurately reconstructed from y, there's no need to store activations, hence is memory-efficient.

Details for back-prop without storing activation are in appendix B.

We first introduce graph neural networks with discrete layers, then extend to the continuous case and introduce graph ordinary differential equations (GODE).

As shown in Fig. 2 , a graph is represented with nodes (marked with circles) and edges (solid lines).

We assign a unique color to each node for ease of visualization.

Current GNNs can generally be represented in a message passing scheme (Fey & Lenssen, 2019) :

where x u k represents states of the uth node in the graph at kth layer and e u,v represents the edge between nodes u and v. N (u) represents the set of neighbor nodes for node u. ζ represents a differentiable, permutation invariant operation such as mean, max or sum.

γ (k) and φ (k) are differentiable functions parameterized by neural networks.

For a specific node u, a GNN can be viewed as a 3-stage model, corresponding to Eq. 9-11: (1) Message passing, where neighbor nodes v ∈ N (u) send information to node u, denoted by message (v,u) .

The message is generated from function φ(·), parameterized by a neural network.

(2) Message aggregation, where a node u aggregates all messages from its neighbors N (u), denoted as aggregation u .

The aggregation function ζ is typically permutation invariant operations such as mean and sum, because graphs are invariant to permutation.

(3) Update, where the states of a node are updated according to its original states x u k−1 and aggregation of messages aggregation u , denoted as γ(·).

We can convert a discrete-time GNN to continuous-time GNN by replacing f in Eq. 2 with the message passing process defined in Eq. 9 to 11, which we call graph ordinary differential equation (GODE).

A diagram of GODE is shown in Fig. 2 .

Because GODE is an ODE in nature, it can capture highly non-linear functions, thus has the potential to outperform its discrete-layer counterparts.

We demonstrate that the asymptotic stability of GODE could be related to the over-smoothing phenomena (Li et al., 2018 ).

It's demonstrated that graph convolution is a special case of Laplacian smoothing (Li et al., 2018) , which can be written as Y = (I − γD −1/2LD−1/2 )X where X and Y are the input and output of a graph-conv layer respectively,Ã = A + I where A is the adjacency matrix, andD is the corresponding degree matrix ofÃ, and γ is a positive scaling constant.

When modified from a discrete model to a continuous model, the continuous smoothing process is:

Since all eigenvalues of the symmetrically normalized Laplacian are real and non-negative, then all eigenvalues of the above ODE are real and non-positive.

Suppose all eigenvalues of the normalized Laplacian are non-zero.

In this case, the ODE has only negative eigenvalues, hence the ODE above is asymptotically stable (Lyapunov, 1992) .

Hence as time t grows sufficiently large, all trajectories are close enough.

In the experiments, this suggests if integration time T is large enough, all nodes (from different classes) will have very similar features, thus the classification accuracy will drop.

To evaluate our method on general NODE, we conducted experiments with a CNN-NODE on two benchmark image classification tasks (CIFAR10 and CIFAR100) (Krizhevsky et al., 2009 ).

We also evaluated our method on benchmark graph datasets, including 2 bioinformatic graph classification datasets (MUTAG and PROTEINS), 2 social network graph classification datasets (IMDB-BINRAY, REDDIT-BINARY) (Yanardag & Vishwanathan, 2015) , and 3 citation networks (Cora, CiteSeer and PubMed).

For graph classification tasks, different from the experiment settings in Xu et al. (2018) , we input the raw dataset into our models without pre-processing.

For node classification tasks, we performed transductive inference and strictly followed the train-validation-test split by Kipf & Welling (2016) , where less than 6% nodes are used as training examples.

Details of datasets are summarized in appendix A.

For image classification tasks, we directly modify a ResNet18 into its corresponding NODE model.

For each block, the function is

where f is a sequence of conv − bn − relu − conv − bn − relu layers.

f is the same as residual branch in ResNet, and it can be replaced with any free-form functions.

For tasks on graph datasets, GODE can be applied to any graph neural network by simply replacing f in Eq. 2 with corresponding structures (free-form functions), or replacing F, G in Eq. 8 with other structures (invertible blocks).

To demonstrate that GODE is easily generalized to existing structures, we used several different GNN architectures, including the graph convolutional network (GCN) (Kipf & Welling, 2016) , graph attention network (GAT) (Veličković et al., 2017) , graph network approximated with Chebyshev expansion (ChebNet) (Defferrard et al., 2016) , and graph isomorphism network (GIN) (Xu et al., 2018) .

For a fair comparison, we trained GNNs with different depths of layers (1-3 middle layers, besides an initial layer to transform data into specified channels, and a final layer to generate prediction), and reported the best results among all depths for each model structure.

On the same task, different models use the same hyper-parameters on model structures, such as channel number.

For graph classification tasks, we set the channel number of hidden layers as 32 for all models; for ChebNet, we set the number of hops as 16.

For node classification tasks, we set the channel number as 16 for GCN and ChebNet, and set number of hops as 3 for ChebNet; for GAT, we used 8 heads, and set each head as 8 channels.

For every GNN structure, we experimented with different number of hidden layers (1,2,3), and calculated the mean and variance of accuracy of 10 runs.

We compared the adjoint method and direct back-propagation on the same network, and demonstrated direct back-prop generates higher accuracy.

For CNN-NODE on classification tasks, we directly modify a ResNet18 into NODE18, and report resuls in Table.

1; for graph networks, we train a GODE model with a GCN to parameterize the derivative, and report results in Table.

2.

Empirical performance Direct back-propagation consistently outperformed the adjoint method for both tasks.

This result validates our analysis on the instability of the adjoint method, which is intuitively caused by the instability of the reverse-time ODE.

On image classification tasks, compared to adjoint method, our training method reduces error rate of NODE18 from 19% (37%) to 5%(23%) on CIFAR10 (CIFAR100).

Furthermore, NODE18 has the same number of parameters as ResNet18, but outperforms deeper networks such as ResNet101 on both datasets.

Our method also consistently outperforms the adjoint method on several benchmark graph datasets, as shown in Table.

2.

Robustness to ODE solvers We implemented adaptive ODE solvers of different orders, as shown in Table 1 .

HeunEuler, RK23, RK45 are of order 1, 2, 4 respectively, i.e., for each step forward in time f is evaluated 1, 2, 4 times respectively.

During inference, using different solvers is equivalent to changing model depth (without re-training the network): for discrete-layer models, it generally causes huge error; for continuous models, we observe only around 1% increase in error rate.

This suggests our method is robust to different orders of ODE solvers.

Support for free-form functions Our method supports NODE and GODE models with free-form functions; for example, f in NODE18 in Table.

1 is a free-form function.

We demonstrate that bijective blocks defined as Eq. 8 can be easily generalized: F and G are general neural networks, which can be adapted to different tasks; ψ(α, β) can be any differentiable bijective mapping w.r.t.

α when β is given.

We demonstrate two examples of ψ:

Results for different ψ are reported in Table 3 .

Note that we experimented with different depths and reported the best accuracy for each model, and performed a paired t-test on results from GODE and their discrete-layer counterparts.

Most GODE models outperformed their corresponding discretelayer models significantly, validating the effectiveness of GODE; different ψ functions behaved similarly on our node classification tasks, indicating the continuous-time model is more important than coupling function ψ.

We also validate the lower memory cost, with details in appendix B.

Results for different models on graph classification tasks are summarized in Table 4 .

We experimented with different structures, including GCN, ChebNet and GIN; for corresponding GODE models (marked with ODE), we tested both free-form (marked with "free") and invertible block (marked with "INV").

We performed paired t-test comparing GODE and its discrete-layer counterparts.

For most experiments, GODE models performed significantly better.

This indicates the continuous process model might be important for graph models.

For a NODE and GODE model, during inference, we test the influence of integration time.

Results are summarized in Table.

5.

When integration time is short, the network does not gather sufficient information from neighbors; when integration time is too long, the model is sensitive to over-smooth issue, as discussed in Sec. 4.2.

We observe accuracy drop in both cases.

We propose GODE, which enables us to model continuous diffusion process on graphs.

We propose a memory-efficient direct back-propagation method to accurately determine the gradient for general free-form NODEs, and validate its superior performance on both image classification tasks and graph data.

Furthermore, we related the over-smoothing of GNN to asymptotic stability of ODE.

Our paper tackles the fundamental problem of gradient estimation for NODE; to our knowledge, it's the first paper to improve accuracy on benchmark tasks to comparable with state-of-the-art discrete layer models.

It's an important step to apply NODE from theory to practice.

A DATASETS

We perform experiments on various datasets, including citation networks (Cora, CiteSeer, PubMed), social networks (COLLAB, IMDB-BINARY, REDDIT-BINARY), and bioinformatics datasets (MUTAG, PROTEINS).

Details of each dataset are summarized in Table 1 .

We explain the structure and conduct experiments for the invertible block here.

Structure of invertible blocks Structure of invertible blocks are shown in Fig. 1 .

We follow the work of Gomez et al. (2017) with two important modifications: (1) We generalize to a family of bijective blocks with different ψ in Eq. 8 in the main paper, while Gomez et al. (2017) restrict the form of ψ to be sum.

(2) We propose a parameter state checkpoint method, which enables bijective blocks to be called more than once, while still generating accurate inversion.

The algorithm is summarized in Algo.

2.

We write the pseudo code for forward and backward function as in PyTorch.

Note that we use "inversion" to represent reconstructing input from the output, and use "backward" to denote calculation of the gradient.

To reduce memory consumption, in the forward function, we only keep the outputs y 1 , y 2 and delete all other variables and computation graphs.

In the backward function, we first "inverse" the block to calculate x 1 , x 2 from y 1 , y 2 , then perform a local forward and calculate the gradient x1,x2] .

In this section we demonstrate that our bijective block is memory efficient.

We trained a GODE model with bijective blocks, and compared the memory consumption using our memory-efficient function as in Algo.

2 and a memory-inefficient method as in conventional backpropagation.

Results were measured with a batchsize of 100 on MUTAG dataset.

Depth Memory-efficient Conventional 10 2.2G 5.3G 20 2.6G 10.5G Table 2 : Memory consumption of bijective blocks.

"Conventional" represents storing activation of all layers in cache, "memory-efficient" represents our method in Algo.

2.

Results are summarized in Table.

2.

We measured the memory consumption with different depths, which is the number of ODE blocks.

When depth increases from 10 to 20, the memory by conventional methods increases from 5.3G to 10.5G, while our memory-efficient version only increases from 2.2G to 2.6G.

In theory, our bijective block takes O(1) memory, because we only need to store the outputs in cache, while deleting activations of middle layers.

For memory-efficient network, the slightly increased memory consumption is because states of F, G need to be cached; but this step takes up minimal memory compared to input data.

Algorithm 2: Function for memory-efficient bijective blocks

delete computation graphs generated by F and G return cache, y1, y2

Backward(cache, y1, y2, F , G, ψ,

Proposition 1 For an ODE in the form dz(t) dt = f (z(t), t), denote the Jacobian of f as J f .

If this ODE is stable both in forward-time and reverse-time, then Re(λ i (J f )) = 0 ∀i, where λ i (J f ) is the ith eigenvalue of J f , and Re(λ) is the real part of λ.

Proof Denote s = T − t, where T is the end time.

Notice that the reverse-time in t is equivalent to forward-time in s. Therefore, we have forward-time ODE:

and reverse-time ODE:

Therefore, we have λ(J f ) = −λ(J g ).

For both forward-time and reverse-time ODE to be stable, the eigenvalues of J need to have non-positive real part.

Therefore

The only solution is

D PROOF FOR THEOREM 1

Theorem 1 For bijective block whose forward and reverse mappings are defined as

If ψ(α, β) is a bijective function w.r.t α when β is given, then the block is a bijective mapping.

Proof To prove the forward mapping is bijective, it is equivalent to prove the mapping is both injective and surjective.

Injective We need to prove, if F orward(x 1 , x 2 ) = F orward(x 3 , x 4 ), then x 1 = x 3 , x 2 = x 4 .

The assumption above is equivalent to

Since ψ(α, β) is bijective w.r.t α when β is given, from Eq.(6), we have x 1 = x 3 .

Similarly, condition on x 1 = x 3 and Eq.(5), using bijective property of ψ, we have x 2 = x 4 .

Therefore, the mapping is injective.

Given y 1 , y 2 , we construct

Then for the forward function, given bijective property of ψ, apply F orward and Reverse defined in the proposition statement,

z 1 = ψ(x 1 , G(y 2 )) = ψ ψ −1 y 1 , G(y 2 ) , G(y 2 ) = y 1

Therefore we construct x 1 , x 2 s.t.

F orward(x 1 , x 2 ) = [y 1 , y 2 ].

Therefore the mapping is surjective.

Therefore is bijective.

We use a figure to demonstrate the computation graph, and derive the gradient from the computation graph.

The loss is L, forward pass is denoted with black arrows, gradient back-propagation is shown with red arrows.

We use p to denote each path from θ to L, corresponding to all paths in red that goes from L to θ.

In this section we derive the gradient of parameters in an neural-ODE model from an optimization perspective.

Then we extend from continuous cases to discrete cases.

Notations With the same notations as in the main paper, we use z(t) to denote hidden states z at time t. Denote parameters as θ, and input as x, target as y, and predicted output asŷ.

Denote the loss as J(ŷ, y).

Denote the integration time as 0 to T .

Problem setup The continuous model is defined to follow an ODE: dz(t) dt = f (z(t), t, θ), s.t.

z(0) = x

We assume f is differentiable, since f is represented by a neural network in our case.

The forward pass is defined as:ŷ

The loss function is defined as: J(ŷ, y) = J(z(T ), y) (14) We formulate the training process as an optimization problem:

For simplicity, Eq. 15 only considers one ODE block.

In the case of multiple blocks, z(T ) is the input to the next ODE block.

As long as we can derive dLoss dθ and dLoss dz(0) when dLoss dz(T ) is given, the same analysis here can be applied to the case with a chain of ODE blocks.

We use the Lagrangian Multiplier Method to solve the problem defined in Eq. 15.

For simplicity, only consider one example (can be easily extended to multiple examples cases), the Lagrangian is

Karush-Kuhn-Tucker (KKT) conditions are necessary conditions for an solution to be optimal.

In the following sections we start from the KKT condition and derive our results.

Derivative w.r.t.

λ At optimal point, we have δL δλ = 0.

Note that λ is a function of t, we derive the derivative from calculus of variation.

Consider a cotninuous and differentiable perturbation λ(t) on λ(t), and a scalar , L now becomes a function of ,

It's easy to check the conditions for Leibniz integral rule, and we can switch integral and differentiation, thus:

At optimal λ(t), dL d | =0 = 0 for all continuous differentiable λ(t).

Therefore, dz(t) dt − f (z(t), t, θ) = 0, ∀t ∈ (0, T )

From continuous to discrete case To derive corresponding results in discrete cases, we need to replace all integration with finite sum.

In discrete cases, the ODE condition turns into:

from Eq. 31, we can get:

Re-arranging terms we have:

which is the discrete version of Eq. 29.

Which also corresponds to our analysis in Eq. 10 and 11.

<|TLDR|>

@highlight

Apply ordinary differential equation model on graph structured data