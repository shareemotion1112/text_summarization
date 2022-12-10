Parameters are one of the most critical components of machine learning models.

As datasets and learning domains change, it is often necessary and time-consuming to re-learn entire models.

Rather than re-learning the parameters from scratch, replacing learning with optimization, we propose a framework building upon the theory of \emph{optimal transport} to adapt model parameters by discovering correspondences between models and data, significantly amortizing the training cost.

We demonstrate our idea on the challenging problem of creating probabilistic spatial representations for autonomous robots.

Although recent mapping techniques have facilitated robust occupancy mapping, learning all spatially-diverse parameters in such approximate Bayesian models demand considerable computational time, discouraging them to be used in real-world robotic mapping.

Considering the fact that the geometric features a robot would observe with its sensors are similar across various environments, in this paper, we demonstrate how to re-use parameters and hyperparameters learned in different domains.

This adaptation is computationally more efficient than variational inference and Monte Carlo techniques.

A series of experiments conducted on realistic settings verified the possibility of transferring thousands of such parameters with a negligible time and memory cost, enabling large-scale mapping in urban environments.

The quintessential paradigm in the machine learning pipeline consists of the stages of data acquisition and inference of the given data.

As data become plentiful, or as ones problem set become more diverse over time, it is common to learn new models tailored to the new data or problem.

Contrasting this conventional modeling archetype, we argue that it is often redundant to perform inference and re-learn parameters from scratch.

Such model adaptation procedures are indispensable in application domains such as robotics in which the operating environments change continuously.

For instance, if the model is represented as a Bayesian model, its distribution should be redetermined regularly to adjust for changes in new data.

In this paper, we focus on significantly improving the training time of building Bayesian occupancy maps such as automorphing Bayesian Hilbert maps (ABHMs) Senanayake et al. (2018) by transferring model parameters associated with a set of source datasets to a target dataset in a zero-shot fashion Isele et al. (2016) .

Despite having attractive theoretical properties and being robust, the main reason that hinders models such as ABHM being used in real-world settings is the run-time cost of learning thousands of parameters (main parameters and hyperparameters).

Moreover, these parameters not only vary across different places in the same environment, but also change over time.

We demonstrate domain adaptation of "geometry-dependent spatial features" of the ABHM model from a pool of source domains to the current target domain.

This is efficiently done using the theory of Optimal Transport Arjovsky et al. (2017) .

Since the proposed approach completely bypasses explicitly learning parameters of the Bayesian model using domain adaptation, this process can be thought of as "replacing parameter learning with domain adapatation."

The notation given in Table 1 will be used throughout the rest of the paper.

An occupancy model is typically a parameterized function which gives the probability of a given point in the environment being occupied.

For instance, having learned a function with parameters θ, it is possible to query y * = p(occupied|x * , θ) ∈ [0, 1] for anywhere in the space x * = (longitude, latitude) ∈ R 2 .

The parameters θ must be estimated from data gathered using a LIDAR sensor with labels y = {0, 1} = {free, hit}. The high level idea of ABHM is projecting LIDAR data into the reproducing kernel Hilbert space (RKHS)-a rich high dimensional feature space-and performing Bayesian logistic regression.

The occupancy probability of a point x is given by p(y|x) = sigmoid(

2 )) with weights w ∈ R, kernel hinged at spatial locations h ∈ R 2 , and width of the squaredexponential (SE) kernel γ ∈ R + .

As shown in Figure 1 , here, M SE kernels positioned at M sparial locations {h m } M m=1 are used to project 2D data into a M dimensional vector such that each kernel has more effect from data in its locality.

must be learned from LIDAR data.

Slightly abusing standard notations, in this paper,¯ands ymbols are used to represent the mean and dispersion parameters, respectively.

One of the most important parameters for later discussions is the location parameterh m ∈ R 2 .

Because of the intractable posterior, the parameters of the model are learned Senanayake et al. (2018) using variational inference through probabilistic programming Tran et al. (2017) .

In this section, we propose a framework for swiftly adapting thousands of parameter and hyperparameters of the Bayesian mapping model.

To adapt to domains, we require accurately pre-trained maps from which we can extract spatially relevant features.

In the context of our problem we must extract LIDAR scans (hits and free) with their corresponding model parameters {(h, θ)}. To simplify further discussions, as in Figure 1 , θ is defined as all parameters except the mean location parameterh.

We define source LIDAR data

with corresponding parameters learned from ABHM {θ

as the source atom.

The source is an environment small enough to be trainable with ABHM.

Having determined the source atom, our objective is to determine the new set of parameters

.

As illustrated in Figure 7 , we are looking for a nonlinear mapping technique to convert a source (S) to a target (T ).

We recognize this as an optimal transport (OT) problem.

In occupancy mapping, the probability measures are from LIDAR data.

For a new target dataset, we attempt to obtain the optimal coupling,

for a given D ∈ R N (S) ×N (T ) distance matrix (e.g. Euclidean distance between sourcetarget pairs) with the information entropy of P , r(P ) = − ij P ij log P ij .

This entropic regularization, commonly known as the Sinkhorn distance Cuturi (2013); Genevay et al. (2017) , enables solving the otherwise hard integer programming problem using an efficient iterative algorithm Sinkhorn and Knopp (1967) .

Here, λ controls the amount of regularization.

Having obtained the optimal coupling between source and target LIDAR, as illustrated in Figures 7 (b) -(c), now it is possible to transport corresponding source parameters θ (S) to the target domain.

This is done by associating the parameter positions with source samplesh (S) as a linear map x (S) Perrot et al. (2016) .

Note that all other θ (S) parameters associated with h (S) will also be transported.

This implicit transfer process is depicted in Figure 5 .

Since ABHM can only be executed in small areas due to the high computational cost, we learn individual ABHM maps for different areas and construct a dictionary of source atoms which we call a dictionary of atoms X (S) .

As a result, as depicted in Figure 2 , atoms from various domains will be transferred to the target.

The entire algorithm is given in Algorithm 1.

We used the Carla simulator Dosovitskiy et al. (2017) and KITTI benchmark dataset Geiger et al. (2013) for experiments.

A summary of datasets is listed in Table 5 .

We compared against vanilla variational inference Senanayake and Ramos (2017); Jaakkola and Jordan (1997) and variational inference with reparameterization trick Senanayake et al. (2018) ; Kingma and Welling (2013) .

Intra-domain and inter-domain adaptation Here we consider two paradigms: intradomain and inter-domain transfer.

In intra-domain transfer, the source atoms are generated from the first 10 frames of a particular dataset and parameters are transferred within the Figure 2 : A high-level overview of our method: Parameter Optimal Transport.

Training domains correspond to potentially independent, data-intensive, expensive, yet small-scale prelearned models.

After storing in a dictionary of atoms, representative data-space and modelparameter tuples from the pre-learned set of models, we find data-space correspondences using optimal transport maps via the ranking procedure.

These maps are then used to transport pre-learned parameters to out-of-sample test domains.

Our method is largely insensitive to data-space invariances between source training domains and test domains reducing knowledge loss during the transfer process.

same dataset.

In inter-domain transfer they are transferred to a completely new town.

Results are in Table 4 with 20% randomly sampled test LIDAR beams.

We consider two paradigms: intra-domain and inter-domain transfer.

In intra-domain transfer, the source atoms are generated from the first 10 frames of a particular dataset and parameters are transferred within the same dataset.

In inter-domain transfer they are transferred to a new town.

Results are in Table 4 Building instantaneous maps This experiment demonstrates performance of building instantaneous maps.

For this purpose, we use the two dynamic datasets: SimCarla and RealKITTI.

The source dictionary of atoms was prepared similar to the intra/inter-domain Table 2 : Instantaneous map building in dynamic environments.

Mean and SD are given.

We evaluated the test performance of our model using accuracy (ACC), area under ROC curve (AUC), and negative log-likelihood (NLL) Bishop (2006) .

The higher the ACC and AUC or lower the NLL, the better.

Figure 6 .

Table 2 shows the performance of transferring features extracted from each town to the dynamic datasets.

@highlight

We present a method of adapting hyperparameters of probabilistic models using optimal transport with applications in robotics