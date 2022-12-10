We propose a framework to model the distribution of sequential data coming from a set of entities connected in a graph with a known topology.

The method is based on a mixture of shared hidden Markov models (HMMs), which are trained in order to exploit the knowledge of the graph structure and in such a way that the obtained mixtures tend to be sparse.

Experiments in different application domains demonstrate the effectiveness and versatility of the method.

Hidden Markov models (HMMs) are a ubiquitous tool for modelling sequential data.

They started by being applied to speech recognition systems and from there they have spread to almost any application one can think of, encompassing computational molecular biology, data compression, and computer vision.

In the emerging field of cognitive radars BID12 , for the task of opportunistic usage of the spectrum, HMMs have been recently used to model the occupancy of the channels by primary users BID21 .When the expressiveness of an HMM is not enough, mixtures of HMM have been adopted.

Roughly speaking, mixtures of HMMs can be interpreted as the result of the combination of a set of independent standard HMMs which are observed through a memoryless transformation BID5 BID8 BID22 BID13 .In many real-life settings one does not have a single data stream but an arbitrary number of network connected entities that share and interact in the same medium and generate data streams in real-time.

The streams produced by each of these entities form a set of time series with both intra and inter relations between them.

In neuroimaging studies, the brain can be regarded as a network: a connected system where nodes, or units, represent different specialized regions and links, or connections, represent communication pathways.

From a functional perspective, communication is coded by temporal dependence between the activities of different brain areas BID6 .

Also team sports intrinsically involve fast, complex and interdependent events among a set of entities (the players), which interact as a team BID24 BID23 .

Thus, understanding a player's behavior implies understanding the behavior of his teammates and opponents over time.

The extraction of knowledge from these streams to support the decision-making process is still challenging and the adaptation of HMM to this scenario is immature at best.

BID9 proposed a hybrid approach combining the Self-Organizing Map (SOM) and the HMM with applications in clustering, dimensionality reduction and visualization of large-scale sequence spaces.

Note that the model at each node is limited to a simple HMM.

Wireless local area networks have also been modeled with Markov-based approaches.

For instance, BID1 use HMMs for outlier detection in 802.11 wireless access points.

However, the typical approaches include a common HMM model for all nodes (with strong limited flexibility) and a HMM model per node, independent of the others (not exploring the dependencies between nodes).

BID4 built a sparse coupled hidden Markov model (SCHMM) framework to parameterize the temporal evolution of data acquired with functional magnetic resonance imaging (fMRI).

The coupling is captured in the transition matrix, which is assumed to be a function of the activity levels of all the streams; the model per node is still restricted to a simple HMM.In general, in networked data streams, the stream observed in each sensor is often modeled by HMMs but the intercorrelations between sensors are seldom explored.

The proper modeling of the intercorrelations has the potential to improve the learning process, acting as a regularizer in the learning process.

In here we propose to tackle this void, by proposing as observation model at each node a sparse mixture of HMMs, where the dependencies between nodes are used to promote the sharing of HMM components between similar nodes.

The proposed model finds intersections with distributed sparse representation and multitask learning.

Sparse representation/coding expresses a signal/model f , defined over some independent variable x, as a linear combination of a few atoms from a prespecified and overcomplete dictionary: DISPLAYFORM0 where φ i pxq are the atoms and only a few of the scalars s i are non-zero, providing a sparse representation of f pxq.

Distributed sparse representation BID2 ) is an extension of the standard version that considers networks with K nodes.

At each node, the signal sensed at the same node has its sparsity property because of its intracorrelation, while, for networks with multiple nodes, signals received at different nodes also exhibit strong intercorrelation.

The intra-and inter-correlations lead to a joint sparse model.

An interesting scenario in distributed sparse representation is when all signals/models share the common support but with different non-zero coefficients.

Multitask learning techniques rely on the idea that individual models for related tasks should share some structure (parameters or hyperparameters).

An interesting approach is based on adapting knowledge instead of data, handled by parameter transfer approaches, where parameters for different tasks/models are shared or constrained to be similar BID10 .Inspired by the formulation of equation 1, we propose to model the generative distribution of the data coming from each of the K nodes of a network as a sparse mixture obtained from a dictionary of generative distributions.

Specifically, we shall model each node as a sparse HMM mixture over a 'large' dictionary of HMMs, where each HMM corresponds to an individual atom from the dictionary.

The field knowledge about the similarities between nodes is summarized in an affinity matrix.

The objective function of the learning process promotes reusing HMM atoms between similar nodes.

We formalize now these ideas.

Assume we have a set of nodes Y " t1, ..., Ku connected by an undirected weighted graph G, expressed by a symmetric matrix G P R KˆK .

These nodes thus form a network, in which the weights are assumed to represent degrees of affinity between each pair of nodes (i.e. the greater the edge weight, the more the respective nodes like to agree).

The nodes y in the graph produce D-dimensional sequences X "`x p1q , ..., x pT q˘, x ptq P R D , whose conditional distribution we shall model using a mixture of HMMs: DISPLAYFORM0 where z P t1, ..., M u is a latent random variable, being M the size of the mixture.

Here, ppX|zq is the marginal distribution of observations of a standard first-order HMM: DISPLAYFORM1 where h "´h p0q , ..., h pT q¯, h ptq P t1, ..., Su, is the sequence of hidden states of the HMM, being S the number of hidden states.

Note that the factorization in equation 2 imposes conditional independence between the sequence X and the node y, given the latent variable z. This is a key assumption of this model, since this way the distributions for the observations in the nodes in Y share the same pool of HMMs, promoting parameter sharing among the K mixtures.

Given an observed sequence X and its corresponding node y P Y, the inference problem here consists in finding the likelihood ppX " X|y " yq (from now on, abbreviated as ppX|yq) as defined by equations 2 and 3.

The marginals ppX|zq of each HMM in the mixture may be computed efficiently, in OpS 2 T q time, using the Forward algorithm BID20 .

Then, ppX|yq is obtained by applying equation 2, so inference in the overall model is done in at most OpM S 2 T q time.

As we shall see, however, the mixtures we get after learning will often be sparse (see section 2.2.3), leading to an even smaller time complexity.

Given an i.i.d.

dataset consisting of N tuples pX i , y i q of sequences of observations X i " x p1q i , ..., x pTiq i¯a nd their respective nodes y i P Y, the model defined by equations 2 and 3, may be easily trained using the Expectation-Maximization (EM) algorithm BID7 , (locally) maximizing the usual log-likelihood objective: DISPLAYFORM0 where θ represents all model parameters, namely:1.

the mixture coefficients, α k :" pppz " 1|y " kq, ..., ppz " M |y " kqq, for k " 1, ..., K;2.

the initial state probabilities, π m :"´pph p0q " 1|z " mq, ..., pph p0q " S|z " mq¯, for m " 1, ..., M ;3.

the state transition matrices, A m , where A m s,u :" pph ptq " u|h pt´1q " s, z " mq, for s, u " 1, ..., S and m " 1, ..., M ; 4.

the emission probability means, µ m,s P R D , for m " 1, ..., M and s " 1, ..., S;5.

the emission probability diagonal covariance matrices, σ 2 m,s I, where σ 2 m,s P R`D, for m " 1, ..., M and s " 1, ..., S.Here, we are assuming that the observations are continuous and the emission probabilities ppx ptq |h ptq , zq are gaussian with diagonal covariances.

This introduces almost no loss of generality, since the extension of this work to discrete observations or other types of continuous emission distributions is straightforward.

The procedure to maximize equation 4 using EM is described in Algorithm 1, in section A.1.

The update formulas follow from the standard EM procedure and can be obtained by viewing this model as Bayesian network or by following the derivation detailed in section A.2.

However, the objective 4 does not take advantage of the known structure of G. In order to exploit this information, we introduce a regularization term, maximizing the following objective instead: DISPLAYFORM1 where λ ě 0 controls the relative weight of the two terms in the objective.

The expectations E z"ppz|y"j,θq rppz|y " k, θq have interesting properties which are enlightened and proven in Proposition 1.

Proposition 1.

Let P be the set of all M -nomial probability distributions.

We have:2.

arg min p,qPP E z"p rqpzqs " tp, q P P | @ m P t1, ..., M u : ppz " mqqpz " mq " 0u;3.

max p,qPP E z"p rqpzqs " 1; DISPLAYFORM2 Proof.

By the definition of expectation, DISPLAYFORM3 Statements 1 and 2 follow immediately from the fact that every term in the right-hand side of equation 6 is non-negative.

For the remaining, we note that equation 6 may be rewritten as the dot product of two M -dimensional vectors α p and α q , representing the two distributions p and q, respectively, and we use the following linear algebra inequalities to build an upper bound for this expectation: DISPLAYFORM4 where ||¨|| 1 and ||¨|| 2 are the L 1 and L 2 norms, respectively.

Clearly, the equality E z"p rqpzqs " 1 holds if p and q are chosen from the set defined in statement 4, where the distributions p and q are the same and they are non-zero for a single assignment of z. This proves statement 3.

Now, to prove statement 4, it suffices to show that there are no other maximizers.

The first inequality in equation 7 is transformed into an equality if and only if α p " α q , which means p " q. The second inequality becomes an equality when the L 1 and L 2 norms of the vectors coincide, which happens if and only if the vectors have only one non-zero component, concluding the proof.

Specifically, given two distinct nodes j, k P Y , if G j,k ą 0, the regularization term for these nodes is maximum (and equal to G j,k ) when the mixtures for these two nodes are the same and have one single active component (i.e. one mixture component whose coefficient is non-zero).

On the contrary, if G j,k ă 0, the term is maximized (and equal to zero) when the mixtures for the two nodes do not share any active components.

In both cases, though, we conclude from Proposition 1 that we are favoring sparse mixtures.

We see sparsity as an important feature since: 1 -it allows the coefficients to better capture the graph structure, which is usually sparse, and 2 -it leads to mixtures with fewer components, yielding faster inference and (possibly) less overfitting.

By setting λ " 0, we clearly get the initial objective 4, where inter-node correlations are modeled only via parameter sharing.

As λ Ñ 8, two interesting scenarios may be anticipated.

If G j,k ą 0, @j, k, all nodes will tend do share the same single mixture component, i.e. we would be learning one single HMM to describe the whole network.

If G j,k ă 0, @j, k, and M ě K, each node would tend to learn its own HMM model independently from all the others.

The objective function 5 can still be maximized via EM (see details in section A.3).

However, the introduction of the regularization term in the objective makes it impossible to find a closed form solution for the update formula of the mixture coefficients.

Thus, in the M-step, we need to resort to gradient ascent to update these parameters.

In order to ensure that the gradient ascent iterative steps lead to admissible solutions, we adopt the following reparameterization from BID25 : DISPLAYFORM5 Both learning and inference use the hmmlearn API, with the appropriate adjustments for our models.

For reproducibility purposes, we make our source code, pre-trained models and the datasets publicly available 2 .We evaluate four different models in our experiments: a model consisting of a single HMM (denoted as 1-HMM) trained on sequences from all graph nodes; a model consisting of K HMMs trained independently (denoted as K-HMM), one for each graph node; a mixture of HMMs (denoted as MHMM) as defined in this work (equations 2 and 3), trained to maximize the usual log-likelihood objective (equation 4); a mixture of HMMs (denoted as SpaMHMM) as the previous one, trained to maximize our regularized objective (equation 5).

Models 1-HMM, K-HMM and MHMM will be our baselines.

We shall compare the performance of these models with that of SpaMHMM and, for the case of MHMM, we shall also verify if SpaMHMM actually produces sparser mixtures in general, as argued in section 2.2.3.

In order to ensure a fair comparison, we train models with approximately the same number of possible state transitions.

Hence, given an MHMM or SpaMHMM with M mixture components and S states per component, we train a 1-HMM with « S ?

M states and a K-HMM with « S a M {K states per HMM.

We initialize the mixture coefficients in MHMM and SpaMHMM randomly, while the state transition matrices and the initial state probabilities are initialized uniformly.

Means are initialized using k-means, with k equal to the number of hidden states in the HMM, and covariances are initialized with the diagonal of the training data covariance.

Models 1-HMM and K-HMM are trained using the Baum-Welch algorithm, MHMM is trained using Algorithm 1 and SpaMHMM is trained using Algorithm 2.

However, we opted to use Adam (Kingma & Ba, 2014) instead of "vanilla" gradient ascent in the inner loop of Algorithm 2, since its per-parameter learning rate proved to be beneficial for faster convergence.

A typical Wi-Fi network infrastructure is constituted by K access points (APs) distributed in a given space.

The network users may alternate between these APs seamlessly, usually connecting to the closest one.

There is a wide variety of anomalies that may happen during the operation of such network and their automatic detection is, therefore, of great importance for future mitigation plans.

Some anomalous behaviors are: overloaded APs, failed or crashed APs, persistent radio frequency (RF) interference between adjacent APs, authentication failures, etc.

However, obtaining reliable ground truth annotation of these anomalies in entire wireless networks is costly and time consuming.

Under these circumstances, using data obtained through realistic network simulations is a common practice.

In order to evaluate our model in the aforementioned scenario, we have followed the procedure of BID0 , performing extensive network simulations using OMNeT++ 3 and INET 4 .

The former is a discrete event simulator for modeling communication networks that is used in many problem domains, including wireless networks.

The latter is a framework that provides detailed models for several communication protocols (TCP, IP, IEEE 802.11, etc.) .

Here, we used OMNeT++ and INET to generate traffic in a typical Wi-Fi network setup (IEEE 802.11 WLANg 2.4 GHz in infrastructure mode).Our network consists of 10 APs and 100 users accessing it.

The pairwise distances between APs are known and fixed.

Each sequence contains information about the traffic in a given AP during 10 consecutive hours and is divided in time slots of 15 minutes without overlap.

Thus, every sequence has the same length, which is equal to 40 samples (time slots).

Each sample contains the following 7 features: the number of unique users connected to the AP, the number of sessions within the AP, the total duration (in seconds) of association time of all current users, the number of octets transmitted and received in the AP and the number of packets transmitted and received in the AP.

Anomalies typically occur for a limited amount of time within the whole sequence.

However, in this experiment, we label a sequence as "anomalous" if there is at least one anomaly period in the sequence and we label it as "normal" otherwise.

One of the simulations includes normal data only, while the remaining include both normal and anomalous sequences.

In order to avoid contamination of normal data with anomalies that may occur simultaneously in other APs, we used the data of the normal simulation for training (150 sequences) and the remaining data for testing (378 normal and 42 anomalous sequences).In a Wi-Fi network, as users move in the covered area, they disconnect from one AP and they immediately connect to another in the vicinity.

As such, the traffic in adjacent APs may be expected to be similar.

Following this idea, the weight G j,k , associated with the edge connecting nodes j and k in graph G, was set to the inverse distance between APs j and k and normalized so that max j,k G j,k " 1.

As in BID0 , sequences were preprocessed by subtracting the mean and dividing by the standard deviation and applying PCA, reducing the number of features to 3.

For MHMM, we did 3-fold cross validation of the number of mixture components M and hidden states per component S. We ended up using M " 10 and S " 10.

We then used the same values of M and S for SpaMHMM and we did 3-fold cross validation for the regularization hyperparameter λ in the range r10´4, 1s.

The value λ " 10´1 was chosen.

We also cross-validated the number of hidden states in 1-HMM and K-HMM around the values indicated in section 3.

Every model was trained for 100 EM iterations or until the loss plateaus.

For SpaMHMM, we did 100 iterations of the inner loop on each M-step, using a learning rate ρ " 10´3.Models were evaluated by computing the average log-likelihood per sample on normal and anomalous test data, plotting the receiver operating characteristic (ROC) curves and computing the respective areas under the curves (AUCs).

FIG0 shows that the ROC curves for MHMM and SpaMHMM are very similar and that these models clearly outperform 1-HMM and K-HMM.

This is confirmed by the AUC and log-likelihood results in table 1.

Although K-HMM achieved the best (lowest) average log-likelihood on anomalous data, this result is not relevant, since it also achieved the worst (lowest) average log-likelihood on normal data.

This is in fact the model with the worst performance, as shown by its ROC and respective AUC.The bad performance of K-HMM likely results mostly from the small amount of data that each of the K models is trained with: in K-HMM, each HMM is trained with the data from the graph node (AP) that it is assigned to.

The low log-likelihood value of the normal test data in this model confirms that the model does not generalize well to the test data and is probably highly biased towards the training data distribution.

On the other hand, in 1-HMM there is a single HMM that is trained with the whole training set.

However, the same HMM needs to capture the distribution of the data coming from all APs.

Since each AP has its own typical usage profile, these data distributions are different and one single HMM may not be sufficiently expressive to learn all of them correctly.

MHMM and SpaMHMM combine the advantages and avoid the disadvantages of both previous models.

Clearly, since the mixtures for each node share the same pool of HMMs, every model in the mixture is trained with sequences from all graph nodes (at least in the first few training iterations).

Thus, at this stage, the models may capture behaviors that are shared by all APs.

As mixtures become sparser during training, mixture components specialize on the distribution of a few APs.

This avoids the problem observed in 1-HMM, which is unaware of the AP where a sequence comes from.

We would also expect SpaMHMM to be sparser and have better performance than MHMM, but only the former supposition was true (see figure 2) and by a small difference.

The inexistence of performance gains in SpaMHMM might be explained from the fact that this dataset consists of simulated data, where users are static (i.e. they do not swap between APs unless the AP where they are connected stops working) and so the assumption that closer APs have similar distributions does not bring any advantage.

The human body is constituted by several interdependent parts, which interact as a whole producing sensible global motion patterns.

These patterns may correspond to multiple activities like walking, eating, etc.

Here, we use our model to make short-time prediction of sequences of human joint positions, represented as motion capture (mocap) data.

The current state of the art methodologies use architectures based on deep recurrent neural networks (RNNs), achieving remarkable results both in short-time prediction BID11 BID18 and in long-term motion generation BID16 BID19 .Our experiments were conducted on the Human3.6M dataset from BID14 , which consists of mocap data from 7 subjects performing 15 distinct actions.

In this experiment, we have considered only 4 of those actions, namely "walking", "eating", "smoking" and "discussion".

There, the human skeleton is represented with 32 joints whose position is recorded at 50 Hz.

We build our 32x32-dimensional symmetric matrix G representing the graph G in the following sensible manner: DISPLAYFORM0 there is an actual skeleton connection between joints j and k (e.g. the elbow joint is connected to the wrist joint by the forearm); G j,k " 1, if joints j and k are symmetric (e.g. left and right elbows); G j,k " 0, otherwise.

We reproduced as much as possible the experimental setup followed in BID11 .

Specifically, we down-sampled the data by a factor of 2 and transformed the raw 3-D angles into an exponential map representation.

We removed joints with constant exponential map, yielding a dataset with 22 distinct joints, and pruned our matrix G accordingly.

Training was performed using data from 6 subjects, leaving one subject (denoted in the dataset by "S5") for testing.

We did 3-fold cross-validation on the training data of the action "walking" to find the optimal number of mixture components M and hidden states S for the baseline mixture MHMM.

Unsurprisingly, since this model can hardly overfit in such a complex task, we ended up with M " 18 and S " 12, which were the largest values in the ranges we defined.

Larger values are likely to improve the results, but the learning time would become too large to be practical.

For SpaMHMM, we used these same values of M and S and we did 3-fold cross validation on the training data of the action "walking" to fine-tune the value of λ in the range r10´4, 1s.

We ended up using λ " 0.05.

The number of hidden states in 1-HMM was set to 51 and in K-HMM it was set to 11 hidden states per HMM.

The same values were then used to train the models for the remaining actions.

Every model was trained for 100 iterations of EM or until the loss plateaus.

For SpaMHMM, we did 100 iterations of the inner loop on each M-step, using a learning rate ρ " 10´2.In order to generate predictions for a joint (node) y starting from a given prefix sequence X pref , we build the distribution ppX|X pref , yq (see details in section A.4) and we sample sequences from that posterior.

Our evaluation method and metric again followed BID11 .

We fed our model with 8 prefix subsequences with 50 frames each (corresponding to 2 seconds) for each joint from the test subject and we predicted the following 10 frames (corresponding to 400 ms).

Each prediction was built by sampling 100 sequences from the posterior and averaging.

We then computed the average mean angle error for the 8 sequences at different time horizons.

Results are in table 2.

Among our models (1-HMM, K-HMM, MHMM and SpaMHMM), SpaMHMM outperformed the remaining in all actions except "eating".

For this action in particular, MHMM was slightly better than SpaMHMM, probably due to the lack of symmetry between the right and left sides of the body, which was one of the prior assumptions that we have used to build the graph G. "Smoking" and "discussion" activities may also be highly non-symmetric, but results in our and others' models show that these activities are generally harder to predict than "walking" and "eating'.

Thus, here, the skeleton structure information encoded in G behaves as a useful prior for SpaMHMM, guiding it towards better solutions than MHMM.

The worse results for 1-HMM and K-HMM likely result from the same limitations that we have pointed out in section 3.1: each component in K-HMM is inherently trained with less data than the remaining models, while BID16 0.81 0.94 1.16 1.30 0.97 1.14 1.35 1.46 1.45 1.68 1.94 2.08 1.22 1.49 1.83 1.93 GRU sup.

BID18 0 Table 2 : Mean angle error for short-term motion prediction on Human3.6M for different actions and time horizons.

The results for ERD, LSTM-3LR, SRNN, GRU supervised and QuaterNet were extracted from BID19 .

Best results among our models are in bold, best overall results are underlined.

1-HMM does not make distinction between different graph nodes.

Extending the discussion to the state of the art solutions for this problem, we note that SpaMHMM compares favorably with ERD, LSTM-3LR and SRNN, which are all RNN-based architectures.

Moreover, ERD and LSTM-3LR were designed specifically for this task, which is not the case for SpaMHMM.

This is also true for GRU supervised and QuaterNet, which clearly outperform all remaining models, including ours.

This is unsurprising, since RNNs are capable of modeling more complex dynamics than HMMs, due to their intrinsic non-linearity and continuous state representation.

This also allows their usage for long-term motion generation, in which HMMs do not behave well due their linear dynamics.

However, unlike GRU supervised and QuaterNet, SpaMHMM models the probability distribution of the data directly, allowing its application in domains like novelty detection.

Regarding sparsity, the experiments confirm that the SpaMHMM mixture coefficients are actually sparser than those of MHMM, as shown in figure 2.

We may roughly divide the human body in four distinct parts: upper body (head, neck and shoulders), arms, torso and legs.

Joints that belong to the same part naturally tend to have coherent motion, so we would expect them to be described by more or less the same components in our mixture models (MHMM and SpaMHMM).

Since SpaMHMM is trained to exploit the known skeleton structure, this effect should be even more apparent in SpaMHMM than in MHMM.In order to confirm this conjecture, we have trained MHMM and SpaMHMM for the action "walking" using four mixture components only, i.e. M " 4, and we have looked for the most likely component (cluster) for each joint: DISPLAYFORM0 where C k is, therefore, the cluster assigned to joint k. The results are in figure 3 .

From there we can see that MHMM somehow succeeds on dividing the body in two main parts, by assigning the joints in the torso and in the upper body mostly to the red/'+' cluster, while those in the hips, legs and feet are almost all assigned to the green/'Ÿ' cluster.

Besides, we see that in the vast majority of the cases, symmetric joints are assigned to the same cluster.

These observations confirm that we have chosen Figure 3 : Assignments of joints to clusters in MHMM (left) and SpaMHMM (right).

The different symbols ('˝', 'Ÿ', 'x', '+') and the respective colors (blue, green, orange and red) on each joint represent the cluster that the joint was assigned to.the graph G for this problem in an appropriate manner.

However, some assignments are unnatural: e.g. one of the joints in the left foot is assigned to the red/'+' cluster and the blue/'˝' cluster is assigned to one single joint, in the left forearm.

We also observe that the distribution of joints per clusters is highly uneven, being the green/'Ÿ' cluster the most represented by far.

SpaMHMM, on the other hand, succeeds on dividing the body in four meaningful regions: upper body and upper spine in the green/'Ÿ' cluster; arms in the blue/'˝' cluster; lower spine and hips in the orange/'x' cluster; legs and feet in the red/'+' cluster.

Note that the graph G used to regularize SpaMHMM does not include any information about the body part that a joint belongs to, but only about the joints that connect to it and that are symmetric to it.

Nevertheless, the model is capable of using this information together with the training data in order to divide the skeleton in an intuitive and natural way.

Moreover, the distribution of joints per cluster is much more even in this case, what may also help to explain why SpaMHMM outperforms MHMM: by splitting the joints more or less evenly by the different HMMs in the mixture, none of the HMM components is forced to learn too many motion patterns.

In MHMM, we see that the green/'+' component, for instance, is the most responsible to model the motion of almost all joints in the legs and hips and also some joints in the arms and the red/'+' component is the prevalent on the prediction of the motion patterns of the neck and left foot, which are presumably very different.

In this work we propose a method to model the generative distribution of sequential data coming from nodes connected in a graph with a known fixed topology.

The method is based on a mixture of HMMs where its coefficients are regularized during the learning process in such a way that affine nodes will tend to have similar coefficients, exploiting the known graph structure.

We also prove that the proposed regularizer promotes sparsity in the mixtures, which is achieved through a fully differentiable loss function (i.e. with no explicit L 0 penalty term).

We evaluate the method's performance in two completely different tasks (anomaly detection in Wi-Fi networks and human motion forecasting), showing its effectiveness and versatility.

For future work, we plan to extend/evaluate the usage of SpaMHMM for sequence clustering.

This is an obvious extension that we did not explore thoroughly in this work, since its main focus was modeling the generative distribution of data.

In this context, extending the idea behind SpaMHMM to mixtures of more powerful generative distributions is also in our plans.

As is known, HMMs have limited expressiveness due to the strong independence assumptions they rely on.

Thus, we plan to extend these ideas to develop an architecture based on more flexible generative models for sequence modeling, like those attained using deep recurrent architectures.

After building the usual variational lower bound for the log-likelihood and performing the E-step, we get the following well-known objective:Jpθ, θ -q " ÿ z,H ppX, z, H|y, θ -q log ppX, z, H|y, θq,which we want to maximize with respect to θ and where θ -are the model parameters that were kept fixed in the E-step.

Some of the parameters in the model are constrained to represent valid probabilities, yielding the following Lagrangian: DISPLAYFORM0 Clearly, J r pθq´V r pθ, qq " 1 Nˆl og ppX|y, θq´E z,H"q " log ppX, z, H|y, θq qpz, Hq

1 N D KL pqpz, Hq||ppz, H|X, y, θqq ,which, fixing the parameters θ to some value θ -and minimizing with respect to q, yields the usual solution q˚pz, Hq " ppz, H|X, y, θ -q.

Thus, in the M-step, we want to find:arg max θ V r pθ, q˚q " arg max DISPLAYFORM0 ppX, z, H|y, θ -q log ppX, z, H|y, θq λ 2 ppX|y, θ -q ÿ j,k‰j G j,k E z"ppz|y"j,θq rppz|y " k, θqs DISPLAYFORM1 :" arg max DISPLAYFORM2 whereJpθ, θ -q is as defined in equation 10 and Rpθ, θ -q is our regularization weighted by the data likelihood, which is simply a function of the parameters α:Rpθ, θ -q " 1 2 ppX|y, θ -q ÿ j,k‰j G j,k E z"ppz|y"j,θq rppz|y " k, θqs DISPLAYFORM3 Now, we may build the Lagrangian as done in section A.2.

Since R only depends on the α's, the update equations for the remaining parameters are unchanged.

However, for α, it is not possible to obtain a closed form update equation.

Thus, we use the reparameterization defined in equation 8 and update the new unconstrained parameters β via gradient ascent.

We have: DISPLAYFORM4 DISPLAYFORM5 From equations 23 and 24, we see that the the resulting gradient ∇ α kJ r " ∇ α kJ`λ ∇ α k R is equal to some vector scaled by the joint data likelihood ppX|y, θ -q, which we discard since it only affects the learning rate, besides being usually very small and somewhat costly to compute.

This option is equivalent to using a learning rate that changes at each iteration of the outter loop of the algorithm.

Equation 8 yields the following derivatives: DISPLAYFORM6 Bα k,m Bβ k,l " 1 β k,m ą0´2 σ 1 pβ k,m q σpβ k,m q α k,m α k,l , for l ‰ m.

@highlight

A method to model the generative distribution of sequences coming from graph connected entities.

@highlight

The authors propose a method to model sequential data from multiple interconnected sources using a mixture of common pool of HMM's.