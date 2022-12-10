Origin-Destination (OD) flow data is an important instrument in transportation studies.

Precise prediction of customer demands from each original location to a destination given a series of previous snapshots helps  ride-sharing platforms to better understand their market mechanism.

However, most existing prediction methods ignore the network structure of OD flow data and fail to utilize the topological dependencies among related OD pairs.

In this paper, we propose a latent spatial-temporal origin-destination (LSTOD) model, with a novel convolutional neural network (CNN) filter to learn the spatial features of OD pairs from a graph perspective and an attention structure to capture their long-term periodicity.

Experiments on a real customer request dataset with available OD information from a ride-sharing platform demonstrate the advantage of LSTOD in achieving at least 6.5% improvement in prediction accuracy over the second best model.

Spatial-temporal prediction of large-scale network-based OD flow data plays an important role in traffic flow control, urban routes planning, infrastructure construction, and the policy design of ridesharing platforms, among others.

On ride-sharing platforms, customers keep sending requests with origins and destinations at each moment.

Knowing the exact original location and destination of each future trip allows platforms to prepare sufficient supplies in advance to optimize resource utilization and improve users' experience.

Given the destinations of prospective demands, platforms can predict the number of drivers transferring from busy to idle status.

Prediction of dynamic demand flow data helps ride-sharing platforms to design better order dispatch and fleet management policies for achieving the demand-supply equilibrium as well as decreased passenger waiting times and increased driver serving rates.

Many efforts have been devoted to developing traffic flow prediction models in the past few decades.

Before the rise of deep learning, traditional statistical and machine learning approaches dominate this field (Li et al., 2012; Lippi et al., 2013; Moreira-Matias et al., 2013; Shekhar & Williams, 2007; Idé & Sugiyama, 2011; Zheng & Ni, 2013) .

Most of these models are linear and thus ignore some important non-linear correlations among the OD flows.

Some other methods (Kwon & Murphy, 2000; Yang et al., 2013) further use additional manually extracted external features, but they fail to automatically extract the spatial representation of OD data.

Moreover, they roughly combine the spatial and temporal features when fitting the prediction model instead of dynamically modelling their interactions.

The development of deep learning technologies brings a significant improvement of OD flow prediction by extracting non-linear latent structures that cannot be easily covered by feature engineering. (Xingjian et al., 2015; Ke et al., 2017; Zhou et al., 2018) .

Zhang et al. (2016; modeled the whole city are as an entire image and employed residual neural network to capture temporal closeness.

and also learned traffic as images but they used LSTM instead to obtain the temporal dependency.

Yao et al. (2018b) proposed a Deep Multi-View Spatial-Temporal Network (DMVST-Net) framework to model both spatial and temporal relationships.

However, using standard convolution filters suffers from the problem that some OD flows covered by a receptive field of regular CNNs are not spatially important.

Graph-based neural net-works (GNN) (Kipf & Welling, 2016; Defferrard et al., 2016; Veličković et al., 2017) are proved to be powerful tools in modelling spatial-temporal network structures Li et al., 2017) .

However, none of these frameworks are directly applicable here since both the historical observations and responses to predict are vertex-level variables.

On the contrary, the OD flows we discuss in this paper are generated in the edge space by our definition.

The aim of this paper is to introduce a hierarchical Latent Spatial-Temporal Origin-Destination (LSTOD) prediction model to jointly extract the complex spatial-temporal features of OD data by using some well-designed CNN-based architectures.

Instead of modelling the dynamic OD networks as a sequence of images and applying standard convolution filters to capture their spatial information, we introduce a novel Vertex Adjacent Convolution Network (VACN) that uses an irregular convolution filter to cover the most related OD flows that share common vertecies with the target one.

The OD flows connected by common starting and/or ending vertexes, which may fall into different regions of the flow map, can be spatially correlated and topologically connected.

Moreover, for most ride-sharing platforms, a passenger is more likely to send a new request from the location where his/her last trip ends in.

To learn such sequential dependency, we introduce a temporal gated CNN (TGCNN) and integrate it with VACN by using the sandwich-structured STconv block in order to collectively catch the evolutionary mechanism of dynamic OD flow systems.

A periodically shifted attention mechanism is also used to capture the shift in the long-term periodicity.

Finally, the combined short-term and long-term representations are fed into the final prediction layer to complete the architecture.

Our contributions are summarized as follow:

• To the best of our knowledge, it is the first time that we propose purely convolutional structures to learn both short-term and long-term spatio-temporal features simultaneously from dynamic origin-destination flow data.

• We propose a novel VACN architecture to capture the graph-based semantic connections and functional similarities among correlated OD flows by modeling each OD flow map as an adjacency matrix.

• We design a periodically shift attention mechanism to model the long-term periodicity when using convolutional architecture TGCNN in learning temporal features.

• Experimental results on two real customer demand data sets obtained from a ride-sharing platform demonstrate that LSTOD outperforms many state-of-the-art methods in OD flow prediction, with 7.94% to 15.14% improvement of testing RMSE.

For a given urban area, we observe a sequence of adjacency matrices representing the OD flow maps defined on a fixed vertex set V , which indicates the N selected sub-regions from this area.

We let V = {v 1 , v 2 , . . .

, v N } denote the vertex set with v i being the i-th sub-region.

The shape of each grid v i could be either rectangles, hexagons or irregular sub-regions.

We define the dynamic OD flow maps as {O The goal of our prediction problem is to predict the snapshot O d,t+j ∈ R N ×N in the future time window (t + j) of day d given previously observed data, including both short-term and long-term historical information.

The short-term input data consists of the last p 1 timestamps from t + 1 − p 1 to t, denoted by

The long-term input data is made up of q time series {O d−ϕ,t+j−(p2−1)/2 , . . .

, O d−ϕ,t+j+(p2−1)/2 } of length p 2 for each previous day (d − ϕ) with the predicted time index (t + j) in the middle for ϕ = 1, . . .

, q. We let

,t+j+(p2−1)/2 } denote the entire long-term data.

Increasing p 1 and p 2 leads to higher prediction accuracy, but more training time.

We reformulate the set of short-term OD networks X 1 into a 4D tensor X 1 ∈ R N ×N ×p1×1 and concatenate the long-term snapshots X 2 into a 5D tensor

Each X 2,d−ϕ ∈ R N ×N ×p2×1 for day d − ϕ is a 4D tensor for ϕ = 1, . . .

, q. Therefore, we can finally define our latent prediction problem as follows:

where F (·, ·) represents the LSTOD model, which captures the network structures of OD flow data as well as the temporal dependencies in multiple scales.

A notation table is attached in the appendix.

In this section, we describe the details of our proposed LSTOD prediction model.

See Figure 1 for the architecture of LSTOD.

The four major novelties and functionalities of LSTOD model include

• an end-to-end framework LSTOD constructed by all kinds of CNN modules to process dynamic OD flow maps and build spatio-temporal prediction models;

• a novel multi-layer architecture VACN to extract the network patterns of OD flow maps by propagating through edge connections, which can not be covered by traditional CNNs;

• a special module ST-Conv block used to combine VACN and gated temporal convolutions to coherently learning the essential spatio-temporal representations;

• a periodically shifted attention mechanism which is well designed for the purely convolutional ST-Conv blocks to efficiently utilize the long-term information by measuring its similarities with short-term data.

Before introducing the detailed architecture of VACN, we want to discuss why directly applying standard CNNs to the OD flow map O d,t may disregard the connections between neighboring OD flows in the graph space first.

Figure 2 demonstrates that it fails to capture enough semantic information using the real-world example of ride demands.

For the OD flow starting from v 1 to v 2 , as illustrated in the upper sub-figure, the most related OD flows should be those with either origin or destination being v 1 or v 2 in the past few timestamps.

A certain part of the travel requests from v 1 to v 2 can be matched with some historical finished trips from a third-party location to V 1 by the same group of people, for example a trip from v 3 to v 1 .

However, as the lower-left sub-figure illustrates, some of the OD flows covered by a single CNN filter (the green square) such as the four corners of the kernel window may be topologically far away from the target one in the graph.

More generally, let's consider a target OD flow o To better understand the differences between our proposed VACN over graph edges with those vertex-based convolutions such as GCN (Kipf & Welling, 2016) or GAT (Veličković et al., 2017) , we introduce the concept of line graphs L(G) (Godsil & Royle, 2013) .

Each node in L(G) corresponds to an edge in G, while each individual edge in L(G) can be mapped to a pair of edges in G that connect to a joint vertex.

L(G) contains representing a feature vector at the edge from v i to v j .

The learned representation of each target edge is defined as the weighted sum of those from the same row or column in the adjacency matrix, and those from the row or column in the transposed adjacency matrix.

The output of the layer-wise VACN propagation for the OD flow from v i to v j is defined as follows:

where

are the weights to learn.

F(·) represents an elementwise activation function, such as ReLU(x) = max(0, x).

The first part of (2)

We use temporal gated CNN (TGCNN) instead of RNN-based architectures such as LSTMs to capture the temporal representation, which makes our LSTOD a pure convolutional architecture.

RNNs suffer from the problem of lower training efficiency, gradient instability, and time-consuming convergence.

Moreover, the high dimension of the spatial representations captured by VACN and a potential long temporal sequence length make RNNs notoriously difficult to train.

The CNNs is more flexible in handling various data structures and allows parallel computations to increase training speed.

TGCNN consists of two parts including one being a 3D convolution kernel applied to the spatial representations of all the N 2 OD flows along the time axis and the other being a gated linear unit (GLU) as the gate mechanism.

By employing VACN at each of r successive timeslots, we can build a 4D tensor Y = (Y d,t ) ∈ R N ×N ×r×ms which is then fed into the TGCNN operator:

where G * γ represents the TGCNN kernel and γ includes all the related parameters to learn.

2m t 3D convolutional kernels of size (1 × 1 × K) with zero paddings map the input Y into a single output

, which is split in half to obtain Y 1 and Y 2 with the same number of feature channels.

here denotes the element-wise Hadamard product.

We use the spatial-temporal convolutional block (ST-conv block) to jointly capture the spatialtemporal features of OD flow data, which has a 'sandwich'-structure architecture with a multi-layer VACN operator in the middle connecting the two TGCNNs.

The use of ST-Conv blocks have two major advantages.

First, the block can be stacked or extended based on the dimension and characteristic of the spatio-temporal input data.

Second, a temporal operation is applied before extracting the spatial information, which greatly reduces its computation complexity and memory consumption.

Both the input and output of each individual ST-Conv block are 4D tensors.

For the input of the l-th block Z l−1 ∈ R N ×N ×r l−1 ×c l−1 (Z 0 is the original the OD flow data with c 0 = 1), the output is computed as follows:

where G 1 * γ l 1 and G 0 * γ l 0 are two TGCNN kernels and S * θ l is a multi-layer VACN operator being applied to each timestamp.

The G 1 and G 2 operators from all the stacked ST-Conv blocks employ the same kernel sizes, which are (1 × 1 × K 1 ) and (1 × 1 × K 2 ), respectively.

Thus, we have r l = r l−1 − (K 1 + K 2 − 2).

After applying (r 0 − 1)/(K 0 + K 1 − 2) ST-Conv blocks to the input Z 0 , the temporal length is reduced from r 0 to 1.

When input being the short-term OD flow data

f here squeezes the captured 4D output into a 3D tensor by dropping the temporal axis.

The kernel sizes of each G 1 and

, respectively.

The detailed propagation of the l-th ST-Conv block is defined as

In addition to capturing the the spatial-temporal features from short-term OD flow data X 1 , we also take into account the long-term temporal periodicity due to the potential day-wise cycling patterns insides the OD flow data, decided by customer's travelling schedule and the city's traffic conditions.

Directly applying ST-Con blocks to an extremely long OD sequence which covers previous few days or weeks is computationally expensive.

Only a small set of timestamps from each previous day is necessary to capture the long-term periodicity.

As mentioned, we pick p 2 time intervals for each day d − ϕ when predicting the time window (d, t + j) considering the non-strict long-term periodicity.

This slight time shifting may be caused by unstable traffic peaks, holidays and extreme weather conditions among different days.

Inspired by the recently widely used attention mechanisms (Xu et al., 2015; Yao et al., 2018a; Liang et al., 2018) in spatial-temporal prediction problems, we propose a modified periodically shifted attention to work for the CNN-based ST-Conv blocks here.

Different from Yao et al. (2018a) that measures the similarity between hidden units of LSTMs, the attention here is built on the intermediate outputs of TGCNNs where the concatenations are then fed into a new set of ST-Conv blocks.

For each day (d − ϕ), we apply a series of L 1 ST-Conv blocks to the day-level p 2 -length sequential OD flow data X 2,d−ϕ and reduce the temporal length from p 2 to n 0 LT .

Each block contains two TGCNN layers with the same kernel size 1

and the propagation rule of the l-th ST-Conv blocks is defined as:

with Z

Moreover, score(z

where W 1 , W 2 and v φ are learned projection matrices.

b s is the added bias term.

By assuming that

to build a new 4D day-wise time series Z 0 LT .

and finally apply another set of ST-Conv blocks to it to obtain the long-term spatial-temporal representations, which is denoted by Z LT ∈ R N ×N ×c LT .

c LT is the number of feature channels.

We concatenate the short-term and long-term spatial-temporal representations Z ST and Z LT together along the feature axis as Z = Z ST ⊕ Z LT ∈ R N ×N ×C , where C = c ST + c LT .

Then, Z is modified to a 2D tensor Z ∈ R N 2 ×C by flattening the first two dimensions while keeping the third one.

We apply a fully connected layer to the C feature channels together with an element-wise non-linear sigmoid function to get the final predictions for all the N 2 OD flows.

We normalize the original OD flow data in the training set to (0, 1) by Max-Min normalization and use 'sigmoid' activation for the final prediction layer to ensure that all predictions fall into (0, 1).

The upper and lower bounds are saved and used to denormalize the predictions of testing data to get the actual flow volumes.

We use L 2 loss to build the objective loss during the training.

The model is optimized via Backpropagation Through Time (BPTT) and Adam (Kingma & Ba, 2014) .

The whole architecture of our model is realized using Tensorflow (Abadi et al., 2016) and Keras (Chollet et al., 2015) .

All experiments were run on a cluster with one NVIDIA 12G-memory Titan GPU.

In this section, we compare the proposed LSTOD model with some state-of-the-art approaches for latent traffic flow predictions.

All compared methods are classified into traditional statistical methods and deep-learning based approaches.

We use the demand flow data collected by a ride-sharing platform to examine the finite sample performance of OD flow predictions for each method.

We employ a large-scale demand dataset obtained from a large-scale ride-sharing platform to do all the experiments.

The dataset contains all customer requests received by the platform from 04/01/2018 to 06/30/2018 in two big cities A and B. Within each urban area, N = 50 hexagonal regions with the largest customer demands are selected to build the in total N 2 = 2500 OD flows.

Since one-layer VACN has a computation complexity O(N ) at each of the N 2 entries (globally O(N 3 )), the memory consumption highly increases as N gets bigger.

Considering the computation efficiency and storage limitation, we choose N = 50 here which can cover more than 80% of total demands and thus satisfy the operation requirement of the ride-sharing platform.

We split the whole dataset into two parts.

The data from 04/01/2018 to 06/16/2018 is used for model training, while the other part from 06/17/2017 to 06/30/2017 (14 days) serves as the testing set.

The first two and half months of OD flow data is further divided in half to the training and validation sets.

The size ratio between the two sets is around 4:1.

We let 30 min be the length of each timestamp and the value of the OD flow from v i to v j is the cumulative number of customer requests.

We make predictions for all the 50 2 OD flows in the incoming 1st, 2nd, and 3rd 30 minutes (i.e. t + 1, t + 2, t + 3) by each compared method, given the historical data with varied (p 1 , p 2 ) combinations.

For those model settings incorporating long-term information, we trace back q = 3 days to capture the time periodicity.

We use Rooted Mean Square Error to evaluate the performance of each method:

All state-of-the-art methods to be compared are listed as follows, some of which are modified to work for the OD flow data: (i) Historical average (HA): HA predicts the demand amount at each OD flow by the average value of the same day in previous 4 weeks. (ii) Autoregressive integrated moving average (ARIMA), (iii) Support Vector Machine Regression (SVMR), (iv) Latent Space Model for Road Networks (LSM-RN) (Deng et al., 2016) , (v) Dense + BiLSTM (DLSTM) (Altché & de La Fortelle, 2017) and (vi) Spatiotemporal Recurrent Convolutional Networks (SRCN) .

We only consider latent models in this paper, that is, no external covariates are allowed, while only the historical OD flow data is used to extract the hidden spatial-temporal features.

We tune the hyperparameters of each compared model to obtain the optimal prediction performance.

Specifically, we get (p * , d * , q * ) = (3, 0, 3) for ARIMA and k * = 15, γ * = 2 −5 , λ * = 10 for LSM-RN.

The optimal kernel size of the spatial based CNN kernel is 11 × 11 in SRCN model.

For fair comparison, we set the length of short-term OD flow sequence to be p 1 = 9 (i.e., previous 4.5 hours), q = 3 for long-term data which covers the three most recent days, and the length of each day-level time series p 2 = 5 to capture the periodicity shifting (one hour before and after the predicted time index).

More analysis of how variant (p 1 , p 2 ) combinations may affect the prediction performance of LSTOD will be studied latter.

A two-layer architecture is used by all the deep-learning based methods to extract the spatial patterns inside the OD flow data (L = 2 for both short-term and long-term VACN).

We set the filter size of all deep learning layers in both spatial and temporal space to be 64, including the VACNs and TGCNNs in our LSTOD model with c ST = c LT = 64.

Comparison with state-of-the-art methods.

Table 1 summarizes the finite sample performance for all the competitive methods and our LSTOD model in terms of the prediction RMSE on the testing data of city A. We compute the mean, variance, 25% quantile and 75% quantile of the 14 day-wise RMSE on the testing set.

LSTOD outperforms all other methods on the testing data with the lowest average day-wise RMSE (2.41/2.55/2.67), achieving (8.02%/7.94%/8.24%) improvement over the second best method 'SRCN'.

In general, deep-learning based models perform more stably than traditional methods with smaller variance and narrower confidence intervals.

Both 'ARIMA' and 'LSM-RN' perform poorly, even much worse than HA, indicating that they cannot capture enough short-term spatial-temporal features to get the evolution trend of OD flow data.

Among the deep learning models, LSTOD can more efficiently control the estimation variance compared to all the others.

This demonstrates the advantages of using our spatial-temporal architecture and long-term periodicity mechanism in modelling the dynamic evolution of OD flow networks.

The improvement becomes more significant when the time scale increases since the contributions of long-term periodicity are more emphasized as the short-term signals getting weaker.

The LSTOD performs even better on city B compared to the baseline methods since the long-term periodical pattern in city B may be more significant compared with that in city A. Detailed results about City B are summarized in Table 3 of the appendix.

Comparison with variants of LSTOD.

Table 2 shows the finite sample performance of our proposed model LSTOD and its different variants based on the demand data from city A.

We can see that the complete LSTOD model outperforms the short-term model and the one without using attention mechanisms in terms of smaller means and variances, and narrower confidence intervals.

It indicates that the attention design we use can capture the shift of the day-wise periodicity and extract more seasonal patterns to improve prediction accuracy.

The left sub-plot of Figure 3 compares the predictions by each model against the true values at two selected OD flows on the last 3 testing days in the 60-minute scales.

Two abnormal change points are marked by black circles.

The short-term model fails in this case because it ignores the long-term information.

The complete LSTOD model outperforms the one without using attention mechanisms since it can better catch the shift of the periodicity.

The right sub-plot visualizes the distribution curves of the day-wise RMSEs on the 14 testing days by each of the three compared models.

The lighter tail of the red curve demonstrates that the complete LSTOD is more predictive and stable especially for those unusual cases.

We do some more experiments to show how different hyperparameter configurations influence the model performance.

For more details, please refer to Section E of the appendix.

VACN VS standard local CNN.

In this experiment, we will show that our proposed VACN outperforms standard CNNs in capturing the hidden network structure of the OD flow data.

Given the model setting that N = 50 sub-regions of city A are used to build the dynamic OD flow matrices, the number of pixels being covered by VACN at each single snapshot is 50 × 4 = 200.

For fair comparison, the largest receptive filed of standard CNN should be no bigger than a 15 × 15 window, which includes 225 elements each time.

We consider five different kernel sizes including 5 × 5, 8 × 8, 11 × 11, 14 × 14, and 15 × 15.

We replace VCAN in our model by standard CNN in order to fairly compare its performance.

All hyper-parameters are fixed but only the kernel size of CNNs being changed.

Moreover, we only consider the baseline short-term mode of LSTOD model while ignoring the long-term information.

As Figure 4 illustrates, standard CNN achieves the best performance with the smallest RMSE = 2.64 on testing data with the filter size being 11 × 11, which is still higher than that using VACN with RMSE = 2.54.

Specifically, RMSE increases when the receptive field is larger than 11 × 11 since the spatial correlations among the most related OD flows (sharing common origin or destination nodes) are smoothed with the increase in the filter size ((8 × 2 − 1)/64 > (14 × 2 − 1)/196).

This experiment shows that treating the dynamic demand matrix as an image, and applying standard CNN filters does not capture enough spatial correlations among related OD flows without considering their topological connections from the perspective of graphs.

For more details, please refer to

Batch normalization is used in the VACN component.

The batch size in our experiment was set to 10, corresponding to 10 randomly sampled timestamps and all the 50 2 OD flows in each snapshot.

The initial liearning rate is set to be 10 −4 with a decay rate 10 −6 .

We use early stopping for all the deep learning-based methods where the training process is terminated when the RMSE over validation set has not been improved for 10 successive epochs.

The maximal number of epochs allowed is 100.

In this section, we want to explore how some important hyperparameters of input OD flow data, for example p 1 and p 2 , may affect the performance of our LSTOD model.

Figure 6 (b) compares RMSE on testing data by STOD model with different data settings.

Varied combinations of the short-term sequence length p 1 and the long-term day-level sequence length p 2 are studied.

We can see that the best performance is achieved as (p 1 , p 2 ) = (7, 5) with RMSE = 2.41.

Specifically, settings with different p 1 's under p 2 = 5 consistently outperform those under p 2 = 7.

It may demonstrate that the shift can usually be captured within a short time range, while a longer time sequence may smooth the significance.

Table 4 provides the detailed prediction results for each data setting.

@highlight

We propose a purely convolutional CNN model with attention mechanism to predict spatial-temporal origin-destination flows. 