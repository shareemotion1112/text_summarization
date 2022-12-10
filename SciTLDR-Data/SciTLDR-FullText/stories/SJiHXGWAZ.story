Spatiotemporal forecasting has various applications in neuroscience, climate and transportation domain.

Traffic forecasting is one canonical example of such learning task.

The task is challenging due to (1) complex spatial dependency on road networks, (2) non-linear temporal dynamics with changing road conditions and (3) inherent difficulty of long-term forecasting.

To address these challenges, we propose to model the traffic flow as a diffusion process on a directed graph and introduce Diffusion Convolutional Recurrent Neural Network (DCRNN), a deep learning framework for traffic forecasting that incorporates both spatial and temporal dependency in the traffic flow.

Specifically, DCRNN captures the spatial dependency using bidirectional random walks on the graph, and the temporal dependency using the encoder-decoder architecture with scheduled sampling.

We evaluate the framework on two real-world large-scale road network traffic datasets and observe consistent improvement of 12% - 15% over state-of-the-art baselines

Spatiotemporal forecasting is a crucial task for a learning system that operates in a dynamic environment.

It has a wide range of applications from autonomous vehicles operations, to energy and smart grid optimization, to logistics and supply chain management.

In this paper, we study one important task: traffic forecasting on road networks, the core component of the intelligent transportation systems.

The goal of traffic forecasting is to predict the future traffic speeds of a sensor network given historic traffic speeds and the underlying road networks.

This task is challenging mainly due to the complex spatiotemporal dependencies and inherent difficulty in the long term forecasting.

On the one hand, traffic time series demonstrate strong temporal dynamics.

Recurring incidents such as rush hours or accidents can cause nonstationarity, making it difficult to forecast longterm.

On the other hand, sensors on the road network contain complex yet unique spatial correlations.

FIG0 illustrates an example.

Road 1 and road 2 are correlated, while road 1 and road 3 are not.

Although road 1 and road 3 are close in the Euclidean space, they demonstrate very different behaviors.

Moreover, the future traffic speed is influenced more by the downstream traffic than the upstream one.

This means that the spatial structure in traffic is nonEuclidean and directional.

Traffic forecasting has been studied for decades, falling into two main categories: knowledgedriven approach and data-driven approach.

In transportation and operational research, knowledgedriven methods usually apply queuing theory and simulate user behaviors in traffic BID6 .

In time series community, data-driven methods such as Auto-Regressive Integrated Moving Average (ARIMA) model and Kalman filtering remain popular BID22 BID21 .

However, simple time series models usually rely on the stationarity assumption, which is often violated by the traffic data.

Most recently, deep learning models for traffic forecasting have been developed in BID23 ; BID35 , but without considering the spatial structure.

BID31 and BID24 model the spatial correlation with Convolutional Neural Networks (CNN), but the spatial structure is in the Euclidean space (e.g., 2D images).

BID4 , studied graph convolution, but only for undirected graphs.

In this work, we represent the pair-wise spatial correlations between traffic sensors using a directed graph whose nodes are sensors and edge weights denote proximity between the sensor pairs measured by the road network distance.

We model the dynamics of the traffic flow as a diffusion process and propose the diffusion convolution operation to capture the spatial dependency.

We further propose Diffusion Convolutional Recurrent Neural Network (DCRNN) that integrates diffusion convolution, the sequence to sequence architecture and the scheduled sampling technique.

When evaluated on realworld traffic datasets, DCRNN consistently outperforms state-of-the-art traffic forecasting baselines by a large margin.

In summary:• We study the traffic forecasting problem and model the spatial dependency of traffic as a diffusion process on a directed graph.

We propose diffusion convolution, which has an intuitive interpretation and can be computed efficiently.• We propose Diffusion Convolutional Recurrent Neural Network (DCRNN), a holistic approach that captures both spatial and temporal dependencies among time series using diffusion convolution and the sequence to sequence learning framework together with scheduled sampling.

DCRNN is not limited to transportation and is readily applicable to other spatiotemporal forecasting tasks.•

We conducted extensive experiments on two large-scale real-world datasets, and the proposed approach obtains significant improvement over state-of-the-art baseline methods.

We formalize the learning problem of spatiotemporal traffic forecasting and describe how to model the dependency structures using diffusion convolutional recurrent neural network.

The goal of traffic forecasting is to predict the future traffic speed given previously observed traffic flow from N correlated sensors on the road network.

We can represent the sensor network as a weighted directed graph G = (V, E, W ), where V is a set of nodes |V| = N , E is a set of edges and W ∈ R N ×N is a weighted adjacency matrix representing the nodes proximity (e.g., a function of their road network distance).

Denote the traffic flow observed on G as a graph signal X ∈ R N ×P , where P is the number of features of each node (e.g., velocity, volume).

Let X (t) represent the graph signal observed at time t, the traffic forecasting problem aims to learn a function h(·) that maps T historical graph signals to future T graph signals, given a graph G: DISPLAYFORM0

We model the spatial dependency by relating traffic flow to a diffusion process, which explicitly captures the stochastic nature of traffic dynamics.

This diffusion process is characterized by a random walk on G with restart probability α ∈ [0, 1], and a state transition matrix DISPLAYFORM0 is the out-degree diagonal matrix, and 1 ∈ R N denotes the all one vector.

After many time steps, such Markov process converges to a stationary distribution P ∈ R N ×N whose ith row P i,: ∈ R N represents the likelihood of diffusion from node v i ∈ V, hence the proximity w.r.t.

the node v i .

The following Lemma provides a closed form solution for the stationary distribution.

Lemma 2.1.

BID29 The stationary distribution of the diffusion process can be represented as a weighted combination of infinite random walks on the graph, and be calculated in closed form: DISPLAYFORM1 where k is the diffusion step.

In practice, we use a finite K-step truncation of the diffusion process and assign a trainable weight to each step.

We also include the reversed direction diffusion process, such that the bidirectional diffusion offers the model more flexibility to capture the influence from both the upstream and the downstream traffic.

The resulted diffusion convolution operation over a graph signal X ∈ R N ×P and a filter f θ is defined as: DISPLAYFORM0 where θ ∈ R K×2 are the parameters for the filter and D DISPLAYFORM1 for q ∈ {1, · · · , Q}where X ∈ R N ×P is the input, H ∈ R N ×Q is the output, {f Θq,p,,: } are the filters and a is the activation function (e.g., ReLU, Sigmoid).

Diffusion convolutional layer learns the representations for graph structured data and we can train it using stochastic gradient based method.

Relation with Spectral Graph Convolution Diffusion convolution is defined on both directed and undirected graphs.

When applied to undirected graphs, we show that many existing graph structured convolutional operations including the popular spectral graph convolution, i.e., ChebNet , can be considered as a special case of diffusion convolution (up to a similarity transformation).

Let D denote the degree matrix, and DISPLAYFORM2 2 be the normalized graph Laplacian, the following Proposition demonstrates the connection.

Proposition 2.2.

The spectral graph convolution defined as DISPLAYFORM3 with eigenvalue decomposition L = ΦΛΦ and F (θ) = K−1 0 θ k Λ k , is equivalent to graph diffusion convolution up to a similarity transformation, when the graph G is undirected.

Proof.

See Appendix C.

We leverage the recurrent neural networks (RNNs) to model the temporal dependency.

In particular, we use Gated Recurrent Units (GRU) BID9 , which is a simple yet powerful variant of RNNs.

We replace the matrix multiplications in GRU with the diffusion convolution, which leads to our proposed Diffusion Convolutional Gated Recurrent Unit (DCGRU).

DISPLAYFORM0 where X (t) , H (t) denote the input and output of at time t, r (t) , u (t) are reset gate and update gate at time t, respectively.

G denotes the diffusion convolution defined in Equation 2 and Θ r , Θ u , Θ C are parameters for the corresponding filters.

Similar to GRU, DCGRU can be used to build recurrent neural network layers and be trained using backpropagation through time.

In multiple step ahead forecasting, we employ the Sequence to Sequence architecture BID28 .

Both the encoder and the decoder are recurrent neural networks with DCGRU.

During training, we feed the historical time series into the encoder and use its final states to initialize the decoder.

The decoder generates predictions given previous ground truth observations.

At testing time, ground truth observations are replaced by predictions generated by the model itself.

The discrepancy between the input distributions of training and testing can cause degraded performance.

To mitigate this issue, we integrate scheduled sampling BID2 into the model, where we feed the model with either the ground truth observation with probability i or the prediction by the model with probability 1 − i at the ith iteration.

During the training process, i gradually decreases to 0 to allow the model to learn the testing distribution.

With both spatial and temporal modeling, we build a Diffusion Convolutional Recurrent Neural Network (DCRNN).

The model architecture of DCRNN is shown in FIG1 .

The entire network is trained by maximizing the likelihood of generating the target future time series using backpropagation through time.

DCRNN is able to capture spatiotemporal dependencies among time series and can be applied to various spatiotemporal forecasting problems.

Traffic forecasting is a classic problem in transportation and operational research which are primarily based on queuing theory and simulations BID12 .

Data-driven approaches for traffic forecasting have received considerable attention, and more details can be found in a recent survey paper BID30 and the references therein.

However, existing machine learning models either impose strong stationary assumptions on the data (e.g., auto-regressive model) or fail to account for highly non-linear temporal dependency (e.g., latent space model ; BID11 ).

Deep learning models deliver new promise for time series forecasting problem.

For example, in BID35 ; BID20 , the authors study time series forecasting using deep Recurrent Neural Networks (RNN).

Convolutional Neural Networks (CNN) have also been applied to traffic forecasting.

BID36 convert the road network to a regular 2-D grid and apply traditional CNN to predict crowd flow.

BID8 propose DeepTransport which models the spatial dependency by explicitly collecting upstream and downstream neighborhood roads for each individual road and then conduct convolution on these neighborhoods respectively.

Recently, CNN has been generalized to arbitrary graphs based on the spectral graph theory.

Graph convolutional neural networks (GCN) are first introduced in BID4 , which bridges the spectral graph theory and deep neural networks.

propose ChebNet which improves GCN with fast localized convolutions filters.

BID19 simplify ChebNet and achieve state-of-the-art performance in semi-supervised classification tasks.

BID26 combine ChebNet with Recurrent Neural Networks (RNN) for structured sequence modeling.

BID33 model the sensor network as a undirected graph and applied ChebNet and convolutional sequence model BID14 to do forecasting.

One limitation of the mentioned spectral based convolutions is that they generally require the graph to be undirected to calculate meaningful spectral decomposition.

Going from spectral domain to vertex domain, BID1 propose diffusion-convolutional neural network (DCNN) which defines convolution as a diffusion process across each node in a graph-structured input.

BID17 propose GraphCNN to generalize convolution to graph by convolving every node with its p nearest neighbors.

However, both these methods do not consider the temporal dynamics and mainly deal with static graph settings.

Our approach is different from all those methods due to both the problem settings and the formulation of the convolution on the graph.

We model the sensor network as a weighted directed graph which is more realistic than grid or undirected graph.

Besides, the proposed convolution is defined using bidirectional graph random walk and is further integrated with the sequence to sequence learning framework as well as the scheduled sampling to model the long-term temporal dependency.

We conduct experiments on two real-world large-scale datasets: In both of those datasets, we aggregate traffic speed readings into 5 minutes windows, and apply Z-Score normalization.

70% of data is used for training, 20% are used for testing while the remaining 10% for validation.

To construct the sensor graph, we compute the pairwise road network distances between sensors and build the adjacency matrix using thresholded Gaussian kernel BID27 .

All neural network based approaches are implemented using Tensorflow BID0 , and trained using the Adam optimizer with learning rate annealing.

The best hyperparameters are chosen using the Tree-structured Parzen Estimator (TPE) (Bergstra et al., 2011) on the validation dataset.

Detailed parameter settings for DCRNN as well as baselines are available in Appendix E. TAB1 shows the comparison of different approaches for 15 minutes, 30 minutes and 1 hour ahead forecasting on both datasets.

These methods are evaluated based on three commonly used metrics in traffic forecasting, including (1) Mean Absolute Error (MAE), (2) Mean Absolute Percentage Error (MAPE), and (3) Root Mean Squared Error (RMSE).

Missing values are excluded in calculating these metrics.

Detailed formulations of these metrics are provided in Appendix E.2.

We observe the following phenomenon in both of these datasets.

DISPLAYFORM0

(1) RNN-based methods, including FC-LSTM and DCRNN, generally outperform other baselines which emphasizes the importance of modeling the temporal dependency.

(2) DCRNN achieves the best performance regarding all the metrics for all forecasting horizons, which suggests the effectiveness of spatiotemporal dependency modeling.

(3) Deep neural network based methods including FNN, FC-LSTM and DCRNN, tend to have better performance than linear baselines for long-term forecasting, e.g., 1 hour ahead.

This is because the temporal dependency becomes increasingly non-linear with the growth of the horizon.

Besides, as the historical average method does not depend on short-term data, its performance is invariant to the small increases in the forecasting horizon.

Note that, traffic forecasting on the METR-LA (Los Angeles, which is known for its complicated traffic conditions) dataset is more challenging than that in the PEMS-BAY (Bay Area) dataset.

Thus we use METR-LA as the default dataset for following experiments.

To further investigate the effect of spatial dependency modeling, we compare DCRNN with the following variants: (1) DCRNN-NoConv, which ignores spatial dependency by replacing the transition matrices in the diffusion convolution (Equation 2) with identity matrices.

This essentially means the forecasting of a sensor can be only be inferred from its own historical readings; (2) DCRNN-UniConv, which only uses the forward random walk transition matrix for diffusion convolution; Figure 3 shows the learning curves of these three models with roughly the same number of parameters.

Without diffusion convolution, DCRNN-NoConv has much higher validation error.

Moreover, DCRNN achieves the lowest validation error which shows the effectiveness of using bidirectional random walk.

The intuition is that the bidirectional random walk gives the model the ability and flexibility to capture the influence from both the upstream and the downstream traffic.

To investigate the effect of graph construction, we construct a undirected graph by setting W ij = W ji = max(W ij , W ji ), where W is the new symmetric weight matrix.

Then we develop a variant of DCRNN denotes GCRNN, which uses the sequence to sequence learning with ChebNet graph convolution (Equation 5) with roughly the same amount of parameters.

TAB2 shows the comparison between DCRNN and GCRNN in the METR-LA dataset.

DCRNN consistently outperforms GCRNN.

The intuition is that directed graph better captures the asymmetric correlation between traffic sensors.

Figure 4 shows the effects of different parameters.

K roughly corresponds to the size of filters' reception fields while the number of units corresponds to the number of filters.

Larger K enables the model to capture broader spatial dependency at the cost of increasing learning complexity.

We observe that with the increase of K, the error on the validation dataset first quickly decrease, and then slightly increase.

Similar behavior is observed for varying the number of units.

To evaluate the effect of temporal modeling including the sequence to sequence framework as well as the scheduled sampling mechanism, we further design three variants of DCRNN: (1) DCNN: in which we concatenate the historical observations as a fixed length vector and feed it into stacked diffusion convolutional layers to predict the future time series.

We train a single model for one step ahead prediction, and feed the previous prediction into the model as input to perform multiple steps ahead prediction.

(2) DCRNN-SEQ: which uses the encoder-decoder sequence to sequence learning framework to perform multiple steps ahead forecasting.

(3) DCRNN: similar to DCRNN-SEQ except for adding scheduled sampling.

Figure 7: Visualization of learned localized filters centered at different nodes with K = 3 on the METR-LA dataset.

The star denotes the center, and the colors represent the weights.

We observe that weights are localized around the center, and diffuse alongside the road network.

FIG4 shows the comparison of those four methods with regards to MAE for different forecasting horizons.

We observe that: (1) DCRNN-SEQ outperforms DCNN by a large margin which conforms the importance of modeling temporal dependency.

(2) DCRNN achieves the best result, and its superiority becomes more evident with the increase of the forecasting horizon.

This is mainly because the model is trained to deal with its mistakes during multiple steps ahead prediction and thus suffers less from the problem of error propagation.

We also train a model that always been fed its output as input for multiple steps ahead prediction.

However, its performance is much worse than all the three variants which emphasizes the importance of scheduled sampling.

To better understand the model, we visualize forecasting results as well as learned filters.

FIG5 shows the visualization of 1 hour ahead forecasting.

We have the following observations: (1) DCRNN generates smooth prediction of the mean when small oscillation exists in the traffic speeds FIG5 ).

This reflects the robustness of the model.

(2) DCRNN is more likely to accurately predict abrupt changes in the traffic speed than baseline methods (e.g., FC-LSTM).

As shown in FIG5 (b), DCRNN predicts the start and the end of the peak hours.

This is because DCRNN captures the spatial dependency, and is able to utilize the speed changes in neighborhood sensors for more accurate forecasting.

Figure 7 visualizes examples of learned filters centered at different nodes.

The star denotes the center, and colors denote the weights.

We can observe that (1) weights are well localized around the center, and (2) the weights diffuse based on road network distance.

More visualizations are provided in Appendix F.

In this paper, we formulated the traffic prediction on road network as a spatiotemporal forecasting problem, and proposed the diffusion convolutional recurrent neural network that captures the spatiotemporal dependencies.

Specifically, we use bidirectional graph random walk to model spatial dependency and recurrent neural network to capture the temporal dynamics.

We further integrated the encoder-decoder architecture and the scheduled sampling technique to improve the performance for long-term forecasting.

When evaluated on two large-scale real-world traffic datasets, our approach obtained significantly better prediction than baselines.

For future work, we will investigate the following two aspects (1) applying the proposed model to other spatial-temporal forecasting tasks; (2) modeling the spatiotemporal dependency when the underlying graph structure is evolving, e.g., the K nearest neighbor graph for moving objects.

This research has been funded in part by NSF grants CNS-1461963, IIS-1254206, IIS-1539608, Caltrans-65A0533, the USC Integrated Media Systems Center (IMSC), and the USC METRANS Transportation Center.

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of any of the sponsors such as NSF.

Also, the authors would like to thank Shang-Hua Teng, Dehua Cheng and Siyang Li for helpful discussions and comments.

undirected degree matrix, In-degree/out-degree matrix L normalized graph Laplacian Φ, Λ eigen-vector matrix and eigen-value matrix of L X,X ∈ R N ×P a graph signal, and the predicted graph signal.

DISPLAYFORM0 output of the diffusion convolutional layer.

f θ , θ convolutional filter and its parameters.

f Θ , Θ convolutional layer and its parameters.

DISPLAYFORM1 The first part of Equation 2 can be rewritten as DISPLAYFORM2 As DISPLAYFORM3 O W is sparse, it is easy to see that Equation 4 can be calculated using O(K) recursive sparse-dense matrix multiplication each with time complexity O(|E|).

Consequently, the time complexities of both Equation 2 and Equation 4 are O(K|E|).

For dense graph, we may use spectral sparsification BID7 to make it sparse.

Proof.

The spectral graph convolution utilizes the concept of normalized graph Laplacian DISPLAYFORM0 ChebNet parametrizes f θ to be a K order polynomial of Λ, and calculates it using stable Chebyshev polynomial basis.

DISPLAYFORM1 where DISPLAYFORM2 DISPLAYFORM3 L is similar to the negative random walk transition matrix, thus the output of Equation 5 is also similar to the output of Equation 2 up to constant scaling factor.

BID5 propose to use spatiotemporal nearest neighbor for traffic forecasting (ST-KNN).

Though ST-KNN considers both the spatial and the temporal dependencies, it has the following drawbacks.

As shown in BID13 , ST-KNN performs independent forecasting for each individual road.

The prediction of a road is a weighted combination of its own historical traffic speeds.

This makes it hard for ST-KNN to fully utilize information from neighbors.

Besides, ST-KNN is a non-parametric approach and each road is modeled and calculated separately BID5 , which makes it hard to generalize to unseen situations and to scale to large datasets.

Finally, in ST-KNN, all the similarities are calculated using hand-designed metrics with few learnable parameters, and this may limit its representational power.

BID8 propose DeepTransport which models the spatial dependency by explicitly collecting certain number of upstream and downstream roads for each individual road and then conduct convolution on these roads respectively.

Comparing with BID8 , DCRNN models the spatial dependency in a more systematic way, i.e., generalizing convolution to the traffic sensor graph based on the diffusion nature of traffic.

Besides, we derive DCRNN from the property of random walk and show that the popular spectral convolution ChebNet is a special case of our method.

The proposed approach is also related to graph embedding techniques, e.g., Deepwalk BID25 , node2vec BID15 which learn a low dimension representation for each node in the graph.

DCRNN also learns a representation for each node.

The learned representations capture both the spatial and the temporal dependency and at the same time are optimized with regarding to the objective, e.g., future traffic speeds.

HA Historical Average, which models the traffic flow as a seasonal process, and uses weighted average of previous seasons as the prediction.

The period used is 1 week, and the prediction is based on aggregated data from previous weeks.

For example, the prediction for this Wednesday is the averaged traffic speeds from last four Wednesdays.

As the historical average method does not depend on short-term data, its performance is invariant to the small increases in the forecasting horizon ARIMA kal : Auto-Regressive Integrated Moving Average model with Kalman filter.

The orders are (3, 0, 1), and the model is implemented using the statsmodel python package.

VAR Vector Auto-regressive model BID16 .

The number of lags is set to 3, and the model is implemented using the statsmodel python package.

SVR Linear Support Vector Regression, the penalty term C = 0.1, the number of historical observation is 5.The following deep neural network based approaches are also included.

FNN Feed forward neural network with two hidden layers, each layer contains 256 units.

The initial learning rate is 1e −3 , and reduces to 1 10 every 20 epochs starting at the 50th epochs.

In addition, for all hidden layers, dropout with ratio 0.5 and L2 weight decay 1e −2 is used.

The model is trained with batch size 64 and MAE as the loss function.

Early stop is performed by monitoring the validation error.

FC-LSTM The Encoder-decoder framework using LSTM with peephole BID28 .

Both the encoder and the decoder contain two recurrent layers.

In each recurrent layer, there are 256 LSTM units, L1 weight decay is 2e −5 , L2 weight decay 5e −4 .

The model is trained with batch size 64 and loss function MAE.

The initial learning rate is 1e-4 and reduces to 1 10 every 10 epochs starting from the 20th epochs.

Early stop is performed by monitoring the validation error.

DCRNN : Diffusion Convolutional Recurrent Neural Network.

Both encoder and decoder contain two recurrent layers.

In each recurrent layer, there are 64 units, the initial learning rate is 1e −2 , and reduces to 1 10 every 10 epochs starting at the 20th epoch and early stopping on the validation dataset is used.

Besides, the maximum steps of random walks, i.e., K, is set to 3.

For scheduled sampling, the thresholded inverse sigmoid function is used as the probability decay: DISPLAYFORM0 where i is the number of iterations while τ are parameters to control the speed of convergence.

τ is set to 3,000 in the experiments.

The implementation is available in https://github.com/ liyaguang/DCRNN.

We conduct experiments on two real-world large-scale datasets:• METR-LA This traffic dataset contains traffic information collected from loop detectors in the highway of Los Angeles County BID18 .

We select 207 sensors and collect 4 months of data ranging from Mar 1st 2012 to Jun 30th 2012 for the experiment.

The total number of observed traffic data points is 6,519,002.• PEMS-BAY This traffic dataset is collected by California Transportation Agencies (CalTrans) Performance Measurement System (PeMS).

We select 325 sensors in the Bay Area and collect 6 months of data ranging from Jan 1st 2017 to May 31th 2017 for the experiment.

The total number of observed traffic data points is 16,937,179.The sensor distributions of both datasets are visualized in FIG9 .In both of those datasets, we aggregate traffic speed readings into 5 minutes windows, and apply Z-Score normalization.

70% of data is used for training, 20% are used for testing while the remaining 10% for validation.

To construct the sensor graph, we compute the pairwise road network distances between sensors and build the adjacency matrix using thresholded Gaussian kernel BID27 .

DISPLAYFORM0 where W ij represents the edge weight between sensor v i and sensor v j , dist(v i , v j ) denotes the road network distance from sensor v i to sensor v j .

σ is the standard deviation of distances and κ is the threshold. : Sensor correlations between the center sensor and its neighborhoods for different forecasting horizons.

The correlations are estimated using regularized VAR.

We observe that the correlations are localized and closer neighborhoods usually have larger relevance, and the magnitude of correlation quickly decay with the increase of distance which is consistent with the diffusion process on the graph.

@highlight

A neural sequence model that learns to forecast on a directed graph.

@highlight

The paper proposes the Diffusion Convolutional Recurrent Neural Network architecture for the spatiotemporal traffic forecasting problem

@highlight

Proposes to build a traffic forecasting model using a diffusion process for convolutional recurrent neural networks to address saptio-temporal autocorrelation.