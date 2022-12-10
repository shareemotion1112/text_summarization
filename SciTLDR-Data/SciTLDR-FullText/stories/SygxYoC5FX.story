Different kinds of representation learning techniques on graph have shown significant effect in downstream machine learning tasks.

Recently, in order to inductively learn representations for graph structures that is unobservable during training, a general framework with sampling and aggregating (GraphSAGE) was proposed by Hamilton and Ying and had been proved more efficient than transductive methods on fileds like transfer learning or evolving dataset.

However, GraphSAGE is uncapable of selective neighbor sampling and lack of memory of known nodes that've been trained.

To address these problems, we present an unsupervised method that samples neighborhood information attended by co-occurring structures and optimizes a trainable global bias as a representation expectation for each node in the given graph.

Experiments show that our approach outperforms the state-of-the-art inductive and unsupervised methods for representation learning on graphs.

Graphs and networks, e.g., social network analysis BID7 , molecule screening BID4 , knowledge base reasoning BID19 , and biological proteinprotein networks BID24 ), emerge in many real-world applications.

Learning low-dimensional vector embeddings of nodes in large graphs has been proved effective for a wide variety of prediction and graph analysis tasks BID5 ; BID18 ).

The high-level idea of node embedding is to explore high-dimensional information about the neighborhood of a node with a dense vector embedding, which can be fed to off-the-shelf machine learning approaches to tasks such as node classification and link prediction BID14 ).Whereas previous approaches BID14 ; BID5 ; BID18 ) can transductively learn embeddings on graphs, without re-training they cannot generalize to new nodes that are newly added to graphs.

It is ubiquitous in real-world evolving networks, e.g., new users joining in a social friendship circle such as facebook.

To address the problem, BID8 propose an approach, namely GraphSAGE, to leverage node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen nodes.

Despite the success of GraphSAGE, it randomly and uniformly samples neighbors of nodes, which suggests it is difficult to explore the most useful neighbor nodes.

It could be helpful if we can take advantage of the most relevant neighbors and ignore irrelevant neighbors of the target node.

Besides, GraphSAGE only focuses on training parameters of the hierarchical aggregator functions, but lose sight of preserving the memory of the training nodes, which means when training is finished, those nodes that have been trained over and over again would still be treated like unseen nodes, which causes a huge waste.

To address the first issue, inspired by GAT BID20 ), a supervised approach that assigns different weights to all neighbors of each node in each aggregating layer, we introduce a bi-attention architecture BID16 ) to perform selective neighbor sampling in unsupervised learning scenarios.

In unsupervised representation learning, when encoding embeddings of a positive 1 node pair before calculating their proximity loss BID7 ), we assume that neighbor nodes positive to both of the pair should have larger chance to be selected, since they are statistically more relevant to the current positive pair than other neighbors.

For example, when embedding words like "mouse", in FIG0 , it's more reasonable to choose "keyboard" rather than "cat" as sampled neighbor while maximizing co-occrrence probability between "mouse" and "PC", because "keyboard" also tends to co-occurr with "PC", which means its imformation should be more relevant.

We thus stack a bi-attention architecture BID16 ) on representations aggregated from both side in a positive node pair.

In this way, we learn the most relevant representations for each positive node pair corresponding to their most relevant neighbors, and simply use a fixed-size uniform sampling which allows us to efficiently generate node embeddings in batches.

To address the second issue, we combine the idea behind transductive approaches and inductive approaches, by intuitively applying an additive global embedding bias to each node's aggregated embedding.

The global embedding biases are trainable as well as parameters of aggregator functions and can be considered as a memorable global identification of each node in training sets.

When the training is completed, we generate the embedding for each node by calculating an average of multiple embedding outputs corresponding to different sampled neighbors with respect to different positive nodes.

In this way, nodes that tend to co-occur in short random-walks will have more similar embeddings based on our bi-attention mechanism.

Based on the above-mentioned two techniques, we propose a novel approach, called BIGSAGE (which stands for the BI-attention architeture, global BIas and the original framework GraphSAGE,) to explore most relevant neighbors and preserve previously learnt knowledge of nodes by utilizing bi-attention architecture and introducing global bias, respectively.

In this paper, we focus on unsupervised and inductive node embedding learning for large and evolving network data.

For unsupervised learning, many different approachs have been proposed.

DeepWalk BID14 ) and node2vec BID5 )are two classic approaches that learn node embeddings based on random-walks using or extending the Skip-Gram model.

Similarly, LINE BID18 )seeks to preserve first-and second-order proximity and trains the embedding via negative smpling.

SDNE ) jointly uses unsupervised components to preserve second-order proximity and expolit first-order proximity in its supervised components.

Unlike the methods mentioned above, some approaches were proposed to takes use of not only network structure but also node attributes and potentially node labels.

Such as TRIDNR BID13 ), CENE BID17 ), TADW BID23 ).

GraphSAGE BID8 ), which this paper is motivated from, also requires rich attributes of nodes for sampling and aggregating into embeddings that preserve rich local neighborhood strutural infromation.

Recently, in order to address the problem in large and dense networks of consequently encounted newly jointed nodes/edges, approaches such as BID8 BID20 Bojchevski & Günnemann (2017) were proposed as inductive ways of graph embedding learning and had produced impressive performance across several large-scale inductive benchmarks.

In fact, we find that the key of inductive learning is to learn an embedding encoder that relies on only information from a single node itself and/or its local observable neighborhood, instead of the entire graph as in transductive setting.

Attention mechanism in neural processes have been largely studied in Neuroscience and Computational Neuroscience BID9 ; BID3 ) and since these few years frequently applied in Deep Learning for speech recognition (Chorowski et al. (2015) ), translation BID12 ), question answering BID16 ) and visual identification of objects BID22 .

The principle inside attention mechanism is that focusing on most pertinent parts of the input, rather than using all available information, a large part of which being irrelevant to compute the desirable output.

In this paper, we are inspired by BID16 to construct a bi-attention layer upon aggregators in order to capture the useful part of the neighborhood.

In this section, we propose a hierarchical bi-attended sampling and global-biased aggregating framework (BIGSAGE).

We start by presenting an overview of our framework: the training in Algorithm 1 and the embedding generation in Algorithm 2.

Followingly, section 3.2 gives the detailed implemention of our bi-attention architeture, section 3.3 demonstrates how we combine global bias within the framework.

Given an undirected network G = {V, E, X}, in which a set of nodes V are connected by a set of edges E, and X ∈ R |V |×f is the attribute matrix of nodes.

We denote the global embedding bias matrix as B ∈ R |V |×d , where each row of B represents the d-dimensional global embedding bias of each node.

The hierarchical layer number is set as K, the embedding output of k-th layer is represented by h k , and the final output embedding z.

To learn representations in unsupervised setting, we apply the same graph-based loss function used in the origin GraphSAGE: DISPLAYFORM0 where node v p co-occurs with v on fixed-length random walk BID14 ), sigma is the sigmoid function, P n is a negative sampling distribution, and Q defines the number of negative samples.

Algorithm 1 shows the training of our framework.

Algorithm 1 BIGSAGE: training input: Training graph G train (V train , E train ); node attributes X; global embedding bias matrix B; sampling times T 1: zero initialize(B) 2: h 0 v ← x v , ∀v ∈ V train 3: run random-walk in G train and do negative sampling to gain a set of triplets: DISPLAYFORM1 for t ∈ {1, ..., T } do 7: DISPLAYFORM2 calculate the graph-based loss J G and update model parameters with SGD When generating embeddings with the learned parameters after optimization is done, GraphSAGE encode only one single random-partial-sampled neighborhoods for each node, likely leaving out information of those unselected part.

To fully preseve the structural information around, we first rerun random walk on a full graph that inlcudes the seen and unseen nodes; Then, through our bi-attention mechanism built upon the aggregating layers, we generate the most relevant embeddings of each node w.r.t its positive nodes; Eventually, we take average of these generated embeddings as final embeddings of each node and use them in downstream machine learning tasks.

The generation process is shown in Algorithm 2:Algorithm 2 BIGSAGE: generation of embedding input: Testing graph G test (V test , E test ); node attributes X test ; learned model BIGSAGE output: Vector representations z v for all v ∈ V test 1: run random-walks for each node in G test and gain a set of positive nodes pair: DISPLAYFORM3 use step 5-9 in Algorithm 1 to generate z i , z j 5: DISPLAYFORM4 We now describe our bi-attention achitecture.

Given node n, and node m as a positive node pair, after sampling T times using a uniform fixed-size sampler, we have DISPLAYFORM5 With T different representations corresponding to T different sampled neighborhoods, their similarity matrix can be calculated by our goal is to find the most similar or relevant neighborhood match between n and m within T × T possibilities, so we need to apply softmax on the flattened similarity matrix, and sum up by collumn(row) to gain attention over T neighborhoods of n(m): DISPLAYFORM6 att n = reduce sum(sof tmax(S), 0), att m = reduce sum(sof tmax(S), 1), and apply attention to T Kth layer representations for the final encoded embeddings, DISPLAYFORM7 The aggregation process with bi-attention architecture is illustrated by FIG1 .

At first, we consider simply adding a trainable bias upon the encoder's final ouput for each node in training set, DISPLAYFORM0 By training global bias, Our framework will be able to learn parameters of aggregator functions for inductive learning meanwhile preserve sufficient embedding informations for known nodes.

On one hand, these informations can be reused to produce embeddings for the known nodes or the unknown connected with the known, as supplement to the aggregator.

On the other hand, they can patially offset the uncertainty of the generation braught by the random sampling.

But through further research, we find it more efficient when appling global bias to not only the last layer but also all hidden layers, as follows:

Applying globla bias vectors through all layers improves not only the expressivity of representations on hidden layers, but also the training efficiency.

In fact, the hidden layers' output embeddings and the global biases are now belonging to one single d-dimensional vector space, as a result of which, the parameters from lower layers can be updated more directly by the involving global bias vectors that are adjusted for all different layers including the last layer, which means the loss funtion can now have more instant impact on the lower layers instead of back-propagating from top to bottom.

DISPLAYFORM1 We propose our aggregating process with global bias in Algorithm 3.

DISPLAYFORM2

In this section, we evaluate BIGSAGE against strong baselines on three challenging benchmark tasks: (i) classifying Reddit posts as belonging to different community; (ii) predicting different classes of papers in Pubmed BID15 ); (iii) classifying protein functions across varieties of protein-protein interaction (PPI) graphs BID24 ).

We start by summarizing the overall settings in our comparison experiments, and then present the experiment results of each task.

We also study the seperate effect of our bi-attention architecture and global bias in section 4.4.

We compare BIGSAGE against the following approaches for graph representations learning under a fully unsupervised and inductive setting:• GraphSAGE: Our proposed method originates from the unsupervised variant of Graph-SAGE, a hierarchical sampling and aggregating framwork for inductive learning.

We compare our method against GraphSAGE using three different aggregator: (1) Mean aggregator, which simply takes the elementwise mean of the vectors in h k−1 u∈N (v) ; (2) LSTM aggregator, which adapts LSTMs to encode a random permutation of a node's neighbors' h k−1 ; (3) Maxpool aggregator, which apply an elementwise maxpooling operation to aggregate information across the neighbor nodes.• Graph2Gauss BID1 ): Unlike GraphSAGE and my method, G2G only uses the attributes of nodes to learn their representations, with no need for link information.

Here we compare against G2G to prove that certain trade-off between sampling granularity control and embedding effectiveness does exists in inductive learning scenario.• SPINE, BID6 : Instead of hierarchical neighbor sampling, SPINE uses RootedPageRankLiben-Nowell & Kleinberg (2007) to represent the high-order structural proximities of neighborhood.

The k largest proximities are then employed to aggregate attributes of the corresponding k neighbor nodes.

For our proposed approach and the origin framework, we set the number of hierarchical layers as K = 2 with neighbor sampling sizes S 1 = 20 and S 2 = 10, the number of random-walks for each node as 100 and the walk length as 5.

The sampling time of our bi-attention layer is set as T = 10.

For all methods, the dimensionality of embeddings is set to 256.

Our approach is impemented in Tensorflow BID0 ) and trained with the Adam optimizer BID10 ) at an initial learning rate of 0.0001.

We report the comparison results in TAB0 .

In real-world large and evolving graphs, inductive node embedding learning techniques would require high efficiency of the information extraction strategy as well as stability and robustness of the Reddit is an large internet forum where users can post and comment on any content they are interested in.

The task is to predict the community, that a post belongs to.

We use the exact dataset conducted by BID8 .

In this dataset, each node represents a post and are connected with one another if the same user comments on both of them.

The node attribute is constructed by word2vec embeddings of post contents.

The first 20 days is for training, and the rest for testing/validation.

In all, this dataset contains 232,965 nodes(posts) with an average degree of 492.

The first collumn summarizes the comparison results against GraphSAGE.

For LSTM aggregator, our model shows slightly poorer performance, which is reasonable, because LSTM aggregator causes more differences between multiple sampled neighborhoods, not only the components but also the order, making it harder for our bi-attention layer to capture the proximities between neighborhoods.

One can observe that BIGSAGE outperforms GraphSAGE in both mean and pool aggregator.

Another representative of evolving graphs we evaluate on is Pubmed, one of the commonly used citation network data .

This dataset contains 19717 nodes and 44324 edges.

We remove 20 percengt of the nodes as unseen, the rest for training.

From the second column of TAB1 we find our model have better prediction results in all three aggregators.

Generalizing accross graphs requires inductive methods capable of learning a transferable encoder function rather than the present community structure.

The protein-protein-interaction(PPI) networks dataset consists of 24 graphs corresponding to different human tissues BID24 ).

We use the preprocessed data provided by BID8 , where 20 graphs for training, 2 for validation and 2 for testing.

For each node, there are 50 features representing their positional gene sets, motif gene sets and immunological signatures, and 121 labels set from gene ontology( collected from the Molecular Signatures Database (?)).

This dataset contains 56944 nodes and 818716 edges.

The final collumn of TAB0 shows us that our method outperforms GraphSAGE by 14% at most on the PPI data.

The results on three different aggregators indicates that the Mean-aggregator beats the other two in our method.

And we also quote the micro-averaged F1 score of SPINE which is based on the exact same PPI dataset as ours.

In this section we adjust BIGSAGE for tests on PPI to further study the seperate effect of bi-attention layer and global bias:• BIGSAGE-ba: only with bi-attention layer, no global bias;• BIGSAGE-sg: with bi-attention layer, global bias only applied to embeddings of the last layer;• BIGSAGE-cb: with bi-attention layer, global bias applied to all layer during training but forgotten (reset to zero matrix) while generating embeddings.

From TAB1 , we observe the three variant of BIGSAGE still show certain advance to the origin GraphSAGE, but reasonably less accurate than the origin BIGSAGE.

The result of using only biattention layer proves the effect of bi-attended sampling.

And the comparison between BIGSAGE-sg and BIGSAGE shows the high efficiency of applying global bias through all layers.

From the result of BIGSAGE-cb, we find that even after training with global-bias, it's critical to have the memory of the known nodes, which is stored in the learnt global embedding biases.

We compare Graph2Gauss against our model as well as GraphSAGE on Pubmed and PPI.

Note that Graph2Gauss only needs node attributes for embedding learning.

The comparison results in TAB0 shows that our model and GraphSAGE beat g2g whether in evolving network data or generalizing across different graphs, which proves the significance of neighbor sampling for inductive learning and that certain trade-off exists between encoding granularity and embedding effect.

In this paper, we proposed BIGSAGE, an unsupervised and inductive network embedding approach which is able to preserve local proximity wisely as well as learn and memorize global identities for seen nodes while generalizing to unseen nodes or networks.

We apply a bi-attention architeture upon hierarchical aggregating layers to directly capture the most relevant representations of co-occurring nodes.

We also present an efficient way of combining inductive and transductive approaches by allowing trainable global embedding bias to be retrieved in all layers within the hierarchical aggregating framework.

Experiments demenstrate the superiority of BIGSAGE over the state-of-art baselines on unsupervised and inductive tasks.

@highlight

For unsupervised and inductive network embedding, we propose a novel approach to explore most relevant neighbors and preserve previously learnt knowledge of nodes by utilizing bi-attention architecture and introducing global bias, respectively

@highlight

This proposes an extension to GraphSAGE using a global embedding bias matrix in the local aggregating functions and a method to sample interesting nodes.