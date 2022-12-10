In order to efficiently learn with small amount of data on new tasks, meta-learning transfers knowledge learned from previous tasks to the new ones.

However, a critical challenge in meta-learning is the task heterogeneity which cannot be well handled by traditional globally shared meta-learning methods.

In addition, current task-specific meta-learning methods may either suffer from hand-crafted structure design or lack the capability to capture complex relations between tasks.

In this paper, motivated by the way of knowledge organization in knowledge bases, we propose an automated relational meta-learning (ARML) framework that automatically extracts the cross-task relations and constructs the meta-knowledge graph.

When a new task arrives, it can quickly find the most relevant structure and tailor the learned structure knowledge to the meta-learner.

As a result, the proposed framework not only addresses the challenge of task heterogeneity by a learned meta-knowledge graph, but also increases the model interpretability.

We conduct extensive experiments on 2D toy regression and few-shot image classification and the results demonstrate the superiority of ARML over state-of-the-art baselines.

Learning quickly is the key characteristic of human intelligence, which remains a daunting problem in machine intelligence.

The mechanism of meta-learning is widely used to generalize and transfer prior knowledge learned from previous tasks to improve the effectiveness of learning on new tasks, which has benefited various applications, such as computer vision (Kang et al., 2019; , natural language processing (Gu et al., 2018; Lin et al., 2019) and social good (Zhang et al., 2019; Yao et al., 2019a) .

Most of existing meta-learning algorithms learn a globally shared meta-learner (e.g., parameter initialization (Finn et al., 2017; , meta-optimizer (Ravi & Larochelle, 2016) , metric space (Snell et al., 2017; Garcia & Bruna, 2017; Oreshkin et al., 2018) ).

However, globally shared meta-learners fail to handle tasks lying in different distributions, which is known as task heterogeneity (Vuorio et al., 2018; Yao et al., 2019b) .

Task heterogeneity has been regarded as one of the most challenging issues in meta-learning, and thus it is desirable to design meta-learning models that effectively optimize each of the heterogeneous tasks.

The key challenge to deal with task heterogeneity is how to customize globally shared meta-learner by using task-specific information?

Recently, a handful of works try to solve the problem by learning a task-specific representation for tailoring the transferred knowledge to each task (Oreshkin et al., 2018; Vuorio et al., 2018; Lee & Choi, 2018) .

However, the expressiveness of these methods is limited due to the impaired knowledge generalization between highly related tasks.

Recently, learning the underlying structure across tasks provides a more effective way for balancing the customization and generalization.

Representatively, Yao et al. propose a hierarchically structured meta-learning method to customize the globally shared knowledge to each cluster (Yao et al., 2019b) .

Nonetheless, the hierarchical clustering structure completely relies on the handcrafted design which needs to be tuned carefully and may lack the capability to capture complex relationships.

Hence, we are motivated to propose a framework to automatically extract underlying relational structures from historical tasks and leverage those relational structures to facilitate knowledge customization on a new task.

This inspiration comes from the way of structuring knowledge in knowledge bases (i.e., knowledge graphs).

In knowledge bases, the underlying relational structures across text entities are automatically constructed and applied to a new query to improve the searching efficiency.

In the meta-learning problem, similarly, we aim at automatically establishing the metaknowledge graph between prior knowledge learned from previous tasks.

When a new task arrives, it queries the meta-knowledge graph and quickly attends to the most relevant entities (vertices), and then takes advantage of the relational knowledge structures between them to boost the learning effectiveness with the limited training data.

The proposed meta-learning framework is named as Automated Relational Meta-Learning (ARML).

Specifically, the ARML automatically builds the meta-knowledge graph from meta-training tasks to memorize and organize learned knowledge from historical tasks, where each vertex represents one type of meta-knowledge (e.g., the common contour between birds and aircrafts).

To learn the meta-knowledge graph at meta-training time, for each task, we construct a prototype-based relational graph for each class, where each vertex represents one prototype.

The prototype-based relational graph not only captures the underlying relationship behind samples, but alleviates the potential effects of abnormal samples.

The meta-knowledge graph is then learned by summarizing the information from the corresponding prototype-based relational graphs of meta-training tasks.

After constructing the meta-knowledge graph, when a new task comes in, the prototype-based relational graph of the new task taps into the meta-knowledge graph for acquiring the most relevant knowledge, which further enhances the task representation and facilitates its training process.

Our major contributions of the proposed ARML are three-fold: (1) it automatically constructs the meta-knowledge graph to facilitate learning a new task; (2) it empirically outperforms the state-ofthe-art meta-learning algorithms; (3) the meta-knowledge graph well captures the relationship among tasks and improves the interpretability of meta-learning algorithms.

Meta-learning designs models to learn new tasks or adapt to new environments quickly with a few training examples.

There are mainly three research lines of meta-learning: (1) black-box amortized methods design black-box meta-learners to infer the model parameters (Ravi & Larochelle, 2016; Andrychowicz et al., 2016; Mishra et al., 2018; Gordon et al., 2019) ; (2) gradient-based methods aim to learn an optimized initialization of model parameters, which can be adapted to new tasks by a few steps of gradient descent (Finn et al., 2017; Lee & Choi, 2018; Yoon et al., 2018; Grant et al., 2018) ; (3) non-parametric methods combine parametric meta-learners and non-parametric learners to learn an appropriate distance metric for few-shot classification (Snell et al., 2017; Vinyals et al., 2016; Yang et al., 2018; Oreshkin et al., 2018; Yoon et al., 2019; Garcia & Bruna, 2017) .

Our work is built upon the gradient-based meta-learning methods.

In the line of gradient-based meta-learning, most algorithms learn a globally shared meta-learners from previous tasks (Finn et al., 2017; Li et al., 2017; Flennerhag et al., 2019) , to improve the effectiveness of learning process on new tasks.

However, these algorithms typically lack the ability to handle heterogeneous tasks (i.e., tasks sample from sufficient different distributions).

To tackle this challenge, recent works tailor the globally shared initialization to different tasks by customizing initialization (Vuorio et al., 2018; Yao et al., 2019b) and using probabilistic models (Yoon et al., 2018; .

Representatively, HSML customizes the globally shared initialization with a manually designed hierarchical clustering structure to balance the generalization and customization (Yao et al., 2019b) .

However, the handcrafted designed hierarchical structure may not accurately reflect the real structure and the clustering structure constricts the complexity of relationship.

Compared with these methods, ARML leverages the most relevant structure from the automatically constructed meta-knowledge graph.

Thus, ARML not only discovers more accurate underlying structures to improve the effectiveness of meta-learning algorithms, but also the meta-knowledge graph further enhances the model interpretability.

Few-shot Learning Considering a task Ti, the goal of few-shot learning is to learn a model with a dataset Di = {D , θ), and obtain the optimal parameters θi.

For the regression problem, the loss function is defined based on the mean square error (i.e., (x j ,y j )∈D tr i f θ (xj)−yj 2 2 ) and for the classification problem, the loss function uses the cross entropy loss (i.e., − (x j ,y j )∈D tr i log p(yj|xj, f θ )).

Usually, optimizing and learning parameter θ for the task Ti with a few labeled training samples is difficult.

To address this limitation, meta-learning provides us a new perspective to improve the performance by leveraging knowledge from multiple tasks.

Meta-learning and Model-agnostic Meta-learning In meta-learning, a sequence of tasks {T1, ..., TI } are sampled from a task-level probability distribution p(T ), where each one is a few-shot learning task.

To facilitate the adaption for incoming tasks, the meta-learning algorithm aims to find a well-generalized meta-learner on I training tasks at meta-learning phase.

At meta-testing phase, the optimal meta-learner is applied to adapt the new tasks Tt.

In this way, meta-learning algorithms are capable of adapting to new tasks efficiently even with a shortage of training data for a new task.

Model-agnostic meta-learning (MAML) (Finn et al., 2017) , one of the representative algorithms in gradient-based meta-learning, regards the meta-learner as the initialization of parameter θ, i.e., θ0, and learns a well-generalized initialization θ * 0 during the meta-training process.

The optimization problem is formulated as (one gradient step as exemplary):

At the meta-testing phase, to obtain the adaptive parameter θt for each new task Tt, we finetune the initialization of parameter θ * 0 by performing gradient updates a few steps, i.e.,

In this section, we introduce the details of the proposed ARML.

To better explain how it works, we show its framework in Figure 1 .

The goal of ARML is to facilitate the learning process of new tasks by leveraging transferable knowledge learned from historical tasks.

To achieve this goal, we introduce a meta-knowledge graph, which is automatically constructed at the meta-training time, to organize and memorize historical learned knowledge.

Given a task, which is built as a prototypebased relational structure, it taps into the meta-knowledge graph to acquire relevant knowledge for enhancing its own representation.

The enhanced prototype representations further aggregate and incorporate with meta-learner for fast and effective adaptions by utilizing a modulating function.

In the following subsections, we elaborate three key components: prototype-based sample structuring, automated meta-knowledge graph construction and utilization, and task-specific knowledge fusion and adaptation, respectively.

Given a task which involves either classifications or regressions regarding a set of samples, we first investigate the relationships among these samples.

Such relationship is represented by a graph, called prototype-based relational graph in this work, where the vertices in the graph denote the prototypes of different classes while the edges and the corresponding edge weights are created based on the similarities between prototypes.

Constructing the relational graph based on prototypes instead of raw samples allows us to alleviate the issue raised by abnormal samples.

As the abnormal samples, which locate far away from normal samples, could pose significant concerns especially when only a limited number of samples are available for training.

Specifically, for classification problem, the prototype, denoted by c

, is defined as:

where N tr k denotes the number of samples in class k. E is an embedding function, which projects xj into a hidden space where samples from the same class are located closer to each other while samples from different classes stay apart.

For regression problem, it is not straightforward to construct Figure 1: The framework of ARML.

For each task T i , ARML first builds a prototype-based relational structure R i by mapping the training samples D tr i into prototypes, with each prototype represents one class.

Then, R i interacts with the meta-knowledge graph G to acquire the most relevant historical knowledge by information propagation.

Finally, the task-specific modulation tailors the globally shared initialization θ 0 by aggregating of raw prototypes and enriched prototypes, which absorbs relevant historical information from the meta-knowledge graph.

the prototypes explicitly based on class information.

Therefore, we cluster samples by learning an assignment matrix Pi ∈ R K×N tr .

Specifically, we formulate the process as:

where Pi[k] represents the k-th row of Pi.

Thus, training samples are clustered to K clusters, which serve as the representation of prototypes.

After calculating all prototype representations {c k i |∀k ∈ [1, K]}, which serve as the vertices in the the prototype-based relational graph Ri, we further define the edges and the corresponding edge weights.

The edge weight A Ri (c

where Wr and br represents learnable parameters, γr is a scalar and σ is the Sigmoid function, which normalizes the weight between 0 and 1.

For simplicity, we denote the prototype-based relational graph

K×d represent a set of vertices, with each one corresponds to the prototype from a class, while

gives the adjacency matrix, which indicates the proximity between prototypes.

In this section, we first discuss how to organize and distill knowledge from historical learning process and then expound how to leverage such knowledge to benefit the training of new tasks.

To organize and distill knowledge from historical learning process, we construct and maintain a meta-knowledge graph.

The vertices represent different types of meta-knowledge (e.g., the common contour between aircrafts and birds) and the edges are automatically constructed to reflect the relationship between meta-knowledge.

When serving a new task, we refer to the meta-knowledge, which allows us to efficiently and automatically identify relational knowledge from previous tasks.

In this way, the training of a new task can benefit from related training experience and get optimized much faster than otherwise possible.

In this paper, the meta-knowledge graph is automatically constructed at the meta-training phase.

The details of the construction are elaborated as follows:

Assuming the representation of an vertex g is given by h g ∈ R d , we define the meta-knowledge graph as G = (HG, AG), where HG = {h

G×G denote the vertex feature matrix and vertex adjacency matrix, respectively.

To better explain the construction of the meta-knowledge graph, we first discuss the vertex representation HG.

During meta-training, tasks arrive one after another in a sequence and their corresponding vertices representations are expected to be updated dynamically in a timely manner.

Therefore, the vertex representation of meta-knowledge graph are defined to get parameterized and learned at the training time.

Moreover, to encourage the diversity of meta-knowledge encoded in the meta-knowledge graph, the vertex representations are randomly initialized.

Analogous to the definition of weight in the prototype-based relational graph Ri in equation 4, the weight between a pair of vertices j and m is constructed as:

where Wo and bo represent learnable parameters and γo is a scalar.

To enhance the learning of new tasks with involvement of historical knowledge, we query the prototype-based relational graph in the meta-knowledge graph to obtain the relevant knowledge in history.

The ideal query mechanism is expected to optimize both graph representations simultaneously at the meta-training time, with the training of one graph facilitating the training of the other.

In light of this, we construct a super-graph Si by connecting the prototype-based relational graph Ri with the meta-knowledge graph G for each task Ti.

The union of the vertices in Ri and G contributes to the vertices in the super-graph.

The edges in Ri and G are also reserved in the super-graph.

We connect Ri with G by creating links between the prototype-based relational graph with the meta-knowledge graph.

The link between prototype c j i in prototype-based relational graph and vertex h m in metaknowledge graph is weighted by the similarity between them.

More precisely, for each prototype c and {h m |∀m ∈ [1, G]} as follows:

where γ s is a scaling factor.

We denote the intra-adjacent matrix as AS = {AS (c

K×G .

Thus, for task T i , the adjacent matrix and feature matrix of super-graph

After constructing the super-graph Si, we are able to propagate the most relevant knowledge from meta-knowledge graph G to the prototype-based relational graph Ri by introducing a Graph Neural Networks (GNN).

In this work, following the "message-passing" framework (Gilmer et al., 2017) , the GNN is formulated as:

where MP(·) is the message passing function and has several possible implementations (Hamilton et al., 2017; Kipf & Welling, 2017; Veličković et al., 2018)

is the vertex embedding after l layers of GNN and W (l) is a learnable weight matrix of layer l.

The input

After stacking L GNN layers, we get the information-propagated feature representation for the prototype-based relational graph Ri as the top-K rows of H

After propagating information form meta-knowledge graph to prototype-based relational graph, in this section, we discuss how to learn a well-generalized meta-learner for fast and effective adaptions to new tasks with limited training data.

To tackle the challenge of task heterogeneity, in this paper, we incorporate task-specific information to customize the globally shared meta-learner (e.g., initialization here) by leveraging a modulating function, which has been proven to be effective to provide customized initialization in previous studies Vuorio et al., 2018) .

The modulating function relies on well-discriminated task representations, while it is difficult to learn all representations by merely utilizing the loss signal derived from the test set D ts i .

To encourage such stability, we introduce two reconstructions by utilizing two auto-encoders.

There are two collections of parameters, i.e, CR i andĈR i , which contribute the most to the creation of the task-specific meta-learner.

CR i express the raw prototype information without tapping into the meta-knowledge graph, whileĈR i give the prototype representations after absorbing the relevant knowledge from the meta-knowledge graph.

Therefore, the two reconstructions are built on CR i andĈR i .

To reconstruct CR i , an aggregator AG q (·) (e.g., recurrent network, fully connected layers) is involved to encode CR i into a dense representation, which is further fed into a decoder AG q dec (·) to achieve reconstructions.

Compute the similarity between each prototype and meta-knowledge vertex in equation 6 and construct the super-graph Si 8:

Apply GNN on super-graph Si and get the information-propagated representationĈR i 9:

Aggregate CR i in equation 8 andĈR i in equation 9 to get the representations qi, ti and reconstruction loss Lq, Lt

Compute the task-specific initialization θ0i in equation 10 and update

end for 12:

13: end while Then, the corresponded task representation qi of CR i is summarized by applying a mean pooling operator over prototypes on the encoded dense representation.

Formally,

Similarly, we reconstructĈR i and get the corresponded task representation ti as follows:

The reconstruction errors in Equations 8 and 9 pose an extra constraint to enhance the training stability, leading to improvement of task representation learning.

After getting the task representation qi and ti, the modulating function is then used to tailor the task-specific information to the globally shared initialization θ0, which is formulated as:

where Wg and b g is learnable parameters of a fully connected layer.

Note that we adopt the Sigmoid gating as exemplary and more discussion about different modulating functions can be found in ablation studies of Section 5.

For each task Ti, we perform the gradient descent process from θ0i and reach its optimal parameter θi.

Combining the reconstruction loss Lt and Lq with the meta-learning loss defined in equation 1, the overall objective function of ARML is:

where µ1 and µ2 are introduced to balance the importance of these three items.

Φ represents all learnable parameters.

The algorithm of meta-training process of ARML is shown in Alg.

2.

The details of the meta-testing process of ARML are available in Appendix A.

In this section, we conduct extensive experiments to demonstrate the effectiveness of the ARML on 2D regression and few-shot classification.

We compare our proposed ARML with two types of baselines: (1) Gradient-based meta-learning methods: both globally shared methods (MAML (Finn et al., 2017) , Meta-SGD (Li et al., 2017) ) and task-specific methods (MT-Net (Lee & Choi, 2018) , MUMO-MAML (Vuorio et al., 2018) , HSML (Yao et al., 2019b) , BMAML (Yoon et al., 2018) ) are considered for comparison.

(2) Other meta-learning methods (non-parametric and black box amortized methods): we select globally shared methods VERSA (Gordon et al., 2019) , Prototypical Network (ProtoNet) (Snell et al., 2017) , TapNet (Yoon et al., 2019) , we use the GRU as the encoder and decoder in this structure.

We adopt one layer GCN (Kipf & Welling, 2017) with tanh activation as the implementation of GNN in equation 7.

For the modulation network, we test sigmoid, tanh and Film modulation, and find that sigmoid modulation achieves best performance.

Thus, in the future experiment, we set the sigmoid modulation as modulating function.

More detailed discussion about experiment settings are presented in Appendix B.

Dataset Description In 2D regression problem, we adopt the similar regression problem settings as Vuorio et al., 2018; Yao et al., 2019b; Rusu et al., 2019) , which includes several families of functions.

In this paper, to model more complex relational structures, we design a 2D regression problem rather than traditional 1D regression.

Results and Analysis In Figure 2 , we summarize the interpretation of meta-knowledge graph (see top figure, and more cases are provided in Figure 8 of Appendix G.4) and the the qualitative results (see bottom table) of 10-shot 2D regression.

In the bottom table, we can observe that ARML achieves the best performance as compared to competitive gradient-based meta-learning methods, i.e., globally shared models and task-specific models.

This finding demonstrates that the meta-knowledge graph is necessary to model and capture task-specific information.

The superior performance can also be interpreted in the top figure.

In the left, we show the heatmap between prototypes and meta-knowledge vertices (darker color means higher similarity).

We can see that sinusoids and line activate V1 and V4, which may represent curve and line, respectively.

V1 and V4 also contribute to quadratic and quadratic surface, which also show the similarity between these two families of functions.

V3 is activated in P0 of all functions and the quadratic surface and ripple further activate V1 in P0, which may show the different between 2D functions and 3D functions (sinusoid, line, quadratic and cubic lie in the subspace).

Specifically, in the right figure, we illustrate the meta-knowledge graph, where we set a threshold to filter the link with low similarity score and show the rest.

We can see that V3 is the most popular vertice and connected with V1, V5 (represent curve) and V4 (represent line).

V1 is further connected with V5, demonstrating the similarity of curve representation.

In the few-shot classification problem, we first use the benchmark proposed in (Yao et al., 2019b) , where four fine-grained image classification datasets are included (Aircraft), and FGVCx-Fungi (Fungi)).

For each few-shot classification task, it samples classes from one of four datasets.

In this paper, we call this dataset as Plain-Multi and each fine-grained dataset as subdataset.

Then, to demonstrate the effectiveness of our proposed model for handling more complex underlying structures, in this paper, we increase the difficulty of few-shot classification problem by introducing two image filters: blur filter and pencil filter.

Similar as (Jerfel et al., 2019) , for each image in PlainMulti, one artistic filters are applied to simulate a changing distribution of few-shot classification tasks.

After applying the filters, the total number of subdatasets is 12 and each tasks is sampled from one of them.

This data is named as Art-Multi.

More detailed descriptions of the effect of different filters is discussed in Appendix C.

Following the traditional meta-learning settings, all datasets are divided into meta-training, metavalidation and meta-testing classes.

The traditional N-way K-shot settings are used to split training and test set for each task.

We adopt the standard four-block convolutional layers as the base learner (Finn et al., 2017; Snell et al., 2017) for ARML and all baselines for fair comparison.

The number of vertices of meta-knowledge graph for Plain-Multi and Art-Multi datasets are set as 4 and 8, respectively.

Additionally, for the miniImagenet and tieredImagenet (Ren et al., 2018) , similar as , which tasks are constructed from a single domain and do not have heterogeneity, we compare our proposed ARML with baseline models and present the results in Appendix D.

Overall Performance Experimental results for Plain-Multi and Art-Multi are shown in Table 1 and Table 2 , respectively.

For each dataset, the performance accuracy with 95% confidence interval is reported.

Due to the space limitation, in Art-Multi dataset, we only show the average value of each filter here.

The full results are shown in Table 8 of Appendix E. In these two tables, first, we can observe that task-specific gradient-based models (MT-Net, MUMOMAML, HSML, BMAML) significantly outperforms globally shared models (MAML, Meta-SGD).

Second, compared ARML with other task-specific gradient-based meta-learning methods, the better performance confirms that ARML can model and extract task-specific information more accurately by leveraging the constructed meta-knowledge graph.

Especially, the performance gap between the ARML and HSML verifies the benefits of relational structure compared with hierarchical clustering structure.

Third, as a gradientbased meta-learning algorithm, ARML can also outperform methods of other research lines (i.e., ProtoNet, TADAM, TapNet and VERSA).

Finally, to show the effectiveness of proposed components in ARML, we conduct comprehensive ablation studies in Appendix F. The results further demonstrate the effectiveness of prototype-based relational graph and meta-knowledge graph.

In this section, we conduct extensive qualitative analysis for the constructed meta-knowledge graph, which is regarded as the key component in ARML.

Due to the space limit, we present the results on Art-Multi datasets here and the analysis of Plain-Multi with similar observations are discussed in Appendix G.1.

We further analyze the effect To analyze the learned meta-knowledge graph, for each subdataset, we randomly select one task as exemplary (see Figure 9 of Appendix G.4 for more cases).

For each task, in the left part of Figure 3 , we show the similarity heatmap between prototypes and vertices in meta-knowledge graph, where deeper color means higher similarity.

V0-V8 and P1-P5 denotes the different vertices and prototypes, respectively.

The meta-knowledge graph is also illustrated in the right part.

Similar as the graph in 2D regression, we set a threshold to filter links with low similarity and illustrate the rest of them.

First, we can see that the V1 is mainly activated by bird and aircraft (including all filters), which may reflect the shape similarity between bird and aircraft.

Second, V2, V3, V4 are firstly activated by texture and they form a loop in the meta-knowledge graph.

Especially, V2 also benefits images with blur and pencil filters.

Thus, V2 may represent the main texture and facilitate the training process on other subdatasets.

The meta-knowledge graph also shows the importance of V2 since it is connected with almost all other vertices.

Third, when we use blur filter, in most cases (bird blur, texture blur, fungi blur), V7 is activated.

Thus, V7 may show the similarity of images with blur filter.

In addition, the connection between V7 and V2 and V3 show that classify blur images may depend on the texture information.

Fourth, V6 (activated by aircraft mostly) connects with V2 and V3, justifying the importance of texture information to classify the aircrafts.

In this paper, to improve the effectiveness of meta-learning for handling heterogeneous task, we propose a new framework called ARML, which automatically extract relation across tasks and construct a meta-knowledge graph.

When a new task comes in, it can quickly find the most relevant relations through the meta-knowledge graph and use this knowledge to facilitate its training process.

The experiments demonstrate the effectiveness of our proposed algorithm.

In the future, we plan to investigate the problem in the following directions: (1) we are interested to investigate the more explainable semantic meaning in the meta-knowledge graph on this problem; (2) Figure 3 : Interpretation of meta-knowledge graph on Art-Multi dataset.

For each subdataset, we randomly select one task from them.

In the left, we show the similarity heatmap between prototypes (P0-P5) and meta-knowledge vertices (V0-V7).

In the right part, we show the meta-knowledge graph.

we plan to extend the ARML to the continual learning scenario where the structure of meta-knowledge graph will change over time; (3) our proposed model focuses on tasks where the feature space, the label space are shared.

We plan to explore the relational structure on tasks with different feature and label spaces.

The work was supported in part by NSF awards #1652525 and #1618448.

The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies.

Algorithm 2 Meta-Testing Process of ARML Require: Training data D tr t of a new task T t 1: Construct the prototype-based relational graph R t by computing prototype in equation 2 and weight in equation 4 2: Compute the similarity between each prototype and meta-knowledge vertice in equation 6 and construct the super-graph St 3: Apply GNN on super-graph St and get the updated prototype representationĈR t 4: Aggregate CR t in equation 8,ĈR t in equation 9 and get the representations qt, tt 5: Compute the task-specific initialization θ0t in equation 10 6: Update parameters

In 2D regression problem, we set the inner-loop stepsize (i.e., α) and outer-loop stepsize (i.e., β) as 0.001 and 0.001, respectively.

The embedding function E is set as one layer with 40 neurons.

The autoencoder aggregator is constructed by the gated recurrent structures.

We set the meta-batch size as 25 and the inner loop gradient steps as 5.

In few-shot image classification, for both Plain-Multi and Art-Multi datasets, we set the corresponding inner stepsize (i.e., α) as 0.001 and the outer stepsize (i.e., β) as 0.01.

For the embedding function E, we employ two convolutional layers with 3 × 3 filters.

The channel size of these two convolutional layers are 32.

After convolutional layers, we use two fully connected layers with 384 and 128 neurons for each layer.

Similar as the hyperparameter settings in 2D regression, the autoencoder aggregator is constructed by the gated recurrent structures, i.e., AG t , AG t dec AG q , AG q dec are all GRUs.

The meta-batch size is set as 4.

For the inner loop, we use 5 gradient steps.

For the gradient-based baselines (i.e., MAML, MetaSGD, MT-Net, BMAML.

MUMOMAML, HSML), we use the same inner loop stepsize and outer loop stepsize rate as our ARML.

As for non-parametric based meta-learning algorithms, both TADAM and Prototypical network, we use the same meta-training and meta-testing process as gradient-based models.

Additionally, TADAM uses the same embedding function E as ARML for fair comparison (i.e., similar expressive ability).

In this dataset, we use pencil and blur filers to change the task distribution.

To investigate the effect of pencil and blur filters, we provide one example in Figure 4 .

We can observe that different filters result in different data distributions.

All used filter are provided by OpenCV 1 .

For miniimagenet and tieredImagenet, since it do not have the characteristic of task heterogeneity, we show the results in Table 3 and Table 4 , respectively.

In this table, we compare our model with other gradient-based meta-learning models (the top baselines are globally shared models and the bottom baselines are task-specific models).

Similar as , we also apply the standard 4-block convolutional layers for each baseline.

For MT-Net on MiniImagenet, we use the reported results in (Yao et al., 2019b) , which control the model with the same expressive power.

Most task-specific models including ARML achieve the similar performance on the standard benchmark due to the homogeneity between tasks. (Finn et al., 2017) 48.70 ± 1.84% LLAMA 49.40 ± 1.83% Reptile (Nichol & Schulman, 2018) 49.97 ± 0.32% MetaSGD (Li et al., 2017) 50.47 ± 1.87%

MT-Net (Lee & Choi, 2018) 49.75 ± 1.83% MUMOMAML (Vuorio et al., 2018) 49.86 ± 1.85% HSML (Yao et al., 2019b) 50.38 ± 1.85% PLATIPUS 50.13 ± 1.86% ARML 50.42 ± 1.73% Table 4 : Performance comparison on the 5-way, 1-shot tieredImagenet dataset.

Algorithms 5-way 1-shot Accuracy MAML (Finn et al., 2017) 51.37 ± 1.80% Reptile (Nichol & Schulman, 2018) 49.41 ± 1.82% MetaSGD (Li et al., 2017) 51.48 ± 1.79%

MT-Net (Lee & Choi, 2018) 51.95 ± 1.83% MUMOMAML (Vuorio et al., 2018) 52.59 ± 1.80% HSML (Yao et al., 2019b) 52.67 ± 1.85% ARML 52.91 ± 1.83%

We provide the full results table of Art-Multi Dataset in Table 8 .

In this table, we can see our proposed ARML outperforms almost all baselines in every sub-datasets.

In this section, we perform the ablation study of the proposed ARML to demonstrate the effectiveness of each component.

The results of ablation study on 5-way, 5-shot scenario for Art-Multi and PlainMulti datasets are presented in Table 5 and Table 6 , respectively.

Specifically, to show the effectiveness of prototype-based relational graph, in ablation I, we apply the mean pooling to aggregate each sample and then feed it to interact with meta-knowledge graph.

In ablation II, we use all samples to construct the sample-level relational graph without constructing prototype.

In ablation III, we remove the links between prototypes.

Compared with ablation I, II and III, the better performance of ARML shows that structuring samples can (1) better handling the underlying relations (2) alleviating the effect of potential anomalies by structuring samples as prototypes.

In ablation IV, we remove the meta-knowledge graph and use the prototype-based relational graph with aggregator AG q as the task representation.

The better performance of ARML demonstrates the effectiveness of meta-knowledge graph for capturing the relational structure and facilitating the classification performance.

We further remove the reconstruction loss in ablation V and replace the encoder/decoder structure as MLP in ablation VI.

The results demonstrate that the autoencoder structure benefits the process of task representation learning and selected encoder and decoder.

In ablation VII, we share the gate value within each filter in Convolutional layers.

Compared with VII, the better performance of ARML indicates the benefit of customized gate for each parameter.

In ablation VIII and IX, we change the modulate function to Film (Perez et al., 2018) and tanh, respectively.

We can see that ARML is not very sensitive to the modulating activation, and sigmoid function is slightly better in most cases.

Figure 5 : Interpretation of meta-knowledge graph on Plain-Multi dataset.

For each subdataset, one task is randomly selected from them.

In the left figure, we show the similarity heatmap between prototypes (P1-P5) and meta-knowledge vertices (denoted as E1-E4), where deeper color means higher similarity.

In the right part, we show the meta-knowledge graph, where a threshold is also set to filter low similarity links.

We first investigate the impact of vertice numbers in meta-knowledge graph.

The results of Art-Multi (5-way, 5-shot) are shown in Table 7 .

From the results, we can notice that the performance saturates as the number of vertices around 8.

One potential reason is that 8 vertices are enough to capture the potential relations.

If we have a larger datasets with more complex relations, more vertices may be needed.

In addition, if the meta-knowledge graph do not have enough vertices, the worse performance suggests that the graph may not capture enough relations across tasks.

In this part, we provide the case study to visualize the task structure of HSML and ARML.

HSML is one of representative task-specific meta-learning methods, which adapts transferable knowledge by introducing a task-specific representation.

It proposes a tree structure to learn the relations between tasks.

However, the structure requires massive labor efforts to explore the optimal structure.

By contrast, ARML automatically learn the relation across tasks by introducing the knowledge graph.

In addition, ARML fully exploring there types of relations simultaneously, i.e., the prototype-prototype, prototype-knowledge and knowledge-knowledge relations.

To compare these two models, we show the case studies of HSML and ARML in Figure 6 and Figure 7 .

For tasks sampled from bird, bird blur, aircraft and aircraft blur are selected for this comparison.

Following case study settings in the original paper (Yao et al., 2019b) , for each task, we show the soft-assignment probability to each cluster and the learned hierarchical structure.

For ARML, like 3, we show the learned meta-knowledge and the similarity heatmap between prototypes and meta-knowledge vertices.

In this figures we can observe that ARML constructs relations in a more flexible way by introducing the graph structure.

More specifically, while HSML activate relevant node in a fixed two-layer hierarchical way, ARML provides more possibilities to leverage previous learned tasks by leveraging prototypes and the learned meta-knowledge graph.

Published as a conference paper at ICLR 2020

We provide additional case study in this section.

In Figure 8 , we show the cases of 2D regression and the additional cases of Art-Multi are illustrated in Figure 9 .

We can see the additional cases also support our observations and interpretations.

<|TLDR|>

@highlight

Addressing task heterogeneity problem in meta-learning by introducing meta-knowledge graph