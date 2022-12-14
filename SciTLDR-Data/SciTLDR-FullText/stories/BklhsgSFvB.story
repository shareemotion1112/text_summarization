Multi-task learning has been successful in modeling multiple related tasks with large, carefully curated labeled datasets.

By leveraging the relationships among different tasks, multi-task learning framework can improve the performance significantly.

However, most of the existing works are under the assumption that the predefined tasks are related to each other.

Thus, their applications on real-world are limited, because rare real-world problems are closely related.

Besides, the understanding of relationships among tasks has been ignored by most of the current methods.

Along this line, we propose a novel multi-task learning framework - Learning To Transfer Via Modelling Multi-level Task Dependency, which constructed attention based dependency relationships among different tasks.

At the same time, the dependency relationship can be used to guide what knowledge should be transferred, thus the performance of our model also be improved.

To show the effectiveness of our model and the importance of considering multi-level dependency relationship, we conduct experiments on several public datasets, on which we obtain significant improvements over current methods.

Multi-task learning (Caruana, 1997) aims to train a single model on multiple related tasks jointly, so that useful knowledge learned from one task can be transferred to enhance the generalization performance of other tasks.

Over the last few years, different types of multi-task learning mechanisms (Sener & Koltun, 2018; Guo & Farooq, 2018; Ish, 2016; Lon, 2015) have been proposed and proved better than single-task learning methods from natural language processing (Palmer et al., 2017) and computer vision (Cortes et al., 2015) to chemical study (Ramsundar et al., 2015) .

Despite the success of multi-task learning, when applying to 'discrete' data (graph/text), most of the current multi-task learning frameworks (Zamir et al., 2018; Ish, 2016) only leverage the general task dependency with the assumption that the task dependency remains the same for (1) different data samples; and (2) different sub-structures (node/word) in one data sample (graph/text).

However, this assumption is not always true in many real-world problems.

(1) Different data samples may have different task dependency.

For example, when we want to predict the chemical properties of a particular toxic molecule, despite the general task dependency, its representations learned from toxicity prediction tasks should be more significant than the other tasks.

(2) Even for the same data sample, different sub-structures may have different task dependency.

Take sentence classification as an example.

Words like 'good' or 'bad' may transfer more knowledge from sentiment analysis tasks, while words like 'because' or 'so' may transfer more from discourse relation identification tasks.

In this work, to accurately learn the task dependency in both general level and data-specific level, we propose a novel framework, 'Learning to Transfer via ModellIng mulTi-level Task dEpeNdency' (L2T-MITTEN).

The general task dependency is learned as a parameterized weighted dependency graph.

And the data-specific task dependency is learned with the position-wise mutual attention mechanism.

The two-level task dependency can be used by our framework to improve the performance on multiple tasks.

And the objective function of multi-task learning can further enhance the quality of the learned task dependency.

By iteratively mutual enhancement, our framework can not only perform better on multiple tasks, but also can extract high-quality dependency structures at different levels, which can reveal some hidden knowledge of the datasets.

Another problem is that to transfer task-specific representations between every task pair, the number of transfer functions will grow quadratically as the number of tasks increases, which is unaffordable.

To solve this, we develop a universal representation space where all task-specific representations get mapped to and all target tasks can be inferred from.

This decomposition method reduces the space complexity from quadratic to linear.

We validate our multi-task learning framework extensively on different tasks, including graph classication, node classification, and text classification.

Our framework outperforms all the other state-ofthe-art (SOTA) multi-task methods.

Besides, we show that L2T-MITTEN can be used as an analytic tool to extract interpretable task dependency structures at different levels on real-world datasets.

Our contributions in this work are threefold:

??? We propose a novel multi-task learning framework to learn to both general task dependency and data-specific task dependency.

The learned task dependency structures can be mutually enhanced with the objective function of multi-task learning.

??? We develop a decomposition method to reduce the space complexity needed by transfer functions from quadratic to linear.

??? We conduct extensive experiments on different real-world datasets to show the effectiveness of our framework and the importance of modelling multi-level task dependency.

According to a recent survey (Ruder, 2017) , existing multi-task learning methods can be categorized by whether they share the parameters hardly or softly.

For hard parameter sharing, a bottom network will be shared among all the tasks, and each individual task will have its own task-specific output network.

The parameter sharing of the bottom network reduces the parameter needed to be learned and thus can avoid over-fitting to a specific task.

However, when the tasks are not relevant enough (Sha, 2002; Baxter, 2011) , the shared-bottom layers will suffer from optimization conflicts caused by mutually contradicted tasks.

If the bottom model is not capable enough to encode all the necessary knowledge from different tasks, this method will fail to correctly capture all the tasks.

Besides, Dy & Krause (2018) points out that the gradients of some dominant task will be relatively larger than gradients of other tasks.

This dominant phenomenon will be more obvious when the proportions of labeled data between tasks are uneven, in which case the model will be majorly optimized on data-rich tasks.

To alleviate this problem, some recent works (Sener & Koltun, 2018; Dy & Krause, 2018) try to dynamically adjust the task weight during the training stage.

Sener & Koltun (2018) casts the multi-task learning to a multi-objective optimization problem, and they use a gradient-based optimization method to find a Pareto optimal solution.

Dy & Krause (2018) proposes a new normalization method on gradients, which attempts to balance the influences of different tasks.

Recently, Guo & Farooq (2018) proposes to apply Mixture-of-Experts on multi-task learning, which linearly combines different experts (bottoms) by learnable gates.

Because different experts can capture different knowledge among tasks, this model can, to some extent, model the dependency relationship among tasks.

Methods using soft parameter sharing (Lon, 2015; Ish, 2016; Dai et al., 2015; Yan, 2017) do not keep the shared bottom layers.

Instead, for soft-parameter models, most of the model parameters are task-specific.

Lon (2015) focuses on reducing the annotation effort of the dependency parser tree.

By combining two networks with a L2 normalization mechanism, knowledge from a different source language can be used to reduce the requirement of the amount of annotation.

Further, in some existing works, the shallow layers of the model will be separated from other layers, and be used as the feature encoders to extract task-specific representations.

For example, Ish (2016) proposes a Cross-Stitch model which is a typical separate bottom model.

Cross-Stitch model will be trained on different tasks separately to encode the task-specific representations from different bottom layers.

Then, a cross-stitch unit is used a as a gate to combine those separately trained layers.

Yan (2017) introduces the tensor factorization model to allow common knowledge to be shared at each layer in the network.

By the strategy proposed in their work, parameters are softly shared across the corresponding layers of the deep learning network and the parameter sharing ratio will be determined by the model itself.

We also note that some recent works (Zamir et al., 2018; Lan et al., 2017; Liu et al., 2018 ) can learn to capture task dependency.

Zamir et al. (2018) computes an affinity matrix among tasks based on whether the solution for one task can be sufficiently easily read out of the representation trained for another task.

However, it can only capture the general task dependency.

Lan et al. (2017) uses a sigmoid gated interaction module between two tasks to model their relation.

But it will suffer from quadratic growth in space as the number of tasks increases.

Liu et al. (2018) utilizes a shared network to share features across different tasks and uses the attention mechanism to automatically determine the importance of the shared features for the respective task.

However, there is no knowledge transferred or interaction between tasks.

In this section, we propose our framework L2T-MITTEN, which can end-to-end learn the task dependency in both general and data-specific level, and help to assemble multi-task representations from different task-specific encoders.

To formulate our framework, we first start by briefly introducing a general setting of multi-task learning.

For each task t ??? {1, ..., T }, we have a corresponding dataset {(X

k=1 with N (t) data samples, where

k represent the feature vector of k-th data sample and y

k is its label.

We would like to train T models for these tasks, and each model has its own parameter W (t) .

Note that for different multi-task learning frameworks, these parameters can be shared hardly or softly.

The goal of multi-task learning is to improve general performance by sharing information among related tasks.

The total loss of multi-task learning is calculated as:

3.2 ARCHITECTURE OVERVIEW Figure 1 : The overall architecture of our multi-task learning framework (L2T-MITTEN).

For each input data, we will transfer its task-specific representations among different tasks, and assemble the transferred representations via a task-specific Interaction Unit to get the final representation for each task.

As is shown in Figure 1 , our framework consists of three components: Task-specific Encoders, Transfer Block, and Readout Block.

The Task-specific Encoders consists of T separate feature encoders, which can be any type of feed-forward networks based on specific data.

Unlike hard parameter sharing methods that tie the bottom encoders' parameters together, we keep each feature encoder separate to efficiently extract task-specific knowledge.

In this way, for a given data sample X k , we can use these encoders to get task-specific representations

, where E t (X k ) is the representation of X k for task t. To conduct multi-task learning, one can simply use the representation of each task alone to predict labels without sharing any parameters.

However, this model will suffer for tasks without sufficient labeled data.

Therefore, we would like to (1) transfer the knowledge among these T tasks and (2) assemble the transferred representations, which is what we do in the Transfer Block.

The Readout Block also consists of T separate readout modules depending on the specific data.

The detailed architecture for different tasks can be found in Appendix B.

In the Transfer Block, the first step is to transfer the task-specific representations from source to target tasks.

A naive way is to use a transfer function F i???j (??) to transfer the task-specific representation from the space of task i to task j for every task pair:

where W i???j ??? R d??d , and d is the dimension of the task-specific representation.

However, this will result in a total number of T 2 transfer functions.

Thus, to prevent the quadratic growth, we develop a universal representation space where all task-specific representations get mapped to and all target tasks can be inferred from.

More specifically, we decompose each transfer function F i???j (??) to F T j ??? F Si (??).

Assume that we are trying to transfer the task-specific representation E i (X k ) from task i to task j, where i, j ??? {1, 2, ..., T }.

We can decompose the transfer matrix W i???j to S i and T j .

In this way, we only need 2T transfer functions in total for F Si (??) and F T j (??).

The space complexity is reduced from O(T 2 ) to O(T ).

Here we denote the transferred representation from task i to task j as:

where S i , T j ??? R d??d and d is a hyper-parameter.

Figure 2: Position-wise mutual attention mechanism.

With the transferred representations, the next step of the Transfer Block is to assemble the transferred representations with respect to the multi-level task dependency.

Here, the multi-level task dependency consists of two parts: (1) the general task dependency and (2) the data-specific task dependency.

The multi-level task dependency is modelled by the position-wise mutual attention mechanism as shown in Figure 2 .

To model the general task dependency, we represent it by a parameterized weighted dependency graph D ??? R T ??T .

The learnable weight of this parameterized dependency graph represents the transferable weight between any task pair.

Note that the dependency graph is asymmetrical.

In this way, the negative influence of irrelevant tasks can be reduced as much as possible.

Further, even for the same task pair, the transferable weight (dependency) may be different for (1) different data samples; (2) different sub-structure (node/word) in one data sample (graph/text).

Therefore, we study the data-specific task dependency in depth.

To efficiently model the dataspecific task dependency, we consider the mutual attention between representations of the same data sample under source and target tasks.

Given H i???j , the transferred representation from the task i to task j, and E j (X k ), the original representation from the target task, we get the position-wise mutual attention by:

where W Qi , W Kj ??? R d??d are the query and key projection matrices, d is a hyper-parameter, ??? is the Hadamard product, and SUM is used to eliminate the last dimension (d ).

We use Hadamard product instead of matrix multiplication because we only want the sub-structure in a given data sample to interact with its counterpart under other tasks.

Take graph data as an example, a certain node of one graph will only give attention to the same node of that graph under other tasks.

Then, for a target task j, we obtain (1) a set of general task dependency

; and (2) a set of data-specific task dependency

.

To integrate them, we first scale data-specific task dependency A j by the general task dependency D j .

And then, we calculate the weighted sum of the transferred representations according to the multi-level task dependency.

The final assembled representationX j k (for data sample k and task j) is as follow:

whereD is the normalized version of D, W V i ??? R d??d is the value projection matrix, W Oj ??? R d??d is the output projection matrix.

Note that ?? here is a scalar parameter used to prevent the vanish gradient problem of two normalization operations.

In this section, we evaluate the performance of our proposed L2T-MITTEN approach against several classical and SOTA approaches on two application domains: graph and text.

In graph domain, we train a multitask Graph Convolutional Network (GCN) for both graph-level and node-level classification.

And in text domain, we train a multitask Recurrent Neural Network (RNN) for text classification.

Further, we provide visualization and analysis on the learned hidden dependency structure.

Codes and datasets will be released.

For graph-level classification, we use Tox21 and SIDER (Kuhn et al., 2010) .

Tox21: Toxicology in the 21st Century (Tox21) is a database measuring the toxicity of chemical compounds.

This dataset contains qualitative toxicity assays for 8014 organic molecules on 12 different targets including nuclear receptors and stress response pathways.

In our experiment, we treat each molecule as a graph and each toxicity assay as a binary graph-level classification task (for 12 tasks in total).

The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug reactions.

This dataset contains qualitative drug side-effects measurements for 1427 drugs on 27 side-effects.

In our experiment, we treat each drug (organic molecule) as a graph and the problem of predicting whether a given drug induces a side effect as a individual graph-level classification tasks (for 27 tasks in total).

For node-level classification, we use DBLP (Tang et al., 2008) and BlogCatalog (IV et al., 2009) .

In the dataset, authors are represented by nodes and its feature is generated by titles of their papers.

Two authors are linked together if they have co-authored at least two papers in 2014-2019.

We use 18 representative conferences as labels.

An author is assigned to multiple labels if he/she has published papers in some conferences.

The processed DBLP dataset is also published in our repository.

BlogCatalog:

The BlogCatalog is a collection of bloggers.

In the dataset, bloggers are represented as nodes, and there is a link between two bloggers if they are friends.

The interests of each blogger can be tagged according to the categories that he/she published blogs in.

This dataset uses 39 categories as labels.

Each blogger is assigned to multiple labels if he/she published a blog in some categories.

For text classification, we use TMDb 1 dataset.

TMDb: The Movie Database (TMDb) dataset is a collection of information for 4803 movies.

For each movie, the dataset includes information ranging from the production company, production country, release date to plot, genre, popularity, etc.

In our experiment, we select plots as the input with genres as the label.

We treat the problem of predicting whether a given plot belongs to a genre as an individual text-level classification tasks (for 20 tasks in total).

A summary of the five datasets is provided in Appendix A.

We compare our L2T-MITTEN approach with both classical and SOTA approaches.

The details are given as follows:

Single-task method:

Single-Task: Simply train a network consists of encoder block and readout block for each task separately.

Classical multi-task method:

Shared-Bottom (Caruana, 1997) : This is a widely adopted multi-task learning framework which consists of a shared-bottom network (encoder block in our case) shared by all tasks, and a separate tower network (readout block in our case) for each specific task.

The input is fed into the sharedbottom network, and the tower networks are built upon the output of the shared-bottom.

Each tower will then produce the task-specific outputs.

Cross-Stitch (Ish, 2016) : This method uses a "cross-stitch" unit to learn the combination of shared and task-specific representation.

The "cross-stitch" unit is a k ?? k trainable matrix (k is the number of tasks) which will transfer and fuse the representation among tasks by the following equation:

where x i is the output of the lower level layer for task i, ?? ij is the transfer weight from task j to task i, andx i is the input of the higher level layer for task i.

MMoE (Guo & Farooq, 2018) : This method adopts the Multi-gate Mixture-of-Expert structure.

This structure consists of multiple bottom networks (experts), and multiple gating networks which take the input features and output softmax gates assembling the experts with different weights.

The assembled features are then passed into the task-specific tower networks.

All the baseline models use the same encoder and readout block for each task.

The architecture details are provided in Appendix B.

We partition the datasets into 80:20 training/testing sets (i.e. each data sample can either appear in the training or testing set) and evaluate our approach under multiple settings 2 : (1) Sufficient setting: all tasks have sufficient labeled training data; (2) Imbalanced setting: some tasks have more labeled training data than others; (3) Deficient setting: all tasks have deficient labeled training data.

Models are trained for 100 epochs using the ADAM optimizer.

We report the performance of our approach and baselines on graph classification, node classification and text classification tasks in terms of AUC-ROC score in Table 1 and 2 respectively.

3 From the above result, first of all, we can see that the multi-task methods outperform the single-task method in most cases which shows the effectiveness of knowledge transfer and multi-task learning.

Further, we can see that our proposed L2T-MITTEN approach outperforms both classical and SOTA in most tasks.

Finally, our approach shows significant improvement under deficient labeled training data setting, since our approach leverages the structure of the data sample itself to guide the transfer among tasks.

Secondly, we found that in the real-world dataset, like DBLP dataset, our model can outperform other SOTA methods significantly, which demonstrate the importance of taking multi-level dependency into consideration.

Note that the Single-Task can achieve the second-best result.

This fact indicates that in the real-world dataset, tasks may be irrelevant to each other.

Our multi-level task dependency can be more effective to prevent the influence of other irrelevant tasks.

Furthermore, we conduct experiments on the text classification dataset, TMDb.

The Cross-Stitch model achieves the best result when the label ratio for every task is 80%.

However, our task can achieve the best result for partially labeled setting (partially 10%) and few labeled setting (all 10%).

This fact demonstrates that our directed task dependency graph can effectively prevent the negative knowledge be transferred among different tasks when the training label is few.

For visualization and analysis of the learned multi-level task dependency structure, we will take DBLP as an example here due to its simplicity in interpreting and understanding.

First, in Figure 3a , where we directly visualize the learned general task dependency matrix, we can see our approach indeed captures the task dependency structure in general, i.e. conferences from the same domain are more likely to be in the same sub-tree.

Moreover, in Figure 3b we plot the authors (nodes) according to the learned data-specific task dependency matrix and we can see that there are some clusters formed by authors.

Further, we visualize the mean value of the data-specific task dependency for each cluster, as shown in Figure 3c .

We can see that different cluster does have different task dependency.

This is desirable since when predicting if an author has published papers in some conferences, authors from different domains should have different transfer weight among conferences (tasks).

As a summary, it is demonstrated that our approach can capture the task dependency at multiple levels according to specific data.

We propose L2T-MITTEN, a novel multi-task learning framework that (1) employs the positionwise mutual attention mechanism to learn the multi-level task dependency; (2) transfers the taskspecific representations between tasks with linear space-efficiency; and (3) uses the learned multilevel task dependency to guide the inference.

We design three experimental settings where training data is sufficient, imbalanced or deficient, with multiple graph/text datasets.

Experimental results demonstrate the superiority of our method against both classical and SOTA baselines.

We also show that our framework can be used as an analytical tool to extract the task dependency structures at different levels, which can reveal some hidden knowledge of tasks and of datasets A DATASET SUMMARY Figure 4 , in the Encoder Block, we use several layers of graph convolutional layers (Kipf & Welling, 2016) followed by the layer normalization (Ba et al., 2016) .

In the Readout Block, for graph-level task, we use set-to-set (Vinyals et al., 2015) as the global pooling operator to extract the graph-level representation which is later fed to a classifier; while for node-level task, we simply eliminate the global pooling layer and feed the node-level representation directly to the classifier.

Figure 4: Graph convolutional networks architecture.

Note that in node-level task, the Set2Set layer (global pooling) is eliminated.

The text model uses long short-term memory (LSTM) architecture in their Encoder Block, and the dot-product attention in the Readout Block, as shown in Figure 5 .

The dot-product attention used to get the text-level representation is as follows:

where O ??? R n??d is the output of the LSTM, H n ??? R 1??d is the hidden state for the last word, ?? ??? R n??1 is attention weight for each word, and?? ??? R 1??d is the text-level representation (n is the number of words, d is the feature dimension for each word).

@highlight

We propose a novel multi-task learning framework which extracts multi-view dependency relationship automatically and use it to guide the knowledge transfer among different tasks.