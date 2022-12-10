Neural Network (NN) has achieved state-of-the-art performances in many tasks within image, speech, and text domains.

Such great success is mainly due to special structure design to fit the particular data patterns, such as CNN capturing spatial locality and RNN modeling sequential dependency.

Essentially, these specific NNs achieve good performance by leveraging the prior knowledge over corresponding domain data.

Nevertheless, there are many applications with all kinds of tabular data in other domains.

Since there are no shared patterns among these diverse tabular data, it is hard to design specific structures to fit them all.

Without careful architecture design based on domain knowledge, it is quite challenging for NN to reach satisfactory performance in these tabular data domains.

To fill the gap of NN in tabular data learning, we propose a universal neural network solution, called TabNN, to derive effective NN architectures for tabular data in all kinds of tasks automatically.

Specifically, the design of TabNN follows two principles: \emph{to explicitly leverages expressive feature combinations} and \emph{to reduce model complexity}. Since GBDT has empirically proven its strength in modeling tabular data, we use GBDT to power the implementation of TabNN.

Comprehensive experimental analysis on a variety of tabular datasets demonstrate that TabNN can achieve much better performance than many baseline solutions.

Recent years have witnessed the extraordinary success of Neural Networks (NN), especially Deep Neural Networks, in achieving state-of-the-art performances in many domains, such as image classification BID27 , speech recognition BID25 , and text mining BID22 .

Beside enlarged model capacity, such great achievement of NN is mainly due to the deliberate design of its structures derived from prior knowledge over the certain domain data.

For example, Convolutional Neural Networks (CNN) BID40 have become the standard solution to address image classification since it can capture the spatial locality by using "Local Receptive Field" BID40 , which is a common pattern in image data.

Recurrent Neural Networks (RNN) BID29 , as another example, has been widely-used on speech recognition and language modeling because its recurrent structure can effectively model the sequential dependency among speech and text data.

In contrast to most of tasks in image, speech, or text domains whose input yields natural spatial or temporal dimension, many other real-world applications, e.g., click through rate prediction BID24 , time series forecasting BID49 BID11 , web search ranking BID0 BID8 , etc, bear structured input consisting of multi-dimension meaningful features.

Typically, such input data can be generalized as the tabular data, as each row of the tabular corresponds to one data example and each column denotes an individual meaningful feature.

Despite the success of CNN and RNN over computer vision, speech recognition, and natural language process, adopting NN over tabular data receives far less attention and yet remains quite challenging.

In particular, as illustrated in previous studies BID18 , it usually leads to unsatisfactory performance on tabular data by directly using Fully Connected Neural Network (FCNN), because its fully connected model structure leads to very complex optimization hyper-planes with a high risk of falling into local optimums.

Moreover, since different applications usually indicate various effective feature combinations within their respective tabular data, it is quite beneficial to recognize such feature combinations and take advantage of them to design the effective NN model on their tabular data, which however has not been well studied yet.

To address these challenges, we identify two principles for the purpose of designing effective NN models on tabular data: (1) To explicitly leverage expressive feature combinations.

Rather than blindly pouring all features together into FCNN and learning via back-propagation to discover the implicit feature combinations, it will be beneficial to let NN explicitly leverage the expressive feature combinations.

(2) To reduce model complexity.

Contrary to highly-complex FCNN with too many parameters leading to higher risk of over-fitting or falling into local optimums, it is vital to reduce the complexity of NN models by removing unnecessary parameters and encouraging parameter sharing.

Inspired by these two principles, we propose a universal neural network solution, called TabNN, to derive effective NN architectures for tabular data in all kinds of tasks automatically, by leveraging the knowledge learned by GBDT model (Gradient Boosting Decision Tree) BID19 BID15 BID12 , which has empirically proven its strength in modeling tabular data BID12 .

More specifically, the GBDT-powered TabNN consists of four major steps: (1) Automatic Feature Grouping (AFG) automatically discovers feature groups implying effective partial combinations based on GBDT-powered knowledge.

(2) Feature Group Reduction (FGR) attempts to further cluster feature groups in order to encourage parameter sharing within the same clusters, which can accordingly reduce the complexity of the resulting NN models.

(3) Recursive Encoder with Shared Embedding (RESE) aims at designing a both effective and efficient NN architecture over clustered tabular feature groups, based on the results of FGR and the feature group importance powered by GBDT.

(4) Transfer Structured Knowledge from GBDT (TSKG) further leverages structured knowledge within GBDT model to provide an effective initialization for the obtained NN architecture.

To illustrate the effectiveness of the proposed TabNN solution, we conduct extensive experiments on various publicly available datasets with tabular data.

Comprehensive experimental analysis has shown that TabNN cannot only create effective NN architectures for various tabular data but also achieves much better performance than other solutions.

In summary, the contributions of this paper are multi-fold:• We identify two principles for the purpose of designing effective NN models on tabular data.• We propose TabNN, a general solution for deriving effective NN models for tabular data by leveraging the data knowledge learned by GBDT.• Extensive experiments show that the proposed method is an off-of-shelf model, which can be ready to use in any kinds of tabular data efficiently and achieves state-of-the-art performance.

Tabular Data Learning by Tree-Based Models.

Tree-based methods, such as GBDT and Random Forest BID2 , have been widely applied in many real-world applications, e.g., click through rate prediction BID41 and web search ranking (Burges, 2010), etc., and have become the first choice in various well-recognized data mining competitions BID12 .

The success of GBDT and other tree-based methods over tabular data majorly relies on their capability on iteratively picking the features with the most statistical information gain to build the trees BID23 BID60 .

Therefore, even if there are amounts of features in the tabular data, GBDT can automatically choose the most useful features to fit the targets well.

However, tree-based models still yield two obvious shortages: (1) Hard to be integrated into complex end-to-end frameworks.

GBDT or other tree-based models cannot back-propagate the error directly to their inputs, thus they cannot be easily plugged into a complex end-to-end framework.

To solve this problem, soft decision trees or neural decision trees have been proposed BID6 BID50 BID56 BID33 BID38 BID20 by using differentiable decision functions, instead of non-differentiable axis aligned splits, to construct trees.

However, abandoning axis aligned splits will lose the automatic feature selection ability, which is important for learning from tabular data.

BID17 propose to use target propagation to pass back the error for nondifferentiable functions.

However, target propagation is inefficient compared with back-propagation as it needs to learn many additional models to propagate the errors.

(2) Hard to learn from streaming data.

Many real-world applications, such as online advertising, continuously generate the large scale of streaming data.

Unfortunately, learning tree-based click prediction and recommendation models over streaming data is quite difficult since it usually needs global statistical information to select split points.

There have been some works that try to efficiently learn trees from streaming data BID34 BID21 BID3 .

However, these models are specifically designed for the single tree model and their performance cannot achieve the same accuracy as using full data at once.

XGBoost BID12 and LightGBM BID36 also provided a simple solution: they learn the structures of trees at first, then, keep the tree structures fixed and update the leaf outputs by the streaming data.

Although this solution is simple and efficient, the performance is still worse than learning from all data at once.

Tabular Data Learning by NN.

These obvious shortages of tree-based methods encourage increasing efforts in applying NN to learn the model over tabular data.

Many recent studies attempt to use NN in a variety of applications with tabular data, including the click-through rate prediction BID54 BID26 and recommendation system BID62 BID13 BID14 BID66 .

Most of them, in fact, focus on how to pre-process categorical features to better adapt to NN.

Meanwhile, many numerical features, which are also very important in tabular data, are not well utilized in these works.

To sum up, there has no universal NN solution to fit all kinds of tabular data well.

The method proposed in this paper aims to fill in this gap and provide an off-of-shelf and universal NN solution.

Combine NNs with Trees.

Due to the respective pros and cons of NN and tree-based methods, there have been emerging efforts that proposed to combine the NNs and tree-based methods.

In general, these efforts can be categorized into two classes: (1) Tree-like NN.

As pointed by BID31 , there have been some tree-like NNs, which have decision ability like decision trees to some extent, e.g. GoogLeNet .

Rota BID57 and BID38 also introduced the tree-like structure and decision ability into NN.

However, these works mainly focused on computer vision tasks without attention to tabular data.

proposed the soft binning function to simulate decision trees in NN, which is, however, very inefficient as it enumerates all possible decisions.

proposed NNRF, which used tree-like NN and random feature selection to improve the learning from tabular data.

Nevertheless, NNRF simply uses random feature combinations, without leveraging the information from data itself.

(2) Convert Trees to NN.

Another track of works tried to convert the trained decision trees to NNs BID59 BID1 BID55 BID5 BID30 .

However, these works are inefficient as they use a redundant and usually very sparse NN to represent a simple decision tree.

When there are many trees, such conversion solution has to construct a very wide NN to represent them, which is unfortunately hard to be applied to realistic scenarios.

Network Architecture Search.

Apart from converting the tree-based model to NN, other major efforts BID68 BID52 BID45 proposed to search neural architectures towards a better performance for NNs.

However, most of them merely focused on the non-tabular data in computer vision, speech recognition, and natural language process.

Particularly, their search space just includes specific structures like convolutional layers or pool layers, which are hardly migrated to the learning from tabular data.

Furthermore, such search methods are quite time-consuming they often enumerate the combinations in a large search space.

Given the aforementioned challenges in building or search NN architecture for tabular data, in this paper, we propose an efficient and strategical way to automatically derive effective NN architecture for tabular data, which will be described in details in the following section.

To derive effective NN architecture for tabular data, the design of TabNN follows two key principles:(1) To explicitly leverage expressive feature combinations.

Rather than blindly pouring all features together into FCNN and learning via back-propagation to discover the implicit feature combinations, it will be beneficial to let TabNN explicitly leverage the expressive feature combinations, meaning that the combination of a certain set of features yields great information gain with respect to the learning task.

Compared to learned implicit feature combinations in FCNN, such explicit feature combinations are more robust and can significantly increase the generalization ability of TabNN.

(2) To reduce model complexity.

Too many parameters (i.e. weights or trainable variables) to learn, like FCNN, usually lead to complex optimization hyper-planes so as to result in a high risk of over-fitting.

Therefore, to improve the efficiency as well as the effectiveness of learned NN model, it is critical for TabNN to reduce the complexity of designed NN architecture by removing the unnecessary parameters and encouraging parameter sharing.

In this paper, based on these two principles, we propose a GBDT-powered TabNN.

Specifically, as shown in Alg.

1, TabNN contains four major steps: (1) Automatic Feature Grouping (AFG, Line 2-3): to follow the first principle, this step automatically discovers the effective feature groups (i.e. expressive feature combinations) from a tabular dataset D by leveraging GBDT.

Therefore, the designed NN model can explicitly leverage the feature combinations derived from feature groups.

We employ G to stand for the set of all feature groups.

The cardinal of G, i.e. the number feature groups, may be very large as GBDT often requires many trees to achieve good performance.

(2) Feature Group Reduction (FGR, Line 4): AFG may produce many feature groups and therefore results in too many parameters.

To reduce the parameters and encourage the parameter sharing as guided by the second principle, we cluster these feature groups into k sets, i.e. G 1 , · · · , G k , based on the similarity over these feature groups.

Since there are common features over the clustered feature groups, we can leverage this characteristic to significantly reduce the parameters, by reusing the embedding of these common features in the derived architecture.

(3) Recursive Encoder with Shared Embedding (RESE, Line 6): we design a both effective and efficient NN architecture over clustered tabular feature groups, based on the results of FGR step and the feature group importance powered by GBDT.

(4) Transfer Structural Knowledge from GBDT (TSKG, Line 8-10): beside feature grouping knowledge, trees in GBDT also contain rich structural knowledge.

This step aims at transferring the structural knowledge in GBDT to the obtained NN architecture.

In the rest of this section, we will dive into more details of these steps one by one.

1 Donate the dataset as D, and the number of feature group sets as DISPLAYFORM0 Construct RESE module for Gj with θj 1 Initialize all sets G1, · · · , G k to be empty set 2 Initialize hyper-parameters n and α DISPLAYFORM1 Randomly initialize a processing order π(G) DISPLAYFORM2 Automatic Feature Grouping.

The AFG component is designed under the guidance of the first principle to determine which expressive feature combinations should be explicitly utilized by TabNN.

Since different tabular data may indicate various expressive feature combinations, it is inappropriate to recognize a predefined static feature grouping for all kinds of tabular data.

Therefore, it is necessary to design a dynamical and automatic approach to identify important feature combinations for tabular data.

Although many popular methods, such as correlation test, feature clustering, principal component analysis, etc., can be applied to obtain the feature groups dynamically, they fail to identify expressive complex combination among features within the same group.

On the other hand, the tree-based models provide a goldmine for discovering rich non-linear dependencies among features BID60 .

Specifically, those features within one tree is indeed a well-processed feature group with rich expressiveness.

Inspired by that, it becomes quite natural to use the tree-based model to automatically find expressive feature groups.

Among various options of tree-based method, in this paper, we adopt GBDT for two major reasons: first, GBDT has been widely used to model tabular data of many real-world applications; moreover, as gradient boosting will adjust the learning targets for different trees as latest residuals, GBDT can learn many diverse trees such that it can create many diverse feature groups.

More formally, suppose the set of trees trained in the GBDT model is T , we will use the features within the same tree t ∈ T as a feature group g ∈ G. Since a GBDT often contains many trees to achieve good performance, there will be many feature groups.

The characteristics of GBDT have decided that such feature groups can have many overlapping features, which enable us to merge these feature groups into much compact sets to reduce the complexity by parameter sharing in TabNN.Feature Group Reduction.

To find similar feature groups for parameter sharing, FGR is designed to merge all feature groups into k sets with the minimum number of common features in one set maximized.

More formally, the objective of the FGR is to maximize the value of min 1≤j≤k | g∈Gj g|, where | · | stands for the number of features in the set.

Indeed, there are two major challenges to address FGR.

The first one is the computational complexity, i.e., the FGR problem is NP-hard.

In fact, we have following theorem: Theorem 1.

The NP-hard P m||C max schedule problem can be reduced to the FGR problem.

Proof.

(Sketch.)

In the well known P m||C max problem BID39 , there are n jobs to be scheduled on m identical machines.

Each job j has a process time p j .

The objective is to design a schedule plan to minimize the max load C max of all machines.

To show the hardness of FGR problem, we proof that any instance of P m||C max problem can be reduced to a instance of FGR problem.

W.l.g., we suppose all the process time are integers and their summation is N , i.e. N = n j=1 p j .

Now we set k = m and construct an instance of FGR problem.

We define the total feature sets as F = {1, 2, · · · , N }.

The j-th feature group g j = F \ {N j−1 + 1, · · · , N j }, where N j = j i=1 p i .

These feature groups satisfy the following property: the size of g j1 ∩ g j2 ∩ · ·

· ∩ g jt is exactly N − p j1 − · · · − p jt .

According to this property, we can find that minimizing the load on one machine is equivalent to maximizing the intersection of corresponding feature groups.

Thus, the P m||C max problem can be reduced to the FGR problem, which means the FGR problem is harder than the P m|C max problem.

Consequently, the FGR problem is NP-hard.

Another challenge lies in that, with the increasing number of feature groups, their intersections usually keep shrinking, which unfortunately makes it hard to share weights for similar feature groups.

To address this challenge, we adopt soft intersections instead of the origin one.

We define the α-soft intersection as the set of features which are covered by α fraction of all feature groups.

For convenience, the operator · α is used to calculate the size of α-soft intersection of feature groups.

As the origin FGR is a special case with α = 1, thus, the soft intersection version of FGR is also an NP-hard problem.

Due to the NP-hardness of this problem, it is impossible to compute an optimal solution efficiently even when k is given in advance.

Thus, we take a heuristic approach for the efficiency purpose.

Our algorithm is shown in Alg.

2.

In this algorithm, we enumerate all feature groups in a random order and add it greedily into the feature group set with the greatest gain (Line 7 and 8).

This procedure will be repeated n times and the one with largest minimum α-soft intersection will be selected as the final sets of feature groups (Line 9 and 10).

Although such a simple solution cannot guarantee an optimal solution, it is very efficient and can provide a sub-optimal solution.

Recursive Encoder with Shared Embedding.

After FGR generates k sets of feature groups, it still remains challenging to organize many feature groups within a single set into an efficient NN architecture.

Fortunately, the resulting sets of FGR yield two characteristics that can inspire an efficient design.

In particular, the first one corresponds to diverse importance of different feature groups within one set due to the varying importance of trees in GBDT, and such important difference can stimulate a more efficient recursive NN architecture to let more important feature groups have the more direct impact on the task.

Furthermore, the second characteristic correlates to many common features within one resulting set of FGR caused by the α-soft intersection, and such common features can share parameters for the purpose of efficient learning when constructing the NN architecture.

Inspired by these two characteristics, we propose a recursive encoder with shared embedding (RESE) approach to constructing NN architectures based on feature group sets generated by FGR.

The whole RESE architecture is summarized in FIG1 , in which the circles stand for the neurons and all arrows stand for fully connections to these neurons; and, for convenience, we use smaller indices to represent the layers closer to the output layer.

As shown in this figure, RESE takes advantage of a recursive NN architecture to allow more important feature groups to contribute more directly on the task.

In particular, we first reorder the feature group in each set according to the descending importance of corresponding trees, and define the i-th important feature group in set G j as G j,i .

Then, we arrange more important group as the input of layers closer to the output.

To further reduce the model complexity, RESE is designed to exponentially increase the number of feature groups in layer l i along with increasing i. For example, RESE can put G j,1 (the most important feature group in G j ) as input in layer l 1 , {G j,2 , G j,3 } in layer l 2 , {G j,4 , · · · , G j,7 } in layer l 3 , and so on.

In this way, the designed recursive architecture will be quite compact since it has reduced the number of layers logarithmically.

To encourage using sharing parameters for common features within each feature group set, we first extract the common features from G j and concatenate them as a vectorx j and use it with corresponding embedding component (green circles) as shared input to all layers.

More specifically, as illustrated in FIG1 , the input of each layers consists of the common featuresx j with their embedding components, external feature vector (for layers except l 0 ), and the output of previous layer (the layer with larger index).

For layers l i with 1 ≤ i ≤ log 2 (|G j | + 1) , the external feature vector x j,i is a concatenation of features in {G j,p ∪ G j,p+1 ∪ . . .

∪ G j,2p−1 } \x j , for p = 2 i−1 .

Note that the size of x j,i is small in practice as the common features are excluded.

In summary, the designed architecture arranges the feature groups in an efficient way by leveraging their importance.

Moreover, as shown in FIG1 , the embedding of common featuresx j are reused.

By employing share embedding, we can not only reduce the number of parameters but also result in more efficient back-propagation over common features in the deeper layers.

REMARK: For the completeness of content without disturbing the elaboration of the main design, we organize more details in this remark part.

(1) Shrink of feature representation: for the purpose of information extraction and parameters reduction, the output dimension of each layer is set to be a fraction of the dimension of its input.

Specifically, this fraction is set to 0.25 for raw feature inputs (such asx j and x j,i ), and to 0.5 for other inputs (such as the output of concatenation components).(2) Non-linear activation: for all neurons, we use batch normalization BID32 following by a ReLU BID51 as the non-linear activation.

(3) Multiple FC layers before output: to enhance the expressiveness ability, there are multiple fully connected layers, which is defined by hyper-parameter, between layer l 0 and the output layer.

(4) Final combination: as shown in FIG1 , all outputs of k RESE modules will be concatenated as the inputs of a final fully connected layer.

Transfer Structural Knowledge from GBDT.

Besides the knowledge of expressive feature combination, the GBDT model also contains rich structural knowledge which is quite invaluable to further improve the learning efficiency and model effectiveness.

In this paper, we adopt the knowledge distillation technology BID28 to transfer GBDT's structural knowledge into more effective model initialization for TabNN.Formally, we define corresponding trees of G j in set j as an ordered set T j , and T j,i represents the i-th important tree in G j .

To transfer structural knowledge of these trees, we first use the training data D to go through these trees one by one.

For each data sample, we will get a set of leaf indices output by trees in G j .

As the categorical data are hard to handle by NN, we extend these indices with one-hot representation and denote it as a vector L j,d for the d-th sample in D. Let L j stand for the one-hot vectors for all data samples.

In fact, the pairwise data (D, L j ) can sufficiently represent the structural knowledge over the training data, since different data samples will go through the different paths in trees according to their feature values and finally reach a leaf node.

This part is corresponding to the Line 8 in the Alg.

1.However, as there are many leaf node in the tree set T j , the dimension of L j,d could be very high.

Thus, learning from it could be extremely inefficient.

So we adopt embedding technology BID48 to reduce the dimension while retain important information in L j,d .

To speed up the embedding learning, rather than using the unsupervised AutoEncoder BID4 ) method, we use a FCNN with one hidden layer to learn the embedding supervised.

More specifically, based on bijection relations between leaf indices and leaf values, the one-hot coding L j,d of leaf index is taken as input, while the corresponding leaf value is taken as the training label.

Then the output of the hidden layer is the embedding of L j,d , which is defined as H j,d .

We denote the whole embedding set as H j .

The Line 9 in Alg.

1 is corresponding to this part.

After H j is prepared, we use the data (D, H j ) to pre-train the parameter θ j of the RESE module corresponding to G j .

After all RESE modules are initialized, we can concatenate these modules together and normally train the whole architecture from the ground truths.

In this section, we conduct thorough evaluations on TabNN 1 by comparing its performance with several baseline methods over a couple of public tabular datasets.

Through the experiments, we mainly demonstrate the advantages of TabNN over the tabular data with numerical features and lowcardinality categorical ones, since high-dimension and sparse data, on the other hand, has been well modeled by existing Deep-and-Wide NN methods BID13 .

The basic information of the used tabular datasets are listed in TAB0 and more details can be found in A.1.

We can find that these datasets cover diverse real-world applications.

To ensure an efficient learning of NN over tabular features, we normalize all numerical features and convert categorical features to numerical vectors by tool "categorical-encoding" (Scikit-learn, 2018) with "binary" and "leave-one-out" encoding.

In the following experiments, we compare TabNN with the following baselines: (1) GBDT is a widely used tree-based learning algorithm for modeling tabular data.

(2) FCNN is the traditional NN solution for tabular data.

To achieve the best performance, we use NNI BID47 to search the best hyper-parameter settings.

(3) NRF (GBDT) BID5 converts the regression trees to NN.

For the fair comparison with other baselines, we convert from GBDT to NN, rather than Random Forest BID2 .

As NRF (GBDT) only work for regression tasks, we cannot compare its performance on some classification tasks.

(4) NNRF ) is a recent NN solution for the tabular data.

As NNRF is only designed for classification tasks, we cannot compare with it on all datasets.

Since these baselines are introduced in Sec. 2, we do not provide more details about them in this section due to the space restriction.

To further evaluate the effectiveness of knowledge brought from GBDT, we add two simplified variants of TabNN for comparison: (1) TabNN (R) randomly clusters features into several groups without using any knowledge from GBDT; (2) TabNN (S) only uses structural tree knowledge while keeps using random feature combinations.

Except for FCNN, we did not search the hyper-parameters for TabNN and other baselines.

And for TabNN, we set the learning rate as 0.001, k = 10, and α = 0.5 in all experiments.

More setting details are available in A.2.Overall Performance Comparison.

TAB2 compares the performance of TabNN with all baselines on five tabular datasets.

From this table, we can find that TabNN outperforms all other baselines on all datasets.

In particular, even though we did not search the hyper-parameters for TabNN, it is still much better than well-tuned FCNN.

A further comparison of training curves between TabNN and FCNN, as shown in Fig. 2 , illustrates that TabNN convergences much faster than FCNN.

Moreover, though both NRF (GBDT) and TabNN can leverage the knowledge learned by GBDT, TabNN can further improve the performance and outperform GBDT, while NRF (GBDT) will be over-fitting when continued training the NN converted from GBDT.

In addition, while we have tried many settings to fine-tune NNRF, the performance of it remains even worse than FCNN.

Since NNRF paper only reports experiment results on small datasets (less than ten thousand samples), we hypothesize that NNRF may work well for small datasets but not suitable for large datasets as used in our experiments.

To summarize, all these experimental results demonstrate that TabNN can outperform all kinds of baselines and achieve superior performance in tabular data learning.

Analysis of TabNN.

In this part, we further investigate the importance of key components in TabNN.

The performance gap between TabNN (S) and TabNN, as shown in TAB2 , indicates that feature grouping knowledge from GBDT plays a vital role in TabNN.

Similarly, the comparison between TabNN (S) and TabNN (R), as shown in TAB2 , implies that the structural knowledge from GBDT also yields a large contribution to the performance of TabNN.

Besides the gaps in final performance, tree knowledge also boosts TabNN with faster convergence, as shown in Fig. 2 .

Obviously, these results reveal the importance of tree knowledge brought by GBDT in TabNN.

To disclose the importance of RESE module in parameters reduction, TAB1 shows the number of parameters used for one feature group set with varying number of trees (i.e. feature groups).

The basic approach Concat simply concatenate many FCNNs, each of which is learned based on one feature group.

From this table, we can find that the parameters in Concat increases linearly with growing number of trees, while RESE can significantly reduce the size of parameters logarithmically.

Thus, RESE module indeed plays an important role to reduce the model complexity.

To sum up, these results demonstrate that the key components in TabNN are indispensable to enhance the effectiveness and efficiency in tabular data learning.

Compared with GBDT in Streaming Data Learning.

As mentioned in Sec. 2, one shortage of GBDT is the difficulty in learning from streaming data.

To demonstrate the advantage of TabNN in streaming data, we design a simulation experiment based on FLIGHT data.

Specifically, we use the data samples of the first 4 months in the year 2007 to train an initial model, and then update it once a month, i.e. one batch contains the data corresponding to one month.

On the other hand, we train two GBDT baselines: one is only trained by the data of first 4 months without further learning from steaming data, while another one will continued use streaming data to update leaf values ('refit' function in LightGBM BID46 ).

As shown in TAB3 , GBDT without the sequential update is the worst as expected, while TabNN achieves the best results in streaming data learning.

Furthermore, this table also implies that, with using more data, TabNN can give rise to more significant performance improvement.

These results demonstrate the advantage of TabNN in streaming data learning.

To fill the gap of NN in tabular data learning, we propose a universal neural network solution, called TabNN, which can derive the effective neural architectures automatically for tabular data.

The design of TabNN follows two principles, one as explicitly leveraging expressive feature combinations and the other as reducing model complexity.

Since GBDT is proven to be effective in tabular data, we leverage GBDT to power the implementation of TabNN.

Specifically, TabNN first leverages GBDT to automatically identify expressive feature groups and then clusters feature groups into sets to encourage parameter sharing.

After that, TabNN utilizes tree importance knowledge from GBDT to construct recursive NN architectures.

To enhance the training efficiency and learning performance, tree structural knowledge is also utilized to provide an effective initialization for the derived architecture.

Extensive experiments on various tabular datasets show the advantages of TabNN in modeling tabular data and demonstrate the necessity of designed components in TabNN.

A.1 DATASET DETAILS.The information of the selected tabular datasets are listed in TAB0 .

Among these datasets, YAHOO BID10 and LETOR are both the learning-to-rank datasets, and we solve them as regression problems (i.e. the pointwise ranking problems).

PROTEIN BID63 and A9A BID37 BID53 are the classification datasets, which are downloaded from LIBSVM website BID9

The concrete model setting for all used models is listed in Table 5 .

Table 5 : Model Setting Details.

For all the experiments, we use the released LightGBM BID46 with learning rate 0.15 and strict 100 trees even when meeting multi-class classification dataset.

Besides, the leaf number is set to 0.5 × #feature on most datasets (except 0.4 on A9A), and is limited to the range [32, 128] to avoid underfitting or overfitting.

By the way, the GBDT used in all the other models is set as same.

We use batch normalization and ReLU as activation, and AdamW BID44 as optimizer for FCNN.

As for the hyper-parameters and structures setting, we use NNI BID47 toolkit to run various models with different settings 64 times on each dataset, then select the best one among all models to be our baseline.

The hyper-parameter searching in NNI contains learning rate, batch size and FCNN structure (the number of layers and corresponding hidden units).

The searched best model on each dataset is listed in TAB5 , all of which outperform the human setting one.

NRF(GBDT) Based on the author's released code (JohannesMaxWel, 2018), we change the Random Forest with GBDT in NRF for the fair comparison.

Besides, considering the tree dependency in GBDT, we train the whole sparse neural network converted by NRF at once.

NNRF According to the original paper , we set the depth of NNRF to log 2 (#class) + 1, set the sampled input feature dimension to √ #f eature and use 150 NNs with bootstrapping for ensemble.

In all experiments, we set the learning rate to 0.001, batch size to 128, optimizer to AdamW, k = 10, α = 0.5 and add three hidden layers with 200, 100, 50 units correspondingly before the 20-dimension output of RESE module.

@highlight

We propose a universal neural network solution to derive effective NN architectures for tabular data automatically.

@highlight

A new Neural Network training procedure, designed for tabular data, that seeks to leverage feature clusters extracted from GBDTs.

@highlight

Proposal for a hybrid machine learning algorithm using Gradient Boosted Decision Trees and Deep Neural Networks, with intended research direction on tabular data.