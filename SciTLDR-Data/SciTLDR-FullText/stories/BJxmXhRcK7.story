Recent deep multi-task learning (MTL) has been witnessed its success in alleviating data scarcity of some task by utilizing domain-specific knowledge from related tasks.

Nonetheless, several major issues of deep MTL, including the effectiveness of sharing mechanisms, the efficiency of model complexity and the flexibility of network architectures, still remain largely unaddressed.

To this end, we propose a novel generalized latent-subspace based knowledge sharing mechanism for linking task-specific models, namely tensor ring multi-task learning (TRMTL).

TRMTL has a highly compact representation, and it is very effective in transferring task-invariant knowledge while being super flexible in learning task-specific features, successfully mitigating the dilemma of both negative-transfer in lower layers and under-transfer in higher layers.

Under our TRMTL, it is feasible for each task to have heterogenous input data dimensionality or distinct feature sizes at different hidden layers.

Experiments on a variety of datasets demonstrate our model is capable of significantly improving each single task’s performance, particularly favourable in scenarios where some of the tasks have insufficient data.

Multi-task learning (MTL) (Caruana, 1997; BID15 is an approach for boosting the overall performance in each individual task by learning multiple related tasks simultaneously.

In the deep learning context, jointly fitting sufficiently flexible deep neural networks (DNNs) to data of multiple tasks can be seen as adding an inductive bias to the deep models, which could be beneficial to learn feature representations preferable by all tasks.

Recently, the deep MTL has gained much popularity and been successfully explored in an abroad range of applications, such as computer vision BID32 BID16 , natural language processing BID14 , speech recognition BID5 and so on.

However, a number of key challenges posed by the issues of ineffectiveness, inefficiency and inflexibility in deep MTL are left largely unaddressed.

One major challenge is how to seek effective information sharing mechanisms across related tasks, which is equivalent to designing better parameter sharing patterns in the deep networks.

Some previous work BID32 BID30 tried to solve this problem by means of hard parameter sharing BID21 , where the bottom layers are all shared except with one branch per task at the top layers.

Although being simple and robust to over-fitting BID0 , this kind of architecture can be harmful when learning high-level task-specific features, since it focuses only on common low-level features of all tasks.

Moreover, these common features may be polluted by some noxious tasks, leading to the negative transfer in low-level features among tasks BID31 ).

An alternative line of work mitigate this issue to some extent by following the soft parameter sharing strategy BID21 , under which one separate DNN is learned for each task with its own set of parameters, and the individual DNNs are implicitly linked by imposing constraints on the aligned weights.

The deep MTL models of this type include using 2 norm regularization BID4 , trace norm regularization BID29 and tensor norm priors BID12 BID13 .The lack of efficiency in model complexity gives rise to another great challenge for current deep MTL.

The above soft-sharing based deep models (one set of parameters per task) typically involve enormous number of trainable parameters and require extremely large storage and memory.

It is thus usually infeasible to deploy those deep MTL models on resource-constrained devices such as mobile The overall sharing mechanisms of MRN, two variants of DMTRL (for the setting of CNN) and our TRMTL w.r.t.

two tasks.

The shared portion is depicted in yellow.

The circles, squares and thin rectangles represent tensor cores, matrices and vectors, respectively.

MRN: original weights are totally shared at the lower layers and the relatedness between tasks at the top layers is modeled by tensor normal priors.

DMTRL (TT or Tucker): all layer-wise weights must be equal-sized so as to be stacked and decomposed into factors.

For each task, almost all the factors are shard at each layer except the very last 1D vector.

Such pattern of sharing is identical at all layers.

TRMTL: layer-wise weights are separately encoded into TR-formats for different tasks, and a subset of latent cores are selected to be tied across two tasks.

The portions of sharing can be different from layer to layer.

phones and wearable computers.

BID28 alleviated the issue by integrating tensor factorization with deep MTL and proposed deep multi-task representation learning (DMTRL).

Specifically, they first stack up the layer-wise weights from all tasks and then decompose them into low-rank factors, yielding a succinct deep MTL model with fewer parameters.

Despite the compactness of the model, DMTRL turns out to be rather restricted on sharing knowledge effectively.

This is because, as shown in FIG0 , DMTRL (TT or Tucker) shares almost all fractions of layer-wise weights as common factors, leaving only a tiny portion of weights to encode the task-specific information.

Even worse, such pattern of sharing must be identical across all hidden layers, which is vulnerable to the negative transfer of the features.

As an effect, the common factors become highly dominant at each layer and greatly suppress model's capability in expressing task-specific variations.

The last challenge arises from the flexibility of architecture in deep MTL.

Most of deep MTL models force tasks to have the equal-sized layer-wise weights or input dimensionality.

This restriction makes little sense for the case of loosely-related tasks, since individual tasks' features (input data) can be quite different and the sizes of layer-wise features (input data) may vary a lot from task to task.

In this work, we provide a generalized latent-subspace based solution to addressing aforementioned difficulties of deep MTL, from all aspects of effectiveness, efficiency and flexibility.

Regarding the effectiveness, we propose to share different portions of weights as common knowledge at distinct layers, so that each individual task can better convey its private knowledge.

As for the efficiency, our proposal shares knowledge in the latent subspace instead of original space by utilizing a general tensor ring (TR) representation with a sequence of latent cores BID21 .

One motivation of TR for MTL is it generalizes other chain structured tensor networks , especially tensor train (TT) BID18 , in terms of model expressivity power, as TR can be formulated as a sum of TT networks.

On the other hand, TR is able to approximate tensors using lower overall ranks than TT does , thus yielding a more compact and sparselyconnected model with significantly less parameters for deep MTL.

Adopting TR-format with much lower ranks could bring more benefits to deep MTL if we tensorize a layer-wise weight of each task into a higher-order weight tensor, since the weight can be decomposed into a relatively larger number but smaller-sized cores.

This in turn facilitates the sharing of cores at a finer granularity and further enhances the effectiveness of sharing.

Additionally, BID34

Figure 2: The demonstration of four types generalizations from DMTRL-TT to our TRMTL. affects one specific scale of the resolution.

Such observation also provides a natural inspiration for encoding weights of deep MTL via chain based tensor networks and then sharing the cores across tasks.

For the last challenge, the flexibility of deep MTL networks is maximally retained in our proposal by parameterizing one DNN per task, while the discrepancy between tasks' features is also taken into account by encoding layer-wise weights of different tasks using distinct number of cores.

In this way, the network of each task may possess its own size of weight (or input dimensionality).

We refer to our framework as tensor ring multi-task learning (TRMTL), as depicted in FIG0 .

With above properties, TRMTL achieves the state-of-the-art performance on a variety of datasets and we validate that each individual task can gain much benefit from the proposed architecture.

The classical matrix factorization based MTL BID8 BID20 BID26 requires the dimensionality of weight vectors {w t ∈ R M } T t=1of T tasks to be equal-sized, so that these weights could be stacked up into one weight matrix W ∈ R M ×T .

BID8 assumes W to be low-rank and factorizes it as W = LS.

Here, L ∈ R M ×K consists of K task-independent latent basis vectors, whereas each column vector of S ∈ R K×T is task-specific and contains the mixing coefficients of these common latent bases.

BID28 extended this matrix based MTL to its tensorial counterpart DMTRL by making use of tensor factorization.

Likewise, DMTRL starts by putting the equal-sized weight matrices {W t ∈ R M ×N } T t=1 side by side along the 'task' mode to form a 3rd-order weight tensor W ∈ R M ×N ×T .

In the case of CNN, this weight tensor corresponds to a 5th-order filter tensor K ∈ R H×W ×U ×V ×T .

DMTRL then factorizes W (or K), for instance via TT-format, into 3 TTcores (or 5 TT-cores for K) BID28 .

Analogously, the first 2 TT-cores (or the first 4 TT-cores) play exactly the same role as L for the common knowledge; the very last TT-core is in fact a matrix (similar to S), with each column representing the task-specific information.

Our TRMTL differs widely with DMTRL and generalizes DMTRL from a variety of aspects.

In order to reach TRMTL from DMTRL-TT, one needs to take four major types of generalizations (G1-G4), as demonstrated in Figure 2 .

Firstly (in G1), TRMTL tensorizes the weight into a higherorder weight tensor before factorizing it.

By doing so, the weight can be embedded into more latent cores than that of just 3 cores (or 5 cores) in DMTRL, which yields a more compact model and makes the sharing at a finer granularity feasible.

Secondly (in G2), DMTRL stringently requires that the first D-1 cores (D is weight tensor's order) must be all shared at every hidden layer, only the last vector is kept for private knowledge.

By contrast, TRMTL allows for any sharing pattern at distinct layer.

Thirdly (in G3), there is no need for layer-wise weights to be equal-sized and stacked into one big tensor as in TRMTL, each task may have its individual input dimensionality.

Finally (in G4), TRMTL further generalizes TT to TR-format.

For each task in DMTRL, the first core must be a matrix and the last core must be a vector (with both border rank and outer mode size being 1).

Notice that our TRMTL also conceptually subsumes DMTRL-Tucker in terms of the first three aspects of generalizations (G1-G3).

It is also worth mentioning that BID25 only applies Figure 3 : The diagrams of a 4th-order tensor and its TT-format and TR-format.

DISPLAYFORM0 TR-format for weight compression in a single deep net, whereas ours incorporates a more general tensor network into the deep MTL context.

The two methods differ in goals and applications.

BID13 lately proposed MRN which incorporates tensor normal priors over the parameter tensors of the task-specific layers.

MRN jointly learns the transferable features as well as multilinear relationship among tasks, with the objective to alleviate both under-transfer and negative-transfer of the knowledge.

However, like methods BID32 BID19 BID2 , MRN follows the architecture where all the lower layers are shared, which may harm the transferability if tasks are loosely correlated.

In addition, the relatedness of tasks is captured by the covariance structures over features, classes and tasks.

Constantly updating these covariance matrices (via SVD in BID13 ) becomes computationally prohibitive for large scale networks.

Compared to above mentioned non-latent-subspace methods, TRMTL is highly compact and hence needs much fewer parameters, which is obviously advantageous in tasks with small sample size.

High-order tensors BID6 ) are referred to as multi-way arrays of real numbers.

Let W ∈ R N1×···×N D be a Dth-order tensor in calligraphy letter, where D is called mode or way.

Some very original work have successfully applied tensor decompositions to applications such as imaging analysis BID23 BID24 and computer vision BID22 .

A recent tensor ring decomposition (TR) ) decomposes a tensor W into a sequence 3rd-order latent cores that are multiplied circularly.

An example of TR-format is illustrated in Figure 3 .

In TR-format, any two adjacent latent cores are 'linked' by a common dimension of size R k+1 , k ∈ {1, ..., D}. In particular, the last core is connected back to the first core by satisfying the border rank condition DISPLAYFORM0 Compared with TT-format (Oseledets, 2011), TR generalizes TT by relaxing the border rank condition.

conclude that TR is more flexible than TT w.r.t.

low-rank approximation.

The authors observe the pattern of ranks distribution on cores tend to be fixed in TT.

In TT, the ranks of middle cores are often much larger than those of the side cores, while TR-ranks has no such drawbacks and can be equally distributed on cores.

The authors also claim that, under the same approximation accuracy, the overall ranks in TR are usually much smaller than those in TT, which makes TR a more compact model than TT.

For more favorable properties, such as TR is invariant under circular dimensional permutation, we refer readers to BID21 .

In general, our tensor ring multi-task learning (TRMTL) learns one DNN per task by representing the original weight of each layer with a tensor ring layer (TRL), i.e., utilizing a sequence of TR-cores.

Then, a subset of TR-cores are tied across multiple tasks to encode the task-independent knowledge, while the rest TR-cores of each task are treated as private cores for task-specific knowledge.

We start the section by describing the tensor ring layer (TRL), which lays a groundwork for our TR based deep MTL approach.

Following the TT-matrix BID17 representation, TR is able to represent a large matrix more compactly via TR-matrix format.

Specifically, let DISPLAYFORM0 In this way, one can establish a one-to-one correspondence between a matrix element W(i, j) and a tensor element W((φ 1 (i), ψ 1 (j)), ..., (φ D (i), ψ D (j))) using the compound index (φ k (·), ψ k (·)) for mode k ∈ {1, ..., D}. We formulate the TR-matrix format as DISPLAYFORM1 where DISPLAYFORM2 represents the r 1 th row vector of the DISPLAYFORM3 Notice that the third line in equation 1 implies TRL is more powerful than TT based layer in terms of the modeling expressivity, as TRL can in fact be written as a sum of R 1 TT layers.

In the deep MTL context, the benefits of tensorization in our TRL are twofold : a sparser, more compact tensor network format for each task and a potentially finer sharing granularity across the tasks.

With TRL, the training can be conducted by applying the standard stochastic gradient descent based methods on the cores.

Note that TRL is similar to the recently proposed TR based weight compression BID25 for neural network, but we adopt a different 4th-order latent cores in TRmatrix.

As for CNN setting, one can easily extend TR to a convolutional kernel K ∈ R DISPLAYFORM4

Our sharing strategy is to partition each layer's parameters into task-independent TR-cores as well as task-specific TR-cores.

More specifically, for some hidden layer of an individual task t ∈ {1, ..., T }, we begin with reformulating the layer's weights W t ∈ R Ut×Vt in terms of TR-cores by means of TRL, where DISPLAYFORM0 .

Next, the layer's input tensor H t can be transformed into layer's output tensor DISPLAYFORM1 where the common TR-cores subset {G (·) com } has c elements which can be arbitrarily chosen from the set of all D t cores, leaving the rest cores {G (·) t } as task-specific TR-cores.

Pay close attention that our TRMTL neither restricts on which cores to share, nor restricts the shared cores to be in an consecutive order.

Finally, we reshape tensor Y t back into a vector output y t ∈ R Vt .

Note that the portion of sharing, which is mainly measured by c, can be set to different values from layer to layer.

According to equation 3, TRMTL represents each weight element in weight matrix as function of a sequence product of the slice matrices of the corresponding shared cores and private cores.

Intuitively, this strategy suggests the value of each weight element is partially determined by some common latent factors, and meanwhile, also partially affected by some private latent factors.

Thus, our sharing is carried out in an distributed fashion.

This is more efficient than conventional sharing strategies in which each weight element is either 100% shared or 100% not shared.

There are various strategies on how to select the shard cores w.r.t.

both the location and the number.

BID34 find that distinct cores control an image at different scales of resolution.

The authors demonstrate this by decomposing a tensorized 2D image into TR-cores, and then adding noise to one specific core at a time.

They show the core in the first location controls small-scale patches while the core in the last location influences on large-scale partitions.

Motivated by this, in current work, we preferentially share the features from the detailed scale to the coarse scale, which means we follow a natural left-to-right order in location to select different c number of cores at distinct layers.

A more sophisticated and possible option is to automatically select sharable core pairs that have highest similarity.

We may consider two cores as a candidate pair if the same perturbation of the two cores induces similar changes in the errors of respective tasks.

In this way, one can adaptively select most similar cores from tasks according to a certain threshold, leaving the rest as private cores.

We compare our TRMTL with single task learning (STL), MRN BID13 , two variants of DMTRL BID28 .

To be fair, all the methods are adopted with same network architecture.

We repeat the experiments five times and record the average classification accuracy.

As for the sharing, we tensorize the layer-wise weight into a Dth-order tensor, whose D modes have roughly the same dimensionality, such that the cores are approximately equal if we assume the same TR-ranks.

Therefore, we may measure the faction of sharing by the number of cores c, which is needed to tune via cross validation.

The search space of this hyper-parameter grows rapidly as number of the layers increase.

In practice, we can mitigate this issue a lot by following a useful guidance that this number tends to decrease as the layers increase.

Another solution is to apply a greedy search on c layer by layer to effectively reduce the searching space.

At last, we employ a similar trick introduced in BID28

We conduct our experiments on following datasets : MNIST LeCun et al. (1998) contains handwritten digits from zero to nine.

For this dataset, the task A is to classify the odd digits and the task B is to classify the even ones.

CIFAR-10 (Krizhevsky & Hinton, 2009) contains 60, 000 colour images of size 32 × 32 from 10 object classes.

We assign 10 classes into 3 tasks, in which task A relates to non-animals; task B comprises 4 animal classes including 'cat', 'dog', 'deer' and 'horse'; tasks C contains the remaining 2 classes.

Omniglot (Lake et al., 2015) consists of 1623 unique characters from 50 alphabets.

There are only 20 examples for every character, drawn by a different person at resolution of 105 × 105.

We divide the whole alphabets into five tasks (A to E), each of which links to the alphabets from 10 different languages.

In the Omniglot-MNIST multi-dataset setting, the task A is assigned to classify the first 10 alphabets, while the task B is to recognize 10 digits.

Due to the paper limit, please refer to the appendix for architectures and more experimental results.

In order to see how sharing styles affect our performance, we examine various patterns from three representative categories, as shown in Figure 4 .

For instance, the patterns in 'bottom-heavy' category mean more parameters are shared at the bottom layers than the top layers, while 'top-heavy' indicates the opposite style.

The validation is conducted on MNIST using MLP with three tensorized hidden layers, each of which is encoded using 4 TR-cores.

The pattern '014', for example, means the c are 0, 1 and 4 from lower to higher layers, respectively.

We gauge the transferability between tasks with unbalanced training samples by the averaged accuracy on the small-sample tasks.

Clearly, the 'bottom-heavy' patterns achieve significantly better results than those from the other two categories.

The pattern '420' makes a lot sense and obviously outperforms the pattern '044' in Figure 4, '044' overlaps all weights at the top layers but shares nothing at the bottom layer.

Within each category, TRMTL is robust to small perturbation of c for pattern selection.

For example, also in Table 1 , both '410' and '420' patterns obtain similarly good performance.

As for the model complexity, STL and MRN have enormous 6060K and 3096K parameters, since they share weights in the original space.

DMTRL-Tucker and TT (1800 vs 1800) are parameterized by a large number of parameters of 1194K and 1522K.

With TRMTL, this number is significantly down to 13K.

The huge reduction is mainly due to the tensorization and the resulting more sparser TRL with overall lower ranks.

In this section, we like to verify the effectiveness of different models in transferring the useful knowledge from data-abundant task to data-scarcity task.

To this end, we first test on CIFAR dataset using CNN with settings where each task may have insufficient training samples like 5%, 10% or 50%.

Figure 5 illusrates how the accuracies of one task (two tasks) vary with sample fractions, given the remaining two tasks (one task) get access to the full data.

We observe the trends in which the accuracies of our model exceed the other competitors by a relatively large margin (shown in solid lines), in the cases of limited training samples, e.g., 5% or 10%.

In the mean time, the advantage of our TRMTL is still significant in terms of the averaged accuracies of three tasks (shown in dash lines), which implies the data-scarcity task has little bad influence on the data-abundant tasks.

Our second test is carried out on the Omniglot with CNN architecture.

We now test a more challenging case, where only 1 task (task C) has sufficient samples while the samples of the other 4 tasks (task A, B, D and E) are limited.

FIG2 demonstrates the amount of the accuracy changes for each task, both with and without the aid of the data-rich task.

We observe our TRMTL is able to make the most of the useful knowledge from task C and cause the accuracy to increase for most of the time.

Particularly, the gap of the accuracy enhancement is more obvious for the case of 10% data.

We next show the advantage of our method in handling tasks with heterogeneous inputs.

In this test, the tasks are assigned to input images with different spatial sizes or distinct channels (i.e. RGB or grayscale).

In order to apply DMTRL, one has to first convert the heterogeneous inputs into equalsized features using one hidden layer with totally unshared weights, so that the weights in following layers can be stacked up and factorized.

To better show the influence of heterogeneous inputs on the competitors, we adopt MLP with 4 hidden layers.

For a good pattern of our TRMTL, such as '5410', the first hidden layer of each task is encoded into 6 TR-cores, 5 of which can be shared.

As recorded in TAB3 , DMTRL based methods behave significantly worse than our TRMTL by a very large margin.

The poor performance of DMTRL is induced by fact that lowest features from related tasks cannot be shared at all because of the heterogeneous input dimensionality.

Our TRMTL also finds its usefulness when applied to multiple datasets, where the tasks are loosely related.

We verify this through recognizing character symbols (task A on Omniglot) and handwritten digits (task B on MNIST) at the same time.

Task A is much harder than task B, as each character in task A has much fewer training samples.

TRMTL is established using three hidden layers with 5 cores at each layer.

Task A and B are partially shared by 2 cores at the first layer.

To apply DMTRL, we use a similar strategy as previous section.

As expected, TRMTL outperforms other methods and TRMTL-211 significantly improves task A by 4.2%, 4.9% and 4.7% w.r.t.

STL, whereas both DMTRL-Tucker and TT fail in the Omniglot task with poor accuracies.

Table 3 : The results of multi-dataset tasks on Omniglot (task A) and MNIST (task B).

In this paper, we have introduced a novel knowledge sharing mechanism for connecting task-specific models in deep MTL, namely TRMTL.

The proposed approach models each task separately in the form of TR representation using a sequence latent cores.

Next, TRMTL shares the common knowledge by ting any subset of layer-wise TR cores among all tasks, leaving the rest TR cores for private knowledge.

TRMTL is highly compact yet super flexible to learn both task-specific and task-invariant features.

TRMTL is empirically verified on various datasets and achieves the stateof-the-art results in helping the individual tasks to improve their overall performance.

Table 4 : Performance comparison of STL, MRN, DMTRL and our TRMTL on CIFAR-10 with unbalanced training samples, e.g., '5% vs 5% vs 5%' means 5% of training samples are available for the respective task A, task B and task C. TR-ranks R = 10 for TRMTL.

In this section, we conduct more experiments on CIFAR-10 dataset.

We adopt the following architecture: We show more results on the effectiveness of different models when transferring the useful knowledge from data-abundant task to data-scarcity task.

For this purpose, we begin with the test cases where all of task have insufficient training samples, e.g., '5% vs 5% vs 5%'.

After that, we compare the precision improvement of the individual task(s) when the other task(s) is (are) equipped with the whole training data.

Table 4 records the results of our two best patterns ('4431' and '4421') , as well as the one with 'bad' pattern '4444'.

Clearly, TRMTL ('4431' and '4421') outperforms other methods in nearly all the cases.

As for task A, for instance, the precision of TRMTL-4431 is increased by 1.7% when the data of the task C becomes 100%.

Even more, such enhancement further grows up to 5.5% in the situation that both task B and C's training samples are fully available.

This is in contrast to MRN whose precision improvements are merely 0.4% and 3.0% in the corresponding scenarios.

Again, the performance of TRMTL-4431 is superior to that of TRMTL-4444, indicating sharing all nodes like '4444' is not a desirable style.

DISPLAYFORM0 It is also interesting to get an idea on what our model has learned via the visualization of the high level features.

FIG3 illustrates the task-specific features of our TRMTL (and DMTRL) using t-SNE for the dimensionality reduction.

We can see a clear pattern of the clustered features produced by our model that are separated for different classes, which could be more beneficial the downstream classification tasks.

For this dataset, we adopt a similar architecture as in the previous experiment for CNN as where the last two convolution layers and first fully connected layer are represented using TRL with the input/output feature modes of TR-cores being {2, 2, 2}, {4, 2, 2}, and {2, 2, 2, 2}, {4, 4, 2, 2}, and {18, 12, 12, 9}, {4, 4, 4, 4}. The best sharing pattern of our model is '432', which is selected by CV.

TAB7 summarizes the performance of the compared methods when the distinct fractions of data are used as training data.

Our TRMTL obtains the best overall performance in both data-rich and data-scarcity situations. [4, 4, 4, 4, 4, 6] [4, 4, 4, 4, 4, 6] [4, 4, 4, 4, 4, 6]

@highlight

a deep multi-task learning model adapting tensor ring representation

@highlight

A variant of tensor ring formulation for multi-task learning by sharing some of the TT cores for learning "common task" while learning individual TT cores for each separate task