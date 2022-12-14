Many real applications show a great deal of interest in learning multiple tasks from different data sources/modalities with unbalanced samples and dimensions.

Unfortunately, existing cutting-edge deep multi-task learning (MTL) approaches cannot be directly applied to these settings, due to either heterogeneous input dimensions or the heterogeneity in the optimal network architectures of different tasks.

It is thus demanding to develop knowledge-sharing mechanism to handle the intrinsic discrepancies among network architectures across tasks.

To this end, we propose a flexible knowledge-sharing framework for jointly learning multiple tasks from distinct data sources/modalities.

The proposed framework allows each task to own its task (data)-specific network design, via utilizing a compact tensor representation, while the sharing is achieved through the partially shared latent cores.

By providing more elaborate sharing control with latent cores, our framework is effective in transferring task-invariant knowledge, yet also being efficient in learning task-specific features.

Experiments on both single and multiple data sources/modalities settings display the promising results of the proposed method, especially favourable in insufficient data scenarios.

Multi-task learning (MTL) (Caruana, 1997; Maurer et al., 2016) is an approach for boosting the overall performance of each individual task by learning multiple related tasks simultaneously.

In the deep learning setting, jointly fitting sufficiently flexible deep neural networks (DNNs) to data of multiple tasks can be seen as adding an inductive bias to the deep models, which can facilitate the learning of feature representations that are preferable by all tasks.

Recently, the deep MTL has been successfully explored in a broad range of applications, such as computer vision (Zhang et al., 2014; Misra et al., 2016) , natural language processing (Luong et al., 2015; , speech recognition Huang et al., 2015) and so on.

Nevertheless, one key challenge in deep MTL remains largely unaddressed, that is, almost all existing deep MTL approaches (Yang & Hospedales, 2017; Long et al., 2017) restrict themselves only to the setting of multi-label learning (or multi-output regression) (Zhang & Yang, 2017) .

In other words, different tasks must be fed with input data from the same source (or domain).

Such requirement, however, seriously limits the applicability of those models to a more realistic scenario of deep MTL, where the tasks involve distinct data sources (domains) with unbalanced sample sizes or dimensions.

More specifically, tasks from some domains with abundant samples or small input dimensions are relatively easy to handle, whereas tasks from other domains are quite challenging due to the insufficient training data and large dimensionality.

For instance, classifying hand-written digits (MNIST dataset (LeCun et al., 1998) ) is somewhat similar to the recognition of hand-drawn characters (Omniglot dataset (Lake et al., 2015) ).

The Omniglot task is much harder than the MNIST task, as each character in Omniglot has only 20 training samples, while the input dimensionality is about 15 times larger than MNIST digit.

As another example, predicting binary attributes (i.e., 'young', 'bald', 'receding hairline') from human face images (CelebA dataset (Liu et al., 2015) ) ought to be related to the age group classification using human photos taken in the wild (Adience dataset (Eidinger et al., 2014) ).

The Adience task turns out to be the more difficult one since the wild images are not preprocessed and 7.6 times fewer than CelebA samples.

Hence, it makes good sense to jointly learn these multi-task representation learning (DMTRL) for CNN setting and our TRMTL (general setting and CNN setting) w.r.t.

two tasks.

The shared portion is depicted in yellow.

MRN: original weights are totally shared at the lower layers and the relatedness between tasks at the top layers is modelled by tensor normal priors.

DMTRL (TT or Tucker): all layer-wise weights must be equal-shape so as to be stacked and decomposed into factors.

For each task, almost all the factors are shard at each layer except the very last 1D vector.

Such pattern of sharing is identical at all layers.

TRMTL (General): layer-wise weights are separately encoded into TR-formats for different tasks, and a subset of latent cores are selected to be tied across two tasks.

The portions of sharing can be different from layer to layer.

TRMTL (CNN): spatial cores (height and width cores) in the tensorized convolutional kernel are shared, while cores of input/output channels of the kernel are task-specific.

tasks to extract better feature representations, especially for the hard tasks, which could be achieved through transferring domain-specific knowledge from easy tasks.

Unfortunately, existing cutting-edge deep MTL models are only suited for the multi-label learning where different tasks share the same training inputs (i.e., X i = X j for i = j, where X i denotes the input for task T i ), and thus cannot be directly applied to above learning scenarios.

This is due to those models fail to provide knowledge-sharing mechanisms that can cope with the intrinsic discrepancies among network architectures across tasks.

Such discrepancies either arise from the heterogeneous dimensions of input data or from the heterogeneous designs of layer-wise structures.

Conventionally, knowledge-sharing mechanisms of deep MTL can be hard or soft parameter sharing (Ruder, 2017) .

Hard sharing models (Zhang et al., 2014; Yin & Liu, 2017) share all parameters at the lower layers but with no parameters being shared at the upper layers across tasks.

Soft sharing models (Duong et al., 2015; Yang & Hospedales, 2016; Long & Wang, 2015) , on the other hand, learn one DNN per task with its own set of parameters, and the tasks are implicitly connected through imposing regularization terms on the aligned weights.

The common issue with above mechanisms is that, for the sharing part, the network architectures of all tasks are strictly required to be identical.

It turns out that some of the tasks have to compromise on a sub-optimal network architecture, which may lead to the deterioration in the overall performance.

Ideally, at all potentially shared layers, each task should be capable of encoding both task-specific and task-independent portions of variation.

To overcome this limitation, we propose a latent-subspace knowledge-sharing mechanism that allows to associate each task with distinct source (domain) of data.

By utilizing tensor representation, different portions of parameters can be shared via latent cores as common knowledge at distinct layers, so that each task can better convey its private knowledge.

In this work, we realize our proposed framework via tensor ring (TR) format and refer it as tensor ring multi-task learning (TRMTL), as shown in Figure 1 .

Our main contributions are twofold: (1) we offer a new distributed knowledge-sharing mechanism that can address the discrepancies of network architectures among tasks.

Compared to existing deep MTL models that are only for multi-label learning, the joint learning of tasks from multi-datasets (multi-domains) with heterogeneous architectures becomes feasible.

(2) we provide a TR-based implementation of the proposed framework, which further enhances the performance of deep MTL models in terms of both compactness and expressive power.

High-order tensors (Kolda & Bader, 2009 ) are referred to as multi-way arrays of real numbers.

Let W ??? R N1??????????N D be a Dth-order tensor in calligraphy letter, where D is called mode or way.

Some original work have successfully applied tensor decompositions to applications such as imaging analysis and computer vision (Vasilescu & Terzopoulos, 2002; 2003) .

As a special case of tensor networks, the recent TR decomposition ) decomposes a tensor W into a sequence of 3rd-order latent cores that are multiplied circularly.

An example of TR-format is illustrated in Figure 2 .

In TR-format, any two adjacent latent cores are 'linked' by a common dimension of size R k+1 , k ??? {1, ..., D}. In particular, the last core is connected back to the first core by satisfying the border rank condition

Compared with tensor train (TT) format (Oseledets, 2011), TR generalizes TT by relaxing the border rank condition. concludes that TR is more flexible than TT w.r.t.

low-rank approximation.

The authors observe the pattern of ranks distribution on cores tend to be fixed in TT.

In TT, the ranks of middle cores are often much larger than those of the side cores, while TR-ranks has no such drawbacks and can be equally distributed on cores.

The authors find that, under the same approximation accuracy, the overall ranks in TR are usually much smaller than those in TT, which makes TR a more compact model than TT.

Our general framework learns one DNN per task by representing the original weight of each layer with a tensor representation layer, i.e., utilizing a sequence of latent cores.

Then, a subset of cores are tied across multiple tasks to encode the task-independent knowledge, while the rest cores of each task are treated as private cores for task-specific knowledge.

We start the section by describing the tensor representation layer, which lays a groundwork for our deep MTL approach.

Our TR-based implementation is called tensor ring representation layer (TRRL).

Following the spirit of TT-matrix (Novikov et al., 2015) representation, TR is able to represent a large matrix more compactly via TR-matrix format.

Specifically, let W be a matrix of size

In this way, one can establish a one-to-one correspondence between a matrix element W(i, j) and a tensor element W((?? 1 (i), ?? 1 (j)), ..., (?? D (i), ?? D (j))) using the compound index (?? k (??), ?? k (??)) for mode k ??? {1, ..., D}. We formulate the TR-matrix format as

where 'Tr' is the trace operation.

G (k) denotes the kth latent core, while

Notice that the third line in equation 1 implies TRRL is more powerful than TT layer in terms of model expressivity, as TRRL can in fact be written as a sum of R 1 TT layers.

In the deep MTL context, the benefits of tensorization in our TRRL are twofold : a sparser, more compact tensor network format for each task and a potentially finer sharing granularity across the tasks.

3.2 THE PROPOSED KNOWLEDGE-SHARING FRAMEWORK 3.2.1 THE GENERAL FORMULATION Our sharing strategy is to partition each layer's parameters into task-independent TR-cores as well as task-specific TR-cores.

More specifically, for some hidden layer of an individual task t ??? {1, ..., T }, we begin with reformulating the layer's weights W t ??? R Ut??Vt in terms of TR-cores by means of TRRL, where

.

Next, the layer's input tensor H t can be transformed into layer's output tensor

where the common TR-cores subset {G (??) com } has C elements which can be arbitrarily chosen from the set of all D t cores, leaving the rest cores {G (??) t } as task-specific TR-cores.

Pay attention that our TRMTL neither restricts on which cores to share, nor restricts the shared cores to be in a consecutive order.

Finally, we reshape tensor Y t back into a vector output y t ??? R Vt .

Note that the portion of sharing, which is mainly measured by C, can be set to different values from layer to layer.

According to equation 2, TRMTL represents each weight element in weight matrix as function of a sequence product of the slice matrices of the corresponding shared cores and private cores.

Intuitively, this strategy suggests the value of each weight element is partially determined by some common latent factors, and meanwhile, also partially affected by some private latent factors.

Thus, our sharing is carried out in an distributed fashion.

This is more efficient than conventional sharing strategies in which each weight element is either 100% shared or 100% not shared.

Although we describe our general framework in terms of TR format, it is straightforward to implement our framework with other tensor network representations, such as Tucker (Tucker, 1966) , TT (Novikov et al., 2015) , projected entangled pair states (PEPS) and entanglement renormalization ansatz (MERA) , as long as each layer-wise weight matrix is tensorized and decomposed into a sequences latent cores.

Our model can be easily extended to convolutional kernel K ??? R H??W ??I??O , where H ?? W is the spatial sizes and I and O are the input and output channels.

Note that here TRRL is similar to TR based weight compression Wang et al. (2018) , but we use a different 4th-order latent cores in TR-matrix.

As one special case of our general framework (TRMTL-CNN), we just share the spatial cores (height cores and width cores) in the tensorized kernel (via TRRL), while cores corresponding to input/output channels may differ from task to task :

where C is typically 1 for small-sized spatial dimensions.

Thus, there is no need to specify how many and which cores to share for TRMTL-CNN.

4 EXPERIMENTAL RESULTS

We compare our TRMTL with single task learning (STL), MRN (Long et al., 2017) , two variants of DMTRL (Yang & Hospedales, 2017) .

We repeat the experiments 5 times and record the average accuracy.

The detailed settings and more experimental results are in the supplementary material.

Before the sharing, we first tensorize the layer-wise weight into a Dth-order tensor, whose D modes have roughly the same dimensionality, such that the cores have approximately equal sizes if we assume the same TR-ranks.

In this manner, we may measure the fraction of knowledge sharing by the number of shared cores.

D is empirically set to be from 4 to 6 in most of our tests.

For simplicity, we assume TR-ranks to be identical for all TR-cores across layers for each task.

We choose TR-ranks by cross validation on a range of values among 5, 10 and 15.

Note that there is no tensorization step in DMTRL in (Yang & Hospedales, 2017) , and DMTRL selects TT-ranks via tensor decomposition according to some specified threshold.

Our general TRMTL is highly flexible as we impose no restrictions on which cores to be shared and where to share across tasks.

In practical implementation, we may need to trade-off some model flexibility for the ease of sharing-pattern selection by introducing some useful prior knowledge about the domain.

For instance, many vision tasks tend to share more cores at the lower layers than upper layers.

There are various strategies on how to select the shard cores w.r.t.

both the location and the number.

Authors of (Zhao et al., 2017 ) discover that distinct cores control an image at different scales of resolution.

The authors demonstrate this by decomposing a tensorized 2D image and then adding noise to one specific core at a time.

They show the core in the first location controls smallscale patches while the core in the last location influences on large-scale partitions.

Motivated by this, under the general formulation 3.2.1, we preferentially share the features from the detailed scale to the coarse scale, which means we follow a natural left-to-right order in location to select different C number of cores at distinct layers.

C is needed to tune via cross validation.

In practice, we apply a greedy search on C layer by layer to effectively reduce the searching space.

Another practical option is to prune the searching space by following the very useful guidance that C tends to decrease as the layers increase.

For certain CNN based architectures, we adopt the special case TRMTL-CNN.

Since the cores produced by tensorized convolutional kernel have their specific roles, we just share the cores that are associated to the spatial dimensions (height and width cores), leaving input/output cores being task-specific.

In our tests, C is just 1 due to the small spatial kernels, thus eliminating the need of the tuning of this hyper-parameter.

We begin our test with data from single domain source to validate the basic properties of our model.

Our first validation test is conducted on MNIST, where the task A is to classify the odd digits and the task B is to classify the even ones.

To see how sharing styles and hyper-parameter C can affect the performance, we examine various patterns from three representative categories, as shown in Figure 3 .

For instance, the patterns in 'bottom-heavy' category mean more parameters are shared at the bottom layers than the top layers, while 'top-heavy' indicates the opposite style.

The validation is conducted on MNIST using multi-layer perceptron (MLP) with three tensorized hidden layers, each of which is encoded using 4 TR-cores.

The pattern '014', for example, means the C are 0, 1 and 4 from lower to higher layers, respectively.

We gauge the transferability between tasks with unbalanced training samples by the averaged accuracy on the small-sample tasks.

Clearly, the 'bottom-heavy' patterns achieve significantly better results than those from the other two categories.

The pattern '420' is reasonable and obviously outperforms the pattern '044' in Figure 3 , since '044' overlaps all weights at the top layers but shares nothing at the bottom layer.

Within each category, TRMTL is robust to small perturbation of C for pattern selection, both '410' and '420' obtain similarly good performance.

We also examine the complexity of the compared models on MNIST.

In Table 1 , STL and MRN have enormous 6, 060K and 3, 096K parameters, since they share weights in the original space.

DMTRLTucker and TT with pre-train trick (Yang & Hospedales, 2017) are parameterized by 1, 194K and 1, 522K parameters.

In contrast, TRMTL achieves the best accuracies while the numbers of parameters are significantly down to 16K and 13K.

The huge reduction is due to the tensorization and the resulting more sparser TRRL with overall lower ranks.

Our next validation is carried out on Omniglot dataset (Krizhevsky & Hinton, 2009 ) to verify efficacy of knowledge transfer from data-abundance tasks to data-scarcity ones within one source of data domain.

Omniglot data Lake et al. (2015) consists of 1, 623 unique characters from 50 alphabets with resolution of 105 ?? 105.

We divide the whole alphabets into 5 tasks (task A to task E), each of which links to the alphabets from 10 languages.

We now test a more challenging case, where only 1 task (task C) has sufficient samples while the samples of the other 4 tasks are limited.

Figure 4 demonstrates the amount of the accuracy changes for each task, both with and without the aid of the data-rich task.

We observe our TRMTL is able to make the most of the useful knowledge from task C and significantly boosts the accuracies of all other tasks.

In our last validation, we like to explore whether the proposed sharing mechanism also works for recurrent neural networks.

Hence, we test on UFC11 dataset (Liu et al., 2009 ) that contains 1, 651 Youtube video clips from 11 actions, which are converted to the resolution of 120 ?? 180 ?? 3.

We assign 5 actions ('basketball', 'biking', 'diving', 'golf swinging' and 'horse back riding') to the task A and leave the rest 6 actions ('soccer juggling', 'swinging', 'tennis swinging', 'trampoline jumping', 'volleyball spiking' and 'walking') as the task B. The RNN is implemented using onelayer long short-term memory (LSTM) with input length of 190.

The weights corresponding to the input video are tensorized and encoded into 4 TR-cores, whose input and output dimensions are [8, 20, 20, 18] and [4, 4, 4, 4] , respectively.

The TR-rank is set to [2, 2, 2, 2, 2].

Only one layer of cores need to be shared and they are shared in a left-to-right order.

The recognition precisions w.r.t.

the number of shared cores are recorded in Table 2 .

We find that sharing TR-cores between tasks via our TRMTL significantly improves the performance comparing to no sharing case, and sharing all 4 TR-cores achieves the best results for this RNN situation.

In this section, we show the key advantage of our method in handling multiple tasks defined on distinct data domains, where the optimal network architectures of the tasks could be different. [7, 7, 5, 5, 3, 3] [7, 7, 4, 4] out modes [7, 7, 5, 5, 3, 3] [7, 7, 5, 5] FC2 in modes [7, 7, 5, 5, 3, 3] [7, 7, 5, 5] out modes [7, 7, 5, 5, 3, 3] [7, 7, 5, 5] FC3 in modes [7, 7, 5, 5, 3, 3 We first verify on Omniglot and MNIST combination, where task A is to classify hand-drawn characters from first 10 alphabets, while task B is to recognize 10 hand-written digits.

Task A is much harder than task B, as each character in task A has a very fewer training samples (only 20 per character).

Table 3 shows the architecture specification of TRMTL using 4 layers MLP, we can see task A and task B possess their respective layer-wise network structures, while different portions of cores could be partially shared across layers.

In contrast, to apply DMTRL, one has to first convert the heterogeneous inputs into equal-sized features using one hidden layer with totally unshared weights, so that the weights in following layers with same shape can be stacked up.

In Table 4 , TRMTL obtains similar results to its competitors for the easier MNIST task, while both TRMTL-200 and 211 significantly outperform STL and DMTRL by a large margin for the more difficult Omniglot task.

The poor performance of DMTRL due to its architecture's not being able to share any feature at the bottom hidden layer but has to share almost all the features at upper layers.

We also conduct experiments on the challenging Office-Home dataset (Venkateswara et al., 2017) to evaluate the effectiveness of TRMTL in handling data from distinct domains.

The dataset contains over 10,000 images collected from different domains including Art, Clipart, Product and Real World, which forms task A to task D, respectively.

Each task is assigned to recognize 65 object categories presenting in offices or homes.

The image styles and the levels of difficulty vary from task to task, e.g., images from Product (task C) have empty backgrounds while images of Art (task A) have complicated backgrounds.

We train three FC layers on the features extracted from images of each task using pre-trained VGG-16 (Simonyan & Zisserman, 2014) .

In Figure 5 , our TRMTL variants consistently outperform other competitors by a large margin, i.e., over 5% in accuracy for the toughest task A when 80% samples are available.

The noticeable improvements are mainly credited to our sharing mechanism, which effectively shares the common signature of object identity across tasks regardless of their individual image styles.

For TRMTL, we observe TRMTL-HT exceeds TRMTL-HM by at least 2% in the averaged accuracy and by 1% in the hardest task A, showing the efficacy of employing non-identical architectures on sharing high-level features.

To further illustrate the merit of sharing knowledge using heterogeneous architectures, we next apply our TRMTL directly to the raw images via CNNs.

We test on two large-scale human face datasets: Adience Eidinger et al. (2014) and CelebA Liu et al. (2015) .

Adience dataset contains 26, 580 unfiltered 227 ?? 227 face photos taken in the wild with variation in appearance, pose, lighting and etc; CelebA has a total number of 202, 599 preprocessed face images of resolution 218 ?? 178.

For this test, the task A assigned to Adience data is to predict the label of age group that a person belongs to (8 classes), and we associate the task B to CelebA data to classify 40 binary facial attributes.

Note that task A is much harder than task B, as the number of samples in Adience (with face images in the wild) is about 7.6 times fewer than that of CelebA (with cropped and aligned face images).

Since Adience bears similarity to CelebA, we are interested to see whether the performance of the tough task (task A) can be enhanced by jointly learning on two domains of data.

The heterogeneous architectures of TRMTL-CNN are shown in Table 5 , in which we adopt the special case of TRMTL by sharing the spatial cores (e.g., [7, 7] , [5, 5] and [3, 3] ) in convolutional kernel yet preserving the differences w.r.t.

input and output channel cores.

We focus on comparing the heterogeneous case with the homogeneous case in Table 5 where the shared structures are identical.

As expected, in Figure 5 , our TRMTL significantly outperforms other methods on the hard task A. In the meantime, TRMTL obtains the best averaged accuracies on two tasks in nearly all cases, indicating the data-scarcity task A has little harmful impact on the data-abundant task B. For our TRMTL, we also observe TRMTL-HM exhibits worse accuracies than TRMTL-HT, which implies that a comprise on an identical CNN design for all tasks, such as input/output channel core and stride size, lead to deteriorated overall performance.

The test also shows the effectiveness of TRMTL in sharing low-level features with heterogeneous architectures.

Our general TRMTL framework relies on the manual selection of shared cores, i.e., one need to specify the number of shared cores C at each layer if we choose to share the cores in a left-to-right order across tasks.

Although we can employ some efficient heuristics, the search space of this hyperparameter may grow rapidly as number of the layers increase.

Besides the greedy search, a more sophisticated and possible option is to automatically select sharable core pairs that have the highest similarity.

We may consider two cores as a candidate pair if the same perturbation of the two cores induces similar changes in the errors of respective tasks.

In this way, one can adaptively select most similar cores from tasks according to a certain threshold, leaving the rest as private cores.

We should also point out that tensorization operation plays a key role in our proposed sharing mechanism.

Due to the tensorization, the cores can be shared in a much finer granularity via our TRMTL framework.

Furthermore, tensorizing weight matrix into high-order weight tensor yields more compact tensor network format (with much lower overall ranks), and thus a higher compression ratio for parameters.

In contrast, DMTRL tends to produce a lot more parameters without tensorization.

In this work, we have extended the conventional deep MTL to a broader paradigm where multiple tasks may involve more than one source data domain.

To resolve the issues caused by the discrepancies among different tasks' network structures, we have introduced a novel knowledge sharing framework for deep MTL, by partially sharing latent cores via tensor network format.

Our method is empirically verified on various learning settings and achieves the state-of-the-art results in helping tasks to improve their overall performance.

of T tasks to be equal-sized, so that these weights could be stacked up into one weight matrix W ??? R M ??T .

The work (Kumar & Daume III, 2012 ) assumes W to be low-rank and factorizes it as W = LS.

Here, L ??? R M ??K consists of K task-independent latent basis vectors, whereas each column vector of S ??? R K??T is task-specific and contains the mixing coefficients of these common latent bases.

Yang & Hospedales (2017) extended this to its tensorial counterpart deep multi-task representation learning (DMTRL) by making use of tensor factorization.

Likewise, DMTRL starts by putting the equal-shaped weight matrices

side by side along the 'task' mode to form a 3rd-order weight tensor W ??? R M ??N ??T .

In the case of CNN, this weight tensor corresponds to a 5th-order filter tensor K ??? R H??W ??U ??V ??T .

DMTRL then factorizes W (or K), for instance via TT-format, into 3 TT-cores (or 5 TT-cores for K) Yang & Hospedales (2017) .

Analogously, the first 2 TT-cores (or the first 4 TT-cores) play exactly the same role as L for the common knowledge; the very last TT-core is in fact a matrix (similar to S), with each column representing the task-specific information.

The fundamental difference between our TRMTL and DMTRL is that ours can tailor heterogeneous network structures to various tasks.

In contrast, DMTRL is not flexible enough to deal with such variations with tasks.

Specifically, our TRMTL differs widely with DMTRL and generalizes DMTRL from a variety of aspects.

In order to reach TRMTL from DMTRL-TT, one needs to take four major types of generalizations (G1-G4), as shown in Figure 6 .

Firstly (in G1), TRMTL tensorizes the weight into a higher-order weight tensor before factorizing it.

By doing so, the weight can be embedded into more latent cores than that of just 3 cores (or 5 cores) in DMTRL, which yields a more compact model and makes the sharing at a finer granularity feasible.

Secondly (in G2), DMTRL stringently requires that the first D-1 cores (D is weight tensor's order) must be all shared at every hidden layer, only the last vector is kept for private knowledge.

By contrast, TRMTL allows for any sharing pattern at distinct layer.

Thirdly (in G3), there is no need for layerwise weights to be equal-sized and stacked into one big tensor as in TRMTL, each task may have its individual input domains.

Finally (in G4), TRMTL further generalizes TT to TR-format.

For each task in DMTRL, the first core must be a matrix and the last core must be a vector (with both border rank and outer mode size being 1).

Notice that our TRMTL also conceptually subsumes DMTRLTucker in terms of the first three aspects of generalizations (G1-G3).

It is also worth mentioning that (Wang et al., 2018) only applies TR-format for weight compression in a single deep net, whereas ours incorporates a more general tensor network framework into the deep MTL context.

The authors of (Long et al., 2017 ) lately proposed multilinear relationship network (MRN) which incorporates tensor normal priors over the parameter tensors of the task-specific layers.

However, like methods (Zhang et al., 2014; Ouyang et al., 2014; Chu et al., 2015) , MRN follows the architecture where all the lower layers are shared, which is also not tailored for the extended MTL paradigm, and may harm the transferability if tasks are not that tightly correlated.

In addition, the relatedness of tasks is captured by the covariance structures over features, classes and tasks.

Constantly updating these covariance matrices (via SVD in (Long et al., 2017) ) becomes computationally prohibitive for large scale networks.

Compared to these non-latent-subspace methods, TRMTL is highly compact and needs much fewer parameters, which is obviously advantageous in tasks with small sample size.

The detailed specification of network architecture and factorized TRRL representation of the experiments on MNIST dataset are recorded in Table 6 .

In Table 7 , our TRMTL achieves the best results and is robust to small perturbation of C for pattern selection, since both '410' and '420' patterns obtain similarly good performance.

For the Omniglot Dataset, we adopt a similar architecture as in the previous experiment for CNN as

, where the last two convolution layers and first fully connected layer are represented using TRRL with the input/output feature modes of TR-cores being {2, 2, 2}, {4, 2, 2}, and {2, 2, 2, 2}, {4, 4, 2, 2}, and {18, 12, 12, 9}, {4, 4, 4, 4}. Table 8 displays the details of network specification.

The best sharing pattern of our model is '432'.

Figure 7 demonstrates the amount of the accuracy changes for each task (for the case of 50% training data), both with and without the aid of the datarich task.

Table 9 summarizes the performance of the compared methods when the distinct fractions of data are used as training data.

Our TRMTL obtains the best overall performance in both data-rich and data-scarcity situations.

In this section, we also conduct more experiments on CIFAR-10 dataset.

We assign 10 classes into 3 tasks, in which task A relates to non-animals; task B comprises 4 animal classes including 'cat', 'dog', 'deer' and 'horse'; task C contains the remaining 2 classes.

We like to verify the performance of different models in transferring the useful knowledge from data-abundant task to data-scarcity task within one source of data domain.

To this end, we first test on CIFAR dataset with settings where each task may have insufficient training samples like 5%, 10% or 50%.

For this test, we adopt the following architecture:

, where C3 stands for a 3 ?? 3 convolutional layer.

We employ the general form of TRL on the last two CNN layers and first two FC layers where the most of the parameters concentrate, yielding 4 TR-cores per layer.

Figure 8 illustrates how the accuracies of one task (two tasks) vary with sample fractions, given the remaining two tasks (one task) get access to the full data.

We observe the trends in which the accuracies of our model exceed the other competitors by a relatively large margin (shown in solid lines), in the cases of limited training samples, e.g., 5% or 10%.

In the mean time, the advantage of our TRMTL is still significant in terms of the averaged accuracies of three tasks (shown in dash lines), which implies the data-scarcity task has little bad influence on the data-abundant tasks.

Table 10 reports the results of our two best patterns ('4431' and '4421') , as well as the one with 'bad' pattern '4444'.

Clearly, TRMTL ('4431' and '4421') outperforms other methods in nearly all the cases.

As for task A , for instance, the precision of TRMTL-4431 is increased by 1.7% when the data of task C becomes 100%.

Even more, such enhancement further grows up to 5.5% in the situation that both task B and C's training samples are fully available.

This is in contrast to MRN whose precision improvements are merely 0.4% and 3.0% in the corresponding scenarios.

Again, It is also interesting to get an idea on what our model has learned via the visualization of the high level features.

Figure 9 illustrates the task-specific features of our TRMTL (and DMTRL) using t-SNE for the dimensionality reduction.

We can see a clear pattern of the clustered features produced by our model that are separated for different classes, which could be more beneficial the downstream classification tasks.

In this section, we show the advantage of our method in handling tasks with heterogeneous inputs within single source of data domain.

For this test, the tasks are assigned to input images with different spatial sizes or distinct channels (i.e. RGB or grayscale) on CIFAR-10 dataset.

In order to apply DMTRL, one has to first convert the heterogeneous inputs into equal-sized features using one hidden layer with totally unshared weights, so that the weights in following layers can be stacked up and factorized.

To better show the influence of heterogeneous inputs on the competitors, we adopt MLP with 4 hidden layers.

The architectures for the heterogenous spatial sizes case and distinct channels case are shown in Table 11 and 12, respectively.

For a good pattern of our TRMTL, such [4, 4, 4, 4, 4, 6] [4, 4, 4, 4, 4, 6] [4, 4, 4, 4, 4, 6] FC2 input modes [8, 8, 6, 4, 4] [8, 8, 6, 4, 4] [8, 8, 6, 4 , 4] output modes [8, 8, 6, 4, 8] [8, 8, 6, 4, 8] [8, 8, 6, 4, 8] FC3 input modes [8, 8, 6, 8, 4] [8, 8, 6, 8, 4] [8, 8, 6, 8, 4 ] output modes [8, 8, 6, 8, 8] [8, 8, 6, 8, 8] [8, 8, 6, 8, 8] FC4 input modes [8, 8, 6, 8, 8] [8, 8, 6, 8, 8] [8, 8, 6, 8, 8] [8, 8, 8, 7 , 7] out modes [4, 4, 4, 5, 5]

<|TLDR|>

@highlight

a distributed latent-space based knowledge-sharing framework for deep multi-task learning