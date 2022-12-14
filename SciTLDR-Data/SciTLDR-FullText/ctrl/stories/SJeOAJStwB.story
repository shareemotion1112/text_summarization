Federated learning, where a global model is trained by iterative parameter averaging of locally-computed updates, is a promising approach for distributed training of deep networks; it provides high communication-efficiency and privacy-preservability, which allows to fit well into decentralized data environments, e.g., mobile-cloud ecosystems.

However, despite the advantages, the federated learning-based methods still have a challenge in dealing with non-IID training data of local devices (i.e., learners).

In this regard, we study the effects of a variety of hyperparametric conditions under the non-IID environments, to answer important concerns in practical implementations: (i) We first investigate parameter divergence of local updates to explain performance degradation from non-IID data.

The origin of the parameter divergence is also found both empirically and theoretically. (ii) We then revisit the effects of optimizers, network depth/width, and regularization techniques; our observations show that the well-known advantages of the hyperparameter optimization strategies could rather yield diminishing returns with non-IID data. (iii) We finally provide the reasons of the failure cases in a categorized way, mainly based on metrics of the parameter divergence.

Over the recent years, federated learning (McMahan et al., 2017) has been a huge success to reduce the communication overhead in distributed training of deep networks.

Guaranteeing competitive performance, the federated learning permits each learner to compute their local updates of each round for relatively many iterations (e.g., 1 epoch, 10 epochs, etc.), which provides much higher communication-efficiency compared to the conventional data parallelism approaches (for intra-datacenter environments, e.g., Dean et al. (2012) ; Chen et al. (2016) ) that generally require very frequent gradient aggregation.

Furthermore, the federated learning can also significantly reduce data privacy and security risks by enabling to conceal on-device data of each learner from the server or other learners; thus the approach can be applied well to environments with highly private data (e.g., personal medical data), it is now emerging as a promising methodology for privacypreserving distributed learning along with differential privacy-based methods (Hard et al., 2018; Yang et al., 2018; Bonawitz et al., 2019; Chen et al., 2019) .

On this wise, the federated learning takes a simple approach that performs iterative parameter averaging of local updates computed from each learners' own dataset, which suggests an efficient way to learn a shared model without centralizing training data from multiple sources; but hereby, since the local data of each device is created based on their usage pattern, the heterogeneity of training data distributions across the learners might be naturally assumed in real-world cases.

Hence, each local dataset would not follow the population distribution, and handling the decentralized non-IID data still remains a statistical challenge in the field of federated learning (Smith et al., 2017) .

For instance, Zhao et al. (2018) observed severe performance degradation in multi-class classification accuracy under highly skewed non-IID data; it was reported that more diminishing returns could be yielded as the probabilistic distance of learners' local data from the population distribution increases.

LearnerUpdate(k, w): // Run on learner k B ???(split P k into batches of size B) for each local epoch ?? from 1 to E do for each batch b ??? B do w ??? w ??? ????? (w; b) end for end for return w to server Contributions.

To address the non-IID issue under federated learning, there have been a variety of recent works 1 ; nevertheless, in this paper we explore more fundamental factors, the effects of various hyperparameters.

The optimization for the number of local iterations per round or learning rates has been handled in several literatures (e.g., Huang et al. (2018) ; Li et al. (2019c) ; Wang et al. (2019) ); by extension we discuss, for the first time to the best of our knowledge, the effects of optimizers, network depth/width, and regularization techniques.

Our contributions are summarized as follows: First, as a root cause of performance degradation from non-IID data, we investigate parameter divergence of local updates at each round.

The parameter divergence can be regarded as a direct response to learners' local data being non-IID sampled from the population distribution, of which the excessive magnitude could disturb the performance of the consequent parameter averaging.

We also investigate the origin of the parameter divergence in both empirical and theoretical ways.

Second, we observe the effects of well-known hyperparameter optimization methods 2 under the non-IID data environments; interestingly, some of our findings show highly conflicted aspects with their positive outcomes under "vanilla" training 3 or the IID data setting.

Third, we analyze the internal reasons of our observations in a unified way, mainly using the parameter divergence metrics; it is identified that the rationale of the failures under non-IID data lies in some or all of (i) inordinate magnitude of parameter divergence, (ii) its steep fall phenomenon (described in Section 4.2), and (iii) excessively high training loss of local updates.

In this study, Algorithm 1 is considered as a federated learning method, and it is written based on FedAvg (McMahan et al., 2017) .

4 We note that this kind of parameter averaging-based approach has been widely discussed in the literature, under various names, e.g., parallel (restarted) SGD (Zhang et al., 2016; Yu et al., 2019) and local SGD (Lin et al., 2018; Stich, 2019) .

In our experiments with Tensorflow (Abadi et al., 2016) , 5 we consider the multi-class classification tasks on CIFAR-10 (Krizhevsky & Hinton, 2009) and SVHN (Netzer et al., 2011) datasets.

2 we use the term hyperparameter optimization methods and hyperparametric methods interchangeably.

3 This term refers to the non-distributed training with a single machine, using the whole training examples.

4 Regarding the significance of the algorithm, we additionally note that Google is currently employing it on their mobile keyboard application (Gboard) (Hard et al., 2018; Yang et al., 2018; Bonawitz et al., 2019; Chen et al., 2019) .

In this study we deal with image classification, which is also considered as the main applications of federated learning along with the language models (McMahan et al., 2017) .

5 Our source code is available at https://github.com/fl-noniid/fl-noniid Baseline network model: For the baseline deep network, we consider a CNN model that has three 3 ?? 3 convolutional layers with 64, 128, 256 output channels, respectively; and then three fully-connected layers with 512, 256, 10 output sizes, respectively (see Appendix A.1 for more detailed description).

We use the term NetA-Baseline to denote this baseline model throughout this paper.

Regularization configuration: For weight decay, we apply the method of decoupled weight decay regularization (Loshchilov & Hutter, 2019) based on the fact that weight decay is equivalent to L 2 regularization only for pure SGD (Loshchilov & Hutter, 2019; Zhang et al., 2019) .

The baseline value of the weight decay factor is set to 0.00005.

As our regularization baseline, we consider not to apply any other regularization techniques additionally.

We importantly note that if without any particular comments, the results described in the following sections are ones obtained using the above baseline configurations of the network model and regularization.

Environmental configuration: We consider 10 learners to have each 5000 nonoverlapping training examples; Table 1 summarizes our configuration of data settings; Non-IID(N) denotes a data setting that lets each learner to have training examples only for N class(es).

The data settings in the Table 1 deal with data balanced cases where learners have the same amount of local data, and they are mainly considered in the following sections; we additionally note that one can refer to Appendix C.8 for the experiments with data unbalanced cases.

For the IID and the non-IID data settings, T = 200 and 300 are used respectively, while E = 1 and minibatch size of 50 are considered commonly for the both.

6 One can find the remaining configurations for the experiments in Appendix A.

Parameter divergence is recently being regarded as a strong cause of diminishing returns from decentralized non-IID data in federated learning (Zhao et al., 2018) (it is sometimes expressed in another way, gradient/loss divergence (Li et al., 2019b; c; Liu et al., 2019; Wang et al., 2019) ).

For the divergence metrics, many of the literatures usually handle the difference of each learner's local model parameters from one computed with the population distribution; it eventually also causes parameter diversity between the local updates as the data distributions become heterogeneous across learners.

A pleasant level of parameter divergence could rather imply exploiting rich decentralized data (IID cases); however, if the local datasets are far from the population distribution, the consequent parameter averaging of the highly diverged local updates could lead to bad solutions away from the global optimum (non-IID cases).

and Non-IID(2) from the population distribution are 1.0 and 1.6, respectively.

The origin of parameter divergence.

In relation, it has been theoretically proven that the parameter divergence (between the global model parameters under FedAvg and those computed by vanilla SGD training) is directly related to the probabilistic distance of local datasets from the population distribution (see Proposition 3.1 in Zhao et al. (2018) ).

In addition to it, for multi-class classification tasks, we here identify in lower level, that if data distributions in each local dataset are highly skewed and heterogeneous over classes, subsets of neurons, which have especially big magnitudes of the gradients in back propagation, become significantly different across learners; this leads to inordinate parameter divergence between them.

As illustrated in Figure 1 , under the IID data setting, the weight values in the output layer are evenly distributed relatively evenly across classes if the neurons of the model are initialized uniformly.

However, we can observe under the non-IID data settings that the magnitudes of the gradients are distributed depending on each learner's data distribution.

We also provide the corresponding theoretical analysis in Appendix B.

Metrics.

To capture parameter divergence under federated learning, we define the following two metrics using the notations in Algorithm 1.

Since in our analysis we compare different network architectures or training settings together in a set, the number of neurons in the probed layers can become different, and values of model parameters can highly depend on the experimental manipulations; thus instead of Euclidean distance, in the two divergence metrics we use cosine distance that enables normalized (qualitative) measures.

We also note that PD-VL is defined assuming the balancedness of data amount between learners, i.e., the same numbers of local iterations per round.

The reason of probing parameter divergence being important is that the federated learning are performed based on iterative parameter averaging.

That is, investigating how local updates are diverged can give a clue whether the subsequent parameter averaging yields positive returns; the proposed divergence metrics provide two ways for it.

k is a subset (or the universal set) of w t k , we define parameter divergence between local updates as

In addition, assume that P k is identical ???k ??? K, and let w t ???1 be the vanilla-updated parameters, that is, the model parameters updated on the global parameters (i.e., w t???1 ) using IID training data during the same number of iterations with the actual learners (i.e., P k /B).

Then, for z

Relationship among probabilistic distance, parameter divergence, and learning performance.

We consider Non-IID(5) and Non-IID(2) for non-IID data settings.

Here we use earth mover's distance (EMD), also known as Wasserstein distance, to measure probabilistic distance of each data settings from the population distribution; the value becomes 1.0 and 1.6 for Non-IID(5) and Non-IID(2), respectively.

From the middle and right panels of Figure 2 , it is seen that greater EMDs lead to bigger parameter divergence (refer to also Figure 9 in the appendix).

Also, together with the left panel, we can observe the positive correlation between parameter divergence and learning performance.

Therefore, we believe the parameter divergence metrics can help to reveal the missing link between data non-IIDness and the consequent learning performance.

Note that one can also refer to the similar analysis with more various EMD in Zhao et al. (2018) .

From now on we describe our findings for the effects of various hyperparameter optimization methods with non-IID data on the federated learning algorithm.

The considered hyperparametric methods have been a huge success to improve performance in deep learning; however, here we newly identify that under non-IID data settings, they could give negative/diminishing effects on performance of the federated learning algorithm.

The following is the summary of our findings; we provide the complete experimental results and further discussion in the next subsection and the appendix.

Effects of optimizers.

Unlike non-adaptive optimizers such as pure SGD and momentum SGD (Polyak, 1964; Nesterov, 1983) , Adam (Kingma & Ba, 2015) could give poor performance from non-IID data if the parameter averaging is performed only for weights and biases, compared to all the model variables (including the first and second moment) being averaged.

Here we importantly note that both momentum SGD and Adam require the additional variables related to momentum as well as weights and biases; throughout the rest of the paper, the terms (optimizer name)-A and (optimizer name)-WB are used to refer to the parameter averaging being performed for all the variables 7 and only for weights & biases, respectively.

Effects of network depth/width.

It is also known that deepening "plain" networks (which simply stacks layers, without techniques such as information highways (Srivastava et al., 2015) and shortcut connection (He et al., 2016)) yields performance degradation at a certain depth, even under vanilla training; however this phenomenon gets much worse under non-IID data environments.

On the contrary, widening networks could help to achieve better outcomes; in that sense, the global average pooling (Lin et al., 2014) could fail in this case since it significantly reduces the channel dimension of the (last) fully-connected layer, compared to using the max pooling.

Effects of Batch Normalization.

The well-known strength of Batch Normalization (Ioffe & Szegedy, 2015) , the dependence of hidden activations in the minibatch (Ioffe, 2017), could become a severe drawback in non-IID data environments.

Batch Renormalization (Ioffe, 2017) helps to mitigate this, but it also does not resolve the problem completely.

Effects of regularization techniques.

With non-IID data, regularizations techniques such as weight decay and data augmentation could give excessively high training loss of local updates even in a modest level, which offsets the generalization gain.

We now explain the internal reasons of the observations in the previous subsection.

Through the experimental results, we were able to classify the causes of the failures under non-IID data into three categories; the following discussions are described based on this.

8 Note that our discussion in this subsection is mostly made from the results under Nesterov momentum SGD and on CIFAR-10; the complete results including other optimizers (e.g., pure SGD, Polyak momentum SGD, and Adam) and datasets (e.g., SVHN) are given in Appendix C.

Inordinate magnitude of parameter divergence.

As mentioned before, bigger parameter divergence is the root cause of diminishing returns under federated learning methods with non-IID data.

By extension, here we observe that even under the same non-IID data setting, some of the considered hyperparametric methods yield greater parameter divergence than when they are not applied.

For example, from the left plot of Figure 3 , we see that under the Non-IID(2) setting, the parameter divergence values (in the last fully-connected layer) become greater as the network depth increases (note that NetA-Baseline, NetA-Deeper, and NetA-Deepest have 3, 6, and 9 convolutional layers, respectively; see also Appendix A.1 for their detailed architecture).

The corresponding final test accuracy was found to be 74.11%, 73.67%, and 68.98%, respectively, in order of the degree of shallowness; this fits well into the parameter divergence results.

Since the NetA-Deeper and NetA-Deepest have twice and three times as many model parameters as NetA-Baseline, it can be expected enough that the deeper models yield bigger parameter divergence in the whole model; but our results also show its qualitative increase in a layer level.

In relation, we also provide the results using the modern network architecture (e.g., ResNet (He et al., 2016) ) in Table 8 of the appendix.

From the middle plot of the figure, we can also observe bigger parameter divergence in a high level of weight decay under the Non-IID(2) setting.

Under the non-IID data setting, the test accuracy of about 72 ??? 74% was achieved in the low levels (??? 0.0001), but weight decay factor of 0.0005 yielded only that of 54.11%.

Hence, this suggests that with non-IID data we should apply much smaller weight decay to federated learning-based methods.

Here we note that if a single iteration is considered for each learner's local update per round, the corresponding parameter divergence will be of course the same without regard to degree of weight decay.

However, in our experiments, the great number of local iterations per round (i.e., 100) made a big difference of the divergence values under the non-IID data setting; this eventually yielded the accuracy gap.

We additionally observe for the non-IID cases that even with weight decay factor of 0.0005, the parameter divergence values are similar to those with the smaller factors at very early rounds in which the norms of the weights are relatively very small.

In addition, it is observed from the right plot of the figure that Dropout (Hinton et al., 2012; Srivatava et al., 2014 ) also yields bigger parameter divergence under the non-IID data setting.

The corresponding test accuracy was seen to be a diminishing return with Nesterov momentum SGD (i.e., using Dropout we can achieve +2.85% under IID, but only +1.69% is obtained under non-IID(2), compared to when it is not applied; see Table 2 ); however, it was observed that the generalization effect of the Dropout is still valid in test accuracy for the pure SGD and the Adam (refer to also Table 13 in the appendix).

Steep fall phenomenon.

As we see previously, inordinate magnitude of parameter divergence is one of the notable characteristics for failure cases under federated learning with non-IID data.

However, under the non-IID data setting, some of the failure cases have been observed where the test accuracy is still low but the parameter divergence values of the last fully-connected layer decrease (rapidly) over rounds; as the round goes, even the values were sometimes seen to be lower than those of the comparison targets.

We refer to this phenomenon as steep fall phenomenon.

It is inferred that these (unexpected abnormal) sudden drops of parameter divergence values indicate going into poor local minima (or saddles); this can be supported by the behaviors that test accuracy increases plausibly at very early rounds, but the growth rate quickly stagnates and eventually becomes much lower than the comparison targets.

The left plot of Figure 4 shows the effect of the Adam optimizer with respect to its implementations.

Through the experiments, we identified that under non-IID data environments, the performance of Adam is very sensitive to the range of model variables to be averaged, unlike the non-adaptive optimizers (e.g., momentum SGD); its moment variables should be also considered in the parameter averaging together with weights and biases (see also Table 3 ).

The poor performance of the Adam-WB under the Non-IID(2) setting would be from twice as many momentum variables as the momentum SGD, which indicates the increased number of them affected by the non-IIDness; thus, originally we had thought that extreme parameter divergence could appear if the momentum variables are not averaged together with weights and biases.

However, it was seen that the parameter divergence values under the Adam-WB was seen to be similar or even smaller than under Adam-A (see also Figure 11 in the appendix).

Instead, from the left panel we can observe that the parameter divergence of Adam-WB in the last fully-connected layer is bigger than that of Adam-A at the very early rounds (as we expected), but soon it is abnormally sharply reduced over rounds; this is considered the steep fall phenomenon.

The middle and the right plots of the figure also show the steep fall phenomenon in the last fullyconnected layer, with respect to network width and whether to use Batch Normalization, respectively.

In the case of the NetC models, NetC-Baseline, NetC-Wider, and NetC-Widest use the global average pooling, the max pooling with stride 4, and the max pooling with stride 2, respectively, after the last convolutional layer; the number of neurons in the output layer becomes 2560, 10240, and 40960, respectively (see also Appendix A.1 for their detailed architecture).

Under the Non-IID(2) setting, the corresponding test accuracy was found to be 64.06%, 72.61%, and 73.64%, respectively, in order of the degree of wideness.

In addition, we can see that under Non-IID(2), Batch Normalization 9 yields not only big parameter divergence (especially before the first learning rate drop) but also the steep fall phenomenon; the corresponding test accuracy was seen to be very low (see Table 3 ).

The failure of the Batch Normalization stems from that the dependence of batchnormalized hidden activations makes each learner's update too overfitted to the distribution of their local training data.

Batch Renormalization, by relaxing the dependence, yields a better outcome; however, it still fails to exceed the performance of the baseline due to the significant parameter divergence.

To explain the impact of the steep fall phenomenon in test accuracy, we provide Figure 5 , which indicates that the loss landscapes for the failure cases (e.g., Adam-WB and with Batch Normalization) commonly show sharper minima that leads to poorer generalization (Hochreiter & Schmidhuber, 9 For its implementations into the considered federated learning algorithm, we let the server get the proper moving variance by 1997; Keskar et al., 2017) , and the minimal value in the bowl is relatively greater.

10 Here it is also observed that going into sharp minima starts even in early rounds such as 25th.

Excessively high training loss of local updates.

The final cause that we consider for the failure cases is excessively high training loss of local updates.

For instance, from the left plot of Figure 6 , we see that under the Non-IID(2) setting, NetB-Baseline gives much higher training loss than the other models.

Here we note that for the NetB-Baseline model, the global average pooling is applied after the last convolutional layer, and the number of neurons in the first fully-connected layer thus becomes 256 ?? 256; on the other hand, NetB-Wider and NetB-Widest use the max pooling with stride 4 and 2, which make the number of neurons in that layer become 1024 ?? 256 and 4096 ?? 256, respectively (see also Appendix A.1 for their details).

The experimental results were shown that NetB-Baseline has notably lower test accuracy (see Table 4 ).

We additionally remark that for NetBBaseline, very high losses are observed under the IID setting, and their values even are greater than in the non-IID case; however, note that one have to be aware that local updates are extremely easy to be overfitted to each training dataset under non-IID data environments, thus the converged training losses being high is more critical than the IID cases.

The middle and the right plot of the figure show the excessive training loss under the non-IID setting when applying the weight decay factor of 0.0005 and the data augmentation, respectively.

In the cases of the high level of weight decay, the severe performance degradation appears compared to when the levels are low (i.e., ??? 0.0001) as already discussed.

In addition, we observed that with Nesterov momentum SGD, the data augmentation yields a diminishing return in test accuracy (i.e., with the data augmentation we can achieve +3.36% under IID, but ???0.16% is obtained under non-IID(2), compared to when it is not applied); with Adam the degree of the diminishment becomes higher (refer to Table 12 in the appendix).

In the data augmentation cases, judging from that the 10 Based on Li et al. (2018) , the visualization of loss surface was conducted by L(??, ??) = (?? * + ???? + ????), where ?? * is a center point of the model parameters, and ?? and ?? is the orthogonal direction vectors.

parameter divergence values are not so different between with and without it, we can identify that the performance degradation stems from the high training loss (see Figures 30 and 31 in the appendix).

Here we additionally note that unlike on the CIFAR-10, in the experiments on SVHN it was seen that the generalization effect of the data augmentation is still valid in test accuracy (see Table 12 ).

In this paper, we explored the effects of various hyperparameter optimization strategies for optimizers, network depth/width, and regularization on federated learning of deep networks.

Our primary concern in this study was lied on non-IID data, in which we found that under non-IID data settings many of the probed factors show somewhat different behaviors compared to under the IID setting and vanilla training.

To explain this, a concept of the parameter divergence was utilized, and its origin was identified both empirically and theoretically.

We also provided the internal reasons of our observations with a number of the experimental cases.

In the meantime, the federated learning has been vigorously studied for decentralized data environments due to its inherent strength, i.e., high communication-efficiency and privacy-preservability.

However, so far most of the existing works mainly dealt with only IID data, and the research to address non-IID data has just entered the beginning stage very recently despite its high real-world possibility.

Our study, as one of the openings, handles the essential factors in the federated training under the non-IID data environments, and we expect that it will provide refreshing perspectives for upcoming works.

A EXPERIMENTAL DETAILS

In the experiments, we consider CNN architectures, as illustrated in Figure 7 .

In the network configurations, three groups of 3 ?? 3 convolutional layers are included that have 16 ?? m, 128, and 256 output channels, respectively; n denotes the number of the layers in each convolutional group.

The first two groups are followed by 3 ?? 3 max pooling with stride 2; the last convolutional layer is followed by either the 3 ?? 3 max pooling with stride s or the global average pooling.

In the case of fully-connected layers, we use two types of the stacks: (i) three layers, of which the output sizes are 256 ?? u, 256, and 10, respectively; and (ii) a single layer, of which the output size is 10.

In addition, we use the ReLU and the softmax activation for the hidden weight layers and the output layer, respectively.

Table 6 summarizes the network models used in the experiments.

In the experiments, we initialize the network models to mostly follow the truncated normal distribution with a mean of 0 based on He et al. (2015), however we fix the standard deviation to 0.05 for the first convolutional group and the last fully-connected layer.

For training, minibatch stochastic optimization with cross-entropy loss is considered.

Specifically, we use pure SGD, Nesterov momentum SGD (Polyak, 1964; Nesterov, 1983) , and Adam (Kingma & Ba, 2015) as optimization methods; initial learning rates are set to 0.05, 0.01, and 0.001, respectively for each optimizer.

We drop the learning rate by 0.1 at 50% and 75% of the total training iterations, respectively.

Regarding the environmental configurations, we predetermine each of learners' local training dataset in a random seed; the training examples are allocated so that they do not overlap between the learners.

To report the experimental results, we basically considered to run the trials once, but as for unstable ones in the preliminary tests, we chose the middle results of several runs.

In every result plot, the values are plotted at each round.

In relation to the federated learning under non-IID data, so far there have been several works for providing theoretical bounds to explain how does the degree of the non-IIDness of decentralized data affect the performance, with respect to its degree (e.g., Zhao et al. (2018) (2019)).

Inspired by them, here we further study how does the non-IIDness make the model parameters of each learner diverged.

In this analysis, we consider training deep networks for multi-class classification.

Based on the notations in Algorithm 1, the SGD update of learner k at round t + 1 is given as

where f q (x; w) is the posterior probability for class q ??? Q (Q is the label space), obtained from model parameters w with data examples (x, y), and p k (y = q) is the probability that the label of a data example in P k is q. In this equation, w

is the model parameters after the ?? -th local iterations in the round t + 1 (R is the number of local iterations of each learner per round).

Herein we note that w t+0 k ( w t ) is the global model parameters received from the server at the round t + 1; we use the term to distinguish it from the term w t k (which indicates the local update that has sent back to the server at round t).

Then, by the linearity of the gradient, we obtain

where d q denote the neurons, in the (dense) output layer of the model w, that are connected to the output node for class q.

with the fixed k. At round t+1, suppose that for learner

where (a q )

.

Then, we can get

From this, we can identify that the parameter difference,

with the fixed q. At round t+1, suppose that for class q,

Then, similar with Equation 1, we can have

C THE COMPLETE EXPERIMENTAL RESULTS

In this section we provide our complete experimental results.

Before the main statement, we first note that in the following figures, C ij denotes the j-th convolutional layer in the i-th group, and F j denotes the j-th fully-connected layer (in relation, refer to Appendix A.1).

In addition, we remind that in this paper "vanilla" training refers to non-distributed training with a single machine, using the whole data examples; for the vanilla training, we trained the networks for 100 epochs.

Here we investigate the effect of optimizers.

We importantly note that both momentum SGD and Adam require the additional variables related to momentum as well as weights and biases; the terms (optimizer name)-A and (optimizer name)-WB are used to refer to the parameter averaging being performed for all the variables and only for weights & biases, respectively.

The experimental results are provided in Table 7 and Figures 10 and 11 .

From the table, interestingly we can notice that under the non-IID data setting, there exists a huge performance gap between Adam-A and Adam-WB (??? 7%), unlike the momentum SGD trials.

At the initial steps of this study, we had thought that the poor performance of Adam-WB would be from the following: Since Adam requires twice as many momentum variables as momentum SGD, extreme parameter divergence could appear if they are not averaged together with weights and biases.

However, unlike our expectations, the parameter divergence values under the Adam-WB was seen to be similar or even smaller than under Adam-A. Nevertheless, we can observe the followings for the non-IID cases: First, the parameter divergence of Adam-WB in F 3 is bigger than that of Adam-A at the very early rounds (as we expected), but soon it is abnormally sharply reduced over rounds; this can be considered the steep fall phenomenon.

Second, Adam-WB leads to higher training loss of each learner.

We guess that these two caused the severe degradation of Adam-WB in test accuracy.

Here we investigate the effect of network depth.

Since deepening networks also indicates that there becomes having more parameters to be averaged in the considered federated learning algorithm, we had predicted especially under non-IID data settings that depending on their depth, it would yield bigger parameter divergence in the whole model and the consequent diminishing returns compared to under the vanilla training and the IID data setting; the test accuracy results show it as expected (see Table 8 ).

11 Moreover, it is also seen from Figure 12 that parameter divergence increases also qualitatively (i.e., in a layer level) under the non-IID data setting, as the number of convolutional layers increases.

Note that for C 21 and C 31 , the divergence pattern is resulted as opposed to that of C 11 and F 3 ; however, the values of C 11 and F 3 would be more impactful as mentioned in Footnote 8.

We additionally remark from the figure that the sharp reduction of parameter divergence (in the convolutional layers) at the very early rounds when using NetA-Deepest indicates the parameter averaging algorithm did not work properly.

Correspondingly, the test accuracy values in the early period were seen to be not much different from the initial one.

Following the previous subsection, from now on we investigate the effect of network width.

Contrary to the results in the Section C.2, it is seen from Table 9 that widening networks provides positive effects for the considered federated learning algorithm under the non-IID data setting.

Especially, one can see that compared to the max pooling trials, while the global average pooling yields higher test accuracy in the vanilla training (with the minibatch size of 50), its performance gets significantly worse under the non-IID data setting (remind that NetB-Baseline and NetC-Baseline use the global average pooling after the last convolutional layer).

Focusing on the NetC models, we here make the following observations for the non-IID data setting from Figures 15, 18, and 21: First, the considered federated learning algorithm provides bigger parameter divergence in F 1 as its width decreases (note that each input size of F 1 is 256, 1024, and 4096 for NetC-Baseline, NetC-Wider, and NetC-Widest, respectively), especially during the beginning rounds (e.g., for the NMom-A case, until about 50 rounds).

Unlike in Section C.2, here we can identify that even though the parameter size of is the smallest under the global averaging pooling, it rather yields the biggest qualitative parameter divergence.

Second, the steep fall phenomenon appears in F 1 for the NetC- Baseline case.

Third, the global average pooling gives too high training loss of each learner.

All the three observations fit well into the failure of the global average pooling.

We additionally note that when using NetC-Baseline, the results under the IID data setting shows very high loss values; this leads to diminishing returns for the pure SGD and the NMom-A cases, compared to the vanilla training results with the minibatch size of 50.

However, the corresponding degradation rate is seen to be much higher under the non-IID data setting.

This is because the local updates are extremely easy to be overfitted to the training data under the non-IID data setting; thus the converged training losses being high becomes much more critical.

Here we investigate the effect of weight decay.

From Table 10 , it is seen that under the non-IID data setting we should apply much smaller weight decay for the considered federated learning algorithm than under the vanilla training or the IID data setting.

For its internal reason, Figures 22, 23 , and 24 show that under the non-IID data setting, the considered federated learning algorithm not only converges to too high training loss (of each learner) but also causes excessive parameter divergence when the weight decay factor is set to 0.0005.

Here we note that if a single iteration is considered for each learner's local update per round, the corresponding parameter divergence will be of course the same without regard to degree of weight decay.

However, in our experiments, the great number of local iterations per round (i.e., 100) made a big difference of the divergence values under the non-IID data setting; this eventually yielded the accuracy gap.

In addition, we further observe under the non-IID data setting that even with the weight decay factor of 0.0005, the test accuracy increases similarly with its smaller values at very early rounds, in which the norm values of the weights are relatively much smaller.

Moreover we also conducted additional experiments for the related regularization techniques, FedProx (Li et al., 2019b) .

Under the FedProx, in order to make local updates do not deviate excessively from the current global model parameters, at each round t each learner uses the following surrogate loss function that adds a proximal term to the original objective function: (w) + ?? 2 w ??? w t???1 2 .

Figure 8 shows the experimental results; as seen from the figure, in our implementation FedProx did not provide dramatic improvement in final accuracy, but we can observe that it could yield not only lower parameter divergence but also faster convergence speed (especially before the first learning rate drop).

One can find the corresponding complete results in Figure 25 .

Here we investigate the effect of Batch Normalization.

For its implementations into the considered federated learning algorithm, we let the server get the proper moving variance by

??? E ?? 2 at each round, by allowing each learner k collect E ?? 2 k as well (2018)).

It is natural to take this strategy especially under the non-IID data setting; otherwise, a huge problem would arise due to bad approximation of the moving statistics.

Also, it is additionally remarked that for Batch Renormalization we simply used ?? = 0.01, r max = 2, and d max = 2 in the experiments (see Ioffe (2017) for the description of the three hyperparameters).

Table 11 that under the non-IID data setting, the performance significantly gets worse if Batch Normalization is employed to the baseline; this would be rooted in that the dependence of batch-normalized hidden activations makes each learner's update too overfitted to the distribution of their local training data.

The consequent bigger parameter divergence is observed in Figures 26,  27 , and 28.

On the contrary, Batch Renormalization, by relaxing the dependence, yields a better outcome; although its parameter divergence is seen greater in some layers than under Batch Normalization, it does not lead to the steep fall phenomenon while the Batch Normalization does in F 3 .

Nevertheless, the Batch Renormalization was still not able to exceed the performance of the baseline due to the significant parameter divergence.

In the implementation of data augmentation, we used random horizontal flipping, brightness & contrast adjustment, and 24??24 cropping & resizing in the pipeline.

From Table 12 , we identify that under the non-IID data setting, the data augmentation yields diminishing returns for the PMom-A, NMom-A, and Adam-A cases on CIFAR-10, compared to under the IID data setting; under Adam-A, especially it gives even a worse outcome.

However, it is seen that the corresponding parameter divergence is almost similar between with and without the data augmentation (refer to Figures 30  and 31 ).

Instead, we are able to notice that the diminishing outcomes from the data augmentation had been eventually rooted in local updates' high training losses.

Here we note that in the pure SGD case, very high training loss values are found as well under the IID data setting when the data augmentation was applied (see Figure 29) ; this leads to lower test accuracy compared to the baseline, 83.09 (83.20) 82.01 (82.11) 76.63 (76.68) similar to under the non-IID cases.

Also, it is additionally noted that unlike on the CIFAR-10, in the experiments on SVHN it was observed that the generalization effect of the data augmentation is still valid in test accuracy.

In the experiments, we employed Dropout with the rates 0.2 and 0.5 for convolutional layers and fully-connected layers, respectively.

The results show that under the non-IID data setting, the Dropout provides greater parameter divergence compared to the baselines, especially in F 3 (see Figures 32, 33, and 34) ; this leads to diminishing returns for the PMom-A and NMom-A cases on CIFAR-10, compared to under the IID data setting.

However, we can observe from Table 13 that the effect of the Dropout is still maintained positive for the rest of the cases.

As remarked in (McMahan et al., 2017) , since the federated learning do not require centralizing local data, data unbalancedness (i.e., each learner has various numbers of local data examples) would be also naturally assumed in the federated learning along with non-IIDness.

In relation, we also conducted the experiments under the unbalanced cases.

Table 14 summarizes the considered unbalanced data settings; they were constructed similarly to (Li et al., 2019b) so that the number of data examples per learner follows a power law.

The experimental results under the unbalanced settings are summarized in Table 15 .

From the table, it is observed that our findings in Section 4.1 are still valid under the unbalanced data settings.

In addition, we can also see that for the unbalanced cases, the performance under Non-IID(2) setting is worse mostly than that of balanced cases while they show similar values under the IID data setting; this indicates that the negative impact of data unbalancedness is not as great as that of the nonIIDness, but it becomes much bigger when the two are combined.

<|TLDR|>

@highlight

We investigate the internal reasons of our observations, the diminishing effects of the well-known hyperparameter optimization methods on federated learning from decentralized non-IID data.