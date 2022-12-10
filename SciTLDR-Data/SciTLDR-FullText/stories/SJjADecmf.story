The fault diagnosis in a modern communication system is traditionally supposed to be difficult, or even impractical for a purely data-driven machine learning approach, for it is a humanmade system of intensive knowledge.

A few labeled raw packet streams extracted from fault archive can hardly be sufficient to deduce the intricate logic of underlying protocols.

In this paper, we supplement these limited samples with two inexhaustible data sources: the unlabeled records probed from a system in service, and the labeled data simulated in an emulation environment.

To transfer their inherent knowledge to the target domain, we construct a directed information flow graph, whose nodes are neural network components consisting of two generators, three discriminators and one classifier, and whose every forward path represents a pair of adversarial optimization goals, in accord with the semi-supervised and transfer learning demands.

The multi-headed network can be trained in an alternative approach, at each iteration of which we select one target to update the weights along the path upstream, and refresh the residual layer-wisely to all outputs downstream.

The actual results show that it can achieve comparable accuracy on classifying Transmission Control Protocol (TCP) streams without deliberate expert features.

The solution has relieved operation engineers from massive works of understanding and maintaining rules, and provided a quick solution independent of specific protocols.

A telecommunications network is a collection of distributed devices, entirely designed and manufactured by humans for a variety of transmission, control and management tasks, striving to provide a transparent channel between external terminals, via an actual internal relay process node by node.

As a typical conversation in the style of client and server, the two linked nodes send their messages in the form of packets, encapsulated the load with miscellaneous attributes in headers to ensure the correctness, consistency, and smoothness of the entire process.

A typical header includes packet sequence number, source and destination addresses, control bits, error detection codes, etc.

The large-scale network cannot always work ideally, due to its inherent complexity inside massive devices and their interactions.

When there is a malfunction of a device, either caused by the traffic overload, or software bugs, or hardware misconfiguration, or malicious attacks, it will be reflected on the packet streams that pass through, such as packet loss, timeout, out of order, etc.

System administrators captured those suspicious streams and sent back to the service center for cautious offline analysis, which is time-consuming and domain-specific.

The primary challenge of automatic diagnosis is that, it is almost impossible to formalize all the logic inside the system and make them available to artificial intelligence.

A typical modern communication system consists of tens of thousands devices end-to-end and runs based on a list of hundreds of protocols layer-by-layer BID6 ).

If we could figure out the latent states of protocols by constructing specific features from raw bytes, the subsequent classification tasks would be quite straightforward and easy to implement.

For instance, the Transmission Control Protocol (TCP) relies on sequence numbers to judge the receiving order of packets, which may be just big integers roughly linearly growing from the view of machine learning models.

Another example is a few critical control bits may reside among much more useless bits, such as checksum codes, which is harmful noises for models.

Even we have the patience to dive into all the industrial protocols and build up an exhausted feature library; eventually, we will fail again to achieve the target of automation, one of the main advantages of the modern data-driven approach.

Another difficulty is scarce of labeled samples.

In spite of there are seemingly numerous packet flows running through the Internet all the time, the real valid faults occur at random and occupy only a tiny portion of whole traffic volume.

The actual labeled data are usually collected from the archive of fault cases, which is hard to have enough samples for all possible categories, or cannot at least cover them completely.

The previous works on this issue mainly follow two technical routes: 1) a traditional two-phase framework, using expert features and some general-propose classifiers BID1 ); 2) an end-to-end approach based on deep learning for automatic feature extraction (Javaid et al. (2016) ).

All these prior arts seldom use the generative models, which is usually more promising for expressing structural relationship among random variables.

And they may fuse 1-2 data sources in semi-supervised setting (Javaid et al. (2016) ), but not scale to even more data sources.

In this paper, we resort to a generative model to mimic the messages in a terminals conversation and enrich the target data domain from two abundant but different information sources: labeled but from simulation, and genuine but unlabeled.

The transfer and semi-supervised demands are integrated into an intuitive framework, composed of a connected graph of multiple simple Generative Adversarial Networks (GANs)' components, trained in an alternative optimization approach.

The contribution of this paper includes: 1) combine three kinds of data sources in a generative approach, to solve the small-sample problem with a simulation environment; 2) extend the two players in usual GANs to a system of multiple ones, still keeping its merit of end-to-end training; 3) verify its effect on our practice problem of packet sequence classification.

The left of paper is organized as below: first, we introduce the previous work selectively in network anomaly detection and the research frontier in the generative neural network.

Next, we present the model and algorithm in detail with feature design at different levels.

The results of experiments are followed in Section 4.

Finally, we conclude the whole article.

The anomaly detection in communication packets has been long-term studied, either for the Quality of Service (QoS) management or instruction detection.

The article BID1 summarized the past works based on their applied technologies, which almost cover all popular machine learning methods before 2012.1 The works after that mainly switched to deep learning as it goes popular, which are surveyed by ourselves.

In general, all of these keep developing with both two aspects: the level of automation and the ability of models.

BID4 started to train neural networks (NN) on labeled data to build models for more categories of anomalies than hard-coded rules, with an expert feature library and three-layer perceptron classifiers.

Later, BID0 verified the feasibility of Self Organizing Map (SOM) in an unsupervised scenario, where the unexpected packets were automatically saved for analyzers.

BID2 used the online learning to quickly adapt to the newly occurred attacking samples feedbacked by users, without the effort of retraining.

BID11 used the self-taught learning to enrich the dataset from unlabeled live stream, and build up a comprehensive feature space for embedding.

The enhancement can be observed even using a simple K-Nearest-Neighbors.

On the other hand, the neural network models also advance consistently.

The early attempts include PCA NNLiu et al. FORMULA3 , Wavelet NNSun et al. (2009 ), etc.

Yin et al. (2017 used Recurrent Neural Network (RNN) on classifying 5 categories of packet flows, and achieved obviously better results than models ignoring the temporal orders.

BID16 designed an NN with 6-dimensional manual feature vectors and 3 hidden layers for inherent mapping as input and claimed accuracy improvements after testing.

Javaid et al. (2016) also used self-taught learning, similar to BID11 , but with more sophisticated models.

It extracted features automatically from unlabeled data by a sparse auto-encoder, and classify them by a single hidden layer perceptron.

To our best knowledge, it is the first time we employ the generative neural networks for the semisupervised and transfers learning simultaneously on this problem.

The classical GANs compose of two neural components contesting with each other in a zero-sum game BID7 ).

It can be extended to more components for the following purposes (but not limited to): 1) reflecting the relationship between multiple random variables.

BID12 solved the multi-class classification by adding an extra classifier to denote the conditional probability p(y|X).

The newly classifier can shift the burden of predicting labels from identifying fake samples, which is blended in the previous work BID14 .

This triple-player formulation makes the model even clearer and brings more stableness during training.

In multi-view learning, BID3 defined one discriminator for each view and enabled the distribution estimation over all possible output y if any subset of view on a particular input X is fixed.

2) Enhancing the training process.

BID8 addressed the mode collapse problem in GANs by training many generators, which envisions to discriminate itself to an additional classifier and fool the original discriminator meanwhile.

It improved the steadiness of training significantly.

Instead, BID5 used multiple discriminators to ensemble a more powerful one; on the other hand, the training process is retarded by a predefined function (similar to soft-max) on the top of them to match the generator's capability better.

In this paper, we only focus on the former purpose.

The packet stream in reality is a sequence of attributed events e = {(t i , {a i,j })|i ∈ N, j ∈ N }, where t i is the timestamp sealed by the receiver, and a i,j is a tuple of key-value parsed from packet headers.

The label c of e can be K classes of anomalies containing 1 special class for normality.

To focus the main aspect of our problem, we prefer to simplify it by two assumptions: 1) anomalies can only happen at one side of the communication, such as server side, to prevent the number of possible situations from blowing up to K 2 .

It seldom happens that, both terminals have problems simultaneously and produce a complicated interaction that can fully not be diagnosed by a onesided model.

In fact, we can train two individual models for both sides, and thus the records from client side can be removed from the train set in experiments.2) The continuous valued (or finegrained in 10 −6 s) timestamps are ignored, while only their ascending index is kept, from 1 to T .

We insert dummy packets, which replicate sequence id from the previous one and fill all other fields with 0, to denote an occurence of timeout between two consecutive items.

The overall number of dummy packets is informative to models since it indicates how many periods the opposite side has not responded.

It is justified because most protocols are indifferent to the exact time intervals during sending/receiving unless they exceed the predefined timeout threshold.

The available content of attributes depends on how much effort we want to pay for an accurate inspection of packet headers.

The clearer we want to know about the states of every system running, the much we should know about the details of a given protocol.

There are 3 levels of feature engineering: 1) raw bytes of headers, which needs little effort; 2) the numerical values (sequence index, integer or Boolean) parsed by a data scheme indicating their positions and necessity; 3) the latent states defined in protocols, based on complete domain knowledge and Finite-State Machine (FSM) -driven analysis.

For instance, a packet at level 1 may be only a binary sequence, like 1001 . . .

1000, which is unreadable for humans.

At level 2 FIG0 , it turns to be a data structure with proper data types, but without semantics defined by the protocol draft.

The array of structures can further be arranged into a multi-dimensional vector.

At level 3 FIG0 , the inherent states are explicitly parsed out, and the sequence finally achieves its most compact representation -a categorical vector.

NN can digest all levels above, though an extra effort is needed for discrete values at level 3, discussed later in Sec. 3.2.2.

A simplified version of FSM for a sender during TCP's transmission phase.

In the typical situation, the sender shuttles between the Ready and Listen state.

When there is some error occurred in the direction that data goes, the system will retreat to a previous position and resend the datum; when the acknowledge packet loses in the coming direction, a timeout will be triggered, and the system will resend the last piece of data.

In practice, these states are implicitly encoded in the headers, and the computer program has to analyze them based on TCP protocol.

Assume that the N -dimensional sequence Y = {y t : t ∈ T } is generated from repetitive applying of a function G on a latent M -dimensional vector z: DISPLAYFORM0 Where c is a categorical variable to switch between the types of anomalies, and θ is the parameters of G. We assume that, if z ∼ N (0, σ 2 ) and c ∼ Cat(π), the resulted Y will conform to a distribution where our observationsŶ come from.

In the adversary approach, we guide the training of G by minimizing the distance (cross entropy, etc.) between the p(Y, c) and observedp(Y, c), via the transformation of an extra binary discriminator D r : DISPLAYFORM1 where H is the cross entropy between two distributions.

Similarly, we define a function G s that transforms Y to its correspondence Y s in the simulation world, and a function D s to discriminate them from real simulation data: DISPLAYFORM2 .

For the unlabeled data, we define a D c to distinguish them from the generated samples without labels c: DISPLAYFORM3 Figure 2: The layout of building blocks in our solution.

The arrows denote for the information flow among blocks, and they are connected as three forward paths in (a), representing real, simulated and unlabeled information sources colored as orange, green, and blue.

The noises driven the whole process is assumed to be sampled from two independent sources, z ∼ N (0, σ 2 ) and c ∼ Cat(π).

The π are estimated directly based on real labeled data.

In (b), the white block C is trained with a fixed G, colored with shading..

The overall optimization goal is a convex combination of Eq. 3, 5 and 7: DISPLAYFORM4 where λ s , λ c is the coefficients to adjust the importance of targets.

Once we obtain the p(Y, c) in the implicit form of G, the classifier p(c|Y) can be derived with the marginal distribution p(c) according to the Bayes rule.

The neural components and their connections are shown in Fig. 2 .

There are 3 information sinks in Fig. 2a) , each of which connects to exact 2 data sources and corresponds to one part of loss function in Eq. 8; minimizing them concurrently, is equivalent to make the data come from different paths look the same with the best effort.

Note that the graph G is connected, directed and acyclic, which makes any topological order (must exist based on graph theory) of nodes is viable to a gradient descent approach.

It provides a convenience for the design of optimization heuristics in Sec. 3.2.3.

The trained G will be frozen in a secondary training process in Fig. 2b ), to consistently offer the classifier C its pseudo-samples with labels until convergence.

The neural blocks in Fig. 2 can be built from even more fine-grained modules: 1 an input layer concatenating a vector of z (or y) and an optional one-hot vector for c; 2 a Long Short-Term Memory (LSTM) layer mapping a sequence to a vector; 3 a LSTM layer reversely mapping a vector to a sequence; 4 a fully connected layer; 5 a sigmoid (or softmax) layer classifying real/fake samples (or outputting labels).

The generators G is built by 1 + 3 , and G s is 2 + 3 , and C is 2 + 5 ; all D, D s , D u share a same structure, which is 2 + 1 + 5 .

The modules 2 , 3 and 4 are kept as one single hidden layer here, after the multi-layer structures have been tried in practice and their improvement was found to be negligible.

For the discrete state sequence at feature level 3, it is feasible to map its discrete values into continuous vector globally by Word2vec during preprocessing, since the C is what we finally interested for diagnosis and the intermediate results are not visible to end users indeed.

We need a heuristics to guide the whole optimization process of G, depicted in Algo.

1.

During every mini-batch iterations of training, every time we select a forward path whose loss function contributes most in the Eq. 8, weighted by λs, and update the θ of blocks on the way in the form of gradient descent.

The all three sub-loss functions will be updated after G has been modified.

Once the process has found a selected target that does not contribute to the overall goal, the update will be rolled back, and the algorithm will switch to others.

The failure record for any target will not bring into next batch.

For individual blocks, RMSprop is preferred for components containing a recurrent layer, including G, G s and C, while all discriminators use Adam instead.

The real labeled data are collected from the archive of fault cases, probed at the core section of a wireless telecom site in service.

The problematic TCP streams have 4 categories: 1) uplink packet loss at random, 2) downlink packet loss at random, 3) packet loss periodically when load exceeds capacity, 4) checksum error at random, 5) healthy but false alarmed.

The faults in the connection establishment and release phase are not considered here.

In the historical records, all we have are 18 samples, unevenly distributed from category 1-5, which are 5, 7, 3, 1, and 2.The unlabeled data are captured from the same site, having 2000 real samples sufficient for training propose.

Though its size is much larger than the labeled ones, it can hardly contain valid anomalies by an arbitrary, short-term collection.

It provides a reference for the average condition of service quality for a specific site.

The simulation is conducted from a mirror environment in labs, with a similar or maybe simplified configuration.

The 5 types of anomalies are generated in equal ratio, and each of them has 400 records.

The phenomenon of occurring errors here is much more evident than reality: 1) for uplink and 2) downlink packets, the portability of loss is 50%, 3) the stream have 50% of time transmitting over the limit of throughput; 4) the probability of checksum error is 50%.The sequence lengths of all synthetic data are all fixed to 500.The data preprocessing on the TCP header is based on different levels discussed in Sec. 3.1, where the latent states of TCP include normal, resend, and timeout, distilled from 7 useful attributes, and also from 24 bytes of raw binary records.

A simple FSM is defined to compress the attributes to several states, according to TCP's standard logic.

All sequences are split into the size of 500, and the ones shorter than that are padded by 0.

The performance of our multi-class problem is evaluated by the accuracy Acc = N correct /N , which is only metrics cared by operation engineers.

They rank all the possible causes descendingly based on the model's output and take the whole list as a recommendation, which acts as a shortcut leading them to the underlying causes much more quickly, especially compared to their existing manualbased approach.

We measure two variations of accuracy: averaging on samples as above, and averaging on class, i.e., K c=1 N Nc Acc c , to emphasize the performance on minor-classes.

3-fold cross-validation is used for the set of real labeled data, with all other 2 datasets always assisting the train partition for every fold.

The program is based on Keras 2 and a GAN library 3 .

The one-to-many recurrent layer of Keras is implemented merely in the form of many-to-many, by repeating the input as a vector of equal length as output.

The dimension of noise X is set to 20, and hidden dimension of LSTM is 10 with L2 regularization.

The learning rates of all components are configured to 10 −3 .

All other parameters are kept by default.

We have two kinds of factors to combine into a workable model: feature levels and data sources, shown in Tab.

1.

The solution in Fig. 2a) can be trimmed based on available data sources.

In the group 1 of Tab.1, we give two referential models for comparison, one (Line 1) is the dummy model which always give the most frequent class as prediction, and the other (Line 2) is the result if we only use the simulation data both for train and test with one standalone classifier.

With the deliberately amplified anomalies and evenly distributed classes, the performance can be quite ideal, and to some extent be an empirical upper limit of diagnosis.

It can be observed in group 4∼5 that, the simulation data are crucial for substantial enhancement by adding more typical anomalies, while the unlabeled contributes slightly via only supplying normal samples.

The improvements in weighted accuracy are more obvious, which is more than twice that of dummy model.

On the other hand, the features still act an essential role in our problem.

The level 3 feature can always perform better than the rest two levels, while the level 1 can especially not produce any meaningful results.

The level 2 can approximate the best performance with the help of massive data, offering an always worse but still acceptable accuracy, without the effort of understanding protocols.

The evolution of losses of 3 discriminators is demonstrated in FIG2 , and the classifier is shown in FIG2 ).

All loss curves of GANs' components converge to their roughly horizontal limits, with more or less mutually influenced fluctuations, caused by the continuous attempts to seek for a better equilibrium in every batch.

However, these efforts seem to merely make the overall loss tremble, instead of moving to lower places.

The variances of simulated data are obviously much smaller than real data, which may be ascribed to the sufficient size of data and evenly distribution among classes, whereas the real data are of imbalance, and have few samples to validate the improvements during training.

We terminated the whole process at iteration 10 4 , and use the trained G to gain a corresponding C, which is much easier to train as the loss goes steadily to its convergence level, shown in FIG2 ).

In this paper, the widely used semi-supervised and transfer learning requirements have been implemented in an integrated way, via a system of cooperative or adversarial neural blocks.

Its effectiveness has been verified in our application of packet flow classification, and it is hopeful to be a widely adopted method in this specific domain.

The work also prompts us that, complex machine learning tasks and their compound loss functions can be directly mapped into connected networks, and their optimization process can be designed over an entire graph, rather than each individual's hierarchical layers.

In future work, we may study how to apply this approach to even larger scale tasks, and make a theoretical analysis of the existence of equilibrium and why we can always reach it.

@highlight

semi-supervised and transfer learning on packet flow classification, via a system of cooperative or adversarial neural blocks