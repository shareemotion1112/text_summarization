Incremental class learning involves sequentially learning classes in bursts of examples from the same class.

This violates the assumptions that underlie  methods for training standard deep neural networks, and will cause them to suffer from catastrophic forgetting.

Arguably, the best method for incremental class learning is iCaRL, but it requires storing  training examples for each class, making it challenging to scale.

Here, we propose FearNet for incremental class learning.

FearNet is a generative model that does not store previous examples, making it memory efficient.

FearNet uses a brain-inspired dual-memory system in which new memories are consolidated from a network for recent memories inspired by the mammalian hippocampal complex to a network for long-term storage inspired by medial prefrontal cortex.

Memory consolidation is inspired by mechanisms that occur during sleep.

FearNet also uses a module inspired by the basolateral amygdala for determining which memory system to use for recall.

FearNet achieves state-of-the-art performance at incremental class learning on image (CIFAR-100, CUB-200) and audio classification (AudioSet) benchmarks.

In incremental classification, an agent must sequentially learn to classify training examples, without necessarily having the ability to re-study previously seen examples.

While deep neural networks (DNNs) have revolutionized machine perception BID26 , off-the-shelf DNNs cannot incrementally learn classes due to catastrophic forgetting.

Catastrophic forgetting is a phenomenon in which a DNN completely fails to learn new data without forgetting much of its previously learned knowledge BID29 .

While methods have been developed to try and mitigate catastrophic forgetting, as shown in BID23 , these methods are not sufficient and perform poorly on larger datasets.

In this paper, we propose FearNet, a brain-inspired system for incrementally learning categories that significantly outperforms previous methods.

The standard way for dealing with catastrophic forgetting in DNNs is to avoid it altogether by mixing new training examples with old ones and completely re-training the model offline.

For large datasets, this may require weeks of time, and it is not a scalable solution.

An ideal incremental learning system would be able to assimilate new information without the need to store the entire training dataset.

A major application for incremental learning includes real-time operation on-board embedded platforms that have limited computing power, storage, and memory, e.g., smart toys, smartphone applications, and robots.

For example, a toy robot may need to learn to recognize objects within its local environment and of interest to its owner.

Using cloud computing to overcome these resource limitations may pose privacy risks and may not be scalable to a large number of embedded devices.

A better solution is on-device incremental learning, which requires the model to use less storage and computational power.

In this paper, we propose an incremental learning framework called FearNet (see Fig. 1 ).

FearNet has three brain-inspired sub-systems: 1) a recent memory system for quick recall, 2) a memory system for long-term storage, and 3) a sub-system that determines which memory system to use for a particular example.

FearNet mitigates catastrophic forgetting by consolidating recent memories into long-term storage using pseudorehearsal BID34 .

Pseudorehearsal allows the network to revisit previous memories during incremental training without the need to store previous training examples, which is more memory efficient.

Figure 1: FearNet consists of three braininspired modules based on 1) mPFC (longterm storage), 2) HC (recent storage), and 3) BLA for determining whether to use mPFC or HC for recall.

Problem Formulation:

Here, incremental class learning consists of T study-sessions.

At time t, the learner receives a batch of data B t , which contains N t labeled training samples, i.e., B t = {(x j , y j )} Nt j=1 , where x j ??? R d is the input feature vector to be classified and y j is its corresponding label.

The number of training samples N t may vary between sessions, and the data inside a study-session is not assumed to be independent and identically distributed (iid).

During a study session, the learner only has access to its current batch, but it may use its own memory to store information from prior study sessions.

We refer to the first session as the model's "base-knowledge," which contains exemplars from M ??? 1 classes.

The batches learned in all subsequent sessions contain only one class, i.e., all y j will be identical within those sessions.

Novel Contributions: Our contributions include:1.

FearNet's architecture includes three neural networks: one inspired by the hippocampal complex (HC) for recent memories, one inspired by the medial prefrontal cortex (mPFC) for long-term storage, and one inspired by the basolateral amygdala (BLA) that determines whether to use HC or mPFC for recall.2.

Motivated by memory replay during sleep, FearNet employs a generative autoencoder for pseudorehearsal, which mitigates catastrophic forgetting by generating previously learned examples that are replayed alongside novel information during consolidation.

This process does not involve storing previous training data.3.

FearNet achieves state-of-the-art results on large image and audio datasets with a relatively small memory footprint, demonstrating how dual-memory models can be scaled.

Catastrophic forgetting in DNNs occurs due to the plasticity-stability dilemma BID0 .

If the network is too plastic, older memories will quickly be overwritten; however, if the network is too stable, it is unable to learn new data.

This problem was recognized almost 30 years ago BID29 BID14 , methods developed in the 1980s and 1990s are extensively discussed, and French argued that mitigating catastrophic forgetting would require having two separate memory centers: one for the long-term storage of older memories and another to quickly process new information as it comes in.

He also theorized that this type of dual-memory system would be capable of consolidating memories from the fast learning memory center to longterm storage.

Catastrophic forgetting often occurs when a system is trained on non-iid data.

One strategy for reducing this phenomenon is to mix old examples with new examples, which simulates iid conditions.

For example, if the system learns ten classes in a study session and then needs to learn 10 new classes in a later study session, one solution could be to mix examples from the first study session into the later study session.

This method is known as rehearsal, and it is one of the earliest methods for reducing catastrophic forgetting BID22 .

Rehearsal essentially uses an external memory to strengthen the model's representations for examples learned previously, so that they are not overwritten when learning data from new classes.

Rehearsal reduces forgetting, but performance is still worse than offline models.

Moreover, rehearsal requires storing all of the training data.

BID34 argued that storing of training examples was inefficient and of "little interest," so he introduced pseudorehearsal.

Rather than replaying past training data, in pseudorehearsal, the algorithm generates new examples for a given class.

In BID34 , this was done by creating random input vectors, having the network assign them a label, and then mixing them into the new training data.

This idea was revived in BID8 , where a generative autoencoder was used to create pseudo-examples for unsupervised incremental learning.

This method inspired FearNet's approach to memory consolidation.

Pseudorehearsal is related to memory replay that occurs in mammalian brains, which involves reactivation of recently encoded memories in HC so that they can be integrated into long-term storage in mPFC BID31 .Recently there has been renewed interest in solving catastrophic forgetting in supervised learning.

Many new methods are designed to mitigate catastrophic forgetting when each study session contains a permuted version of the entire training dataset (see BID20 ).

Unlike incremental class learning, all labels are contained in each study session.

PathNet uses an evolutionary algorithm to find the optimal path through a large DNN, and then freezes the weights along that path BID11 .

It assumes all classes are seen in each study session, and it is not capable of incremental class learning.

Elastic Weight Consolidation (EWC) employs a regularization scheme that redirects plasticity to the weights that are least important to previously learned study sessions BID24 .

After EWC learns a study session, it uses the training data to build a Fisher matrix that determines the importance of each feature to the classification task it just learned.

EWC was shown to work poorly at incremental class learning in BID23 .The Fixed Expansion Layer (FEL) model mitigates catastrophic forgetting by using sparse updates BID6 .

FEL uses two hidden layers, where the second hidden layer (i.e., the FEL layer) has connectivity constraints.

The FEL layer is much larger than the first hidden layer, is sparsely populated with excitatory and inhibitory weights, and is not updated during training.

This limits learning of dense shared representations, which reduces the risk of learning interfering with old memories.

FEL requires a large number of units to work well BID23 .

Figure 2: iCaRL's performance depends heavily on the number of exemplars per class (EPC) that it stores.

Reducing EPC from 20 (blue) to 1 (red) severely impairs its ability to recall older information.

Gepperth & Karaoguz (2016) introduced a new approach for incremental learning, which we call GeppNet.

GeppNet uses a self-organizing map (SOM) to reorganize the input onto a two-dimensional lattice.

This serves as a long-term memory, which is fed into a simple linear layer for classification.

After the SOM is initialized, it can only be updated if the input is sufficiently novel.

This prevents the model from forgetting older data too quickly.

GeppNet also uses rehearsal using all previous training data.

A variant of GeppNet, GeppNet+STM, uses a fixed-size memory buffer to store novel examples.

When this buffer is full, it replaces the oldest example.

During pre-defined intervals, the buffer is used to train the model.

GeppNet+STM is better at retaining base-knowledge since it only trains during its consolidation phase, but the STM-free version learns new data better because it updates the model on every novel labeled input.iCaRL BID33 is an incremental class learning framework.

Rather than directly using a DNN for classification, iCaRL uses it for supervised representation learning.

During a study session, iCaRL updates a DNN using the study session's data and a set of J stored examples from earlier sessions (J = 2, 000 for CIFAR-100 in their paper), which is a kind of rehearsal.

After a study session, the J examples retained are carefully chosen using herding.

After learning the entire dataset, iCaRL has retained J/T exemplars per class (e.g., J/T = 20 for CIFAR-100).

The DNN in iCaRL is then used to compute an embedding for each stored example, and then the mean embedding for each class seen is computed.

To classify a new instance, the DNN is used to compute an embedding for it, and then the class with the nearest mean embedding is assigned.

iCaRL's performance is heavily influenced by the number of examples it stores, as shown in Fig. 2 .

FearNet is heavily inspired by the dual-memory model of mammalian memory BID28 , which has considerable experimental support from neuroscience BID12 BID38 BID25 BID4 BID39 BID15 .

This theory proposes that HC and mPFC operate as complementary memory systems, where HC is responsible for recalling recent memories and mPFC is responsible for recalling remote (mature) memories.

GeppNet is the most recent DNN to be based on this theory, but it was also independently explored in the 1990s in French (1997) and BID3 .

In this section, we review some of the evidence for the dual-memory model.

One of the major reasons why HC is thought to be responsible for recent memories is that if HC is bilaterally destroyed, then anterograde amnesia occurs with old memories for semantic information preserved.

One mechanism HC may use to facilitate creating new memories is adult neurogenesis.

This occurs in HC's dentate gyrus BID2 BID9 .

The new neurons have higher initial plasticity, but it reduces as time progresses BID7 .In contrast, mPFC is responsible for the recall of remote (long-term) memories BID4 .

BID39 and BID15 showed that mPFC plays a strong role in memory consolidation during REM sleep.

BID28 and BID10 theorized that, during sleep, HC reactivates recent memories to prevent forgetting which causes these recent memories to replay in mPFC as well, with dreams possibly being caused by this process.

After memories are transferred from HC to mPFC, evidence suggests that corresponding memory in HC is erased (Poe, 2017).Recently, BID25 performed contextual fear conditioning (CFC) experiments in mice to trace the formation and consolidation of recent memories to long-term storage.

CFC experiments involve shocking mice while subjecting them to various visual stimuli (i.e., colored lights).

They found that BLA, which is responsible for regulating the brain's fear response, would shift where it retrieved the corresponding memory from (HC or mPFC) as that memory was consolidated over time.

FearNet follows the memory consolidation theory proposed by BID25 .

FearNet has two complementary memory centers, 1) a short-term memory system that immediately learns new information for recent recall (HC) and 2) a DNN for the storage of remote memories (mPFC).

FearNet also has a separate BLA network that determines which memory center contains the associated memory required for prediction.

During sleep phases, FearNet uses a generative model to consolidate data from HC to mPFC through pseudorehearsal.

Pseudocode for FearNet is provided in the supplemental material.

Because the focus of our work is not representation learning, we use pre-trained ResNet embeddings to obtain features that are fed to FearNet.

FearNet's HC model is a variant of a probabilistic neural network BID37 .

HC computes class conditional probabilities using stored training examples.

Formally, HC estimates the probability that an input feature vector x belongs to class k as DISPLAYFORM0 (1) DISPLAYFORM1 where > 0 is a regularization parameter and u k,j is the j'th stored exemplar in HC for class k. All exemplars are removed from HC after they are consolidated into mPFC.FearNet's mPFC is implemented using a DNN trained both to reconstruct its input using a symmetric encoder-decoder (autoencoder) and to compute P mP F C (C = k|x).

The autoencoder enables us to The mPFC and BLA sub-systems in FearNet.

mPFC is responsible for the long-term storage of remote memories.

BLA is used during prediction time to determine if the memory should be recalled from short-or long-term memory.use pseudorehearsal, which is described in more detail in Sec. 4.2.

The loss function for mPFC is DISPLAYFORM2 where L class is the supervised classification loss and L recon is the unsupervised reconstruction loss, as illustrated in FIG1 .

For L class , we use standard softmax loss.

L recon is the weighted sum of mean squared error (MSE) reconstruction losses from each layer, which is given by DISPLAYFORM3 where M is the number of mPFC layers, H j is the number of hidden units in layer j, h encoder,(i,j) and h decoder, (i,j) are the outputs of the encoder/decoder at layer j respectively, and ?? j is the reconstruction weight for that layer.

mPFC is similar to a Ladder Network BID32 , which combines classification and reconstruction to improve regularization, especially during lowshot learning.

The ?? j hyperparameters were found empirically, with ?? 0 being largest and decreasing for deeper layers (see supplementary material).

This prioritizes the reconstruction task, which makes the generated pseudo-examples more realistic.

When training is completed during a study session, all of the data in HC is pushed through the encoder to extract a dense feature representation of the original data, and then we compute a mean feature vector ?? c and covariance matrix ?? c for each class c. These are stored and used to generate pseudo-examples during consolidation (see Sec. 4.2).

We study FearNet's performance as a function of how much data is stored in HC in Sec. 6.2.

During FearNet's sleep phase, the original inputs stored in HC are transferred to mPFC using pseudo-examples created by an autoencoder.

This process is known as intrinsic replay, and it was used by Draelos et al. FORMULA1 for unsupervised learning.

Using the class statistics from the encoder, pseudo-examples for class c are generated by sampling a Gaussian with mean ?? c and covariance matrix ?? c to obtainx rand .

Then,x rand is passed through the decoder to generate a pseudo-example.

To create a balanced training set, for each class that mPFC has learned, we generate m pseudo-examples, where m is the average number of examples per class stored in HC.

The pseudo-examples are mixed with the data in HC, and the mixture is used to fine-tune mPFC using backpropagation.

After consolidation, all units in HC are deleted.

During prediction, FearNet uses the BLA network ( FIG1 ) to determine whether to classify an input x using HC or mPFC.

This can be challenging because if HC has only been trained on one class, it will put all of its probability mass on that class, whereas mPFC will likely be less confident.

The output of BLA is given by A (x) and will be a value between 0 and 1, with a 1 indicating mPFC should be used.

BLA is trained after each study session using only the data in HC and with pseudoexamples generated with mPFC, using the same procedure described in Sec. 4.2.

Instead of using solely BLA to determine which network to use, we found that combining its output with those of mPFC and HC improved results.

The predicted class?? is computed a?? DISPLAYFORM0 where DISPLAYFORM1 ?? is the probability of the class according to HC weighted by the confidence that the associated memory is actually stored in HC.

BLA has the same number of layers/units as the mPFC encoder, and uses a logistic output unit.

We discuss alternative BLA models in supplemental material.

Evaluating Incremental Learning Performance.

To evaluate how well the incrementally trained models perform compared to an offline model, we use the three metrics proposed in BID23 .

After each study session t in which a model learned a new class k, we compute the model's test accuracy on the new class (?? new,t ), the accuracy on the base-knowledge (?? base,t ), and the accuracy of all of the test data seen to this point (?? all,t ).

After all T study sessions are complete, a model's ability to retain the base-knowledge is given by DISPLAYFORM0 , where ?? of f line is the accuracy of a multi-layer perceptron (MLP) trained offline (i.e., it is given all of the training data at once).

The model's ability to immediately recall new information is measured by and ??? all are relative to an offline MLP model, so a value of 1 indicates that a model has similar performance to the offline baseline.

This allows results across datasets to be better compared.

Note that ??? base > 1 and ??? all > 1 only if the incremental learning algorithm is more accurate than the offline model, which can occur due to better regularization strategies employed by different models.

Datasets.

We evaluate all of the models on three benchmark datasets TAB1 : CIFAR-100, CUB-200, and AudioSet.

CIFAR-100 is a popular image classification dataset containing 100 mutually-exclusive object categories, and it was used in BID33 to evaluate iCaRL.

All images are 32 ?? 32 pixels.

CUB-200 is a fine-grained image classification dataset containing high resolution images of 200 different bird species BID40 .

We use the 2011 version of the dataset.

AudioSet is an audio classification dataset BID16 .

We use the variant of AudioSet used by BID23 , which contains a 100 class subset such that none of the classes were super-or sub-classes of one another.

Also, since the AudioSet data samples can have more than one class, the chosen samples had only one of the 100 classes chosen in this subset.

DISPLAYFORM1 For CIFAR-100 and CUB-200, we extract ResNet-50 image embeddings as the input to each of the models, where ResNet-50 was pre-trained on ImageNet BID21 .

We use the output after the mean pooling layer and normalize the features to unit length.

For AudioSet, we use the audio CNN embeddings produced by pre-training the model on the YouTube-8M dataset BID1 .

We use the pre-extracted AudioSet feature embeddings, which represent ten second sound clips (i.e., ten 128-dimensional vectors concatenated in order).Comparison Models.

We compare FearNet to FEL, GeppNet, GeppNet+STM, iCaRL, and an onenearest neighbor (1-NN).

FEL, GeppNet, and GeppNet+STM were chosen due to their previously reported efficacy at incremental class learning in BID23 .

iCARL is explicitly designed for incremental class learning, and represents the state-of-the-art on this problem.

We compare against 1-NN due to its similarity to our HC model.

1-NN does not forget any previously observed examples, but it tends to have worse generalization error than parametric methods and requires storing all of the training data.

In each of our experiments, all models take the same feature embedding as input for a given dataset.

This required modifying iCaRL by turning its CNN into a fully connected network.

We performed a hyperparameter search for each model/dataset combination to tune the number of units and layers (see Supplemental Materials).Training Parameters.

FearNet was implemented in Tensorflow.

For mPFC and BLA, each fully connected layer uses an exponential linear unit activation function BID5 .

The output of the encoder also connects to a softmax output layer.

Xavier initialization is used to initialize all weight layers BID19 , and all of the biases are initialized to one.

BLA's architecture is identical to mPFC's encoder, except it has a logistic output unit, instead of a softmax layer.mPFC and BLA were trained using NAdam.

We train mPFC on the base-knowledge set for 1,000 epochs, consolidate HC over to mPFC for 60 epochs, and train BLA for 20 epochs.

Because mPFC's decoder is vital to preserving memories, its learning rate is 1/100 times lower than the encoder.

We performed a hyperparameter search for each dataset and model, varying the model shape (64-1,024 units), depth (2-4 layers), and how often to sleep (see Sec. 6.2).

Across datasets, mPFC and BLA performed best with two hidden layers, but the number of units per layer varied across datasets.

The specific values used for each dataset are given in supplemental material.

In preliminary experiments, we found no benefit to adding weight decay to mPFC, likely because the reconstruction task helps regularize the model.

Unless otherwise noted, each class is only seen in one unique study-session and the first baseknowledge study session contains half the classes in the dataset.

We perform additional experiments to study how changing the number of base-knowledge classes affects performance in Sec. 6.2.

Unless otherwise noted, FearNet sleeps every 10 study sessions across datasets.

TAB2 shows incremental class learning summary results for all six methods.

FearNet achieves the best ??? base and ??? all on all three datasets.

FIG3 shows that FearNet more closely resembles the offline MLP baseline than other methods.

??? new measures test accuracy on the most recently trained class.

1 For FearNet, this measures the performance of HC and BLA.

??? new does not account for how well the class was consolidated into mPFC which happens later during a sleep phase; however, ??? all does account for this.

FEL achieves a high ??? new score because it is able to achieve nearly perfect test accuracy on every new class it learns, but this results in forgetting more quickly than FearNet.

1-NN is similar to our HC model; but on its own, it fails to generalize as well as FearNet, is memory inefficient, and is slow to make predictions.

The final mean-class test accuracy for the offline MLP used to normalize the metrics is 69.9% for CIFAR-100, 59.8% for CUB-200, and 45.8% for AudioSet.

Table 3 : FearNet performance when the location of the associated memory is known using an oracle versus using BLA.

Novelty Detection with BLA.

We evaluated the performance of BLA by comparing it to an oracle version of FearNet, i.e., a version that knew if the relevant memory was stored in either mPFC or HC.

Table 3 shows that FearNet's BLA does a good job at predicting which network to use; however, the decrease in ??? new suggests BLA is sometimes using mPFC when it should be using HC.

FIG5 , it is better able to retain its base-knowledge, but this reduces its ability to recall new information.

In humans, sleep deprivation is known to impair new learning BID41 , and that forgetting occurs during sleep BID30 .

Each time FearNet sleeps, the mPFC weights are perturbed which can cause it to gradually forget older memories.

Sleeping less causes HC's recall performance to deteriorate.

Table 4 : Multi-modal incremental learning experiment.

FearNet was trained with various base-knowledge sets (column-header) and then incrementally trained on all remaining data.

Multi-Modal Incremental Learning.

As shown in Sec. 6.1, FearNet can incrementally learn and retain information from a single dataset, but how does it perform if new inputs differ greatly from previously learned ones?

This scenario is one of the first shown to cause catastrophic forgetting in MLPs.

To study this, we trained FearNet to incrementally learn CIFAR-100 and AudioSet, which after training is a 200-way classification problem.

To do this, AudioSet's features are zero-padded to make them the same length as CIFAR-100s.

Table 4 shows the performance of FearNet for three separate training paradigms: 1) FearNet learns CIFAR-100 as the baseknowledge and then incrementally learns AudioSet; 2) FearNet learns AudioSet as the baseknowledge and then incrementally learns CIFAR-100; and 3) the base-knowledge contains a 50/50 split from both datasets with FearNet incrementally learning the remaining classes.

Our results suggest FearNet is capable of incrementally learning multi-modal information, if the model has a good starting point (high base-knowledge); however, if the model starts with lower base-knowledge performance (e.g., AudioSet), the model struggles to learn new information incrementally (see Supplemental Material for detailed plots).Base-Knowledge Effect on Performance.

In this section, we examine how the size of the baseknowledge (i.e., number of classes) affects FearNet's performance on CUB-200.

To do this, we varied the size of the base-knowledge from 10-150 classes, with the remaining classes learned incrementally.

Detailed plots are provided in the Supplemental Material.

As the base-knowledge size increases, there is a noticeable increase in overall model performance because 1) mPFC has a better learned representation from a larger quantity of data and 2) there are not as many incremental learning steps remaining for the dataset, so the base-knowledge performance is less perturbed.

FearNet's mPFC is trained to both discriminate examples and also generate new examples.

While the main use of mPFC's generative abilities is to enable psuedorehearsal, this ability may also help make the model more robust to catastrophic forgetting.

BID18 observed that unsupervised networks are more robust (but not immune) to catastrophic forgetting because there are no target outputs to be forgotten.

Since the pseudoexample generator is learned as a unsupervised reconstruction task, this could explain why FearNet is slow to forget old information.

Table 5 : Memory requirements to train CIFAR-100 and the amount of memory that would be required if these models were trained up to 1,000 classes.

Table 5 shows the memory requirements for each model in Sec. 6.1 for learning CIFAR-100 and a hypothetical extrapolation for learning 1,000 classes.

This chart accounts for a fixed model capacity and storage of any data or class statistics.

FearNet's memory footprint is comparatively small because it only stores class statistics rather than some or all of the raw training data, which makes it better suited for deployment.

An open question is how to deal with storage and updating of class statistics if classes are seen in more than one study sessions.

One possibility is to use a running update for the class means and covariances, but it may be better to favor the data from the most recent study session due to learning in the autoencoder.

FearNet assumed that the output of the mPFC encoder was normally distributed for each class, which may not be the case.

It would be interesting to consider modeling the classes with a more complex model, e.g., a Gaussian Mixture Model.

BID34 showed that pseudorehearsal worked reasonably well with randomly generated vectors because they were associated with the weights of a given class.

Replaying these vectors strengthened their corresponding weights, which could be what is happening with the pseudo-examples generated by FearNet's decoder.

The largest impact on model size is the stored covariance matrix ?? c for each class.

We tested a variant of FearNet that used a diagonal ?? c instead of a full covariance matrix.

TAB5 shows that performance degrades, but FearNet still works.

FearNet can be adapted to other paradigms, such as unsupervised learning and regression.

For unsupervised learning, FearNet's mPFC already does a form of it implicitly.

For regression, this would require changing mPFC's loss function and may require grouping input feature vectors into similar collections.

FearNet could also be adapted to perform the supervised data permutation experiment performed by BID20 and BID24 .

This would likely require storing statistics from previous permutations and classes.

FearNet would sleep between learning different permutations; however, if the number of classes was high, recent recall may suffer.

In this paper, we proposed a brain-inspired framework capable of incrementally learning data with different modalities and object classes.

FearNet outperforms existing methods for incremental class learning on large image and audio classification benchmarks, demonstrating that FearNet is capable of recalling and consolidating recently learned information while also retaining old information.

In addition, we showed that FearNet is more memory efficient, making it ideal for platforms where size, weight, and power requirements are limited.

Future work will include 1) integrating BLA directly into the model (versus training it independently); 2) replacing HC with a semi-parametric model; 3) learning the feature embedding from raw inputs; and 4) replacing the pseduorehearsal mechanism with a generative model that does not require the storage of class statistics, which would be more memory efficient.

A SUPPLEMENTAL MATERIAL A.1 MODEL HYPERPARAMETERS TAB1 shows the training parameters for the FearNet model for each dataset.

We also experimented with various dropout rates, weight decay, and various activation functions; however, weight decay did not work well with FearNet's mPFC.

TAB1 : FearNet Training Parameters TAB2 shows the training parameters for the iCaRL framework used in this paper.

We adapted the code from the author's GitHub page for our own experiments.

The ResNet-18 convolutional neural network was replaced with a fully-connected neural network.

We experimented with various regularization strategies to increase the initial base-knowledge accuracy with weight decay working the best.

The values that are given as a range of values are the hyperparameter search spaces.

TAB9 shows the training parameters for GeppNet and GeppNet+STM.

Parameters not listed here are the default parameters defined by BID17 .

The values that are given as a range of values are the hyperparameter search spaces.

A.3 BLA VARIANTS Our BLA model is a classifier that determines whether a prediction should be made using HC (recent memory) or mPFC (remote memory).

An alternative approach would be to use an outlier detection algorithm that determines whether the data being processed by a sub-network is an outlier for that sub-network and should therefore be processed by the other sub-network.

To explore this alternative BLA formulation, we experimented with three outlier detection algorithms: 1) one-class support vector machine (SVM) BID36 , 2) determining if the data fits into a Gaussian distribution using a minimum covariance determinant estimation (i.e., elliptical envelope) (Rousseeuw BID35 , and 3) the isolation forest BID27 .

All three of these methods set a rejection criterion for if the test sample exists in HC; whereas the binary MLP reports a probability on how likely the test sample resides in HC.

TAB5 : Performance of different BLA variants.

Pseudocode for FearNet's training and prediction algorithms are given in Algorithms 1 and 2 respectively.

The variables match the ones defined in the paper.

Algorithm 1: FearNet Training Data: X,y Classes/Study-Sessions: T; K: Sleep Frequency; Initialize mPFC with base-knowledge; Store ?? t , ?? t for each class in the base-knowledge; for c ??? T /2 to T do Store X, y for class c in HC; if c % K == 0 then Fine-tune mPFC with X, y in HC and pseudoexamples generated by mPFC decoder; Update ?? t , ?? t for all classes seen so far; Clear HC; else Update BLA;Algorithm 2: FearNet Prediction Data: X A (X) ??? P BLA (C = 1|X); ?? ??? max k P HC (C=k|X)A(X) 1???A(X); if ?? > max k P mP F C (C = k|X) then return arg max k P HC (C = k |X); else return arg max k P mP F C (C = k |X);A.5 MULTI-MODAL LEARNING EXPERIMENT Fig. S1 shows the plots for the multi-modal experiments in Sec. 6.2.

The three base-knowledge experiments were 1) CIFAR-100 is the base-knowledge and AudioSet is trained incrementally, 2) AudioSet is the base-knowledge and then AudioSet is trained incrementally, and 3) the base-knowledge is a 50/50 mix of the two datasets and then the remaining classes are trained incrementally.

For all three base-knowledge experiments, we show the mean-class accuracy on the base-knowledge and the entire test set.

FearNet works well when it adequately learns the base-knowledge (Experiment #1 and #3); however, when FearNet learns it poorly, incremental learning deteriorates.

A.6 BASE-KNOWLEDGE EFFECT ON PERFORMANCE Figure S1 : Detailed plots for the multi-modal experiment.

The top row is when the base-knowledge was CIFAR-100, the middle row is when the base-knowledge was AudioSet, and the bottom row is when the base-knowledge was a 50/50 mix from the two datasets.

The left column represents the mean-class accuracy on the base-knowledge test set and the right column computes mean-class accuracy on the entire test set.remains relatively even because the size of the base-knowledge has no effect on the HC model's ability to immediately recall new information; however, there is a very slight decrease that corresponds to the BLA model erroneously favoring mPFC in a few cases.

Most importantly, ??? all sees an increase in performance because; like ??? base , there are not as many sleep phases to perturb older memories in mPFC.

<|TLDR|>

@highlight

FearNet is a memory efficient neural-network, inspired by memory formation in the mammalian brain, that is capable of incremental class learning without catastrophic forgetting.

@highlight

This paper presents a novel solution to an incremental classification problem based on a dual memory system. 