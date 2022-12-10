Recurrent neural network(RNN) is an effective neural network in solving very complex supervised and unsupervised tasks.

There has been a significant improvement in RNN field such as natural language processing, speech processing, computer vision and other multiple domains.

This paper deals with RNN application on different use cases like Incident Detection , Fraud Detection , and Android Malware Classification.

The best performing neural network architecture is chosen by conducting different chain of experiments for different network parameters and structures.

The network is run up to 1000 epochs with learning rate set in the range of 0.01 to 0.5.Obviously, RNN performed very well when compared to classical machine learning algorithms.

This is mainly possible because RNNs implicitly extracts the underlying features and also identifies the characteristics of the data.

This lead to better accuracy.

In today's data world, malware is the common threat to everyone from big organizations to common people and we need to safeguard our systems, computer networks, and valuable data.

Cyber-crimes has risen to the peak and many hacks, data stealing, and many more cyber-attacks.

Hackers gain access through any loopholes and steal all valuable data, passwords and other useful information.

Mainly in android platform malicious attacks increased due to increase in large number of application.

In other hand its very easy for persons to develop multiple malicious malwares and feed it into android market very easily using a third party software's.

Attacks can be through any means like e-mails, exe files, software, etc.

Criminals make use of security vulnerabilities and exploit their opponents.

This forces the importance of an effective system to handle the fraudulent activities.

But today's sophisticated attacking algorithms avoid being detected by the security Email address: harishharunn@gmail.com (Mohammed Harun Babu R) mechanisms.

Every day the attackers develop new exploitation techniques and escape from Anti-virus and Malware softwares.

Thus nowadays security solution companies are moving towards deep learning and machine learning techniques where the algorithm learns the underlying information from the large collection of security data itself and makes predictions on new data.

This, in turn, motivates the hackers to develop new methods to escape from the detection mechanisms.

Malware attack remains one of the major security threat in cyberspace.

It is an unwanted program which makes the system behave differently than it is supposed to behave.

The solutions provided by antivirus software against this malware can only be used as a primary weapon of resistance because they fail to detect the new and upcoming malware created using polymorphic, metamorphic, domain flux and IP flux.

The machine learning algorithms were employed which solves complex security threats in more than three decades BID0 .

These methods have the capability to detect new malwares.

Research is going at a high phase for security problems like Intrusion Detection Systems(IDS), Mal-ware Detection, Information Leakage, etc.

Fortunately, today's Deep Learning(DL) approaches have performed well in various long-standing AI challenges BID1 such as nlp, computer vision, speech recognition.

Recently, the application of deep learning techniques have been applied for various use cases of cyber security BID2 .It has the ability to detect the cyber attacks by learning the complex underlying structure, hidden sequential relationships and hierarchical feature representations from a huge set of security data.

In this paper, we are evaluating the efficiency of SVM and RNN machine learning algorithms for cybersecurity problems.

Cybersecurity provides a set of actions to safeguard computer networks, systems, and data.

This paper is arranged accordingly where related work are discussed in section 2 the background knowledge of recurrent neural network (RNN) in section 3 .In section 4 proposed methodology including description,data set are discussed and at last results are furnished in Section 5.

Section 6 is conclude with conclusion.

In this section related work for cybersecurity use cases is discussed : Android Malware Classification (T1), Incident Detection (T2) , and Fraud Detection (T3).

The most commonly used approach for Malware detection in Android devices is the static and dynamic approach BID3 .

In the static approach, all the android permissions are collected by unpacking the application and whereas, in dynamic approach, the run-time execution attributes like system calls, network connections, electricity, user interactions and efficient utilization of memory.

Most of the commercial systems used today use both the static and dynamic approach.

For low computational cost, resource utilization, time resource Static analysis is mainly preferred for Android devices.

Meanwhile dynamic analysis has the advantage to detect metamorphic and polymorphic malware.

BID4 have evaluated the performance of traditional ML algorithms for malware detection on Android devices without using the API calls and permission as features.

MalDozer proposed the use of API calls with deep learning approach to detect the Android malware and classify them accordingly BID5 .

BID6 API calls contains schematic information which helps in understand the intention of the app indirectly without any user interface.

Using embedding techniques at training phase API calls are extracted using DEX assembly BID5 which helps in effective malware detection on neural networks.

The security issues in cloud computing are briefly discussed in BID7 .

BID8 proposed ML-based anomaly detection that acts on the network, service and work-flow layers.

A hybrid of both machine learning and rulebased systems are combined for intrusion detection in the cloud infrastructure BID9 .

BID10 shows how Incident Detection can perform well than intrusion detection.

In BID11 discusses a detailed study on 6 different traditional ML classifiers in finding the credit card frauds, financial frauds.

Credit card frauds are detected using Convolution Neural Networks.

Fraud Detection in crowd sourcing projects is discussed in BID12 .Statistical Fraud Detection method model is trained to discriminate the fraudulent and non fraudulent using supervised and unsupervised methods in credit card frauds.

BID6 Especially in communication networks Fraud Detection are rectified using supervised learning by statistical learning of behaviour of networks us using Bayesian network approach.

Data mining approaches related to financial Fraud Detection are discussed in BID13 .

BID14 mainly discusses the Fraud Detection in today's new Online e-commerce transaction using Recurrent Neural Network(RNN) which performed very well.

Based on this a detailed survey is conducted in BID15 .

The risks and trust involved in e-commerce market are detailed studied in BID16 .

The first task is an android classification task.

The dataset is created from a set of APK packages files collected from the Opera Mobile Store from Jan to Sep 2014 is used.

This dataset consists of API(Application Programming Interface) information for 61,730 APK files where 30,897 files for training and 30,833 files for testing BID17 .

The second task is incident detection.

This dataset contains operational log file that was captured from Unified Threat Management (UTM) of UniteCloud.

Task 3 is Fraud Detection.

This dataset is anonymised data that was unified using the highly correlated rule based uniformly distributed synthetic data (HCRUD) approach by considering similar distribution of features.

To find an optimal result, three trails of experiment with 700 epochs has run with learning rate varying in the range [0.01-0.5].

The highest 10-fold cross-validation accuracy was obtained by using the learning rate of 0.01.

There was a sudden decrease in accuracy at learning rate 0.05 and finally attained highest accuracy at learning rates of 0.035, 0.045 and 0.05 in comparison to learning rate 0.01.

This accuracy may have been enhanced by running the experiments till 1000 epochs.

As more complex architectures we have experimented with, showed less performance within 500 epochs, so 0.01 as learning rate for the rest of the experiments by taking training time and computational cost into account.

The RNN 1 to 6 layer network topology are used in order to find an optimum network structure for our input data since we don't know the optimal number of layers and neurons.

We run 3 trails of experiments for each RNN network toplogy.

Each trail of the experiment was run till 700 epochs.

It was observed that most of the deep learning architectures learn the normal category patterns of input data within 400 epochs itself.

The number of epochs required to learn the malicious category data usually varies.

This complex architecture networks required a large number of iterations in order to reach the best accuracy.

At last, we obtained the bestperformed network topology for each use case.

For Task Two and Task Three, 3 layer RNN network performed well.

For Task One, the 6 layer RNN network gave a good performance in comparison to the 4 layer RNN.

Then we decided to use 6 layer RNN network for the rest of the experiments.

10-fold cross-validation accuracy of each RNN network topology for all use cases is shown in TAB2 .

An intuitive overview of our proposed RNN architecture for all use cases is shown in FIG0 This consists of the input layer with six hidden layers and an output layer.

An input layer contains 4896 neurons for Task One, 9 neurons for Task Two and 12 neurons for Task Three.

An output layer contains 2 neurons for Task One, 3 neurons for Task Two and 2 neurons for Task Three.

The detailed structure and configuration of proposed RNN architecture are shown in TAB2 .

The neurons in input to hidden layer and hidden to output layer are fully connected.

The proposed Recurrent Network is composed of recurrent layers, fully-connected layers, batch normalization layers and dropout layers.

It contains the recurrent units/neurons.

The units have self-connection/loops.

This helps to carry out the previous time step information for the future time step.

Batch Normalization and Regularization: To obviate overfitting and speed up the RNN model training, Dropout (0.001) BID18 and Batch Normalization BID19 was used in between fully-connected layers.

A dropout removes neurons with their connections randomly.

In our alternative architectures for Task 1, the recurrent networks could easily overfit the training data without regularization even when trained on large number samples.

Classification:

For classification, the final fully connected layer follows sigmoid activation function for Task One and Task Two, softmax for Task Three.

The fully connected layer absorb the non-linear kernel and sigmoid layer output zero (benign) and output one (malicious), softmax provides the probability score for each class.

The prediction loss for Task One and Task Two is estimated using binary cross entropy DISPLAYFORM0 (1) where vector predicted probability is denoted by pd testing data set, ed is a vector of the expected class label, values are either 0 or 1.The prediction loss for Task Three is estimated using categorical-cross entropy DISPLAYFORM1 where pd true probability distribution, ed is predicted probability distribution.

We have used sgd as an optimizer to minimize the loss of binary-cross entropy and categorical-cross entropy.

We have evaluated the proposed RNN model against classical machine learning classifier SVM, on 3 different cybersecurity use cases.1.Identifying Android malware based on API information, 2.Incident Detection over unified threat management (UTM) operation on Unite Cloud, 3.Fraud Detection in financial transactions.

The detailed results of proposed RNN model on 3 different use cases are displayed in TAB3 .

In this paper performance of RNN Vs other classical machine learning classifiers are evaluated for cybersecuriy use cases such as Android malware classification, incident detection, and fraud detection.

In all the three

<|TLDR|>

@highlight

Recurrent neural networks for Cybersecurity use-cases