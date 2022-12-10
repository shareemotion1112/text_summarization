Deep neural networks (DNNs) have witnessed as a powerful approach in this year by solving long-standing Artificial intelligence (AI) supervised and unsupervised tasks exists in natural language processing, speech processing, computer vision and others.

In this paper, we attempt to apply DNNs on three different cyber security use cases: Android malware classification, incident detection and fraud detection.

The data set of each use case contains real known benign and malicious activities samples.

These use cases are part of Cybersecurity Data Mining Competition (CDMC) 2017.

The efficient network architecture for DNNs is chosen by conducting various trails of experiments for network parameters and network structures.

The experiments of such chosen efficient configurations of DNNs are run up to 1000 epochs with learning rate set in the range [0.01-0.5].

Experiments of DNNs performed well in comparison to the classical machine learning algorithm in all cases of experiments of cyber security use cases.

This is due to the fact that DNNs implicitly extract and build better features, identifies the characteristics of the data that lead to better accuracy.

The best accuracy obtained by DNNs and XGBoost on Android malware classification 0.940 and 0.741, incident detection 1.00 and 0.997, and fraud detection 0.972 and 0.916 respectively.

The accuracy obtained by DNNs varies -0.05%, +0.02%, -0.01% from the top scored system in CDMC 2017 tasks.

In this era of technical modernization, explosion of new opportunities and efficient potential resources for organizations have emerged but at the same time these technologies have resulted in threats to the economy.

In such a scenario proper security measures plays a major role.

Now days, hacking has become a common practice in organizations in order to steal data and information.

This highlights the need for an efficient system to detect and prevent the fraudulent activities.

cyber security is all about the protection of systems, networks and data in the cyberspace.

Malware remains one of the maximum enormous security threats on the Internet.

Malware are the softwares which indicate malicious activity of the file or programs.

These are unwanted programs since they cause harm to the intended use of the system by making it behave in a very different manner than it is supposed to behave.

Solutions with Antivirus and blacklists are used as the primary weapons of resistance against these malwares.

Both approaches are not effective.

This can only be used as an initial shelter in real time malware detection system.

This is primarily due to the fact that both approaches completely fails in detecting the new malware that is created using polymorphic, metamorphic, domain flux and IP flux.

Machine learning algorithms have played a pivotal role in several use cases of cyber security BID0 .

Fortunately, deep learning approaches are prevailing subject in recent days due to the remarkable performance in various long-standing artificial intelligence (AI) supervised and unsupervised challenges BID1 .

This paper evaluates the effectiveness of deep neural networks (DNNs) for cyber security use cases: Android malware classification, incident detection and fraud detection.

The paper is structured as follows.

Section II discusses the related work.

Section III discusses the background knowledge of deep neural networks (DNNs).

Section IV presents the proposed methodology including the description of the data set.

Results are displayed in Section V. Conclusion is placed in Section VI.

This section discusses the related work for cyber security use cases: Android malware classification, incident detection and fraud detection.

Static and dynamic analysis is the most commonly used approaches in Android malware detection BID2 .

In static analysis, android permissions are collected by unpacking or disassembling the app.

In dynamic analysis, the run-time execution characteristics such as system calls, network connections, power consumption, user interactions and memory utilization.

Mostly, commercial systems use combination of both the static and dynamic analysis.

In Android devices, static analysis is preferred due to the following advantageous such as less computational cost, low resource utilization, light-weight and less time consuming.

However, dynamic analysis has the capability to detect the metamorphic and polymorphic malwares.

In BID3 evaluated the performance of traditional machine learning classifiers for android malware detection with using the permission, API calls and combination of both the API calls and permission as features.

These 3 different feature sets were collected from the 2510 APK files.

All traditional machine learning classifiers performance is good with combination of API calls and permission feature set in comparison to the API calls as well as permission.

BID4 proposed MalDozer that use sequences of API calls with deep learning to detect Android malware and classify them to their corresponding family.

The system has performed well in both private and public data sets, Malgenome, Drebin.

Recently, the privacy and security for cloud computing is briefly discussed by BID5 .

The discussed various 28 cloud security issues and categorized those issues into five major categories.

BID6 proposed machine learning based anomaly detection that acts on different layers e.g. the network, the service, or the workflow layers.

BID7 discussed the issues in creating the intrusion detection for the cloud infrastructure.

Also, how rule based and machine learning based system can be combined as hybrid system is shown.

BID8 discussed the security problems in cloud and proposed incident detection system.

They showed how incident detection system can perform well in comparison to the intrusion detection.

In BID9 did comparative study of six different traditional machine learning classifiers in identifying the financial fraud.

In BID10 discussed the applicability of data mining approaches for financial fraud detection.

Deep learning is a sub model of machine learning technique largely used by researchers in recent days.

This has been applied for various cyber security use cases BID11 , BID12 , BID13 , BID14 , BID15 , BID16 , BID17 , BID18 .

Following, this paper proposes a unique DNN architecture which works efficiently on various cyber security use cases.

The purpose of this section is to discuss the concepts of deep neural networks (DNNs) architecture concisely and promising techniques behind to train DNNs.

Artificial neural networks (ANNs) represent a directed graph in which a set of artificial neuron generally called as units in mathematical model that are connected together with edges.

This influenced by the characteristics of biological neural networks, where nodes represent biological neurons and edges represent synapses.

A feed forward network is a type of ANNs.

A feed forward network (FFN) consists of a set of units that are connected together with edges in a single direction without formation of a cycle.

They are simple and most commonly used algorithm.

Multi-layer perceptron (MLP) is a subset of FFN that consist of 3 or more layers with a number of artificial neurons, termed as units.

The 3 layers are input layer, a hidden layer and output layer.

There is a possibility to increase the number of hidden layers when the data is complex in nature.

So, the number of hidden layer is parameterized and relies on the complexity of the data.

These units together form an acyclic graph that passes information or signals in forward direction from layer to layer without the dependence of past input.

MLP can be written as O : R p × R q where p and q are the size of the input vector x = x 1 , x 2 , · · · , x p−1 , x p and output vector O(x) respectively.

The computation of each hidden layer Hl i can be mathematically formulated as follows.

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 When then network consist of l hidden layers then the combined representation of them can be generally defined as, Rectified linear units (ReLU ) have been turned out to be more proficient and are capable of accelerating the entire training process altogether BID19 .

Selecting ReLU is a more efficient way when considering the time cost of training the vast amount of data.

The reason being that not only does it substantially speeds up the training process but also possesses some advantages when comparing to the traditional activation function including logistic sigmoid function and hyperbolic tangent function BID20 .

We refer to neurons with this nonlinearity following BID21 .

DISPLAYFORM4

We consider TensorFlow BID22 in conjunction with Keras BID23 as software framework.

To increase the speed of gradient descent computations of deep learning architectures, we use with GPU enabled TensorFlow in single NVidia GK110BGL Tesla k40.

All deep learning architectures are trained using the back propagation through time (BPTT) technique.

Task 1 (Android Malware Classification): This data set includes 37,107 unique API information from 61,730 APK files BID24 .

These APK (application package) files were collected from the Opera Mobile Store over the period of January to September of 2014.

When a user runs an application, a set of APIs will be called.

Each API is related to a particular permission.

The execution of the API may solely achieve success within the case that the permission is granted by the user.

These permissions are grouped into Normal, Dangerous, Signature and Signature Or System in Android.

These permissions are explicitly mentioned in the AndroidManifest.xml file of APK by application developers.

Task 2 (Incident Detection): This dataset contains operational log file that was captured from Unified Threat Management (UTM) of UniteCloud BID25 .

UniteCloud uses resilient private cloud infrastructure to supply e-learning and e-research services for tertiary students and staffs in New Zealand.

Unified Threat Management is a rule based real-time running system for UniteCloud server.

Each sample of a log file contains nine features.

These features are operational measurements of 9 different sensors in UTM system.

Each sample is labeled based on the knowledge related to the incident status of the log samples.

Task 3 (Fraud Detection): This dataset is anonymised data that was unified using the highly correlated rule based uniformly distributed synthetic data (HCRUD) approach by considering similar distribution of features BID26 .

The detailed statistics of Task 1, Task 2 and Task 3 data sets are reported in TAB0 In order to find an optimal learning rate, we run two trails of experiment till 500 epochs with learning rate varying in the range [0.01-0.5].

The highest 10-fold cross validation accuracy was obtained by using the learning rate of 0.1.

There was a sudden decrease in accuracy at learning rate 0.2 and finally attained highest accuracy at learning rates of 0.35, 0.45 and 0.45 in comparison to learning rate 0.1.

This accuracy may have been enhanced by running the experiments till 1000 epochs.

As more complex architectures we have experimented with, showed less performance within 500 epochs, we decided to use 0.1 as learning rate for the rest of the experiments after considering the factors of training time and computational cost.

The following network topologies are used in order to find an optimum network structure for our input data.

1) DNN 1 layer 2) DNN 2 layer 3) DNN 3 layer 4) DNN 4 layer 5) DNN 5 layer For all the above network topologies, we run 2 trails of experiments.

Each trail of experiment was run till 500 epochs.

It was observed that most of the deep learning architectures learn the normal category patterns of input data within 600 epochs.

The number of epochs required to learn the malicious category data usually varies.

The complex architecture networks required large number of iterations in order to reach the best accuracy.

Finally, we obtained the best performed network topology for each use case.

For Task 2 and Task 3, 4 layer DNNs network performed well.

For Task 1, the performance of 5 layer DNNs network is good in comparison to the 4 layer DNNs.

We decided to use 5 layer DNNs network for the rest of the experiments.

10-fold cross validation accuracy of each DNNs network topology for all use cases is shown in TAB0 .

An intuitive overview of proposed DNNs architecture, Deep-Net for all use cases is shown in FIG0 This contains an input layer, 5 hidden layer and output layer.

An input layer contains 4896 neurons for Task 1, 9 neurons for Task 2 and 12 neurons for Task 3.

An output layer contains 2 neurons for Task 1, 3 neurons for Task 2 and 2 neurons for Task 3.

The details about the structure and configuration details of proposed DNNs architecture is shown in TAB0 .

The units in input to hidden layer and hidden to output layer are fully connected.

DNNs network is trained using the backpropogation mechanism BID1 .

The proposed deep neural network is composed of fully-connected layers, batch normalization layers and dropout layers.

Fully-connected layers: The units in this layer have connection to every other unit in the succeeding layer.

Thats why this layer is called as fullyconnected layer.

Generally, these fully-connected layers map the data into high dimension.

The more the dimensions the data has the more accurate the data will be in determining the accurate output.

It uses ReLU as non-linear activation function.

Batch Normalization and Regularization: To obviate over fitting and speed up DNNs model training, Dropout (0.01) BID27 and Batch Normalization BID28 was used in between fully-connected layers.

A dropout removes neurons with their connections randomly.

In our alternative architectures for Task 1, the deep networks could easily overfit the training data without regularization even when trained on large number samples.

Classification:

For classification, the final fully connected layer follows sigmoid activation function for Task 1 and Task 2, sof tmax for Task 3.

The fully connected layer absorb the non-linear kernel and sigmoid layer output 0 (benign) and 1 (malicious), sof tmax provides the probability score for each class.

The prediction loss for Task 1 and Task 2 is estimated using binary cross entropy DISPLAYFORM0 where pd is a vector of predicted probability for all samples in testing data set, ed is a vector of expected class label, values are either 0 or 1.The prediction loss for Task 3 is estimated using categorical-cross entropy DISPLAYFORM1 where ed is true probability distribution, pd is predicted probability distribution.

We have used sgd as an optimizer to minimize the loss of binary-cross entropy and categorical-cross entropy.

We evaluate proposed DNNs model against classical machine learning classifier, on three different cyber security use cases.

The first use case is identifying Android malware based on API information, the second use case is incident detection over unified threat management (UTM) operation on UniteCloud and the third use case is fraud detection in financial transactions.

During training, we pass matrix of shape 30897*4896 for Task 1, 70000*9 for Task 2 and 70000*9 for Task 3 to the input layer of DNNs.

These inputs are passed to more than one hidden layer (specifically 5) and output layer contains 1 neuron for Task 1 and Task 2, 3 neurons for Task TAB0 XGBoost is short for Extreme Gradient Boosting, where the term Gradient Boosting is proposed in the paper Greedy Function Approximation BID29 .

XGBoost is based on this original model.

XGBoost is used for the given supervised learning problems (Task1, Task2 and Task3), where we use the training data (with multiple features) to predict a target variable.

Here "multi:softmax" is used to perform the classification.

After the observation and experiment, "max depth" of the tree set it as 20.

10 fold cross validation is performed to observe the training accuracy.

Except Task 1, data are loaded as it is using Pandas 1 .

The "NaN" values are replaced with 0.

In Task 1 the data is represented as a term -document matrix, where the vocabulary built using the API indication numbers in train and test.

The scikit-learn BID11 count vectorizer is used to develop the termdocument matrix.

On the successive representation, the data are fed to the XG Booster for prediction.

The winner of CDMC 2017 tasks has acheived 0.9405, 0.9998 and 0.9824 on Task 1, Task 2 and Task 3 respectively using Random Forest classifier with Python scikit-learn BID11 .

The proposed method has performed well on Task 2 in comparision to the winner of CDMC 2017 and the accuracy obtained by DNNs varies -0.05%, -0.01% from the winner of CDMC 2017.

The reported results of DNNs can be further enhanced by simply adding hidden layers to the existing architecture that we are incompetent to try.

Moreover, the proposed method can implicitly obtain the best features itself.

This paper has evaluated the performance of deep neural networks (DNNs) for cyber security uses cases: Android malware classification, incident detection and fraud detection.

Additionally, other classical machine learning classifier is used.

In all cases, the performance of DNNs is good in comparison to the classical machine learning classifier.

Moreover, the same architecture is able to perform better than the other classical machine learning classifier in all use cases.

The reported results of DNNs can be further improved by promoting training or stacking a few more layer to the existing architectures.

This will be remained as one of the direction towards the future work.

@highlight

Deep-Net: Deep Neural Network for Cyber Security Use Cases