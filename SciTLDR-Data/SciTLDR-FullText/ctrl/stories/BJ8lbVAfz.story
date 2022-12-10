While self-organizing principles have motivated much of early learning models, such principles have rarely been included in deep learning architectures.

Indeed, from a supervised learning perspective it seems that topographic constraints are rather decremental to optimal performance.

Here we study a network model that incorporates self-organizing maps into a supervised network and show how gradient learning results in a form of a self-organizing learning rule.

Moreover, we show that such a model is robust in the sense of its application to a variety of  areas, which is believed to be a hallmark of biological learning systems.

Machine learning has made significant improvements, specifically with deep neural network models BID13 , BID0 , BID4 .

Deep learning was made possible by much faster computer technology such as GPUs, and with algorithmic advancement such as BID17 BID3 BID10 , BID2 .

Learning tasks that, due to their complexity or data volume, were impossible to execute a decade ago, now can be run in reasonable time scale.

The improvements allow the applications of Deep Learning to many real world problems.

Learning good internal representations is a key aspect of deep learning.

Indeed, it is interesting to recall that the first breakthrough in deep learning came from an application of unsupervised pretraining with gradient-based fine tuning BID7 .

Restricted Boltzmann Machines (RBMs) BID6 and Autoencoders BID1 BID8 are utilized for constructing the hidden layers of early models such as Deep Belief Networks (DBN) and Deep Boltzmann Machine (DBM), , BID16 .

While much research of deep learning research focuses on learning efficiency and running performances, there is much less research into the understanding of the formation of internal representation in hierarchical neural networks.

Moreover, while self-organizing maps have been and integral part of biologically motivated learning theories since the 1970s BID19 , BID12 the role of such self-organizing mechansims are less understood in modern deep learning theories.

Topographical self-organization is often observed in biological neural networks BID11 , BID15 and thus may give new insights in understanding learning and self-organization in artificial neural networks.

Here, we propose a network that combines aspects of self-organization into a supervised network model for classification.

More specifically, in this study we modify the previously proposed Restricted Radial Basis Function Networks (rRBF) BID5 with a softmax output layer that is trained on a crossentropic cost function.

This is more consistent with a probabilistic interpretation of the class membership output function than the previous implementation which allows a more clear derivation of the emergence of the self-organizing learning aspects of this network.

We call this modified network the Softmax Restricted Radial Basis Function Networks (S-rRBF).

Through this network we argue that it is possible to build a learning model in which unsupervised self-organization and supervised learning are just different aspect of a single learning mechanism.

We show that the network achieves compatible performance with other deep network architectures while having the added feature of robustness in the sense that it compares favorable with the best performers in the studied examples, while the best performer changes for different applications.

While the results are consistent with BID20 'No free lunch theorem', they also highlight that robustness against variation of applications and not the best performance is an important part in flexible learner, which is thought to be of importance when understanding biological learning systems.

We highlight our ideas here with well understood applications examples of moderate complexity.

However, the proposed architecture can also be scaled to deeper layers and hence applied to deeper learning problems.

The main contribution here is showing algebraically the emergence of the selforganizing structures from supervised gradient learning.

We believe that this research opens new insights into the relation between unsupervised and supervised learning.

Also, we illustrate on some examples the internal representation in the competitive layer and compare it to a standard selforganizing map (SOM) BID12 and to t-Stochastic Neighborhood Embedding (t-SNE) BID18 ) that represents a deeper transformation of the feature space.

Softmax Restricted Radial Basis Function Networks (S-rRBF) is a hierarchical neural network that has one or more hidden layers where the neurons are aligned in a two dimensional grid.

For simplicity we will restrict our discussion to networks with one hidden layer as shown in FIG0 .

The S-rRBF is developed based on Restricted Radial Basis Function Networks (rRBF) introduced in BID5 .

Here, unlike the original rRBF that has a sigmoidal output layer and quadratic cost function, S-rRBF adopts the softmax output layer with a cross entropy cost function.

These modifications yields clearer understanding on the relation between the internal self-organization with supervised learning process.

So far, most studies treat self-organizing and supervised learning as two unrelated learning mechanisms.

Here, we argue that with the proposed S-rRBF it is possible to build a learning model in which self-organization is an integrated process of supervised learning, and thus giving a new perspective on the learning process of artificial neural networks.

The dynamics of the S-rRBF is as follows.

Suppose the S-rRBF is trained on a data set DISPLAYFORM0 .., C}, and d is the dimension of the input while C is the number of classes and thus the number of output neurons.

Given input, X i , the j-th hidden neuron generates output, h i j , as DISPLAYFORM1 (1) DISPLAYFORM2 In Eq. 1, σ() is a neighborhood function defined as DISPLAYFORM3 DISPLAYFORM4 where dist(win, j, t) is the Euclidean distance between the winning neuron and the j-th neuron on the two-dimensional grid of the hidden layer, while t, and t end , is the current epoch, and the target epoch when the learning process is terminated.

The activation function of a hidden neuron in S-rRBF is similar to that of Radial Basis Function Networks (RBF) BID14 ) except that in S-rRBF it is topologically restricted by the neighborhood function σ(win, j).The output of the hidden neurons are then propagated to the output layers, where the k-th output, O k , in the output layer is defined as DISPLAYFORM5 The conditional probability that the S-rRBF classifies the input into the class k is given by DISPLAYFORM6 The S-rRBF is then trained to minimize the cross entropy, DISPLAYFORM7 Considering that Y i ∈ {1, ..., C}, Eq. 6 can be rewritten as DISPLAYFORM8 In Eq. 7, Π( DISPLAYFORM9 To minimize the cross entropy, its gradients are calculated as DISPLAYFORM10 Hence, the modification of the weight vector leading to the j-th output neuron is as follows.

DISPLAYFORM11 Eq. 9 shows that the values of connection weights leading to an output neuron are increased if that neuron is associated with the true label of the input and are decreased otherwise.

Consequently these modifications increase the probability that the S-rRBF predicts the correct class.

Also, DISPLAYFORM12 In calculating Eq. 10, considering the weight vector W n is only relevant to the output of the n-th hidden neuron, h n , the equation can be rewritten as DISPLAYFORM13 is the weighted average of the connection weights from the n-th hidden neuron to the output layer, with the conditional probabilities of the all possible classes being selected as the predicted class by the S-rRBF as the weighting coefficients.

Defining,ṽ n = l v ln P (Y i = l|W, V, X i ), Eq. 11 can be expressed as DISPLAYFORM14 When the true class of the given input X i is K, hence Π(Y i = K) = 1 and 0 for all other classes, Eq. 12 becomes as follows, DISPLAYFORM15 Hence the modification of the n-th reference vector is given by DISPLAYFORM16 Eq. 14 shows that a self-organizing process, similar to that of Kohonen's SOM BID12 shown in Eq. 15, occurs in the internal layer during the supervised training process of S-rRBF.

The self-organization occurs as a mathematical implication of the cost function minimization.

It shows that it is possible to link topological self-organization and supervised learning, which are often treated as different learning mechanism, in a single supervised learning model.

DISPLAYFORM17 The recent surge of deep learning triggers a natural interest in learning representations.

Representations extracted by Autoencoders, RBMs and other recent deep models have been extensively studied.

However, although often observed in biological neural networks, so far topographical representations in hierarchical learning models of artificial neural networks have not been well studied.

Here, we show that a topographical structure is feasible for internal representation in hierarchical supervised learning of neural networks.

It is important to mentioned that the internal self-organization here is different from that of a SOM, in that in a SOM, as shown in Eq. 15, the reference vector is always modified towards the input vector while the direction of self-organization in S-rRBF is regulated by the sign of (v Kn (t) −ṽ n (t)).

The sign of this regularization term is decided by the relative value of the weight connecting the neuron associated with the n-th reference with the output associated with the true class of the input.

If the weight leading to the output neuron associated with the true class is larger than the expected value of weight connection leading from the neuron associated with the n-th reference vector, a "positive" self-organization as in SOM occurs, while if the value of the weight is below average a "negative" self-organization that moves the reference vector away from the input occurs.

In this context, the self-organization process in SOM is label-free, while in S-rRBF it is label-oriented.

While the internal self-organization in this study is not fully unsupervised, as it depends on the labels of the input, it does not require the exact information of the output error but only relative value of connection weight from a particular hidden neuron leading to the output neuron.

Hence, it is not supervised in the strict sense either.

We consider that the semi-supervised self-organization here is a good starting point in further study to connect unsupervised learning with the supervised learning scheme.

Furthermore, the term e − X(t)−Wn(t) 2 in Eq. 14 also triggers a dropout effect BID2 , BID17 , resulting in a sparse network in that hidden neurons associated with reference vectors that differ greatly from the input X are inhibited.

We tested the S-rRBF on a variety of standard machine learning benchmark problems and compared it against three deep learning models, namely a Deep Belief Network (DBN), a Stacked Autoencoder (SAE), and a ReLU MLP with softmax output layers and crossentropic cost function.

The average classification error rates over 15-fold cross validation test are shown in Table.

1.

In those experiments, the number of hidden neurons, as well as the structures for the deep neural networks were empirically tried, and the results of the best settings were registered for comparison.

In TAB0 the performance of the best algorithm is highlighted in bold.

The results indicate that although S-rRBF does not always outperform the three deep networks, it generally compares favorably with the best performing deep model.

To show the properties of the internal representation of the S-rRBF, the resulting internal representations of the S-rRBF for some of the benchmark problems are shown.

The first one is the internal representation for the Iris problem, a well known 3-classed problem where one of the classes are linearly separable from the other non-linearly separable two.

The self-organized internal topographical representation of the S-rRBF for this problem is shown in Fig. 2a .

For comparison, 2-D visualizations of SOM andt-SNE are respectively shown in Fig. 2b and Fig. 2c .

In these 2-D maps, each class is represented with different marker and color, while × on the maps show the overlapping representation of some data belonging to contrasting classes.

The size of the marker reflects the number of data that it represents.

It should be noted that for SOM and t-SNE the 2-D representations are constructed based only on the similarities of the data while their labels are irrelevant.

The internal representation of the S-rRBF is different in that during the learning process, the directions of the topological self-organization are regulated by the labels of the data.

Hence, the internal representation of the S-rRBF is context-dependent.

The three representations reflect the problem's separability well.

This is an easy problem in that the data distribution is consistent with the labels distribution.

The simplicity of the problem is also obvious from the high classification performances of the compared algorithms in TAB0 .The second example is Bank Marketing Data, a 48-D, 3-classed problem.

The low dimensional representations of SOM in Fig. 3b and t-SNE in Fig. 3c indicate that there are many overlapping data belonging to contrasting classes that make this problem a relatively difficult one.

Figure 3a indicates that the S-rRBF generates a nice topographical internal representation that illustrates how the classifier separates the two classes.

The × in S-rRBF's representation indicates the area where data are likely to be misclassified.

The third example is the recently proposed "Fashion MNIST", an apparel-related image classification problem BID21 .

Some of the data of this data set are shown in Fig. 5 .

This data set has the same dimensionality, class number and data size than the traditional MNIST data set.

The SOM and t-SNE representations of this problem are respectively shown in Fig. 5b and Fig. 5a shows that it did not form a clearly distinctive class representations as in the previous two examples.

The internal representations offer understanding on how the S-rRBF self-organizes the data to be further classified in the output layer.

In this research we showed that it is possible to build a hierarchical neural network that self-organizes with context-relevant topographical internal representation.

More specifically, we showed that topographical self-organization can emerge as an implication of the supervised learning.

Thus, the two learning processes of self-organization and supervised learning, which are often considered to be unrelated, are can be viewed as two different aspects of a single learning mechanism.

The two learning processes are only distinguished by the layers where they occurs.

The internal self-organization in this network is not fully unsupervised.

However, the direction of the self-organization process in a hidden neuron is only decided by the relative value of the connection weight leading from the neuron to the output neuron relevant to the true label of the input and thus not dependent on the supervised error.

The experiments show that the classification performance of the proposed model is comparable to that of standard supervised networks.

While the proposed model does not always outperform existing conventional models, we found that the performance was comparable to the best performer for most of the diverse benchmark applications.

Specific machine learning methods often perform well on datasets for which they have been designed, but it is well acknowledged that sufficient performance in a variety of tasks is useful in many applications such as robotics and probably to understand better human abilities.

Another advantage of our system is its 2-dimensional internal layer offers auxiliary visual information on its learning representations.

The S-rRBFcan can readily expanded into deep networks.

As layered networks transfer transform inputs (physical stimuli) into labels (concepts) in a layer by layer manner, the visualization of internal layers in multi-layered S-rRBF can be considered as concept-forming visualization.

The visualization can potentially offer new insights for machine learning.

<|TLDR|>

@highlight

integration of self-organization and supervised learning in a hierarchical neural network

@highlight

The paper discusses learning in a neural network with three layers, where the middle layer is topographically organized and investigates interplay between unsupervised and hierarchical supervised learning in biological context.

@highlight

A supervised variant of Kohonen's self-organizing map (SOM), but where the linear output layer is replaced with squared error by a softmax layer with cross-entropy.

@highlight

Proposes a model using hidden neurons with self-organising activation function, whose outputs feed to classifier with softmax output function. 