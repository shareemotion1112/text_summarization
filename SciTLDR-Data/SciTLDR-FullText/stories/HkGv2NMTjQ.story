State of the art sound event classification relies in neural networks to learn the associations between class labels and audio recordings within a dataset.

These datasets typically define an ontology to create a structure that relates these sound classes with more abstract super classes.

Hence, the ontology serves as a source of domain knowledge representation of sounds.

However, the ontology information is rarely considered, and specially under explored to model neural network architectures.

We propose two ontology-based neural network architectures for sound event classification.

We defined a framework to design simple network architectures that preserve an ontological structure.

The networks are trained and evaluated using two of the most common sound event classification datasets.

Results show an improvement in classification performance demonstrating the benefits of including the ontological information.

Humans can identify a large number of sounds in their environments e.g., a baby crying, a wailing ambulance siren, microwave bell.

These sounds can be related to more abstract categories that aid interpretation e.g., humans, emergency vehicles, home.

These relations and structures can be represented by ontologies BID0 , which are defined for most of the available datasets for sound event classification (SEC).

However, sound event classification rarely exploits this additional available information.

Moreover, although neural networks are the state of the art for SEC BID1 BID2 BID3 , they are rarely designed considering such ontologies.

An ontology is a formal representation of domain knowledge through categories and relationships that can provide structure to the training data and the neural network architecture.

The most common type of ontologies are based on abstraction hierarchies defined by linguistics, where a super category represents its subcategories.

Generally, the taxonomies are defined by either nouns or verbs e.g., animal contains dog and cat, dog contains dog barking and dog howling.

Examples of datasets are ESC-50 BID4 , UrbanSounds BID5 , DCASE BID6 , AudioSet BID7 .

Another taxonomy can be defined by interactions between objects and materials, actions and descriptors e.g., contains Scraping, which contains Scraping Rapidly and Scraping a Board BID8 BID9 BID10 .

Another example of this type is given by physical properties, such as frequency and time patterns BID11 BID12 BID13 .

There are multiple benefits of considering hierarchical relations in sound event classifiers.

They can allow the classifier to back-off to more general categories when encountering ambiguity among subcategories.

They can disambiguate classes that are acoustically similar, but not semantically.

They can be used to penalize classification differently, where miss classifying sounds from different super classes is worse than within the same super class.

Lastly, they can be used as domain knowledge to model neural networks.

In fact, ontological information has been evaluated in computer vision BID14 and music BID15 , but has rarely been used for sound event classification.

Ontology-based network architectures have showed improvement in performance along with other benefits.

Authors in BID16 proposed an ontology-based deep restricted Boltzmann machine for textual topic classification.

The architecture replicates the tree-like structure adding intermediate layers to model the transformation from a super class to its sub classes.

Authors showed improved performance and reduced overfitting in training data.

Another example used a perceptron for each node of the hierarchy, which classified whether an image corresponded to such class or not BID17 .

Authors showed an improvement in performance due to the ability of class disambiguation by comparing predictions of classes and sub classes.

Motivated by these approaches and by the flexibility to adapt structures in a deep learning model we propose our ontology-based networks detailed in the following section.

In this section we present a framework to deal with ontological information using deep learning architectures.

First, we describe a set of assumptions we consider along this paper.

In particular, we describe the type of ontologies we work and some of their implications.

Later, we present a Feed-forward model that includes the discussed constraints, defining our proposed ontological layer.

Second, in order to preserve an embedding space consistent with the ontological structure, we extended the learning model to compute ontology-based embeddings using Siamese Neural Networks.

The framework is defined to make use of the ontology structure and to model the neural network architectures.

It should be noted that we considered ontologies with two levels, which are the most common in sound event datasets.

Nevertheless, the presented framework can be easily generalized to more levels.

In our framework, we considered the training data {(x 1 , y 1 ), ..., (x n , y n )}, where x i ∈ X is an audio representation, which is associated to a set of labels given by the ontology DISPLAYFORM0 In this case, C i is the set of possible classes at i-level.

Assuming a hierarchical relation, we can consider that each possible class in C i is mapped to one element in C i+1 .

The higher the value of i, the higher the level in the ontology.

For example, consider the illustration of an ontology in FIG0 .

In this case k = 2, C 1 = {cat , dog, breathing , eating , sneezing , violin , drums , piano , beep , boing , train , siren} and C 2 = {nature , human , music , effects , urban}. As the figure shows, every element in C 1 is related to one element in C 2 ; e.g., cat belongs to nature, or drums belongs to music.

Furthermore, for a given representation x ∈ X , if we know the corresponding label y 1 in C 1 , we can infer its label in C 2 .

This intuition can be formalized using a probabilistic formulation, where it is Net linear + softmax Figure 2 :

Architecture of the Feed-forward Network with Ontological Layer.

The blue column represents the acoustic feature vector, the red columns are the output probabilities for both levels.straightforward to see that, assuming p(y 2 |y 1 , x) = p(y 2 |y 1 ), the following is satisfied: DISPLAYFORM1 Therefore, if we want to estimate p(y 2 |x) using a model, we just need to compute the estimation of p(y 1 |x) and sum the values corresponding to the children of y 2 .

This case is valid for inference time, however, it is not clear that using the representation and label (x, y 1 ) should be enough to train the model.

If at training time we can make use of knowledge to relate the different classes in y 1 , it should improve the performance of the model, specially at making predictions for classes y 2 .In the following sections we take our proposed framework and use it to design ontology-based neural network architectures.

In this section, we describe how we use our proposed framework to design the architecture.

Also, we introduce the ontological layer, which makes use of the ontology structure.

The Feed-forward Network (FFN) with Ontological Layer consists of a base network (Net), an intermediate vector z, and two outputs, one for each ontology level.

The base network weights are learned at every parameter update.

The base network utilizes an input vector of audio features x and generates a vector z. This vector is used to generate two outputs, p(y 1 |x) a probability vector for C 1 and p(y 2 |x) a probability vector for C 2 .

First, the vector z is passed to a softmax layer of the size of C 1 .

Then, this output is multiplied by the ontological layer M and generates a layer of size of C 2 .

Once the FFN is trained, it can be used to predict any class in C 1 and C 2 for any input x.

The ontological layer reflects the relation between super classes and sub classes given by the ontology.

To describe how we used this layer, we refer to Equation 3, where p(y 2 |x) is the sum of all the values of p(y 1 |x) corresponding to the children of y 2 .

If we consider this equation as a directed graph where M is the |C 2 | × |C 1 | incidence matrix, then, it is clear that Equation 3 can be rewritten as, DISPLAYFORM0 Note that the ontological layer M defines the weights of a standard layer connection.

Although we do not consider that these weights are trainable, they are part of our training data.

In order to train this model, we simply propose to apply gradient-based method to minimize the loss function L, which is a convex combination between two categorical cross-entropy functions; L 1 the categorical cross entropy corresponding to p(y 1 |x) and L 2 corresponding to p(y 2 |x).

Formally, DISPLAYFORM1 Hence, we consider λ ∈ [0, 1] as a hyper parameter to be tuned.

Note that, when λ = 1, we are reducing the problem to train a standard classifier just using the information from the first level of the ontology.

In this section, we describe how we learned the ontology-based embeddings.

Our goal is to create embeddings that preserved the ontological structure.

We used a Siamese neural network (SNN), which enforces samples of the same class to be closer, while separating samples of different classes.

If two samples belong to different subclasses, but they belong to the same super class, they are closer than two samples that belong to different super classes.

The architecture of the SNN with the Feed-forward Network with Ontological Layer is shown in FIG1 .

The blue rows represent the acoustic feature vectors of two different samples; they can be from the same subclass, different subclass but same super class, or different super class.

Then, the twin networks have the same base architecture (Net) with shared weights.

The weights are learned simultaneously at every parameter update.

The white rows represent the ontological embeddings used to compute a Similarity metric (Euclidean Distance), where the distance of the embeddings z 1 and z 2 should indicate how different x 1 and x 2 are with respect to the ontology.

For this work, we imposed that the distance between z 1 and z 2 is close to 0 if the samples are from the same subclass, close to 5 if they are from different sub classes, but the same super class, and close to 10 if they are from different super classes.

Finally, the red rows are the output probabilities for both levels, p(y 1 |x 1 ), p(y 1 |x 2 ), p(y 2 |x 1 ) and p(y 2 |x 2 ).To train the Feed-forward Model with Ontological layer using Ontology-based embeddings, we provided the three types of pairs of audio examples and applied a gradient-based method to minimize DISPLAYFORM0

In this section, we evaluate the sound event classification performance of the ontological-based neural network architectures.

We present the datasets and its ontologies, the baseline and proposed architectures, and the classification performance at different levels of the hierarchy.

Making Sense of Sounds Challenge 2 -MSoS: The dataset is designed for a challenge which objective is to classify the most abstract classes or highest level in its taxonomy.

The ontology, illustrated in FIG0 has two levels, the lowest level 1, has 97 classes and the highest level 2, has 5 classes.

The audio files were taken from Freesound data base, the ESC-50 dataset and the Cambridge-MT Multitrack Download Library.

The development dataset consists of 1500 audio files divided into the five categories, each containing 300 files.

The number of different sound types within each category is not balanced.

The evaluation dataset consists of 500 audio files, 100 files per category.

All files have an identical format: single-channel 44.1 kHz, 16-bit .wav files.

We randomly partitioned the set in 80% for training and tuning parameters and 10% for testing.

All files are exactly 5 seconds long, but may feature periods of silence.

The official blind evaluation set of the challenge consisted on 500 files distributed among the 5 classes.

Urban Sounds -US8K: The dataset is designed to evaluate classification of urban sounds, which are organized using a taxonomy with more nodes than the annotated number of classes.

Due to this reason, we adjusted the taxonomy to avoid redundant levels with only one annotated child.

The resulting ontology is illustrated in FIG2 , with two levels, the lowest level 1, has 10 classes and the highest level 2, has 4 classes.

The audio files were taken from Freesound data base and corresponded to real field recordings.

All files have an identical format: single-channel 44.1 kHz, 16-bit .wav files.

The dataset contains 8,732 audio files divided into 10 stratified subsets.

We used 9 folds to train and tune parameters and one fold for testing.

We used state-of-the-art Walnet features BID1 to represent audio recordings.

For each audio, we computed a 128-dimensional logmel-spectrogram vector and transformed it via a convolutional neural network (CNN) that was trained separately on the balanced set of AudioSet.

The architecture of the base network (Net) considered in this experiment, shown in Fig. 2 , is a feed-forward multi-layer perceptron network.

It consists of 4 layers: the input layer of dimensionality 1024, which takes audio feature vectors, 2 dense layers of dimensionality 512 and 256, respectively, and the output layer of dimensionality 128, which is the dimensionality of the vector z. The dense layers utilize Batch Normalization, a dropout rate of 0.5 and the ReLU activation function; max(0, x), where x is input to the function.

We tuned the parameters in the Net box as well as the parameters that transformed z into p(y 1 |x).

We considered baseline models in both level 1 and 2 for the different data sets.

In this case, the baseline models did not consider any ontological information, hence the models consist of the Base Network Architecture with the addition of an output layer that was either for level 1 or level 2.Note that for level 1 this is equivalent to training the Feed-forward model with Ontological Layer using λ = 1.

Indeed, with λ = 1 the loss function associated to level 2 is not considered.

For level 2, the baseline model is different from the Feed-forward model with λ = 0, because in the baseline model there is no layer corresponding to the prediction of y 1 .

Table 1 shows the results of baseline models for both, MSoS and US8K data set in level 1 and level 2.The baseline performance of the development set in the MSoS challenge was reported to be 0.81 for level 2 and no baseline was provided for level 1.

To validate the architecture presented in Section 2.2 and analyze the utility of the ontological layer, we trained models taking different values of λ.

FIG3 shows the effect of λ in both data sets.

In general, we observe that considering values different from 0 and 1 helps to increase the performance.

Note that the classification in both level is affected by the ontological layer.

In the case of MSoS data set, the best performance was obtained using λ = 0.8, getting 0.74 and 0.913 of accuracy in level 1 and 2 respectively.

Thus, using the ontological structure we can get an absolute improvement of 5.4% and 6% respect baseline models.

Running the same experiment on the US8K data set, we observe a smaller improvement.

The best performance was obtained using λ = 0.7, being the accuracy of 0.82 and 0.86 for level 1 and 2 respectively.

This means an improvement of 2.5% and 0.2% only, respect baseline models.

Best result is achieved using λ = 0.8 (Right) Results in US8K data set.

Best result is achieved using λ = 0.7 Figure 6 : The MSoS t-SNE plots of the samples in classes from level 2 (1st and 3rd) and level 1 (2nd and 4th).

The first two boxes are from the base network vectors and the second two boxes are the ontology-based embeddings.

We observe in 1st and 3rd, the groups of classes in level 2 and in 2nd and 4th the same level 2 groups, but using the level 1 class samples.

The ontology-based embeddings results in tighter and better defined clusters.

We tested the architecture described in Section 2.3 to evaluate the performance of the ontology-based embeddings for sound event classification.

Additionally, we include t-SNE plots, to illustrate how the embeddings cluster at different levels.

We processed the Walnet audio features and chose different super and sub class pairs to train the Siamese neural network to produce the ontology-based embeddings.

The embeddings are passed to the architecture of the base network (Net), which is the same as the one used in the previous section.

We trained the SNN for 50 epochs using the Adam algorithm.

We also tuned the hyper-parameters of the SNN to achieve good performance with the input features that are described in the next section.

We also tried different number of pairs for the input training data, from 100 to 1,000,000 pairs and found that 100,000 yielded the best performance.

For the loss function we used the values derived in the previous experiment.

We used the value of 0.8 for the lambda of the classifiers of level 2 and 0.2 for the classifiers in level 1, and 0.2 for the similarity metric.

Modifying the lambdas in the loss function affected the overall performance.

The results in Table 1 show that the accuracy performance of MSoS and US8K were respectively as follows, in level 1 0.736 and 0.818, and in level 2 0.886 and 0.856.

Based on these results we made the following conclusions.

The performance of this architecture is better than the baseline, but slightly under performed the method without the embeddings.

Nevertheless, the ontology-based embeddings have the benefit of better grouping as illustrated in Figure6.

We took the MSoS data and created the t-SNE plots (perplexity=30) of the classes in level 2 and level 1.

We observed that the FF + Ontology vectors and the ontology-based embeddings provided clustered groups of level 2 classes.

However, the ontology-based embeddings have tighter and better defined clusters.

In the case of the US8K data set performance was limited.

We think this was because the number of sub classes was similar to the number of super classes.

We had 10 sub classes for 4 classes unlike the MSoS data set, where we had 97 sub classes and 5 classes.

It seems when the ratio between the number of sub classes and the number of classes is not large, the contribution of the ontology is negligible.

Both approaches were used to compete in the Making Sense of Sounds Challenge.

The baseline for the blind evaluation set was 0.80 accuracy for level 2.

The Feed-forward Network with Ontological Layer achieved 0.88 while using the ontological-embeddings achieved 0.89.

Again, both architectures outperformed significantly the baseline.

In this paper we proposed a framework to design neural networks for sound event classification using hierarchical ontologies.

We have shown two methods to add such structure into deep learning models in a simple manner without adding more learnable parameters.

We used a Feed-forward Network with an ontological layer to relate predictions of different levels in the hierarchy.

Additionally, we proposed a Siamese neural Network to compute ontology-based embeddings to preserve the ontology in an embedding space.

The embeddings plots showed clusters of super classes containing different sub classes.

Our results in the datasets and MSoS challenge improved over the baselines.

We expect that our results pave the path to further explore ontologies and other relations, which is fundamental for sound event classification due to wide acoustic diversity and limited lexicalized terms to describe sounds.

@highlight

We present ontology-based neural network architectures for sound event classification.