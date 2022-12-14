Automatic classification of objects is one of the most important tasks in engineering and data mining applications.

Although using more complex and advanced classifiers can help to improve the accuracy of classification systems, it can be done by analyzing data sets and their features for a particular problem.

Feature combination is the one which can improve the quality of the features.

In this paper, a structure similar to Feed-Forward Neural Network (FFNN) is used to generate an optimized linear or non-linear combination of features for classification.

Genetic Algorithm (GA) is applied to update weights and biases.

Since nature of data sets and their features impact on the effectiveness of combination and classification system, linear and non-linear activation functions (or transfer function) are used to achieve more reliable system.

Experiments of several UCI data sets and using minimum distance classifier as a simple classifier indicate that proposed linear and non-linear intelligent FFNN-based feature combination can present more reliable and promising results.

By using such a feature combination method, there is no need to use more powerful and complex classifier anymore.

A quick review of engineering problems reveals importance of classification and its application in medicine, mechanical and electrical engineering, computer science, power systems and so on.

Some of its important applications include disease diagnosis using classification methods to diagnosis Thyroid (Temurtas (2009)), Parkinson BID4 ) and Alzheimers disease BID7 ); or fault detection in power systems such as BID6 ) which uses classification methods to detect winding fault in windmill generators; BID12 ) using neuro-fuzzy based classification method to detect faults in AC motor; and also fault detection in batch processes in chemical engineering BID22 ).

In all classification problems extracting useful knowledge and features from data such as image, signal, waveform and etcetera can lead to design efficient classification systems.

As extracted data and their features are not usually suitable for classification purpose, two major approaches can be substituted.

First approach considers all the classifiers and tries to select effective ones, even if their complexity and computational cost are increased.

Second approach focusing on the features, enhances the severability of data, and then uses improved features and data for classification.

Feature combination is one of the common actions used to enhance features.

In classic combination methods, deferent features vectors are lumped into a single long composite vector BID19 ).

In some modern techniques, in addition to combination of feature vectors, dimension of feature space is reduced.

Reduction process can be done by feature selection, transmission, and projection or mapping techniques, such as Linear Discriminate Analysis (LDA), Principle Component Analysis (PCA), Independent Component Analysis (ICA) and boosting BID19 ).

In more applications, feature combination is fulfilled to improve the efficiency of classification system such as BID3 ), that PCA and Modular PCA (MPCA) along Quad-Tree based hierarchically derived Longest Run (QTLR) features are used to recognize handwritten numerals as a statistical-topological features combination.

The other application of feature combination is used for English character recognition, here structure and statistical features combine then BP network is used as a classifier ).

Feature combination has many applications; however before using, some questions should be answered: which kind of combination methods is useful for studied application and available data set.

Is reduction of feature space dimension always useful?

Is linear feature combination method better than non-linear one?In this paper, using structure of Feed-Forward Neural Network (FFNN) along with Genetic Algorithm (GA) as a powerful optimization algorithm, Linear Intelligent Feature Combination (LIFC) and Non-Linear Intelligent Feature Combination (NLIFC) systems is introduced to present adaptive combination systems with the nature of data sets and their features.

In proposed method, original features are fed into semi-FFNN structure to map features into new feature space, and then outputs of this intelligent mapping structure are classified by minimum distance classifier via cross-validation technique.

In each generation, weights and biases of semi-FFNN structure are updated by GA and correct recognition rate (or error recognition rate) is evaluated.

In the rest of this paper, overview of minimum distance classifier, Feed-Forward Neural Network structure and Genetic Algorithm are described in sections2, 3and 4, respectively.

In section 5, proposed method and its mathematical consideration are presented.

Experimental results, comparison between proposed method and other feature combinations and classifiers using the same database are discussed in section 6.

Eventually, conclusion is presented in section 7.

Minimum Distance classifier (or 1-nearest neighbor classifier) is one of the simplest classification methods, which works based on measured distance between an unknown input data and available data in classes.

Distance is defined as an index of similarity, according this definition, the minimum distance means the maximum similarity.

Distance between two vectors can be calculated in various procedures, such as Euclidian distance, Normalized Euclidian distance, Mahalanobis distance, Manhattan distance and etcetera.

Euclidian distance is the most prevalent procedure that is presented in 1.

DISPLAYFORM0 Where D is the distance between two vectors X and Y .

||X ??? Y ||means second norm of Euclidian distance.

Notation n is dimension of X and Y whereX = (x 1 , x 2 , , x n ) and Y = (y 1 , y 2 , , y n ).

FIG0 shows the concept of a minimum distance classifier.

As it can be seen, distance between unknown input data and C2 is the minimum distance among all distances therefore this input data assigns to class C.

Artificial Neural Networks (ANNs) are designed based on a model of human brain and its neural cells.

Although human knowledge is much more limited than brain; its performance can be understood according to observation and physiology and anatomy information of brain BID13 ).

Prominent trait of ANN is its ability to learn complicated problems between input and output vectors.

In general, these networks are capable to model many non-linear functions.

This ability lets neural networks be used in practical problems such as comparative diagnoses and controlling nonlinear systems.

Nowadays, different topologies are proposed for implementing ANN in supervised, unsupervised and reinforcement applications.

Feed-forward is a dominant used topology in supervised learning procedure.

Feed-forward topology for an ANN is shown in FIG1 .

As it can be seen, information is fed into ANN via input layer which distribute just input information into the main body of ANN.

In this transmission the quantities of information are changed through multiplying by synapse weights of connection between input layer and next layer.

Applying activation functions in next layers, updated information arrive at output layer.

General equation is given by 2.

It is noticeable that in this structure information flow from input to output and there is not any feedback, also there is not any disconnection and jump connection between layers.

DISPLAYFORM0 Where, coefficients g and s are activation functions of N 2 and N 1s, respectively.

w1s are synapse weights between input layer and hidden layer, also w2s are synapse weights between hidden layer and output layer.

In evolution theory, particles of population evolve themselves to be more adaptable to their environment.

Therefore the particles that can do this better have more chance to survive.

These algorithms are stochastic optimization techniques.

In this kind of techniques, information of each generation is transferred to next generation by chromosome.

Each chromosome consists of gens and any gen illustrates an especial feature or behavior.

Genetic Algorithm (GA) is one of the most well known evolutionary algorithms.

In GA's process, first of all, initial population is created based on necessities of problem.

After that, objective function is evaluated.

In order to achieve the best solution, off springs are created from parents in reproduction step by crossover and mutation.

Consequently the best solution is obtained after determined iterations BID11 ).

As mentioned before, variant methods may be used to improve the ability of classification system.

In some cases, we interest in more complex and powerful classifier, although helpful, it often reduces decision making speed and increases computational cost.

The other way is using pre-processing on training data before changing the kind of classifier or its complexity.

Feature combination is one of the most common ways used to enhance the quality of features, so simple classifiers can discriminate them easily.

In the most feature combination methods, such as LDA, PCA, ICA, MPCA and etcetera the main strategy is to reduce the feature space dimension, whereas based on nature of data sets features sometimes dimension reduction is needed, combination of features in same dimension is enough sometimes, and also increase of feature space dimension may be useful sometimes.

The main idea of proposed method in this paper is to applying linear or non-linear intelligent features map in new solution space, in this method discriminative of data is increased.

In general, proposed method is illustrated as FIG2 and can be represented as follow: Let R be solution space; according to the mapping concepts, we have: DISPLAYFORM0 Where, Superscriptsn and m are dimensions of solution space (or feature dimensions) before and after mapping process respectively.

If n > m, then feature dimension is reduced from n-dimension to m-dimension by transfer function f .

If n = m, then there is not any change dimensionality and only transfer function is applied on features.

Feature dimension is also increased for n < m.

Equation 1 describes the only generality of issue, whereas in proposed method not only the feature space dimension is changed and transfer function is applied, but also features are combined in linear or non-linear format.

As shown in FIG3 X = x 1 , x 2 , , x n is an input data, Y = y 1 , y 2 , , y m is an output data and F is a transfer function which can be a typical Super polynomial like a feedforward neural network structure.

After feeding X into this structure, if m be a unit (m = 1), we would project all features into one axis (or one dimension), and so we have: DISPLAYFORM1 And if m be more than unit (m > 1), then we have: DISPLAYFORM2 . . .

DISPLAYFORM3 Where function g may be linear or non-linear activation (or transfer) function.

In this paper, as it can be seen in Fig. 5 (a) and Eq. 6 in the case of linear transfer function (like purelin) y s are only the weighted summation of primary features (x s ).

DISPLAYFORM4 Function g can be non-linear as shown in Eq. 7and Fig. 5 (b) .

This non-linear function is a kind of sigmoid transfer function which can be changed by coefficient ?? .

DISPLAYFORM5 Figure 5: Linear and non-linear used activation function in proposed method.

Now finding the optimum weights and biases for increasing separability of features that leads to increase efficiency of classification system is the heart of this paper.

GA is utilized as one of the most accurate and powerful optimization tool.

It is applied on features of each data after establishing the structure of mentioned intelligent mapping system and considering weights and biases with initial random value as shown in FIG4 .Then a very simple classifiers used that is minimum distance classifier and cross-validation technique -one-leave one-out (Bishop FORMULA1 ) and error recognition rate is calculated.

In this step, stop criteria is evaluated which is the least error recognition rate or the given number of generation for GA.

If neither of stop criterions is satisfied, GA updates weights and biases.

This process is done again and again until one of the stop criterions become satisfied.

In other word mapping system (intelligent combination system) is a fitness function of GA and weights, and biases are GAs chromosomes.

In order to evaluate the ability of proposed combination methods four classification tasks of UCI data sets are used BID2 ).

Useful information of studied data sets 1 is described as follow (see 1).

The Iris data contains 50 samples from three species, namely, Iris setosa, Iris versicolor, and Iris virginica.

Sepal length, sepal width, petal length and petal width are four features extracted from each species.

Wine: These data are the results of a chemical analysis of wines grown in the same region in Italy and they are derived from three different cultivars.

The analysis determines the quantities of 13 features extracted from each type of wine.

These features are Alcohol, Malic acid, Ash, Alcalinity of Ash, Magnesium, total phenols, flavanoids, non-flavanoid phenols, proanthocyanins, color intensity, hue OD289/OD315 of diluted wines, and proline.

Moreover, this dataset contains 178 samples categorized in three classes.

The Glass data set consists of 214 samples of nine features from every specie: building-windows-float-processed, building-windows-non-float-processed, vehicle-windows-floatprocessed, containers, tableware, and heal ware.

And extracted features are Refractive Index, Sodium, Magnesium, Aluminum, Silicon, Potassium, Calcium, Barium, and Iron.

Ionosphere: This radar data was collected by a system in Goose Bay, Labrador.

This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts.

This radar data are categorized in two groups; "Good" and "Bad". "Good" radar returns show evidence of some types of structure in the ionosphere.

And "Bad" returns dont do so, their signals pass through the ionosphere.

As mentioned before, studying the nature of data set in order to design efficient combination and classification systems may be so important.

Therefore all possible condition (namely: dimension reduction, dimension increase, only combination of features in same dimension, linear and nonlinear mapping) are considered.

For each data set, classification using cross-validation is applied ten times.

Classification parameters of GA are also considered similar for all data sets to present same condition, as shown in Table ( 2).Coefficient ?? is 0.2for all non-linear feature combinations.

Fig.7 shows typical convergence curves of error recognition rate for studied data sets.

Figure 7 : Typical convergence curves of error recognition rates for all studied data set of UCI using GA.

It is worth noting that; Correct recognition rate = 100 Error recognition rate.

Tables FORMULA2 and FORMULA3 present obtained results from all studied data sets in all mentioned condition.

In each condition, classification is done 10 times.

Minimum, maximum and average of correct classification rates are calculated to evaluate accuracy and reliability.

It should be mentioned that proposed method works based on error recognition rate, but we report correct recognition rate, here.

In order to clear this Tables, consider Iris which has 4 features or dimensions.

In first condition features are projected and combined into lower dimension (with 2 dimensions).

In second, only combination of features is fulfilled under intelligent combination function and in third one, features are projected and combined into higher dimension (with 8 dimensions).

In all conditions, classification is done using linear (L) and non-linear (NL) transfer function.

As it can be implied from Tables (3) and FORMULA3 , best combination form for Iris is non-linear mapping that dimension of feature space is reduced, although obtained results in different condition are approximately similar winner condition is more reliable and accurate.

The best classification rate for Wine is obtained by non-linear combination method while feature space dimension is increased.

It is completely different for Glass in order to achieve efficient classification system for Glass it is enough to combine features without any changing in feature space dimension.

Also, using non-linear combination feature while dimension is reduced can lead to best recognition rates for Ionosphere.

In order to compare the performance of proposed method with other combination methods, two common used combination methods, LDA and PCA, are considered in this section.

Both methods reduce dimension of feature space.

LDA reduce dimension of features to (C-1) dimensions which C is the number of classes.

PCA is also reduced feature dimension, but in PCA projected feature Table 3 : Obtained results for Iris and Wine for 10 times classification in 6 conditions.

Table 4 : Obtained results for Iris and Wine for 10 times classification in 6 conditions.

space dimension may be absolutely less than original feature space dimension.

TAB4 shows the mapping spaces for LDA and PCA.

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 In addition to LDA and PCA, obtained results have been compared with other reported classification rate that used same data sets in other literatures as shown in TAB5 .

Obtained results compared with other results, show the importance of work on data and features before using complex classifiers with high computational costs.

In all data sets, both proposed methods (LIFC and NLIFC) provide high quality features, so a simple classifier such as minimum distance classifier can discriminates classes and presents easily acceptable classification rate: for Iris correct recognition arte is increased from % 94.66 to % 100and for Wine this rate is increased from % 76.96 to % 99.43.

Also correct recognition rate reaches to % 86.91 for Glass and % 97.15 for Ionosphere.

In order to design more efficient classification system extracting useful knowledge and features from data set is so important and helpful.

In many cases, it is more reasonable to spend time and energy to analyze features instead of using more complex classifiers with high computational costs.

In this paper intelligent feature combination is proposed to enhance the quality of features and then minimum distance classifier is used as a simple classifier to obtain results.

Obtained results confirm that kind of combination method depends on nature of data set and its features.

For some datasets using non-linear mapping system while reducing dimension of the feature space is useful and sometimes using linear mapping system while increasing the dimension of the feature space leads to design the classification system more efficiently.

For Iris and Ionosphere using non-linear intelligent mapping system while reducing the dimension of feature space results correct recognition rates of %100 and % 97.15respectively.

Using non-linear intelligent mapping while increasing dimension of feature space leads to obtain correct recognition rate of % 99.43 for Wine.

It is so interesting that the best result for Glass obtains when features are combined by non-linear mapping without any change in dimension of feature space.8 REFERENCES

<|TLDR|>

@highlight

A method for enriching and combining features to improve classification accuracy