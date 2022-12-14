Many problems with large-scale labeled training data have been impressively solved by deep learning.

However, Unseen Class Categorization (UCC) with minimal information provided about target classes is the most commonly encountered setting in industry, which remains a challenging research problem in machine learning.

Previous approaches to UCC either fail to generate a powerful discriminative feature extractor or fail to learn a flexible classifier that can be easily adapted to unseen classes.

In this paper, we propose to address these issues through network reparameterization, \textit{i.e.}, reparametrizing the learnable weights of a network as a function of other variables, by which we decouple the feature extraction part and the classification part of a deep classification model to suit the special setting of UCC, securing both strong discriminability and excellent adaptability.

Extensive experiments for UCC on several widely-used benchmark datasets in the settings of zero-shot and few-shot learning demonstrate that, our method with network reparameterization achieves state-of-the-art performance.

The rich and accessible labeled data has fueled the revolutionary successes of deep learning in various tasks, e.g., visual recognition BID7 ), object detection BID20 ), machine translation BID1 ), etc.

However, requiring numerous annotated data severely limits the applicability of deep learning algorithms to Unseen Class Categorization (UCC) for which we only have access to a limited amount of information, which is frequently encountered in industrial applications.

Recently, an increasing number of approaches have been proposed to solve UCC with the help of either attribute descriptions (zero-shot learning (ZSL)) BID9 ; BID30 ) or one/a few labeled samples for each class (few-shot learning (FSL)) BID22 ; BID29 ).Previous approaches to UCC mainly have the following characteristics and limitations: (i) To obtain powerful discriminative feature representation, they often train a deep classification model employing state-of-the-art multi-class classification techniques.

However, such models are hard to be adapted to new classes with limited supervision information due to the high volume of model parameters and the gradual updating scheme. (ii) To ensure the consistency of training and test settings and adaptability to new classes, previous methods often train a deep model in an episode fashion BID26 ), sometimes along with some specially designed meta-learning updating rules BID4 ).

With episode-based training, the model acquires adaptability to new tasks after many training episodes using the knowledge it grasps during the training.

However, the episode-based training strategy severely limits the model's capability of extracting discriminative features, because it does not fully exploit the diversity and variance of all classes within the training dataset.

The trained model treats the classes in each episode as new classes and attempts to separate them.

Therefore, it does not have memory of the competing information of these classes against all the other ones in the whole dataset beyond the current episode.

Due to the neglect of this global (dataset-wise rather than episode-wise) discriminative information, the feature extraction capability of the model is suppressed, thus limiting the UCC performance.

To address these issues, we propose to secure both powerful discriminability of feature extraction and strong adaptability of model classification through network reparameterization, i.e., reparametrizing the learnable weights of a network as a function of other variables.

We decouple the feature extraction module and the classification module of a deep classification model, learn the former as a standard multi-class classification task to obtain a discriminative feature extractor, and learn the latter employing a light deep neural network that generates generic classification weights for unseen classes given limited exemplar information.

We train the classification weight generator by following the episode-based training scheme to secure the adaptability.

Our method can be flexibly applied to both ZSL and FSL, where the exemplar information about unseen classes are provided in the form of either the semantic attributes or one/a few labeled samples.

Extensive experiments show that our proposed method achieves state-of-the-art performance on widely-used benchmark datasets for both tasks.

With regard to the form of the exemplar information provided about unseen classes, UCC can be classified as zero-shot learning and few-shot learning.

ZSL requires recognizing unseen classes based on their semantic descriptions.

It is approached by finding an embedding space where visual samples and semantic descriptions of a class are interacted so that the semantic description of an unseen class can be queried by its visual samples.

Since the embedding space is often of high dimension, finding the best match of a given vector among many candidates shall inevitably encounter the hubness problem BID17 ), i.e., some candidates will be biased to be the best matches for many of the queries.

Depending on the chosen embedding space, the severeness of this problem varies.

Some approaches select the semantic space as the embedding space and project visual features to the semantic space BID10 ; BID5 .

Projecting the visual features into a often much lower-dimensional semantic space shrinks the variance of the projected data points and thus aggravates the hubness problem.

Alternatively, some methods project both visual and semantic features into a common intermediate space BID0 ; BID23 ; BID31 ).

However, due to lacking training samples from unseen classes, these methods are prone to classify test samples into seen classes BID21 ) (for the generalized ZSL setting, seen classes are included when testing).

Recently, BID30 proposed to choose the visual space as the embedding space and learned a mapping from the semantic space to visual space.

Benefiting from the abundant data diversity in visual space, this method can mitigate the hubness problem at some extent.

However, the limitation of this method is that it strives only to learn a mapping from semantic space to visual space such that the visual samples of a class coincide with the associated semantic description; it however neglects the separation information among visual features of different classes.

Our method avoids this problem.

We formulate bridging the semantic space and the visual space as a visual feature classification problem conditioned on the semantic features.

We learn a deep neural network that generates classification weights for the visual features when fed with the corresponding semantic features.

By nature of a classification problem, both intra-class compactness (visual features of the same classes are assigned with the same label) and inter-class separability (visual features of different classes are assigned with different labels) are exploited, hence resulting in a better mapping.

FSL aims to recognize unseen classes when provided with one/a few labeled samples of these classes.

A number of methods address it from the perspective of deep metric learning by learning deep embedding models that output discriminative feature for any given images BID19 ; BID26 BID22 ; BID24 ; BID23 ).

The difference lies in the loss functions used.

More common approaches are based on meta-learning, also called learning to learn, which is to learn an algorithm (meta-learner) that outputs a model (the learner) that can be applied on a new task when given some information (meta-data) about the new task.

Following this line, approaches such as META-LSTM BID18 ), MAML BID4 ), Meta-SGD ), DEML+Meta-SGD ), Meta-Learn LSTM BID18 ), Meta-Networks BID13 ), and REPTILE BID14 ) aim to optimize the meta-learned classifiers to be easily fine-tuned on new few-shot tasks using the small-scale support set provided.

The common limitation of the above methods is that they adopt the episode-based training scheme to secure adaptability to new classes, which however compromises the capability of discriminative feature extraction due to the forgetting of global (dataset-wise) competing information of among classes beyond individual episodes.

Perhaps closest to our approach, BID6 proposed the DFSVL algorithm which approaches FSL also in virtue of classification weight generation.

The major limitation of DFSVL is that it obtains classification weights for unseen classes simply as a mixture of feature embeddings of support images of novel classes and attended pretrained weights of base (seen) classes, which is too weak to bridge feature embeddings and classification weights.

Besides, it cannot bridge information across different domains (due to dimension inconsistency) so that is not applicable for ZSL.

We instead learn a network to generate classification weights directly from feature embeddings of support images; it is more powerful and flexible to solve both ZSL and FSL within the same framework.

We focus on Unseen Class Categorization (UCC), which is to recognize objects of unseen classes given only minimal information (a few labeled samples or the attributes) about the classes.

Formally, suppose we have three sets of data DISPLAYFORM0

Our main contribution in this paper is the proposed framework that can address both ZSL and FSL with minimal changes.

FIG0 diagrams our framework.

Instead of jointly learning the feature extraction network weights and classification weights, which results in a heavy model that is hard to be adjusted for novel classes with limited supervision information, we reparametrize the learnable weights of a classification model as the combination of learnable parameters of a feature extraction model and a weight generation model.

In other words, we decouple the feature extraction network f ?? and the classification weight W of a standard classification network.

We train f ?? as a standard multiclass classification task and learn another network g ?? to generate the classification weight W. Since f ?? is trained as a standard multi-class classification task to distinguish all classes within the training set, it is supposed to be able to generate more discriminative feature representations for images of unseen classes than that generated by a model trained in episode-based fashion where the model is train to distinguish several classes within mini-batches.

Meanwhile, we train g ?? in episode-based fashion by constantly sampling new classes and minimizing the classification loss (cross entropy loss on top of Softmax outputs) using the weights generated by g ?? .

After training, whenever some new classes come, along with supporting information in the form of either attribute vectors (ZLS) or a few-labeled samples (FSL), g ?? is supposed to be able to generate generic classification weights that can effectively classify query images that belong to these new classes.

Thanks to this network reparameterization strategy, we are able to get a powerful and flexible UCC model.

We adopt the cosine similarity based cross entropy loss to train the weight generator g ?? .

Traditional multi-layer neural networks use dot product between the output vector of previous layer and the incoming weight vector as the input to activation function.

BID12 recently showed that replacing the dot product with cosine similarity can bound and reduce the variance of the neurons and thus result in models of better generalization.

BID6 further showed that using the cosine similarity instead of dot product for calculating classification score in the last fullyconnected layer of deep neural network brings benefit for classification, with some minor revisions.

We adopt this technique to train our weight generator g ?? .

The classification score of a sample (e x , y) is calculated as DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 where s is a learnable scalar controlling the peakiness of the probability distribution generated by the softmax operator BID6 ), w j is the classification weight for class j generated by neural network g ?? taking supporting information of the class as input, x is the input image, a j is the attribute vector for class j for ZSL, x i,j is the i-th input image of class j for FSL, j = 1, ..., N f , and N f is the number of shots for FSL.In a typical UCC task T , the loss function is calculated as DISPLAYFORM4 where ?? is a hyper-parameter weighting the l 2 -norm regularization of the learnable parameters of neural network g ?? .

For ZSL, we are provided with semantic class attributes S = A t ??? A u as the assistance for UCC.

The basic assumption for existing ZSL algorithms is that the visual-attribute relationship learned from seen classes in a certain embedding space is class-invariant and can be applied to unseen classes.

With this assumption, existing methods either project visual features to semantic space or reversely project semantic features to visual space, or alternatively project both visual and semantic features to an intermediate space.

In any case, the coincidence of visual and semantic features of a class is utilized to learn the visual-attribute relationship.

BID30 recently showed that it is advantageous to select the visual space as the embedding space because the abundance of data diversity in the visual space can significantly mitigate the so-called "hubness" problem.

Their objective function is as follows: DISPLAYFORM0 where f ?? is a feature extraction model which outputs a representation vector f ?? (x i ) using image x i as input.

h ?? is a mapping function which projects attribute vector a yi of class y i to the embedding space where f ?? (x i ) lies.

Through minimizing the least square embedding loss, the visual-attribute relationship can be established.

With this relationship, in the testing stage, the attributes A u of unseen classes are mapped to the visual feature embedding space in which the visual feature of an images of any unseen class can find the best class attribute through nearest neighbor searching.

One can observe that this method learns the visual-attribute relationship by only utilizing the coincidence of the visual samples of a class with the associated semantic description.

It however neglects to explore the inter-class separation of different classes, which shall be crucial to further avoid the hubness problem.

To remedy this, we reformulate the learning of visual-attribute relationship from a regression problem to a visual feature classification problem.

We directly learn a network g ?? that outputs the classification weights for classifying visual features and use the cross-entropy loss on top of Softmax outputs to guide learning g ?? .

Through this reformulation, both intra-class compactness and inter-class separability are elegantly exploited for learning the visual-attribute relationship: DISPLAYFORM1 and DISPLAYFORM2 3.

Calculate loss according to Eq. 5 4.

Update g ?? through back-propagation. end while Visual features of the same classes should be assigned with the same label (compactness), while visual features of different classes are assigned with different labels (separability).We follow the network reparameterization scheme by decoupling the feature extraction module f ?? and the classification weight module which is generated by g ?? .

The feature extraction module f ?? is trained as a standard multi-class classification task to enable us to obtain a discriminative feature representation for any given image.

To learn g ?? , we adopt the episode based training scheme by continuously exposing g ?? with new (randomly sampled) ZSL tasks so as to secure good performance when new real tasks arrive in the testing stage.

More specifically, we keep randomly sampling from D t = {X t , Y t } and A t ZSL tasks and feeding them to the network.

Each task consists of M z classes and the associated M z attribute vectors.

For each class, we randomly sample N z images.

With a batch of M z N z images B v and M z attribute vectors B a , we train g ?? by minimizing the loss function defined in Eq. 5.

In the testing stage, given attributes of unseen classes A u , or S = A t ??? A u for all (seen and unseen) classes as in generalized ZSL setting, we generate the corresponding classification weights using g ?? .

The generated classification weights, integrated with the feature extraction network f ?? serve to classify images of unseen classes.

Algorithm 1 outlines the main steps of our method for ZSL.

For FSL, one/a few labeled samples D s = {X s , Y s } for each unseen class are provided to help recognize objects of these classes.

Our novel categorization framework can be easily extended from ZSL to FSL, simply by replacing the semantic attribute vectors with feature embedding vectors as the input to the classification weight generation network g ?? .

To train g ?? , we keep randomly sampling FSL tasks from D t = {X t , Y t }, each of which consists of a support set and a query set.

Images in the both sets are from the same classes.

The support set consists of M f classes and N f images for each class.

With the feature embeddings B e of the M f N f images as input, g ?? generates the classification weights for the M f classes, which are then used to classify the feature embeddings of images from the query set.

Note that if N f > 1, i.e., each class has multiple support samples, we average the embeddings of all images belonging to the same class and feed the averaged embedding to g ?? .

Similar to ZSL, we learn the resulting model by optimizing the loss function defined in Eq. 5.

Algorithm 2 outlines the main steps for FSL.One of the most distinct aspects of our method from the existing ones is that we decouple the feature extraction module and the classifier module of the deep classification model, and train each module on the most beneficial tasks.

We train the feature extraction module as a standard multi-class classification task.

This is motivated by the observation that a simple classifier (e.g., nearest neighbor), when taking as input features obtained by a powerful extractor, can outperform some sophisticated FSL models that use weaker feature extraction models.

For example, as shown in Fiugre 2, using nearest neighbor (NN) as the classifier, we can achieve better one-shot classification accuracy than a recent FSL algorithm PROTO NET BID22 ), when using features extracted by ResNet18 BID7 The reason for this surprising result is that the episode-based training scheme of existing FSL methods inherently suppresses obtaining a powerful feature extractor:

In each episode, the model is fed with a new FSL task that is assumed to have no relationship with the previous ones.

The model is trained to separate well the several classes within the task.

However, since all training tasks are sampled from the training dataset, one class shall appear in many tasks.

The inter-class separation across the whole dataset is neglected by existing FSL methods.

Therefore, there is a dilemma for existing FSL algorithms: They need to be trained in an episodebased fashion to ensure flexibility, but which in return compromises feature discriminability.

To avoid this awkward situation, our proposed method decoupling the network and training different components in different ways ensures powerful discriminability and strong adaptability.

We evaluate our framework for both zero-shot learning and few-shot learning tasks.

Datasets and evaluation settings.

We employ the most widely-used zero-shot classification datasets for performance evaluation, namely, AwA1 (Lampert et al. FORMULA1 ), AwA2 BID29 ), CUB BID27 ), SUN BID16 ) and aPY BID3 ).

The statistics of the datasets are shown in Table 1 .

We follow the GBU setting proposed in BID29 ) and evaluate both the conventional ZSL setting and the generalized ZSL (GZSL) setting.

In the conventional ZSL, test samples are restricted to the unseen classes, while in the GZSL, they may come from either seen classes or unseen classes.

Implementation details.

Following BID29 ), we adopt ResNet101 as our feature extraction model f ?? which results in a 2048-dimension vector for each input image.

For the weight generation model g ?? , we utilize two FC+ReLU layers to map semantic vectors to visual classification weights.

The dimension of the intermediate hidden layer are 1600 for all the five datasets.

We train g ?? with Adam optimizer and a learning rate 10 ???5 for all datasets by 1,000,000 randomly sample ZSL tasks.

Each task consists of 32 randomly sampled classes, 4 samples for each class, i.e., M z = 32 and N z = 4.

The hyper-parameters ?? is chosen as 10 ???4 , 10 ???3 , 10 ???3 , 10 ???5 and 10 ???4 for AwA1, AwA2, CUB, SUN and aPY, respectively.

Our model is implemented with PyTorch.

Experimental results.

TAB3 shows the experimental results.

For the conventional ZSL setting, our method reaches the best for three out of the five datasets, while being very close to the best for one of the left two.

Remarkably, our method consistently outperforms DEM BID30 ) for all the five datasets, which substantiates the benefit of our method of taking consideration of interclass separability when learning the mapping from semantic space to visual space.

For GZSL setting where seen classes are also included to be the candidates, our method significantly outperforms all competing methods, reaching performance gains over the second best even about 30% in the AWA1 dataset.

We analyze the reason for our dramatic advantage is that our method considers inter-class separation during the training stage so that the resultant classification weights for the seen classes possess good separation property after training.

When they are concatenated with the classification weights generated from semantic descriptions of unseen classes in the testing stage, they shall be

AwA2 CUB aPY SUN ZSL GZSL ZSL GZSL ZSL GZSL ZSL GZSL ZSL GZSL DAP (Lampert et al. FORMULA1 44.1 0.0 46.1 0.0 40.0 1.7 33.8 4.8 39.9 4.2 CONSE BID15 45.6 0.4 44.5 0.5 34.3 1.6 26.9 0.0 38.8 6.8 SSE BID31 60.1 7.0 61.0 8.1 43.9 8.5 34.0 0.2 51.5 2.1 DEVISE BID5 54.2 13.4 59.7 17.1 52.0 23.8 39.8 4.9 56.5 16.9 SJE BID0 65.6 11.3 61.9 8.0 53.9 23.5 32.9 3.7 53.7 14.7 LATEM BID28 55.1 7.3 55.8 11.5 49.3 15.2 35.2 0.1 55.3 14.7 ESZSL BID21 ) 58.2 6.6 58.6 5.9 53.9 12.6 38.3 2.4 54.5 11 ALE BID0 ) 59.9 16.8 62.5 14.0 54.9 23.7 39.7 4.6 58.1 21.8 SYNC BID2 54.0 8.9 46.6 10.0 55.6 11.5 23.9 7.4 56.3 7.9 SAE BID9 53.0 1.8 54.1 1.1 33.3 7.8 8.3 0.4 40.3 8.8 DEM BID30 68.4 32.8 67.1 30.5 51.7 19.6 35.0 11.1 61.9 20.5 RELATION NET BID23 quite discriminative to discern that the incoming images do not belong to their classes.

From the perspective of hubness problem, since the classification weights for seen class have good separation property, the weight vectors are less likely to be clustered in the embedding space, so that the risk is reduced that some candidates are selected as the nearest neighbors for many query images.

Datasets and evaluation settings.

We evaluate few-shot classification on two widely-used datasets, Mini-ImageNet BID26 ) and CUB BID27 ).

The Mini-ImageNet dataset has 60,000 images from 100 classes, 600 images for each class.

We follow previous methods and use the splits in BID18 for evaluation, i.e., 64, 16, 20 classes as training, validation, and testing sets, respectively.

The CUB dataset is a fine-grained dataset of totally 11,788 images from 200 categories of birds.

As the split in BID18 , we use 100, 50, 50 classes for training, validation, and testing, respectively.

For both datasets, we resize images to 224??224 to meet the requirement of our adopted feature extraction network.

Following the previous methods, we evaluate both 5-way 1-shot and 5-way 5-shot classification tasks where each task instance involves classifying test images from 5 sampled classes with 1 (1-shot) or 5 (5-shot) randomly sampled images for each class as the support set.

In order to reduce variance we repeat the evaluation task 600 times and report the mean of the accuracy with a 95% confidence interval.

Implementation details.

We use ResNet18 as our feature extraction model f ?? which results in a 512-dimension vector for each input image after average pooling.

We train f ?? on the two experimental datasets by following the standard classification learning pipeline: We use Adam optimizer with an initial learning rate 10 ???3 which decays to the half every 10 epochs.

The model is trained with 100 epochs.

As for g ?? , we use two FC+ReLU layers, same as in ZSL.

The dimension of the intermediate hidden layer is 512 for both datasets.

We train g ?? using Adam optimizer with a learning rate 10 ???5 and set the hyper-parameters ?? = 10 ???5 for both datasets.

The model is trained with 60000 randomly sampled FSL tasks, each of which consist of 5 classes, with 1 or 5 samples as the support samples and another 15 as the query samples.

Experimental results.

TAB4 shows the results of the proposed method and the most recent ones.

From the table, we can get some interesting observations.

First, the baseline method "ResNet18 + NN" beats most competing FSL algorithms where various sophisticated strategies are used.

Meanwhile, the accuracy of feeding the classifier of PROTO NET with features obtained by ResNet18 ("ResNet18 feat.

+ PROTO NET classifier") is much higher than that obtained by training PROTO NET end to end with ResNet18 as the base model ("ResNet18 + PROTO NET").

These results support our analysis that the episode-based training scheme adopted by existing FSL approaches suppresses the discriminability of the feature extraction model.

Second, compared with the baseline methods "ResNet18 feat.

+ NN" and "ResNet18 feat.

+ PROTO NET classifier", which use the same feature representations as our method, we get obvious improvements.

This substantiates the benefit of the proposed weight generation strategy for FSL.

Third, compared with the existing methods, our method reaches the best in the both datasets for both 1-shot and 5-shot evaluation settings, often by large margins.

This shows the great advantage of our method for handling the FSL problem.

As we can see above, our method dramatically outperforms existing methods for the GZSL setting.

The advantage is much more significant than that for the ZSL setting.

We have analyzed the reason is that the classification weights generated from the attributes of seen classes show good separation property so that the hubness problem is not as severe as that for other methods.

The hubness problem refers that in ZSL, some candidate points are prone to be the nearest neighbors of many query points when the dimension is high.

So, if the candidate points are more evenly distributed in the space, the less severe of the hubness problem should be.

To validate this, we use t-SNE BID25 ) to visualize the classification weight vectors generated from all 200 class semantic vectors in the CUB dataset.

As a comparison, we do the same thing for DEM BID30 ) which also learns mapping from semantic space to visual space.

The result is shown in FIG4 .

We can observe that the points are more evenly distributed for our method than that for DEM.

This further validates the benefit of our method in avoiding the hubness problem.

In this paper, we propose a flexible framework for unseen class categorization with limited information provided about these classes.

We secure two key factors, a powerful feature extractor and a flexible classifier, through network reparameterization.

We decouple the feature extraction module and the classification module of a deep model for UCC.

The feature extraction module is learned in a standard multi-class classification framework and the classification weight vector is generated by a network from exemplar information of the unseen classes.

We train the classification weight generator in an episode-by-episode fashion to enable it flexibility for new tasks.

Applying our framework for zero-shot learning (ZSL), we achieve much better results especially for the generalized ZSL setting than the state-of-the-art owing to our incorporation of inter-class separation information for learning the mapping from semantic space to visual space.

For few-shot learning (FSL), we also achieve remarkable performance gains relative to existing methods due to the flexible scheme that make it possible a powerful feature extraction model and a flexible weight generation model.

@highlight

A unified frame for both few-shot learning and zero-shot learning based on network reparameterization