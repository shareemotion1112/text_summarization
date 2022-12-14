Although few-shot learning research has advanced rapidly with the help of meta-learning, its practical usefulness is still limited because most of the researches assumed that all meta-training and meta-testing examples came from a single domain.

We propose a simple but effective way for few-shot classification in which a task distribution spans multiple domains including previously unseen ones during meta-training.

The key idea is to build a pool of embedding models which have their own metric spaces and to learn to select the best one for a particular task through multi-domain meta-learning.

This simplifies task-specific adaptation over a complex task distribution as a simple selection problem rather than modifying the model with a number of parameters at meta-testing time.

Inspired by common multi-task learning techniques, we let all models in the pool share a base network and add a separate modulator to each model to refine the base network in its own way.

This architecture allows the pool to maintain representational diversity and each model to have domain-invariant representation as well.

Experiments show that our selection scheme outperforms other few-shot classification algorithms when target tasks could come from many different domains.

They also reveal that aggregating outputs from all constituent models is effective for tasks from unseen domains showing the effectiveness of our framework.

Few-shot learning in the perspective of meta-learning aims to train models which can quickly solve novel tasks or adapt to new environments with limited number of examples.

In case of few-shot classification, models are usually evaluated on a held-out dataset which does not have any common class with the training dataset.

In the real world, however, we often face harder problems in which novel tasks arise arbitrarily from many different domains even including previously unseen ones.

In this study, we propose a more practical few-shot classification algorithm to generalize across domains beyond the common assumption, i.e., meta-training and meta-testing within a single domain.

Our approach to cover a complex multi-domain task distribution is to construct a pool of multiple models and learn to select the best one given a novel task through meta-training over various domains.

This recasts task-specific adaption across domains as a simple selection problem, which could be much easier than manipulating high-dimensional parameters or representations of a single model to adapt to a novel task.

Furthermore, we enforce all models to share some of the parameters and train per-model modulators with model-specific parameters on top of that.

By doing so, each model could keep important domain-invariant features while the model pool has representational diversity as a whole without a significant increase of model parameters.

We train and test our algorithms on various image classification datasets with different characteristics.

Experimental results show that the proposed selection scheme outperforms other state-of-theart algorithms in few-shot classification tasks from many different domains without being given any knowledge of the domain which the task belongs to.

We also show that even few-shot classification tasks from previously unseen domains, i.e., domains which have never appeared during meta-training, can be done successfully by averaging outputs of all models.

We follow the common setting of few-shot classification in the meta-learning community (Vinyals et al., 2016) .

For a N -way, K-shot classification task, an episode which consists of a support set S = {(x i represent examples and their correct labels respectively and T is the number of query examples.

Once a model has been trained with respect to a number of random episodes at meta-training time, it is expected to predict a correct label for an unlabeled query given only a few labeled examples (i.e., support set) even if all these came from classes which have never appeared during meta-training.

Based on this setting, we try to build a domain-agnostic meta-learner beyond the common metalearning assumptions, i.e., meta-training within a single domain and meta-testing within the same domain.

We perform meta-training over multiple diverse domains, which we call source domains

where M is the number of source domains, expecting to obtain a domaingeneralized meta-learner.

Since we presume that one particular dataset defines its own domain, we realize this idea by training this meta-learner on various tasks from many different datasets.

In our study, the trained meta-learner is meta-tested on a target domain D T for two types of crossdomain few-shot classification tasks.

One is a task which is required to classify from held-out classes of multiple source domains (i.e., D T ??? {D S1 , D S2 , ?? ?? ?? , D S M }) without knowing from which dataset each task is sampled.

This could be used to evaluate whether the meta-learner is capable of adapting to a complex task distribution across multiple domains.

We also tackle a task sampled from previously unseen datasets during the meta-training (i.e., D Si ??? D T = ??? for all i), which requires to generalize over out-of-distribution tasks in domain-level.

Basically, we perform metric-based meta-learning to learn a good metric space in which the support and query examples from the same class are located closely over various domains.

While recent meta-learning methods have been proposed to train a single model commonly applicable to various potential tasks and to learn to adjust the model to a particular task for further improvement (Rusu et al., 2019; Oreshkin et al., 2018; Ye et al., 2018; Triantafillou et al., 2019) , we train a pool of multiple embedding models each of which defines a different metric space and a meta-model to select the best model from them given a novel task.

This makes task-specific adaptation easier and more effective to learn because our approach is required to solve only a simple classification problem, i.e., choose one of all pre-trained models, instead of learning to manipulate high-dimensional model components, such as model parameters or activations, directly to adapt to novel tasks from various domains.

Rather than training each model separately, we take an approach that all models share a considerable amount of parameters and they are differentiated by adding per-model modulators as done usually in multi-task learning (Ruder, 2017) .

The rationale behind this is to let our model pool capture good domain-invariant features by the shared parameters as well as have diversity, which is desirable to represent the complex cross-domain task distribution, without a significant increase of the number of parameters.

To realize this idea, we first build a base network f E (??; ??) shared among all models.

One large virtual dataset is constructed by aggregating all source datasets.

The base network is trained on this dataset following typical supervised learning procedure (Figure 1(a) ).

In the next step, we build one model per source domain by adding a per-model modulator with a parameter set ?? i on top of the frozen base network.

We then train each modulator on one dataset D Si by performing metric-based metalearning in the same way as the Prototypical Networks (ProtoNet) (Snell et al., 2017) (Figure 1(b) ).

Finally, we have a pool of embedding models which are ready for non-parametric classification in the same way as ProtoNet.

As shown in Figure 2 , we add the modulator components to the base network in a per-layer basis following the idea proposed in (Rebuffi et al., 2018) .

This way has turned out to be more effective than the conventional approach, i.e., adding a few extra layers for each model, for domain-specific representation.

Moreover, this allows each modulated model to have the same computational cost at inference time as the base network's because all modulator components can be fused into existing convolution 3??3 operations.

We try two modulator architectures, convolution 1??1 and channel-wise transform (i.e., per-channel trainable scale and bias).

The former shows slightly better performance whereas the latter uses much less parameters only incurring negligible memory overhead to the pool.

More details of the architecture including the number of parameters can be found in Appendix B.

After the construction of the pool, we build a meta-model to predict the most suitable model from all constituent models in the pool for a given task as the final step of our training.

By training this model over a number of episodes sampled randomly from all available source datasets, we expect this ability to be generalized to novel tasks from various domains including unseen ones during meta-training.

As depicted in Figure 1 (c), this meta-model parameterized by ??, which we call a model selection network, is trained in order to map a task representation z task for a particular task to an index of the best model in the model pool.

The task representation is obtained by passing all examples in the support set of the task through the base network and taking a mean of all resulting embedded vectors to form a fixed-length summary of the given task.

During meta-training, the index of the best model, which is the ground truth label for training the selection network, is generated by measuring the classification accuracy values of all models in the pool given the query set and picking an index of one which has the highest accuracy.

In our setup, the task-specific adaptation is reduced to a (M +1)-way classification problem when we have M +1 embedding models including the base network learned from M available source datasets.

Learning this classifier could be far simpler than learning to adapt model parameters (Rusu et al., 2019 ), embedded vectors (Ye et al., 2018 or per-layer activations (Oreshkin et al., 2018) to a particular task because their dimensions are usually much larger than that of our selection network outputs, i.e., the number of the pre-trained models.

The overall training procedure for constructing the pool and training the selection network is summarized in Algorithm 1 in Appendix C. One way of the inference is to predict a class with the best single model chosen by the selection network f S (??; ??) for a given support set (Figure 3(a) ).

Following the method proposed in ProtoNet as shown in Equation 1, a class prediction?? to a query example x q is made by finding a class whose prototype c y is the closest one to an embedded vector of the given query example.

Specifically, the prototype for the class y is defined as a mean of embedded vectors of all examples in the support set belonging to a class y and squared Euclidean distances d y i to these prototypes are compared between the classes.??

Another way to benefit from the model pool is to combine outputs from all available embedding models for inference (Figure 3(b) ).

We take the simplest approach to average outputs of all models in the level of class prediction probability.

As described in Equation 2, we collect output probabilities p(y | x q ; i) based on the relative distances to class prototypes d y i for a given task.

Then, our final prediction?? would be a class to maximize a mean of these probabilities from all M + 1 models.

As the last step, we adopt test-time 'further adaptation' proposed in (Chen et al., 2019) , which turned out to make additional performance improvement in most cases.

Both experimental results with and without the further adaptation are presented in Section 3 and Appendix D with its implementation details in Appendix C.

3.1 SETUP

We use eight image classification datasets, denoted as Aircraft, CIFAR100, DTD, GTSRB, ImageNet12, Omniglot, SVHN, UCF101 and Flowers, introduced in the Visual Decathlon Challenge (Rebuffi et al., 2017) for evaluation.

These are considered as eight different domains in our experiments.

All datasets are resplit for the few-shot classification, i.e., no overlap of classes between meta-training and meta-testing.

More details about the datatsets can be found in Appendix A.

We denote our methods using the model picked by the selection network as DoS (Domaingeneralized method by Selection) and DoS-Ch, which modulate the base network with convolution 1??1 and channel-wise transform respectively.

We also explore our averaging-based methods, DoA (Domain-generalized method by Averaging) and DoA-Ch, which generate an output by averaging class prediction probabilities of all constituent models modulated by the proposed two types of modulators.

Our algorithms are compared with Fine-tune, Simple-Avg, ProtoNet (Snell et al., 2017) , FEAT (Ye et al., 2018) and ProtoMAML (Triantafillou et al., 2019) .

Fine-tune is a baseline method, which adds a linear classifier on top of the pre-trained base network and fine-tune it with the support set examples for 100 epochs at meta-testing.

In Simple-Avg, we train an embedding model independently on each source domain without sharing any parameters and perform inference by averaging class prediction probabilities of all these models.

FEAT and ProtoMAML are the state-of-the-art algorithms focusing on single domain and cross-domain setups respectively.

TADAM (Oreshkin et al., 2018) was also tested but excluded from the results because its training did not converge in our setup.

All these algorithms are tested by our own implementations.

We pre-train all models of the compared algorithms with our base network since the pre-training has shown additional performance gain empirically.

Then, we meta-train these models in an algorithmspecific way on randomly generated episodes except the Fine-tune, the non-episodic baseline.

At each episode, a target domain is chosen randomly from source domains, then a target task is sampled from that domain.

The trained models are tested for 5-way classification tasks with 1-shot and 5-shot configurations.

The test results are averaged over 600 test episodes with 10 queries per class.

Other details about the training are explained in Appendix C.

We evaluate the above-mentioned algorithms in a multi-domain test setup constructed using all available datasets.

Specifically, we meta-train a model for each algorithm on all available eight datasets.

Then, we meta-test the trained model for various tasks sampled from these eight datasets without knowing which dataset the task comes from.

Figure 4 depicts test accuracy for each target dataset and the average accuracy over the eight datasets.

Our selection methods, DoS and DoS-Ch, outperform other few-shot classification methods in most cases.

Two state-of-the art algorithms, FEAT and ProtoMAML, do not seem as effective as ours under this complex task distribution across domains.

ProtoMAML shows comparable or better results in some cases, but much inferior results in other cases.

ProtoNet seems relatively stable, but does not produce better results in any case.

FEAT works worse than these two algorithms in most cases.

Although our averaging methods, DoA and DoA-Ch, seem competitive in many cases, they are outperformed by the selection methods always.

This implies that the learned selection network is working properly, which is highly likely to select the model with the modulator trained on the same domain as the given task even without any information about the domain at testing time.

Another Table A6 .) implication is that the best single model might be better than the averaging approach if a model from the same domain exists.

It is also worth noting that DoS-Ch is quite competitive despite much less number of parameters than DoS.

We also report the results on unseen domains in Figure 5 .

Given a target dataset for test, we train all models on other seven datasets.

Therefore, we end up with eight different models for each algorithm because there exist eight different combinations of the source datasets.

Our methods still outperform other algorithms in this more challenging setting.

This reveals that our approach can be generalized to novel domains as well.

Differently from the seen domain cases, our averaging methods, DoA and DoA-Ch, perform better than all other algorithms including our selection methods.

It seems to make sense since the averaging could induce natural synergy between beneficial models even if we do not know which models are good for a given task.

However, our averaging methods significantly outperform Simple-Avg, which implies that our way of the pool construction to encourage keeping domain-invariant features is another key factor to the high performance of our averaging methods.

We conduct experiments with varying number of source datasets.

Following the common real-world situation, we add from the largest dataset to the smallest one to our sources for meta-training.

Tables  1 and 2 show the experimental results with 2, 4, and 6 source datasets on seen and unseen domains respectively.

Our selection and averaging methods outperform others consistently on seen and unseen domains similarly to the previous results.

Apart from comparing between the algorithms, it is commonly observed over all algorithms that the added source often harms the performance.

For example, the CIFAR100 tasks tend to work poorer as the number of source datasets increases in the seen domain case.

This means that we should pay more attention to avoiding negative transfer between heterogeneous domains.

S: source datasets, T: target dataset A:

Aircraft, C: CIFAR100, G: GTSRB, I: ImageNet12, O: Omniglot, U: UCF101, F: Flowers S: source datasets, T: target dataset A:

Aircraft, C: CIFAR100, G: GTSRB, I: ImageNet12, O: Omniglot, U: UCF101, F: Flowers 4 RELATED WORKS Few-shot learning has been studied actively as an effective means for a better understanding of human learning or as a practical learning method only requiring a small number of training examples (Lake et al., 2015; Li et al., 2006) .

Meta-learning is one of the most popular techniques to solve the few-shot learning problems, which include learning a task-invariant metric space (Snell et al., 2017; Vinyals et al., 2016) , learning to optimize (Andrychowicz et al., 2016; Ravi & Larochelle, 2017) or learning good weight initialization (Finn et al., 2017; Nichol et al., 2018) for forthcoming tasks.

Follow-up studies showed that the metric-based meta-learning could be improved further by learning to modulate that metric space in a task-specific manner (Gidaris & Komodakis, 2018; Oreshkin et al., 2018; Qiao et al., 2018; Ye et al., 2018) .

Similarly, it has been reported that the task-common initial parameters could be refined for a given task producing task-specific initialization (Rusu et al., 2019; Vuorio et al., 2018; Yao et al., 2019) .

Recent few-shot learning studies have tried to tackle challenging problems under more realistic assumptions.

Some studies dealt with few-shot learning under domain shift between training and testing (Kang & Feng, 2018; Wang & Hebert, 2016) .

A more realistic evaluation method was proposed for few-shot learning to overcome limitations of the current popular benchmarks including the lack of domain divergence (Triantafillou et al., 2019) .

One study performed an extensive and fair comparative analysis of well-known few-shot learning methods (Chen et al., 2019) .

Our network architecture is inspired by the parameter sharing strategies for multi-task learning (Ruder, 2017) and multi-domain learning with domain-specific adaptation (Rebuffi et al., 2018) because they have been known to lead to efficient parameterization and positive knowledge transfer between heterogeneous entities.

Similar to our approach, a few suggestions combined multiple models to benefit from their diversity (Dvornik et al., 2019; Park et al., 2019) .

Our work also has something in common with the mixture-of-experts approach (Shazeer et al., 2017) in a sense that a part of a large scale model would be executed conditionally benefiting from a large amount of the learned knowledge at low computational cost.

Our research is also related to domain adaptation or generalization (Ganin et al., 2016; Motiian et al., 2017) .

However, most of the researches about these topics assume tasks with the same classes in both training and testing whereas our methods do not impose such limitations.

Interestingly, some studies showed that the episodic training which is commonly adopted in many few-shot learning techniques, was also useful for domain generalization (Li et al., 2018; .

We proposed a new few-shot classification method which is capable of dealing with many different domains including unseen domains.

The core idea was to build a pool of embedding models, each of which was diversified by its own modulator while sharing most of parameters with others, and to learn to select the best model for a target task through cross-domain meta-learning.

The simplification of the task-specific adaptation as a small classification problem made our selection-based algorithm easy to learn, which in turn helped the learned model to work more effectively for multidomain few-shot classification.

The architecture with one shared model and disparate modulators encouraged our pool to maintain domain-invariant knowledge as well as cross-domain diversity.

It helped our algorithms to generalize to heterogeneous domains including unseen ones even when we used one best model solely or all models collectively.

We believe that there is still a large room for improvement in this challenging task.

It would be one promising extension to find the optimal way to build the pool without the constraint on the number of models (i.e., one model per dataset) so that it can work even with a single source dataset with large diversity.

Soft selection or weighted averaging can be also thought as one of future research directions because a single model or uniform averaging is less likely to be optimal.

We can also consider a more scalable extension to allow continual expansion of the pool only by training a modulator for an incoming source domain without re-training all existing models in the pool.

Although the number of parameters does not increase much by virtue of the parameter sharing between models, the computational cost in the averaging-based methods needs to be improved over the current linear increase with the number of models.

In our experiments, we use the Visual Decathlon Challenge dataset (Rebuffi et al., 2017) which consists of ten datasets for image classification listed below.

??? FGVC-Aircraft Benchmark (Aircraft, A) (Maji et al., 2013) ??? CIFAR100 (CIFAR100, C) (Krizhevsky, 2009) ??? Daimler Mono Pedestrian Classification Benchmark (DMPCB) (Munder & Gavrila, 2006) ??? Describable Texture Dataset (DTD, D) (Cimpoi et al., 2014) ??? German Traffic Sign Recognition Benchmark (GTSRB, G) (Stallkamp et al., 2012) ??? ImageNet ILSVRC12 (ImageNet12, I) (Russakovsky et al., 2015) ??? Omniglot (Omniglot, O) (Lake et al., 2015) ??? Street View House Numbers (SVHN) (Netzer et al., 2011) ??? UCF101 (UCF101, U) (Soomro et al., 2012) ??? Flowers102 (Flowers, F) (Nilsback & Zisserman, 2008) The categories and the number of images of each dataset as well as the image sizes are significantly different.

All images have been resized isotropically to 72 ?? 72 pixels so that each image from various domains has the same size.

Daimler Mono Pedestrian Classification has only 2 classes, pedestrian and non-pedestrian.

We excluded it from our experiments as we are considering 5-way classification tasks.

SVHN was also excluded since SVHN has only 10 digit classes from 0 to 9, which were too few to split for metatraining and meta-testing.

To use the remaining eight datasets for multi-domain few-shot classification, we divide the examples into roughly 70% training, 15% validation, and 15% testing classes.

For ILSVRC12, we follow the split of Triantafillou et al. (Triantafillou et al., 2019) to adopt class hierarchy, and we use random class splits for other datasets.

The number of classes at each split is shown in Table A1 .

We only use train and validation sets of the Visual Decathlon because the labels of the test set is not publicly available.

B ARCHITECTURES Figure 2 shows the architecture of the embedding network f E (??; ??, ?? i ), which processes an input image and produces a 512-dimensional embedding vector.

The embedding network is based on the Table A2 : The comparison of the number of parameters for convolution 1 ?? 1 and channel-wise transform modulators.

This is the case when the number of source domains is 8.

ResNet-18 architecture (He et al., 2016) , which consists of one convolutional layer with 64 7 ?? 7 filters followed by 4 macro blocks, each having 64-128-256-512 3 ?? 3 filters.

Figure 2 (a) and Figure  2 (b) depict how the base network is modulated by the convolution 1 ?? 1 modulator and the channelwise transform modulator, respectively.

These modulators are placed within each residual block of the macro blocks, same as the previous works in ( Rebuffi et al., 2018) and (Perez et al., 2018) .

The number of parameters for two modulators are shown in Table A2 .

The values on the first row are the number of modulator parameters that are additionally applied to the embedding network.

Note that the channel-wise transform modulator has much fewer parameters than the convolution 1??1 modulator.

In particular, the channel-wise transform modulator has negligible number of parameters compared to that of the base network, i.e., ResNet-18.

For each embedding model, the convolution 1 ?? 1 modulator has about 10% of the number of parameters that the base network has whereas the channel-wise transform modulator requires only less than 1%.

The selection network f S (??; ??) is a two-layered MLP (multi-layer perceptron) network, which receives an embedding vector produced by the embedding network as an input and performs the best model index prediction.

Two layers are a linear layer of 512??128 and a linear layer of 128??(M +1), where M is the number of source domains.

Algorithm 1 describes the overall training procedure to construct the model pool and the selection network.

Although we trained three components in a sequential manner, joint training of these components seems to make sense also.

For the fair comparison with Fine-tune method, we also apply algorithm-specific refinement at metatesting time, inspired by 'further adaptation' in (Chen et al., 2019) , to all other algorithms including ours.

A linear classifier is placed on top of the embedding network of the ProtoNet, the self-attention module of the FEAT or the modulated embedding network of our models.

During meta-testing, other parameters are fixed and the classifier is fine-tuned using the support examples for 100 iterations per episode.

In case of FEAT, the classifier is trained for 100 epochs per query example not per episode because FEAT modulates a representation space for each query.

We also adjust the number of adaptation of the ProtoMAML to 100 for the better task-adaptation as done in (Chen et al., 2019) .

The hyperparameters including the learning rate are selected by grid search based on the validation accuracy.

For FEAT and ProtoMAML, Adam optimizer is used for training and the learning rate and weight decay are set to be 0.0001.

Other models are also trained using Adam optimizer with the learning rate 0.001 but without any regularization including the weight decay.

In Simple-Avg, each embedding model is trained on a separate source dataset following the method proposed in the Prototypical Networks (Snell et al., 2017) .

We can obtain a little higher performance with this approach than training each model following the standard supervised learning procedure.

All reported test accuracy values are averaged over 600 test episodes with 10 queries per class.

The overall training procedure Input: Training data from

Step 1: Build a base network 1: Build one large classification dataset (x agg , y agg ) by aggregating all classes from D S .

2: Learn ?? by optimizing f E (x; ??, ?? 0 ) for the aggregated dataset (?? 0 : no modulation).

Step 2: Add modulators through intra-domain episodic training 1: while not converged do 2:

Sample one domain D Si from D S , then sample one episode (S, Q) from D Si .

Learn ?? i by optimizing f E (x; ??, ?? i ) for (S, Q) while keeping ?? fixed.

4: end while

Step 3: Build a selection network through cross-domain episodic training 1: while not converged do 2:

Sample one domain D Si from D S , then sample one episode (S, Q) from D Si .

Get a task representation z task by averaging embedding vectors of S from the base network.

Measure accuracies of M + 1 available embedding models {f E (x; ??, ?? i )} M i=0 for (S, Q).

Set the best model index y sel to the index of the model with the highest accuracy.

Learn ?? by training f S (z task ; ??) so as to predict y sel for (S, Q).

7: end while

We present the experimental results when we do not apply the further adaptation scheme introduced in (Chen et al., 2019) .

Specifically, ProtoNet, FEAT, and our models are tested without additional linear classifiers f c (??; ??).

The number of parameter update steps in ProtoMAML is reduced to 3, which is not enough to have the models fine-tuned.

Tables A3 and A4 show the results tested on seen and unseen domains, respectively.

We can see that accuracy drops in almost all cases compared to corresponding cases with further adaptation whose results are in Section 3 in the main text, but our models generally do better than other methods in any experimental settings.

The measured numbers show that the individual models of our DoA perform better than those in the Simple-Avg, which explains the higher performance of the proposed method partly.

Additionally, we can observe that major contributors (i.e., the models with higher accuracy) tend to change every episode in our DoA whereas only two models seem to play dominant roles regardless of the given episode.

This implies that our method for constructing the model pool provides the averaging model with more beneficial diversity.

Equation (3) shows the loss (loss sel ) used to train the selection network f S (??; ??).

Here, acc m is the classification accuracy of the model with the modulator parameterized by ?? m in the pool.

The accuracy is measured for query examples in a given episode by making a prediction in the same way with the Protypical Networks (Snell et al., 2017) , where the class whose prototype is the closest to the embedding vector of a given query example is picked as the final prediction.

@highlight

We address multi-domain few-shot classification by building multiple models to represent this complex task distribution in a collective way and simplifying task-specific adaptation as a selection problem from these pre-trained models.

@highlight

This paper tackles few-shot classification with many different domains by building a pool of embedding models to capture domain-invariant and domain-specific features without a significant increase in the number of parameters.