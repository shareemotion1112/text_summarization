In many applications, the training data for a machine learning task is partitioned across multiple nodes, and aggregating this data may be infeasible due to storage, communication, or privacy constraints.

In this work, we present Good-Enough Model Spaces (GEMS), a novel framework for learning a global satisficing (i.e. "good-enough") model within a few communication rounds by carefully combining the space of local nodes' satisficing models.

In experiments on benchmark and medical datasets, our approach outperforms other baseline aggregation techniques such as ensembling or model averaging, and performs comparably to the ideal non-distributed models.

There has been significant work in designing distributed optimization methods in response to challenges arising from a wide range of large-scale learning applications.

These methods typically aim to train a global model by performing numerous communication rounds between distributed nodes.

However, most approaches treat communication reduction as an objective, not a constraint, and seek to minimize the number of communication rounds while maintaining model performance.

Less explored is the inverse setting-where our communication budget is fixed and we aim to maximize accuracy while restricting communication to only a few rounds.

These few-shot model aggregation methods are ideal when any of the following conditions holds:• Limited network infrastructure: Distributed optimization methods typically require a connected network to support the collection of numerous learning updates.

Such a network can be difficult to set up and maintain, especially in settings where devices may represent different organizational entities (e.g., a network of different hospitals).• Privacy and data ephemerality: Privacy policies or regulations like GDPR may require nodes to periodically delete the raw local data.

Few-shot methods enable learning an aggregate model in ephemeral settings, where a node may lose access to its raw data.

Additionally, as fewer messages are sent between nodes, these methods have the potential to offer increased privacy benefits.• Extreme asynchronicity: Even in settings where privacy is not a concern, messages from distributed nodes may be unevenly spaced and sporadically communicated over days, weeks, or even months (e.g., in the case of remote sensor networks or satellites).

Few-shot methods drastically limit communication and thus reduce the wall-clock time required to learn an aggregate model.

Throughout this paper, we reference a simple motivating example.

Consider two hospitals, A and B, which each maintain private (unshareable) patient data pertinent to some disease.

As A and B are geographically distant, the patients they serve sometimes exhibit different symptoms.

Without sharing the raw training data, A and B would like to jointly learn a single model capable of generalizing to a wide range of patients.

The prevalent learning paradigm in this settingdistributed or federated optimization-dictates that A and B share iterative model updates (e.g., gradient information) over a network.

From a meta-learning or multitask perspective, we can view each hospital (node) as a separate learning task, where our goal is to learn a single aggregate model which performs well on each task.

However, these schemes often make similar assumptions on aggregating data and learning updates from different tasks.

As a promising alternative, we present good-enough model spaces (GEMS), a framework for learning an aggregate model over distributed nodes within a small number of communication rounds.

Intuitively, the key idea in GEMS is to take advantage of the fact that many possible hypotheses may yield 'good enough' performance for a learning task on local data, and that considering the intersection between these sets can allow us to compute a global model quickly and easily.

Our proposed approach has several advantages.

First, it is simple and interpretable in that each node only communicates its locally optimal model and a small amount of metadata corresponding to local performance.

Second, each node's message scales linearly in the local model size.

Finally, GEMS is modular, allowing the operator to tradeoff the aggregate model's size against its performance via a hyperparameter .We make the following contributions in this work.

First, we present a general formulation of the GEMS framework.

Second, we offer a method for calculating the good-enough space on each node as a R d ball.

We empirically validate GEMS on both standard benchmarks (MNIST and CIFAR-10) as well as a domain-specific health dataset.

We consider learning convex classifiers and neural networks in standard distributed setups as well as scenarios in which some small global held-out data may be used for fine-tuning.

We find that on average, GEMS increases the accuracy of local baselines by 10.1 points and comes within 43% of the (unachievable) global ideal.

With fine-tuning, GEMS increases the accuracy of local baselines by 41.3 points and comes within 86% of the global ideal.

Distributed Learning.

Current distributed and federated learning approaches typically rely on iterative optimization techniques to learn a global model, continually communicating updates between nodes until convergence is reached.

To improve the overall runtime, a key goal in most distributed learning methods is to minimize communication for some fixed model performance; to this end, numerous methods have been proposed for communication-efficient and asynchronous distributed optimization (e.g., Dekel et al., 2012; Recht et al., 2011; Dean et al., 2012; Li et al., 2014; Shamir et al., 2014; Richtárik & Takáč, 2016; Smith et al., 2018; McMahan et al., 2017) .

In this work, our goal is instead to maximize performance for a fixed communication budget (e.g., only one or possibly a few rounds of communication).One-shot/Few-shot Methods.

While simple one-shot distributed communication schemes, such as model averaging, have been explored in convex settings (Mcdonald et al., 2009; Zinkevich et al., 2010; Zhang et al., 2012; Shamir et al., 2014; Arjevani & Shamir, 2015) , guarantees typically rely on data being partitioned in an IID manner and over a small number of nodes relative to the total number of samples.

Averaging can also perform arbitrarily poorly in non-convex settings, particularly when the local models converge to differing local optima (Sun et al., 2017; McMahan et al., 2017) .

Other one-shot schemes leverage ensemble methods, where an ensemble is constructed from models trained on distinct partitions of the data (Chawla et al., 2004; Mcdonald et al., 2009; Sun et al., 2017) .

While these ensembles can often yield good performance in terms of accuracy, a concern is that the resulting ensemble size can become quite large.

In Section 4, we compare against these one-shot baselines empirically, and find in that GEMs can outperform both simple averaging and ensembles methods while requiring significantly fewer parameters.

Meta-learning and transfer learning.

The goals of metalearning and transfer learning are seemingly related, as these works aim to share knowledge from one learning process onto others.

However, in the case of transfer learning, methods are typically concerned with one-way transfer-i.e., optimizing the performance of a single target model, not jointly aggregate knowledge between multiple models.

In meta-learning, such joint optimization is performed, but similar to traditional distributed optimization methods, it is assumed that these models can be updated in an iterative fashion, with potentially numerous rounds of communication being performed throughout the training process.

Version Spaces.

In developing GEMS, we draw inspiration from work in version space learning, an approach for characterizing the set of logical hypotheses consistent with available data (Mitchell, 1978) .

Similar to (Balcan et al., 2012) , we observe that if each node communicates its version space to the central server, the server can return a consistent hypothesis in the intersection of all node version spaces.

However, (Mitchell, 1978; Balcan et al., 2012) assume that the hypotheses of interest are consistent with the observed data-i.e., they perfectly predict the correct outcomes.

Our approach significantly generalizes to explore imperfect, noisy hypotheses spaces as more commonly observed in practice.

As in traditional distributed learning, we assume a training set DISPLAYFORM0 ..} as the subset of training examples belonging to node k, such that DISPLAYFORM1 We assume that a single node (e.g., a central server) can aggregate updates communicated in the network.

Fixing a function class H, our goal is to learn an aggregate model h G ∈ H that approximates the performance of the optimal model h * ∈ H over S while limiting communication to one (or possibly a few) rounds of communication.

In developing a method for model aggregation, our intuition is that the aggregate model should be at least good-enough over each node's local data, i.e., it should achieve some minimum performance for the task at hand.

Thus, we can compute h G by having each node compute and communicate a set of locally good-enough models to a central server, which learns h G from the intersection of these sets.

DISPLAYFORM2 denote a model evaluation function, which determines whether a given model h is good-enough over a sample of data points {(x i , y i )} d ⊆ S. In this work, define "good-enough" in terms of the accuracy of h and a threshold : DISPLAYFORM3 Using these model evaluation functions, we formalize the proposed approach for model aggregation, GEMS, in Algorithm 1.

In GEMS, each node k = 1, ..., K computes the set of models DISPLAYFORM4 and sends it to the central node.

After collecting H 1 , ...H K , the central node selects h G from the intersection of the sets, ∩ i H i .

When granted access to a small sample of public data, the server can additionally use this auxiliary data further fine-tune the selected h ∈ ∩

i H i , an approach we discuss further below.

FIG3 visualizes this approach for a model class with only two weights (w 1 and w 2 ) and two learners ("red" and "blue").

The 'good-enough' model space, H k , for each learner is a set of regions over the weight space (the blue regions correspond to one learner and the red regions correspond to second learner).

The final aggregate model, h G , is selected from the area in which the spaces intersect.

For a fixed hypothesis class H, applying Algorithm 1 requires two components: (i) a mechanism for computing H k over every node, and (ii) a mechanism for identifying the aggregate model, h G ∈ ∩ k H k .

In this work, we present methods for two types of models: convex models and simple neural networks.

For convex models, we find that H k can be approximated as R d -ball in the parameter space, requiring only a single round of communication between nodes to learn h G .

For neural networks, we apply Algorithm 1 to each layer in a step-wise fashion, compute H k as a set of independent R d -balls corresponding to every neuron in the layer, and identify intersections between different neurons.

This requires one round of communication per layer (a few rounds for the entire network).We can compute these R d balls by fixing the center at the optimal local model on a device.

The radius for the ball is computed via binary search: at each iteration, the node samples a candidate hypothesis h and evaluates Q(h, S k ).

The goal is to identify that largest radius such that all models located in the R d ball are good-enough.

Algorithm 2 presents a simple method for constructing H k .

More details can be found in Appendix A (convex setting) and Appendix B (neural network setting).

DISPLAYFORM5 Node k computes good-enough model space, H k , according to (1) 4: end for DISPLAYFORM6 else 10:Set R upper = R

end if 12: end while 13: Return H k Fine-tuning.

In many contexts, a small sample of public data S public may be available to the central server.

This could correspond to a public research dataset, or devices which have waived their privacy right.

The coordinating server can fine-tune H G on S public by updating the weights for a small number of epochs.

We find that fine-tuning is particularly useful for improving the quality of the GEMS aggregate model, H G , compared to other baselines.

We now present the evaluation results for GEMS on three datasets: MNIST (LeCun et al., 1998) , CIFAR-10 (Krizhevsky & Hinton, 2009) , and HAM10000 (Tschandl et al., 2018) , a medical imaging dataset.

HAM10000 (HAM) consists of images of skin lesions, and our model is tasked with distinguishing between 7 types of lesions.

Full details can be found in Appendix C.1.

We focus on the performance of GEMS for neural networks, and discuss results for convex models in Appendix A. We partitioned data by label, such that all train/validation images corresponding to a particular label would be assigned to the same node.

We consider three baselines: 1) global, a model trained on data aggregated across all nodes, 2) local, the average performance of models trained locally on each node, and 3) naive average, a parameter-wise average of all local models.

All results are reported on the aggregated test set consisting of all test data across all nodes.

Fine-tuning consists of updating the last layer's weights of the GEMS model for 5 epochs over a random sample of 1000 images from the aggregated validation data.

We report the average accuracy (and standard deviation) of all results over 5 trials.

Neural network performance.

We evaluated the neural network variant of GEMS on simple two layer feedforward neural networks TAB0 ).

The precise network configuration and training details are outlined in Appendix C.4.

In the majority of cases, the untuned GEMS model outperforms the local/average baselines.

Moreover, fine-tuning has a significant impact, and tuned GEMS model 1) significantly outperforms every baseline, and 2) does not degrade as K increases.

In Appendix F, we demonstrate that GEMS is more parameter efficient than ensemble baselines, delivering better accuracy with fewer parameters.

Fine-tuning.

The results in TAB0 suggest that fine-tuning on a holdout set of samples S public has a significant effect on the GEMS model.

We evaluate the effect of fine-tuning as the number of public data samples (the size of the tuning set) changes.

For neural networks FIG1 ), finetuned GEMS consistently outperforms 1) the finetuned baselines, and 2) a 'raw' model trained directly on S public .

This suggest that the GEMS model is learning weights that are more amenable to fine-tuning, and are perhaps capturing better representations for the overall task.

Though this advantage diminishes as the tuning sample size increases, the advantage of GEMS is especially pronounced for smaller samples, and achieves remarkable improvements with just 100 images.

Intersection Analysis.

In certain cases, GEMS may not find an intersection between different nodes.

This occurs when the task is too complex for the model, or is set too high.

In practice, we notice that finding an intersection requires us to be conservative (e.g low values) when setting for each node.

We explain this by our choice to represent H k as an R d ball.

Though R d balls are easy to compute and intersect, they're fairly coarse approximations of the actual good-enough model space.

To illustrate node behavior at different settings of , we defer the reader to experiments performed in Appendix G.

In summary, we introduce GEMS, a framework for learning an aggregated model across different nodes within a few rounds of communication.

We validate one approach for constructing good-enough model spaces (as R d balls) on three datasets for both convex classifiers and simple feedforward networks.

Despite the simplicity of the proposed approach, we find that it outperforms a wide range of baselines for effective model aggregation TAB0

We provide a more detailed explanation of the GEMS algorithm for convex settings.

Consider the class of linear separators f w (·) parameterized by a weight vector w ∈ R d .

For each node k, we compute H k as R d -ball in the parameter space, represented as a tuple (c k ∈ R d , r k ∈ R) corresponding to the center and radius.

Formally, H k = {w ∈ R d |||c k − w|| 2 ≤ r k }.

Fixing as our minimum acceptable performance, we want to compute H k such that ∀w ∈ H k , Q(w, S k ) = 1.

Intuitively, every model contained within the d-ball should have an accuracy greater than or equal to .

DISPLAYFORM0 over node data S k , is fixed hyperparameter, and Q(·) is a minimum accuracy threshold defined according to Eq. 1.

R max and ∆ define the scope and stopping criteria for the binary search.

Intersection: Given K nodes with individual DISPLAYFORM1 ., K}. We pick a point in this intersection by solving: DISPLAYFORM2 which takes a minimum value of 0 when w ∈ ∩ i H i .

This w can be improved by fine-tuning on a limited sample of 'public data'.

We provide a more detailed explanation of the GEMS algorithm applied to neural networks.

First, we observe that the final layer of an MLP is a linear model.

Hence, we can apply the method above with no modification.

However, the input to this final layer is a set of stacked, non-linear transformations which extract feature from the data.

For these layers, the approach presented above faces two challenges:1.

Node specific features: When the distribution of data is non-i.i.d across nodes, different nodes may learn different feature extractors in lower layers.2.

Model Isomorphisms: MLPs are extremely sensitive to the weight initialization.

Two models trained on the same set of samples (with different initializations) may have equivalent behavior despite learning weights.

In particular, reordering a model's hidden neurons (within the same layer) does not alter the model's predictions, but corresponds to a different weight vector w.

In order to construct H k for hidden layers, we modify the approach presented in Appendix A, applying it to individual hidden units.

Formally, let the ordered set DISPLAYFORM0 Broadly, Q neuron returns 1 if the output of f w over z j−1 is within of z j l , and −1 otherwise.

We can now apply Algorithm 2 to each neuron.

Formally:1.

Each node k learns a locally optimal model m k , with optimal neuron weights w j l * , over all j, l.2.

Fix hidden layer j = 1.

Apply Algorithm 2 to each DISPLAYFORM1 , with Q(·) according to Eq 3 and predefined hyperparameter j .

Denote the R d ball constructed for neuron l as H k j,l .

to the central server which constructs the aggregate hidden layer f Gj,· such ∀i, k, ∃i : f Gj,i ∈ H k j,i .

This is achieved by greedily applying Eq 2 to tuples in the cartesian product H 1 j,· × ... × H K j,· .

Neurons for which no intersection exists are included in f Gj,· , thus trivially ensuring the condition above.4.

The server sends h Gj,· to each node, which insert h Gj,· at layer j in their local models and retrain the layers above j.5.

Increment j, and return to (2) if any hidden layers remain.

Step FORMULA10 is expensive for large L and K as |H DISPLAYFORM0 A simplifying assumption is that if H k j,i and H k j,l are 'far', then the likelihood of intersection is low.

Operationalizing this, we can perform k-means clustering over all neurons.

In step (3), we now only look for intersections between tuples of neurons in the same cluster.

Neurons for which no intersection exists are included in f Gj,· .

For notational clarity, we denote the number of clusters with which k-means is run as m , in order to distinguish it from device index k.

We describe preprocessing/featurization steps for our empirical results.

MNIST.

We used the standard MNIST dataset.

CIFAR-10.

We featurize CIFAR-10 (train, test, and validation sets) using a pretrained ImageNet VGG-16 model (Simonyan & Zisserman, 2014) from Keras.

All models are learned on these featurized images.

HAM10000.

The HAM dataset consists of 10015 images of skin lesions.

Lesions are classified as one of seven potential types: actinic keratoses and intraepithelial carcinoma (akiec), basal cell carcinoma (bcc), benign keratosis (bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions (vasc).

As FIG4 shows, the original original dataset is highly skewed, with almost 66% of images belonging to one class.

In order to balance the dataset, we augment each class by performing a series of random transformations (rotations, width shifts, height shifts, vertical flips, and horizontal flips) via Keras (Chollet et al., 2015) .

We sample 2000 images from each class.

We initially experimented with extracting ImageNet features (similar to our proceedure for CIFAR-10).

However, training a model on these extractions resulted in poor performance.

We constructed our own feature extractor, by training a simple convolutional network on 66% of the data, and trimming the final 2 dense layers.

This network contained 3 convolutional layers (32, 64, 128 filters with 3 × 3 kernels) interspersed with 2 × 2 MaxPool layers, and followed by a single hidden layer with 512 neurons.

Given K nodes, we partitioned each dataset in order to ensure that all images corresponding to the same class belonged to the same node.

TAB3 provides an explicit breakdown of the label partitions for each of the three datasets, across the different values of K we experimented with.

We divided each dataset into train, validation, and test splits.

All training occurs exclusively on the train split and all results are reported for performance on the test split.

We use the validation split to construct each node's

Our convex model consists of a simple logistic regression classifier.

We train with Adam, a learning rate of 0.001, and a batch size of 32.

We terminate training when training accuracy converges.

Our non-convex model consists of a simple two layer feedforward neural network.

For MNIST and HAM, we fix the hidden layer size to 50 neurons.

For CIFAR-10, we fix the hidden layer size to 100 neurons.

We apply dropout (Srivastava et al., 2014 ) with a rate of 0.5 to the hidden layer.

We train with Adam, a learning rate of 0.001, and a batch size of 32.

We terminate training when training accuracy converges.

We evaluate the convex variant of GEMS on logistic classifiers.

The results for all three datasets for a varying number nodes K is presented in TAB6 .

Fine-tuning consists of updating the weights of the GEMS model for 5 epochs over a random sample of 1000 images from the aggregated validation data.

Training details are provided in Appendix C.3In a convex setting, we find that GEMS frequently defaults to a weighted average of the parameters.

Hence, the GEMS results closely mirror naive averaging.

As the number of agents increases, both untuned GEMS and the baselines significantly decrease in performance.

However, tuned GEMS remains relatively consistent, and outperforms all other baselines.

We use = 0.70 for MNIST, = 0.40 for HAM, and = 0.20 for CIFAR-10.

E. Neural Network Results Table 4 presents the neural network results for MNIST.

We use = 0.7 for the final layer, and let j denote the deviation allowed for the hidden neurons (as defined in Eq 3).

For neural networks, GEMS provides a modular framework to tradeoff between the model size and performance, via hyperparameters m (the number of clusters created when identifying intersections) and j (the maximum output deviation allowed for hidden neurons).

Intuitively both parameters control the number of hidden neurons in the aggregate model h G .

Table 7 compares adjustments for j and m on CIFAR-10 for 5 nodes against an ensemble of local device models.

We observe that the GEMS performance correlates with the number of hidden neurons, and that GEMS outperforms the ensemble method at all settings (despite having fewer parameters).

For ease of clarity, we describe the model size in terms of the number of hidden neurons.

For ensembles, we sum the hidden neurons across all ensemble members.

All results are averaged over 5 trials, with standard deviations reported.

We notice that in order for GEMs to find an intersection, we have to set conservatively.

We illustrate this phenomenon in FIG8 .

We consider the convex MNIST case (K = 2), and do a grid search over different values of for each node.

We plot whether an intersection was identified, and the resulting accuracy at that setting TAB0 Model Aggregation via Good-Enough Model Spaces TAB0 Model Aggregation via Good-Enough Model Spaces

@highlight

We present Good-Enough Model Spaces (GEMS), a framework for learning an aggregate model over distributed nodes within a small number of communication rounds.