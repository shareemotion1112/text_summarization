Loss functions play a crucial role in deep metric learning thus a variety of them have been proposed.

Some supervise the learning process by pairwise or tripletwise similarity constraints while others take the advantage of structured similarity information among multiple data points.

In this work, we approach deep metric learning from a novel perspective.

We propose instance cross entropy (ICE) which measures the difference between an estimated instance-level matching distribution and its ground-truth one.

ICE has three main appealing properties.

Firstly, similar to categorical cross entropy (CCE), ICE has clear probabilistic interpretation and exploits structured semantic similarity information for learning supervision.

Secondly, ICE is scalable to infinite training data as it learns on mini-batches iteratively and is independent of the training set size.

Thirdly, motivated by our relative weight analysis, seamless sample reweighting is incorporated.

It rescales samples’ gradients to control the differentiation degree over training examples instead of truncating them by sample mining.

In addition to its simplicity and intuitiveness, extensive experiments on three real-world benchmarks demonstrate the superiority of ICE.

Deep metric learning (DML) aims to learn a non-linear embedding function (a.k.a.

distance metric) such that the semantic similarities over samples are well captured in the feature space (Tadmor et al., 2016; Sohn, 2016) .

Due to its fundamental function of learning discriminative representations, DML has diverse applications, such as image retrieval (Song et al., 2016) , clustering (Song et al., 2017) , verification (Schroff et al., 2015) , few-shot learning (Vinyals et al., 2016) and zero-shot learning (Bucher et al., 2016) .

A key to DML is to design an effective and efficient loss function for supervising the learning process, thus significant efforts have been made (Chopra et al., 2005; Schroff et al., 2015; Sohn, 2016; Song et al., 2016; Law et al., 2017; Wu et al., 2017) .

Some loss functions learn the embedding function from pairwise or triplet-wise relationship constraints (Chopra et al., 2005; Schroff et al., 2015; Tadmor et al., 2016) .

However, they are known to not only suffer from an increasing number of non-informative samples during training, but also incur considering only several instances per loss computation.

Therefore, informative sample mining strategies are proposed (Schroff et al., 2015; Wu et al., 2017; Wang et al., 2019b) .

Recently, several methods consider semantic relations among multiple examples to exploit their similarity structure (Sohn, 2016; Song et al., 2016; Law et al., 2017) .

Consequently, these structured losses achieve better performance than pairwise and triple-wise approaches.

In this paper, we tackle the DML problem from a novel perspective.

Specifically, we propose a novel loss function inspired by CCE.

CCE is well-known in classification problems owing to the fact that it has an intuitive probabilistic interpretation and achieves great performance, e.g., ImageNet classification (Russakovsky et al., 2015) .

However, since CCE learns a decision function which predicts the class label of an input, it learns class-level centres for reference (Zhang et al., 2018; Wang et al., 2017a) .

Therefore, CCE is not scalable to infinite classes and cannot generalise well when it is directly applied to DML (Law et al., 2017) .

With scalability and structured information in mind, we introduce instance cross entropy (ICE) for DML.

It learns an embedding function by minimising the cross entropy between a predicted instance-level matching distribution and its corresponding ground-truth.

In comparison with CCE, given a query, CCE aims to maximise its matching probability with the class-level context vector (weight vector) of its ground-truth class, whereas ICE targets at maximising its matching probability with it similar instances.

As ICE does not learn class-level context vectors, it is scalable to infinite training classes, which is an intrinsic demand of DML.

Similar to (Sohn, 2016; Song et al., 2016; Law et al., 2017; Goldberger et al., 2005; Wu et al., 2018) , ICE is a structured loss as it also considers all other instances in the mini-batch of a given query.

We illustrate ICE with comparison to other structured losses in Figure 1 .

A common challenge of instance-based losses is that many training examples become trivial as model improves.

Therefore, we integrate seamless sample reweighting into ICE, which functions similarly with various sample mining schemes (Sohn, 2016; Schroff et al., 2015; Shi et al., 2016; Wu et al., 2017) .

Existing mining methods require either separate time-consuming process, e.g., class mining (Sohn, 2016) , or distance thresholds for data pruning (Schroff et al., 2015; Shi et al., 2016; Wu et al., 2017) .

Instead, our reweighting scheme works without explicit data truncation and mining.

It is motivated by the relative weight analysis between two examples.

The current common practice of DML is to learn an angular embedding space by projecting all features to a unit hypersphere surface (Song et al., 2017; Law et al., 2017; MovshovitzAttias et al., 2017) .

We identify the challenge that without sample mining, informative training examples cannot be differentiated and emphasised properly because the relative weight between two samples is strictly bounded.

We address it by sample reweighting, which rescales samples' gradient to control the differentiation degree among them.

Finally, for intraclass compactness and interclass separability, most methods (Schroff et al., 2015; Song et al., 2016; Tadmor et al., 2016; Wu et al., 2017) use distance thresholds to decrease intraclass variances and increase interclass distances.

In contrast, we achieve the target from a perspective of instance-level matching probability.

Without any distance margin constraint, ICE makes no assumptions about the boundaries between different classes.

Therefore, ICE is easier to apply in applications where we have no prior knowledge about intraclass variances.

Our contributions are summarised: (1) We approach DML from a novel perspective by taking in the key idea of matching probability in CCE.

We introduce ICE, which is scalable to an infinite number of training classes and exploits structured information for learning supervision.

(2) A seamless sample reweighting scheme is derived for ICE to address the challenge of learning an embedding subspace by projecting all features to a unit hypersphere surface.

(3) We show the superiority of ICE by comparing with state-of-the-art methods on three real-world datasets.

Heated-up, NormFace, TADAM, DRPR, Prototypical Networks, Proxy-NCA.

These methods calculate the similarities between a query and class centres (a.k.a.

proxies or prototypes) instead of other instances (Zhang et al., 2018; Wang et al., 2017a; Oreshkin et al., 2018; Law et al., 2019; Snell et al., 2017; Movshovitz-Attias et al., 2017) .

In Heated-up and NormFace, the class centres are learned parameters of a fully connected layer, which is similar to Center Loss .

While in TADAM, DRPR, and Prototypical Networks, a class centre is the mean over all embeddings of a class.

By comparing a sample with other examples other than class centres, more informative instances can contribute more in ICE.

NCA (Goldberger et al., 2005) , S-NCA (Wu et al., 2018) .

NCA learns similarity relationships between instances.

Since original NCA learns the whole training data and its time complexity is quadratically proportional to the scale of training data, S-NCA is proposed recently with linear time complexity with respect to the training data size.

Instead, ICE is scalable to infinite training data by iterative learning on randomly sampled small-scale instances matching tasks.

S-NCA and NCA share the same learning objective.

However, they treat the event of all similar instance being correctly recognised as a whole by a sum accumulator.

Instead, we maximise the probability of every similar sample being correctly identified individually.

Therefore, ICE's optimisation task is harder, leading to better generalisation.

N-pair-mc (Sohn, 2016) .

The aim of N-pair-mc is to identify one positive example from N − 1 negative examples of N − 1 classes (one negative example per class).

In other words, only one (a) A query versus learned parametric class centroids.

All T classes in the training set are considered.

Prior work: CCE, Heated-up (Zhang et al., 2018) , NormFace (Wang et al., 2017a) .

(b) A query versus non-parametric class means.

Only classes in the mini-batch are considered.

Representative work: TADAM (Oreshkin et al., 2018) , DRPR (Law et al., 2019) , Prototypical Networks (Snell et al., 2017) .

(c) N-pair-mc (Sohn, 2016) : A query versus one instance per class.

A mini-batch has to be 2 examples per class rigidly.

Only one instance per negative class is randomly sampled out of 2.

(d) NCA (Goldberger et al., 2005) and S-NCA (Wu et al., 2018) : A query versus the rest instances.

Figure 1: Our ICE and related losses.

The first row shows prior work of a query versus class centres/means while the second row displays the work of a query versus instances.

Note that the cross entropy computation and interpretation are different in different losses.

For a mini-batch, we show two classes, i.e., circle and rectangle, with 3 examples per class except N-pair-mc which requires 2 samples per class.

The icons are at the right bottom.

GT means ground-truth matching distribution.

When illustrating the losses of a query versus instances in (c), (d) and (e), we index those instances with numbers for clarity, except the query.

positive and one negative instance per class are considered per loss computation by simulating CCE exactly.

Instead, ICE exploits all negative examples to benefit from richer information.

When constructing mini-batches, N-pair-mc requires expensive offline class mining and samples 2 images per class.

According to (Sohn, 2016 ) N-pair-mc is superior to NCA.

Hyperbolic (Nickel & Kiela, 2018) .

It aims to preserve the similarity structures among instances as well.

However, it learns a hyperbolic embedding space where the distance depends only on norm of embeddings.

Instead, we learn an angular space where the similarity depends only on the angle between embeddings.

Besides, Hyperbolic requires a separate sampling of semantic subtrees when the dataset is large.

Mining informative examples or emphasising on them are popular strategies in DML: 1) Mining non-trivial samples during training is crucial for faster convergence and better performance.

Therefore, sample mining is widely studied in the literature.

In pairwise or triplet-wise approaches (Schroff et al., 2015; Wu et al., 2017; Huang et al., 2016; , data pairs with higher losses are emphasized during gradient backpropagation.

As for structured losses, Lifted Struct (Song et al., 2016 ) also focuses on harder examples.

Furthermore, (Sohn, 2016) and (Suh et al., 2019) propose to mine hard negative classes to construct informative input mini-batches.

Proxy-NCA (Movshovitz-Attias et al., 2017) addresses the sampling problem by learning class proxies.

2) Assigning higher weights to informative examples is another effective scheme (Wang et al., 2019a; c) .

Beyond, there are some other novel perspectives to address sample mining or weighting, e.g., hardness-aware examples generation (Zheng et al., 2019) and divide-and-conquer of the embedding space (Sanakoyeu et al., 2019) .

Our proposed ICE has a similarity scaling factor which helps to emphasise more on informative examples.

Moreover, as described in (Schroff et al., 2015) , very hard negative pairs are likely to be outliers and it is safer to mine semi-hard ones.

In ICE, the similarity scaling factor is flexible in that it controls the emphasis degree on harder samples.

Therefore, a proper similarity scaling factor can help mine informative examples and alleviate the disturbance of outliers simultaneously.

What makes ours different is that we do not heuristically design the mining or weighting scheme.

Instead, it is built-in and we simply scale it as demonstrated in Section 3.4.

We remark that Prototypical Networks, Matching Networks (Vinyals et al., 2016) and NCA are also scalable and do not require distance thresholds.

Therefore, they are illustrated and differentiated in Figure 1 .

Matching Networks are designed specifically for one-shot learning.

Similarly, (Triantafillou et al., 2017) design mAP-SSVM and mAP-DLM for few-shot learning, which directly optimises the retrieval performance mAP when multiple positives exist.

FastAP (Cakir et al., 2019) is similar to (Triantafillou et al., 2017) and optimises the ranked-based average precision.

Instead, ICE processes one positive at a time.

Beyond, the setting of few-shot learning is different from deep metric learning: Each mini-batch is a complete subtask and contains a support set as training data and a query set as validation data in few-shot learning.

Few-shot learning applies episodic training in practice.

Remarkably, TADAM formulates instances versus class centres and also has a metric scaling parameter for adjusting the impact of different class centres.

Contrastively, ICE adjusts the influence of other instances.

Furthermore, ours is not exactly distance metric scaling since we simply apply naive cosine similarity as the distance metric at the testing stage.

That is why we interpret it as a weighting scheme during training.

is an input mini-batch, where x i ∈ R h×w×3 and y i ∈ {1, ..., C} represent i-th image and the corresponding label, respectively; {x

is a set of N c images belonging to c-th class, ∀c, N c ≥ 2.

The number of classes C is generally much smaller than the total number of classes T in the training set (C T ).

Note that T is allowed to be extremely large in DML.

Given a sufficient number of different mini-batches, our goal is to learn an embedding function f that captures the semantic similarities among samples in the feature space.

We represent deep embeddings of X as {{f

.

Given a query, 'positives' and 'negatives' refer to samples of the same class and different classes, respectively.

CCE is widely used in a variety of tasks, especially classification problems.

As demonstrated in (Liu et al., 2016) , a deep classifier consists of two joint components: deep feature learning and linear classifier learning.

The feature learning module is a transformation (i.e., embedding function f ) composed of convolutional and non-linear activation layers.

The classifier learning module has one neural layer, which learns T class-level context vectors such that any image has the highest compatibility (logit) with its ground-truth class context vector:

where

is the learned parameters of the classifier.

During training, the goal is to maximise the joint probability of all instances being correctly classified.

The identical form is to minimise the negative log-likelihood, i.e., L CCE (X; f, W).

Therefore, the learning objective of CCE is:

In contrast to CCE, ICE is a loss for measuring instance matching quality (lower ICE means higher quality) and does not need class-level context vectors.

We remark that an anchor may have multiple positives, which are isolated in separate matching distributions.

There is a matching distribution for every anchor-positive pair versus their negatives as displayed in Figure 1e .

Let f c a be a random query, we compute its similarities with the remaining points using dot product.

We define the probability of the given anchor x c a matching one of its positives x c i (i = a) as follows:

where f

We remark: (1) Dot product measures the similarity between two vectors; (2) Eq. (3) represents the probability of a query matching a positive while Eq. (1) is the probability of a query matching its ground-truth class.

To maximise p(x (Kullback & Leibler, 1951) between the predicted and ground-truth distributions, which is equivalent to minimise their cross entropy.

Since the ground-truth distribution is one-hot encoded, the cross-entropy is − log p(x c i |x c a ).

To be more general, for the given anchor x c a , there may exist multiple matching points when N c > 2, i.e., |{x c i } i =a | = N c − 1 > 1.

In this case, we predict one matching distribution per positive point.

Our goal is to maximise the joint probability of all positive instances being correctly identified, i.e.,

.

A case of two positives matching a given query is described in Figure 1e .

In terms of mini-batch, each image in X serves as the anchor iteratively and we aim to maximise the joint probability of all queries {{p x c a } Nc a=1 } C c=1 .

Equivalently, we can achieve this by minimising the sum of all negative log-likelihoods.

Therefore, our proposed ICE on X is as follows:

Following the common practice in existing DML methods, we apply L 2 -normalisation to feature embeddings before the inner product.

Therefore, the inner product denotes the cosine similarity.

The similarity between two feature vectors is determined by their norms and the angle between them.

Without L 2 normalisation, the feature norm can be very large, making the model training unstable and difficult.

With L 2 normalisation, all features are projected to a unit hypersphere surface.

Consequently, the semantic similarity score is merely determined by the direction of learned representations.

Therefore, L 2 normalisation can be regarded as a regulariser during training 1 .

Note that the principle is quite different from recent hyperspherical learning methods (Liu et al., 2017a; a; a) .

They enforce the learned weight parameters to a unit hypersphere surface and diversify their angles.

In contrast, feature normalisation is output regularisation and invariant to the parametrisation of the underlying neural network (Pereyra et al., 2017) .

In summary, our learning objective is:

The feature L 2 -normalisation layer is implemented according to Wang et al. (2017a) .

It is a differentiable layer and can be easily inserted at the output of a neural net.

Intrinsic sample weighting.

We find that ICE emphasises more on harder samples from the perspective of gradient magnitude.

We demonstrate this by deriving the partial derivatives of L ICE (X; f ) with respect to positive and negative examples.

Given the query x c a , the partial derivative of its any positive instance is derived by the chain rule:

Since ||f Similarly, the partial derivative of its any negative sample is: .

Clearly, the harder negative samples own higher matching probabilities and weights.

Relative weight analysis.

In general, the relative weight (Tabachnick et al., 2007) is more notable as the exact weight will be rescaled during training, e.g., linear post-processing by multiplying the learning rate.

Therefore, we analyse the relative weight between two positive points of the same anchor (i = k = a):

Similarly, the relative weight between two negative points of the same anchor (o = c, l = c) is:

Note that the positive relative weight in Eq. (9) is only decided by f Non-linear scaling for controlling the relative weight.

Inspired by (Hinton et al., 2015) , we introduce a scaling parameter to modify the absolute weight non-linearly:

where s ≥ 1 is the scaling parameter.

In contrast to p and w,p andŵ represent the rescaled matching probability and partial derivative weight, respectively.

We remark that we scale the absolute weight

Batch setting: C classes, N c images from c-th class, batch size N = C c=1 N c .

Hyper-setting: The scaling parameter s and the number of iterations τ .

Input: Initialised embedding function f , iteration counter iter = 0.

Output: Updated f .

for iter < τ do iter = iter + 1.

Sample one mini-batch randomly X = {{x

Step 1: Feedforward X into f to obtain feature representations {{f

Step 2: Compute the similarities between an anchor and the remaining instances.

Every example serves as the anchor iteratively.

Step 3: Gradient back-propagation to update the parameters of f using Eq. (15).

end for non-linearly, which is an indirect way of controlling the relative weight.

We do not modify the relative weight directly and Eq. (9) and Eq. (10) are only for introducing our motivation.

Our objective is to maximise an anchor's matching probability with its any positive instance competing against its negative set.

Therefore, we normalise the rescaled weights based on each anchor:

.

Note that the denominators in Eq. (13) and (14) are the accumulated weights of positives and negatives w.r.t.

x c a , respectively.

Although there are much more negatives than positives, the negative set and positive set contribute equally as a whole, as indicated by 1/2.

N = C c=1 N c is the total number of instances in X. We select each instance as the anchor iteratively and treat all anchors equally, as indicated by 1/N .

It is worth noting that during back-propagation, the magnitudes of partial derivatives in Eq. (7) and Eq. (8)

To make it more clear and intuitive for understanding, we now analyse a naive case of ICE, where there are two samples per class in every mini-batch, i.e., ∀c, N c = 2, |{x c i } i =a | = N c − 1 = 1.

In this case, for each anchor (query), there is only one positive among the remaining data points.

As a result, the weighting schemes in Eq. (13) for positives and Eq. (14) for negatives can be simplified:

Firstly, we have N anchors that are treated equally as indicated by 1/N .

Secondly, for each anchor, we aim to recognise its positive example correctly.

However, there is a sample imbalance problem because each anchor has only one positive and many negatives.

ICE addresses it by treating the positive set (single point) and negative set (multiple points) equally, i.e., 1/2 in Eq. (16) and Eq. (17) 2 .

Finally, as there are many negative samples, we aim to focus more on informative ones, i.e., harder negative instances with higher matching probabilities with a given anchor.

The non-linear transformation can help control the relative weight between two negative points.

The weighting scheme shares the same principle as the popular temperature-based categorical cross entropy (Hinton et al., 2015; Oreshkin et al., 2018) .

The key is that we should consider not only focusing on harder examples, but also the emphasis degree.

Algorithm 1 summarises the learning process with ICE.

As presented there, the input data format of ICE is the same as CCE, i.e., images and their corresponding labels.

In contrast to other methods which require rigid input formats (Schroff et al., 2015; Sohn, 2016) , e.g., triplets and n-pair tuplets, ICE is much more flexible.

We iteratively select one image as the anchor.

For each anchor, we aim to maximise its matching probabilities with its positive samples against its negative examples.

Therefore, the computational complexity over one mini-batch is O(N 2 ), being the same as recent online metric learning approaches (Song et al., 2016; Wang et al., 2019b) .

Note that in FaceNet (Schroff et al., 2015) and N -pair-mc (Sohn, 2016) , expensive sample mining and class mining are applied, respectively.

For data augmentation and preprocessing, we follow (Song et al., 2016; .

In detail, we first resize the input images to 256 × 256 and then crop it at 227 × 227.

We use random cropping and horizontal mirroring for data augmentation during training.

To fairly compare with the results reported in (Song et al., 2017) , we use a centre cropping without horizontal flipping in the test phase.

For the embedding size, we set it to 512 on all datasets following (Sohn, 2016; Law et al., 2017; Wang et al., 2019a) .

To compare fairly with (Song et al., 2017; Law et al., 2017; MovshovitzAttias et al., 2017) , we choose GoogLeNet V2 (with batch normalisation) (Ioffe & Szegedy, 2015) as the backbone architecture initialised by the publicly available pretrained model on ImageNet (Russakovsky et al., 2015) .

We simply change the original 1000-neuron fully connected layers followed by softmax normalisation and CCE to 512-neuron fully connected layers followed by the proposed ICE.

For faster convergence, we randomly initialise the new layers and optimise them with 10 times larger learning rate than the others as in (Song et al., 2016) .

We implement our algorithm in the Caffe framework (Jia et al., 2014) .

The source code will be available soon.

Datasets.

Following the evaluation protocol in (Song et al., 2016; , we test our proposed method on three popular fine-grained datasets including CARS196 (Krause et al., 2013) , CUB-200-2011 (Wah et al., 2011 and SOP (Song et al., 2016) .

A summary of the datasets is given in 2 The weight sum of negatives: Table 1 .

We also keep the same train/test splits.

We remark that to test the generalisation and transfer capability of the learned deep metric, the training and test classes are disjoint.

Evaluation protocol.

We evaluate the learned representations on the image retrieval task in terms of Recall@K performance (Song et al., 2016) .

Given a query, its K nearest neighbours are retrieved from the database.

Its retrieval score is one if there is an image of the same class in the K nearest neighbours and zero otherwise.

Recall@K is the average score of all queries.

Training settings.

All the experiments are run on a single PC equipped with Tesla V100 GPU with 32GB RAM.

For optimisation, we use the stochastic gradient descent (SGD) with a weight decay of 1e −5 and a momentum of 0.8.

The base learning rate is set as 1e

Remarks.

For a fair comparison, we remark that the methods group (Ustinova & Lempitsky, 2016; Harwood et al., 2017; Wang et al., 2017b; Suh et al., 2019; Zheng et al., 2019) using GoogLeNet V1 and another group (Wu et al., 2017; Cakir et al., 2019; Sanakoyeu et al., 2019 ) using ResNet-50 (He et al., 2016 are not benchmarked.

Besides, ensemble models Opitz et al., 2017; Kim et al., 2018; Xuan et al., 2018) are not considered.

HTL (Ge et al., 2018 ) also uses GoogLeNet V2, but it constructs a hierarchical similarity tree over the whole training set and updates the tree every epoch, thus being highly unscalable and expensive in terms of both computation and memory.

That is why HTL achieves better performance on small datasets but performs worse than ours on the large dataset SOP.

Finally, there are some other orthogonal deep metric learning research topics that are worth studying together in the future, e.g., a robust distance metric (Yuan et al., 2019) and metric learning with continuous labels .

In GoogLeNet V2, there are three fully connected layers of different depth.

We refer them based on their depth: L for the low-level layer (inception-3c/output), M for the mid-level layer (inception-4e/output) and H for the high-level layer (inception5b/output).

By default, we use only 'H'.

We also report the results of their combination (L, M, H) for reference following RLL (Wang et al., 2019a) .

Competitors.

All the compared baselines, Triplet Semihard (Schroff et al., 2015) , Lifted Struct (Song et al., 2016) , N -pair-mc (Sohn, 2016) , Struct Clust (Song et al., 2017 ), Spectral Clust (Law et al., 2017 , Proxy-NCA (Movshovitz-Attias et al., 2017) , RLL (Wang et al., 2019a) and our ICE are trained and evaluated using the same settings: (1) GoogLeNet V2 serves as the backbone network; (2) All models are initialised with the same pretrained model on ImageNet; (3) All apply the same data augmentation during training and use a centre-cropped image during testing.

The results of some baselines (Schroff et al., 2015; Song et al., 2016; Sohn, 2016) are from (Song et al., 2017) , which means they are reimplemented there for a fair comparison.

In addition, the results of vanilla GoogLeNet V2 pretrained on ImageNet without fine-tuning and with fine-tuning via minimising CCE are reported in (Law et al., 2017) , which can be regarded as the most basic baselines.

Among these baselines, Proxy NCA is not scalable as class-level proxies are learned during training.

Struct Clust and Spectral Clust are clustering-motivated methods which explicitly aim to optimise the clustering quality.

We highlight that clustering performance Normalised Mutual Information (NMI) (Schütze et al., 2008) is not a good assessment for SOP (Law et al., 2017) because SOP has a large number of classes but only 5.3 images per class on average.

Therefore, we only report and compare Recall@K performance.

Results.

Table 3 compares the results of our ICE and those of the state-of-the-art DML losses.

ICE achieves the best Recall@1 performance on all benchmarks.

We observe that only RLL achieves comparable performance in a few terms.

However, RLL is more complex since it has three hyperparameters in total: one weight scaling parameter and two distance margins for positives and negatives, respectively.

In addition, its perspective is different since it processes the positive set together similarly with (Triantafillou et al., 2017; Wang et al., 2019c) .

We note that (Wang et al., 2019c ) is also complex in designing weighting schemes and contains four control hyper-parameters.

However, our Recall@1 on SOP is 77.3%, which is only 0.9% lower than 78.2% of (Wang et al., 2019c) .

It is also worth mentioning that among these approaches, except fine-tuned models with CCE, only our method has a clear probability interpretation and aims to maximise the joint instance-level matching probability.

As observed, apart from being unscalable, CCE's performance is much worse than the state-of-the-art methods.

Therefore, ICE can be regarded as a successful exploration of softmax regression for learning deep representations in DML.

The t-SNE visualisation (Van Der Maaten, 2014) of learned embeddings are available in the supplementary material.

We empirically study the impact of the weight scaling parameter s, which is the only hyperparameter of ICE.

It functions similarly with the popular sample mining or example weighting (Wang et al., 2019a; b; c) widely applied in the baselines in Table 3 .

Generally, different s corresponds to different emphasis degree on difficult examples.

When s is larger, more difficult instances are assigned with relatively higher weights.

In general, small datasets are more sensitive to minor changes of hyper-settings and much easier to overfit.

Therefore, the experiments are conducted on the large dataset SOP.

The results are shown in Table 2 .

Note that when s is too small, e.g., s = 1, we observe that the training does not converge, which demonstrates the necessity of weighting/mining samples.

The most significant observation is that focusing on difficult samples is better but the emphasis degree should be properly controlled.

When s increases from 16 to 64, the performance grows gradually.

However, when s = 80, we observe the performance drops a lot.

That may be because extremely hard samples, e.g., outliers, are emphasised when s is too large.

In this paper, we propose a novel instance-level softmax regression framework, named instance cross entropy, for deep metric learning.

Firstly, the proposed ICE has clear probability interpretation and exploits structured semantic similarity information among multiple instances.

Secondly, ICE is scalable to infinitely many classes, which is required by DML.

Thirdly, ICE has only one weight scaling hyper-parameter, which works as mining informative examples and can be easily selected via cross-validation.

Finally, distance thresholds are not applied to achieve intraclass compactness and interclass separability.

This indicates that ICE makes no assumptions about intraclass variances and the boundaries between different classes.

Therefore ICE owns general applicability.

A.1 BATCH CONTENT We evaluate the impact of batch content which consists of C classes and k images per class, i.e., ∀c, N c = k. The batch size N = C × k is set to 180.

In our experiments, we change the number of classes C from 36 to 90, and the number of images k from 2 to 5, while keeping the batch size unchanged.

Table 4 shows the results on SOP dataset.

We observe that when there are more classes in the mini-batch, the performance is better.

We conjecture that as the number of classes increases, the mini-batch training becomes more difficult and helps the model to generalise better.

To explore different batch size N , we fix k = 2 and only change C. In this case, N = C × 2.

Table 5 shows that as the number of classes increases, the performance grows.

In detail, when the number of classes increases from 50 to 90, the performance raises from 74.4% to 77.3% accordingly.

One reason may be that as the number of classes increases, it fits the global structure of the test set better, where there are a large number of classes but only a few positive examples.

In addition, the increasing difficulty of mini-batch training can help the model to generalise better.

The dimension of feature representations is an important factor in many DML methods.

We conduct experiments on SOP to see the influence of different embedding size.

The results are presented in Table 6 .

We observe that when the embedding size is very small, e.g., 64, the performance is much worse.

The performance increases gradually as the embedding size grows.

The t-SNE visualisation (Van Der Maaten, 2014) of learned embeddings is available in Figures 2, 3 , 4.

@highlight

We propose instance cross entropy (ICE) which measures the difference between an estimated instance-level matching distribution and its ground-truth one. 