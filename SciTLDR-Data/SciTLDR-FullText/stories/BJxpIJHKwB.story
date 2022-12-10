Few shot image classification aims at learning a classifier from limited labeled data.

Generating the classification weights has been applied in many meta-learning approaches for few shot image classification due to its simplicity and effectiveness.

However, we argue that it is difficult to generate the exact and universal classification weights for all the diverse query samples from very few training samples.

In this work, we introduce Attentive Weights Generation for few shot learning via Information Maximization (AWGIM), which addresses current issues by two novel contributions.

i) AWGIM generates different classification weights for different query samples by letting each of query samples attends to the whole support set.

ii) To guarantee the generated weights adaptive to different query sample, we re-formulate the problem to maximize the lower bound of mutual information between generated weights and query as well as support data.

As far as we can see, this is the first attempt to unify information maximization into few shot learning.

Both two contributions are proved to be effective in the extensive experiments and we show that AWGIM is able to achieve state-of-the-art performance on benchmark datasets.

While deep learning methods achieve great success in domains such as computer vision (He et al., 2016) , natural language processing (Devlin et al., 2018) , reinforcement learning (Silver et al., 2018) , their hunger for large amount of labeled data limits the application scenarios where only a few data are available for training.

Humans, in contrast, are able to learn from limited data, which is desirable for deep learning methods.

Few shot learning is thus proposed to enable deep models to learn from very few samples (Fei-Fei et al., 2006) .

Meta learning is by far the most popular and promising approach for few shot problems (Vinyals et al., 2016; Finn et al., 2017; Snell et al., 2017; Ravi & Larochelle, 2016; Rusu et al., 2019) .

In meta learning approaches, the model extracts high level knowledge across different tasks so that it can adapt itself quickly to a new-coming task (Schmidhuber, 1987; Andrychowicz et al., 2016) .

There are several kinds of meta learning methods for few shot learning, such as gradient-based (Finn et al., 2017; Ravi & Larochelle, 2016) and metric-based (Snell et al., 2017; Sung et al., 2018) .

Weights generation, among these different methods, has shown effectiveness with simple formulation (Qi et al., 2018; Qiao et al., 2018; Gidaris & Komodakis, 2018; .

In general, weights generation methods learn to generate the classification weights for different tasks conditioned on the limited labeled data.

However, fixed classification weights for different query samples within one task might be sub-optimal, due to the few shot challenge.

We introduce Attentive Weights Generation for few shot learning via Information Maximization (AWGIM) in this work to address these limitations.

In AWGIM, the classification weights are generated for each query sample specifically.

This is done by two encoding paths where the query sample attends to the task context.

However, we show in experiments that simple cross attention between query samples and support set fails to guarantee classification weights fitted to diverse query data since the query-specific information is lost during weights generation.

Therefore, we propose to maximize the lower bound of mutual information between generated weights and query, support data.

As far as we know, AWGIM is the first work introducing Variational Information Maximization in few shot learning.

The induced computational overhead is minimal due to the nature of few shot problems.

Furthermore, by maximizing the lower bound of mutual information, AWGIM gets rid of inner update without compromising performance.

AWGIM is evaluated on two benchmark datasets and shows state-of-the-art performance.

We also conducted detailed analysis to validate the contribution of each component in AWGIM.

2 RELATED WORKS 2.1 FEW SHOT LEARNING Learning from few labeled training data has received growing attentions recently.

Most successful existing methods apply meta learning to solve this problem and can be divided into several categories.

In the gradient-based approaches, an optimal initialization for all tasks is learned (Finn et al., 2017) .

Ravi & Larochelle (2016) learned a meta-learner LSTM directly to optimize the given fewshot classification task.

Sun et al. (2019) learned the transformation for activations of each layer by gradients to better suit the current task.

In the metric-based methods, a similarity metric between query and support samples is learned. (Koch et al., 2015; Vinyals et al., 2016; Snell et al., 2017; Sung et al., 2018; Li et al., 2019a) .

Spatial information or local image descriptors are also considered in some works to compute richer similarities (Lifchitz et al., 2019; Li et al., 2019b; Wertheimer & Hariharan, 2019) .

Generating the classification weights directly has been explored by some works.

Gidaris & Komodakis (2018) generated classification weights as linear combinations of weights for base and novel classes.

Similarly, Qiao et al. (2018) and Qi et al. (2018) both generated the classification weights from activations of a trained feature extractor.

Graph neural network denoising autoencoders are used in (Gidaris & Komodakis, 2019) .

Munkhdalai & Yu (2017) proposed to generate "fast weights" from the loss gradient for each task.

All these methods do not consider generating different weights for different query examples, nor maximizing the mutual information.

There are some other methods for few-shot classification.

Generative models are used to generate or hallucinate more data in Wang et al., 2018; Chen et al., 2019) .

Bertinetto et al. (2019) and used the closed-form solutions directly for few shot classification.

integrated label propagation on a transductive graph to predict the query class label.

Attention mechanism shows great success in computer vision (Xu et al., 2015; Parmar et al., 2018) and natural language processing (Bahdanau et al., 2015; Vaswani et al., 2017) .

It is effective in modeling the interaction between queries and key-value pairs from certain context.

Based on the fact that keys and queries point to the same entities or not, people refer to attention as self attention or cross attention.

In this work, we use both types of attention to encode the task and query-task information.

The work most similar to ours is Attentive Neural Processes , which also employs self and cross attention.

However, we are using attention for few-shot image classification via maximizing the mutual information.

In stark contrast, worked on regression from the perspective of a stochastic process and the variational objective is optimized.

Given two random variables x and y, mutual information I(x; y) measures the decrease of uncertainty in one random variable when another is known.

It is defined as the Kullback-Leibler divergence between joint distribution p(x, y) and product of marginal distributions p(x) ⊗ p(y),

When x and y are independent, p(x, y) = p(x) ⊗ p(y) so that I(x, y) = 0, indicating that knowing x does not reveal any information about y. When y is a deterministic function of x, I(x, y) achieves its maximum value.

Mutual information has been widely applied in applications such as Generative Adversarial Networks , self-supervised learning (Hjelm et al., 2019) , visual question generation Krishna et al. (2019) and so on.

Similarly, the attentive path enables the query samplex to be equipped with task knowledge.

Both paths are achieved by attention mechanism.x ap is repeated to concatenate with X cp .

The weight generator g takes these concatenated representations as input to generate classification weights W specific forx, denoted by the colorful matrix with slash.

It can be used to predict the class label forx and X. W is also used to reconstruct the inputs of the generator g by two networks r 1 and r 2 .

In this way, the lower bound of mutual information is maximized and g is forced to generate classification weights sensitive to different query samples.

In this section, we provide the problem formulation first.

Then the proposed model is described in Sec. 3.3.

The objective function, which maximizes the mutual information between certain variables, and theoretical analysis are provided in Sec. 3.4.

Following many popular meta-learning methods for few shot classification, we formulate the problem under episodic training paradigm (Vinyals et al., 2016; Finn et al., 2017) .

One N -way K-shot task sampled from an unknown task distribution P (T ) includes support set and query set:

where S = {(x cn;k , y cn;k )|k = 1, ..., K; n = 1, ..., N }, Q = {(x 1 , ...,x |Q| )}.

Support set S contains N K labeled samples.

Query set Q includesx and we need to predict labelŷ forx based on S. During meta-training, the meta-loss is estimated on Q to optimize the model.

During metatesting, the performance of meta-learning method is evaluated on Q, provided the labeled S. The classes used in meta-training and meta-testing are disjoint so that the meta-learned model needs to learn the knowledge transferable across tasks and adapt itself quickly to novel tasks.

Our proposed approach follows the general framework to generate the classification weights (Qi et al., 2018; Qiao et al., 2018; Rusu et al., 2019; Gidaris & Komodakis, 2018; .

In this framework, there is a feature extractor to output image feature embeddings.

The meta-learner needs to generate the classification weights for different tasks.

Latent Embedding Optimization (LEO) (Rusu et al., 2019 ) is one of the weights generation methods that is most related to our work.

In LEO, the latent code z is generated by h conditioned on support set S, described as z = h(S).

h is instantiated as relation networks (Santoro et al., 2017) .

Classification weights w can be decoded from z with l, w = l(z).

In the inner loop, we use w to compute the loss (usually cross entropy) on the support set and then update z:

where L S indicates that the loss is evaluated on S only.

The updated latent code z is used to decode new classification weights w with generating function l. w is adopted in the outer loop for query set Q and the objective function of LEO then can be written as min

Here θ stands for the parameters of h and l and we omit the regularization terms for clarity.

LEO avoids updating high-dimensional w in the inner loop by learning a lower-dimensional latent space, from which sampled z can be used to generate w. The most significant difference between LEO and AWGIM is that we do not need inner updates to adapt the model.

Instead, AWGIM is a feedforward network trained to maximize the mutual information so that it fits to different tasks well.

On the other hand, AWGIM learns to generate optimal classification weights for each query sample while LEO generates fixed weights conditioned on the support set within one task.

In Section 3.4 we will show LEO can be casted as a special case of AWGIM under certain conditions.

The framework of our proposed method is shown in Figure 1 .

Assume that we have a feature extractor, which can be a simple 4-layer Convnet or a deeper Resnet.

All the images included in the sampled task T are processed by this feature extractor and represented as d-dimensional vectors afterwards, i.e., x cn;k ,x ∈ R d .

There are two paths to encode the task context and the individual query sample respectively, which are called contextual path and attentive path.

The outputs of both paths are concatenated together as input to the generator for classification weights.

Generated classification weights are used to not only predict the label ofx, but also maximize the lower bound of mutual information between itself and other variables, which will be discussed in the following section 3.4.

The encoding process includes two paths, namely the contextual path and attentive path.

The contextual path aims at learning representations for only the support set with a multi-head self-attention network f cp sa (Vaswani et al., 2017) .

The outputs of contextual path X cp ∈ R N K×d h 1 thus contain richer information about the task and can be used later for weights generation.

Existing weights generation methods generate the classification weights conditioned on the support set only, which is equivalent to using contextual path.

However, the classification weights generated in this way might be sub-optimal.

This is because estimating the exact and universal classification weights from very few labeled data in the support set is difficult and sometimes impossible.

The generated weights are usually in lack of adaptation to different query samples.

We address this issue by introducing attentive path, where the individual query example attends to the task context and then is used to generate the classification weights.

Therefore, the classification weights are adaptive to different query samples and aware of the task context as well.

In the attentive path, a new multi-head self-attention network f ap sa on the support set is employed to encode the global task information.

f ap sa is different from f cp sa in contextual path because the selfattention network in contextual path emphasizes on generating the classification weights.

On the contrary, outputs of self-attention here plays the role of providing the V alue context for different query samples to attend in the following cross attention.

Sharing the same self-attention networks might limit the expressiveness of learned representations in both paths.

The cross attention network f ap ca applied on each query sample and task-aware support set is followed to produceX ap ∈ R |Q|×d h .

We use multi-head attention with h heads in both paths.

In one attention block, we produce h different sets of queries, keys and values.

Multi-head attention is claimed to be able to learn more comprehensive and expressive representations from h different subspaces (Vaswani et al., 2017; Voita et al., 2019) .

More details of these two paths can be found in A.2.

We replicate X cp ∈ R N K×d h andX ap ∈ R |Q|×d h for |Q| and N K times respectively and reshape them afterwards.

Then we have X cp ∈ R |Q|×N K×d h andX ap ∈ R |Q|×N K×d h .

These two tensors are concatenated to become X cp⊕ap ∈ R |Q|×N K×2d h .

X cp⊕ap can be interpreted that each query sample has its own latent representations for support set to generate specific classification weights, which are both aware of the task-context and adaptive to individual query sample.

cp⊕ap is decoded by the weights generator g : R 2d h → R 2d .

We assume that the classification weights follow Gaussian distribution with diagonal covariance.

g outputs the distribution parameters and we sample the weights from learned distribution during meta-training.

The sampled classification weights are represented as W ∈ R |Q|×N K×d .

To reduce complexity, we compute the mean value on K classification weights for each class to have W f inal ∈ R |Q|×N ×d .

Therefore, ith query sample has its specific classification weight matrix W f inal i,:,:

∈ R N ×d .

The prediction for query data can be computed byXW f inalT .

The support data X is replicated for |Q| times and reshaped as X s ∈ R |Q|×N K×d .

So the prediction for support data can also be computed as X s W f inalT .

Besides the weights generator g, we have another two decoders r 1 :

They both take the generated weights W as inputs and learn to reconstruct X cp andX ap respectively.

The outputs of r 1 and r 2 are denoted as X cp re ,X ap re ∈ R |Q|×N K×d h .

The reason we are using reconstruction as auxiliary tasks will be discussed in following Sec. 3.4.

In this section, we perform the analysis for one query sample without loss of generality.

The subscripts for classification weights are omitted for clarity.

In general, we use (x, y) and (x,ŷ) to represent support and query samples respectively.

Since the classification weights w generated from g are encoded with attentive path and contextual path, it is expected that we can directly have the query-specific weights.

However, we show in the experiments that simply doing this does not outperform a weight generator conditioned only on the S significantly, which implies that the generated classification weights from two paths are not sensitive to different query samples.

In other words, the information from attentive path is not kept well during the weights generation.

To address this limitation, we propose to maximize the mutual information between generated weights w and support as well as query data.

The objective function can be described as maxI((x,ŷ); w) + (x,y)∈S I((x, y); w)

According to the chain rule of mutual information, we have I((x,ŷ); w) = I(x; w) + I(ŷ; w|x).

Equation 6

Directly computing the mutual information in Equation 7 is intractable since the true posteriori distributions like p(ŷ|x, w), p(x|w) are still unknown.

Therefore, we use Variational Information Maximization (Barber & Agakov, 2003; to compute the lower bound of Equation 5.

We use p θ (x|w) to approximate the true posteriori distribution, where θ represents the model parameters.

As a result, we have

H(·) is the entropy of a random variable.

H(x) is a constant value for given data.

We can maximize this lower bound as the proxy for the true mutual information.

Similar to I(x; w),

(13) p θ (x|w), p θ (x, y|w) are used to approximate the true posteriori distribution p(x|w) and p(x, y|w).

Put the lower bounds back into Equation 7.

Omit the constant entropy terms and the expectation subscripts for clarity, we have the new objective function as

The first two terms are maximizing the log likelihood of label for both support and query data with respective to the network parameters, given the generated classification weights.

This is equivalent to minimizing the cross entropy between prediction and ground-truth.

We assume that p θ (x|w) and p θ (x|w) are Gaussian distributions.

r 1 and r 2 are used to approximate the mean of these two Gaussian distributions.

Therefore maximizing the log likelihood is equivalent to reconstruct x cp and x ap with L2 loss.

Thus the loss function to train the network can be written as is not equal to real log likelihood and we have to decide the weightage for each one.

λ 1 , λ 2 , λ 3 are thus hyper-parameters for trade-off of different terms.

With the help of last three terms, the generated classification weights are forced to carry information about the support data and the specific query sample.

In LEO (Rusu et al., 2019) , the inner update loss is computed as cross entropy on support data.

If we merge the inner update into outer loop, then the loss becomes the summation of first two terms in Equation 15.

However, the weight generation in LEO does not involve specific query samples, thus making reconstructingx ap impossible.

In this sense, LEO can be regarded as a special case of our proposed method, where (1) only contextual path exits and (2) λ 2 = λ 3 = 0.

The encoding process in contextual path results in computational complexity O((N K)

2 ) due to self-attention.

Similarly, the computational complexity of attentive path is O((N K) 2 + |Q|(N K)).

In total, the complexity is O((N K) 2 + |Q|(N K)).

However, because of the nature of few-shot learning problem, the value of (N K) 2 is usually negligible.

The value of |Q| depends on the setting and the cross attention can be implemented parallelly via matrix multiplication.

Therefore, the induced computational overhead will be negligible.

AWGIM avoids the inner update without compromising the performance, which furthers reduces both training and inference time significantly.

The empirical evaluation is presented in A.3.4.

We conduct experiments on miniImageNet (Vinyals et al., 2016) and tieredImageNet (Ren et al., 2018) , two commonly used benchmark datasets, to compare with other methods and analyze our model.

Both datasets are subsets of ILSVRC-12 dataset (Russakovsky et al., 2015) .

miniImageNet contains 100 randomly sampled classes with 600 images per class.

We follow the train/test split in (Ravi & Larochelle, 2016) , where 64 classes are used for meta-training, 16 for meta-validation and 20 for meta-testing.

tieredImageNet is a larger dataset compared to miniImageNet.

There are 608 classes and 779,165 images in total.

They are selected from 34 higher level nodes in ImageNet (Deng et al., 2009) hierarchy.

351 classes from 20 high level nodes are used for meta-training, 97 from 6 nodes for meta-validation and 160 from 8 nodes for meta-testing.

We use the image features in LEO (Rusu et al., 2019) provided by the authors 2 .

They trained a 28-layer Wide Residual Network (Zagoruyko & Komodakis, 2016) on the meta-training set.

Each image then is represented by a 640 dimensional vector, which is used as the input to our model.

For N -way K-shot experiments, we randomly sample N classes from meta-training set and each of them contains K samples as the support set and 15 as query set.

Similar to other works, we train 5-way 1-shot and 5-shot models on two dataset.

During meta-testing, 600 N -way K-shot tasks are sampled from meta-testing set and the average accuracy for query set is reported with 95% confidence interval, as done in recent works (Finn et al., 2017; Snell et al., 2017; Rusu et al., 2019) .

We use TensorFlow (Abadi et al., 2016) to implement our method and the code will be made available.

d = 640 is the dimension of feature embeddings.

d h is set to be 128.

The number of heads h in attention module is set to be 4.

g, r 1 and r 2 are 2-layer MLPs with 256 hidden units.

We decide λ 1 = 1, λ 2 = λ 3 = 0.001 by meta-validation performance. (Finn et al., 2017) Conv-4 48.70 ± 1.84% 63.11 ± 0.92% Meta LSTM (Ravi & Larochelle, 2016) Conv-4 43.44 ± 0.77% 60.60 ± 0.71% Prototypical Nets (Snell et al., 2017) Conv-4 49.42 ± 0.78% 68.20 ± 0.66% Relation Nets (Sung et al., 2018) Conv-4 50.44 ± 0.82% 65.32 ± 0.70% SNAIL (Mishra et al., 2018) Resnets-12 55.71 ± 0.99% 68.88 ± 0.92% TPN Resnets-12 59.46 75.65 MTL (Sun et al., 2019) Resnets-12 61.20 ± 1.80% 75.50 ± 0.80

Dynamic (Gidaris & Komodakis, 2018) WRN-28-10 60.06 ± 0.14% 76.39 ± 0.11% Prediction (Qiao et al., 2018) WRN-28-10 59.60 ± 0.41% 73.74 ± 0.19% DAE-GNN (Gidaris & Komodakis, 2019) WRN-28-10 62.96 ± 0.15% 78.85 ± 0.10% LEO (Rusu et al., 2019) WRN-28-10 61.76 ± 0.08% 77.59 ± 0.12% AWGIM (ours) WRN-28-10 63.12 ± 0.08% 78.40 ± 0.11% Table 2 : Accuracy comparison with other approaches on tieredImageNet.

The results are averaged on 600 tasks from meta-testing set with 95% confidence interval.

Best results are highlighted.

Model Feature Extractor 5-way 1-shot 5-way 5-shot MAML (Finn et al., 2017) Conv-4 51.67 ± 1.81% 70.30 ± 1.75% Prototypical Nets (Snell et al., 2017) Conv-4 53.31± 0.89% 72.69 ± 0.74% Relation Nets (Sung et al., 2018) Conv-4 54.48 ± 0.93% 71.32 ± 0.78% TPN Conv-4 59.91 ± 0.96% 72.85 ± 0.74% MetaOptNet Resnets (2017) is used to optimize the network with weight decay 1 × 10 −6 .

The initial learning rate is set to 0.0002 for 5-way 1-shot and 0.001 for 5-way 5-shot, which is decayed by 0.2 for every 15,000 iterations.

We train the model for 50,000 iterations.

Batch size is 64 for 5-way 1-shot and 32 for 5-way 5-shot.

Similar to LEO (Rusu et al., 2019) , we first train the model on meta-training set and choose the optimal hyper-parameters by validation results.

Then we train the model on meta-training and meta-validation sets together using fixed hyper-parameters.

We compare the performance of our approach AWGIM on two datasets with several state-of-theart methods proposed in recent years.

The results of MAML, Prototypical Nets, Relation Nets on tieredImageNet are evaluated by .

The results of Dynamic on miniImageNet with WRN-28-10 as the feature extractor is reported in (Gidaris & Komodakis, 2019) .

The other results are reported in the corresponding original papers.

We also include the backbone network structure of the used feature extractor for reference.

The results on miniImageNet and tieredImageNet are shown in Table 1 and 2 respectively.

The top half parts of Table 1 and 2 display the methods belonging with different meta learning categories, such as metric-based(Matching Networks, Prototypical Nets), gradient-based (MAML, MTL), graph-based (TPN).

The bottom part shows the classification weights generation approaches including Dynamic, Prediction, DAE-GNN, LEO and our proposed AWGIM.

AWGIM can outperform all the methods in top parts of two table.

Comparing with other classification weights generation methods in the bottom part, AWGIM still shows very competitive performance, namely the best on tieredImageNet and close to the state-of-the-art on miniImageNet.

We note that all the classification weights generation methods are using WRN-28-10 as backbone network, which makes the comparison fair.

In particular, AWGIM can outperform LEO in all settings.

Table 3 : Analysis of our proposed AWGIM.

In the top half, the attentive path is removed to compare with LEO.

In the bottom part, ablation analysis with respective to different components is provided.

We also shuffle the generated classification weights randomly to show that they are indeed optimal for different query samples.

We perform detailed analysis on AWGIM, shown in Table 3 .

We include the results of LEO Rusu et al. (2019) for reference.

"Generator in LEO" means that there is no inner update in LEO.

In the upper part of the table, we first studied the effect of attentive path.

We implemented two generators including only the contextual path during encoding.

"Generator conditioned on S with IM" indicates that we add the cross entropy loss and reconstruction loss for support set.

It can be observed that "Generator conditioned on S only" is trained with cross entropy on query set, which is similar to "Generator in LEO" without inner update.

It is able to achieve similar or slightly better results than "Generator in LEO", which implies that self-attention is no worse than relation networks used in LEO to model task-context.

With information maximization, our generator is able to obtain slightly better performance than LEO.

The effect of attention is investigated by replacing the attention modules with 2-layer MLPs, which is shown as "MLP encoding".

More specifically, one MLP in contextual path is used for support set and another MLP in attentive path for query samples.

We can see that even without attention to encode the task-contextual information, "MLP encoding" can achieve accuracy close to LEO, for the sake of information maximization.

However, if we let λ 1 = λ 2 = λ 3 = 0 for MLP encoding, the performance drops significantly, which demonstrates the importance of maximizing the information.

We conducted ablation analysis with respective to λ 1 , λ 2 and λ 3 to investigate the effect of information maximization.

First, λ 1 , λ 2 and λ 3 are all set to be 0.

In this case, the accuracy is similar to "generator conditioned on S only", showing that the generated classification weights are not fitted for different query samples, even with the attentive path.

It can also be observed that maximizing the mutual information between weights and support is more crucial since λ 1 = λ 2 = 0 degrades accuracy significantly, comparing with λ 3 = 0.

We further investigate the relative importance of the classification on support as well as reconstruction.

λ 1 = 0 affects the performance noticeably.

We conjecture that the support label prediction is more critical for information maximization.

The classification weights are generated specifically for each query sample in AWGIM.

To this point, we shuffle the classification weights between query samples within the same classes and between different classes as well to study whether the classification weights are adapted for different query samples.

Assume there are T query samples per class in one task.

W f inal ∈ R |Q|×N ×d can be reshaped into W f inal ∈ R N ×T ×N ×d .

Then we shuffle this weight tensor along the first and second axis randomly.

The results are shown as "random shuffle between classes" and "random shuffle in class" in Table 3 .

For 5-way 1-shot experiments, the random shuffle between classes degrades the accuracy noticeably while the random shuffle in class dose not affect too much.

This indicates that when the support data are very limited, the generated weights for query samples from the same class are very similar to each other while distinct for different classes.

When there are more labeled data in support set, two kinds of random shuffle show very close or even the same results in 5-way 5-shot experiments, which are both worse than the original ones.

This implies that the generated classification weights are more diverse and specific for each query sample in 5-way 5-shot setting.

The possible reason is that larger support set provides more knowledge to estimate the optimal classification weights for each query example.

More analysis is provided in Appendix A.3.

In this work, we introduce Attentive Weights Generation via Information Maximization (AWGIM) for few shot image classification.

AWGIM learns to generate optimal classification weights for each query sample within the task by two encoding paths.

To guarantee this, the lower bound of mutual information between generated weights and query, support data is maximized.

As far as we know, AWGIM is the first work utilizing mutual information techniques for few shot learning.

The effectiveness of AWGIM is demonstrated by state-of-the-art performance on two benchmark datasets and extensive analysis.

The multi-head attention can be described as

X h1 ∈ R N K×d h is the matrix where each row stands for one support sample x cn;k h1 .

For one N -way K-shot task, the outputs of f sa cp are represented by a matrix X cp ∈ R N K×d h .

The attentive path is instantiated by attention, similar to contextual path.

First, a MLP f 2 :

andx h2 .

Then we employ another H-head selfattention network f

cn;k h2 to encode the global task information to each support sample,

The cross attention between query and context-aware support samples are computed aŝ

HereX ap ∈ R |Q|×d h is the matrix form ofx q , where each query sample is context-aware.

∈ R 2d h , where i, j stands for ith query sample and jth support sample.

x cp⊕ap is decoded by the weights generator g : R 2d h → R 2d .

We assume that the classification weights follow Gaussian distribution with diagonal covariance and we sample the weights from this distribution during meta-training, shown in Equation 23 and 24.

A.3 EXPERIMENTAL ANALYSIS A.3.1 FEW SHOT REGRESSION AWGIM can be applied to few shot regression task by slight modification.

During meta-training, we set the number of classes N equal to 1 and adapt the cross entropy loss to mean square error.

We use the data points (x, y) as inputs to AWGIM and generate weight as well as bias parameters for a three layer MLP with hidden dimension 40.

This is consistent with few shot regression experimental setting in LEO.

The few shot regression tasks are constructed as either sinusoidal or linear regression tasks.

For sinusoidal regression tasks, the amplitude range is

We replace the multi-head attention in the two paths with single-head attention and conduct the 5-way 1-shot and 5-way 5-shot experiments on miniImageNet dataset.

The results are shown in Table  4 .

We can see clearly that multi-head attention improve the performance.

In particular, for 5-way 1-shot experiment, single head attention gives results close to MLP encoding, which indicates that single head attention struggles when data are extremely scarce.

We compare AWGIM with LEO in terms of convergence speed.

The batch size is set to be 16 for both methods.

We use the hyper-parameters tuned by authors to train LEO.

The accuracy of metavalidation set during meta-training on 5-way 1-shot miniImageNet is plotted, shown in Figure 3 .

we can see clearly that AWGIM converges faster than LEO and outperforms LEO except for the first few iterations.

We measure the inference time of AWGIM to show that it induces minimal computational overhead.

In comparison, we use "MLP encoding" in two paths, which has time complexity O(N K + |Q|).

We use two set-ups on miniImageNet and the batch size is set to be 64.

100 batches are processed and we report the average consumed time for one batch.

All these experiments on done with the same GPU and workstation.

The results are shown in Table 5 .

It can be observed that the usage of self-attention and cross attention in AWGIM occurs negligible overhead, compared with MLP encoding.

This is because the values of N, K, |Q| are all relatively small and matrix multiplication further can be processed very fast by GPU.

We visualize the generated classification weights by t-SNE (Maaten & Hinton, 2008 ).

First we sample 400 tasks from meta-validation set of 5-way 1-shot miniImageNet experiment.

Each task contains 5 query samples from 5 different classes.

Thus in total there are 400 × 5 × 5 = 10, 000 weight vectors to visualize.

As comparison, inputs to the generator g are also plotted.

The visualization results are shown in Figure 4 .

The inputs to g are displayed in (a, b) and the generated classification weights in (c, d) .

From the comparison between (a) and (c), we can see the decoded weights for each class in (c) are clustered closer than (a) in general.

Red and blue dots in (b, d) denotes the classification weights for two query samples from two classes within one task.

It can be observed that g can generate adapted weights for different query samples.

This is consistent with Table 3 , where the results of "random shuffle between classes" suggest that query samples from different class have distinct classification weights.

(a) (b) (c) (d) Figure 4 : t-SNE visualization of the inputs to g in (a, b) and the generated classification weights in (c, d).

Blue and red dots in (b) and (d) are the classification weights for two query samples in the same task.

@highlight

A novel few shot learning method to generate query-specific classification weights via information maximization.