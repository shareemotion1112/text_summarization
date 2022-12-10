There is a growing interest in automated neural architecture search (NAS).

To improve the efficiency of NAS, previous approaches adopt  weight sharing method to force all models share the same set of weights.

However, it has been observed that a model performing better with shared weights does not necessarily perform  better when trained alone.

In this paper, we analyse existing weight sharing one-shot NAS approaches from a Bayesian point of view and identify the posterior fading problem, which compromises the effectiveness of shared weights.

To alleviate this problem, we present a practical approach to guide the parameter posterior towards its true distribution.

Moreover, a hard latency constraint is introduced during the search so that the desired latency can be achieved.

The resulted method, namely Posterior Convergent NAS (PC-NAS), achieves state-of-the-art performance under standard GPU latency constraint on ImageNet.

In our small search space, our model PC-NAS-S attains76.8% top-1 accuracy, 2.1% higher than MobileNetV2 (1.4x) with the same latency.

When adopted to our large search space, PC-NAS-L achieves 78.1% top-1 accuracy within 11ms.

The discovered architecture also transfers well to other computer vision applications such as object detection and person re-identification.

Neural network design requires extensive experiments by human experts.

In recent years, there has been a growing interest in developing algorithmic NAS solutions to automate the manual process of architecture design (Zoph & Le, 2016; Liu et al., 2018a; Zhong et al., 2018; .

Despite remarkable results, early works on NAS (Real et al., 2018; Elsken et al., 2017) are limited to searching only using proxy or subsampled dataset due to the exorbitant computational cost.

To overcome this difficulty, (Bender et al., 2018; Pham et al., 2018) attempted to improve search efficiency via sharing weights across models.

These approaches utilize an overparameterized network (supergraph) containing every single model, which can be further divided into two categories.

The first category is continuous relaxation method (Liu et al., 2018c; Cai et al., 2018) , which keeps a set of so called architecture parameters to represent the model, and updates these parameters alternatively with supergraph weights.

The resulting model is obtained using the architecture parameters at convergence.

The continuous relaxation method suffers from the rich-get-richer problem (Adam & Lorraine, 2019) , which means that a better-performed model at the early stage would be trained more frequently (or have larger learning rates).

This introduces bias and instability to the search process.

The other category is referred to as one-shot method (Brock et al., 2017b; Bender et al., 2018; Chu et al., 2019) , which divides the NAS procedure into a training stage and a searching stage.

In the training stage, the supergraph is optimized along with either dropping out each operator with certain probability or sampling uniformly among candidate architectures.

In the search stage, a search algorithm is applied to find the architecture with the highest validation accuracy with shared weights.

The one-shot approach ensures the fairness among all models by sampling architecture or dropping out operator uniformly.

However, as identified in (Adam & Lorraine, 2019; Chu et al., 2019; Bender et al., 2018) , the problem of one-shot method is that the validation accuracy of the model with shared weights is not predictive to its true performance.

In this paper, we formulate NAS as a Bayesian model selection problem (Chipman et al., 2001 ).

This formulation is especially helpful in understanding the one-shot approaches in a theoretical way, which in turn provides us a guidance to fundamentally addressing one of the major issues of one-shot approaches.

Specially, we show that shared weights are actually a maximum likelihood estimation of a proxy distribution to the true parameter distribution.

Most importantly, we identify the common issue of weight sharing, which we call Posterior Fading, i.e., as the number of models in the supergraph increases, the KL-divergence between true parameter posterior and proxy posterior also increases.

To alleviate the Posterior Fading problem, we proposed a practical approach to guide the convergence of the proxy distribution towards the true parameter posterior.

Specifically, we divide the training of supergraph into several intervals and maintain a pool of high potential partial models and progressively update this pool after each interval .

At each training step, a partial model is sampled from the pool and complemented to a full model.

To update the partial model pool, we first generate candidates by extending each partial model and evaluate their potentials, keeping the best performancing ones.

The search space is effectively shrunk in the upcoming training interval.

Consequently, the parameter posterior get close to the desired true posterior during this procedure.

Main contributions of our work is concluded as follows:

• We for the first time analyse one-shot approaches from a Bayesian point of view and identify the associated disadvantage which we call Posterior Fading.

• Guided by the theoretical result, we introduce a novel NAS algorithm fundamentally different from existing one-shot methods, which guides the proxy distribution to converge towards the true parameter posterior.

To examine the effectiveness of our newly proposed method, we benchmark its performance on ImageNet (Russakovsky et al., 2015) against the existing methods.

In one typical search space (Cai et al., 2018) , our PC-NAS-S attains 76.8% top-1 accuracy, 0.5% higher and 20% faster than EfficientNet-B0 (Tan & Le, 2019a) , which is the previous state-of-the-art model in mobile setting.

To further demonstrate the advantage of our method, we test it on a larger space and our PC-NAS-L boosts the accuracy to 78.1%.

Early neural architecture search (NAS) (Miller et al., 1989; Liu et al., 2018b; Real et al., 2017; Zoph & Le, 2016; Baker et al., 2017a; Wang et al., 2019 ) methods normally involves with reinforcement learning or neuro-evolution.

This type of NAS is typically considered as an agent-based explore and exploit process, where an agent (e.g. an evolution mechanism or a recurrent neural network(RNN)) is introduced to explore a given architecture space with training a network in the inner loop to get an evaluation for guiding exploration.

Such methods are computationally expensive and hard to be used on large-scale datasets, e.g. ImageNet.

Recent works (Pham et al., 2018; Brock et al., 2017a; Liu et al., 2018c; Cai et al., 2018) try to alleviate this computation cost via modeling NAS as a single training process of an over-parameterized network that comprises all candidate models, in which weights of the same operators in different models are shared.

ENAS (Pham et al., 2018) reduces the computation cost by orders of magnitude, while requires an RNN agent and focuses on small-scale datasets (e.g. CIFAR10).

One-shot NAS (Brock et al., 2017b) trains the over-parameterized network along with droping out each operator with increasing probability.

Then it use the pre-trained over-parameterized network to evaluate randomly sampled architectures.

DARTS (Liu et al., 2018c) additionally introduces a real-valued architecture parameter for each operator and alternately train operator weights and architecture parameters by back-propagation.

ProxylessNAS (Cai et al., 2018) binarize the real-value parameters in DARTS to save the GPU cumputation and memory for training the over-parameterized network.

The paradigm of ProxylessNAS (Cai et al., 2018) and DARTS (Liu et al., 2018c) introduce unavoidable bias since operators of models performing well in the beginning will easily get trained more and normally keep being better than other.

But they are not necessarily superior than others when trained from scratch.

figure) operators.

However, only one operator in each mixop is invoked at a time for each batch.

In (b), partial models 1 and 2 in the pool consist of choices in mixop 1 and 2.

We extend these 2 partial models to one mixop 3.

6 extended candidate models are evaluated and ranked in(c).

In (d), the new pool consists of the top-2 candidate models ranked in (c).

Other relevant works are ASAP and XNAS , which introduce pruning during the training of over-parameterized networks to improve the efficiency of NAS.

Similar to these approaches, we start with an over-parameterized network and then reduce the search space to derive the optimized architecture.

Instead of focusing on the speed-up of training and only pruning by evaluating the architecture parameters, we further improve the rankings of models and evaluate operators directly on validation set.

In this section, we first formulate neural architecture search in a Bayesian manner.

Utilizing this setup, we introduce the PC-NAS algorithm and analyse its advantage comparing to previous approaches.

Finally, we discuss the search algorithm combined with latency constraint.

Suppose we have K different models M = {m 1 , ..., m K } under consideration for data D, and p(D|θ k , m k ) describes the probability density of data D given model m k and its associated parameters θ k .

The Bayesian approach proceeds by assigning a prior probability distribution p(θ k |m k ) to the parameters of each model, and a prior probability p(m k ) to each model.

In order to ensure fairness among all models, we set the model prior p(m k ) a uniform distribution.

Under previous setting, we can drive

where

Since p(m k ) is uniform, the Maximum Likelihood Estimation (MLE) of m k is just the maximum of (2).

It can be inferred that, p(θ k |m k ) is crucial to the solution of the model selection.

We are interested in attaining the model with highest test accuracy in a trained alone manner, thus the parameter prior p(θ k |m k ) is just the posterior p alone (θ k |m k , D) which means the distribution of θ k when m k is trained alone on dataset D. Thus we would use the term true parameter posterior to refer p alone (θ k |m k , D).

We constrain our discussion in the setting which is frequently used in NAS literature for simplicity.

As a building block of our search space, a mixed operator (mixop), denoted by

The search space is defined by L mixed operators (layers) connected sequentially interleaved by downsampling as in Fig. 1(a) .

The network architecture (model) m is defined by a vector

representing the choice of operator for layer l. The parameter for the operator o at the l-th layer is denoted as θ lo .

The parameters of the supergraph are denoted by θ which includes {θ lo |l ∈ {1, 2, ..., L}, o ∈ O}. In this setting, the parameters of each candidate operator are shared among multiple architectures.

The parameters related with a specific model m k is denoted

which is a subset of the parameters of the supergraph θ.

Obtaining the p alone (θ k |m k , D) or a MLE of it for every model is computationally intractable.

Therefore, the one-shot method trains the supergraph by dropping out each operator (Brock et al., 2017b) or sampling different architectures (Bender et al., 2018; Chu et al., 2019) and utilize the shared weights to evaluate single model.

In this work, we adopt the latter training paradigm while the former one could be easily generalized.

Suppose we sample a model m k and optimize the supergraph with a mini-batch of data based on the objective function L alone :

where − log p(θ|m k ) is a regularization term.

Thus minimizing this objective equals to making MLE to p alone (θ|m k , D).

When training the supergraph, we sample many models m k , and then train the parameters for these models, which corresponds to a stochastic approximation of the following objective function:

This is equivalent to adopting a proxy parameter posterior as follows:

Maximizing p share (θ|D) is equivalent to minimizing L share .

We take one step further to assume that the parameters at each layer are independent, i.e.

Due to the independence, we have

where

The KL-divergence between p alone (θ l,o |m k , D) and p share (θ l,o |D) is as follows:

The KL-divergence is just the summation of the cross-entropy of p alone (θ l,o |m k , D) and p alone (θ l,o |m i , D) where i = k. The cross-entropy term is always positive.

Increasing the number of architectures would push p share away from p alone , namely the Posterior Fading.

We conclude that non-predictive problem originates naturally from one-shot supergraph training, as a typical search space contains a huge 10 21 architectures.

Thus if we effectively reduce the number of architectures in (10) during training, the KL divergence would decrease.

This is the intuition of our PC-NAS algorithm.

Inputs: G(supergraph), L(num of mixops in G), m (partial candidate), Lat(latency constraint), S(evaluation number), D val (validataion set) Scores = ∅ for i = 1 : S do m * = expand(m ) randomly expand m to full depth L if Latency(m * ) > Lat

The naive approach to mitigate the posterior fading problem is to limit the number of candidate models inside the supergraph.

However, large number of candidates is demanded for NAS to discover promising models.

Due to this conflict, we present PC-NAS which adopt progressive search space shrinking.

The resulted algorithm divides the training of shared weights into L intervals, where L is the number of mixed operators in the search space.

The number of training epochs of a single interval is denoted as T i .

Partial model pool is a collection of partial models.

At the l-th interval, a single partial model

The size of partial model pool is denoted as P .

After the l-th interval, each partial model in the pool will be extended by the N operators in l-th mixop.

Thus there are P × N candidate extended partial models with length l. These candidate partial models are evaluated and the top-P among which are used as the partial model pool for the interval l + 1.

An illustrative exmaple of partial model pool update is in Fig. 1(b

Candidate evaluation with latency constraint We define the potential of a partial model to be the expected validation accuracy of the models which contain the partial model.

where the validation accuracy of model m is denoted by Acc(m).

We estimate this value by uniformly sampling valid models and computing the average of their validation accuracy using one mini-batch.

We use S to denote the evaluation number, which is the total number of sampled models.

We observe that when S is large enough, the potential of a partial model is fairly stable and discriminative among candidates.

See Algorithm 1 for pseudocode.

The latency constraint is imposed by discarding invalid full models when calculating potentials of partial models in the pool.

Training based on partial model pool The training iteration of the supergraph along with the partial model pool has two steps.

First, for a partial model from the pool, we randomly sample the missing operator {o l+1 , o l+2 , ..., o L } to complement the partial model to a full model.

Then we optimize θ using the sampled full model and mini-batch data.

Initially, the partial model pool is empty.

Thus the supergraph is trained by uniformly sampled models, which is identical to previous one-shot training stage.

After the initial training, all operators in the first mixop are evaluated.

The top P operators form the partial model pool in the second training stage.

Then, the supergraph resumes training and the training procedure is identical to the one discussed in last paragraph.

Inspired by warm-up, the first stage is set much more epochs than following stages denoted as T w .

The whole PC-NAS process is elaborated in algorithm 2.

The number of models in shrunk search space at the interval l is strictly less than interval l − 1.

At the final interval, the number of cross-entropy terms in (10) are P-1 for each architectures in final pool.

Thus the parameter posterior of PC-NAS would move towards the true posterior during these intervals.

Inputs: P (size of partial model pool), G(supergraph), O i (the ith operator in mixed operator), L(num of mixed operators in G), T w (warm-up epochs), T i (interval between updation of partial model pool), D train (train set), D val (validataion set), Lat(latency constraint) PartialModels = ∅ Warm-up(G, D train , T w ) uniformly sample models from G and train

We demonstrate the effectiveness of our methods on ImageNet, a large scale benchmark dataset, which contains 1,000,000 training samples with 1000 classes.

For this task, we focus on models that have high accuracy under certain GPU latency constraint.

We search models using PC-NAS, which progressively updates a partial model pool and trains shared weights.

Then, we select the model with the highest potential in the pool and report its performance on the test set after training from scratch.

Finally, we investigate the transferability of the model learned on ImageNet by evaluating it on two tasks, object detection and person re-identification.

Dataset and latency measurement: As a common practice, we randomly sample 50,000 images from the train set to form a validation set during the model search.

We conduct our PC-NAS on the remaining images in train set.

The original validation set is used as test set to report the performance of the model generated by our method.

The latency is evaluated on Nvidia GTX 1080Ti and the batch size is set 16 to fully utilize GPU resources.

Search spaces: We use two search spaces.

We benchmark our small space similar to ProxylessNAS (Cai et al., 2018) and FBNet for fair comparison.

To test our PC-NAS method in a more complicated search space, we add 3 more kinds of operators to the small space's mixoperators to construct our large space.

Details of the two spaces are in A.1.

We use PC-NAS to search in both small and large space.

To balance training time and performance, we set evaluation number S = 900 and partial model pool size P = 5 in both experiments.

Ablation study of the two values is in 4.4.

When updating weights of the supergraph, we adopt mini-batch nesterov SGD optimizer with momentum 0.9, cosine learning rate decay from 0.1 to 5e-4, batch size 512, and L2 regularization with weight 1e-4.

The warmup epochs T w and shrinking interval T i are set 100 and 5, thus the total training of supergraph lasts 100+20×5 = 200 epochs.

After searching, we select the best one from the top 5 final partial models and train it from scratch.

Similar to EfficientNet and MixNet, We add squeeze-and-excitation (SE) layers (Hu et al., 2018) Table 1 shows the performance of our model on ImageNet.

We set our target latency at 10ms according to our measurement of mobile setting models on GPU.

Our search result in the small space, namely PC-NAS-S, achieves 76.8% top-1 accuracy under our latency constraint, which is 0.5% higher than EffcientNet-B0 (in terms of absolute accuracy improvement), 1% higher than MixNet-S. If we slightly relax the time constraint, our search result from the large space, namly PC-NAS-L, achieves 78.1% top-1 accuracy, which improves top-1 accuracy by 1.8% compared to EfficientNet-B0, 2.3% compared to MixNet-S. Both PC-NAS-S and PC-NAS-L are faster than EffcientNet-b0 and MixNet-S. We validate our PC-NAS's transferability on two tasks, object detection and person re-identification.

We use COCO (Lin et al., 2014 ) dataset as benchmark for object detection and Market-1501 (Zheng et al., 2015) for person re-identification.

For the two dataset, PC-NAS-L pretrained on ImageNet is utilized as feature extractor, and is compared with other models under the same training script.

For object detection, the experiment is conducted with the two-stage framework FPN (Lin et al., 2017) .

Table 2 shows the performance of our PC-NAS model on COCO and Market-1501.

For COCO, our approach significantly surpasses the mAP of MobileNetV2 as well as ResNet50.

Compare to the standard ResNet101 backbone, our model achieves comparable mAP quality with almost 1/3 parameters and 2.3× faster speed.

Similar phenomena are found on Market-1501.

In this section, we further study the impact of hyperparameters on our method within our small space on ImageNet.

The hyperparameters include warm-up, training epochs Tw, partial model pool size P , and evaluation number S. We tried setting Tw as 100 and 150 with fixed P = 5 and S = 900.

The resulted models of these two settings show no significant difference in top-1 accuracy (less than 0.1%), shown as in Fig. 2a .

Thus we choose warm-up training epochs as 100 in our experiment to save computation resources.

For the influence of P and S, we show the results in Fig. 2a .

It can be seen that the top-1 accuracy of the models found by PC-NAS increases with both P and S. Thus we choose P = 5, S = 900 in the experiments for better performance.

we did not observe significant improvement when further increasing these two hyperparameters.

Effectiveness of shrinking search space: To assess the role of space shrinking, we train the supergraph of our large space using One-Shot (Brock et al., 2017b) method without any shrinking of the search space.

Then we conduct model search on this supergraph by progressively updating a partial model pool in our method.

The resulted model using this setting attains 77.1% top-1 accuracy on ImageNet, which is 1% lower than our PC-NAS-L. We further double the value of P or S, and the results are listed in Table 3 .

From these results, we could conclude that our search method alone already lead to impressive performance.

And our space shrinking training strategy improves model ranking and brings extra 0.7%-1% accuracy boost.

We add another comparison as follows.

First, we select a batch of models from the candidates of our final pool under small space and evaluate their stand alone top-1 accuracy.

Then we use One-Shot to train the supergraph also under small space without shrinking.

Finally, we shows the model rankings of PC-NAS and One-Shot using the accuracy obtained from inferring the models in the supergraphs trained with the two methods.

The difference is shown in Fig. 2b , the pearson correlation coefficients between stand-alone accuracy and accuracy in supergraph of One-Shot and PC-NAS are 0.11 and 0.92, thus models under PC-NAS's space shrinking can be ranked by their accuracy evaluated on sharing weights much more precisely than One-Shot.

Effectiveness of our search method: To investigate the importance of our search method, we utilize Evolution Algorithm (EA) to search for models with the above supergraph of our large space trained with One-Shot.

The top-1 accuracy of discovered model drops furthur to 75.9% accuracy, which is 2.2% lower than PC-NAS-L .

We implement EA with population size 5, aligned to the value of pool size P in our method, and set the mutation operation as randomly replace the operator in one mixop operator to another.

We constrain the total number of validation images in EA the same as ours.

The results are also shown in Table 3 .

In this paper, a new architecture search approach called PC-NAS is proposed.

We study the conventional weight sharing approach from Bayesian point of view and identify a key issue that compromises the effectiveness of shared weights.

With the theoretical motivation, a practical method The operators in our spaces have structures described by either Conv1x1-ConvNxM-Conv1x1 or Conv1x1-ConvNxM-ConvMxN-Conv1x1.

We define expand ratio as the ratio between the channel numbers of the ConvNxM in the middle and the input of the first Conv1x1.

Small search space Our small search space contains a set of MBConv operators (mobile inverted bottleneck convolution (Sandler et al., 2018) ) with different kernel sizes and expand ratios, plus Identity, adding up to 10 operators to form a mixoperator.

The 10 operators in our small search space are listed in the left column of Table 5 , where notation OP X Y represents the specific operator OP with expand ratio X and kernel size Y.

Large search space We add 3 more kinds of operators to the mixoperators of our large search space, namely NConv, DConv, and RConv.

We use these 3 operators with different kernel sizes and expand ratios to form 10 operators exclusively for large space, thus the large space contains 20 operators.

For large search space, the structure of NConv, DConv are Conv1x1-ConvKxK-Conv1x1 and Conv1x1-ConvKxK-ConvKxK-Conv1x1, and that of RConv is Conv1x1-Conv1xK-ConvKx1-Conv1x1.

The kernel sizes and expand ratios of operators exclusively for large space are lised in the right column of Table 5 , where notation OP X Y represents the specific operator OP with expand ratio X and K=Y.

There are altogether 21 mixoperators in both small and large search spaces.

Thus our small search space contains 10 21 models, while the large one contains 20 21 .

The specifications of PC-NAS-S and PC-NAS-L are shown in Fig. 3 .

We observe that PC-NAS-S adopts either high expansion rate or large kernel size at the tail end, which enables a full use of high level features.

However, it tends to select small kernels and low expansion rates to ensure the model remains lightweight.

PC-NAS-L chooses lots of powerful bottlenecks exclusively contained in the large space to achieve the accuracy boost.

The high expansion rate is not quite frequently seen which is to compensate the computation utilized by large kernel size.

Both PC-NAS-S and PC-NAS-L tend to use heavy operator when the resolution reduces, circumventing too much information loss in these positions.

224×224×

3  112×112×16  112×112×

16  56×56×

32  56×56×

32  56×56×

32  56×56×

32  28×28×64  28×28×64  28×28×64  28×28×64  14×14×136  14×14×136  14×14×136  14×14×136  14×14×136  14×14×136  14×14×136  14×14×136  7×7×264  7×7×264  7×7×264

<|TLDR|>

@highlight

Our paper identifies the issue of existing weight sharing approach in neural architecture search and propose a practical method, achieving strong results.

@highlight

Author identifies an issue with NAS called posterior fading and introduces Posterior Convergent NAS to mitigate this effect