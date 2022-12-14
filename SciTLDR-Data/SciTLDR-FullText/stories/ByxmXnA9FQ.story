With the recently rapid development in deep learning, deep neural networks have been widely adopted in many real-life applications.

However, deep neural networks are also known to have very little control over its uncertainty for test examples, which potentially causes very harmful and annoying consequences in practical scenarios.

In this paper, we are particularly interested in designing a higher-order uncertainty metric for deep neural networks and investigate its performance on the out-of-distribution detection task proposed by~\cite{hendrycks2016baseline}. Our method first assumes there exists a underlying higher-order distribution $\mathcal{P}(z)$, which generated label-wise distribution $\mathcal{P}(y)$ over classes on the K-dimension simplex, and then approximate such higher-order distribution via parameterized posterior function $p_{\theta}(z|x)$ under variational inference framework, finally we use the entropy of learned posterior distribution $p_{\theta}(z|x)$ as uncertainty measure to detect out-of-distribution examples.

However, we identify the overwhelming over-concentration issue in such a framework, which greatly hinders the detection performance.

Therefore, we further design a log-smoothing function to alleviate such issue to greatly increase the robustness of the proposed entropy-based uncertainty measure.

Through comprehensive experiments on various datasets and architectures, our proposed variational Dirichlet framework with entropy-based uncertainty measure is consistently observed to yield significant improvements over many baseline systems.

Recently, deep neural networks BID18 have surged and replaced the traditional machine learning algorithms to demonstrate its potentials in many real-life applications like speech recognition BID10 , image classification BID11 , and machine translation BID34 BID32 , reading comprehension BID27 , etc.

However, unlike the traditional machine learning algorithms like Gaussian Process, Logistic Regression, etc, deep neural networks are very limited in their capability to measure their uncertainty over the unseen test cases and tend to produce over-confident predictions.

Such overconfidence issue BID1 ) is known to be harmful or offensive in real-life applications.

Even worse, such models are prone to adversarial attacks and raise concerns in AI safety BID9 BID23 .

Therefore, it is very essential to design a robust and accurate uncertainty metric in deep neural networks in order to better deploy them into real-world applications.

Recently, An out-of-distribution detection task has been proposed in BID12 as a benchmark to promote the uncertainty research in the deep learning community.

In the baseline approach, a simple method using the highest softmax score is adopted as the indicator for the model's confidence to distinguish in-from out-ofdistribution data.

Later on, many follow-up algorithms BID21 BID19 BID30 BID4 have been proposed to achieve better performance on this benchmark.

In ODIN BID21 , the authors follow the idea of temperature scaling and input perturbation BID25 to widen the distance between in-and out-of-distribution examples.

Later on, adversarial training BID19 ) is introduced to explicitly introduce boundary examples as negative training data to help increase the model's robustness.

In BID4 , the authors proposed to directly output a real value between [0, 1] as the confidence measure.

The most recent paper BID30 leverages the semantic dense representation into the target labels to better separate the label space and uses the cosine similarity score as the confidence measure.

These methods though achieve significant results on out-of-distribution detection tasks, they conflate different levels of uncertainty as pointed in BID22 .

For example, when presented with two pictures, one is faked by mixing dog, cat and horse pictures, the other is a real but unseen dog, the model might output same belief as {cat:34%, dog:33%, horse:33%}. Under such scenario, the existing measures like maximum probability or label-level entropy BID21 BID30 BID12 will misclassify both images as from out-of-distribution because they are unable to separate the two uncertainty sources: whether the uncertainty is due to the data noise (class overlap) or whether the data is far from the manifold of training data.

More specifically, they fail to distinguish between the lower-order (aleatoric) uncertainty BID5 , and higherorder (episdemic) uncertainty BID5 , which leads to their inferior performances in detecting out-domain examples.

In order to resolve the issues presented by lower-order uncertainty measures, we are motivated to design an effective higher-order uncertainty measure for out-of-distribution detection.

Inspired by Subjective Logic BID14 BID37 BID29 , we first view the label-wise distribution P(y) as a K-dimensional variable z generated from a higher-order distribution P(z) over the simplex S k , and then study the higher-order uncertainty by investigating the statistical properties of such underlying higher-order distribution.

Under a Bayesian framework with data pair D = (x, y), we propose to use variational inference to approximate such "true" latent distribution P(z) = p(z|y) by a parameterized Dirichlet posterior p ?? (z|x), which is approximated by a deep neural network.

Finally, we compute the entropy of the approximated posterior for outof-distribution detection.

However, we have observed an overwhelming over-concentration problem in our experiments, which is caused by over-confidence problem of the deep neural network to greatly hinder the detection accuracy.

Therefore, we further propose to smooth the Dirichlet distribution by a calibration algorithm.

Combined with the input perturbation method BID21 BID16 , our proposed variational Dirichlet framework can greatly widen the distance between in-and out-of-distribution data to achieve significant results on various datasets and architectures.

The contributions of this paper are described as follows:??? We propose a variational Dirichlet algorithm for deep neural network classification problem and define a higher-order uncertainty measure.??? We identify the over-concentration issue in our Dirichlet framework and propose a smoothing method to alleviate such problem.

In this paper, we particularly consider the image classification problem with image input as x and output label as y. By viewing the label-level distribution DISPLAYFORM0 lying on a K-dimensional simplex S k , we assume there exists an underlying higher-order distribution P(z) over such variable z. As depicted in FIG0 , each point from the simplex S k is itself a categorical distribution P(y) over different classes.

The high-order distribution P(z) is described by the probability over such simplex S k to depict the underlying generation function.

By studying the statistical properties of such higher-order distribution P(z), we can quantitatively analyze its higher-order uncertainty by using entropy, mutual information, etc.

Here we consider a Bayesian inference framework with a given dataset D containing data pairs (x, y) and show the plate notation in FIG1 , where x denotes the observed input data (images), y is the groundtruth label (known at training but unknown as testing), and z is latent variable higher-order variable.

We assume that the "true" posterior distribution is encapsulated in the partially observable groundtruth label y, thus it can be viewed as P(z) = p(z|y).

During test time, due to the inaccessibility of y, we need to approximate such "true" distribution with the given input image x.

Therefore, we propose to parameterize a posterior model p ?? (z|x) and optimize its parameters to approach such "true" posterior p(z|y) given a pairwise input (x, y) by minimizing their KL-divergence D KL (p ?? (z|x)||p(z|y)).

With the parameterized posterior p ?? (z|x), we are able to infer the higher-order distribution over z given an unseen image x * and quantitatively study its statistical properties to estimate the higher-order uncertainty.

In order to minimize the KL-divergence D KL (p ?? (z|x)||p(z|y)), we leverage the variational inference framework to decompose it into two components as follows (details in appendix): DISPLAYFORM1 where L(??) is better known as the variational evidence lower bound, and log p(y) is the marginal likelihood over the label y. DISPLAYFORM2 Since the marginal distribution p(y) is constant w.r.t ??, minimizing the KL-divergence D KL (p ?? (z|x)||p(z|y)) is equivalent to maximizing the evidence lower bound L(??).

Here we propose to use Dirichlet family to realize the higher-order distribution p ?? (z|x) = Dir(z|??) due to its tractable analytical properties.

The probability density function of Dirichlet distribution over all possible values of the K-dimensional stochastic variable z can be written as: DISPLAYFORM3 where ?? is the concentration parameter of the Dirichlet distribution and DISPLAYFORM4 is the normalization factor.

Since the LHS (expectation of log probability) has a closed-formed solution, we rewrite the empirical lower bound on given dataset D as follows: DISPLAYFORM5 where ?? 0 is the sum of concentration parameter ?? over K dimensions.

However, it is in general difficult to select a perfect model prior to craft a model posterior which induces an the distribution with the desired properties.

Here, we assume the prior distribution is as Dirichlet distribution Dir(??) with concentration parameters?? and specifically talk about three intuitive prior functions in FIG2 .

The first uniform prior aggressively pushes all dimensions towards 1, while the *-preserving priors are less strict by allowing one dimension of freedom in the posterior concentration parameter ??.

This is realized by copying the value from k th dimension of posterior concentration parameter ?? to the uniform concentration to unbind ?? k from KL-divergence computation.

Given the prior concentration parameter??, we can obtain a closed-form solution for the evidence lower bound as follows: DISPLAYFORM6 ?? denotes the gamma function, ?? denotes the digamma function.

We write the derivative of L(??) w.r.t to parameters ?? based on the chain-rule: DISPLAYFORM7 ????? , where is the Hardamard product and DISPLAYFORM8 is the Jacobian matrix.

In practice, we parameterize Dir(z|??) via a neural network with ?? = f ?? (x) and re-weigh the two terms in L(??) with a balancing factor ??.

Finally, we propose to use mini-batch gradient descent to optimize the network parameters ?? as follows: DISPLAYFORM9 where B(x, y) denotes the mini-batch in dataset D. During inference time, we use the marginal probability of assigning given input x to certain class label i as the classification evidence: DISPLAYFORM10 Therefore, we can use the maximum ??'s index as the model prediction class during inference?? = arg max i p(y = i|x) = arg max i ?? i .

After optimization, we obtain a parametric Dirichlet function p ?? (z|??) and compute its entropy E as the higher-order uncertainty measure.

Formally, we write the such metric as follows: DISPLAYFORM0 where ?? is computed via the deep neural network f ?? (x).

Here we use negative of entropy as the confidence score C(??).

By investigating the magnitude distribution of concentration parameter ?? for in-distribution test cases, we can see that ?? is either adopting the prior ?? = 1.0 or adopting a very large value ?? 1.0.

In order words, the Dirichlet distribution is heavily concentrated at a corner of the simplex regardless of whether the inputs are from out-domain region, which makes the model very sensitive to out-of-distribution examples leading to compromised detection accuracy.

In order to resolve such issue, we propose to generally decrease model's confidence by smoothing the concentration parameters ??, the smoothing function can lead to opposite behaviors in the uncertainty estimation of in-and out-of-distribution data to enlarge their margin.

Concentration smoothing In order to construct such a smoothing function, we experimented with several candidates and found that the log-smoothing function?? = log(?? + 1) can achieve generally promising results.

By plotting the histogram of concentration magnitude before and after log-scaling in FIG3 , we can observe a very strong effect in decreasing model's overconfidence, which in turn leads to clearer separation between in-and out-of-distribution examples (depicted in FIG3 ).

In the experimental section, we detail the comparison of different smoothing functions to discuss its impact on the detection accuracy.

Input Perturbation Inspired by fast gradient sign method BID9 , we propose to add perturbation in the data before feeding into neural networks: DISPLAYFORM1 where the parameter denotes the magnitude of the perturbation, and (x, y) denotes the input-label data pair.

Here, similar to BID21 our goal is also to improve the entropy score of any given input by adding belief to its own prediction.

Here we make a more practical assumption that we have no access to any form of out-of-distribution data.

Therefore, we stick to a rule-of-thumb value = 0.01 throughout our experiments.

Table 1 : Classification accuracy of Dirichlet framework on various datasets and architectures.

Detection For each input x, we first use input perturbation to obtainx, then we feed it into neural network f ?? (x) to compute the concentration ??, finally we use log-scaling to calibrate ?? and compute C(??).

Specifically, we compare the confidence C(??) to the threshold ?? and say that the data x follows in-distribution if the confidence score C(??) is above the threshold and that the data x follows out-of-distribution, otherwise.

In order to evaluate our variational Dirichlet method on out-of-distribution detection, we follow the previous paper BID12 BID21 to replicate their experimental setup.

Throughout our experiments, a neural network is trained on some in-distribution datasets to distinguish against the out-of-distribution examples represented by images from a variety of unrelated datasets.

For each sample fed into the neural network, we will calculate the Dirichlet entropy based on the output concentration ??, which will be used to predict which distribution the samples come from.

Finally, several different evaluation metrics are used to measure and compare how well different detection methods can separate the two distributions.

These datasets are all available in Github 1 .??? In-distribution: CIFAR10/100 BID16 ) and SVHN BID24 , which are both comprised of RGB images of 32 ?? 32 pixels.??? Out-of-distribution: TinyImageNet , LSUN BID38 and iSUN BID35 , these images are resized to 32 ?? 32 pixels to match the in-distribution images.

Before reporting the out-of-distribution detection results, we first measure the classification accuracy of our proposed method on the two in-distribution datasets in Table 1 , from which we can observe that our proposed algorithm has minimum impact on the classification accuracy.

In order to make fair comparisons with other out-of-distribution detectors, we follow the same setting of BID21 ; BID39 ; DeVries & Taylor (2018); BID30 to separately train WideResNet BID39 ) (depth=16 and widening factor=8 for SVHN, depth=28 and widening factor=10 for CIFAR100), VGG13 BID31 , and ResNet18 BID11 ) models on the in-distribution datasets.

All models are trained using stochastic gradient descent with Nesterov momentum of 0.9, and weight decay with 5e-4.

We train all models for 200 epochs with 128 batch size.

We initialize the learning with 0.1 and reduced by a factor of 5 at 60th, 120th and 180th epochs.

we cut off the gradient norm by 1 to prevent from potential gradient exploding error.

We save the model after the classification accuracy on validation set converges and use the saved model for out-of-distribution detection.

We measure the quality of out-of-distribution detection using the established metrics for this task BID12 BID21 BID30 .

FORMULA1 BID5 .We report our VGG13's performance in TAB2 and ResNet/WideResNet's performance in Table 3 under groundtruth-preserving prior, where we list the performance of Baseline BID12 , ODIN Liang et al. (2017) , Bayesian Neural Network BID5 2 , SemanticRepresentation BID30 and Learning-Confidence (DeVries & Taylor, 2018) .

The results in both tables have shown remarkable improvements brought by our proposed variational Dirichlet framework.

For CIFAR datasets, the achieved improvements are very remarkable, however, the FPR score on CIFAR100 is still unsatisfactory with nearly half of the out-of-distribution samples being Table 3 : Experimental results for ResNet architecture, where Semantic refers to multiple semantic representation algorithm BID30 wrongly detected.

For the simple SVHN dataset, the current algorithms already achieve close-toperfect results, therefore, the improvements brought by our algorithm is comparatively minor.

In order to individually study the effectiveness of our proposed methods (entropy-based uncertainty measure, concentration smoothing, and input perturbation), we design a series of ablation experiments in FIG5 .

From which, we could observe that concentration smoothing has a similar influence as input perturbation, the best performance is achieved when combining these two methods.

Here we mainly experiment with four different priors and depict our observations in FIG6 .

From which, we can observe that the non-informative uniform prior is too strong assumption in terms of regularization, thus leads to inferior detection performances.

In comparison, giving model one dimension of freedom is a looser assumption, which leads to generally better detection accuracy.

Among these two priors, we found that preserving the groundtruth information can generally achieve slightly better performance, which is used through our experiments.

We also investigate the impact of different smoothing functions on the out-of-distribution detection accuracy.

For smoothing functions, we mainly consider the following function forms: tection accuracy, and use x, x 2 as baselines to investigate the impact of compression capability of smoothing function on the detection accuracy.

From FIG7 , we can observe that the first three smoothing functions greatly outperforms the baseline functions.

Therefore, we can conclude that the smoothing function should adopt two important characteristics: 1) the smoothing function should not be bounded, i.e. the range should be [0, ???] .

2) the large values should be compressed.

DISPLAYFORM0

The novelty detection problem BID26 has already a long-standing research topic in traditional machine learning community, the previous works BID33 BID8 BID28 have been mainly focused on low-dimensional and specific tasks.

Their methods are known to be unreliable in high-dimensional space.

Recently, more research works about detecting an anomaly in deep learning like BID0 and BID19 , which propose to leverage adversarial training for detecting abnormal instances.

In order to make the deep model more robust to abnormal instances, different approaches like BID2 ; ; ; BID17 have been proposed to increase deep model's robustness against outliers during training.

Another line of research is Bayesian Networks BID7 BID5 BID15 , which are powerful in providing stochasticity in deep neural networks by assuming the weights are stochastic.

However, Bayesian Neural Networks' uncertainty measure like variational ratio and mutual information rely on Monte-Carlo estimation, where the networks have to perform forward passes many times, which greatly reduces the detection speed.

In this paper, we aim at finding an effective way for deep neural networks to express their uncertainty over their output distribution.

Our variational Dirichlet framework is empirically demonstrated to yield better results, but its detection accuracy on a more challenging setup like CIFAR100 is still very compromised.

We conjecture that better prior Dirichlet distribution or smoothing function could help further improve the performance.

In the future work, we plan to apply our method to broader applications like natural language processing tasks or speech recognition tasks.??? LSUN (out-of-distribution): The Large-scale Scene UNderstanding dataset (LSUN) BID38 has a test set consisting of 10,000 images from 10 different scene classes, such as bedroom, church, kitchen, and tower.

We downsample LSUN's original image and create 32 ?? 32 images as an out-of-distribution dataset.??? iSUN (out-of-distribution): The iSUN dataset BID35 ) is a subset of the SUN dataset, containing 8,925 images.

All images are downsampled to 32 ?? 32 pixels.

Here we investigate the impact of KL-divergence in terms of both classification accuracy and detection errors.

By gradually increasing the weight of KL loss (increasing the balancing factor ?? from 0 to 10), we plot their training loss curve in FIG8 .

With a too strong KL regularization, the model's classification accuracy will decrease significantly.

As long as ?? is within a rational range, the classification accuracy will become stable.

For detection error, we can see from FIG8 that adopting either too large value or too small value can lead to compromised performance.

For the very small value ?? ??? 0, the variational Dirichlet framework degrades into a marginal log-likelihood, where the concentration parameters are becoming very erratic and untrustworthy uncertainty measure without any regularization.

For larger ?? > 1, the too strong regularization will force both in-and outof-distribution samples too close to prior distribution, thus erasing the difference between in-and out-of-distribution becomes and leading to worse detection performance.

We find that adopting a hyper-parameter of ?? = 0.01 can balance the stability and detection accuracy.

Here we particularly investigate the out-of-distribution detection results of our model on CIFAR100 under different scenarios.

Our experimental results are listed in TAB5 : Experimental results for ResNet architecture, where Semantic refers to multiple semantic representation algorithm BID30

@highlight

A new framework based variational inference for out-of-distribution detection

@highlight

Describes a probabilistic approach to quantifying uncertainty in DNN classification tasks that outperforms other SOTA methods in the task of out-of-distribution detection.

@highlight

A new framework for out-of-distribution detection, based on variaitonal inference and a prior Dirichlet distribution, that reports state of the art results on several datasets.

@highlight

An out-of distribution detection via a new method to approximate the confidence distribution of classification probability using variational inference of Dirichlet distribution.