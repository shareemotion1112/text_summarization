Attention mechanisms have advanced the state of the art in several machine learning tasks.

Despite significant empirical gains, there is a lack of theoretical analyses on understanding their effectiveness.

In this paper, we address this problem by studying the landscape of population and empirical loss functions of attention-based neural networks.

Our results show that, under mild assumptions, every local minimum of a two-layer global attention model has low prediction error, and attention models require lower sample complexity than models not employing attention.

We then extend our analyses to the popular self-attention model, proving that they deliver consistent predictions with a more expressive class of functions.

Additionally, our theoretical results provide several guidelines for designing attention mechanisms.

Our findings are validated with satisfactory experimental results on MNIST and IMDB reviews dataset.

Significant research in machine learning has focused on designing network architectures for superior performance, faster convergence and better generalization.

Attention mechanisms are one such design choice that is widely used in many natural language processing and computer vision tasks.

Inspired by human cognition, attention mechanisms advocate focusing on the relevant regions of input data to solve a desired task rather than ingesting the entire input.

Several variants of attention mechanisms have been proposed, and they have advanced the state of the art in machine translation (Bahdanau et al., 2014; Luong et al., 2015; Vaswani et al., 2017) , image captioning (Xu et al., 2015) , video captioning (Pu et al., 2018) , visual question answering (Lu et al., 2016) , generative modeling (Zhang et al., 2018) , etc.

In computer vision, spatial/ spatio-temporal attention masks are employed to focus only on the relevant regions of images/ video frames for the underlying downstream task (Mnih et al., 2014) .

In natural language tasks, where input-output pairs are sequential data, attention mechanisms focus on the most relevant elements in the input sequence to predict each symbol of the output sequence.

Hidden state representations of a recurrent neural network are typically used to compute these attention masks.

The most popular implementation of this paradigm is self-attention (Vaswani et al., 2017) , which uses correlation among the elements of the input sequence to learn an attention mask.

Substantial empirical evidence demonstrating the effectiveness of attention mechanisms motivates us to study the problem from a theoretical lens.

In this work, we attempt to understand the loss landscape of neural networks employing attention.

Analyzing the loss landscape and optimization of neural networks is an open area of research, and is a challenging problem even for two-layer neural networks (Poggio & Liao, 2017; Rister & Rubin, 2017; Soudry & Hoffer, 2018; Zhou & Feng, 2017; Mei et al., 2018b; Soltanolkotabi et al., 2017; Ge et al., 2017; Nguyen & Hein, 2017a; Arora et al., 2018) .

Convergence of gradient descent for two-layer neural networks has been studied in Mei et al., 2018b; Du et al., 2019) .

Ge et al. (2017) shows that there is no bad local minima for two-layer neural nets under a specific loss landscape design.

Unfortunately, these results cannot directly be applied to attention mechanisms, as attention modifies the network structure and introduces additional parameters which are jointly optimized with the model.

To the best of our knowledge, our work presents the first theoretical analysis on attention-based models.

Our main result shows that, under some mild conditions, every stationary point of attention models achieve a low prediction error.

We perform an asymptotic analysis where we show that expected prediction on error goes to 0 as n ??? ???. We also show that attention models achieve lower sample complexity than the models not employing attention.

We then discuss how the result can be extended to recurrent attention and multi layer cases, and discuss the effect of regularization.

In addition, we show how attention further helps improve the loss landscape by studying three properties: number of linear regions, flatness of local minima and small sample size training.

We validate our theoretical results with experiments on MNIST and IMDB reviews dataset.

Attention mechanisms are modules that help neural networks focus only on the relevant regions of input data to make predictions.

To study such behavior, we analyze different types of attention models.

We start with a naive global attention model.

Then we analyze the most popular self-attention model and discuss the extension to recurrent attention in Appendix.

In the naive global attention model, we consider a dataset D = {x i , y i } N i=1 , x i ??? R p , y i ??? R, where the output y i depends only on certain regions of input x i , i.e., y i = f (a x i ), where a is an attention mask, and f (.) is the ground-truth function that is used to generate the dataset and the vector a ??? [0, 1] p .

The set of entries {a i |a i = 0} corresponds to the relevant region of the input, while the complementary set {a i |a i = 0} corresponds to the irrelevant region.

We consider a two-layer ReLU-based neural network to approximate the function f * .

The network architecture consists of a linear layer followed by rectified linear units (ReLU) as a non-linearity and a second linear layer.

Denote the weights of the first layer by w

(1) , the weights of the second layer by w (2) , and the ReLU function by ??(??).

Then the response function for the input x can be written as f (x) = w (2)T ??( w (1) , x ).

We call the above function "baseline model" since it does not employ any attention.

To incorporate attention, we introduce the attention mask a as additional neural network parameters.

The attention model we use can be written as:

In this paper, we focus on the regression task which minimizes the following loss function:L = E (x,y)???D f (x) ??? y 2 2 .

While we present analysis on the regression task, our theory can easily be extended to classification tasks as well.

After a thorough analysis of global attention, we analyze a more practical self-attention setup, which comes from the transformer model proposed in Vaswani et al. (2017) .

The input x i = (x 1 i , . . . , x p i ) ??? R t??p , where x j i are t-dimensional vectors.

Each x i corresponds to independent sentences for i = 1, . . .

, n, and x j 's are the fixed dimensional vector embedding of each word in sentence x.

w Q ,w K ??? R dq??t are the query and key matrices, and w V ??? R dv??t is the value matrix.

For each input x i , the key is calculated as: K i = (w K x i ) T ??? R p??dq ; For z th vector in the input, the query vector is computed as: Q z i = (w Q x z i ) T ??? R 1??dq for z = 1, . . .

, p. The value matrix V = w V x i ??? R dv??p .

Then the self-attention w.r.t to the z th vector in the input x i is computed as:

for z = 1, . . . , p. And a

).

This self-attention vector represents the interaction between different words in each sentence.

The value vector for each word in the sentence x z i can be calculated as

dv .

This value vector is then passed to a 2-layer MLP parameterized by w

(1) ??? R pdv??d and w (2) ??? R d??1 , resulting in the following generative model:

where vec(??) represents the vectorization of a matrix.

We also discuss the extension to recurrent attention model and multi-layer self-attention model in Appendix due to the page limit.

Note that naive global attention is not widely used in practice, because here we assume the attention mask is globally fixed for all points, but not a function of input.

In real world application, the attention weights depend on the input, such as the self-attention framework we just introduced.

Despite the limitation of naive global attention model, it is a fundamental building block of attention models, and needed to be analyzed first for two following reasons.

First, this fixed attention shares the core idea of attention: There is a specific intrinsic structure in data, and this intrinsic structure requires that we should assign different weights to different input features accordingly.

And this weight assigning strategy should be learned from data.

In naive global attention, attention weights a are parameters themselves; In the self-attention model, the attention weights a depend on a function parameterized by value/key/query matrices, and we learn these matrices as attention parameters.

And for both global and self-attention, we jointly learn the network weights(w (1) ,w (2) ) and attention parameters(a in global attention, value/key/query matrices in self-attention) at the same time.

Therefore this naive global attention is a good starting point for analyzing attention mechanisms.

Second, we can gain helpful insights on attention by analyzing the global attention case.

The number of non-zero elements of a in global attention represents both the size of attention parameters and the sparsity level of attention.

In the standard self-attention model, size of attention parameters is determined by the size of value/key/query matrices.

And the sparsity level is how many words we allow one word to attend to.

By studying the effect of this quantity, we can have a better understanding of how the sparsity and parameter size of attention affect the model performance and sample complexity.

The detailed discussions can be found in Section 3.

In Section 3.1, we analyze the loss landscape for the the naive global attention model, which is defined as

where

, and S is the parameter space of a. Section 3.2 extends the loss landscape analysis to the self-attention model with a = f (x).

Section 3.3 discusses the extension to recurrent attention model.

Section 3.4 discusses the effect of regularization and how the result can be extended to multi-layer networks.

To approximate the non-differentiable ReLU ??, we use the softplus activation function ?? ?? (x), i.e.,

Note ?? ?? converges to ReLU as ?? ??? ??? (Glorot et al., 2011) .

Theoretical results with ?? ?? (x) will hold for any arbitrarily large ?? .

For ease of notation, we still use ??(x) to denote the softplus function.

We first analyze the asymptotic prediction error of local minimum with large sample size.

Let the covariance matrix of ??(w

, where x ??k represents k th feature in x, and a k is the attention mask for x ??k .

?? ( w (1)T , x a is the first order derivative with respect to the value in ??(??) and it belongs to

.

Before proceeding, we introduce several necessary assumptions:

(A3) The output y can be specified by the two-layer neural network up to an independent subGaussian error with variance ?? 2 , i.e., there exists a set of parameter (a , w (1) , w (2) ), such that

(A4) a 0 ??? s 0 such that s 0 ??? p, which represents the sparsity of the attention model, and 0 ??? a i ??? 1 for any i = 1, ?? ?? ?? , p.

(A5) ?? min (?? ?? ) ??? C ?? , and when the estimation of ?? is inaccurate, i.e, E( ??(

, then there exists a feature x ??k and for the t th element of h t (x ??k ), it satisfies sd(h t (x ??k )) = O(1) and cor(u, h t (x ??k )) = O(1) for w (2) in the form of ??

(A7) The sum of the weights a 1 = 1.

These assumptions are mild when the dimension of x is large.

The justification of these assumptions are provided in the appendices.

Given these assumptions, we show in Theorem 1 that the sample complexity required to converge to a good minimum is reduced with attention mechanism.

Theorem 1.

Under (A1) to (A6), for any ?? > 0, suppose

where ?? = C 1 C 2 C x .

Then with probability tending to 1, any stationary point (??,w (1) ,w (2) ) of the objective function (4) satisfies the following prediction error bound:

The sample complexity bound of Theorem 3 provides helpful insight for understanding attention mechanisms.

With the sparsity structure of attention mask a, attention mechanisms constrain the parameters in a smaller space, thus reducing the variance and the covering number.

This leads to lower sample complexity compared to the baseline model not employing attention.

It is straightforward to calculate the sample complexity bound for the baseline model.

To achieve the same error bound, we substitute s 0 with p in the bound, and this results in a much larger value.

When s 0 is fixed, we can see up to a log term, prediction error ?? is proportion to n ???1/2 , which is the optimal rate of convergence in regression.

This shows that the bound is tight in this aspect.

Next, we extend Theorem 1 to the attention model with an additional sum-to-one constraint (A7).

The discussion of the following Corollary 1 is provided in the appendix.

Then with probability tending to 1, any stationary point (??,w

(1) ,w (2) ) of the objective function (4) satisfies the following prediction error bound:

In this section, we extend our previous analysis to the self-attention model.

In self-attention, the attention mask is no more fixed globally, but instead a function of the input.

We begin by analyzing a self-attention model with a known attention function f (??), in which the weight a is not optimized together with w. Similar bound can be derived for the following model:

, where ?? = C 1 C 2 C x , with probability converging to 1, any stationary point (w (1) ,w (2) ) of the objective function equation 5 satisfies that:

Proposition 1 implies that if the self-attention mask can be precisely computed, global attention results can be extended to self-attention ones.

However, the function f (??) is not necessary known, and needs to be learnt in real world applications.

Therefore the self-attention setup as we introduced in section 2 is more desired in real-world setting.

Denoting w = (w (1) , w (2) , w Q , w K , w V ), the two-layer self-attention model can be estimated by:

We now introduce necessary assumptions for analyzing self-attention model.

(A8) There exist C 5 ,C 6 and C 7 such that w

(A9) The output y can be predicted by the two-layer network (3) with an independent sub-Gaussian error with variance ?? 2 , i.e, there exists a set of parameters (a ,

(A10) We assume (A5) and (A6) holds, substituting x i a with vec(w

)) ??k is the k-th element of the value matrices.

The assumption (A9) states that self-attention model can correctly predict the conditional mean E(y i |x i ).

Note that (A9) encompasses a more expressive class of models than (A3), which includes the models used in practice such as the transformers. (A10) is parallel to (A5) and (A6).

Under these assumptions, we can obtain its sample complexity as given by following theorem: Theorem 2.

Under (A1), (A2), and (A8) to (A10), for any ?? > 0, given the sample size:

where ?? = C 1 C 2 C x , with probability tending to 1, any stationary point (w

Remark: Theorem 2 shows that with the help of self-attention, we can achieve consistent prediction under more expressive class of models (assumption (A9)) which considers the interactions between vectors in data.

It is worth pointing out that both global attention model and baseline model do not have consistency for the class of models beyond the ones stated in (A3).

In other words, consistent prediction on the data distribution generated from equation 3 using baseline and global attention models requires introducing larger parameter space, for example, using more layers of network or more units in each layer.

Self-attention model, on the other hand, achieves the more accurate estimation by constraining the parameter space and input space.

And parallel to the sparsity level s 0 in Theorem 1, a proper choice of value/query/key matrices can help reduce sample complexity.

If a sparse attention(i.e.

one word should attend to all words, but only some relevant words), the sample complexity can also be further reduced similar with Theorem 1.

Sample complexity analysis can be extended to recurrent neural networks.

This is included in the appendix.

The key messages from our analysis include: (1) A good design of recurrent framework can help the network converge to a good stationary point with small sample complexity, and (2) An arbitrarily complex framework increases the sample complexity.

Since in real world, the optimal recurrent framework is unknown, careful design choice has to be made for obtaining good sample complexity.

3.4 DISCUSSION: REGULARIZATION AND BEYOND 2 LAYERS So far our analyses provide some theoretical justification on how attention mechanisms help learn superior models.

Furthermore, our analysis also suggests proper regularization is helpful in training an attention model.

An 1 regularization on attention weights and 2 regularization on network weights are effective in reducing the sample complexity.

We also find that imposing constraints and regularization on network weights can help remove sharp minima, and keep flat minima with good generalization.

Detailed discussions on regularization are provided in Appendix Section D.1.

Also, Theorem 1 and 2 can be extended to multi-layer attention network, under the assumptions parallel to (A8), (A9) and (A10): There exists a correct multi-layer self-attention network can specify the model, and the bias and gradient with respect to network weights are not uncorrelated.

Under these assumptions, we provide sample complexity bound for multi-layer self-attention models.

And all the discussions and insights are applied to multi-layer models.

Explicit assumptions, discussions and theorems are provided in Appendix Section D.2.

We avoid multi-layer setting in main context because it leads to over-complicated derivations and assumption justifications, and will distract readers from main idea of the paper.

We believe two-layer models are representative enough to provide theoretical evidence on why attention reduces sample complexity.

In this section, we further investigate three additional properties on how attention mechanisms improve the landscape of neural networks.

First, we show that in global attention model, attention mechanisms reduce unnecessary number of linear regions and maintain a low approximation error; Second, we show that flatness properties of minima are retained when attention mechanisms are used for both global attention and self-attention.

Furthermore, our analysis indicates that for attention models, smaller sample size suffices to converge to good minima which generalize well in prediction.

Finally, we show that the perfect in-sample prediction on small sample size is also achieved in attention networks for both global attention and self-attention.

We first study how attention mechanisms affect the number of linear regions (Montufar et al., 2014) in a wide two-layer network, where the number of units in the hidden layer is larger than the sparsity of the attention mask matrix.

Remark: The theorem implies that when appropriate attention mechanism is used, the number of linear regions reduces leading to a simpler landscape, yet the approximation error remains small.

This leads to lower sample complexity for achieving a desired prediction error.

More detailed discussion can be found in the appendices.

The result of Theorem 3 also applies to the self-attention with different attention sparsity(i.e.

allowing how many words we allow one word to attend to).

Many recent works, such as Keskar et al. (2016) , argue that flatter local minima tend to generalize well.

However, in a recent study, Dinh et al. (2017) observes that by scale transformation, the minima which are observationally equivalent, can be arbitrarily sharp, and the operator norm of a Hessian matrix can also be arbitrarily large.

We will show that this fact also holds for the global attention mechanism, if no constraint on parameter (??,w (1) ,w (2) ) is imposed.

Here we introduce the definition of -flatness as in Hochreiter & Schmidhuber (1997) .

In the following Theorem, we analyze the flatness of stationary point for both naive global and self-attention model.

Theorem 4.

(a) Consider the two-layer ReLU neural network with naive global attention in Section 3.1:

and a minimum ?? = (??,w (1) ,w (2) ) satisfying that?? = 0,w (1) = 0,w (2) = 0.

For any > 0, C(L, ??, ) has an infinite volume, and for any M > 0, we can find a stationary point such that the largest eigenvalue of ??? 2 L(??) is larger than M;

(b) Consider the two-layer ReLU neural network with self-attention mechanism as stated in Section 3.2:

and a minimum ?? = (

has an infinite volume, and for any M > 0, we can find a stationary point such that the largest eigenvalue of ??? 2 L(??) is larger than M.

Theorem 4 indicates that property on flatness of minima is maintained when attention mechanism is applied, and there exist good sharp minima, coinciding with the observation in Dinh et al. (2017) .

However, there is no guarantee that all sharp minima are good in generalization.

Revisiting our analysis in Section 3, the restriction on the parameter space (see (A2)) help remove these sharp minima.

Specifically, (A2) provides an upper bound on the magnitude of (a, w (1) , w (2) ) and (A5) bounds the magnitude of ??( w (1) , x i a ) from below.

These constraints control the parameter space and remove all sharp minima generated in Theorem 4 in which ?? 1 or ?? 2 goes to infinity.

The 2 bounds in (A2) can be achieved through a proper 2 regularization (See Section 3.4).

We conclude by studying the local minima of wide neural networks in small sample regime. (Nguyen & Hein, 2017b) proved that a two-layer neural network model can always achieve perfect empirical estimation error when the same size is small.

Here, we extend this result for global and self-attention model.

The discussion of Theorem 5 is deferred to the appendices.

) of the objective function (4), is a global minimum; (6) is a global minimum.

5.1 SAMPLE COMPLEXITY

Theorem 1 proves that attention models require a lower sample complexity than baseline models, i.e., attention models require fewer samples to achieve the same test error as baseline models.

This result is validated empirically in this experiment.

To mimic the assumptions of Theorem 1, we consider ground truth two-layer neural network G * (x) is formed using random weights as inputs.

The network G * maps the input vector x ??? R 256 to 10-dimensional output.

The input vector x is randomly sampled such that each element is drawn i.i.d from N (0, 1).

An attention mask a * is then constructed with k randomly chosen elements as 1 and the rest as 0.

The ground-truth labels are generated from y = G * (x a * ).

To test the sample complexity, we generate multiple datasets, each containing 10k, 14k, 16k, 18k, 20k and 50k unique samples respectively using the scheme mentioned in the previous section.

A common test set of 5000 samples is created to evaluate each of the models.

A regression model is then trained on each of these datasets.

All models are trained with SGD optimizer with a fixed learning rate of 10 ???3 .

Table 1 reports test errors for baseline and attention models at 400k iterations as the number of training samples vary.

We observe that attention models need fewer training samples than baseline models to achieve a desired error.

For instance, to attain the desired error 0.07, attention models need 14000 samples, whereas baseline models need 20000 samples.

We would like to point out that improvements obtained by attention models is not because of increase in model parameters.

As shown in Table 1 , the number of parameters in baseline and attention models are comparable.

Hence, the performance gain is solely due to the attention mechanism.

Regularization: In section 3.4, we discuss how regularizing attention vector helps obtain a better attention model.

To empirically validate this claim, we train a model with L 1 regularization on the attention vector.

Same experimental setting as Section 5.1 is used, a regression model is trained using a two-layer neural network.

We add the L 1 penalty on the attention vector to the objective:

The results are also shown in Table 1 .

We observe that models trained with L 1 regularization achieves better sample complexity than its unregularized counterpart.

We also observe the faster convergence when the models were regularized, as shown in the appendices.

We extend the sample complexity experiments to self-attention model discussed in Section 3.2.

Since our model is tailored towards natural language tasks, we consider the problem of sentiment classification on IMDB reviews dataset (Maas et al., 2011) .

Note that our analysis needs fixed length sentences which hardly holds true in any NLP dataset.

So, we zero-pad all our sentences to make their length equal the maximum sentence length in the dataset (2142 for IMDB reviews).

For every input word, we first obtain their corresponding pre-trained GloVE embeddings (??? R 100 ) which is then passed to the neural network.

As a baseline model, we flatten the input to one large vector of dimension 2142 ?? 100 and pass it to a 1-hidden layer MLP with 256 hidden units.

For self-attention model, we use w

Once the attended features are computed per equation 2, it is passed to a 1-hidden layer MLP with 256 hidden units as the baseline model.

All models were trained using Adam optimizer with learning rate 10 ???3 .

This was the setting that gave the best performance among optimizer and learning rate configurations we tried.

A comparison of sample complexity of baseline model and the self-attention model is provided in Table 2 .

We clearly observe that self-attention model requires low sample complexity to achieve the same error as the baseline model.

To test if improvements are obtained in attention model due to increase in model parameters, we ran the baseline model with twice the number of parameters as the self-attention model.

Even with a large parameter size, baseline model performs poorly compared to self-attention models.

This experiment studies the convergence of the empirical risk of the baseline and attention models.

A modified MNIST dataset called NoisyMNIST is constructed where the images of digits from the MNIST dataset is embedded in noise as shown in Panel (a) of Figure 2 .

We consider the classification task to predict the labels of the digit in each image.

Since the ground truth label depends only on certain regions of input, NoisyMNIST mimics the data generating process we consider in this paper.

For the baseline model, we train a two-layer neural network with 128 hidden units is using stochastic gradient descent.

For attention models, the input tensor is multiplied element-wise with a learned attention mask a as in equation 1.

The attended input is then passed to the two-layer network.

The convergence plots for baseline and attention models are plotted in Figure 2 .

We observe that the attention model converges faster, and to a better minimum than the baseline model.

Similar behavior is observed for different learning rate and scale configurations as shown in the appendices.

Figure 3: Top 100 eigenvalues of the Hessian matrix for baseline and attention models.

We study the Hessian matrix of the loss surface to validate the loss landscape of attention models.

The same classification setup as the previous experiment is considered.

The following two-layer neural network architecture was employed: 576 ??? 16 ??? 10.

The Hessian matrix of loss landscape about the computed minimum was, respectively, computed for the baseline and attention models, and their top k sorted eigenvalues are plotted in Figure 3.

Baseline models exhibit higher eigenvalues than attention models, so the loss landscape of attention models are flatter than the baseline models.

Since flat landscapes lead to better generalization, models with attention generalize better than models without attention as shown in Section 5.2.

In this paper, we study the loss landscape of two-layer neural networks on global and self attention models, and show that attention mechanisms help reduce the sample complexity and achieve consistent predictions in the large sample regime.

Additionally, by analyzing the number of linear regions, the loss landscape under small sample regime, and flatness of local minima, we demonstrate that attention mechanisms produce a well behaved loss landscape that leads to a good minima.

Extensive empirical studies on NoisyMNIST dataset and IMDB reviews dataset validate our theoretical findings.

In appendices, Section A presents the extensions of our analyses to recurrent attention model; Section B provides detailed justification of our assumptions; Section C discusses the implications of theorem results in more detail(Theorem 1, Corollary 1, Theorem 3 and Theorem 5 in order); Section D discusses the regularization effect in attention networks and potential extensions beyond 2 layers and to CNN/RNN.

Finally We provide proofs and additional experiment results in section E and F separately.

Here we consider analyzing the representative recurrent attention framework in Bahdanau et al. (2014) .

In the recurrent attention network, we still have each data point

t??p , corresponding to p words with t-dimensional embedding.

Then the generative model can be represented as:

Analogous to NLP setting, a(x i ) is a unknown function mapping x i to a t-dimensional vector, where a(x i ) j represents the effect of the j th word in the sentence for point i. To simplify the model, we use data features themselves as their annotation, then for time stamp k = 1, . . .

, T , The recurrent attention model estimates a(x i ) as follows:

where score(??) is the scoring function representing how well the inputs around position j and the output at position i match.

It can be dot product or MLP.

And f (??) is the function to update s k .

Suppose the parameter set inside these two functions are w a and w f with number of parameters as d a and d f accordingly.

Here we show that when these two functions are expressive enough, recurrent attention network will also have sample complexity bound parallel to previous sections.

Here we introduce necessary assumptions.

(A11) The output y can be predicted by the two-layer network with an independent sub-Gaussian error with variance ?? 2 , i.e, there exists a set of parameters (w

(A13) We assume w a 2 ??? C 8 and w f 2 ??? C 9 .

(A11) and (A13) are parallel to (A1) to (A5) in global attention case.

They can be justified similar as them, which is discussed in Section B of appendices.

Now we can provide following sample complexity bound extended from previous sections.

Theorem 6.

Under (A1),(A2),(A11) to (A13), there exists a sufficient large T , for any ?? > 0, suppose

where ?? = C 1 C 2 C x , such that if there exist stationary point(s), then with probability tending to 1, any stationary point (w (1) ,w (2) ,w f ,w a ) satisfies the following prediction error bound:

Remark: This theorem provide a sample complexity bound for recurrent attention network.

It holds when such 'good stationary point' exists.

It also shows a trade-off between a complicated recurrent attention network and the sample complexity bound.

If f (??) and a(??) are properly selected, they will be sufficient expressive to obtain good stationary points, and also the number of parameters d w and d f will not be too large.

In this way.

an ideal sample complexity bound to these good stationary points can be achieved as theorem says.

However, with a over complicated design in these functions, the sample complexity bound will be large; With a over simple design, such good stationary points don't exist.

It is parallel to a trade-off between approximation error and estimation error in learning theory.

The theory implies a good design of recurrent structure will help achieve an optimal sample complexity in recurrent attention model.

In this section, we discuss the rationality of Assumptions (A1) to (A6).

Note that (A1) to (A6) are required to prove the main result in Theorem 3 and they have also been studied by Keskar et al. First of all, (A1) and (A2) require upper bounds on the input x i and 2 bound for network weights.

It is a standard assumption in landscape analysis (Mei et al., 2018a; b) , and also it is crucial to remove sharp minima which may not generalize well (Keskar et al., 2016; Dinh et al., 2017) .(See remark after Theorem 4).

These assumptions can be achieved through regularization.

(A3) requires that this two-layer network are rich enough to specify the condition mean E(y i |x i ).

It has been studied that general bounded functions with a Fourier representation on [???1, 1] can be well approximated by the defined two-layer network(Barron & Klusowski, 2018).

(A4) requires a sparse structure on a ; otherwise, the model would be equivalent to the baseline model, simply just choose attention masks all being 1.

And a ??? ??? 1 requires the attention weight ranges from 0 to 1.

(A5) is a technical condition for our analysis of stationary point. (A5) includes two parts.

Both of them hold naturally when dimensionality is large.

Firstly, the lower eigenvalue bound assumes ??( w (1) , x i a ) is not degenerated.

If eigenvalue assumption is violated, the model is equivalent to a network with fewer number of units in hidden layers, and we can study this equivalent degenerated one instead; This assumption also guarantees us to remove sharp minima (Keskar et al., 2016; Dinh et al., 2017) , same as (A1) and (A2).

Secondly, (A5) assumes when ??( w (1) , x i a ) is not well estimated, there exists an 'active feature'.

The correlation between this feature and bias cannot be cancelled by the direction of a specific linear combination of ??( w (1) , x i a ).

Intuitively, this assumption says that the correlation between h t (x ??k ) and E(y|x ??k ) cannot be fully explained by a fixed linear combination if there is some systematic bias in ??( w (1) , x i a ).

Since there is systematic bias, it is reasonable to assume this systematic bias cannot be uncorrelated with all the directions spanned by h t (x).

This correlation assumption between h t (x ??j ) and u is parallel to the full column rank condition in (Nguyen & Hein, 2017b) .

Considering all active terms in h t (x ??k ), they span a larger space comparing to ??( w (1) , x i a ), considering d is a fixed dimension.

Thus it is a natural assumption if we have reasonable large dimensionality and the model doesn't degenerate.

What's more, with required large sample size, we can also straight forwardly evaluate this assumption by checking empirical correlation, and avoid this type of bad minima through random initialization and proper gradient descent type algorithm(Allen-Zhu et al., 2019).

And it also will not affect the key structure of our proof.

Specifically, for the sd(h t (x ??k )) = O(1) part, since h t (x ??k ) = x ??k a k (w

, where ?? ( w (1) , x i a ) t are O(1) with positive probability otherwise the network always predicts zero, therefore we have sd(h t (x ??k )) = O(1) as long as sd(x ??k ) = O(1) and a k = O(1).

For the correlation assumption cor(u, h t (x ??k )) = O(1), if this assumption doesn't hold, cor(u, h(xk)) 2 = o(1) for all k = 1, . . .

, p. By linear combination, for any vector z = (z 1 , . . . , z k ) with bounded 2 norm, we have cor(u, z

By the arbitrary choice of our z, it means that all the directions of z are almost not correlated with u. When p and s 0 are high-dimensional,w (1) (x a)((w (2) t ?? ( w (1) , x a ) spans all the directions as p ??? ???, thus there must be some direction is correlated with u. Therefore it is reasonable to assume a O(1) correlation occurs.

(A6) assumes that we fit a model with expectation close to truth.

We can always achieve this by centralization.

With these assumptions, we just make sure we remove unnecessary minima, such that we can concentrate on analyzing the behaviour of good stationary points of our interest.

Assumptions have no influence on the main idea of our following theorem: showing the required sample complexity to approach a good minimum is reduced with attention mechanism.

Remark: The assumptions are also related to the question that when attention mechanism should be applied in network. (Zhou et al., 2015) shows that attention sometimes can be badly used in certain cases, which shares the similar philosophy with our analyses.

For example, our theoretical analyses depend on the assumption (A3), that is, the model can be correctly specified when attention mechanism applies.

This can be violated when all the variables are useful and they all need to be included in the model.

In this case, the model with attention will be inconsistent.

It indicates that in this case when assumptions are violated, neural networks can achieve the precise estimation only through the over-parameterization of w (1) .

When we compare sample complexity of attention model versus baseline model, we say the key difference will be that s 0 is substituted with p, which can be a larger diverging dimensionality.

Constants C 1 and C 2 can be regarded remain same when we compare attention model with the baseline for following reasons: (1) We assume same generative model, and the network size is the same, thus the optimal weight is known to be same.

To make sure trained network weight is on the same scale with the optimal one, it is fair to keep there 2 bound constant same.

(2) In this framework, we study the effect when p diverges to ??? as n ??? ???. In this aspect, we don't expect the weight norm also diverges, since a diverging weight leads to overfitting.

By imposing 2 regularizations on weights, we can always control the upper bound of 2 norm.

Therefore it is reasonable to assume its norm is bounded by a sufficient large constant.

By imposing 2 regularization on weights, we can always control the upper bound of 2 norm by this sufficient large constant.

(3) Even with overfitting, C 2 in baselines are expected to be even larger due to the overfitting effect(Ch.7.

Goodfellow et al. (2016) .

It will further explain why sample complexity of baseline is even larger comparing with fixing C 1 and C 2 .

In the experiment result, we also observe that the 2 norm of weights from baseline is larger or equal to the attention network.

Given the assumption (A7), we do not need s 0 on the sample complexity bound for the sum-to-one global attention model.

However, if we rescale x properly, the result will be parallel to Theorem 1 result.

In a sufficiently wide network, ).

Given p log p???s0 log s0 p???s0 ??? p p???s log p, since p p???s is close to p when s is relatively small, the result still holds when n 1 is larger than the order of p.

For illustration, the bounds are plotted in Figure 1 in paper.

The red line is for baseline model with p = 100, and others are attention model with different sparsity level s 0 .

In general we can see the bound for attention model is smaller than that for baseline model.

The implication is that, when proper attention mechanism is applied, the approximation error remains small, and we can use a simpler landscape structure with less number of linear regions to reduce the estimation error.

This is the why we can achieve specific prediction error rate with a smaller sample complexity.

In assumptions, rank(?? ( w (1) , x i a ) i=1,2,.

.n ) = n is a mild assumption in a wide network with over-parameterization.

We can see that as long as we choose the number of units d to be larger than n, the linear dependence of w (1) , x i a i=1,2,..n holds with measure zero.

In other words, almost surely this matrix has full column rank n.

Thus after the nonlinear activation, The full column rank still holds almost surely.

This assumption is similar to the condition in Theorem 3.8 of Nguyen & Hein (2017b) , where the number of units in some layer is larger than the sample size.

When the sample size is smaller than the number of units in the network, this theorem holds for the network without attention.

It has been proved by Nguyen & Hein (2017b) and Soudry & Carmon (2016) under different conditions.

Our analyses provide some theoretical justification on how attention mechanisms help learn superior models.

The benefit mainly comes from that the attention weight a shrinks the whole parameter space, while this space is still large enough to capture all the necessary information.

Thus the gradient and Hessian are more controllable in this space, and the landscape of loss function behaves better compared to the baseline model.

It will be shown again in Theorem 4 in the following section.

This fact holds for both global and self-attention model.

Our theory further validates that as long as the attention masks are learnt well, the performance is expected to improve.

And this effect can be more significant when the weight is sparse.

We also find that imposing constraints and regularization on network weights can also help remove sharp minima, and keep flat minima with good prediction.

These discussions can be found in D.1.

The main idea of Theorem 1 and 2 can also be extended to multi-layer networks under the assumption that bias and gradients of weights are not uncorrelated.

They can be found in D.2.

Our analyses indicate that it is worth considering approaches to control the bounds for the network weight matrices and pursue more accurate estimation of attention weighted input space.

Motivated by this message, we suggest two possible regularization methods, which help to improve currently used attention mechanisms.

??? 2 Regularization on w It is known that in optimization, 2 regularization on w is equivalent to specific 2 bound for w. For our two-layer model, if we choose regularization level properly, the 2 bound for w (1) , w (2) ( also w Q ,w K and w V in self-attention model) will match our assumption.

In practice, a proper regularization should be as large as possible, as long as a nice in-sample prediction is still achieved.

This is important for attention mechanisms to keep prediction power while improving the landscape behavior of loss function.

??? 1 Regularization on a to achieve sparse attention weight As we discussed, 1 regularization can help achieve sparsity and more precise estimation.

For our 2-layer model, imposing 1 regularization will not affect the analysis since we didn't use the gradient w.r.t a in the analysis.

It means the same theoretical guarantee holds for the regularized network.

At the same time a will be more sparse, practically lead to a more precise estimation, and more interpretable result.

This idea can also be adapted to self-attention model.

In self-attention, we can add a regularization to a = sof tmax(

With this regularization, only part of the value matrices proceeds to the decoding procedure.

In real world application such as in transformer, the sparsity in a i corresponds to the case when there are some words in a sentence should not attend each other.

In this case, a sparse a self help us only focus attention between useful words in a sentence, thus improve predition.

It is worth mentioning that, through our theorem, we can tell why 1 loss are more helpful than 2 loss under sparsity assumption.

Both 1 loss and 2 loss will control the magnitude of attention mask, however, 1 loss can also help control sparsity level s 0 , and we can see that the sample complexity bound is proportion to s 2 0 .

Although 2 loss on attention mask can also help reduce the sample complexity, its effect will only be reflected in the log term, therefore it is less effective.

Similarly in Theorem 2, if we assume the attention to be sparse(i.e.

one word should not attend to all words, but only some relevant words), then 1 loss can help further reduce the sample complexity.

Regularization is one experimental approach for estimating a sparse and precise a and w. If some other methods can achieve this target, they are also the right directions to reduce sample complexity and improve current attention structures.

Here we discuss how our results of Theorem 1 and 2 can be extended to multi-layer neural nets.

First we consider a D-layer network with naive global attention structure:

???w (k) , under the assumption that at least one gradient term ???f (w k ) for k = 1, . . .

, D has correlation at rate O(1) with bias u(this assumption is parallel to (A5)), we can show the expectation term E x,y (???R(w (k) )) ??? O(??).

Then using the similar -covering and uniform convergence technique as Theorem 1 goes, we can still show attention mechanism leads to a smaller -covering number and tighter Hoeffding bound, thus leading to a smaller sample complexity bound comparing to baseline model.

Then we consider a D-layer network with self-attention structure.

We denote the k th self-attention layer follows g k (x

is the output of (k ??? 1)

a is calcualted in the same way with two-layer self-attention network.

Then we have

, and w D1 ??? R ?? , w D2 ??? R ?? .

In this way, the network calculate self-attention D times and finally produce the final prediction.

It is worth mentioning that, To obtain a scalar prediction in regression model, we flatten the value matrix of the last layer as same as the two-layer model.

We still denote

) ??? E(y|x)).

Then the necessary assumptions parallel to (A2), (A9) and (A10) are as follows.

??? (A14) All weights w kj for k = 1, . . .

, D and j = 1, 2 satisfy w kj 2 ??? C 10 .

And we assume the prediction is centered, i.e. E(u) = 0.

??? (A15) The output y can be predicted by the D-layer self-attention network with an independent sub-Gaussian error with variance ?? 2 , i.e, there exists a set of parameters

??? (A16) There exists interger k and r such that k ??? {1, . . .

, D} and r ??? {1, 2}, such that cor(???f (w kr ), u) = O(1) and sd(???f (w kr ) = O(1).

Then we can have the following theorem of sample complexity bound of multi-layer self-attention model:

Theorem 7.

Under (A1) and (A14) to (A16), we assume the weight term w kr satisfies (A16) with ???f (w kr ) 2 ??? c k .

d self is the total number of parameters in all value, query, key matrices.

Then for any ?? > 0, given the sample size:

where ?? = C 1 C 2 C x , with probability tending to 1, any stationary point (w (1) ,w (2) ,w Q ,w K ) of the objective function (6) satisfies that:

Remark: Because multi-layer self-attention models include a large parameter set with complicated gradients, the assumptions are not as intuitive as the two-layer model.

But the main assumptions are parallel, such that the bias u i cannot be uncorrelated with all possible directions.

And this assumption is reasonable considering the high-dimensionality nature of networks.

The extension of this multi-layer model is omitted in the main paper since it leads to over-complicated derivations and complicated assumption discussion and will distract readers from main ideas of the paper.

We believe the two-layer attention model is representative enough to provide theoretical evidence on why attention reduces sample complexity.

Our analyses are based on fully connected network, and may not be able to apply to more involved task such as CNN and RNN directly.

But we believe the key message of our analyses also provide insights in analyzing CNN/RNN with attention: The attention mechanisms can help us effectively shrink the parameter space, thus reducing most of the noise and unnecessary variability in training.

Thus the stationary solutions are more likely to generalize well.

That's why we start with analyzing the naive global attention model and self-attention models, which can help inspire the analysis of these complicated tasks.

More detailed analysis/experiments with CNN/RNN are important future work.

To this point, our experiments aim at validating the theoretical analyses and explaining why attention works in general.

Proof.

The proof is divided into two parts.

Firstly, we study the landscape of population risk in part (a), then we evaluate the convergence of empirical risk to the population risk in part (b).

We introduce necessary notations beforehand.

To emphasize the role of x and y separately, here we denote R(w (1) , w (2) , a) = E y|x (R n (w (1) , w (2) , a)), which is the expectation of the empirical loss gradient with respect to y, treating x as random, and ???R(w) as corresponding derivatives.

And we denote E x (???R(w (1) , w (2) , a)) = E x,y (R n (w (1) , w (2) , a)), which is the expectation of the empirical loss function with expectation to both x and y. In our analysis, first we will study E x (???R(w (1) , w (2) , a)), then we analyze E y|x R n (w (1) , w (2) , a).

The motivation of using

, a) comes from that it simplifies the part (b) analysis of empirical risk convergence, since the randomness of x has been included in the population risk analysis, making advantage that the noise is independent from predictors.

In the proof, we may use o(??) for vector/matrix case.

In these cases, it means that every element in vector/matrix is o(??).

In the proof, we will regard d as an arbitrary large fixed value, not diverging with n.

and u i as the version with specified sample index.

Then the derivatives of population risk with expection to y can be presented as follows:

(1) , w (2) ), all the u i are zero, and all the gradients expectations are zero.

Thus for any true set of parameter (a , w

(1) , w (2) ), they have zero gradient expectation automatically.

And the key of our proof is showing that with high probability, any parameter (a,

, because their gradients w.r.t to w (2) or w

(1) must be bounded away from zero.

By our assumption (A2) and (A4), our parameters w (1) ,w (2) and a are inside the 2 balls

.

By Lemma 5.2 in Vershynin (2010), we know the -covering number N 1 ,N 2 ,N 3 for these three balls are upper bounded by:

Then we know 3 -covering number for the union of all three parameters N 3 satisfies that N 3 ??? N 1 N 2 N 3 .

For the ease of notation, we denote ?? = (a, w (1) , w (2) ).

Let ?? = {?? 1 , ?? ?? ?? , ?? N } be a corresponding cover with N 3 elements.

Then we can always find ?? such that for any feasible ??, there exists j ??? [N ] such that max( w

2 , a (j) ??? a 2 ) ??? .

In this proof, we use parenthesis subscription (j) to represent elements in the cover, to distinguish it from other subscriptions.

By triangle inequality, we have for v = 1, 2:

And in the following section, we prove that if

Here the subscription k without parenthesis corresponds to k th feature.

First we consider ???R(w (2) ).

By (A4), there are at most s 0 nonzero elements in x a, and we denote ?? 1,active as the 1 norm on the elements with corresponding non-zero attention mask.

Thus by inequality between norms, we have:

.

Then we consider the case when cov (u, ??( w (1) , x i a )) 2 = o(??).

In this case, we denote r = cov(E(y|x), ??( w (1) , x a )) ??? R d , and denote the covariance matrix for ??( w (1) , x a ) as:

, using subtraction and addition, we have: ?? ?? w (2) = r + o(??) With (A5), covariance matrix ?? ?? is invertible with smallest eigenvalue lower bounded, we have w (2) = ?? ???1 ?? r + o(??).

Next we argue the following term by contradiction:

where ?? ( w (1) , x a ) represents the function corresponding to the true parameter set (w (1) , w (2) , a ).

If we have equation 9 violated, we know:

where we use E( ??(

, and we know ?? ( w (1) , x a ) corresponding to true parameter set is finite.

Since we have derived w (2) = ?? ???1 ?? r, with lower bounded eigenvalue assumption, we can derive:

We know if we have cov (u, ??( w (1) ,

.

Therefore we conclude w (2) = w (2) + o(??).

Plugging back to the formula of u in equation 12, we have:

Here we conclude the contradiction.

When condition equation 9 is violated, we derive

Now we are ready to study ???R(w (1) ).

Recall in assumption (A5), we denote

, where x ??k represents k th feature in x, and a k is the attention weight for x ??k .

And we have proved that E( ??(

Thus by (A5), we can find k, t such that sd(h t (x ??k )) = O(1) and cor(u, h t (x ??k )) = O(1).

Then we have

Therefore we have:

Here we conclude that we can always find k ??? {1, . . .

, p}, such that E(???R(w

With this conclusion, we move to bound the gradient term

With any fixed parameter set, we can calculate:

where we use again that there are at most s 0 nonzero elements in x a:

From the last section, we know there exists a constant c such that

k )) 2 ??? c?? for some constant c. Suppose E x (???R(w (2) )) 2 ??? c??, and we denote

x , And we know v i 2 2 is bounded as we derived.

Therefore we can use Hoeffding bound on the 2 norm in direction of E x (???R(w (2) (j) )), since we know the variance on this direction is smaller than ?? 2 .

Denoting ???R(w

(j) ) as the gradient with respect to j th parameter set in -cover for j ??? {1, . . .

, N }:

?? 2 ) By union bound, we have:

Secondly we analyze ???R(w (2) ) ??? ???R(w

(j) ) 2 term.

Here we use u i to represent the prediction error for i th instant with respect to parameter (a, w (1) , w (2) ), and use u i(j) to represent the term with respect to the parameter from j th element in -cover set.

By triangle inequality, we have:

We choose = c?? 3s0C 2

x C 2 1 C2

, and plug back above results to equation 8, then at least with prob-

Finally we can conclude that with probability 1???o n (1), for any (a,

Applying the same technique, we can show that with probability 1 ??? o n (1), we have ???R(w

So far, we have shown that for population risk with respect to y, with high probability, all the parameter sets with poor prediction in expectation, i.e E(|w

, their population risk gradient with expectation to y must be away from zero.

Now we move forward to show that empirical risk will converge to the popular risk, i.e. ???R n (w) ??? ???R(w).

Thus these parameter sets cannot have zero empirical gradient.

In aspect of three parameter sets, they can be represented as:

With (A3), we know that i ??? subG(0, C 2 4 ), thus

ing the bound for ??( w (1) , x i a ) we have derived in last section, with sample size n ?? 2 c 2 ?? 2 log( s0C1C2Cx c?? )(pd + p + d), conclude that with probability 1 ??? o p (1):

???R n (w

Recalling part (a), under the first case that w.h.p ???R(w (2) ) 2 ??? c?? 3 for any parameter (a, w

(1) , w (2) ) with w (2) ??( w (1) , X ?? ) ??? E(y|X) 2 ??? ??.

Combining this with (10), we can conclude that for any positive constant ?? > 0, with required sample size, with high probability that ???R n (w (2) ) 2 > 0, thus they cannot be stationary solution for our loss function.

As we stated, if we are in another case, we have w.h.p ???R(w

, we can use same techniques to show w.h.p ???R n (w (2) ) 2 > 0.

In other words, under our assumptions, all the stationary points (??,w (1) ,w (2) ) in our programming satisfy the prediction error upper bound rate ?? w.h.p.

Proof.

To extend the result from Theorem 3 to corollary 1, we simply substitute the 2 norm bound x i a 2 , from s 0 C x to C x since a 2 ??? a 1 = 1.

All the other parts keep the same.

Thus the only difference is that we remove s 0 in the bound comparing with Theorem 3.

Proof.

This proposition is a direct result of Theorem 3.

Since the assumptions for a still hold, all the bounds apply.

The only different is that since a is not optimized together, we don't have to consider the -cover number for a in the maximum operator.

This leads to a slightly tighter sample complexity bound in corollary 1 comparing with theorem 3.

Proof.

Similar with Theorem 1, we obtained a new -covering bound:

where

And also, the new v i and z ik terms are:

where

)).

Under assumptions, using the same agrument with Theorem 3, we can show there exists a constant c such that either E(???R(w (2 )) 2 ??? c?? or there exist k ??? 1, . . .

, p such that E(???R(w

X , considering a self is normalized by softmax function for each vector in set.

Parallel to Theorem 3, by hoeffding bound and union bound, we have:

?? 2 ) Secondly we bound ???R(w (2) ) ??? ???R(w (2) ) j 2 term by subtraction and addition:

recalling that u i(j) and a i(j) are corresponding to the j th epsilon cover.

We choose = c?? 3C 2 1 C 2 2 C 2 7 C 2 x , and combine the above results.

Then at least with probability

Thus with this required sample complexity, we have

3 .

In the same way, we can in the case when show ???R(w

Finally we can conclude that with high probability, any parameter (a, w

(1) , w (2) ) with

.

Then following the same empirical risk convergence argument, we show that with high probability they cannot be stationary point.

Proof.

First with a 0 = s 0 , we know all the inputs x i with corresponding a i = 0, will be inactive in the network.

We can omit all these inactive inputs.

Then we split n 1 units into s 0 group, with n1 s0 number of units in each group, and discard the leftover units.

s 0 different groups correspond to s 0 active inputs with non-zero attention weight.

Inside each group, for example in j th group, denoting q = n1 s0 , we choose the input weights and biases for i = 1, 2, ?? ?? ?? , q as:

. . .

here we assign w j to be a row vector with j th variable equal to 1 and all other entries to be 0.

And in the second layer, we choose w (2) = (w 3 , ?? ?? ?? , w 3 ), where

q+1 ), corresponding to h 1 to h q in each group.

Then the designed network has q linear regions inside each group, giving by the intervals:

Each of these intervals has a subset that is mapped by w 3 h(x) onto the interval (0,1).

Montufar et al. (2014) Therefore the total number of linear regions is lower bounded by n1 s0

s0 .

Proof.

Here we define an (?? 1 , ?? 2 ) scale transformation such that:

Then we know the jacobian determinant for T ??1,??2 is ??

.

Let r > 0 such that B ??? (r, ??) is in C(L, ??, ) and has empty intersection with (a, w

(1) , w (2) ) = 0.

Since pd > d, we assign ?? 2 ??? ???, such that the jacobian determinant goes to infinity, and the volume of C(L, ??, ) goes to infinity.

For the Hessian matrix, without loss of generality, we assume there is a positive diagonal element ?? > 0 in a.

Therefore the Frobenius norm

is lower bounded by ?? ???2 1 ??.

Further we apply the fact that the biggest eigenvalue of a symmetric matrix X is larger than c X F , and pick ?? 1 < c?? M , then we have the biggest eigenvalue of ??? 2 L(T ??1,??2 (??)) is larger than M. Therefore there exists a stationary point such that the operator norm for Hessian is arbitrary large.

Thus we finish proving part (a).

Then we consider part (b).

Similar with part (a), we define an ?? scale transformation such that:

And all the value,query and key matrices remain the same.

Then we know the jacobian determinant for T ?? = ?? (pdv???1)d .

Since pd v d ??? d, as we assign ?? ??? ???, such that the jacobian determinant goes to infinity, and the volume of C(L, ??, ) goes to infinity.

For the Hessian matrix, we still assume a positive diagonal element ?? > 0 in w (1) .

Similarly we have the Frobenius norm

is lower bounded by ?? ???2 ??.

When we choose sufficient small ??, we have the biggest eigenvalue of ??? 2 L(T ??1,??2 (??)) is larger than any constant M .

Therefore there exists a stationary point such that the operator norm for Hessian is arbitrary large.

Proof.

For global attention in part (a), We start with calculating the gradient of the empirical loss function ???R n (w (2) ), where R n (w (1) , w (2) , a) = 1 2n

Denoting u i = (w (2)T ??( w (1) , x i a ) ??? y).

The derivatives can be presented as follows:

?? ( w (1) , x i a )))

By assumption, rank(??( w (1) , x i a ) i=1,...,n ) = n, thus solving the linear system, we must have u i = 0 for any i = 1, 2, ..., n to satisfy that ???R n (w (2) ) = 0.

Thus we know that the loss is exactly zero inside sample.

Thus it must be a global minimum.

Part (b) can be proved by substituting a to a in part (a).

Here we only consider the derivatives with respect to w

(1) and w (2) , they can be presented as follows:

))(w ) ) i=1,...,n ) = n, thus solving the linear system, we must have u i = 0 for any i = 1, 2, ..., n to satisfy that ???R n (w (2) ) = 0.

Thus we know that the loss is exactly zero inside sample.

Thus it must be a global minimum.

Proof.

First, we obtained new -covering bound for the parameter set (w (2) , w (1) , w f , w a ):

And N ??? ??

and u i as the version with specified sample index.

Then the derivatives of population risk with expection to y can be presented as follows:

a(x i )x j i )(w X with normalized attention weight.

Same argument follows for the case when E(???R(w (1 )) k 2 ??? c?? Then follow the same approach as Theorem 1 and 2, we obtain the sample complexity bound: n ?? 2 c 2 ?? 2 log(

E.9 PROOF OF THEOREM 7

Proof.

Under assumption (A1) and (A14), we know all input features and weights are bounded.

Therefore we know ???f (w kr ) 2 is Lipschitz continuous function, and we denote its Lipschitz constant L k .

For w kr , we can derive that:

Under (A14) to (A16), if we have E(f (x) ??? E(y|x)) 2 ?? 2 , then:

E(???R(w kr )) 2 sd(f (w kr ) 2 )sd(u)cor(???f (w kr ), u) O(??)

Then similar with Theorem 1 and 2, we construct an -cover over all parameters ?? := (w k1 , w k2 , w V , w Q , w K ), and we denote it as {?? 1 , . . .

, ?? N } such that for any feasible parameter, there exist j ??? [N ] such that the maximum 2 distance to ?? j is smaller than .

By calculating the number of parameters in all matrices in ??, we have

Denoting ???R(w kr (j) ) as the gradient with respect to j th parameter set in -cover for j ??? {1, . . .

, N }:

) ??? E x (???R(w Secondly we analyze ???R(w kr ) ??? ???R(w kr (j) ) 2 term.

As we have shown that the gradient is Lipschitz continuous, thus we have:

We choose = c?? 3L k , then at least with probability 1???O(N exp(???n c 2 ?? 2 ?? 2 )), we have ???R(w (2) ) 2 > c?? 3 .

Therefore we can choose n log(

o(1).

Finally we can conclude that with probability 1 ??? o n (1), for any (a, w (1) , w (2) ) such that E(w (2) ??( w (1) , x ?? ) ??? E(y|x)) 2 ??? ??, we have ???R(w (2) ) 2 > c?? 3 .

Then following the convergence of empirical risk procedure of Theorem 1, we show with probability going to 1 such that ???R n (w (2) ) 2 > 0 and all parameters with prediction error O(??) cannot be stationary point as long as n log(

Thus we complete the proof.

We present additional results on convergence of 2??? layer neural networks on classification task involving Noisy-MNIST dataset (Section 5.1 of main paper).

This dataset is formed by embedding an s ?? s digit image in a noisy image of size 48 ?? 48.

We show convergence results for three settings: s = {5, 8, 15}. In each setting, the learning rate of models are varied.

The plots are shown in Figures 4,5, and 6.

We observe that in every setting, attention models converge faster than the baseline models not employing attention.

In this section, we present convergence plots for sample complexity experiments discussed in Section 5.3 of the main paper.

Regression task is considered in this experiment.

Convergence plots of baseline model, attention model and regularized attention models are plotted for various sizes of the datsset N s as shown in Figure 7 .

We observe that for all N s values, regualrized attention model converge fastest followed by attention model which is then followed by baseline model.

So, we conclude that in addition to achieving improved sample complexity, attention models converge faster than models not employing attention.

Also, regularization helps speed up the model convergence.

Figure 7: Convergence plots for regression task varying the number of samples N s .

"Baseline" indicates 2??? layer NN not using attention, "attention" denotes attention models, and "attention regualrized" denotes attention models trained with L 1 regularization

<|TLDR|>

@highlight

We analyze the loss landscape of neural networks with attention and explain why attention is helpful in training neural networks to achieve good performance.

@highlight

This paper proves from the theoretical perspective that attention networks can generalize better than non-attention baselines for fixed-attention (single-layer and multi-layer) and self-attention in the single layer setting.