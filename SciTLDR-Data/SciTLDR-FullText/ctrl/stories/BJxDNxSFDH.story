Recent few-shot learning algorithms have enabled models to quickly adapt to new tasks based on only a few training samples.

Previous few-shot learning works have mainly focused on classification and reinforcement learning.

In this paper, we propose a few-shot meta-learning system that focuses exclusively on regression tasks.

Our model is based on the idea that the degree of freedom of the unknown function can be significantly reduced if it is represented as a linear combination of a set of sparsifying basis functions.

This enables a few labeled samples to approximate the function.

We design a Basis Function Learner network to encode basis functions for a task distribution, and a Weights Generator network to generate the weight vector for a novel task.

We show that our model outperforms the current state of the art meta-learning methods in various regression tasks.

Regression deals with the problem of learning a model relating a set of inputs to a set of outputs.

The learned model can be thought as function y = F (x) that gives a prediction y ∈ R dy given input x ∈ R dx where d y and d x are dimensions of the output and input respectively.

Typically, a regression model is trained on a large number of data points to be able to provide accurate predictions for new inputs.

Recently, there have been a surge in popularity on few-shot learning methods (Vinyals et al., 2016; Koch et al., 2015; Gidaris & Komodakis, 2018) .

Few-shot learning methods require only a few examples from each task to be able to quickly adapt and perform well on a new task.

These few-shot learning methods in essence are learning to learn i.e. the model learns to quickly adapt itself to new tasks rather than just learning to give the correct prediction for a particular input sample.

In this work, we propose a few shot learning model that targets few-shot regression tasks.

Our model takes inspiration from the idea that the degree of freedom of F (x) can be significantly reduced when it is modeled a linear combination of sparsifying basis functions.

Thus, with a few samples, we can estimate F (x).

The two primary components of our model are (i) the Basis Function Learner network which encodes the basis functions for the distribution of tasks, and (ii) the Weights Generator network which produces the appropriate weights given a few labelled samples.

We evaluate our model on the sinusoidal regression tasks and compare the performance to several meta-learning algorithms.

We also evaluate our model on other regression tasks, namely the 1D heat equation tasks modeled by partial differential equations and the 2D Gaussian distribution tasks.

Furthermore, we evaluate our model on image completion as a 2D regression problem on the MNIST and CelebA data-sets, using only a small subset of known pixel values.

To summarize, our contributions for this paper are:

• We propose to address few shot regression by linear combination of a set of sparsifying basis functions.

• We propose to learn these (continuous) sparsifying basis functions from data.

Traditionally, basis functions are hand-crafted (e.g. Fourier basis).

• We perform experiments to evaluate our approach using sinusoidal, heat equation, 2D Gaussian tasks and MNIST/CelebA image completion tasks.

An overview of our model as in meta-training.

Our system learns the basis functions Φ that can result in sparse representation for any task drawn from a certain task distribution.

The basis functions are encoded in the Basis Function Learner network.

The system produces predictions for a regression task by generating a weight vector, w for a novel task, using the Weights Generator network.

The prediction is obtained by taking a dot-product between the weight vector and learned basis functions.

Regression problems has long been a topic of study in the machine learning and signal processing community (Myers & Myers, 1990; Specht, 1991) .

Though similar to classification, regression estimates one or multiple scalar values and is usually thought of as a single task problem.

A single model is trained to only perform regression on only one task.

Our model instead reformulates the regression problem as a few-shot learning problem, allowing for our model to be able to perform regressions of tasks sampled from the same task distribution.

The success achieved by deep neural networks heavily relies on a large amount of data, especially labelled ones.

As labelling data is time-consuming and labor-intensive, learning from limited labelled data is drawing more and more attention.

A prominent approach is meta learning.

Meta learning, also referred as learning to learn, aims at learning an adaptive model across different tasks.

Meta learning has shown potential in style transfer (Zhang et al., 2019) , visual navigation (Wortsman et al., 2018) , etc.

Meta learning has also been applied to few-shot learning problems, which concerns models that can learn from prior experiences to adapt to new tasks.

Lake et al. (2011) proposed the one-shot classification problem and introduced the Omniglot data set as a few-shot classification data set, similar to MNIST (LeCun, 1998) for traditional classification.

Since then, there has been a surge of meta learning methods striving to solve few-shot problems.

Some meta learning approaches learn a similarity metric (Snell et al., 2017; Vinyals et al., 2016; Koch et al., 2015) between new test examples with few-shot training samples to make the prediction.

The similarity metric used here can be Euclidean distance, cosine similarity or more expressive metric learned by relation networks (Sung et al., 2018) .

On the other hand, optimization-based approaches learn how to optimize the model directly.

Finn et al. (2017) learned an optimal initialization of models for different tasks in the same distribution, which is able to achieve good performance by simple gradient descent.

Rusu et al. (2019) learned how to perform gradient descent in the latent space to adapt the model parameters more effectively.

Ravi & Larochelle (2016) employed an LSTM to learn an optimization algorithm.

Generative models are also proposed to overcome the limitations resulted from few-shot setting Hariharan & Girshick, 2017; Wang et al., 2018) .

Few-shot regression tasks are used among various few-shot leaning methods (Finn et al., 2017; Rusu et al., 2019; Li et al., 2017) .

In most existing works, these experiment usually does not extend beyond the sinusoidal and linear regression tasks.

A prominent family of algorithms that tackles a similar problem as few-shot regression is Neural Processes (Garnelo et al., 2018b; a; Kim et al., 2019) .

Neural Processes algorithms model the distributions of the outputs of regression functions using Deep Neural Networks given pairs of input-output pairs.

Similar to Variational Autoencoders (Kingma & Welling, 2013) , Neural Processes employ a Bayesian approach in modelling the output distribution of regression function using an encoder-decoder architecture.

Our model on the other hand employs a deterministic approach where we directly learn a set of basis functions to model the output distribution.

Our model also does not produce any latent vectors but instead produces predictions via a dot product between the learned basis functions and weight vector.

Our experiment results show that our model (based on sparse linear combination of basis functions) compares favorably to Neural Processes (based on conditional stochastic processes).

Our proposed sparse linear representation framework for few shot regression makes the few shot regression problem appears to be similar to another research problem called dictionary learning (DL) (Tosic & Frossard, 2011) , which focuses on learning dictionaries of atoms that provide efficient representations of some class of signals.

However the differences between DL and our problem are significant: Our problems are continuous rather than discrete as in DL, and we only observe a very small percentage of samples.

Detailed comparison with DL is discussed in the appendix.

3 PROPOSED METHOD

We first provide problem definition for few-shot regression.

We aim at developing a model that can rapidly regress to a variety of equations and functions based on only a few training samples.

We assume that each equation we would like to regress is a task T i sampled from a distribution p(T ).

We train our model on a set of training tasks, S train , and evaluate it on a separate set of testing tasks, S test .

Unlike few-shot classification tasks, the tasks distribution p(T ) is continuous for regression task in general.

Each regression task is comprised of training samples D train and validation samples D val , for both the training set S train and testing set S test , D train is comprised of K training samples and labels D train = {(x

Here we discuss our main idea.

We would like to model the unknown function y = F (x) given only D train = {(x k t , y k t )|k = 1...K}. With small K, e.g. K = 10, this is an ill-posed task, as F (x) can take any form.

As stated before, we assume that each function we would like to regress is a task T i drawn from an unknown distribution p(T ).

To simplify discussion, we assume scalar input and scalar output.

Our idea is to learn sparse representation of the unknown function F (x), so that a few samples {(x k t , y k t )|k = 1...K} can provide adequate information to approximate the entire F (x).

Specifically, we model the unknown function F (x) as a linear combination of a set of basis functions {φ i (x)}:

Many handcrafted basis functions have been developed to expand F (x).

For example, the Maclaurin series expansion (Taylor series expansion at x = 0) uses {φ i (x)} = {1, x, x 2 , x 3 , ...}:

If F (x) is a polynomial, (2) can be a sparse representation, i.e. only a few non-zero, significant w i , and most w i are zero or near zero.

However, if F (x) is a sinusoid, it would require many terms to represent F (x) adequately, e.g.:

In (3), M is large and M K. Given only K samples {(x k t , y k t )|k = 1...K}, it is not adequate to determine {w i } and model the unknown function.

On the other hand, if we use the Fourier basis instead, i.e., {φ i (x)} = {1, sin(x), sin(2x), ..., cos(x), cos(2x), ...}, clearly, we can obtain a sparse representation: we can adequately approximate the sinusoid with only a few terms.

Under Fourier basis, there are only a few non-zero significant weights w i , and K samples are sufficient to estimate the significant w i and approximate the function.

Essentially, with a sparsifying basis {φ i (x)}, the degree of freedom of F (x) can be significantly reduced when it is modeled using (1), so that K samples can well estimate F (x).

Our approach is to use the set of training tasks drawn from p(T ) to learn {φ i (x)} that result in sparse representation for any task drawn from p(T ).

The set of {φ i (x)} is encoded in the Basis Function Learner Network that takes in x and outputs

T .

In our framework, Φ(x) is the same for any task drawn from p(T ), as it encodes the set of {φ i (x)} that can sparsely represent any task from p(T ).

We further learn a Weights Generator Network to map the K training samples of a novel task to a constant vector w = [w 1 , w 2 , ..., w M ]

T .

The unknown function is modeled as w T Φ(x).

An overview of our model is depicted in Figure 1 .

Given a regression task T with

The model is then applied to make prediction for any input x. During meta-training, the validation set D val = {x n p , y n p |n = 1...N } for a task T is given.

The prediction is produced by taking a dot product between task-specific weights vector, w and the set of learned basis functions:

To train our model, we design a loss function L that consists of three terms.

The first term is a mean-squared error between the validation set labels y n p ∈ D val and the predicted y n pred .

We also add two penalty terms on the weights vector w generated for each task.

The first penalty term is on the L1 norm of the generated weight vectors.

This is to encourage the learned weight vectors to be sparse in order to approximate the unknown function with a few significant basis functions.

The second penalty term is on the L2 norm of the generated weights vector.

This is used to reduce the variance of the estimated weights as commonly used in regression (Zou & Hastie (2005) ).

The full loss function L is as follows:

where λ 1 and λ 2 represents the weightage of the L1 and L2 terms respectively.

Note that, it turns out that our loss function for meta learning is is similar to that of the Elastic Net Regression (Zou & Hastie, 2005) with both L1 and L2 regularization terms.

However, the difference is significant: Instead of focusing on a single regression task as in (Zou & Hastie, 2005) , we use this loss function to learn (i) the parameter θ for the Basis Function Learner network, which encodes the sparsifying basis functions for any task drawn from a task distribution, and (ii) the parameter ψ for the Weight Generator network, which produces the weights for any novel task drawn from the same task distribution. (Finn et al., 2017) 1.13 ± 0.18 0.77 ± 0.11 0.48 ± 0.08 Meta-SGD (Li et al., 2017) 0.90 ± 0.16 0.53 ± 0.09 0.31 ± 0.05 EMAML (small) (Yoon et al., 2018) 0.885 ± 0.117 0.615 ± 0.091 0.371 ± 0.048 EMAML (large) 0.783 ± 0.101 0.537 ± 0.079 0.307 ± 0.040 BMAML (small) (Yoon et al., 2018) 0.927 ± 0.116 0.735 ± 0.104 0.459 ± 0.058 BMAML (large) 0.878 ± 0.108 0.675 ± 0.094 0.442 ± 0.055 NP (Garnelo et al., 2018b) 0.640 ± 0.205 0.561 ± 0.234 0.421 ± 0.088 CNP (Garnelo et al., 2018a) 0.910 ± 0.234 0.630 ± 0.222 0.393 ± 0.145 ANP (Kim et al., 2019) 0.488 ± 0.188 0.216 ± 0.082 0.095 ± 0.068

Ours (small) 0.363 ± 0.018 0.169 ± 0.007 0.076 ± 0.004 Ours (large) 0.199 ± 0.010 0.062 ± 0.003 0.027 ± 0.002

In this section we describe the experiments we ran and introduce the types of regression task used to evaluate our method.

For all of our experiments, we set the learning rate to 0.001 and use the Adam Optimizer (Kingma & Ba, 2014) as the optimization method to preform stochastic gradient decent on our model.

We implement all our models using the Tensorflow (Abadi et al., 2016) library.

In the following subsections, we decribe each of experiments in more detail.

We include the experiments on the 1D Heat Equation and 2D Gaussian regression tasks in the appendix.

For all 1D Regression tasks, the Basis Function Learner consists of two fully connected layers with 40 hidden units.

For the loss function we set λ 1 = 0.001 and λ 2 = 0.0001.

Sinusoidal Regression.

We first evaluate our model on the sinusoidal regression task which is a few-shot regression task that is widely used by other few-shot learning methods as a few-shot learning task to evaluate their methods on (Finn et al., 2017; Li et al., 2017; Rusu et al., 2019) .

The target function is defined as y(x) = Asin(ωx + b), where amplitude A, phase b, frequency ω are the parameters of the function.

We follow the setup exactly as in (Li et al., 2017) .

We sample the each parameters uniformly from range

.

We train our model on tasks of batch size 4 and 60000 iterations for 5,10 and, 20 shot cases, where each training task contains K ∈ {5, 10, 20} training samples and 10 validation samples.

We compare our method against recent few-shot learning methods including Meta-SGD (Li et al., 2017) , MAML (Finn et al., 2017) , EMAML ,BMAML (Yoon et al., 2018) and the Neural Processes family of methods including Neural Processes (Garnelo et al., 2018b) Conditional Neural Processes (Garnelo et al., 2018a) and Attentive Neural Processes (Kim et al., 2019) .

We use the officially released code for these three methods 1 .

We show the results in Table 1 .

We provide two variants our model in this experimental setup.

The two models differ only in the size of the Weights Generator.

For the "small" model the Weights Generator consist of B = 1 self-attention blocks followed by a fully connected layer of 40 hidden units.

The self-attention block consists of three parallel weight projections of 40 dimensions followed by fully connected layers of 80 and 40 hidden units respectively.

The "large" model consists of B = 3 self-attention blocks also followed by a fully connected layer of 40 hidden units.

Each self-attention block has weight projections of 64 dimensions followed by fully connected layers of 128 and 64 hidden units respectively.

Both MAML and Meta-SGD uses an architecture of 2 fully connected layers with 40 hidden units which is similar to the architecture of the Basis Learner network, though both Meta-SGD and MAML both have additional optimization for individual tasks.

The Neural Process family of methods uses encoder archtecture of 4 fully connected layers with 128 hidden units and decoder architecture of 2 fully connected layers of 128 hidden units respectively which is more similar in architecture our larger model.

Similarly, we also compare our methods against two variants of EMAML and BMAML.

The "small" model consist of 2 fully connected layers with 40 hidden units each while the "large" model consists of 5 fully connected layers with 40 hidden units each.

This is to ensure fair comparison as both BMAML and EMAML lack a separate network to generate weight vectors but are ensemble methods that aggregate results from M p number of model instances.

We set the number of model instances in BMAML and EMAML to 10.

Alternative Sinusoidal Regression We also evaluate our method on another version of the sinusoidal task as introduced by Yoon et al. (2018) .

The range of A remain the same while the range of b is increased to [0, 2π] and the range of ω is increased to [0.5, 2.0].

An extra noise term, is also added the function y(x).

For noise , we sample it from distribution N ∼ (0, (0.01A)

2 ).

We also fix the total number of our tasks used during training to 1000 as in (Yoon et al., 2018) .

For this experimental setup we also include an ensemble version of our model where we train 10 separate instance of our model on the same 1000 tasks and aggregate their results by taking a mean of the predictions.

We evaluate our model for both 10 shot and 5 shot cases and show the mean-squared error results in Table 2 .

For this experimental setup, we calculate the mean-squared error from 10 randomly points from 1000 advanced sinusoidal tasks.

Our results show that our method outperforms all recent few-shot regression methods in sinusoidal tasks.

We also tested our method on more challenging image data, as done in (Garnelo et al., 2018a; b; Kim et al., 2019) .

We use MNIST and CelebA datasets (Liu et al., 2015) here for qualitative and quantitative comparison.

Each image can be regarded as a continuous function f : R 2 → R dy , where d y = 1 if the image is gray-scale or or d y = 3 if it is RGB.

The input x ∈ R 2 to f is the normalized coordinates of pixels and the output y ∈ R dy is the normalized pixel value.

The size of the images is 28 × 28 in MNIST and rescaled to 32 × 32 in CelebA. During meta-training, we randomly sample K points from 784(1024) pixels in one image as D train and another K points as D val to form a regression task.

In the meta-testing stage, the MSE is evaluated on 784(1024)−K pixels.

60,000(162,770) images are used for meta-training and 10,000 for meta-testing for MNIST(CelebA) dataset.

We compare our methods with NP family: CNP (Garnelo et al., 2018a) , NP (Garnelo et al., 2018b) and ANP (Kim et al., 2019) for K = 50 and K = 100.

Deeper network structure is adopted due to the complexity of regression on image data.

Namely, we use 5 fully connected layers with 128 hidden units in Basis Function Learner and 3 attention blocks in Weight Generator for our method.

The encoders and decoders in NP family are all MLPs including 4 fully connected layers with 128 hidden units.

Thus, the comparison is fair in terms of network capacity.

All the models are trained for 500 epochs with batch size 80.

The MSE on 10,000 tasks from meta-testing set is reported with 95% confidence interval, shown in Table 3 .

The top results are highlighted.

It can be observed that our method outperforms two of three NP methods and achieves MSE very close to most recent ANP.

The outputs of regression on CelebA image data are high-dimension predictions, which demonstrates the effectiveness of our method in such challenging tasks.

Note that ANP significantly improves upon NP and CNP using cross-attention, which can potentially be applied to our method as well.

In this subsection we provide some deeper analysis on the basis functions that are learned by our method.

In particular, we provide some further evidence to our claim that our method learns a set of sparsifying basis functions that correspond to the regression tasks that we would like to model.

To demonstrate the sparsity of basis functions, we take only the S largest weights in terms of |w| and their corresponding basis functions and illustrate the predicted regression function with the combination of only the S weights and basis functions.

We conduct this experiment on both the sinusoidal regression task and the more difficult image completion task and show these S-weights predictions in Figures 3 and 2b respectively.

The figures illustrate that our method is able to produce a good prediction of the regression function with only a fraction of the full set learned basis function (40 for the sinusoidal task, 128 for the MNIST image completion task).

This demonstrates the sparsity of Φ(x) as most of the prediction is carried out by just a small number of basis functions.

This also demonstrates that our method is able to force most of the information of F (x) to be contained in a few terms.

Therefore, using K samples to estimate the weights of these few important terms could achieve a good approximation of F (x).

In this subsection we detail some ablation studies on our model to test the validity of certain design choices of our model.

In particular we focus on the effects of the addition of self-attention operations in the Weights Generator and also the effects of using different penalty terms on our loss function.

To test out the effects of adding the selfattention operations to our model, we conduct a simple experiment where we replace the self attention operations in the self-attention block with just a single fully connected layer of equal dimensions as the self-attention weight projection.

Essentially, this reduces the Weights Generator to be just a series of fully connected layers with residual connections and layer normalization.

We compare the simpler model performance on the sinusoidal regression task as specified in Table 1 with our original model and show the results in Table 4 .

The results show that adding the self-attention operations do improve our methods performance on the 1D sinusoidal regression task.

We also conducted experiments to test the effects of the different penalty terms on the the generated weights vector.

In this ablation study, we compared our models trained using different variants of the .

Similar to the previous study, we evaluate them on their performance on the sinusoidal regression task as specified in Table 1 .

The variants we tested out are: (i) Loss function with only the L1-norm penalty term ; (ii) Loss function with only the L2-norm penalty term (iii) Loss function with both L1 and L2-norm penalty terms.

To demonstrate the sparsity of the weights vectors of each variant, we also show the a breakdown of the magnitude of the learned weight vectors over 100 sinusoidal tasks.

We group the weight vectors into three groups : |w| less than 0.02 to indicate weights that are near zero, |w| between 0.02 and 1 and weights with magnitude more than 1.

We show the results of the different variants in Table 5 .

We also present histograms of the magnitude of the learned weight vectors in Figure 4 The results do show that the combination of both L1 and L2 penalty terms do ultimately give the best performance for the sinusoidal regression task.

In terms of sparsity, the model trained with only the L1 loss term do gives the highest percentage of sparse weights though we found the model with both L1 and L2 terms do give a better performance while still maintaining a relatively high percentage of near zero weights.

We propose a few-shot meta learning system that focuses exclusively on regression tasks.

Our model is based on the idea of linear representation of basis functions.

We design a Basis Function Learner network to encode the basis functions for the entire task distribution.

We also design a Weight generator network to generate the weights from the K training samples of a novel task drawn from the same task distribution.

We show that our model has competitive performance in in various few short regression tasks.

In this section we illustrate all of the individual non-zero basis functions learned by our model for the sinusoidal task.

These functions are shown Figure 5 .

Note that out of 40 of the basis functions, only 22 of the learned basis functions are non-zero functions, further demonstrating that indeed our method is forcing the model to learn a set of sparse functions to represent the tasks.

Furthermore, it can be seen that the learned basis functions all correspond to the different "components" of sinusoidal function: most of the learned functions seem to represent possible peaks, or troughs if multiplied with a negative weight at various regions of the input range whereas the top four basis function seem to model the more complicated periodic nature of the sinusoidal functions.

Adding on to the experiments in Section 4.3, we also illustrate what happens when do the exact opposite.

We take the prediction using the full set of weight vectors/basis function and study the effect of the prediction when we remove certain basis function from the prediction.

Similar to the previous experiment, we remove the basis function by order of magnitude starting with the basis function with the largest corresponding |w|.

we conduct this experiment on the sinusoidal regression task and illustrate the results in Figure 6 .

Similarly, this study also demonstrates the importance of certain basis functions as removing them caused the prediction the change drastically.

In particular, notice that for sinusoidal task, removing just 4 of the most important basis functions resulted in a less accurate prediction than using just 10 of the most important basis functions.

Here we provide more details on the architecture of the Weights Generator Network.

As mentioned previously in Section 3.3.

The Weights Generator Network consists of a series of self attention blocks followed by a final fully connected layer.

We define a self attention block as such: An attention block consists of a self attention operation on the input of self attention block.

Following the self-attention operation, the resultant embedding is further passed through two fully connected layers.

A residual connection (He et al., 2016 ) from the output of the self-attention operation to the output of the second fully connected layer.

Finally, resultant embedding of the residual connection is then passed though a a layer normalization operation (Ba et al., 2016) .

Note that the input of the first self attention block will always be the input to the Weights Generator network, (Φ(x k t ), y k t ) whereas the inputs to subsequent attention blocks are the outputs of the previous attention block.

For the self-attention operation, the input is transformed into query, key and value vectors though their respective weight projections.

These query, key and value vectors, Q, K and V then go through a scaled dot-product self-attention operation as introduced by (Vaswani et al., 2017) :

We also evaluate our method on another 1D Regression task, the 1D heat Equation task, we define it as such: Consider a 1-dimensional rod of length L with both of its ends connected to heat sinks, i.e. the temperature of the ends will always be fixed at 0K unless a heat source is applied at the end.

a constant point heat source is then applied to a random point s on the rod such the the heat point source will always have a temperature of 1.0K.

We would like the model the temperature u(x, t) at each point of the rod a certain time t after applying the heat source until the temperature achieves equilibrium throughout the rod.

The temperature at each point x after time t is given by the heat equation:

∂x 2 For our experiments, we set L to be 5 and randomly sample K points of range [0, 5] on the heat equation curve.

We fix the total number of tasks used during training to 1000 and evaluate our model on both 10 shot and 5 shot cases, similar to the experimental setup for the Advanced Sinusoidal tasks.

We also compare our results to both EMAML and BMAML on this regrssion task and add an ensemble version of method for comparison.

The results of our evaluation is presented in Table 6 .

2D Gaussian.

We also evaluated our method on the for the 2D Gaussian regression tasks.

For this task, we train our model to predict the probability distribution function of a two-dimensional Gaussian distribution.

We train our model from Gaussian distribution task with mean ranging from (−2, −2) to (2, 2) and standard deviation of range [0.1, 2].

We fix the standard deviation to be of the same value in both directions.

Similar to the heat equation, we use the same setup as the Advanced Sinusoidal task and compare our methods to EMAML and BMAML.

We evaluate our model on 10, 20 and 50 shot case.

The results of our evaluation is presented in Table 7 .

Qualitative results on CelebA datasets.

We provide the qualitative results on CelebA datasets in Figure 7 .

We note that the RGB images are complex 2D functions.

We choose them to evaluate so that we can see the results more directly, not to compare with image inpainting methods, which is also mentioned in (Garnelo et al., 2018a) .

The results in Figure 7a are consistent with Figure 2a .

The regression results from our method are visually better than NP and CNP.

The predictions using first S largest weights are shown in Figure 7b .

The 2D image function is usually predicted with less than 50 weights, which suggests that the information of the 2D function is kept in several terms.

Our proposed sparse linear representation framework for few shot regression makes the few shot regression problem appears to be similar to another research problem called dictionary learning (DL), which focuses on learning dictionaries of atoms that provide efficient representations of some class of signals (Tosic & Frossard, 2011) .

However the differences between DL and our problem are significant: Our problems are continuous rather than discrete as in DL, and we only observe a very small percentage of samples.

Specifically, for a given y ∈ R n , the goal of DL is to learn the dictionary (n by M ) Φ for some sparse w:

In typical DL, the entire y is given.

Also, M > n for an overcomplete dictionary (Figure 8 ).

In few shot regression, the goal is to predict the entire continuous function y = F (x).

Therefore, viewing this as the setup in (7), n is infinite.

Moreover, only a few (K) samples of y is given: y k t = F (x k t ).

The locations of the given samples (x k t ) are different for different y (different task).

Therefore, our problem is significantly different and more difficult than DL.

Typical DL algorithms solve (7) and return Φ, which is a n by M matrix of finite dimensions (the dictionary).

In our setup, the basis matrix Φ has infinite entries, and Φ is encoded by the proposed Basis Function Learner network.

GT NP CNP ANP Ours K=100

<|TLDR|>

@highlight

We propose a method of doing few-shot regression by learning a set of basis functions to represent the function distribution.