The recent rise in popularity of few-shot learning algorithms has enabled models to quickly adapt to new tasks based on only a few training samples.

Previous few-shot learning works have mainly focused on classification and reinforcement learning.

In this paper, we propose a few-shot meta-learning system that focuses exclusively on regression tasks.

Our model is based on the idea that the degree of freedom of the unknown function can be significantly reduced if it is represented as a linear combination of a set of appropriate basis functions.

This enables a few labelled samples to approximate the function.

We design a Feature Extractor network to encode basis functions for a task distribution, and a  Weights Generator to generate the weight vector for a novel task.

We show that our model outperforms the current state of the art meta-learning methods in various regression tasks.

Regression deals with the problem of learning a model relating a set of inputs to a set of outputs.

The learned model can be thought as function y = F (x) that will give a prediction y ∈ R dy given input x ∈ R dx where d y and d x are dimensions of the output and input respectively.

Typically, a regression model is trained on a large number of data points to be able to provide accurate predictions of new inputs.

Recently, there have been a surge in popularity on few-shot learning methods (Vinyals et al., 2016; BID7 BID4 .

Few-shot learning methods require only a few examples from each task to be able to quickly adapt and perform well on a new task.

The fewshot learning model in essence is learning to learn i.e. the model learns to quickly adapt itself to new tasks rather than just learning to give the correct prediction for a particular input sample.

In this work, we propose a few shot learning model that targets few-shot regression tasks.

We evaluate our model on the sinusoidal regression tasks and compare our model's performance to several meta-learning algorithms.

We further introduce two more regression tasks, namely the 1D heat equation task modeled by partial differential equations and the 2D Gaussian distribution task.

Regression problems has long been a topic of study in the machine learning and signal processing community BID11 BID16 .

Though similar to classification, regression estimates one or multiple scalar values and is usually thought of as a single task problem.

A single model is trained to only perform regression on a only one task.

Our model instead reformulates the regression problem as a few-shot learning problem, allowing for our model to be able to perform regressions of tasks sampled from the same task distribution.

The problem of meta-learning has similarly long been a topic of interest in the general machine learning community (Thrun & Pratt, 1998; BID15 BID12 .

Meta learning has been applied to few-shot learning problem, which is concerned with models that can learn from prior experiences to adapt to new tasks.

BID8 first proposed the one-shot FIG0 : An overview of our model.

Note that during meta-training, we use the true task labels of the regression task as input to the Weights Generator to train both the Weights Generator and Feature Extractor, whereas the generated task labels from the Task Label Generator are only used during meta-testing.classification problem in 2011 and introduced the Omniglot data set as a few-shot classification data set similar to how the MNIST data set (LeCun, 1998) is for traditional classification.

Since then, there has been a surge of few-shot learning methods (Vinyals et al., 2016; BID3 BID4 BID14 , but most of them focus on few shot classification and reinforcement learning domains.

We first discuss our idea.

We aim at developing a model that can rapidly adapt to regress a novel function based on only a few samples from this function.

Specifically, we would like to model the unknown function y = F (x) given only D train = {(x k , y k )|k = 1...K}. With small K, e.g. K = 10, this is an intractable task, as F (x) can take any form.

We follow the common setup: we assume that each function we would like to regress is a task T i drawn from an unknown distribution p(T ).To simplify discussion, we assume scalar input and scalar output.

Our idea is to learn sparse or compressible representation of the unknown function F (x), so that a few samples {(x k , y k )|k = 1...K} can provide adequate information to estimate F (x).

Specifically, we model the unknown function F (x) as a linear combination of a set of basis functions {φ i (x)}: DISPLAYFORM0 Many basis functions have been developed to expand F (x).

For example, the Maclaurin series expansion (Taylor series expansion at x = 0) uses {φ i (x)} = {1, x, x 2 , x 3 , ...}: DISPLAYFORM1 If F (x) is a polynomial, (2) can be a compressible representation, i.e. only a few non-zero/significant w i .

However, if F (x) is a sinusoid, it would require many terms to represent F (x) adequately, e.g.: DISPLAYFORM2 In (3), M is large and M K. Given only K samples {(x k , y k )|k = 1...K}, it is not adequate to determine {w i } and model the unknown function.

On the other hand, if we use the Fourier basis instead, i.e., {φ i (x)} = {1, sin(x), sin(2x), ..., cos(x), cos(2x), ...}, clearly, we can obtain a sparse representation: we can represent a sinusoid with only a few terms.

Under Fourier basis, there are only a few non-zero linear weights w i , and K samples are sufficient to estimate w i and estimate the function.

Essentially, with an appropriate {φ i (x)}, the degree of freedom of F (x) can be significantly reduced when it is modeled using (1), so that K samples can well estimate F (x).Our approach is to use the set of training tasks drawn from p(T ) to learn {φ i (x)} that result in sparse/compressible representation for any task drawn from p(T ).

The set of {φ i (x)} is encoded in the Feature Extractor that takes in x and outputs DISPLAYFORM3 T .

In our framework, Φ(x) is the same for any task drawn from p(T ), as it encodes the set of {φ i (x)} that can sparsely represent any task from p(T ).

We further learn a Weights Generator to map the K training samples of a novel task to a constant vector w = [w 1 , w 2 , ..., w M ]T .

The unknown function is modeled as w T Φ(x).

We hereby introduce our few-shot regression model in detail.

Given a regression task T with DISPLAYFORM0 ..

K}, the model is tasked to predict the entire regression function across a value range.

The training samples, x ∈ R dx first passed though the Feature Extractor which is represented as a function, F (x|θ F ) with trainable parameters θ F .

The Feature Extractor outputs a high dimensional feature representation, f ∈ R d f , where d f is the dimension of the feature representation, for each training point of the task T .

Note that d f is the number of basis functions encoded in the Feature Extractor.

The feature representation x f , together with the labels y ∈ R dy and task labels t ∈ R dt generated from the Task Label Generator are then passed through the Weights Generator.

The Weights Generator, represented as a function G(f , y, t|θ G ), with trainable parameters θ G , outputs a weights vector, w k for each training sample of a regression task.

The final weights vector, w for task T is then obtained by taking a mean of the k weight vectors.

The Weights Generator itself consists a series self-attention blocks with scaled dot product attention introduced by Vaswani et al. (2017) .

Each of the self-attention modules allows the weights generator to "look" at the embedding of the Weights Generator's input to let generator "choose" the parts of the embedding which is most useful in generating the optimal weights for each of the training sample.

The model is then able to make predictions on set of points D pred = {x n |n = 1...N } for task T , by taking a dot product between task-specific weights vector, w and feature representation of the prediction set.

DISPLAYFORM1 Note that for all of our regression experiments, y has a dimension of 1.

However, our model is able to predict regression tasks with higher dimensional y by outputting d y weight vectors from the Weights Generator, the predictions can be obtained by doing a dot product at each dimension of y between the individual weight vectors and f .

An overview of our model can be found in FIG0 .

Outside of the label information y at the sample level, few-shot regression tasks, unlike other fewshot learning tasks also possess additional label information at the task level.

These task level labels are parameters that describe a regression function that we can leverage to improve the performance of a few-shot regression model.

For example, a sinusoidal function has parameters labels such as the amplitude, phase and frequency.

We dub these task-level labels as task labels, t and we use it as an additional input to the Weights Generator.

Though we assume that the model to have access to task labels during the training phase, it is unrealistic to assume that such information will be available or reliable as well during testing.

Therefore, we introduce the Task Label Generator as an additional component to our model.

We represent it as a function T (x f , y|θ T ) with trainable parameters θ T .

It takes in feature representation x f and labels y of the regression task T and attempts to output the correct task labels, t g for T .

Similar to the Weights Generator, we also employ the use of self-attention blocks within the Task Label Generator to enable the Task Label Generator to "look" at parts of the input that are most useful to generating the correct task labels.

We evaluate our model on three few-shot regression tasks.

The first task is the sinusoidal regression task which has been used in many recent few-shot learning papers BID3 BID14 BID10 .

We also introduce two more regression tasks, namely the 1D heat equation task BID2 and the higher dimensional 2D Gaussian distribution task.

For the sinusoidal task we compare our model to Meta-SGD proposed by BID10 .

We also compare our results on sinusoidal task to that of Yoon et al. (2018) where they add a noise component and only train with limited number of tasks for added complexity.

We follow their set up of training of model with only 1000 tasks but we used a pre-trained Task Label Generator instead.

For 1D Heat Equation and 2D Gaussian, we compare our models performance against MAML BID3 and include results of our model with and without using the Task Label Generator.

We calculate mean squared error across 1000 test tasks for all regression tasks and present our results in TAB0 .

Our model manages to outperform both Meta-SGD and BMAML on both variants of the sinusoidal task.

Our model also manage to achieve a superior performance on 1D Heat Equation and 2D Gaussian.

Our results show that even without the use of task labels, our model setup is already able to perform well in the two regression tasks.

Furthermore, we conduct a ablation study to study the effects of adding the Task Label Generator.

We compare three variants of our model an evaluate them on the advanced sinusoidal task .For the first variant when we use the true task labels in both training and testing.

In the second variant, we do not use task labels at all.

Finally we compare the two variants to the base model.

We show the results of this study in TAB1 .

We propose a few-shot meta learning system that focuses exclusively on regression tasks.

Our model is based on the idea of linear representation of basis functions.

We design a Feature extractor network to encode the basis functions for the entire task distribution.

We design a Weight generator network to generate the weights from the K training samples of a novel task drawn from the same task distribution.

We show that our model has competitive performance in in various few short regression tasks.

A TECHNICAL DETAILS

We represent the feature extractor as a a 2 layer 40-dimensional fully connected network which transforms each 1-dimensional input sample into a 40-dimensional feature, Each hidden layer in the feature extractor also followed by a ReLU non-linearity activation function BID13 .

The Weights Generator takes in a 41 + t n dimensional input where t n is the dimension of the t ask label of each task.

The input is passed through a fully connected layer of dimension 64 to produce the embedding for each of the task's training samples.

The extracted embedding is then passed through a series of 3 attention blocks.

Each block is comprised of a dot-product self-attention operation followed by two fully connected layers of dimensions 128 and 64 respectively and finally a layer normalization operation BID1 at the end.

The first fully connected layer after the selfattention module in each block is also followed by a ReLU non-linearity activation function BID13 .

Each block also has a residual connection BID5 from the embedding to the output of the second fully connected layer.

The attention function we used is the scaled dot-product attention introduced by Vaswani et al. FORMULA0 : DISPLAYFORM0 where Q, K and V represent the Query, Key and Value of the Attention module respectively.

d k represents the dimension of K and is used as a scaling factor.

In our case as we using self-attention, Q, K and V are all represented by the task embedding.

The output of the attention blocks are then passed though another fully connected layer of dimension 40 to ensure the same dimensionality as the sample features.

The final weight vector, w is obtained by taking the mean of the output.

The Task Label Generator takes in a 41 dimensional input has a similar architecture to the Weights Generator.

It consists of a 64 dimensional fully connected layer followed by 3 attention blocks similar to that in the Weights Generator.

The Task Label Generator outputs a t n dimensional task label for each training sample and the final task label, t is obtained by taking the mean of all the individual task labels.

The generated task label is then passed through a sigmoid activation layer.

We train our model in two stages.

In the first stage, only the Feature Extractor and Weights Generator are updated.

In this stage we use the true task labels from the regression tasks as input to the Weights Generator.

The loss function used is mean squared error between the labels of the validation set labels and the prediction.

DISPLAYFORM0 Where j is number of samples in the validation set.

As the Feature Extractor is tasked to only map a low dimensional samples x to high dimensional features f , through training the Feature Extractor eventually learns to map samples to features that are most suited for the task domain.

The Feature Extractor can therefore be thought as implicitly modeling and capturing the task distribution p(T ).For any given regression task, the set of features f is fixed.

The reason that the model is able to predict different y value from the same f is purely due to the weights vector w.

As the Weights Generator has access to both sample label and task label information, it is trained to output weights vectors that capture a specific task instance T i within the task distribution p(T ).In the second stage, we train the entire model including the Task Label Generator.

In this stage the predicted task labels from the Task Label Generator are used as input to the Weights Generator instead of the true task labels.

We also use a identical mean-squared loss function for the this stage of training but instead updating all three sets of parameters θ F , θ G and θ T .

Here we provide details on all of our regression tasks and some additional training details of our model.

For all the regression tasks we normalize the task labels to a [0, 1] range using it for training.

We also provide visual results of our model's performance on all three regression tasks in the following section .

We use the amplitude, phase and frequency values to form a 3-dimensional task labels for the sinusoidal task

For the heat equation task, we define it as such: Consider a 1-dimensional rod of length L with both of its ends connected to heat sinks, i.e. the temperature of the ends will always be fixed at 0K unless a heat source is applied at the end.

a constant point heat source is then applied to a random point s on the rod such the the heat point source will always have a temperature of 1.0K.

We would like the model the temperature u(x, t) at each point of the rod a certain time t after applying the heat source until the temperature achieves equilibrium throughout the rod.

The temperature at each point x after time t is given by the heat equation: DISPLAYFORM0 In our experiments, we set L to be 5 and randomly sample 10 points of position range [0, 5] on the heat equation curve for a 10-shot regression task.

We use both the x and t to form a 2 dimensional task label for the heat equation task.

For the 2D Gaussian tasks, the model is trained to predict the probability distribution function of a two-dimensional Gaussian distribution.

We train our model from Gaussian distribution task with mean ranging from (−2, −2) to (2, 2) and standard deviation of range [0.1, 2].

We fix the standard deviation to be of the same value in both directions.

The model is once again trained on 10 randomly sampled input points for a 10-shot regression task.

We plot the predictions of points in the range of [µ − 3 ,µ + 3] where µ is mean of the Gaussian task..

Similar to the heat equation tasks, we use the mean and standard deviation values to form a 3 dimensional task label.

For all of our experiments except the one comparing against BMAML, we train both our model on regression tasks of batch size 32 for 50000 steps for both training stages.

For experiment comparing against BMAML for the sinusoidal task, we train our model on tasks of batch size 10 for 10000 steps using a pre-trained Task Label Generator trained on 50000 training steps.

We also limit the data set in this experiment to just 1000 regression tasks for fair comparison.

We use the Adam Optimizer BID6 as the optimization method to preform stochastic gradient decent on our model.

We implement all our models using the Tensorflow BID0 library and train them on a Nvidia GTX 1080 Ti GPU.

In this section, we further illustrate our justification made for our model by conducting an additional ablation study.

We intend to show that that learned features f indeed correspond to a set of basis functions {φ i (x)} that correspond to the function F (x).

If that is indeed the case, each of the learned basis function φ i (x) should correspond to certain characteristics of the regression function, and removing one basis function from the final prediction should give noticeable difference in the models final prediction.

Thus, we conduct an ablation study where we remove one of the dimensions in f and visualize the change in the final prediction.

We conduct this study on sinusoidal regression task show the results in Figure 2 .

The results shows that removing certain dimensions from the feature vector does give significantly different results and does correspond to certain characteristics of the regression function.

Namely for the first case, the removed feature correspond the accurate magnitude and position of the regression curve whereas for the second case, the removed feature correspond to the "shape" of the regression curve at range [-2.0 ,2.4].dim 9 from f is removed dim 35 from f is removed Figure 2 : Results of the ablation study where we remove one dimension in f from the final prediction and compare the change to that of the original prediction with all of the dimensions in f .

The prediction made with reduced dimension is represented by the blue dashed line.

In this section we show some example visual results of our model's results on the three regression tasks.

In Figure 3 , we show some examples of our models results on the sinusoidal task.

In Figure 4 , we show examples of our model results on the 1D heat equation task and compare it against the results from MAML BID3 .

Finally in Figure 5 , we show examples our model results on the 2D Gaussian task.

MAML BID3 Our ModelFigure 4: Our models performance on the 1D Heat Equation Task.

We also provide a comparison the performace of MAML on the left.

Figure 5: Our Model's results on 2D Gaussian Tasks.

@highlight

We propose a few-shot learning model that is tailored specifically for regression tasks

@highlight

This paper proposes a novel shot-learning method for small sample regression problems.

@highlight

A method that learns a regression model with a few samples and outperforms other methods.