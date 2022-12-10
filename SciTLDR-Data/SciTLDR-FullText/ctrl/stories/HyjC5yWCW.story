Learning to learn is a powerful paradigm for enabling models to learn from data more effectively and efficiently.

A popular approach to meta-learning is to train a recurrent model to read in a training dataset as input and output the parameters of a learned model, or output predictions for new test inputs.

Alternatively, a more recent approach to meta-learning aims to acquire deep representations that can be effectively fine-tuned, via standard gradient descent, to new tasks.

In this paper, we consider the meta-learning problem from the perspective of universality, formalizing the notion of learning algorithm approximation and comparing the expressive power of the aforementioned recurrent models to the more recent approaches that embed gradient descent into the meta-learner.

In particular, we seek to answer the following question: does deep representation combined with standard gradient descent have sufficient capacity to approximate any learning algorithm?

We find that this is indeed true, and further find, in our experiments, that gradient-based meta-learning consistently leads to learning strategies that generalize more widely compared to those represented by recurrent models.

Deep neural networks that optimize for effective representations have enjoyed tremendous success over human-engineered representations.

Meta-learning takes this one step further by optimizing for a learning algorithm that can effectively acquire representations.

A common approach to metalearning is to train a recurrent or memory-augmented model such as a recurrent neural network to take a training dataset as input and then output the parameters of a learner model (Schmidhuber, 1987; Bengio et al., 1992; Li & Malik, 2017a; BID0 .

Alternatively, some approaches pass the dataset and test input into the model, which then outputs a corresponding prediction for the test example (Santoro et al., 2016; Duan et al., 2016; Wang et al., 2016; Mishra et al., 2018) .

Such recurrent models are universal learning procedure approximators, in that they have the capacity to approximately represent any mapping from dataset and test datapoint to label.

However, depending on the form of the model, it may lack statistical efficiency.

In contrast to the aforementioned approaches, more recent work has proposed methods that include the structure of optimization problems into the meta-learner (Ravi & Larochelle, 2017; Finn et al., 2017a; Husken & Goerick, 2000) .

In particular, model-agnostic meta-learning (MAML) optimizes only for the initial parameters of the learner model, using standard gradient descent as the learner's update rule (Finn et al., 2017a) .

Then, at meta-test time, the learner is trained via gradient descent.

By incorporating prior knowledge about gradient-based learning, MAML improves on the statistical efficiency of black-box meta-learners and has successfully been applied to a range of meta-learning problems (Finn et al., 2017a; b; Li et al., 2017) .

But, does it do so at a cost?

A natural question that arises with purely gradient-based meta-learners such as MAML is whether it is indeed sufficient to only learn an initialization, or whether representational power is in fact lost from not learning the update rule.

Intuitively, we might surmise that learning an update rule is more expressive than simply learning an initialization for gradient descent.

In this paper, we seek to answer the following question: does simply learning the initial parameters of a deep neural network have the same representational power as arbitrarily expressive meta-learners that directly ingest the training data at meta-test time?

Or, more concisely, does representation combined with standard gradient descent have sufficient capacity to constitute any learning algorithm?We analyze this question from the standpoint of the universal function approximation theorem.

We compare the theoretical representational capacity of the two meta-learning approaches: a deep network updated with one gradient step, and a meta-learner that directly ingests a training set and test input and outputs predictions for that test input (e.g. using a recurrent neural network).

In studying the universality of MAML, we find that, for a sufficiently deep learner model, MAML has the same theoretical representational power as recurrent meta-learners.

We therefore conclude that, when using deep, expressive function approximators, there is no theoretical disadvantage in terms of representational power to using MAML over a black-box meta-learner represented, for example, by a recurrent network.

Since MAML has the same representational power as any other universal meta-learner, the next question we might ask is: what is the benefit of using MAML over any other approach?

We study this question by analyzing the effect of continuing optimization on MAML performance.

Although MAML optimizes a network's parameters for maximal performance after a fixed small number of gradient steps, we analyze the effect of taking substantially more gradient steps at meta-test time.

We find that initializations learned by MAML are extremely resilient to overfitting to tiny datasets, in stark contrast to more conventional network initialization, even when taking many more gradient steps than were used during meta-training.

We also find that the MAML initialization is substantially better suited for extrapolation beyond the distribution of tasks seen at meta-training time, when compared to meta-learning methods based on networks that ingest the entire training set.

We analyze this setting empirically and provide some intuition to explain this effect.

In this section, we review the universal function approximation theorem and its extensions that we will use when considering the universal approximation of learning algorithms.

We also overview the model-agnostic meta-learning algorithm and an architectural extension that we will use in Section 4.

The universal function approximation theorem states that a neural network with one hidden layer of finite width can approximate any continuous function on compact subsets of R n up to arbitrary precision (Hornik et al., 1989; Cybenko, 1989; Funahashi, 1989) .

The theorem holds for a range of activation functions, including the sigmoid (Hornik et al., 1989) and ReLU (Sonoda & Murata, 2017) functions.

A function approximator that satisfies the definition above is often referred to as a universal function approximator (UFA).

Similarly, we will define a universal learning procedure approximator to be a UFA with input (D, x ) and output y , where (D, x ) denotes the training dataset and test input, while y denotes the desired test output.

Furthermore, Hornik et al. (1990) showed that a neural network with a single hidden layer can simultaneously approximate any function and its derivatives, under mild assumptions on the activation function used and target function's domain.

We will use this property in Section 4 as part of our meta-learning universality result.

Model-Agnostic Meta-Learning (MAML) is a method that proposes to learn an initial set of parameters θ such that one or a few gradient steps on θ computed using a small amount of data for one task leads to effective generalization on that task (Finn et al., 2017a) .

Tasks typically correspond to supervised classification or regression problems, but can also correspond to reinforcement learning problems.

The MAML objective is computed over many tasks {T j } as follows: DISPLAYFORM0 where D Tj corresponds to a training set for task T j and the outer loss evaluates generalization on test data in D Tj .

The inner optimization to compute θ Tj can use multiple gradient steps; though, in this paper, we will focus on the single gradient step setting.

After meta-training on a wide range of tasks, the model can quickly and efficiently learn new, held-out test tasks by running gradient descent starting from the meta-learned representation θ.

While MAML is compatible with any neural network architecture and any differentiable loss function, recent work has observed that some architectural choices can improve its performance.

A particularly effective modification, introduced by Finn et al. (2017b) , is to concatenate a vector of parameters, θ b , to the input.

As with all other model parameters, θ b is updated in the inner loop via gradient descent, and the initial value of θ b is meta-learned.

This modification, referred to as a bias transformation, increases the expressive power of the error gradient without changing the expressivity of the model itself.

While Finn et al. (2017b) report empirical benefit from this modification, we will use this architectural design as a symmetry-breaking mechanism in our universality proof.

We can broadly classify RNN-based meta-learning methods into two categories.

In the first approach (Santoro et al., 2016; Duan et al., 2016; Wang et al., 2016; Mishra et al., 2018) , there is a meta-learner model g with parameters φ which takes as input the dataset D T for a particular task T and a new test input x , and outputs the estimated outputŷ for that input: DISPLAYFORM0 The meta-learner g is typically a recurrent model that iterates over the dataset D and the new input x .

For a recurrent neural network model that satisfies the UFA theorem, this approach is maximally expressive, as it can represent any function on the dataset D T and test input x .In the second approach (Hochreiter et al., 2001; Bengio et al., 1992; Li & Malik, 2017b; BID0 Ravi & Larochelle, 2017; Ha et al., 2017) , there is a meta-learner g that takes as input the dataset for a particular task D T and the current weights θ of a learner model f , and outputs new parameters θ T for the learner model.

Then, the test input x is fed into the learner model to produce the predicted outputŷ .

The process can be written as follows: DISPLAYFORM1 Note that, in the form written above, this approach can be as expressive as the previous approach, since the meta-learner could simply copy the dataset into some of the predicted weights, reducing to a model that takes as input the dataset and the test example.

1 Several versions of this approach, i.e. Ravi & Larochelle (2017); Li & Malik (2017b) , have the recurrent meta-learner operate on order-invariant features such as the gradient and objective value averaged over the datapoints in the dataset, rather than operating on the individual datapoints themselves.

This induces a potentially helpful inductive bias that disallows coupling between datapoints, ignoring the ordering within the dataset.

As a result, the meta-learning process can only produce permutation-invariant functions of the dataset.

In model-agnostic meta-learning (MAML), instead of using an RNN to update the weights of the learner f , standard gradient descent is used.

Specifically, the predictionŷ for a test input x is: DISPLAYFORM2 where θ denotes the initial parameters of the model f and also corresponds to the parameters that are meta-learned, and corresponds to a loss function with respect to the label and prediction.

Since the RNN approaches can approximate any update rule, they are clearly at least as expressive as gradient descent.

It is less obvious whether or not the MAML update imposes any constraints on the learning procedures that can be acquired.

To study this question, we define a universal learning procedure approximator to be a learner which can approximate any function of the set of training datapoints D T and the test point x .

It is clear how f MAML can approximate any function on x , as per the UFA theorem; however, it is not obvious if f MAML can represent any function of the set of input, output pairs in D T , since the UFA theorem does not consider the gradient operator.

The first goal of this paper is to show that f MAML (D T , x ; θ) is a universal function approximator of (D T , x ) in the one-shot setting, where the dataset D T consists of a single datapoint (x, y).

Then, we will consider the case of K-shot learning, showing that f MAML (D T , x ; θ) is universal in the set of functions that are invariant to the permutation of datapoints.

In both cases, we will discuss meta supervised learning problems with both discrete and continuous labels and the loss functions under which universality does or does not hold.

We first introduce a proof of the universality of gradient-based meta-learning for the special case with only one training point, corresponding to one-shot learning.

We denote the training datapoint as (x, y), and the test input as x .

A universal learning algorithm approximator corresponds to the ability of a meta-learner to represent any function f target (x, y, x ) up to arbitrary precision.

We will proceed by construction, showing that there exists a neural network functionf (·; θ) such that f (x ; θ ) approximates f target (x, y, x ) up to arbitrary precision, where θ = θ − α∇ θ (y, f (x)) and α is the non-zero learning rate.

The proof holds for a standard multi-layer ReLU network, provided that it has sufficient depth.

As we discuss in Section 6, the loss function cannot be any loss function, but the standard cross-entropy and mean-squared error objectives are both suitable.

In this proof, we will start by presenting the form off and deriving its value after one gradient step.

Then, to show universality, we will construct a setting of the weight matrices that enables independent control of the information flow coming forward from x and x , and backward from y.

We will start by constructingf , which, as shown in FIG0 is a generic deep network with N + 2 layers and ReLU nonlinearities.

Note that, for a particular weight matrix W i at layer i, a single gradient step W i − α∇

Wi can only represent a rank-1 update to the matrix W i .

That is because the gradient of W i is the outer product of two vectors, DISPLAYFORM0 , where a i is the error gradient with respect to the pre-synaptic activations at layer i, and b i−1 is the forward post-synaptic activations at layer i − 1.

The expressive power of a single gradient update to a single weight matrix is therefore quite limited.

However, if we sequence N weight matrices as N i=1 W i , corresponding to multiple linear layers, it is possible to acquire a rank-N update to the linear function represented by W = N i=1 W i .

Note that deep ReLU networks act like deep linear networks when the input and pre-synaptic activations are non-negative.

Motivated by this reasoning, we will constructf (·; θ) as a deep ReLU network where a number of the intermediate layers act as linear layers, which we ensure by showing that the input and pre-synaptic activations of these layers are non-negative.

This allows us to simplify the analysis.

The simplified form of the model is as follows: DISPLAYFORM1 where φ(·; θ ft , θ b ) represents an input feature extractor with parameters θ ft and a scalar bias transformation variable θ b , N i=1 W i is a product of square linear weight matrices, f out (·, θ out ) is a function at the output, and the learned parameters are θ := {θ ft , θ b , {W i }, θ out }.

The input feature extractor and output function can be represented with fully connected neural networks with one or more hidden layers, which we know are universal function approximators, while N i=1 W i corresponds to a set of linear layers with non-negative input and activations.

Next, we derive the form of the post-update predictionf (x ; θ ).

DISPLAYFORM2 , and the error gradient ∇ z = e(x, y).

Then, the gradient with respect to each weight matrix W i is: DISPLAYFORM3 Therefore, the post-update value of DISPLAYFORM4 where we will disregard the last term, assuming that α is comparatively small such that α 2 and all higher order terms vanish.

In general, these terms do not necessarily need to vanish, and likely would further improve the expressiveness of the gradient update, but we disregard them here for the sake of the simplicity of the derivation.

Ignoring these terms, we now note that the post-update value of z when x is provided as input intof (·; θ ) is given by DISPLAYFORM5 Our goal is to show that that there exists a setting of W i , f out , and φ for which the above function, f (x , θ ), can approximate any function of (x, y, x ).

To show universality, we will aim to independently control information flow from x, from y, and from x by multiplexing forward information from x and backward information from y. We will achieve this by decomposing W i , φ, and the error gradient into three parts, as follows: DISPLAYFORM6 ( 2) where the initial value of θ b will be 0.

The top components all have equal numbers of rows, as do the middle components.

As a result, we can see that z will likewise be made up of three components, which we will denote asz, z, andž.

Lastly, we construct the top component of the error gradient to be 0, whereas the middle and bottom components, e(y) andě(y), can be set to be any linear (but not affine) function of y. We will discuss how to achieve this gradient in the latter part of this section when we define f out and in Section 6.In Appendix A.3, we show that we can choose a particular form ofW i , W i , andw i that will simplify the products of W j matrices in Equation 1, such that we get the following form for z : DISPLAYFORM7 where A 1 = I, B N = I, A i can be chosen to be any symmetric positive-definite matrix, and B i can be chosen to be any positive definite matrix.

In Appendix D, we further show that these definitions of the weight matrices satisfy the condition that the activations are non-negative, meaning that the modelf can be represented by a generic deep network with ReLU nonlinearities.

Finally, we need to define the function f out at the output.

When the training input x is passed in, we need f out to propagate information about the label y as defined in Equation 2.

And, when the test input x is passed in, we need a different function defined only on z .

Thus, we will define f out as a neural network that approximates the following multiplexer function and its derivatives (as shown possible by Hornik et al. (1990) ): DISPLAYFORM8 where g pre is a linear function with parameters θ g such that ∇ z = e(y) satisfies Equation 2 (see Section 6) and h post (·; θ h ) is a neural network with one or more hidden layers.

As shown in Appendix A.4, the post-update value of f out is DISPLAYFORM9 Now, combining Equations 3 and 5, we can see that the post-update value is the following: DISPLAYFORM10 In summary, so far, we have chosen a particular form of weight matrices, feature extractor, and output function to decouple forward and backward information flow and recover the post-update function above.

Now, our goal is to show that the above functionf (x ; θ ) is a universal learning algorithm approximator, as a function of (x, y, x ).

For notational clarity, we will use DISPLAYFORM11 to denote the inner product in the above equation, noting that it can be viewed as a type of kernel with the RKHS defined by B iφ (x; θ ft , θ b ).2 The connection to kernels is not in fact needed for the proof, but provides for convenient notation and an interesting observation.

We then define the following lemma:Lemma 4.1 Let us assume that e(y) can be chosen to be any linear (but not affine) function of y. Then, we can choose θ ft , θ h , {A i ; i > 1}, {B i ; i < N } such that the function DISPLAYFORM12 can approximate any continuous function of (x, y, x ) on compact subsets of R dim(y) .

Intuitively, Equation 7 can be viewed as a sum of basis vectors A i e(y) weighted by k i (x, x ), which is passed into h post to produce the output.

There are likely a number of ways to prove Lemma 4.1.

In Appendix A.1, we provide a simple though inefficient proof, which we will briefly summarize here.

We can define k i to be a indicator function, indicating when (x, x ) takes on a particular value indexed by i. Then, we can define A i e(y) to be a vector containing the information of y and i. Then, the result of the summation will be a vector containing information about the label y and the value of (x, x ) which is indexed by i. Finally, h post defines the output for each value of (x, y, x ).

The bias transformation variable θ b plays a vital role in our construction, as it breaks the symmetry within k i (x, x ).

Without such asymmetry, it would not be possible for our constructed function to represent any function of x and x after one gradient step.

In conclusion, we have shown that there exists a neural network structure for whichf (x ; θ ) is a universal approximator of f target (x, y, x ).

We chose a particular form off (·; θ) that decouples forward and backward information flow.

With this choice, it is possible to impose any desired post-update function, even in the face of adversarial training datasets and loss functions, e.g. when the gradient points in the wrong direction.

If we make the assumption that the inner loss function and training dataset are not chosen adversarially and the error gradient points in the direction of improvement, it is likely that a much simpler architecture will suffice that does not require multiplexing of forward and backward information in separate channels.

Informative loss functions and training data allowing for simpler functions is indicative of the inductive bias built into gradient-based meta-learners, which is not present in recurrent meta-learners.

Our result in this section implies that a sufficiently deep representation combined with just a single gradient step can approximate any one-shot learning algorithm.

In the next section, we will show the universality of MAML for K-shot learning algorithms.

Now, we consider the more general K-shot setting, aiming to show that MAML can approximate any permutation invariant function of a dataset and test datapoint ({(x, y) i ; i ∈ 1...K}, x ) for K > 1.

Note that K does not need to be small.

To reduce redundancy, we will only overview the differences from the 1-shot setting in this section.

We include a full proof in Appendix B.In the K-shot setting, the parameters off (·, θ) are updated according to the following rule: DISPLAYFORM0 Defining the form off to be the same as in Section 4, the post-update function is the following: DISPLAYFORM1 In Appendix C, we show one way in which this function can approximate any function of ({(x, y) k ; k ∈ 1...K}, x ) that is invariant to the ordering of the training datapoints {(x, y) k ; k ∈ 1...

K}. We do so by showing that we can select a setting ofφ and of each A i and B i such that z is a vector containing a discretization of x and frequency counts of the discretized datapoints 4 .

If z is a vector that completely describes ({(x, y) i }, x ) without loss of information and because h post is a universal function approximator,f (x ; θ ) can approximate any continuous function of ({(x, y) i }, x ) on compact subsets of R dim(y) .

It's also worth noting that the form of the above equation greatly resembles a kernel-based function approximator around the training points, and a substantially more efficient universality proof can likely be obtained starting from this premise.

In the previous sections, we showed that a deep representation combined with gradient descent can approximate any learning algorithm.

In this section, we will discuss the requirements that the loss function must satisfy in order for the results in Sections 4 and 5 to hold.

As one might expect, the main requirement will be for the label to be recoverable from the gradient of the loss.

As seen in the definition of f out in Equation 4, the pre-update functionf (x, θ) is given by g pre (z; θ g ), where g pre is used for back-propagating information about the label(s) to the learner.

As stated in Equation 2, we require that the error gradient with respect to z to be: DISPLAYFORM0 and where e(y) andě(y) must be able to represent [at least] any linear function of the label y.

We define g pre as follows: DISPLAYFORM1 To make the top term of the gradient equal to 0, we can setW g to be 0, which causes the pre-update predictionŷ =f (x, θ) to be 0.

Next, note that e(y) = W T g ∇ŷ (y,ŷ) andě(y) =w T g ∇ŷ (y,ŷ).

Thus, for e(y) to be any linear function of y, we require a loss function for which ∇ŷ (y, 0) is a linear function Ay, where A is invertible.

Essentially, y needs to be recoverable from the loss function's gradient.

In Appendix E and F, we prove the following two theorems, thus showing that the standard 2 and cross-entropy losses allow for the universality of gradient-based meta-learning.

Theorem 6.1 The gradient of the standard mean-squared error objective evaluated atŷ = 0 is a linear, invertible function of y.

Theorem 6.2 The gradient of the softmax cross entropy loss with respect to the pre-softmax logits is a linear, invertible function of y, when evaluated at 0.

Now consider other popular loss functions whose gradients do not satisfy the label-linearity property.

The gradients of the 1 and hinge losses are piecewise constant, and thus do not allow for universality.

The Huber loss is also piecewise constant in some areas its domain.

These error functions effectively lose information because simply looking at their gradient is insufficient to determine the label.

Recurrent meta-learners that take the gradient as input, rather than the label, e.g. BID0 , will also suffer from this loss of information when using these error functions.

Now that we have shown that meta-learners that use standard gradient descent with a sufficiently deep representation can approximate any learning procedure, and are equally expressive as recurrent learners, a natural next question is -is there empirical benefit to using one meta-learning approach versus another, and in which cases?

To answer this question, we next aim to empirically study the inductive bias of gradient-based and recurrent meta-learners.

Then, in Section 7.2, we will investigate the role of model depth in gradient-based meta-learning, as the theory suggests that deeper networks lead to increased expressive power for representing different learning procedures.

First, we aim to empirically explore the differences between gradient-based and recurrent metalearners.

In particular, we aim to answer the following questions: (1) can a learner trained with MAML further improve from additional gradient steps when learning new tasks at test time, or does it start to overfit?

and (2) does the inductive bias of gradient descent enable better few-shot learning performance on tasks outside of the training distribution, compared to learning algorithms represented as recurrent networks?

Figure 4 : Comparison of finetuning from a MAML-initialized network and a network initialized randomly, trained from scratch.

Both methods achieve about the same training accuracy.

But, MAML also attains good test accuracy, while the network trained from scratch overfits catastrophically to the 20 examples.

Interestingly, the MAMLinitialized model does not begin to overfit, even though meta-training used 5 steps while the graph shows up to 100.To study both questions, we will consider two simple fewshot learning domains.

The first is 5-shot regression on a family of sine curves with varying amplitude and phase.

We trained all models on a uniform distribution of tasks with amplitudes A ∈ [0.1, 5.0], and phases γ ∈ [0, π].

The second domain is 1-shot character classification using the Omniglot dataset (Lake et al., 2011) , following the training protocol introduced by Santoro et al. (2016) .

In our comparisons to recurrent meta-learners, we will use two state-of-the-art meta-learning models: SNAIL (Mishra et al., 2018) and metanetworks (Munkhdalai & Yu, 2017) .

In some experiments, we will also compare to a task-conditioned model, which is trained to map from both the input and the task description to the label.

Like MAML, the task-conditioned model can be fine-tuned on new data using gradient descent, but is not trained for few-shot adaptation.

We include more experimental details in Appendix G.To answer the first question, we fine-tuned a model trained using MAML with many more gradient steps than used during meta-training.

The results on the sinusoid domain, shown in FIG1 , show that a MAML-learned initialization trained for fast adaption in 5 steps can further improve beyond 5 gradient steps, especially on out-of-distribution tasks.

In contrast, a task-conditioned model trained without MAML can easily overfit to out-of-distribution tasks.

With the Omniglot dataset, as seen in Figure 4 , a MAML model that was trained with 5 inner gradient steps can be fine-tuned for 100 gradient steps without leading to any drop in test accuracy.

As expected, a model initialized randomly and trained from scratch quickly reaches perfect training accuracy, but overfits massively to the 20 examples.

Next, we investigate the second question, aiming to compare MAML with state-of-the-art recurrent meta-learners on tasks that are related to, but outside of the distribution of the training tasks.

All three methods achieved similar performance within the distribution of training tasks for 5-way 1-shot Omniglot classification and 5-shot sinusoid regression.

In the Omniglot setting, we compare each method's ability to distinguish digits that have been sheared or scaled by varying amounts.

In the sinusoid regression setting, we compare on sinusoids with extrapolated amplitudes within [5.0, 10.0] and phases within [π, 2π] .

The results in FIG2 and Appendix G show a clear trend that MAML recovers more generalizable learning strategies.

Combined with the theoretical universality results, these experiments indicate that deep gradient-based meta-learners are not only equivalent in representational power to recurrent meta-learners, but should also be a considered as a strong contender in settings that contain domain shift between meta-training and meta-testing tasks, where their strong inductive bias for reasonable learning strategies provides substantially improved performance.

The proofs in Sections 4 and 5 suggest that gradient descent with deeper representations results in more expressive learning procedures.

In contrast, the universal function approximation theorem only requires a single hidden layer to approximate any function.

Now, we seek to empirically explore this theoretical finding, aiming to answer the question: is there a scenario for which model-agnostic meta-learning requires a deeper representation to achieve good performance, compared to the depth of the representation needed to solve the underlying tasks being learned?

Figure 5 : Comparison of depth while keeping the number of parameters constant.

Task-conditioned models do not need more than one hidden layer, whereas meta-learning with MAML clearly benefits from additional depth.

Error bars show standard deviation over three training runs.

To answer this question, we will study a simple regression problem, where the meta-learning goal is to infer a polynomial function from 40 input/output datapoints.

We use polynomials of degree 3 where the coefficients and bias are sampled uniformly at random within [−1, 1] and the input values range within [−3, 3] .

Similar to the conditions in the proof, we meta-train and meta-test with one gradient step, use a mean-squared error objective, use ReLU nonlinearities, and use a bias transformation variable of dimension 10.

To compare the relationship between depth and expressive power, we will compare models with a fixed number of parameters, approximately 40, 000, and vary the network depth from 1 to 5 hidden layers.

As a point of comparison to the models trained for meta-learning using MAML, we trained standard feedforward models to regress from the input and the 4-dimensional task description (the 3 coefficients of the polynomial and the scalar bias) to the output.

These task-conditioned models act as an oracle and are meant to empirically determine the depth needed to represent these polynomials, independent of the meta-learning process.

Theoretically, we would expect the task-conditioned models to require only one hidden layer, as per the universal function approximation theorem.

In contrast, we would expect the MAML model to require more depth.

The results, shown in Figure 5 , demonstrate that the task-conditioned model does indeed not benefit from having more than one hidden layer, whereas the MAML clearly achieves better performance with more depth even though the model capacity, in terms of the number of parameters, is fixed.

This empirical effect supports the theoretical finding that depth is important for effective meta-learning using MAML.

In this paper, we show that there exists a form of deep neural network such that the initial weights combined with gradient descent can approximate any learning algorithm.

Our findings suggest that, from the standpoint of expressivity, there is no theoretical disadvantage to embedding gradient descent into the meta-learning process.

In fact, in all of our experiments, we found that the learning strategies acquired with MAML are more successful when faced with out-of-domain tasks compared to recurrent learners.

Furthermore, we show that the representations acquired with MAML are highly resilient to overfitting.

These results suggest that gradient-based meta-learning has a num-ber of practical benefits, and no theoretical downsides in terms of expressivity when compared to alternative meta-learning models.

Independent of the type of meta-learning algorithm, we formalize what it means for a meta-learner to be able to approximate any learning algorithm in terms of its ability to represent functions of the dataset and test inputs.

This formalism provides a new perspective on the learning-to-learn problem, which we hope will lead to further discussion and research on the goals and methodology surrounding meta-learning.

While there are likely a number of ways to prove Lemma 4.1 (copied below for convenience), here we provide a simple, though inefficient, proof of Lemma 4.1.Lemma 4.1 Let us assume that e(y) can be chosen to be any linear (but not affine) function of y. Then, we can choose θ ft , θ h , {A i ; i > 1}, {B i ; i < N } such that the function DISPLAYFORM0 can approximate any continuous function of (x, y, x ) on compact subsets of R dim(y) .

To prove this lemma, we will proceed by showing that we can choose e, θ ft , and each A i and B i such that the summation contains a complete description of the values of x, x , and y. Then, because h post is a universal function approximator,f (x , θ ) will be able to approximate any function of x, x , and y.

Since A 1 = I and B N = I, we will essentially ignore the first and last elements of the sum by defining B 1 := I and A N := I, where is a small positive constant to ensure positive definiteness.

Then, we can rewrite the summation, omitting the first and last terms: DISPLAYFORM0 Next, we will re-index using two indexing variables, j and l, where j will index over the discretization of x and l over the discretization of x .

DISPLAYFORM1 Next, we will define our chosen form of k jl in Equation 8.

We show how to acquire this form in the next section.

Lemma A.1 We can choose θ ft and each B jl such that DISPLAYFORM2 where discr(·) denotes a function that produces a one-hot discretization of its input and e denotes the 0-indexed standard basis vector.

Now that we have defined the function k jl , we will next define the other terms in the sum.

Our goal is for the summation to contain complete information about (x, x , y).

To do so, we will chose e(y) to be the linear function that outputs J * L stacked copies of y. Then, we will define A jl to be a matrix that selects the copy of y in the position corresponding to (j, l), i.e. in the position j + J *

l.

This can be achieved using a diagonal A jl matrix with diagonal values of 1 + at the positions corresponding to the kth vector, and elsewhere, where k = (j + J * l) and is used to ensure that A jl is positive definite.

As a result, the post-update function is as follows: DISPLAYFORM3 where y is at the position j + J * l within the vector v(x, x , y), where j satisfies discr(x) = e j and where l satisfies discr(x ) = e l .

Note that the vector −αv(x, x , y) is a complete description of (x, x , y) in that x, x , and y can be decoded from it.

Therefore, since h post is a universal function approximator and because its input contains all of the information of (x, x , y), the function f (x ; θ ) ≈ h post (−αv(x, x , y); θ h ) is a universal function approximator with respect to its inputs (x, x , y).A.2 PROOF OF LEMMA A.1In this section, we show one way of proving Lemma A.1:Lemma A.1 We can choose θ ft and each B jl such that DISPLAYFORM4 where discr(·) denotes a function that produces a one-hot discretization of its input and e denotes the 0-indexed standard basis vector.

DISPLAYFORM5 , where θ b = 0.

Since the gradient with respect to θ b can be chosen to be any linear function of the label y (see Section 6), we can assume without loss of generality that θ b = 0.We will chooseφ and B jl as follows: DISPLAYFORM6 where we use E ik to denote the matrix with a 1 at (i, k) and 0 elsewhere, and I is added to ensure the positive definiteness of B jl as required in the construction.

Using the above definitions, we can see that: DISPLAYFORM7 Thus, we have proved the lemma, showing that we can choose aφ and each B jl such that: DISPLAYFORM8

The goal of this section is to show that we can choose a form ofW , W , andw such that we can simplify the form of z in Equation 1 into the following: DISPLAYFORM0 where DISPLAYFORM1 for i < N and B N = I. Recall that we decomposed W i , φ, and the error gradient into three parts, as follows: DISPLAYFORM2 where the initial value of θ b will be 0.

The top components,W i andφ, have equal dimensions, as do the middle components, W i and 0.

The bottom components are scalars.

As a result, we can see that z will likewise be made up of three components, which we will denote asz, z, andž, where, before the gradient update,z = N i=1W iφ (x; θ ft ), z = 0, andž = 0.

Lastly, we construct the top component of the error gradient to be 0, whereas the middle and bottom components, e(y) anď e(y), can be set to be any linear (but not affine) function of y.

Using the above definitions and noting that θ ft = θ ft − α∇ θft = θ ft , we can simplify the form of z in Equation 1, such that the middle component, z , is the following: DISPLAYFORM3 We aim to independently control the backward information flow from the gradient e and the forward information flow fromφ.

Thus, choosing allW i and W i to be square and full rank, we will set DISPLAYFORM4 for i ∈ {1...N } whereM N +1 = I and M 0 = I. Then we can again simplify the form of z : DISPLAYFORM5 where DISPLAYFORM6

In this appendix, we provide a full proof of the universality of gradient-based meta-learning in the general case with K > 1 datapoints.

This proof will share a lot of content from the proof in the 1-shot setting, but we include it for completeness.

We aim to show that a deep representation combined with one step of gradient descent can approximate any permutation invariant function of a dataset and test datapoint ({(x, y) i ; i ∈ 1...K}, x ) for K > 1.

Note that K does not need to be small.

We will proceed by construction, showing that there exists a neural network function f (·; θ) such thatf (·; θ ) approximates f target ({(x, y) k }, x ) up to arbitrary precision, where DISPLAYFORM0 and α is the learning rate.

As we discuss in Section 6, the loss function cannot be any loss function, but the standard cross-entropy and mean-squared error objectives are both suitable.

In this proof, we will start by presenting the form off and deriving its value after one gradient step.

Then, to show universality, we will construct a setting of the weight matrices that enables independent control of the information flow coming forward from the inputs {x k } and x , and backward from the labels {y k }.We will start by constructingf .

With the same motivation as in Section 4, we will constructf (·; θ) as the following:f DISPLAYFORM1 φ(·; θ ft , θ b ) represents an input feature extractor with parameters θ ft and a scalar bias transformation variable θ b , N i=1 W i is a product of square linear weight matrices, f out (·, θ out ) is a readout function at the output, and the learned parameters are θ := {θ ft , θ b , {W i }, θ out }.

The input feature extractor and readout function can be represented with fully connected neural networks with one or more hidden layers, which we know are universal function approximators, while N i=1 W i corresponds to a set of linear layers.

Note that deep ReLU networks act like deep linear networks when the input and pre-synaptic activations are non-negative.

We will later show that this is indeed the case within these linear layers, meaning that the neural network functionf is fully generic and can be represented by deep ReLU networks, as visualized in FIG0 .Next, we will derive the form of the post-update predictionf (x ; θ ).

DISPLAYFORM2 and we denote its gradient with respect to the loss as ∇ z k = e(x k , y k ).

The gradient with respect to any of the weight matrices W i for a single datapoint (x, y) is given by DISPLAYFORM3 Therefore, the post-update value of DISPLAYFORM4 where we move the summation over k to the left and where we will disregard the last term, assuming that α is comparatively small such that α 2 and all higher order terms vanish.

In general, these terms do not necessarily need to vanish, and likely would further improve the expressiveness of the gradient update, but we disregard them here for the sake of the simplicity of the derivation.

Ignoring these terms, we now note that the post-update value of z when x is provided as input intof (·; θ ) is given by DISPLAYFORM5 DISPLAYFORM6 Our goal is to show that that there exists a setting of W i , f out , and φ for which the above function,f (x , θ ), can approximate any function of ({(x, y) k }, x ).

To show universality, we will aim independently control information flow from {x k }, from {y k }, and from x by multiplexing forward information from {x k } and x and backward information from {y k }.

We will achieve this by decomposing W i , φ, and the error gradient into three parts, as follows: DISPLAYFORM7 where the initial value of θ b will be 0.

The top components all have equal numbers of rows, as do the middle components.

As a result, we can see that z k will likewise be made up of three components, which we will denote asz k , z k , andž k .

Lastly, we construct the top component of the error gradient to be 0, whereas the middle and bottom components, e(y k ) andě(y k ), can be set to be any linear (but not affine) function of y k .

We discuss how to achieve this gradient in the latter part of this section when we define f out and in Section 6.

connection to kernels is not in fact needed for the proof, but provides for convenient notation and an interesting observation.

We can now simplify the form off (x , θ ) as the following equation: DISPLAYFORM8 Intuitively, Equation 20 can be viewed as a sum of basis vectors A i e(y k ) weighted by k i (x k , x ), which is passed into h post to produce the output.

In Appendix C, we show that we can choose e, θ ft , θ h , each A i , and each B i such that Equation 20 can approximate any continuous function of ({(x, y) k }, x ) on compact subsets of R dim(y) .

As in the one-shot setting, the bias transformation variable θ b plays a vital role in our construction, as it breaks the symmetry within k i (x, x ).

Without such asymmetry, it would not be possible for our constructed function to represent any function of x and x after one gradient step.

In conclusion, we have shown that there exists a neural network structure for whichf (x ; θ ) is a universal approximator of f target ({(x, y) k }, x ).

In Section 5 and Appendix B, we showed that the post-update functionf (x ; θ ) takes the following form:f DISPLAYFORM0 In this section, we aim to show that the above form off (x ; θ ) can approximate any function of {(x, y) k ; k ∈ 1...K} and x that is invariant to the ordering of the training datapoints {(x, y) k ; k ∈ 1...

K}.

The proof will be very similar to the one-shot setting proof in Appendix A.1Similar to Appendix A.1, we will ignore the first and last elements of the sum by defining B 1 to be I and A N to be I, where is a small positive constant to ensure positive definiteness.

We will then re-index the first summation over i = 2...

N − 1 to instead use two indexing variables j and l as follows:f DISPLAYFORM1 As in Appendix A.1, we will define the function k jl to be an indicator function over the values of x k and x .

In particular, we will reuse Lemma A.1, which was proved in Appendix A.2 and is copied below:Lemma A.1 We can choose θ ft and each B jl such that DISPLAYFORM2 where discr(·) denotes a function that produces a one-hot discretization of its input and e denotes the 0-indexed standard basis vector.

Likewise, we will chose e(y k ) to be the linear function that outputs J * L stacked copies of y k .

Then, we will define A jl to be a matrix that selects the copy of y k in the position corresponding to (j, l), i.e. in the position j + J *

l.

This can be achieved using a diagonal A jl matrix with diagonal values of 1 + at the positions corresponding to the nth vector, and elsewhere, where n = (j + J * l) and is used to ensure that A jl is positive definite.

Published as a conference paper at ICLR 2018As a result, the post-update function is as follows: DISPLAYFORM3 where y k is at the position j + J * l within the vector v(x k , x , y k ), where j satisfies discr(x k ) = e j and where l satisfies discr(x k ) = e l .For discrete, one-shot labels y k , the summation over v amounts to frequency counts of the triplets (x k , x , y k ).

In the setting with continuous labels, we cannot attain frequency counts, as we do not have access to a discretized version of the label.

Thus, we must make the assumption that no two datapoints share the same input value x k .

With this assumption, the summation over v will contain the output values y k at the index corresponding to the value of (x k , x ).

For both discrete and continuous labels, this representation is redundant in x , but nonetheless contains sufficient information to decode the test input x and set of datapoints {(x, y) k } (but not the order of datapoints).Since h post is a universal function approximator and because its input contains all of the information DISPLAYFORM4 ; θ h is a universal function approximator with respect to {(x, y) k } and x .

In this appendix, we show that the network architecture with linear layers analyzed in the Sections 4 and 5 can be represented by a deep network with ReLU nonlinearities.

We will do so by showing that the input and activations within the linear layers are all non-negative.

First, consider the input φ(·; θ ft , θ b ) andφ(·; θ ft , θ b ).

The inputφ(·; θ ft , θ b ) is defined to consist of three terms.

The top term,φ is defined in Appendices A.2 and C to be a discretization (which is non-negative) both before and after the parameters are updated.

The middle term is defined to be a constant 0.

The bottom term, θ b , is defined to be 0 before the gradient update and is not used afterward.

Next, consider the weight matrices, W i .

To determine that the activations are non-negative, it is now sufficient to show that the products DISPLAYFORM0 To do so, we need to show that the products N i=jW i , N i=j W i , and N i=jw i are PSD for j = 1, ..., N .

In Appendix A.2, each B i =M i+1 is defined to be positive definite; and in Appendix A.3, we define the products N i=j+1W i =M j + 1.

Thus, the conditions on the products ofW i are satisfied.

In Appendices A.1 and C, each A i are defined to be symmetric positive definite matrices.

In Appendix A.3, we define DISPLAYFORM1 Thus, we can see that each M i is also symmetric positive definite, and therefore, each W i is positive definite.

Finally, the purpose of the weightsw i is to provide nonzero gradients to the input θ b , thus a positive value for eachw i will suffice.

E PROOF OF THEOREM 6.1Here we provide a proof of Theorem 6.1: Theorem 6.1 The gradient of the standard mean-squared error objective evaluated atŷ = 0 is a linear, invertible function of y.

For the standard mean-squared error objective, (y,ŷ) = 1 2 y −ŷ 2 , the gradient is ∇ŷ (y, 0) = −y, which satisfies the requirement, as A = −I is invertible.

There is a clear trend that gradient descent enables better generalization on out-of-distribution tasks compared to the learning strategies acquired using recurrent meta-learners such as SNAIL.

Right: Here is another example that shows the resilience of a MAML-learned initialization to overfitting.

In this case, the MAML model was trained using one inner step of gradient descent on 5-way, 1-shot Omniglot classification.

Both a MAML and random initialized network achieve perfect training accuracy.

As expected, the model trained from scratch catastrophically overfits to the 5 training examples.

However, the MAML-initialized model does not begin to overfit, even after 100 gradient steps.

F PROOF OF THEOREM 6.2Here we provide a proof of Theorem 6.2:Theorem 6.2 The gradient of the softmax cross entropy loss with respect to the pre-softmax logits is a linear, invertible function of y, when evaluated at 0.For the standard softmax cross-entropy loss function with discrete, one-hot labels y, the gradient is ∇ŷ (y, 0) = c − y where c is a constant vector of value c and where we are denotingŷ as the pre-softmax logits.

Since y is a one-hot representation, we can rewrite the gradient as ∇ŷ (y, 0) = (C − I)y, where C is a constant matrix with value c. Since A = C − I is invertible, the cross entropy loss also satisfies the above requirement.

Thus, we have shown that both of the standard supervised objectives of mean-squared error and cross-entropy allow for the universality of gradientbased meta-learning.

In this section, we provide two additional comparisons on an out-of-distribution task and using additional gradient steps, shown in FIG3 .

We also include additional experimental details.

For Omniglot, all meta-learning methods were trained using code provided by the authors of the respective papers, using the default model architectures and hyperparameters.

The model embedding architecture was the same across all methods, using 4 convolutional layers with 3 × 3 kernels, 64 filters, stride 2, batch normalization, and ReLU nonlinearities.

The convolutional layers were followed by a single linear layer.

All methods used the Adam optimizer with default hyperparameters.

Other hyperparameter choices were specific to the algorithm and can be found in the respective papers.

For MAML in the sinusoid domain, we used a fully-connected network with two hidden layers of size 100, ReLU nonlinearities, and a bias transformation variable of size 10 concatenated to the input.

This model was trained for 70,000 meta-iterations with 5 inner gradient steps of size α = 0.001.

For SNAIL in the sinusoid domain, the model consisted of 2 blocks of the following: 4 dilated convolutions with 2 × 1 kernels 16 channels, and dilation size of 1,2,4, and 8 respectively, then an attention block with key/value dimensionality of 8.

The final layer is a 1 × 1 convolution to the output.

Like MAML, this model was trained to convergence for 70,000 iterations using Adam with default hyperparameters.

We evaluated the MAML and SNAIL models for 1200 trials, reporting the mean and 95% confidence intervals.

For computational reasons, we evaluated the MetaNet model using 600 trials, also reporting the mean and 95% confidence intervals.

Following prior work (Santoro et al., 2016) , we downsampled the Omniglot images to be 28 × 28.

When scaling or shearing the digits to produce out-of-domain data, we transformed the original 105 × 105 Omniglot images, and then downsampled to 28 × 28.

In the depth comparison, all models were trained to convergence using 70,000 iterations.

Each model was defined to have a fixed number of hidden units based on the total number of parameters (fixed at around 40,000) and the number of hidden layers.

Thus, the models with 2, 3, 4, and 5 hidden layers had 200, 141, 115, and 100 units per layer respectively.

For the model with 1 hidden layer, we found that using more than 20, 000 hidden units, corresponding to 40, 000 parameters, resulted in poor performance.

Thus, the results reported in the paper used a model with 1 hidden layer with 250 units which performed much better.

We trained each model three times and report the mean and standard deviation of the three runs.

The performance of an individual run was computed using the average over

<|TLDR|>

@highlight

Deep representations combined with gradient descent can approximate any learning algorithm.