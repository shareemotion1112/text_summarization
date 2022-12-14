The backpropagation algorithm is the de-facto standard for credit assignment in artificial neural networks due to its empirical results.

Since its conception, variants of the backpropagation algorithm have emerged.

More specifically, variants that leverage function changes in the backpropagation equations to satisfy their specific requirements.

Feedback Alignment is one such example, which replaces the weight transpose matrix in the backpropagation equations with a random matrix in search of a more biologically plausible credit assignment algorithm.

In this work, we show that function changes in the  backpropagation procedure is equivalent to adding an implicit learning rate to an artificial neural network.

Furthermore, we learn activation function derivatives in the backpropagation equations to demonstrate early convergence in these artificial neural networks.

Our work reports competitive performances with early convergence on MNIST and CIFAR10 on sufficiently large deep neural network architectures.

Credit assignment BID10 is the task of identifying neurons and weights that are responsible for a desired prediction.

Currently, the backpropagation (BP) algorithm BID9 ) is the de-facto standard for credit assignment in artificial neural networks.

The backpropagation algorithm assigns credit by computing partial derivatives for weights and neurons with respect to the networks cost function.

Variants of the backpropagation procedure have emerged since its conception.

More specifically variants that exploit function changes in the backpropagation procedure.

Feedback Alignment BID5 is considered a biologically plausible alternative to vanilla backpropagation.

Feedback alignment is a variant of the backpropagation algorithm that uses a random weight matrix instead of the weight transpose matrix in the backpropagation equation.

Despite not scaling to the ImageNet dataset BID1 , the algorithm relaxes BP weight symmetry requirements and demonstrate comparable learning capabilities to that of BP on small datasets Similarly, BID0 produced unique backpropagation equations to train an artificial neural network by learning parts of the backpropagation equations.

BID0 report early convergence on unique backpropagation equations for CIFAR10.

In this work, we demonstrate that function changes in the backpropagation equations particularly activation function derivatives is equivalent to adding an implicit learning rate in stochastic gradient descent.

The backpropagation algorithm iteratively computes gradients for each layer, from output to input, in a neural network using the derivative chain rule.

Furthermore, we can interpret the backpropagation algorithm as simply a product of functions.

For convenience sake, we call the partial derivative functions in the backpropagation equation, b-functions.

These b-functions are the partial derivative terms [??? a C, ?? (z L ), w l+1 , ?? l+1 , ?? (z l )] in the backpropagation equation as illustrated in equations 1 and 2.

DISPLAYFORM0 Lillicrap et al. FORMULA0 probe the backpropagation equations to produce a more biologically plausible credit assignment algorithm for artificial neural networks by replacing the b-function (w l ) T with a random matrix B. Similarly, we probe the backpropagation equation by replacing the activation function derivative ?? (z l ) with a b-function g(z l ), as was also explored by BID0 .

Mathematically, this is equivalent to multiplying equation 1 with DISPLAYFORM1 when ?? (z l ) = 0, this is illustrated in equations 3.

DISPLAYFORM2 Within the context of stochastic gradient optimization, replacing the activation function derivative ?? (z l ) with the b-function g(z l ) could be interpreted as having a learning rate DISPLAYFORM3 when ?? (z l ) = 0 as described in equation 4.

DISPLAYFORM4 ??(z l ) is parameterized by z l which infers that the learning rate ??(z l ) is an adaptive learning rate with respect to the pre-activations z l .

Similarly, as ?? l propagates backwards into ?? (l???1) there will be noise = DISPLAYFORM5 propagated backwards into earlier layers.

However, in many cases gradient noise can be beneficial for gradient optimization as described by BID7 .When ?? (z l ) = 0, this replaces the zero gradient with a custom credit value.

It's ideal to have g(z l ) = 0 when ?? (z l ) = 0 else this can be harmful to the gradient optimization step.

In conclusion, the same argument applies to feedback alignment as demonstrated in equation 5, where B is the random matrix.

DISPLAYFORM6

In this experiment, we compare various b-functions changes in a neural network on a simple binary classification task to demonstrate evidence of gradient information despite the bfunction changes.

The data-set for this task is generated from two Gaussian distributions, with mean 1 and -1 respectively and standard deviation of 1.

Each Gaussian represents a classification class.

We replace the ReLU partial derivative in the backpropagation pass with derivatives from other existing activation functions.

The neural network architecture employed for this experiment : Input ->FC1 ->ReLU ->FC2 ->ReLU ->FC3 ->Softmax.

We replaced the ReLU derivative in FC2 and froze all but FC2 weights to visualize the loss landscape against the rate of change of FC2.

Our fully connected components have no bias.

The network was trained on batch gradient descent on Adam (Kingma and Ba, 2014) with a learning rate of 0.1 for 100 epochs.

The loss function is the cross entropy function.

FIG0 shows evidence of the true gradient being propagated through the network despite changes to the ReLU derivative.

Despite the noise that accompanies b-function changes as described in Section 2, the b-function changes still manage to navigate to local minima as shown in FIG0

In this experiment, we find an optimal b-function in place of the ReLU derivative in the backpropagation equation and compare it to vanilla backpropagation with a standard learning rate on an arbitrary gradient optimizer.

The aim is demonstrate that learning an optimal b-function is analogous to learning an optimal learning rate.

Learning b-functions in the backpropagation procedure using gradient optimization techniques is a complex problem.

Hence, to learn these b-functions, we reduce the problem to a black-box optimization problem.

We use Bayesian optimization for black-box optimization.

Following convention, we use Gaussian process (GP) BID8 priors over functions as our probabilistic prior model and Expected Improvement (EI) BID6 as our acquisition function for exploration.

Our learning framework consists of three components.

A target network f (??) which is the subject of the experiment, this could be a convolution neural network.

A function approximator which we call the meta-network.

The meta-network g(??) is a neural network with weights ?? and its role is to approximate a b-function.

An optimization component to learn b-functions for the target network f (??).

Hence, learning b-functions in the backpropagation pass consists of learning weights ?? such that J(??), the cost function for f (??), is minimized in training.

We formulate the black-box optimization problem as follows: the weights ?? of the metanetwork g(??) serves as the input to the black-box.

The area under the loss curve (AULC) when training the target network f (??) serves as the evaluation of the black-box function.

This framework will allow us to learn an optimal b-function in the backpropagation pass to compare its effects to learning an optimal learning rate.

We evaluated our method on CIFAR-10 for a sufficiently large neural architecture, SimpleNet BID2 .

We ran SimpleNet for 5 epochs on CIFAR 10 on the Adadelta optimizer with a learning rate of 0.01.

We replaced the ReLU activation function derivative for one layer in the convolution component.

The learned b-functions are illustrated in FIG1 and their performances in FIG1 .An implicit learning rate could relax the requirement for an explicit learning rate.

A learning rate can be learned as a weight in a network via BP instead of as a hyperparameter.

FIG1 , B-functions discovered for CIFAR10 on SimpleNet using our proposed method.

Corresponding performances of the b-functions described in FIG1 .

Function A converges earlier than backpropagation.

The meta-network architecture consisted of one input, hidden and output neuron.

<|TLDR|>

@highlight

We demonstrate that function changes in the backpropagation is equivalent to an implicit learning rate