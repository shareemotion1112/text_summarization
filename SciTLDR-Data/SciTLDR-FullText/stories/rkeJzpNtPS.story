We propose two approaches of locally adaptive activation functions namely, layer-wise and neuron-wise locally adaptive activation functions, which improve the performance of deep and physics-informed neural networks.

The local adaptation of activation function is achieved by introducing scalable hyper-parameters in each layer (layer-wise) and for every neuron separately (neuron-wise), and then optimizing it using the stochastic gradient descent algorithm.

Introduction of neuron-wise activation function acts like a vector activation function as opposed to the traditional scalar activation function given by fixed, global and layer-wise activations.

In order to further increase the training speed, an activation slope based slope recovery term is added in the loss function, which further accelerate convergence, thereby reducing the training cost.

For numerical experiments, a nonlinear discontinuous function is approximated using a deep neural network with layer-wise and neuron-wise locally adaptive activation functions with and without the slope recovery term and compared with its global counterpart.

Moreover, solution of the nonlinear Burgers equation, which exhibits steep gradients, is also obtained using the proposed methods.

On the theoretical side, we prove that in the proposed method the gradient descent algorithms are not attracted to sub-optimal critical points or local minima under practical conditions on the initialization and learning rate.

Furthermore, the proposed adaptive activation functions with the slope recovery are shown to accelerate the training process in standard deep learning benchmarks using CIFAR-10, CIFAR-100, SVHN, MNIST, KMNIST, Fashion-MNIST, and Semeion data sets with and without data augmentation.

In recent years, research on neural networks (NNs) has intensified around the world due to their successful applications in many diverse fields such as speech recognition , computer vision (Krizhevsky et al., 2012) , natural language translation (Wu et al., 2016) , etc.

Training of NN is performed on data sets before using it in the actual applications.

Various data sets are available for applications like image classification, which is a subset of computer vision.

MNIST (LeCun et al., 1998) and their variants like, Fashion-MNIST (Xiao et al., 2017) , and KMNIST (Clanuwat et al., 2018) are the data sets for handwritten digits, images of clothing and accessories, and Japanese letters, respectively.

Apart from MNIST, Semeion (Brescia, 1994 ) is a handwritten digit data set that contains 1593 digits collected from 80 persons.

SVHN (Netzer et al., 2011) is another data set for street view house numbers obtained from house numbers in Google Street View images.

CI-FAR (Krizhevsky et al., 2009 ) is the popular data set containing color images commonly used to train machine learning algorithms.

In particular, the CIFAR-10 data set contains 50000 training and 10000 testing images in 10 classes with image resolution of 32x32.

CIFAR-100 is similar to the CIFAR-10, except it has 100 classes with 600 images in each class, which is more challenging than the CIFAR-10 data set.

problems, where the approximate solutions of governing equations are obtained, as well as inverse problems, where parameters involved in the governing equation are inferred from the training data.

Highly efficient and adaptable algorithms are important to design the most effective NN which not only increases the accuracy of the solution but also reduces the training cost.

Various architectures of NN like Dropout NN (Srivastava et al., 2014) are proposed in the literature, which can improve the efficiency of the algorithm for specific applications.

Activation function plays an important role in the training process of NN.

In this work, we are particularly focusing on adaptive activation functions, which adapt automatically such that the network can be trained faster.

Various methods are proposed in the literature for adaptive activation function, like the adaptive sigmoidal activation function proposed by (Yu et al., 2002) for multilayer feedforward NNs, while (Qian et al., 2018) focuses on learning activation functions in convolutional NNs by combining basic activation functions in a data-driven way.

Multiple activation functions per neuron are proposed (Dushkoff & Ptucha, 2016) , where individual neurons select between a multitude of activation functions. (Li et al., 2013) proposed a tunable activation function, where only a single hidden layer is used and the activation function is tuned. (Shen et al., 2004) , used a similar idea of tunable activation function but with multiple outputs.

Recently, Kunc and Kléma proposed a transformative adaptive activation functions for gene expression inference, see (Kunc & Kléma, 2019) .

One such adaptive activation function is proposed (Jagtap & Karniadakis, 2019) by introducing scalable hyper-parameter in the activation function, which can be optimized.

Mathematically, it changes the slope of activation function thereby increasing the learning process, especially during the initial training period.

Due to single scalar hyper-parameter, we call such adaptive activation functions globally adaptive activations, meaning that it gives an optimized slope for the entire network.

One can think of doing such optimization at the local level, where the scalable hyper-parameter are introduced hidden layer-wise or even for each neuron in the network.

Such local adaptation can further improve the performance of the network.

Figure 1 shows a sketch of a neuron-wise locally adaptive activation function based physics-informed neural network (LAAF-PINN), where both the NN part along with the physicsinformed part can be seen.

In this architecture, along with the output of NN and the residual term from the governing equation, the activation slopes from every neuron are also contributing to the loss function in the form of slope recovery term.

The rest of the paper is organized as follows.

Section 2 presents the methodology of the proposed layer-wise and neuron-wise locally adaptive activations in detail.

This also includes a discussion on the slope recovery term, expansion of parametric space due to layer-wise and neuron-wise introduction of hyper-parameters, its effect on the overall training cost, and a theoretical result for gradient decent algorithms.

Section 3 gives numerical experiments, where we approximate a nonlinear discontinuous function using deep NN by the proposed approaches.

We also solve the Burgers equation using the proposed algorithm and present various comparisons in appendix B. Section 4 presents numerical results with various standard deep learning benchmarks using CIFAR-10, CIFAR-100, SVHN, MNIST, KMNIST, Fashion-MNIST, and Semeion data sets.

Finally, in section 5, we summarize the conclusions of our work.

We use a NN of depth D corresponding to a network with an input layer, D − 1 hidden-layers and an output layer.

In the k th hidden-layer, N k number of neurons are present.

Each hidden-layer of the network receives an output z k−1 ∈ R N k−1 from the previous layer where an affine transformation of the form

is performed.

The network weights w k ∈ R N k ×N k−1 and bias term b k ∈ R N k associated with the k th layer are chosen from independent and identically distributed sampling.

The nonlinearactivation function σ(·) is applied to each component of the transformed vector before sending it as an input to the next layer.

The activation function is an identity function after an output layer.

Thus, the final neural network representation is given by the composition

where the operator • is the composition operator,

represents the trainable parameters in the network, u is the output and z 0 = z is the input.

In supervised learning of solution of PDEs, the training data is important to train the neural network, which can be obtained from the exact solution (if available) or from high-resolution numerical solution given by efficient numerical schemes and it can be even obtained from carefully performed experiments, which may yield both high-and low-fidelity data sets.

We aim to find the optimal weights for which the suitably defined loss function is minimized.

In PINN the loss function is defined as

where the mean squared error (MSE) is given by

Here {x

represents the residual training points in space-time domain, while {x

represents the boundary/initial training data.

The neural network solution must satisfy the governing equation at randomly chosen points in the domain, which constitutes the physicsinformed part of neural network given by first term, whereas the second term includes the known boundary/initial conditions, which must be satisfied by the neural network solution.

The resulting optimization problem leads to finding the minimum of a loss function by optimizing the parameters like, weights and biases, i.e., we seek to find w * , b * = arg min w,b∈Θ (J(w, b)).

One can approximate the solutions to this minimization problem iteratively by one of the forms of gradient descent algorithm.

The stochastic gradient descent (SGD) algorithm is widely used in machine learning community see, Ruder (2016) for a complete survey.

In SGD the weights are updated as

, where η l > 0 is the learning rate.

SGD methods can be initialized with some starting value w 0 .

In this work, the ADAM optimizer Kingma & Ba (2014) , which is a variant of the SGD method is used.

A deep network is required to solve complex problems, which on the other hand is difficult to train.

In most cases, a suitable architecture is selected based on the researcher's experience.

One can also think of tuning the network to get the best performance out of it.

In this regard, we propose the following two approaches to optimize the adaptive activation function.

Instead of globally defining the hyper-parameter a for the adaptive activation function, let us define this parameter hidden layer-wise as

This gives additional D − 1 hyper-parameters to be optimized along with weights and biases.

Here, every hidden-layer has its own slope for the activation function.

One can also define such activation function at the neuron level as

This gives additional

k=1 N k hyper-parameters to be optimized.

Neuron-wise activation function acts as a vector activation function as opposed to scalar activation function given by L-LAAF and global adaptive activation function (GAAF) approaches, where every neuron has its own slope for the activation function.

The resulting optimization problem leads to finding the minimum of a loss function by optimizing a k i along with weights and biases, i.e., we seek to find (a

The process of training NN can be further accelerated by multiplying a with scaling factor n > 1.

The final form of the activation function is given by σ(na

It is important to note that the introduction of the scalable hyper-parameter does not change the structure of the loss function defined previously.

Then, the final adaptive activation function based neural network representation of the solution is given by

In this case, the set of trainable parametersΘ consists of {w

and {a

In all the proposed methods, the initialization of scalable hyper-parameters is done such that na k i = 1, ∀n.

The main motivation of adaptive activation function is to increase the slope of activation function, resulting in non-vanishing gradients and fast training of the network.

It is clear that one should quickly increase the slope of activation in order to improve the performance of NN.

Thus, instead of only depending on the optimization methods, another way to achieve this is to include the slope recovery term based on the activation slope in the loss function as

where the slope recovery term S(a) is given by

where N is a linear/nonlinear operator.

Although, there can be several choices of this operator, including the linear identity operator, in this work we use the exponential operator.

The main reason behind this is that such term contributes to the gradient of the loss function without vanishing.

The overall effect of inclusion of this term is that it forces the network to increase the value of activation slope quickly thereby increasing the training speed.

We now provide a theoretical result regarding the proposed methods.

The following theorem states that a gradient descent algorithm minimizing our objective functionJ(Θ) in equation 3 does not converge to a sub-optimal critical point or a sub-optimal local minimum, for neither L-LAAF nor N-LAAF, given appropriate initialization and learning rates.

In the following theorem, we treatΘ as a real-valued vector.

Let Jc(0) = M SE F + M SE u with the constant network u Θ (z) = u Θ (z ) = c ∈ R N D for all z, z where c is a constant.

Theorem 2.1.

Let (Θ m ) m∈N be a sequence generated by a gradient descent algorithm asΘ m+1 = Θ m − η m ∇J(Θ).

Assume that J(Θ 0 ) < Jc(0) + S(0) for any c ∈ R N D , J is differentiable, and that for each i ∈ {1, . . .

, N f }, there exist differentiable function ϕ i and input

.

Assume that at least one of the following three conditions holds.

(i) (constant learning rate)

∇J is Lipschitz continuous with Lipschitz constant C (i.e., ∇J(Θ) − ∇J(Θ ) 2 ≤ C Θ −Θ 2 for allΘ,Θ in its domain), and ≤ η m ≤ (2 − )/C, where is a fixed positive number.

(ii) (diminishing learning rate) ∇J is Lipschitz continuous, η m → 0 and ∞ m=0 η m = ∞. (iii) (adaptive learning rate) the learning rate η m is chosen by the minimization rule, the limited minimization rule, the Armjio rule, or the Goldstein rule (Bertsekas, 1997).

Then, for both L-LAAF and N-LAAF, no limit point of (Θ m ) m∈N is a sub-optimal critical point or a sub-optimal local minimum.

The initial condition J(Θ 0 ) < Jc(0) + S(0) means that the initial value J(Θ 0 ) needs to be less than that of a constant network plus the highest value of the slope recovery term.

Here, note that S(1) < S(0).

The proof of Theorem 2.1 is included in appendix A.

In this section, we shall solve a regression problem of a nonlinear function approximation using deep neural network.

The Burgers equation using physics-informed neural network is solved in appendix B. In this test case, a standard neural network (without physics-informed part) is used to approximate a discontinuous function.

In this case, the loss function consists of the data mismatch and the slope recovery term as

The following discontinuous function with discontinuity at x = 0 location is approximated by a deep neural network.

Here, the domain is [−3, 3] and the number of training points used is 300.

The activation function is tanh, learning rate is 2.0e-4 and the number of hidden layers are four with 50 neurons in each layer.

Figure 2 shows the solution (first column), solution in frequency domain (second column) and pointwise absolute error in log scale (third column).

The solution by standard fixed activation function is given in the first row, GAAF solution is given in second row, whereas the third row shows the solution given by L-LAAF without and with (fourth row) slope recovery term.

The solution given by N-LAAF without slope recovery term is shown in the fifth row and with slope recovery term in the sixth row.

We see that the NN training speed increases for the locally adaptive activation functions compared to fixed and globally adaptive activations.

Moreover, both L-LAAF and N-LAAF with slope recovery term accelerate training and yield the least error as compared to other methods.

Figure 3 (top) shows the variation of na for GAAF, whereas the second row, left and right shows the layer-wise variation of na k for L-LAAF without and with the slope recovery term respectively.

The third row, left and right shows the variation of scaled hyper-parameters for N-LAAF without and with the slope recovery term respectively, where the mean value of na k i along with its standard deviation (Std) are plotted for each hidden-layer.

We see that the value of na is quite large with the slope recovery term which shows the rapid increase in the activation slopes.

Finally, the comparison of the loss function is shown in figure 4 for standard fixed activation, GAAF, L-LAAF and N-LAAF without the slope recovery (left) and for L-LAAF and N-LAAF with the slope recovery (right) using a scaling factor of 10.

The Loss function for both L-LAAF and N-LAAF without the slope recovery term decreases faster, especially during the initial training period compared to the fixed and global activation function based algorithms.

The previous sections demonstrated the advantages of adaptive activation functions with PINN for physics related problems.

One of the remaining questions is whether or not the advantage of adaptive activations remains with standard deep neural networks for other types of deep learning applications.

To explore the question, this section presents numerical results with various standard benchmark problems in deep learning.

Figures 5 and 6 shows the mean values and the uncertainty intervals Figure 2: Discontinuous function: Neural network solution using standard fixed activation (first row), GAAF (second row), L-LAAF without (third row) and with (fourth row) slope recovery term, and N-LAAF without (fifth row) and with (sixth row) slope recovery term using the tanh activation.

First column shows the solution which is also plotted in frequency domain (zoomed-view) as shown by the corresponding second column.

Third column gives the point-wise absolute error in the log scale for all the cases.

accelerates the minimization process of the training loss values.

Here, all of GAAF, L-LAAF and N-LAAF use the slope recovery term, which improved the methods without the recovery term.

Accordingly, the results of GAAF are also new contributions of this paper.

In general, L-LAAF improved against GAAF as expected.

The standard cross entropy loss was used for training and plots.

We used pre-activation ResNet with 18 layers (He et al., 2016) for CIFAR-10, CIFAR-100, and SVHN data sets, whereas we used a standard variant of LeNet (LeCun et al., 1998) with ReLU for other data sets; i.e., the architecture of the variant of LeNet consists of the following five layers (with the three hidden layers): (1) input layer, (2) convolutional layer with 64 5 × 5 filters, followed by max pooling of size of 2 by 2 and ReLU, (3) convolutional layer with 64 5 × 5 filters, followed by max pooling of size of 2 by 2 and ReLU, (4) fully connected layer with 1014 output units, followed by ReLU, and (5) Fully connected layer with the number of output units being equal to the number of target classes.

All hyper-parameters were fixed a priori across all different data sets and models.

We fixed the mini-batch size s to be 64, the initial learning rate to be 0.01, the momentum coefficient to be 0.9 and we use scaling factor n = 1 and 2.

The learning rate was divided by 10 at the beginning of 10th epoch for all experiments (with and without data augmentation), and of 100th epoch for those with data augmentation.

In this paper, we present two versions of locally adaptive activation functions namely, layer-wise and neuron-wise locally adaptive activation functions.

Such local activation functions further improve the training speed of the neural network compared to its global predecessor.

To further accelerate the training process, an activation slope based slope recovery term is added in the loss function for both layer-wise and neuron-wise activation functions, which is shown to enhance the performance of the neural network.

Various NN and PINN test cases like nonlinear discontinuous function approximation and Burgers equation respectively, and benchmark deep learning problems like MNIST, CIFAR, SVHN etc are solved to verify our claim.

Moreover, we theoretically prove that no sub-optimal critical point or local minimum attracts gradient descent algorithms in the proposed methods (L-LAAF and N-LAAF) with the slope recovery term under only mild assumptions.

k=1 is a limit point of (Θ m ) m∈N and a sub-optimal critical point or a sub-optimal local minimum.

and h

Following the proofs in (Bertsekas, 1997, Propositions 1.2.1-1.2.4), we have that ∇J(Θ) = 0 and J(Θ) < Jc(0) + S(0), for all three cases of the conditions corresponding the different rules of the learning rate.

Therefore, we have that for all k ∈ {1, . . .

, D − 1},

Furthermore, we have that for all k ∈ {1, . . .

, D − 1} and all j ∈ {1, . . .

, N k },

By combining equation 5-equation 7, for all k ∈ {1, . . .

, D − 1},

which implies that for all a k = 0 since (D − 1)

exp(a k ) = 0.

This implies that J(Θ) = Jc(0) + S(0), which contradicts with J(Θ) < Jc(0) + S(0).

This proves the desired statement for L-LAAF.

For N-LAAF, we prove the statement by contradiction.

Suppose that the parameter vectorΘ consisting of {w

k=1 ∀j = 1, 2, · · · , N k is a limit point of (Θ m ) m∈N and a suboptimal critical point or a sub-optimal local minimum.

Redefine

and h

for all j ∈ {1, . . .

, N k }, where w k,j ∈ R 1×N k−1 and b k,j ∈ R. Then, by the same proof steps, we have that ∇J(Θ) = 0 and J(Θ) < Jc(0) + S(0), for all three cases of the conditions corresponding the different rules of the learning rate.

Therefore, we have that for all k ∈ {1, . . .

, D − 1} and all j ∈ {1, . . .

, N k },

By combining equation 6-equation 8, for all k ∈ {1, . . .

, D − 1} and all j ∈ {1, . . .

, N k }, ,

which implies that for all a

This implies that J(Θ) = Jc(0) + S(0), which contradicts with J(Θ) < Jc(0) + S(0).

This proves the desired statement for N-LAAF.

The Burgers equation is one of the fundamental partial differential equation arising in various fields such as nonlinear acoustics, gas dynamics, fluid mechanics etc, see Whitham (2011) for more details.

The Burgers equation was first introduced by H. Bateman, Bateman (1915) and later studied by J.M. Burgers, Burgers (1948) in the context of theory of turbulence.

Here, we consider the viscous Burgers equation given by equation equation 9 along with its initial and boundary conditions.

The non-linearity in the convection term develops very steep solution due to small˜ value.

We consider the Burgers equation given by

with initial condition u(x, 0) = − sin(πx), boundary conditions u(−1, t) = u(1, t) = 0 and˜ = 0.01/π.

The analytical solution can be obtained using the Hopf-Cole transformation, see Basdevant et al. (1986) for more details.

The number of boundary and initial training points is 400, whereas the number of residual training points is 10000.

The activation function is tanh, learning rate is 0.0008 and the number of hidden layers are 6 with 20 neurons in each layer.

Figure 7 shows the evolution of frequency plots of the solution at three different times using the standard fixed activation function (first row), global adaptive activation function (second row), L-LAAF without (third row) and with (fourth row) slope recovery term, N-LAAF without (fifth row) and with (sixth row) slope recovery term using scaling factor n = 10.

Again, for both L-LAAF and N-LAAF the frequencies are converging faster towards the exact solution (shown by black line) with and without slope recovery term as compared to the fixed and global activation function based algorithms.

decreases faster for all adaptive activations, in particular GAAF.

Even though it is difficult to see from the actual solution plots given by figure 8, one can see from the Figure 10: Burgers equation: comparison of na k for L-LAAF for all six hidden-layers.

First three columns represent results for L-LAAF without slope recovery term whereas the last three columns are with slope recovery term.

In all simulations, the scaling factor n is 10.

Figure 10 shows the comparison of layer-wise variation of na k for L-LAAF with and without slope recovery term.

It can be seen that, the presence of slope recovery term further increases the slope of activation function thereby increasing the training speed.

Similarly, figure 11 shows the mean and standard deviation of na k i for N-LAAF with and without slope recovery term, which again validates that with slope recovery term network training speed increases.

for N-LAAF for all six hidden-layers.

First three columns represent resuls for N-LAAF without the slope recovery term whereas the last three columns are with slope recovery term.

In all simulations, the scaling factor n is 10.

@highlight

Proposing locally adaptive activation functions in deep and physics-informed neural networks for faster convergence