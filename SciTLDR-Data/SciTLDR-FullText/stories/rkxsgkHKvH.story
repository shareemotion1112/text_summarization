Activation is a nonlinearity function that plays a predominant role in the convergence and performance of deep neural networks.

While Rectified Linear Unit (ReLU) is the most successful activation function, its derivatives have shown superior performance on benchmark datasets.

In this work, we explore the polynomials as activation functions (order ≥ 2) that can approximate continuous real valued function within a given interval.

Leveraging this property, the main idea is to learn the nonlinearity, accepting that the ensuing function may not be monotonic.

While having the ability to learn more suitable nonlinearity, we cannot ignore the fact that it is a challenge to achieve stable performance due to exploding gradients - which is prominent with the increase in order.

To handle this issue, we introduce dynamic input scaling, output scaling, and lower learning rate for the polynomial weights.

Moreover, lower learning rate will control the abrupt fluctuations of the polynomials between weight updates.

In experiments on three public datasets, our proposed method matches the performance of prior activation functions, thus providing insight into a network’s nonlinearity preference.

Deep learning methods have achieved excellent results in visual understanding, visual recognition, speech, and natural language processing tasks (Krizhevsky et al. (2012) , Lee et al. (2014) , Goodfellow et al. (2014) , Hochreiter & Schmidhuber (1997) , Oord et al. (2016) , Vaswani et al. (2017) ).

The convolutional neural networks (CNNs) first introduced in LeCun et al. (1999) , is the foundation for numerous vision tasks.

While recurrent neural networks, wavenet and the recent transformers with attention mechanism are the core algorithms used in speech and natural language processing.

The commonality is the importance of deeper architectures that has both theoretical and empirical evidence (Serre et al. (2007) , Simonyan & Zisserman (2015) , Lee et al. (2014) ).

One essential component for deep neural networks is the activation function that enables nonlinearity.

While ReLUs are the most used nonlinearity, sigmoid and hyperbolic tangent are the traditional functions.

Several derivatives of ReLU are presented in recent years that further improve the performance and minimize vanishing gradients issue (Maas et al. (2013) , He et al. (2015a) , Clevert et al. (2015) , Ramachandran et al. (2019) ).

While most are fixed functions, the negative slope for Leaky ReLUs can be adjusted during the network design, and remains constant while training.

Parametric ReLU adaptively changes the negative slope during training using a trainable parameter and demonstrate a significant boost in performance (He et al. (2015a) ).

A relatively new activation function, Swish, is derived by an automated search techniques (Ramachandran et al. (2019) ).

While the parameter β enables learning, the performance difference reported in the study between parametric and non-parametric versions is minimal.

To this end, rather than using a fixed or heavily constrained nonlinearity, we believe that the nonlinearity learned by the deep networks can provide more insight on how they can be designed.

In this work, we focus on the use of polynomials as nonlinearity functions.

We demonstrate the stability of polynomial of orders 2 to 9 by introducing scaling functions and initialization scheme that approximates well known activation functions.

Experiments on three public datasets show that our method competes with state-of-the-art activation functions on a variety of deep architectures.

Despite their imperfections, our method allows each layer to find their preferred nonlinearity during training.

Finally, we show the learned nonlinearities that are both monotonic and non-monotonic.

In this section we review activation functions and their relative formulation.

Sigmoid (σ(x) = (1 + exp(−z)) −1 ), and the hyperbolic tangent (tanh(x) = (exp(x) − exp(−x))/(exp(x) + exp(−x))), are the usual nonlinearity functions for neural networks.

However, the deeper networks suffer vanishing gradient issue (Bengio et al. (1994) ) that have near zero gradients at the initial layers.

Softplus (f (x) = log(1 + exp(x))) initially proposed in Dugas et al. (2000) , is a smoother verion of ReLU whose derivative is a sigmoid function.

Unlike sigmoid or tanh, ReLU (ReLU (x) = max(x, 0)) can have gradient flow as long as the inputs are positive (Hahnioser et al. (2000) , Nair & Hinton (2010) , Glorot et al. (2011) ).

An extension of ReLU, called Leaky ReLUs (LReLU), allows a fraction of negative part to speed-up the learning process by avoiding the constant zero gradients when x < 0 (Maas et al. (2013) ).

LReLU are relatively popular for generative adversarial networks (Radford et al. (2015) ).

While the slope (α) is constant for LReLU, it is a learnable parameter for Parametric ReLU (PReLU), which has achieved better performance on image benchmark datasets (He et al. (2015a) ).

However, Exponential Linear Unit (ELU), another derivative of ReLU, has improved learning by shifting the mean towards zero and ensuring a noise-robust deactivation state (Clevert et al. (2015) ).

Gaussian Error Linear Unit, defined by GeLU (x) = xΦ(x), is a non-monotonic nonlinearity function (Hendrycks & Gimpel (2016) ), where Φ(x) is the cumulative distribution function.

Scaled Exponential Linear Unit (SELU) with self-normalizing property delivers robust training of deeper networks Klambauer et al. (2017) .

where λ = 1.0507 and α = 1.6733

Unlike any of the above, Swish is a result of automated search technique Ramachandran et al. (2019) .

Using a combination of exhaustive and reinforcement learning based search techniques (Bello et al. (2017) , Zoph & Le (2016) ), the authors reported eight novel activation functions (x * σ(βx), max(x, σ(x)), cos(x) − x, min(x, sin(x)), (tan −1 (x)) 2 − x, max(x, tanh(x)), sinc(x) + x, and x * (sinh −1 (x)) 2 ).

Of which, x * σ(βx) (Swish) and max(x, σ(x)) matched or outperformed ReLU in CIFAR experiments, with the former showing better performance on ImageNet and machine translation tasks.

The authors claim that the non-monotonic bump, controlled by β, is an important aspect of Swish.

β can either be a constant or trainable parameter.

To our knowledge this is the first work to search activation functions.

In contrast, we use polynomials with trainable coefficients to learn activation functions that have the ability to approximate a well-known or novel continuous nonlinearity.

Furthermore, we demonstrate its stability and performance on three public benchmark datasets.

We emphasize that our goal is to understand a network's nonlinearity preference given the ability to learn.

An n th order polynomial with trainable weights w j , j ∈ 0, 1, ...n and its derivatives are defined as follows:

Given an input with zero mean and unit variance, a deeper network with multiple f n (x) suffer exploding gradients as f n (x) scales exponentially with increase in order (n) for x > 1 (limx→∞ fn(x)/x=∞).

While this can be avoided by using sigmoid(x), the approximations of f n (sigmoid(x)) are limited for lower weights and saturates for higher weights.

Moreover, the lower weights can suffer from vanishing gradients and the higher weights beacause of exploding gradients.

To circumvent the exploding weights and gradients, we introduce dynamic input scaling, g(.).

Consider x i , i ∈ 1, 2, ...K is the ouput of a dense layer with K neurons, we define the dynamic input scaling g(.) as follows:

2, regardless of the input dimensions.

The choice of √ 2 allowed us to contain f n (g(x)) for higher order's with larger w n , especially in our mixed precision training.

The primary advantage of the g(.) is that for any given order of the polynomial, we have max(f n (g(x i ))) ≤ w j 2 * √ 2 n ; this constraint allows us to normalize the output, resulting in f n (g(

Observing the experiments in Section 4, we limit w j 2 ≤ 3 which allows us to explore prior activations as initilizations.

Essentially, when w j 2 > 3, we renormalize the weights using w j * 3/ w j 2 .

Therefore, we define n th order polynomial activations as:

Unlike the usual, polynomial activation is not for a scalar value and of course, not as simple as ReLU.

The purpose of our proposal is to explore newer nonlinearities by allowing the network to choose an appropriate nonlinearity for each layer and eventually, design better activation functions on the gathered intuitions.

Initialization of polynomial weights play a significant role in the stability and convergence of networks.

We start by examining this issue with random initializations.

In this scenario, we start with Network-1 defined in Section 5.1 and polynomial activation with n = 2.

We train all the networks using stochastic gradient descent (SGD) with a learning rate of 0.1 and a batch size of 256.

To speedup this experiment, we use mixed precision training.

By varying w i ∈ {−1, −0.95, 0.90, ..., 1}, we train each possible combinations (41 3 = 68921) of initialization for an epoch.

We use same initialization for all the three activations.

The weights of both convolution and linear layers are initialized using He et al. (2015a) for each run.

Given the simplicity of MNIST hand written digits dataset, we consider any network with test error of ≤ 10% as fast converging initialization.

In comparison, all the networks reported in Table2 achieved a test error of ≤ 5% after the first epoch.

One interesting observation with this experiment is that the networks initialized with f n (x) ≈ x and f n (x) ≈ −x have always achieved a test error of ≤ 5%.

We extended our experiment to n = 3 with w i ∈ {−1, −0.8, −0.6, ..., 1} and observed the same.

Our second observation is that the weights closer to 0 never converged.

To this end, rather than initializing the polynomial weights with a normal or uniform distribution, we instead used weights that can approximate an activation function, F (x).

The advantage is that we can start with some of the most successful activation functions, and gradually allow the weights to choose an appropriate nonlinearity.

It is important to understand that the local minima for f n (x) ≈ F (x) is not gaurenteed since x does not satisfy boundary conditions, regardless of it being continuous (Weierstrass theorem).

While lower order polynomial suffer more, failure often leads to f n (x) ≈ x, which is stable.

Along with f n (x) = x (w 1 = 1), we investigate ReLU, Swish (β = 1) and TanH approximations as initializations that are derived by minimizing the sum of squared residuals of f n (x) − ReLU (x), f n (x) − Swish(x), and f n (x) − tanh(x) using the least squares within the bounds −5 ≤ x ≤ 5.

Essentially, for an n th order polynomial, we initialize the w j such that the f n (x) ≈ ReLU (x), f n (x) ≈ Swish(x), or f n (x) ≈ tanh(x).

Figure 1 shows the approximations of ReLU, Swish and TanH using polynomials of orders 2 to 9, and their respective initializations used in this experiment are in Table1.

On observation, Swish approximations are relatively more accurate when compared to ReLU and TanH.

Using Network-1 and Network-2 defined in Section 5.1, we evaluate the stability and performance of each of the four initializations for order's 2 to 9 under the following optimization setting: 1) SGD with a learning rate of 0.1 2) SGD with a learning rate of 0.05 3) SGD with a learning rate of 0.01 4) Adam (Kingma & Ba (2014)) with a learning rate of 0.001.

By varying the batch size from 2 3 to 2 10 , we trained Network-1 and Network-2 in each setting for 10 epochs to minimize cross entropy loss on MNIST data.

In total, we train 512 configurations (2 different networks * 8 different orders * 4 different optimization settings * 8 different batch sizes) per initialization, and compute the test error at the end of 10 th epoch.

We observed that ∼ 22% of our networks with order ≥ 6 failed to converge.

On monitoring the gradients, we reduced the learning rate (lr) for the polynomial weights to lr * 2 −0.5 * order 1 , resulting in the convergence of all the 512 configurations.

The issue is that the polynomials fluctuate heavily between the updates due to high backpropogated gradients.

The networks whose activations are initialized with f n (x) ≈ Swish(x) outperformed in 167 experiments, followed by f n (x) ≈ ReLU (x) in 155, f n (x) = x in 148 and f n (x) ≈ tanh(x) in 42.

The results suggest that the Swish approximations as initializations are marginally better when compared to the rest.

It is important to note that the activation after training does not resemble Swish or any other approximations that are used during the initialization.

Instead, allows the network to converge faster.

We evaluate the polynomial activations on three public datasets and compare our work with seven state-of-art activation functions reported in the literature.

On CIFAR, we present the accuracies of the networks with polynomial activations presented in Ramachandran et al. (2019) .

Here on, we initialize polynomial activations with f n (x) ≈ Swish(x) weights.

We set α = 0.01 for LReLUs, α = 1.0 for ELUs, λ = 1.0507 and α = 1.6733 for SELU, and a non-trainable β = 1 for Swish (Swish-1).

We adopt the initilization proposed in He et al. (2015a) for both convolution and linear layers, and implement all the models in PyTorch.

Swish f 9 (x) = 6.5e

+1.8e −9 * x 5 + 4.7e −4 * x 6 − 9.5e −11 * x 7 − 7.0e

The MNIST dataset consists greysale handwritten digits (0-9) of size 28 x 28 pixel, with 60,000 training and 10,000 test samples, no augmentation was used for this experiment.

We consider four different networks to understand the behaviour of the proposed activation for orders 2 to 9 on network depth.

In each network, there are k convolutional layers (ends with an activation function), followed by a dropout (p = 0.2), a linear with 64 output neurons, an activation, and a softmax layer.

The number of convolutional layers, k, in Network-1, Network-2, Network-3 and Network-4 is two, four, six and eight, respectively.

The choice of networks is driven by our curiosity to investigate the stability of the proposed activation function.

The filter size in the first convolutional layer is 7x7, and then the rest are 3x3 filters.

The number of output channels in the first k/2 convolutional layers is 64, and then the rest are 128.

The subsampling is performed by convolutions with a stride of 2 at k/2 and k layer.

We train each configuration for 60 epochs using stochastic gradient descent with a momentum of 0.9.

The initial learning rate is set to 0.01 and reduced to 0.001 at epoch 40 2 .

Realizing the effect of learning rate on polynomial activations from the weight initialization experiments, we use a lower learning rate only for the polynomial weights, lr * 2 −0.5 * order .

Table2 shows the test error (median of 5 different runs) measured at the end of the final epoch.

Polynomial activations with order 2 and 3 either match or outperform other activation functions.

While our method performs well overall, we observed that the order greater than three is unnecessary.

It does not improve accuracy, and increases both complexity and computation cost.

Figure 2 shows the learned polynomial activation functions for network's 1, 2 and 3 across two different runs.

While lower order nonlinearities are similar to parabola, higher orders tend to avoid most information.

We believe the reason to avoid information by the higher orders comes from the simplicity of the data.

Overall, polynomial activations and PReLUs have performed better.

The CIFAR-10 and CIFAR-100 datasets consists of color images from 10 and 100 labels, respectively (Krizhevsky (2009) ).

There are 50,000 training and 10,000 test samples with a resolution of 32 x 32 pixels.

The training data is augmented with random crop and random horizontal flip (Lee et al. (2014) ).

We extend the experiments reported on CIFAR in Ramachandran et al. (2019) with polynomial activations using the ResNet-164 (R164) (He et al. (2015b) ), Wide ResNet 28-10 (WRN) (Zagoruyko & Komodakis (2016) ), and DenseNet 100-12 (Dense) (Huang et al. (2017) ) models.

We replicate the architecture (with two changes) and training setting from the original papers, and switch the ReLU with polynomial activations.

The two changes are 1.

lower the learning rate only for the polynomial weights, 2.

disable the polynomial activation right before the average Ramachandran et al. (2019) pool as the gradients are usually high during backpropogation.

We train using SGD with an initial learning rate of 0.1, a momentum of 0.9 and a weight decay of 0.0005.

Table3 shows the test accuracy (median of 5 different runs) measured at the end of final epoch for both CIFAR-10 and CIFAR-100.

The results are comparable and the best performance occurs with Wide ResNet which is shallow when compared to DenseNet and ResNet with relatively fewer parameters.

In this case, most activations are non-monotonic signifying its importance stated in Ramachandran et al. (2019) .

We also observe that the information allowed through initial layers is lower when compared to the deeper layers.

One reason is that the residual connections is common in all the networks.

Our numbers on CIFAR100 for DenseNet is lower when compared to Ramachandran et al. (2019) , however, we are comparable to the original implementation that reported an accuracy of 79.80 using ReLUs (Huang et al. (2017) ).

Dense (n=3) WRN (n=2) WRN (n=3) R164 (

We proposed a polynomial activation function that learns the nonlinearity using trainable coefficients.

Our contribution is stabilizing the networks with polynomial activation as a nonlinearity by introducing scaling, initilization technique and applying a lower learning rate for the polynomial weights, which provides more insight about the nonlinearity prefered by networks.

The resulting nonlinearities are both monotonic and non-monotonic in nature.

In our MNIST experiments, we showed the stability of our method with orders 2 to 9 and achieved superior perfromance when compared to ReLUs, LReLUs, PReLUs, ELUs, GELUs, SELUs and Swish.

In our CIFAR experiments, the performance by replacing ReLUs with polynomial activations using DenseNet, Residual Networks and Wide Residual Networks is on par with eight state-of-the-art activation functions.

While the increase of parameters is negligible, our method is computationally expensive.

We believe that by designing networks with simpler activations like ReLU for the initial layers, followed by layers with polynomial activations can further improve accuracies.

@highlight

We propose polynomial as activation functions.

@highlight

The authors introduce learnable activation functions that are parameterized by polynomial functions and show results slightly better than ReLU.