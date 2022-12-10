Deep neural networks (DNNs) are widely adopted in real-world cognitive applications because of their high accuracy.

The robustness of DNN models, however, has been recently challenged by adversarial attacks where small disturbance on input samples may result in misclassification.

State-of-the-art defending algorithms, such as adversarial training or robust optimization, improve DNNs' resilience to adversarial attacks by paying high computational costs.

Moreover, these approaches are usually designed to defend one or a few known attacking techniques only.

The effectiveness to defend other types of attacking methods, especially those that have not yet been discovered or explored, cannot be guaranteed.

This work aims for a general approach of enhancing the robustness of DNN models under adversarial attacks.

In particular, we propose Bamboo -- the first data augmentation method designed for improving the general robustness of DNN without any hypothesis on the attacking algorithms.

Bamboo augments the training data set with a small amount of data uniformly sampled on a fixed radius ball around each training data and hence, effectively increase the distance between natural data points and decision boundary.

Our experiments show that Bamboo substantially improve the general robustness against arbitrary types of attacks and noises, achieving better results comparing to previous adversarial training methods, robust optimization methods and other data augmentation methods with the same amount of data points.

In recent years, thanks to the availability of large amounts of training data, deep neural network (DNN) models (e.g., convolutional neural networks (CNNs)) have been widely used in many realworld applications such as handwritten digit recognition ), large-scale object classification BID18 ), human face identification BID2 ) and complex control problems BID12 ).

Although DNN models have achieved close to or even beyond human performance in many applications, they exposed a high sensitivity to input data samples and therefore are vulnerable to the relevant attacks.

For example, adversarial attacks apply a "small" perturbation on input samples, which is visually indistinguishable by humans but can result in the misclassification of DNN models.

Several attacking algorithms have been also proposed, including FGSM BID20 ), DeepFool ), CW BID0 ) and PGD BID11 ) etc., indicating a serious threat against the systems using DNN models.

Many approaches have also been proposed to defend against adversarial attacks.

adversarial training, for example, adds the classification loss of certain known adversarial examples into the total training loss function: BID5 use the FGSM noise for adversarial training and BID11 use the PGD noise as the adversaries.

These approaches can effectively improve the model's robustness against a particular attacking algorithm, but won't guarantee the performance against other kinds of attacks BID0 ).

Optimization based methods take the training process as a min-max problem and minimize the loss of the worst possible adversarial examples, such as what were done by BID19 and BID21 .

The approach can increase the margin between training data points and the decision boundary along some directions.

However, solving the min-max problem on-the-fly generates a high demand for the computational load.

For large models like VGG BID18 ) and ResNet BID6 ), optimizing the min-max problem could be extremely difficult.

A large gap exists between previously proposed algorithms aiming for defending against adversarial attacks and the goal of efficiently improving the overall robustness of DNN models without any hypothesis on the attacking algorithms.

Generally speaking, defending against adversarial attacks can be considered as a special case of increasing the generalizability of machine learning models to unseen data points.

Data augmentation method, which is originally proposed for improving the model generalizability, may also be effective to improve the DNN robustness against adversarial attacks.

Previous studies show that augmenting the original training set with shifted or rotated version of the original data can make the trained classifier robust to shift and rotate transformations of the input BID17 ).

Training with additional data sampled from a Gaussian distribution centered at the original training data can also effectively enahnce the model robustness against natural noise BID1 ).

The recently proposed Mixup method BID23 ) augmented the training set with linear combinations of the original training data and surprisingly improved the DNN robustness against adversarial attacks.

Although these data augmentation methods inspired our work, they may not offer the most efficient way to enhance the adversarial robustness of DNN as they are not designated to defend adversarial attacks.

In this work, we propose Bamboo-a ball shape data augmentation technique aiming for improving the general robustness of DNN against adversarial attacks from all directions.

Bamboo augments the training set with data uniformly sampled on a fixed radius ball around each training data point.

Our theoretical analysis shows that without requiring any prior knowledge of the attacking algorithm, training the DNN classifier with our augmented data set can effectively enhance the general robustness of the DNN models against the adversarial noise.

Our experiments show that Bamboo offers a significantly enhanced model robustness comparing to previous robust optimization methods, without suffering from the high computational complexity of these prior works.

Comparing to other data augmentation method, Bamboo can also achieve further improvement of the model robustness using the same amount of augmented data.

Most importantly, as our method makes no prior assumption on the distribution of adversarial examples, it is able to work against all kinds of adversarial and natural noise.

To authors' best knowledge, Bamboo is the first data augmentation method specially designed for improving the general robustness of DNN against all directions of adversarial attacks and noise.

The remaining of the paper is organized as follows.

Section 2 explains how to measure model robustness and summaries previous research on DNN robustness improvement; In Section 3, we elaborate Bamboo's design principle and the corresponding theoretical analysis.

Section 4 empirically discusses the parameter selection and performance of our method and compares it with some related works; At the end, we conclude the paper and discuss the future work in Section 5.

2.1 MEASUREMENT OF DNN ROBUSTNESS 2.1.1 ROBUSTNESS UNDER GRADIENT BASED ATTACK A metric for measuring the robustness of the DNN is necessary.

BID20 propose the fast gradient sign method (FGSM) noise, which is one of the most efficient and most commonly applied attacking method.

FGSM generates an adversarial example x using the sign of the local gradient of the loss function J at a data point x with label y as shown in Equation (1): DISPLAYFORM0 where controls the strength of FGSM attack.

For its high efficiency in noise generation, the classification accuracy under the FGSM attack with certain has been taken as a metric of the model robustness.

As FGSM attack leverages only the local gradient for perturbing the input, gradient masking BID15 ) that messes up the local gradient can effectively improve the accuracy under FGSM attack.

However, gradient masking has little effect on the decision boundary, so it may not increase the actual robustness of the DNN.

In other words, even a DNN model achieves high accuracy under FGSM attack, it may still be vulnerable to other attacking methods.

BID11 propose projected gradient descent (PGD), which attacks the input with multi-step variant FGSM that is projected into certain space x + S at the vicinity of data point x for each step.

Equation FORMULA1 demonstrates a single step of the PGD noise generation process.

DISPLAYFORM1 Madry et al. FORMULA0 's work shows that comparing to FGSM, adversarial training using PGD adversarial is more likely to lead to a universally robust model.

Therefore the classification accuracy under the PGD attack would also be an effective metric of the model robustness.

Besides these gradient based methods, the generation of adversarial examples can also be viewed as an optimization process.

In this work, we mainly focus on untargeted optimization-based attacks.

BID20 describe the general objective of such attacks as Equation FORMULA2 : DISPLAYFORM0 Where D is the distance measurement, most commonly the Euclidean distance; and C is the classification result of the DNN.

The optimization objective is to find an adversarial example x = x + δ that results in misclassification by paying the minimum distance to the original data point x.

Note that the objective in Equation FORMULA2 includes the classification function of DNN as a constraint.

Due to the nonlinearity and the nonconvexity of the DNN classifier, the objective in Equation FORMULA2 can not be easily optimized.

In order to generate strong optimization-based attacks more efficiently, CW attack BID0 ) was proposed which defines a objective function f such that With the use of f , the optimization problem in Equation (3) can be modified to: DISPLAYFORM1 DISPLAYFORM2 It can be equivalently formulated as: DISPLAYFORM3 where c is a positive constant.

The objective in Equation FORMULA5 can be optimized more easily than that in Equation (3), leading to a higher chance of finding the optimal δ efficiently BID0 ).

BID0 successfully demonstrate that most of the previous works with high performance under FGSM attack would not be robust under their CW attack.

Since the objective of CW attack is to find the minimal possible perturbation strength of a successful attack, the resulted δ will point to the direction of the nearest decision boundary around x, and its strength can be considered as an estimation of the distance between the testing data point and the decision boundary.

Therefore the average strength required for a successful CW attack can be considered as a reasonable measurement of the model robustness.

In this work, we will use the average strength of untargeted CW noise across all the data in testing set as the metric of robustness when demonstrating the effect of parameter tuning on our proposed method.

Both the average CW strength and the testing accuracy under different strengths of FGSM and PGD attacks are taken as the metrics when comparing our method to previous works.

One of the most straightforward ways of analyzing and improving the robustness of a DNN is to formulate the robustness with key factors, such as the shape of the decision boundary or parameters and weights of the DNN model.

's work empirically visualizes the shape of the decision boundary, the observation of which shows that the curvature of the boundary tends to be lower when close to the training data points.

This technique isn't very helpful in practice, mainly due to the difficulty in drawing a theoretical relationship between the decision boundary curvature and the DNN robustness that can effectively guide the DNN training.

Some other works try to derive a bound of the DNN robustness from the network weights BID16 , BID8 ).

These obtained bounds are often too loose to be used as a guideline for robust training, or too complicated to be considered as a factor in the training objective.

A more practical approach is adversarial training.

For example, we can generate adversarial examples from the training data and then include their classification loss to the total loss function of the training process.

As the generation of the adversarial examples usually relies on existing attacking techniques like FGSM (Goodfellow et al. FORMULA0 ), DeepFool BID21 ) or PGD BID11 ), the method can be efficiently optimized for the limited types of known adversarial attacks.

However, it may not promise the robustness against other attacking methods, especially those newly proposed ones.

Alternatively, the defender may online generate the worst-case adversarial examples of the training data and minimize the loss of such adversarial examples by solving a minmax optimization problem during the training process.

For instance, the distributional robustness method BID19 ) trains the weight θ of a DNN model so as to minimize the loss L of adversarial example x which is near to original data point x but has supremum loss, such as DISPLAYFORM0 where γ is a positive constant that tradeoffs between the strength and effectiveness of the generated x .

This method can achieve some robustness improvement, but suffers from high computational cost for optimizing both the network weight and the potential adversarial example.

Also, this work only focuses on small perturbation attacks, so the robustness guarantee may not hold on the improvement of robustness under large attacking strength BID19 ).3 PROPOSED APPROACH

Most of the supervised machine learning algorithms, including the ordinary training process of DNNs, follow the principle of empirical risk minimization (ERM).

ERM tends to minimize the total risk R on the training set data, as stated in Equation FORMULA7 : DISPLAYFORM0 where f (·, θ) is the machine learning model with parameter θ, L is the loss function and P (x, y) is the joint distribution of data and label in the training set.

Such an objective is based on the hypothesis that the testing data has a similar distribution as the training data, so minimizing the loss on the training data would naturally lead to the minimum testing loss.

DNN, as a sufficiently flexible machine learning model, can be well optimized towards this objective and memorize the training set distribution ).

However, the distribution of adversarial examples generated by attacking algorithms may be different from the original training data.

Thus the memorization of DNN models would lead to unsatisfactory performance on adversarial examples BID5 ).As our work aims to improve the model robustness against adversarial attacks, we propose to follow the principle of vicinity risk minimization (VRM) instead of ERM during the training process.

Firstly proposed by BID1 , the VRM principle targets to minimize the vicinity riskR on the virtual data pair (x,ŷ) sampled from a vicinity distributionP (x,ŷ|x, y) generated from the original training set distribution P (x, y).

Consequently, the optimization objective of the VRMbased training can be described as: DISPLAYFORM1 For the choice of vicinity distribution, they use Gaussian distribution centered at original training data, which makes the model more robust to natural noise BID1 ).Now we consider improving the robustness against adversarial attacks.

It would be easier to detect and defense the adversarial attacks if the strength of the perturbation is large, therefore most of the attacking algorithms will apply a constraint on the strength of the perturbation.

So the adversarial examplex can be considered as a point within a r-radius ball around the original data x. Without any prior knowledge of the attacking algorithm, we can consider the adversarial examples as uniformly distributed within the r-radius ball:x ∼ U nif orm(||x − x|| 2 ≤ r).

Following the VRM principle, we can improve the robustness against adversarial attacks by optimizing the objective in Equation FORMULA8 with vicinity distribution: DISPLAYFORM2 However, the input space of DNN model is usually high dimensional.

Directly sampling the virtual data pointx within the r-radius ball may be data inefficient.

Here we propose to further improve the data efficiency by utilizing the geometry analysis of DNN model.

Previous research shows that the curvature of DNN's decision boundary near a training data point would most likely be very small , and the DNN model tends to behave linearly, especially at the vicinity of training data points BID5 , ).

These observations indicate that the objective of minimizing the loss of data points sampled within the ball can be approximated by minimizing the loss of data points sampled on the edge of the ball.

Formally, the vicinity distribution in Equation (9) can be modified to:P (x,ŷ|x, y) = U nif orm(||x − x|| 2 = r) · δ(ŷ, y).(10) By optimizing the VRM objective in Equation FORMULA8 with this vicinity distribution, we can improve the robustness of DNN against adversarial attacks with higher data efficiency in sampling the virtual data points for augmentation.

As explained in the previous section, minimizing the loss of data points uniformly sampled on the edge of a r-radius ball around each point in the training set likely leads to a more robust DNN model against adversarial attacks.

So we propose Bamboo, a ball-shape data augmentation scheme that augments the training set with N virtual data points uniformly sampled from a r-radius ball centered at each original training data point.

In practice, for each data point in the training data, we first sample N perturbations from a Gaussian distribution with zero mean and identity matrix as variance matrix.

Then we normalize the l 2 norm of each perturbation to r. Following the symmetric property of Gaussian distributions, the normalized perturbations should be uniformly distributed on a r-radius ball.

Finally we augment the resulted data points into the training set by adding these normalized perturbations to the original training data.

Algorithm 1 provides a formal description of the process of the proposed data augmentation method.

FIG1 intuitively demonstrates the effect of Bamboo data augmentation.

During the training process, the decision boundary will be formed to surround all the training data points of a certain class.

Since the decision boundary of the DNN model tends to have small curvature around training data points ), including the augmented data on the ball naturally pushes the decision boundary further away from the original training data points, therefore increases the robustness of the learned model.

In such sense, if the DNN model can perfectly fit to the augmented training set, increasing the ball radius will increase the margin between the decision boundary and the original training data, and more points sampled on each ball will make it less likely for the margin to get smaller than r. Figure 2 shows the effect of Bamboo with a simple classification problem.

Here we classify 100 data points sampled from the MNIST class of the digit "3" from another 100 data points in the class of the digit "7" using a multi-layer perceptron with one hidden layer.

The dimension of all data points are reduced to 2-D using PCA for visualization.

Figure 2a shows the decision boundary without data augmentation, where the decision boundary is more curvy and is overfitting to the training data.

In Figure 2b , the decision boundary after applying our data augmentation becomes smoother and is further away from original training points, implying a more robust model with the training set augmented with our proposed Bamboo method.

To analyze the performance of our proposed method, we test it with Cleverhans (Nicolas Papernot FORMULA0 ), a python library based on Tensorflow that provides reliable implementation for most of the previously proposed adversarial attack algorithms.

As mentioned in Section 2.1, for evaluating the effect of parameter r and N on the performance of our model, we use the average strength of successful CW attack BID0 ) as the metric of robustness.

When comparing with previous work, we use both CW attack strength (marked as CW rob in TAB0 ) and the testing accuracy under FGSM attack BID20 ) with = 0.1, 0.3, 0.5 respectively (marked as FGSM1, FGSM3 and FGSM5 in TAB0 ).

The accuracy under 50 iterations of PGD attack BID11 ) with = 0.3 is also evaluated here (marked as PGD3 in TAB0 ).

Moreover, to show the robustness of the trained DNN model to unknown attack, we test the accuracy under Gaussian noise with variance σ = 0.5 (marked as GAU5 in TAB0 ), which demonstrates the robustness against attacks from all directions.

For parameter tuning, we train and test the DNN model on MNIST data set ).Both MNIST and CIFAR-10 data set BID9 ) are used for comparing with previous work.

The MNIST test adopts the network structure provided in the tutorial of Cleverhans, which consists of three convolutional layers with 64 8 × 8 kernels, 128 6 × 6 kernels and 128 5 × 5 kernels respectively followed by a linear layer with 10 output neurons.

For the CIFAR experiment, we choose VGG-16 BID18 ) with 10 output neurons as the DNN model.

The selection of models is made to demonstrate the scalability of these defending methods.

These DNN models, without applying any further training trick, obtain the accuracy of 98.18% on original MNIST testing set after 10 epochs of training.

The accuracy on CIFAR-10 testing set would be 83.95% after 100 epochs of training.

ImageNet BID3 ) is also used to compare between Bamboo and Mixup BID23 ), where we train a ResNet-18 BID6 ) network for 90 epochs with each method.

In order to analyze the effect of the defending methods on the decision boundaries of DNNs around the testing data points, BID7 propose a linear search method to find the distance to the nearest boundary in different directions, where they gradually perturb the input data point along random orthogonal directions.

When the prediction of the perturbed input becomes different to that of the original input, the perturbation distance is used as an estimate of the decision boundary distance.

In our experiments, we follow the setting used in BID7 's work, where we use 784 random orthogonal directions for testing MNIST and 1000 random orthogonal directions for testing CIFAR-10.

For each testing data point, we find the top 20 directions with the smallest decision boundary distance for each training method, showing how the decision boundary change with different defending methods.

We also compute the average of the top 20 smallest distance across all the testing data points, implying the overall effectiveness of different methods on increasing the robustness.

To show the effectiveness of our method, we compare it with FGSM adversarial training BID5 ) with = 0.1, 0.3, 0.5, the state-of-the-art optimization based defending method distributional robust optimization (DIST) BID19 ) and the adversarial training with PGD attack BID11 ).

Newly proposed data augmentation method Mixup BID23 ) is also used for comparison.

For the implementation of these algorithms, we adopt the original implementation of adversarial training in Cleverhans, and the open-sourced Tensorflow implementations that replicate the functionality of the distributional robust optimization method 1 and the Mixup method 2 .

The hyper-parameters of these algorithms are carefully selected to produce the best performance in our experiments.

As mentioned in Section 3.2, the Bamboo augmentation has two hyper-parameters: the ball radius r and the ratio of the augmented data N .

We first analyze how the testing accuracy and model robustness change when tuning these parameters.

Figure 3a shows the influence of r and N to the testing accuracy.

When we fix the radius r, the testing accuracy increases as the number of augmented points grows up.

Adjusting the radius, however, has little impact on the testing accuracy.

Figure 3b presents the relationship between the number of augmented points and CW robustness under different ball radius.

When r is fixed, the robustness improves as data augmentation ratio N increases.

The effectiveness of further increasing N becomes less significant as N gets larger.

Under the same data amount, increasing the radius r can also enhance the robustness, while the effectiveness of increasing r saturates as r gets larger.

According to these observations, in the following experiments, we manually tuned r and N in each experiment setting for the best tradeoff between the robustness and the training cost.

FIG4 shows the top 20 smallest decision boundary on random orthogonal directions for MNIST and CIFAR-10 testing points respectively.

These results provides a visualization of the effect of different training methods on the decision boundary.

From Figure 4a and 5a we can see that adversarial training methods like FGSM (Goodfellow et al. FORMULA0 ) and Madry BID11 ) can improve the decision boundary distance on original vulnerable directions, while may cause other directions to be more vulnerable after training.

Comparing to these previous adversarial training methods, optimization based methods and data augmentation methods, our Bamboo data augmentation can provide largest gain on robustness for the most vulnerable directions, without introducing new vulnerable directions.

The average results over the whole testing set shown in Figure 4b and 5b also proof that Bamboo can further improve the overall robustness of the trained model comparing to previous methods.

TAB0 summarizes the performance of the DNN model trained with Bamboo comparing to other methods.

These results are consisted to our prior observations on the advantages and shortcomings of previous works.

The adversarial training methods can improve the robustness against the attacking methods they are trained on, especially the one with the same strength as used in training.

However, it cannot guarantee the robustness against other kinds of attacks.

Distributional robust op- We also note that the overall performance of this method on CIFAR-10 dataset is not as good as that on MNIST, possibly due to the scalability issue of the min-max optimization as elaborated in Equation (6).

A large-scale CNN and larger input space for the CIFAR-10 experiment may be too complicated to efficiently find an optimal solution.

Although not specially designed against adversarial attack, the performance of Mixup Zhang et al. (2017) is promising on robustness gain and the accuracy against adversarial attack with small strength.

However, the overall robustness achieved by Mixup, indicated by the CW robustness, is not as good as what is achieved by Bamboo.

The ImageNet experiment results showed in TAB1 show the same trend as well.

Comparing to previous methods, Bamboo achieves the highest robustness under CW attack on both MNIST and CIFAR-10 experiments, and the lowest accuracy drop when facing Gaussian noise.

Comparing to adversarial training methods with FGSM and PGD that only work best against the attacks they are trained on, Bamboo demonstrates a higher robustness against a wide range of attacking methods.

Comparing to Distributional robust optimization whose robustness drops quickly as the strength of adversarial attacks goes up, the performance of our method is less sensitive to the change of the attacking strength.

Therefore Bamboo can also be effectively applied against large-strength attacks.

Also, the overall performance of Bamboo is better than Mixup with the same amount of data augmented, implying that Bamboo is more data efficient in improving DNN robustness against adversarial attack.

All these observations lead to the conclusion that our proposed Bamboo method can effectively improve the overall robustness of DNN models, no matter which kind of attack is applied or which direction of noise is added.

In this work we propose Bamboo, the first data augmentation method that is specially designed for improving the overall robustness of DNNs.

Without making any assumption on the distribution of adversarial examples, Bamboo is able to improve the DNN robustness against attacks from all directions.

Previous analysis and experiment results have proven that by augmenting the training set with data points uniformly sampled on a r-radius ball around original training data, Bamboo is able to effectively improve the robustness of DNN models against different kinds of attacks comparing to previous adversarial training or robust optimization methods, and can achieve stable performance on large DNN models or facing strong adversarial attacks.

With the same amount of augmented data, Bamboo is able to achieve better performance against adversarial attacks comparing to other data augmentation methods.

We have shown that the resulted network robustness improves as we increase the radius of the ball or the number of augmented data points.

In future work we will discuss the theoretical relationship between the resulted DNN robustness and the parameters in our method, and how will the change in the scale of the classification problem affect such relationship.

We will also propose new training tricks better suited for training with augmented dataset.

As we explore these theoretical relationships and training tricks in the future, we will be able to apply our method more effectively on any new DNN models to improve their robustness against any kinds of adversarial attacks.

@highlight

The first data augmentation method specially designed for improving the general robustness of DNN without any hypothesis on the attacking algorithms.

@highlight

Proposes a data augmentation training method to gain model robustness against adversarial perturbations, by augmenting uniformly random samples from a fixed-radius sphere centered at training data. 