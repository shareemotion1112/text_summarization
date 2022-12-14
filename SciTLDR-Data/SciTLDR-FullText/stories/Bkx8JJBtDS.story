We propose a novel quantitative measure to predict the performance of a deep neural network classifier, where the measure is derived exclusively from the graph structure of the network.

We expect that this measure is a fundamental first step in developing a method to evaluate new network architectures and reduce the reliance on the computationally expensive trial and error or "brute force" optimisation processes involved in model selection.

The measure is derived in the context of multi-layer perceptrons (MLPs), but the definitions are shown to be useful also in the context of deep convolutional neural networks (CNN), where it is able to estimate and compare the relative performance of different types of neural networks, such as VGG, ResNet, and DenseNet.

Our measure is also used to study the effects of some important "hidden" hyper-parameters of the DenseNet architecture, such as number of layers, growth rate and the dimension of 1x1 convolutions in DenseNet-BC.

Ultimately, our measure facilitates the optimisation of the DenseNet design, which shows improved results compared to the baseline.

Deep neural networks (DNN) have achieved outstanding results in several classification tasks (Huang et al., 2017; He et al., 2016) .

There is some theoretical understanding of the workings of individual elements, such as convolutional filters, activation funtions, and normalisation (Goodfellow et al., 2016; LeCun et al., 2015; Schmidhuber, 2015) .

However, current ideas behind the DNN graph design are still based on ad-hoc principles (Mishkin et al., 2017) .

These principles are largely qualitative and tend to improve classification accuracy -examples of these principles include: an increase of the network depth (Szegedy et al., 2015) , and an increase of the representation dimensionality (by, for example, expanding the number of channels in deeper parts of the DNN) (Huang et al., 2017; He et al., 2016) .

We notice that an effective DNN graph design is largely independent of the data set, as long as the type of data (e.g., images) and task (e.g., classification) are similar.

Hence, we argue that good design principles can be encoded in a quantitative measure of the graph and should be justified by a quantitative assessment of the DNN architecture performance.

An alternative way of designing a DNN graph structure is based on (meta-)optimisation methods (Jenatton et al., 2017; Kandasamy et al., 2019; Mendoza et al., 2016; Snoek et al., 2012) .

Although useful in practice, such optimisation methods add little to our understanding of the design principles of new DNN graphs and are computationally challenging to execute.

DNNs form a hierarchical structure of filters that can be seen as a directed graph.

The first layers of this graph contain neurons that are active for low level patterns, such as edges and patches (in the case of images) (Zeiler & Fergus, 2014) , while deeper layer neurons are active for more complex visual patterns, such as faces or cars, formed by a hierarchical combination of a large number of simpler filters (Zeiler & Fergus, 2014) .

In such representation, each neuron behaves like a binary classifier of the visual pattern learned by the neuron.

Also, the strength of the activation is related to how well the pattern is matched.

We argue that this linear separability promoted by the neurons is the key ingredient behind an effective quantitative measure of model performance.

In this paper, we introduce a new measure that can be a proxy for DNN model performance.

This proposed measure is first formulated in the context of Multi Layer Perceptrons (MLPs) to quantitatively predict the classification accuracy of the model.

Then, we extend the applicability of the measure to predict the classification accuracy of the following CNNs: VGG (Simonyan & Zisserman, 2014) , ResNet (He et al., 2016) and DenseNet (Huang et al., 2017) .

The experiments demonstrate how this quantity can be used to predict the "correct" depth of a simple feed forward DNN with constraints on the parameter budget.

The experiments also show how the proposed quantity can be used to improve the design of DenseNet (Huang et al., 2017) and in the study of the effects of some important "hidden" hyper-parameters such as the dimension of 1 ?? 1 convolutions in the bottlenecks of DenseNet-BC, the number of layers, and the growth rate.

We first define the proposed measure using discrete calculations, then we relax the computation with continuous mathematics to formulate an optimisation problem for the proposed quantity.

We then make some assumptions that facilitate the calculation and optimisation of the measure.

We close the section showing how we can relax these assumptions without loosing predictive power.

The goal of DNN models is to maximise the chances to have a large number of useful patterns recognised by each layer.

In each layer, new low level patterns must be recognised without losing the ability to separate patterns that were separable in previous layers.

One possible way to realise such goal is by increasing the dimension of the representation (Vapnik, 2013) .

This argument can be quantified by computing the number of paths that affect a neuron in layer k as i<k N i , where N i is the number of nodes or channels in layer i, i.e., nodes for multi-layer perceptrons (MLP) and channels for convolutional neural networks (CNN).

Such definition resembles a concept called forward path entropy that studies the hierarchy of complex networks (Corominas-Murtra et al., 2013) .

We argue that modern CNNs are man-made complex networks developed incrementally by a large number of researchers, and so comply with many of the assumptions behind complex networks.

A measure of number of paths in DNNs was proposed by Veit et al. (2016) to explain the effectiveness of residual networks (ResNet) (He et al., 2016) , their work counts the paths of the residual connections, which is different from our approach.

The maximisation of the number of paths defined above, given a fixed number of parameters, implies that the optimal design would be of a very deep network, where each channel (or node) has size 2.

Unfortunately, this clearly does not match the collective wisdom of the field regarding the design of deep learning models.

In this paper, we aim to propose an alternative measure that can explain more effectively the superior performance of recently proposed DNNs.

State-of-the-art DNNs can be represented by a graph that takes in data with a small number of channels for CNNs (e.g., 3 channels for colour images) or a small number of input neurons for MLPs, and after each layer, it expands the dimension of this representation by increasing the number of channels or neurons.

Also when the number of neurons does not increase, the effectively used dimensions of the representation is likely to increase nonetheless.

Increasing the dimension of the representation helps in classification tasks because it makes samples belonging to different classes "more" linearly separable -this is analogous to the kernel trick argument (Vapnik, 2013) .

Consequently, we redefine paths to be the number of additional channels or neurons between two consecutive layers, as in path i = N i ??? N i???1 .

We justify this definition by arguing that for each additional layer i, only the "extra" dimensions (N i ??? N i???1 ) are beneficial to enable new linearly separable patterns without losing the separability of patterns already learned by earlier layers.

These extra dimensions allow to embed the data encoded in the previous layer in a higher dimensional space, where classes become more linearly separable (Cover, 1965; Budinich, 1991; Fink et al., 2017) .

For MLPs, we propose the following objective function to maximise:

subject to a fixed number of parameters P = i N i+1 N i , disregarding the biases for simplicity.

This measure can be extended for CNNs by replacing N i with N i ?? filter width ?? filter height.

To make the calculation of the paths manageable, following the same approach taken in the definition of entropy, we take the log(.) of the objective in equation 1:

Finally, for the definition to work in cases where the number of channels does not change (or decreases) for a few layers, we consider the upper limit of the paths calculated under the constrain of the available neurons:

where the optimal values for D i can be trivially found.

For instance, for a block of layers that have the same number of channels N i , the values of D i optimising Z in equation 3 grow linearly between D 0 and D last , with D 0 typically representing the number of input channels.

In the limit of large number of layers, it is easier to optimise Z in equation 2 as a continuous function of the layer depth.

Then, i becomes the continuous variable ?? (with ?? denoting the total number of layers), where the discrete N i is approximated by the continuous variable n(??).

Replacing the summations with integrals, the total number of parameters is defined by:

and the log of the number of paths is

From equation 4, we have d?? = dp/n 2 , so equation 5 can be re-written as

log (n 2 dn dp ) n 2 dp.

In the next section we show that, by making weak assumptions about the network structure, we can obtain a closed form solution for Z from equation 6.

Assuming that each layer has (1 + ??) times the number of channels of the previous layer, i.e., N i+1 = N i ?? (1 + ??), means that for the continuous case we have n(??) = n 0 e ???? , where n 0 is the number of channels in the first layer.

This allows us to find a closed-form solution for equation 6 (explicit calculations are presented in Appendix A.1),

Figure 1: Z as a function of ?? and n 0 (left -computed from equation 7) and as a function of ?? and n 0 (rightcomputed from equation 7 and inverting equation 10 from the Appendix A.1).

If we set the number of parameters to P = 35 ?? 10 6 as in ResNet101 (He et al., 2016) , the maximum value for Z is at ?? = 0.0178 with the number of channels in the first layer n 0 = 56.5.

This means that the total number of layers is ?? = 167.5 and the optimal size of the last layer is n(??) = 1113 ( Fig. 1) .

These numbers are remarkably close to the actual hyperparameter values for large CNN architectures, such as ResNet101 (He et al., 2016) , trained on large image classification tasks (Deng et al., 2009 ).

This suggests that large image classification tasks (Deng et al., 2009 ) may not benefit from larger and deeper neural networks with a simply-connected feeedfoward architecture.

This also suggests that to really exploit models bigger than the current state-of-the-art CNNs, we will need problems more complex than the 1000-class ImageNet (Deng et al., 2009) .

For the case above with a fixed P , the maximum value for Z is 251.

This number corresponds to 10 109 paths, which represents an extremely large number of ways to activate the output neurons -this can qualitatively explain how CNNs can learn complex tasks.

By relaxing the constrain on the functional form of n(??), defined above, we can rewrite equation 6 as: Z(n) + ??Z(n, ??n) = Z(n + ??n).

A necessary condition for Z being an optimum over the space on functions n is ??Z = 0.

This imposes a set of conditions on n(??) which is equivalent to a differential equation and boundary conditions at ?? = 0 and ?? = ??. Solving numerically this differential equation, as shown in details in Appendix A.2, results in a solution qualitatively very similar to an exponential, confirming the validity of the analytical results obtained above.

The contribution to the paths for convolutions are proportional to the filter width and height, as explained in Sec. 2.1.

For example, if the filter has size 3 ?? 3, the contribution to Z is 9 times what it would be for a filter of size 1 ?? 1.

In the case of the DenseNet-BC bottleneck block, we have first a layer of 1 ?? 1 and then a layer of 3 ?? 3 convolutions (Fig. 5a , middle).

In the original DenseNet paper (Huang et al., 2017) , the number of channels in the 1 ?? 1 convolutions is r = 4 times the number of channels in the 3 ?? 3 convolutions.

Therefore, this particular block has Z = log ((rk)(9k ??? rk)/9), where the 1 ?? 1 layers contributes rk, because its input has a skip connection, and the 3 ?? 3 layer contributes (9k ??? rk)/9 because each convolution has 9 inputs and its input has no skip connection.

If we optimise r for the highest Z, we find r 4.5, with small r favored by economy of parameters.

This explains well the value of 4 used in the original DenseNet-BC (Huang et al., 2017) .

Increasing r further is expected to bring little return in performances, as Z does not increase and the network ends up having unnecessary channels in the 1 ?? 1 filters.

There is a vast collection of papers that describe the "tricks of the trade", consisting of suggestions about network structures, activation functions, different types of network layers, etc. (Goodfellow et al., 2016; LeCun et al., 2015; Schmidhuber, 2015) .

Even though such suggestions are useful from a practical point of view, they are fundamentally qualitative descriptions that lack rigour and offer little insight into the design of new classification models.

Some other papers have aimed to characterise the performance of deep learning model in a post-hoc manner, and consequently offered little help in the design of new deep learning models (Keskar et al., 2016; Lee et al., 2016; Liao et al., 2018; Littwin & Wolf, 2016; Sagun et al., 2016; Soudry & Carmon, 2016 ).

An alternative to a proper understanding of the DNN graph structure is to turn the problem of designing new models into an optimisation task.

Unfortunately, this task often depends on what is infamously known as the grad student optimisation: a gruesome trial and error of innumerous DNN models (Kandasamy et al., 2019) .

Thankfully, smarter approaches have been developed, such as Bayesian optimisation (Jenatton et al., 2017; Kandasamy et al., 2019; Mendoza et al., 2016; Snoek et al., 2012) , reinforcement learning (Zoph & Le, 2016) , evolutionary methods , and random optimisation (Li & Talwalkar, 2019) .

These optimisation methods are useful, but still suffer from scalability and robustness issues to a certain extent.

Also, when the network is finally optimised, there is hardly any insight on why the selected model was chosen.

What we propose in our paper addresses these two issues: 1) we propose a method that increases our current understanding of how deep learning methods work, and 2) our method does not rely on computationally expensive optimisation processes to enable the design of a potentially interesting model.

Therefore, such increasing of our understanding can potentially improve our ability to propose new structures that have higher chances of succeeding at producing more effective DNNs.

Also, the low run-time and memory costs involved in the calculation of our proposed measure makes it significantly more efficient in terms of computational costs than the optimisation approaches mentioned above.

The main challenge involved in our approach is that the computation of Z requires a non-trivial understanding of the model to design the formula to compute Z. We provide general guidelines on how to formulate Z in Sec. 5

We first run experiments using simple setups, in terms of data set and models, to show the effectiveness of our proposed measure.

Then, we run an experiment that compares the proposed measure Z for several DNN models: VGG (Simonyan & Zisserman, 2014) , ResNet (He et al., 2016) and DenseNet (Huang et al., 2017) .

We then optimise the DenseNet architecture to maximise our measure Z. All experiments are run on two data sets.

Fashion-MNIST (Xiao et al., 2017 ) is a 10-class data set that has a training set of 60,000 grey scale images and a testing set of 10,000 images, where each image has 28 ?? 28 pixels.

CIFAR-10 (Krizhevsky & Hinton, 2009 ) has 10 classes and a total of 50,000 32 ?? 32 training colour images and 10,000 testing images -this data set is extended to CIFAR-10+, which relies on standard data augmentation (mirroring and shifting) to build the training set (He et al., 2016) .

All data sets are balanced, i.e., they have the same number of training images per class.

We train the models for the FashionMNIST experiments with sgd+momentum for 30 epochs, learning rate 0.001, momentum 0.9, and mini-batch size of 100.

We follow the experimental setup suggested by Huang et al. (2017) We start with simple experiments involving MLPs on CIFAR-10.

The goal of this experiment is to show an intuitive result to most deep learning practitioners: assuming a fixed parameter budget, the optimal network structure shows a compromise between depth and width.

We follow closely the architectural choices suggested by Pennington et al. (2017) .

Our networks are MLPs containing 500K parameters with a depth in a range from few layers up to 800 layers.

As shown by Pennington et al. (2017) , the training of such architectures involves an optimal choice of initialisation method and learning rate.

In particular, we will use the empirical relation for the dependency of the optimal learning rate with depth (1/L) and the theoretical result for the optimal variance of the weight initialisation (1 ??? ?? 2 w 1/L).

We explore a range of values around the ones suggested by such relations to make sure that we pick the optimal values for the initialisation method and learning rate.

Also, we train all models with weight decay of 10 ???5 .

To simplify the problem, we preprocess CIFAR-10 by running PCA on the training set and keeping the first 100 whitened PCs.

Fig. 2(left) shows that for a given budget of parameters, the maximum trainable depth is obtainable by optimising Z (equation 3).

The best models train to an accuracy comparable to what is obtained by Pennington et al. (2017) .

Fig. 2(right) shows the growth of dimensionality (illustrative of D i in equation 3) of the feature representation in a trained network, starting with 100 dimensions up to the size of the last layer.

ET AL., 2017) In this section, we try our proposed measure Z using a simple CNN, consisting of a variable number of layers ??, a variable number of channels in the first layer n 0 , fixed filter sizes of 3 ?? 3 and final fully connected layer before the softmax activation, and a number of channels that grows exponentially with the layer index according to the function n i = n 0 e ??i .

Figure 3 shows the accuracy of the models (multiple accuracy results at each value of ?? and n 0 are obtained from results of epochs between 10 and 30) and the value of Z as a function of depth ??. In all experiments, the number of parameters of all networks is constrained to be P = 5 ?? 10 5 .

For most models, the computed Z from equation 2 is shown to be a good predictor of the accuracy of the models, supporting the idea that optimising the network architecture to have a large Z is beneficial to get an optimal design.

We estimate Z on several "real world" CNN architectures trained and tested on ImageNet (Deng et al., 2009) .

For VGG (Simonyan & Zisserman, 2014) and ResNet (He et al., 2016) , the number of layers does not grow within each block, so we rely on equation 3 and the results observed in Sec. 4.1 to estimate value of Z. Recall from Sec. 4.1 that the number of channels effectively used by the data in the network grows gradually between the input dimension of the block N in and the output N out .

A linear growth in the number of channels used (D i ) within each block gives the optimal value for Z. For a block with m layers in a standard feed forward CNN, such as VGG16, we have Z = m log ???N , where ???N = Nout???Nin m with the number of parameters defined by P = mN 2 out .

For a block of the ResNet, we have one skip connection every two layers, so the contribution to the number of paths is ??? log N for the layer with skip connection, and ??? log ???N for the layer without skip connection, where we again use that that the optimal configuration for D i grows linearly between N in and N out .

For ResNet, the number of parameters is P = mN 2 out .

In a DenseNet block, the number of channels per layer is equal to the growth-rate k. The dimensionality of the data is equal to the growth rate plus the dimensions of the previous layer.

In each layer ???N = N i ??? N i???1 = k, so Z = m log k. For the DenseNet block, the number of parameters is defined by:

in .

This means that if N in < mk, the number of parameters grows quadratically with m and k. Also, Z grows linearly in depth and logarithmically in growth-rate.

Qualitatively, this means that the optimal design has a small growth-rate and a depth that is not too high (to avoid a quadratic growth of the number of parameters).

This imposes a limit for the maximum depth of each block (m max ??? N in /k), which is, empirically, close to how the DenseNet architectures have been designed.

This principle can be used to explain the effectiveness of DenseNet(-BC) (Huang et al., 2017) blocks and to optimise the design of new DenseNet(-BC) variants.

For all models above, we show how to compute the Z value per block.

Therefore, we also need to take all the blocks into account when computing the final values for Z. Table 1 shows the Z values computed for various networks recently proposed in the field, along with their number of parameters P and classification error results on ImageNet (Deng et al., 2009) .

The quantity Z is a good predictor for the accuracy of the networks, for a given parameter budget P .

We also compare accuracy and Z in Fig. 4 on CIFAR-10+ (Krizhevsky & Hinton, 2009) , which shows the accuracy of models trained with different values of the ratio r for a DenseNet-BC bottleneck block.

In general, it is observed that the accuracy of the model saturates for a ratio larger than ??? 4.5.

Recall from Sec. 2.3, that Z is maximised for r ??? 4.5, and the accuracy remains relatively high for r > 4.5.

We conjecture that regularisation techniques applied during training reduces the number of "effective" channels, corresponding to the set of D i values that maximise Z. Therefore, Z explains when increasing further the number of channels (and parameters) of a 1 ?? 1 layer of a bottleneck layer will not yield improved performances.

To test the usefulness of our measure for the design of a new DenseNet variant, we optimise the block of a DenseNet-40-12 to increase Z (initially for this model, Z = 89) -this is the smallest of the DenseNet architectures proposed by Huang et al. (Huang et al., 2017) .

The optimisation is based on the reduction of the growth rate and increase of the number of layers by the same factor.

This approach is chosen because it is a simple way to increase Z without changing the number of input and output channels (boundary conditions of the blocks).

The first architecture that we propose is a DenseNet-148-3 with 148 layers and growth rate of 3, which produces Z = 148 that compares favourably with the original value.

The number of parameters and classification results for the new and original models are shown in the table of Figure 5b .

We propose another optimisation of the structure of the bottleneck blocks of the DenseNet-BC architecture.

Like before, we focus on optimising one of the smallest elements of the graph, which, we argue, should be largely independent from the data set.

In the original architecture, the BC block is built with a 1 ?? 1 convolutional layer followed by a 3 ?? 3 convolutional layer, giving Z = (rk)((9k ??? rk)/9).

The 1 ?? 1 convolutional layer has 4 times more channels than the corresponding 3 ?? 3 convolutional layer.

We replace the single 1 ?? 1 layer with two 1 ?? 1 layers that are densely connected, as shown in Fig. 5a on the rightmost panel.

With the new design, the number of paths becomes larger, with Z = log ((rk)(rk)(9k ??? 2rk)/9), where rk is the number of channels of each 1 ?? 1 layer.

The optimum Z for the minimum number of parameters is achieved with r = 3 -results are displayed in the table of Figure 5b , where we show two different proposed versions of the DenseNet-BC architecture.

In Figure 4 (right), we show the accuracy of different versions of this architecture as a function of r. Similarly to what happens to the simpler DenseNet-BC, the accuracy of the model increases significantly up to the optimal r, and then remains stable.

This behaviour strengthens our conjecture that regularisation techniques effectively reduce the number of channels and parameters of the network.

We finally performed a single experiment with the DenseNet-denseBC-250-24, r = 3 (last line in the table of Fig. 5b ) improving over the baseline and DenseNet-BC-190-40, which represents the most accurate model from (Huang et al., 2017) .

Our measure (Z) is calculated based only on the graph structure of the model and assumes that the initialisation and training strategies are close to optimal.

We argue that the choice of these strategies is important because it guarantees the realisation of the model potential for a particular classification problem.

However, we see these strategies as somewhat orthogonal to network design.

We show in Fig. 3 that the value of Z is a good predictor of the optimal depth ?? for a simple feed forward CNN.

We also show that the value of Z is a good predictor of accuracy for small values of r (channels of bottleneck layer), and for larger values of r, while Z "saturates", the accuracy tends to remain stable.

We conjecture that this happens because regularisation techniques are likely to reduce the effective number of channels when this allows an increase in the number of paths.

We find indicative support of this mechanism from the study of the dimensionality of the PCA decomposition of MLP classifiers ( Fig. 2(right) ).

Figure 6: This example shows the rules to calculate the contribution to the total number of paths given by each layer in a NN.

Each N i can be ??? than the corresponding channels in the NN.

We will leave the proof for this conjecture for a future work, note that this behaviour does not undermine the method -the implementation of models with optimal Z appears to offer best performance with the minimum number of parameters.

We now provide guidelines on how to formulate Z -see Fig. 6 .

When there is a skip connection around a layer, the number of channels in such layer can contribute fully to the paths (N i ).

When two layers are in sequence, the contribution to the paths is the difference of the channels after the reduction due to regularisation (N i ??? N i???1 ).

When the filter size changes (e.g., 1 ?? 1 followed by 3 ?? 3 convolution), the channels of the first layer need to be divided by the ratio of the dimension of the filters (

).

In the common case of convolution blocks with constant number of channels, the supremum of eq 3 gives that the optimal D i 's grow linearly between N in and N out .

In conclusion, in this manuscript we propose a quantitative principle for predicting the accuracy of a deep learning classifier that takes into account the structure of the DNN graph.

We hope that this will be the basis for more general principles for network design and it will lead to a deeper understanding of DNNs.

As a practical application of this work, we also propose a new DenseNetdenseBC architecture that improves over baseline results.

If we assume that each layer has (1 + ??) the number of channels of the previous one:

the continuous case becomes:

A.2 MAXIMISING Z AS A FUNCTIONAL TO SOLVE CONTINUOUS COUNTING A more effective way to maximise Z over all functions n(??) under the constraint of a fixed number of parameters P relies on the use of calculus of variations.

More specifically, from equation 6, we have

log ((n + ??n) 2 ?? d(n+??n) dp ) (n + ??n) 2 dp,

where a maximum point of the function Z implies ??Z = 0 for any (small) ??n.

By expanding Z +??Z in the first order in ??n, discarding higher order terms, and relying on integration by parts, we obtain:

log ((n + ??n) 2 d(n+??n) dp )(1 ??? 2??n/n) n 2 dp P 0 log (n 2 dn dp ) n 2 ??? 2??n log (n 2 dn dp ) n 3 + 2??n n 3 + d??n dp n 2 dn dp dp P 0 ??? ??? ??? ??? log (n 2 dn dp ) n 2 + ??? ??? ??? ??? ??? 2 log (n 2 dn dp ) n 3 + 2 n 3 ??? d 1 n 2 dn dp dp ??? ??? ??? ??? ??n ??? ??? ??? ??? dp.

In the last step we integrate by parts and set the boundary conditions for n. To maximise the functional in equation 14, we set boundary conditions with ??n(P ) = ??n(0) = 0.

Note that the first term of the integral in equation 14 is just Z, as defined in equation 6.

We search for an optimal n such that ??Z does not grow (at first order), no matter the choice of ??n.

This is equivalent to have ??Z = 0 for any ??n n(p), which means that for any p the factor inside the integral is zero -that is: ??? 2 log (n 2 dn dp ) n 3 + 2 n 3 ??? d 1 n 2 dn dp dp = 0.

In other words, the optimal n(p) has to respect this differential equation.

By defining v = dn dp we can re-write equation 15 as: ??? 2 log (n 2 v) n 3 + 2 n 3 + 2 n 3 + dv dp n 2 v 2 = 0.

Since dv dp = dv dn dn dp = dv dn v, we can remove all the explicit occurrences of p and obtain:

This differential equation is a first order ordinary differential equation and, given the initial condition for v at n(0), can be integrated numerically to obtain a family of solutions for v as a function of n. Then, n(p) and n(??) can be integrated numerically.

Fig. 7 shows a family of solutions for n 0 = 56.5.

The thick line is the solution with the largest Z 272, which is larger than 251, obtained in Sec. 2.2 with the exponential assumption (i.e., n(??) = n 0 e ???? ).

For comparison, we show this exponential function from Sec. 2.2 as a thin black line.

The system predicts a network depth ?? ??? 174 (the highest point in Fig. 7 right panel) , which is comparable with the value of 167.5 found in Sec. 2.2.

@highlight

A quantitative measure to predict the performances of deep neural network models.

@highlight

The paper proposes a novel quantity that counts the number of path in the neural network which is predictive of the performance of neural networks with the same number of parameters.

@highlight

The paper presents a method for counting paths in deep neural networks that arguably can be used to measure the performance of the network.