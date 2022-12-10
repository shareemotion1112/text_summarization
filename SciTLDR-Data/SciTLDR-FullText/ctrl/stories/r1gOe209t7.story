Recently convolutional neural networks (CNNs) achieve great accuracy in visual recognition tasks.

DenseNet becomes one of the most popular CNN models due to its effectiveness in feature-reuse.

However, like other CNN models, DenseNets also face overfitting problem if not severer.

Existing dropout method can be applied but not as effective due to the introduced nonlinear connections.

In particular, the property of feature-reuse in DenseNet will be impeded, and the dropout effect will be weakened by the spatial correlation inside feature maps.

To address these problems, we craft the design of a specialized dropout method from three aspects, dropout location, dropout granularity, and dropout probability.

The insights attained here could potentially be applied as a general approach for boosting the accuracy of other CNN models with similar nonlinear connections.

Experimental results show that DenseNets with our specialized dropout method yield better accuracy compared to vanilla DenseNet and state-of-the-art CNN models, and such accuracy boost increases with the model depth.

Recent years have seen a rapid development of deep neural network in the computer vision area, especially for visual object recognition tasks.

From AlexNet, the winner of ImageNet Large Scale Visual Recognition Challenge (ILSVRC) , to later VGG network BID14 and GoogLeNet , CNNs have shown significant success with its shocking improvement of accuracy.

Researchers gradually realized that the depth of the network always plays an important role in the final accuracy of one model due to increased expressiveness BID2 BID3 .

However, simply increasing the depth does not always help due to the induced vanishing gradient problem BID0 .

ResNet BID4 has been proposed to promote the flow of information across layers without attenuation by introducing identity skip-connections, which sums together the input and the output of several convolutional layers.

In 2016, Densely connected network (DenseNet) BID7 came out, which replaces the simple summation in ResNet with concatenation after realizing the summation may also impede the information flow.

Despite the improved information flow, DenseNet still suffers from the overfitting problem, especially when the network goes deeper.

Standard dropout BID16 has been used to combat such problem, but can not work effectively on DenseNet.

The reasons are twofold: 1) Feature-reuse will be weakened by standard dropout as it could make features dropped at previous layers no longer be used at later layers.

2) Standard dropout method does not interact well with convolutional layers because of the spatial correlation inside feature maps BID19 .

Since dense connectivity increases the number of feature maps tremendously -especially at deep layers -the effectiveness of standard dropout would further be reduced.

In this paper, we design a specialized dropout method to resolve these problems.

In particular, three aspects of dropout design are addressed: 1) Where to put dropout layers in the network?

2) What is the best dropout granularity?

3) How to assign appropriate dropout (or survival) probabilities for different layers?

Meanwhile, we show that the idea to re-design the dropout method from the three aspects also applies to other CNN models like ResNet.

The contributions of the paper can be summarized as follows:• First, we propose a new structure named pre-dropout to solve the possible feature-reuse obstruction when applying standard dropout method on DenseNet.• Second, we are the first to show that channel-wise dropout (compared to layer-wise and unit-wise) fit best for CNN through both detailed analysis and experimental results.• Third, we propose three distinct probability schedules, and via experiments we find out the best one for DenseNet.• Last, we provide a good insight for future practitioners, inspiring them regarding what should be considered and which is the best option when applying the dropout method on a CNN model.

Experiments in our paper suggest that DenseNets with our proposed specialized dropout method outperforms other comparable DenseNet and state-of-art CNN models in terms of accuracy, and following the same idea dropout methods designed for other CNN models could also achieve consistent improvements over the standard dropout method.

In this section, first we will review some basic ideas behind DenseNet, which is also the foundation of our method.

Then the standard dropout method along with some of its variants will be introduced as a counterpart of our proposed dropout approach.

Figure 1 : Examples of the dense block and the DenseNet with standard dropout method.

White spots denote dropped neurons.

The right figure shows that the same feature maps (orange color) after dropout layer will be directly sent to later layers, which makes the dropped features always unavailable to later layers.

DenseNet was first proposed by BID7 , which features a special structure-dense blocks.

Figure 1a gives an example of the dense block.

One dense block consists of several convolutional layers, each of which output k feature maps, where k is referred to as the growth rate in the network.

The most important property of DenseNet is that for each convolutional layer inside the dense block, the input is the concatenation of all feature maps from the preceding layers within the same block, which is also known as the dense connectivity.

With dense connectivity previous output features could be reused at later layers.

DISPLAYFORM0 Equation FORMULA0 shows such kind of relationship clearly.

Here X i represents the output at layer i.[, ] denotes concatenation operation.

L is a composite function of batch normalization (BN) BID10 , rectified linear unit (ReLU) BID1 and a 3 × 3 convolutional layer.

With the help of feature-reuse, DenseNet achieves a better performance and turns out to be more robust BID9 .

However, dense connectivity will cost a large amount of parameters.

To ease the consumption of parameters, a variant structure, named DenseNet-BC, came out.

In this structure, a 1 × 1 layer is added before each 3 × 3 layer to reduce the depth of input to 4k.

Standard dropout method BID16 ) discourages co-adaptation between units by randomly dropping out unit values.

Stochastic depth BID8 extends it to layer level by randomly dropping out whole layers.

Other alternative examples include DropConnect BID20 , which generalizes dropout by dropping individual connections instead of units (i.e., dropping several connections together), and Swapout BID15 which skips layers randomly at a unit level.

These previous works mainly focus on one aspect of dropout method, i.e., the dropout granularity.

We are the first who gives a thorough study of the dropout design from all three aspects: dropout location, dropout granularity and dropout probability.

In our evaluation, we will not only give the overall accuracy improvement, but also the breakdown along all these three aspects.

Meanwhile, all these methods above will impede the feature-reuse in DenseNet since dropped features in previous layers will never be available to later layers.

Figure 1b shows an example of the feature-reuse obstruction when applying standard dropout method on DenseNet.

As previously mentioned, standard dropout method will have some limitations on DenseNet: 1.

it could impede feature-reuse; 2.

The effect of dropout will be weakened by the spatial correlation.

To solve these problems and further improve model generalization ability, we propose the following structures.

DISPLAYFORM0 Figure 2: Examples illustrating data flow in one dense block of standard dropout method and predropout method.

The blue boxes represent dropout layers, while the yellow boxes represent composite functions.

X 0 , X 1 and X 1 are the data tensors.

W stands for the output of the dense block.

Pre-dropout structure aims at solving the possible feature-reuse obstruction when applying standard dropout to DenseNet.

As mentioned in Section 2.2, due to the dense connectivity, standard dropout will make features dropped at previous layers no longer be used at later layers, which will weaken the effect of feature-reuse in DenseNet.

To retain the feature-reuse, we come up with a simple yet novel idea named pre-dropout, which instead places dropout layers before the composite functions so that complete outputs from previous convolutional layers can be transferred directly to later layers before the dropout method is applied.

Meanwhile one extra benefit from pre-dropout is that we can stimulate much more feature-reuse patterns in the network because of different dropout patterns applied before different layers.

Figure 2 illustrates the differences between standard dropout and pre-dropout.

In the following, we will explain these two benefits in details.

Suppose the input to Figure , then it would also be used as the input to the dropout layer B 1 .

The calculation of B 1 could be regarded as element-wisely multiplying X standard 1 with Θ 1 , a tensor of random variables following Bernoulli(p 1 ).

Thus we can get DISPLAYFORM0 where represents element-wise multiplication.

Then X standard 0 and X standard 1 will be concatenated together as the input to L 2 , i.e., X standard 0 DISPLAYFORM1 , here ⊕ denotes concatenation.

So on and so forth.

Finally, the output of the dense block can be written as DISPLAYFORM2 Similarly we can get the mathematical representations from Figure 2b , which shows the data flow in a dense block with pre-dropout method, DISPLAYFORM3 where Θ i,j represents the tensor of dropout layer connecting from the output of layer i to the input of layer j, in which random variables follow Bernoulli(p i,j ).Comparing Equations 2 with 4, we can notice that pre-dropout method will allow better feature-reuse than standard dropout method.

For instance, the outputs of standard dropout would become zero and remains the same as inputs to all following layers when Θ 1 equals to zero.

However pre-dropout solves this problem.

In pre-dropout method, X pre 1 would be multiplied by different independent tensors Θ 1,2 , Θ 1,3 , · · · , Θ 1,n+1 , such that for any specific X pre 1ijk even if Θ 1,2 makes X pre 1ijk be zero at L 2 , we still have a chance to reuse this feature at later layers.

Meanwhile we could also find that pre-dropout method could stimulate much more feature-reuse patterns in the network.

For example, in Figure 2a , once X standard 1 goes through dropout layer B 1 , the same output will always be reused.

Whereas in pre-dropout method, every time before X pre 1 is utilized as the input to the next layer, it would be multiplied by a different tensor.

In Figure 2b , the contributions of X pre 1 are actually two distinct features, since Θ 1,2 and Θ 1,n+1 are independent.

Note that similar feature-reuse obstruction also exists when applying standard dropout on other CNN models with shortcut connections, such as ResNet BID4 , Wide-ResNet BID21 and RoR BID22 .

Thus pre-dropout method could work for those networks as well.

When designing our specialized dropout method, dropout granularity is also an important aspect to be considered.

FIG2 shows three different granularity: unit-wise, channel-wise and layer-wise.

Standard dropout is one kind of unit-wise method, which has been proved useful when applied on fully connected layers BID16 .

It helps improve the generalization capability by breaking the strong dependence between neurons from different layers.

However when applying the same method on convolutional layers, the effectiveness of standard unit-wise method will be hampered due to the strong spatial correlation between neurons within a feature map -although dropping one neuron can stop other neurons from replying on that particular one, they will still be able to learn from the correlated neurons in the same feature map.

To cope with the spatial correlation, we need dropout at better granularity.

Layer-wise method drops the entire layers, whcih refer to the outputs from previous layers, inside the input tensor.

However, layer-wise dropout is prone to discard too many useful features.

Channel-wise method, which will drop a entire feature map at a given probability, strikes a nice trade-off between the two granularity above.

Our experiments also confirmed that channel-wise works the best for regularizing DenseNet.

Meanwhile since the spatial correlation exists in all types of convolutional layers, our analysis above should also work for other CNN models whenever a dropout method is applied.

Channel-wise dropout can still be improved when applied on DenseNet.

Notice that naive channelwise dropout cannot add various degrees of noise to different layers due to the deterministic survival Figure 4: Different probability schedules.

The numbers besides blue boxes represent survival probabilities.

Left figure shows the linearly decaying schedule v 1 which applies linearly decaying probabilities on different layers of DenseNet.

Right figure shows v 3 schedule which applies various probabilities on different portions of the input to a convolutional layer depending on distances between layers generating those portions and the input layer.probability, and since in DenseNet dense connectivity makes the sizes of inputs at different layers quite different, such variation seems to be helpful.

We believe the model would benefit from such kind of noise.

Thus, in our design, to further promote model generalization ability, besides predropout and channel-wise dropout, we also apply stochastic probability method to introduce various degrees of noise to different layers in DenseNet.

In experiment part we compare the stochastic probability method with the deterministic probability method, and results show that stochastic method could get a better accuracy on DenseNet.

Since in CNNs the degree of feature correlation and the importance of features are generally different at different layer depths, such probability schedule would always be desired.

Once we adopt the stochastic method, one natural question arises: how can we assign probabilities for different convolutional layers in order to achieve the best accuracy?

Actually it is hard to find out the best specific probability for each layer.

However, based on some observations we are still able to design some useful probability schedules.

Figure 4 gives examples on the different schedules below.

Same as before, we've done some experiments on them and adopt the best one in our design.

One observation is that in the shallow layers of DenseNet, the number of feature maps used as input is quite limited and as the layers go deeper the number will become larger.

Meanwhile in CNN highlevel features are prone to be repeated.

Thus intuitively we propose a linearly decaying probability schedule.

We refer to it as v 1 .

For this probability schedule, in each dense block, the starting survival probability at the first convolutional layer is 1.0 while the last one is 0.5.

Recall that in Section 3.1 we use Θ i,j to represent the tensor of dropout layer connecting from the output of layer i to the input of layer j, in which random variables follow Bernoulli(p i,j ).

So schedule v 1 will have the following properties, 1.

For fixed j and ∀i, p i,j = C, C is a constant.

Particularly, p i,1 = 1.0 and p i,n+1 = 0.5; 2.

For fixed i, p i,j is monotonically decreasing with j.

To the best of our knowledge, dropout will add per neuron different levels of stochasticity depending on the survival probability and maximum randomness is reached when probability is 0.5.

Meanwhile the sizes of inputs in DenseNet will gradually increase.

So to reduce the total randomness in the model, we design another schedule v 2 , which is also a reverse version of v 1 , i.e., the starting probability for the first layer is 0.5 whereas the last one is 1.0.

Similarly, in this schedule, we will have, 1.

For fixed j and ∀i, p i,j = C, C is a constant.

Particularly, p i,1 = 0.5 and p i,n+1 = 1.0; 2.

For fixed i, p i,j is monotonically increasing with j.

Additionally, we observe that in DenseNet deeper layers tend to rely on high-level features more than low-level features, such phenomenon is also mentioned in BID7 .

Based on that, schedule v 3 is proposed.

For this schedule, we decide survival probabilities for different layers' outputs based on their distances to the convolutional layer, i.e., the most recent output from previous layers will be assigned with the highest probability while the earliest one gets the lowest.

In our implementation, for the input to the last layer in one dense block we assign the most recent output in it with probability 1.0 and the least with 0.5.

Then based on the number of previous layers concatenated, we can calculate the probability difference between two adjacent layers' outputs to decide probabilities for other portions of the input.

The corresponding properties of v 3 can be summarized as, DISPLAYFORM0 , where d(i, j) denotes the distance between layer i and j; 2.

For fixed j, p i,j is monotonically increasing with i. Particularly, when i = j −1, p i,j = 1.0, and p 0,n+1 = 0.5; 3.

For fixed i, p i,j is monotonically decreasing with j. Particularly, when j = i+1, p i,j = 1.0.In conclusion, by using v 3 for inputs to deep layers in DenseNet low-level features from shallow layers will always be dropped with higher probabilities whereas high-level features can be kept with a good chance.

Meanwhile, survival probabilities for the output of one layer will become smaller as layers go deeper, which is also intuitive as outputs from earlier layers have been used for more times so there exists higher probability that such outputs will not be used again later.

Also notice that the idea to apply different dropout probability at different layers can also be applied to other networks since the variation of inputs at different layers are quite common CNN models.

In our experiments, we mainly use two datasets: CIFAR10 and CIFAR100, containing 50k training images and 10k test images, a perfect size for a model of normal size to overfit.

Meanwhile, we apply normal data augmentation on them which includes horizontal flipping and translation.

When implementing our models, we try to retain the same configurations of original DenseNet though further hyper-parameter tuning might generate better results since dropout will slow convergence BID16 .

Briefly four DenseNet structures are used: DenseNet-40, DenseNet-BC-76, DenseNet-BC-100 and DenseNet-BC-147.

The number at the rear of the structure name represents the depth of the network.

Growth rate k for all structures is 12.

We adopt 300 training epochs with batch size 64.

The initial learning rate is 0.1 and is divided by 10 at 50% and 75% of the total number of training epochs.

During training process a fixed survival probability 0.5 is used for non-stochastic dropout methods.

The test error is reported after every epoch and all results in our experiments represent test errors after the final epoch.

As an important part of our work, we want to know how DenseNets with our specialized dropout method compare with other models.

In this section, we use three different DenseNet structures and test on CIFAR10 and CIFAR100 augmentation datasets.

Note that our specialized dropout won't incur additional parameters in the network.

From results in TAB0 , we can find that DenseNets with our specialized dropout method get the best accuracy on both datasets.

In particular specialized dropout could consistently have good improvements over standard dropout method and DenseNet-BC-100 with our specialized dropout which only contains 0.8M parameters could outperform a 1001 layer ResNet model.

Notice that data augmentation (which is always applied) already imposes generalization power to the model, which could make other regularization methods less effective.

Further, based on the results, our specialized dropout method gives better accuracy improvements on larger DenseNets, e.g., on CIFAR100 dataset, the improvements of our specialized dropout method would increase with the depth of the network.

In order to show the effectiveness of predropout structure, we compare unit-wise predropout method with standard dropout on two DenseNet structures.

We run experiments on CIFAR10 augmentation dataset.

Results are shown in TAB1 .

From TAB1 , pre-dropout structure achieves better accuracy than standard dropout on both of the two DenseNet structures.

Such results coincide with our analysis that pre-dropout could incur better feature-reuse and stimulate various features in the network.

Meanwhile according to the analysis in Section 3.2 one factor that could disadvantage pre-dropout here is that correlated features can compensate for dropped features in standard dropout method.

In this section, we use DenseNet-BC-76 structure to compare the three dropout granularity on CIFAR10 and CIFAR100 augmentation datasets respectively.

The results are shown in TAB2 .

TAB2 shows that channel-wise dropout achieves the best accuracy on both of two datasets.

So in our specialized dropout method, we adopt channel-wise dropout granularity.

Also from TAB2 , layer-wise dropout always has the worst performance.

The reason could be that layer-wise dropout discards some useful features at one time, as a result a loss of accuracy is observed.

In Section 3.3, we argue why DenseNet would benefit from the variation of noise at different layers.

In order to validate this idea, we compare the proposed three stochastic probability schedules to the standard version with a uniform dropout probability (0.5) on DenseNet-BC-76.

Our empirical study indicates that v 3 always is the best among the three schedules, whereas v 2 is the worst.

Thus, we pick schedule v 3 as our final specialized dropout method.

TAB3 gives an example result for DenseNet-BC-76 on CIFAR10 augmentation dataset.

Recall that the number of feature maps to shallow layers in DenseNet is very limited and schedule v 2 applies lower survival probabilities on these layers, thus from the results we can see although v 2 reduces the total randomness in the model, a loss of relatively larger quantity of low-level features could still hurt the accuracy.

Furthermore, we can find that the same effect of v 1 also exists in v 3 , i.e., relatively higher survival probabilities are assigned for shallow layers and lower ones for deep layers.

Besides v 3 can also help deep layers rely more on high-level features, which could be the reason making v 3 better than v 1 .

Dense-BC-148 with specialized dropout Figure 5 : Training on CIFAR100.

Thin curves denote training error, and bold curves denote test error.

We also want to figure out the reasons why our specialized dropout method could result in an accuracy improvement.

To reveal the reasons, we compare the training/test errors during the training procedures of the normal DenseNet and the one with our specialized dropout method.

Figure 5 shows such comparison on DenseNet-BC-148.

As shown in the figure, the specialized dropout version reaches slightly higher training error at convergence, but produces lower test error.

This phenomenon indicates that the improvement of accuracy comes from the strong regularization effect brought by the specialized dropout method, which also verifies that the specialized dropout method could improve the model generalization ability.

Following the idea of designing a specialized dropout method for DenseNet, we also want to explore whether such idea could also apply to other state-of-the-art CNN models.

Here we choose AlexNet, VGG-16 and ResNet to conduct the experiments.

Similar to the DenseNet, we design the specialized dropout method for each model from three aspects.

We apply pre-dropout structure and channel-wise granularity for all specialized dropout methods and decide dropout probability by the size of input.

In order to reduce the total randomness in a model, the largest input will have the dropout probability 0 while the smallest one corresponds to the probability 0.5.

Layers between the two will follow a linear increasing/decreasing schedule to assign the dropout probability.

The results are shown in TAB4 .

From results in TAB4 we can see that models with the specialized dropout method all outperform its original counterparts, which indicates that our idea to design a specialized dropout method could also work in other CNN models.

The effectiveness of our idea is also validated by such results.

In this paper, first we show problems of applying standard dropout method on DenseNet.

To deal with these problems, we come up with a new pre-dropout structure and adopt channel-wise dropout granularity.

Specifically, we put dropout before convolutional layers to reinforce feature-reuse inside the model.

Meanwhile we randomly drop some feature maps in inputs of convolutional layers to break dependence among them.

Besides to further promote model generalization ability we introduce stochastic probability method to add various degrees of noise to different layers in DenseNet.

Experiments show that in terms of accuracy DenseNets with our specialized dropout method outperform other CNN models.

<|TLDR|>

@highlight

Realizing the drawbacks when applying original dropout on DenseNet, we craft the design of dropout method from three aspects, the idea of which could also be applied on other CNN models.

@highlight

Application of different binary dropout structures and schedules with the specific aim to regularise the DenseNet architecture.

@highlight

Proposes a pre-dropout technique for densenet which implements the dropout before the non-linear activation function.