Residual networks (Resnets) have become a prominent architecture in deep learning.

However, a comprehensive understanding of Resnets is still a topic of ongoing research.

A recent view argues that Resnets perform iterative refinement of features.

We attempt to further expose properties of this aspect.

To this end, we study Resnets both analytically and empirically.

We formalize the notion of iterative refinement in Resnets by showing that residual architectures naturally encourage features to move along the negative gradient of loss during the feedforward phase.

In addition, our empirical analysis suggests that Resnets are able to perform both representation learning and iterative refinement.

In general, a Resnet block tends to concentrate representation learning behavior in the first few layers while higher layers perform iterative refinement of features.

Finally we observe that sharing residual layers naively leads to representation explosion and hurts generalization performance, and show that simple existing strategies can help alleviating this problem.

Traditionally, deep neural network architectures (e.g. VGG Simonyan & Zisserman (2014) , AlexNet Krizhevsky et al. (2012) , etc.) have been compositional in nature, meaning a hidden layer applies an affine transformation followed by non-linearity, with a different transformation at each layer.

However, a major problem with deep architectures has been that of vanishing and exploding gradients.

To address this problem, solutions like better activations (ReLU Nair & Hinton (2010) ), weight initialization methods Glorot & Bengio (2010) ; He et al. (2015) and normalization methods Ioffe & Szegedy (2015) ; BID0 have been proposed.

Nonetheless, training compositional networks deeper than 15 − 20 layers remains a challenging task.

Recently, residual networks (Resnets He et al. (2016a) ) were introduced to tackle these issues and are considered a breakthrough in deep learning because of their ability to learn very deep networks and achieve state-of-the-art performance.

Besides this, performance of Resnets are generally found to remain largely unaffected by removing individual residual blocks or shuffling adjacent blocks Veit et al. (2016) .

These attributes of Resnets stem from the fact that residual blocks transform representations additively instead of compositionally (like traditional deep networks).

This additive framework along with the aforementioned attributes has given rise to two school of thoughts about Resnets-the ensemble view where they are thought to learn an exponential ensemble of shallower models Veit et al. (2016) , and the unrolled iterative estimation view Liao & Poggio (2016) ; Greff et al. (2016) , where Resnet layers are thought to iteratively refine representations instead of learning new ones.

While the success of Resnets may be attributed partly to both these views, our work takes steps towards achieving a deeper understanding of Resnets in terms of its iterative feature refinement perspective.

Our contributions are as follows:1.

We study Resnets analytically and provide a formal view of iterative feature refinement using Taylor's expansion, showing that for any loss function, a residual block naturally encourages representations to move along the negative gradient of the loss with respect to hidden representations.

Each residual block is therefore encouraged to take a gradient step in order to minimize the loss in the hidden representation space.

We empirically confirm this by measuring the cosine between the output of a residual block and the gradient of loss with respect to the hidden representations prior to the application of the residual block.2.

We empirically observe that Resnet blocks can perform both hierarchical representation learning (where each block discovers a different representation) and iterative feature refinement (where each block improves slightly but keeps the semantics of the representation of the previous layer).

Specifically in Resnets, lower residual blocks learn to perform representation learning, meaning that they change representations significantly and removing these blocks can sometimes drastically hurt prediction performance.

The higher blocks on the other hand essentially learn to perform iterative inference-minimizing the loss function by moving the hidden representation along the negative gradient direction.

In the presence of shortcut connections 1 , representation learning is dominantly performed by the shortcut connection layer and most of residual blocks tend to perform iterative feature refinement.3.

The iterative refinement view suggests that deep networks can potentially leverage intensive parameter sharing for the layer performing iterative inference.

But sharing large number of residual blocks without loss of performance has not been successfully achieved yet.

Towards this end we study two ways of reusing residual blocks: 1.

Sharing residual blocks during training; 2.

Unrolling a residual block for more steps that it was trained to unroll.

We find that training Resnet with naively shared blocks leads to bad performance.

We expose reasons for this failure and investigate a preliminary fix for this problem.

Recently, several papers have investigated the behavior of Resnets (He et al., 2016a) .

In (Veit et al., 2016; Littwin & Wolf, 2016) , authors argue that Resnets are an ensemble of relatively shallow networks.

This is based on the unraveled view of Resnets where there exist an exponential number of paths between the input and prediction layer.

Further, observations that shuffling and dropping of residual blocks do not affect performance significantly also support this claim.

Other works discuss the possibility that residual networks are approximating recurrent networks (Liao & Poggio, 2016; Greff et al., 2016) .

This view is in part supported by the observation that the mathematical formulation of Resnets bares similarity to LSTM (Hochreiter & Schmidhuber, 1997) , and that successive layers cooperate and preserve the feature identity.

Resnets have also been studied from the perspective of boosting theory Huang et al. (2017) .

In this work the authors propose to learn Resnets in a layerwise manner using a local classifier.

Our work has critical differences compared with the aforementioned studies.

Most importantly we focus on a precise definition of iterative inference.

In particular, we show that a residual block approximate a gradient descent step in the activation space.

Our work can also be seen as relating the gap between the boosting and iterative inference interpretations since having a residual block whose output is aligned with negative gradient of loss is similar to how gradient boosting models work.

Humans frequently perform predictions with iterative refinement based on the level of difficulty of the task at hand.

A leading hypothesis regarding the nature of information processing that happens in the visual cortex is that it performs fast feedforward inference (Thorpe et al., 1996) for easy stimuli or when quick response time is needed, and performs iterative refinement of prediction for complex stimuli (Vanmarcke et al., 2016) .

The latter is thought to be done by lateral connections within individual layers in the brain that iteratively act upon the current state of the layer to update it.

This mechanism allows the brain to make fine grained predictions on complex tasks.

A characteristic attribute of this mechanism is the recursive application of the lateral connections which can be thought of as shared weights in a recurrent model.

The above views suggest that it is desirable to have deep network models that perform parameter sharing in order to make the iterative inference view complete.

Our goal in this section is to formalize the notion of iterative inference in Resnets.

We study the properties of representations that residual blocks tend to learn, as a result of being additive in nature, in contrast to traditional compositional networks.

Specifically, we consider Resnet architectures (see FIG0 ) where the first hidden layer is a convolution layer, which is followed by L residual blocks which may or may not have shortcut connections in between residual blocks.

A residual block applied on a representation h i transforms the representation as, DISPLAYFORM0 Consider L such residual blocks stacked on top of each other followed by a loss function.

Then, we can Taylor expand any given loss function L recursively as, DISPLAYFORM1 Here we have Taylor expanded the loss function around h L−1 .

We can similarly expand the loss function recursively around h L−2 and so on until h i and get, DISPLAYFORM2 Notice we have explicitly only written the first order terms of each expansion.

The rest of the terms are absorbed in the higher order terms O(.).

Further, the first order term is a good approximation when the magnitude of F j is small enough.

In other cases, the higher order terms come into effect as well.

Thus in part, the loss equivalently minimizes the dot product between F (h i ) and DISPLAYFORM3 ∂hi , which can be achieved by making F (h i ) point in the opposite half space to that of ∂L(hi) ∂hi .

In other words, h i + F (h i ) approximately moves h i in the same half space as that of − ∂L(hi) ∂hi .

The overall training criteria can then be seen as approximately minimizing the dot product between these 2 terms along a path in the h space between h i and h L such that loss gradually reduces as we take steps from h i to h L .

The above analysis is justified in practice, as Resnets' top layers output F j has small magnitude (Greff et al., 2016) , which we also report in Fig. 2 .Given our analysis we formalize iterative inference in Resnets as moving down the energy (loss) surface.

It is also worth noting the resemblance of the function of a residual block to stochastic gradient descent.

We make a more formal argument in the appendix.

introduce for the purpose of our analysis (described below).

Our main goal is to validate that residual networks perform iterative refinement as discussed above, showing its various consequences.

Specifically, we set out to empirically answer the following questions:

• Do residual blocks in Resnets behave similarly to each other or is there a distinction between blocks that perform iterative refinement vs. representation learning?

• Is the cosine between ∂L(hi) ∂hi and F i (h i ) negative in residual networks?• What kind of samples do residual blocks target?• What happens when layers are shared in Resnets?Resnet architectures: We use the following four architectures for our analysis:1.

Original Resnet-110 architecture: This is the same architecture as used in He et al. (2016b) starting with a 3 × 3 convolution layer with 16 filters followed by 54 residual blocks in three different stages (of 18 blocks each with 16, 32 and 64 filters respectively) each separated by a shortcut connections (1 × 1 convolution layers that allow change in the hidden space dimensionality) inserted after the 18 th and 36 th residual blocks such that the 3 stages have hidden space of height-width 32 × 32, 16 × 16 and 8 × 8.

The model has a total of 1, 742, 762 parameters.

This architecture starts with a 3 × 3 convolution layer with 100 filters.

This is followed by 10 residual blocks such that all hidden representations have the same height and width of 32 × 32 and 100 filters are used in all the convolution layers in residual blocks as well.3.

Avg-pooling Resnet: This architecture repeats the residual blocks of the single representation Resnet (described above) three times such that there is a 2 × 2 average pooling layer after each set of 10 residual blocks that reduces the height and width after each stage by half.

Also, in contrast to single representation architecture, it uses 150 filters in all convolution layers.

This is followed by the classification block as in the single representation Resnet.

It has 12, 201, 310 parameters.

We call this architecture the avg-pooling architecture.

We also ran experiments with max pooling instead of average pooling but do not report results because they were similar except that max pool acts more non-linearly compared with average pooling, and hence the metrics from max pooling are more similar to those from original Resnet.4.

Wide Resnet: This architecture starts with a 3 × 3 convolution layer followed by 3 stages of four residual blocks with 160, 320 and 640 number of filters respectively, and 3 × 3 kernel size in all convolution layers.

This model has a total of 45,732,842 parameters.

For all architectures, we use He-normal weight initialization as suggested in He et al. (2015) , and biases are initialized to 0.For residual blocks, we use BatchNorm→ReLU→Conv→BatchNorm→ReLU→Conv as suggested in He et al. (2016b) .The classifier is composed of the following elements: BatchNorm→ReLU→AveragePool (8,8)

In this experiment we directly validate our theoretical prediction about Resnets minimizing the dot product between gradient of loss and block output.

To this end compute the cosine loss DISPLAYFORM0 .

A negative cosine loss and small F i (.) together suggest that F i (.) is refining features by moving them in the half space of − ∂L(hi) ∂hi , thus reducing the loss value for the corresponding data samples.

Figure 4 shows the cosine loss for CIFAR-10 on train and validation sets.

These figures show that cosine loss is consistently negative for all residual blocks but especially for the higher residual blocks.

Also, notice for deeper architectures (original Resnet and pooling Resnet), the higher blocks achieve more negative cosine loss and are thus more iterative in nature.

Further, since the higher residual blocks make smaller changes to representation (figure 2), the first order Taylor's term becomes dominant and hence these blocks effectively move samples in the half space of the negative cosine loss thus reducing loss value of prediction.

This result formalizes the sense in which residual blocks perform iterative refinement of features-move representations in the half space of − ∂L(hi) ∂hi .

In this section, we are interested in investigating the behavior of residual layers in terms of representation learning vs. refinement of features.

To this end, we perform the following experiments.

2 ratio DISPLAYFORM0 .

For every such block in a Resnet, we measure the 2 ratio of F i (h i ) 2 / h i 2 averaged across samples.

This ratio directly shows how significantly F i (.) changes the representation h i ; a large change can be argued to be a necessary condition for layer to perform representation learning.

Figure 2 shows the 2 ratio for CIFAR-10 on train and validation sets.

For single representationResnet and pooling Resnet, the first few residual blocks (especially the first residual block) changes representations significantly (up to twice the norm of the original representation), while the rest of the higher blocks are relatively much less significant and this effect is monotonic as we go to higher blocks.

However this effect is not as drastic in the original Resnet and wide Resnet architectures which have two 1 × 1 (shortcut) convolution layers, thus adding up to a total of 3 convolution layers in the main path of the residual network (notice there exists only one convolution layer in the main path for the other two architectures).

This suggests that residual blocks in general tend to learn to refine features but in the case when the network lacks enough compositional layers in the main path, lower residual blocks are forced to change representations significantly, as a proxy for the absence of compositional layers.

Additionally, small 2 ratio justifies first order approximation used to derive our main result in Sec. 3.2.

Effect of dropping residual layer on accuracy: We drop individual residual blocks from trained Resnets and make predictions using the rest of network on validation set.

This analysis shows the significance of individual residual blocks towards the final accuracy that is achieved using all the residual blocks.

Note, dropping individual residual blocks is possible because adjacent blocks operate in the same feature space.

Figure 3 shows the result of dropping individual residual blocks.

As one would expect given above analysis, dropping the first few residual layers (especially the first) for single representation Resnet and pooling Resnet leads to catastrophic performance drop while dropping most of the higher residual layers have minimal effect on performance.

On the other hand, performance drops are not drastic for the original Resnet and wide Resnet architecture, which is in agreement with the observations in 2 ratio experiments above.

In another set of experiments, we measure validation accuracy after individual residual block during the training process.

This set of experiments is achieved by plugging the classifier right after each residual block in the last stage of hidden representation (i.e., after the last shortcut connection, if any).

This is shown in figure 5 .

The figures show that accuracy increases very gradually when adding more residual blocks in the last stage of all architectures.

In this section we investigate which samples get correctly classified after the application of a residual block.

Individual residual blocks in general lead to small improvements in performance.

Intuitively, since these layers move representations minimally (as shown by previous analysis), the samples that lead to these minor accuracy jump should be near the decision boundary but getting misclassified by a slight margin.

To confirm this intuition, we focus on borderline examples, defined as examples that require less than 10% probability change to flip prediction to, or from the correct class.

We measure loss, accuracy and entropy over borderline examples over last 5 blocks of the network using the network final classifier.

Experiment is performed on CIFAR-10 using Resnet-110 architecture.

Fig 6 shows evolution of loss and accuracy on three groups of examples: borderline examples, already correctly classified and the whole dataset.

While overall accuracy and loss remains similar across the top residual blocks, we observe that a significant chunk of borderline examples gets corrected by the immediate next residual block.

This exposes the qualitative nature of examples that these feature refinement layers focus on, which is further reinforced by the fact that entropy decreases for all considered subsets.

We also note that while train loss drops uniformly across layers, test sets loss increases after last block.

Correcting this phenomenon could lead to improved generalization in Resnets, which we leave for future work.

A fundamental requirement for a procedure to be truly iterative is to apply the same function.

In this section we explore what happens when we unroll the last block of a trained residual network for more steps than it was trained for.

Our main goal is to investigate if iterative inference generalizes to more steps than it was trained on.

We focus on the same model as discussed in previous section, Resnet-110, and unroll the last residual block for 20 extra steps.

Naively unrolling the network leads to activation explosion (we observe similar behavior in Sec. 4.5).

To control for that effect, we added a scaling factor on the output of the last residual blocks.

We hypothesize that controlling the scale limits the drift of the activation through the unrolled layer, i.e. they remains in a given neighbourhood on which the network is well behaved.

Similarly to Sec. 4.3 we track evolution of We first investigate how unrolling blocks impact loss and accuracy.

Loss on train set improved uniformly from 0.0012 to 0.001, while it increased on test set.

There are on average 51 borderline examples in test set 2 , on which performance is improved from 43% to 53%, which yields slight improvement in accuracy on test set.

Next we shift our attention to cosine loss.

We observe that cosine loss remains negative on the first two steps without rescaling, and all steps after scaling.

Figure 7 shows evolution of loss and accuracy on the three groups of examples: borderline examples, already correctly classified and the whole dataset.

Cosine loss and 2 ratio for each block are reported in Appendix E.To summarize, unrolling residual network to more steps than it was trained on improves both loss on train set, and maintains (in given neighbourhood) negative cosine loss on both train and test set.

Our results suggest that top residual blocks should be shareable, because they perform similar iterative refinement.

We consider a shared version of Resnet-110 model, where in each stage we share all the residual blocks from the 5 th block.

All shared Resnets in this section have therefore a similar number of parameters as Resnet-38.

Contrary to (Liao & Poggio, 2016) we observe that naively sharing the higher (iterative refinement) residual blocks of a Resnets in general leads to bad performance 3 (especially for deeper Resnets).First, we compare the unshared and shared version of Resnet-110.

The shared version uses approximately 3 times less parameters.

In Fig. 8 , we report the train and validation performances of the Resnet-110.

We observe that naively sharing parameters of the top residual blocks leads both to overfitting (given similar training accuracy, the shared Resnet-110 has significantly lower validation performances) and underfitting (worse training accuracy than Resnet-110).

We also compared our shared model with a Resnet-38 that has a similar number of parameters and observe worse validation performances, while achieving similar training accuracy.

We notice that sharing layers make the layer activations explode during the forward propagation at initialization due to the repeated application of the same operation (Fig 8, right) .

Consequently, the norm of the gradients also explodes at initialization (Fig. 8, center) .To address this issue we introduce a variant of recurrent batch normalization (Cooijmans et al., 2016) , which proposes to initialize γ to 0.1 and unshare statistics for every step.

On top of this strategy, we also unshare γ and β parameters.

Tab.

1 shows that using our strategy alleviates explosion problem and leads to small improvement over baseline with similar number of parameters.

We also perform an ablation to study, see Figure.

9 (left), which show that all additions to naive strategy are necessary and drastically reduce the initial activation explosion.

Finally, we observe a similar trend for cosine loss, intermediate accuracy, and 2 ratio for the shared Resnet as for the unshared Resnet discussed in the previous Sections.

Full results are reported in Appendix D.Unshared Batch Normalization strategy therefore mitigates this exploding activation problem.

This problem, leading to exploding gradient in our case, appears frequently in recurrent neural network.

This suggests that future unrolled Resnets should use insights from research on recurrent networks optimization, including careful initialization (Henaff et al., 2016) and parametrization changes (Hochreiter & Schmidhuber, 1997

Our main contribution is formalizing the view of iterative refinement in Resnets and showing analytically that residual blocks naturally encourage representations to move in the half space of negative loss gradient, thus implementing a gradient descent in the activation space (each block reduces loss and improves accuracy).

We validate theory experimentally on a wide range of Resnet architectures.

We further explored two forms of sharing blocks in Resnet.

We show that Resnet can be unrolled to more steps than it was trained on.

Next, we found that counterintuitively training residual blocks with shared blocks leads to overfitting.

While we propose a variant of batch normalization to mitigate it, we leave further investigation of this phenomena for future work.

We hope that our developed formal view, and practical results, will aid analysis of other models employing iterative inference and residual connections.

∂ho , then it is equivalent to updating the parameters of the convolution layer using a gradient update step.

To see this, consider the change in h o from updating parameters using gradient descent with step size η.

This is given by, DISPLAYFORM0 Thus, moving h o in the half space of − ∂L ∂ho has the same effect as that achieved by updating the parameters W, b using gradient descent.

Although we found this insight interesting, we don't build upon it in this paper.

We leave this as a future work.

Here we report the experiments as done in sections 4.2 and 4.1, for CIFAR-100 dataset.

The plots are shown in figures 10, 11 and 12.

The conclusions are same as reported in the main text for CIFAR-10.

Here we plot the accuracy, cosine loss and 2 ratio metrics corresponding to each individual residual block on validation during the training process for CIFAR-10 (figures 13, 14, 5) and 16, 17) .

These plots are recorded only for the residual blocks in the last space for each architecture (this is because otherwise the dimensions of the output of the residual block and the classifier will not match).

In the case of cosine loss after individual residual block, this set of experiments is achieved by plugging the classifier right after each hidden representation and measuring the cosine between the gradient w.r.t.

hidden representation and the corresponding residual block's output.

We find that the accuracy after individual residual blocks increases gradually as we move from from lower to higher residua blocks.

Cosine loss on the other hand consistently remains negative for all architectures.

Finally 2 ratio tends to increase for residual blocks as training progresses.

In this section we extend results from Sec. 4.5.

We report cosine loss, intermediate accuracy, and 2 ratio for naively shared Resnet in FIG0 , and with unshared batch normalization in Fig. ? ?.

In this section we report additional results for unrolling residual network.

Figure 20 shows evolution of cosine loss an 2 ratio for Resnet-110 with unrolled last block for 20 additional steps.

@highlight

Residual connections really perform iterative inference