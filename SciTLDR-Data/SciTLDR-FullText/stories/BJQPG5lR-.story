A widely observed phenomenon in deep learning is the degradation problem: increasing the depth of a network leads to a decrease in performance on both test and training data.

Novel architectures such as ResNets and Highway networks have addressed this issue by introducing various flavors of skip-connections or gating mechanisms.

However, the degradation problem persists in the context of plain feed-forward networks.

In this work we propose a simple method to address this issue.

The proposed method poses the learning of weights in deep networks as a constrained optimization problem where the presence of skip-connections is penalized by Lagrange multipliers.

This allows for skip-connections to be introduced during the early stages of training and subsequently phased out in a principled manner.

We demonstrate the benefits of such an approach with experiments on MNIST, fashion-MNIST, CIFAR-10 and CIFAR-100 where the proposed method is shown to greatly decrease the degradation effect (compared to plain networks) and is often competitive with ResNets.

The representation view of deep learning suggests that neural networks learn an increasingly abstract representation of input data in a hierarchical fashion BID26 BID6 BID7 .

Such representations may then be exploited to perform various tasks such as image classification, machine translation and speech recognition.

A natural conclusion of the representation view is that deeper networks will learn more detailed and abstract representations as a result of their increased capacity.

However, in the case of feed-forward networks it has been observed that performance deteriorates beyond a certain depth, even when the network is applied to training data.

Recently, Residual Networks (ResNets; BID9 and Highway Networks BID22 have demonstrated that introducing various flavors of skip-connections or gating mechanisms makes it possible to train increasingly deep networks.

However, the aforementioned degradation problem persists in the case of plain deep networks (i.e., networks without skip-connections of some form).A widely held hypothesis explaining the success of ResNets is that the introduction of skipconnections serves to improve the conditioning of the optimization manifold as well as the statistical properties of gradients employed during training.

BID19 and BID21 show that the introduction of specially designed skip-connections serves to diagonalize the Fisher information matrix, thereby bringing standard gradient steps closer to the natural gradient.

More recently, BID0 demonstrated that the introduction of skip-connections helps retain the correlation structure across gradients.

This is contrary to the gradients of deep feed-forward networks, which resemble white noise.

More generally, the skip-connections are thought to reduce the effects of vanishing gradients by introducing a linear term BID10 .The goal of this work is to address the degradation issue in plain feed-forward networks by leveraging some of the desirable optimization properties of ResNets.

We approach the task of learning parameters for a deep network under the framework of constrained optimization.

This strategy allows us to introduce skip-connections penalized by Lagrange multipliers into the architecture of our network.

In our setting, skip-connections play an important role during the initial training of the network and are subsequently removed in a principled manner.

Throughout a series of experiments we demonstrate that such an approach leads to improvements in generalization error when compared to architectures without skip-connections and is competitive with ResNets in some cases.

The contributions of this work are as follows:• We propose alternative training strategy for plain feed-forward networks which reduces the degradation in performance as the depth of the network increases.

The proposed method introduces skip-connections which are penalized by Lagrange multipliers.

This allows for the presence of skip-connections to be iteratively phased out during training in a principled manner.

The proposed method is thereby able to enjoy the optimization benefits associated with skip-connections during the early stages of training.• A number of benchmark datasets are used to demonstrate the empirical capabilities of the proposed method.

In particular, the proposed method greatly reduces the degradation effect compared to plain networks and is on several occasions competitive with ResNets.

The hierarchical nature of many feed-forward networks is loosely inspired by the structure of the visual cortex where neurons in early layers capture simple features (e.g., edges) which are subsequently aggregated in deeper layers BID14 .

This interpretation of neural networks suggests that the depth of a network should be maximized, thereby allowing the network to learn more abstract (and hopefully useful) representations BID1 .

However, a widely reported phenomenon is that deeper networks are more difficult to train.

This is often termed the degradation effect in deep networks BID22 BID9 .

This effect has been partially attributed to optimization challenges such as vanishing and shattered gradients BID11 BID0 .In the past these challenges have been partially addressed via the use of supervised and unsupervised pre-training BID2 ) and more recently through careful parameter initialization BID5 BID8 and batch normalization BID15 .

In the past couple of years further improvements have been obtained via the introduction of skip-connections.

ResNets BID9 b) introduce residual blocks consisting of a residual function F together with a skip-connection.

Formally, the residual block is defined as: DISPLAYFORM0 where F l : R n → R n represents some combination of affine transformation, non-linearity and batch normalization parameterized by W l .

The matrix W l parameterizes a linear projection to ensure the dimensions are aligned 1 .

More generally, ResNets are closely related to Highway Networks BID22 where the output of each layer is defined as: DISPLAYFORM1 where · denotes element-wise multiplication.

In Highway Networks the output of each layer is determined by a gating function DISPLAYFORM2 inspired from LSTMs.

We note that both ResNets and Highway Networks were introduced with the explicit goal of training deeper networks.

Inspired by the success of the these methods, many variations have been proposed.

BID12 propose DenseNet, where skip-connections are passed from all previous activations.

BID13 propose to shorten networks during training by randomly dropping entire layers, leading to better gradient flow and information propagation, while using the full network at test time.

Recently, the goal of learning deep networks without skip-connections has begun to receive more attention.

BID25 propose a novel re-parameterization of weights in feedforward networks which they call the Dirac parameterization.

Instead of explicitly adding a skipconnection, they model the weights as a residual of the Dirac function, effectively moving the skipconnection inside the non-linearity.

In related work, BID0 propose to initialize weights in a CReLU activation function in order to preserve linearity during the initial phases of training.

This is achieved by initializing the weights in a mirrored block structure.

During training the weights are allowed to diverge, resulting in non-linear activations.

Finally, we note that while the aforementioned approaches have sought to train deeper networks via modifications to the network architecture (i.e., by adding skip-connections) success has also been obtained by modifying the non-linearities BID4 BID16 .

The goal of this work is to train deep feed-forward networks without suffering from the degradation problem described in previous sections.

To set notation, we denote x 0 as the input and x L as the output of a feed-forward network with L layers.

Given training data {y, x 0 } it is possible to learn parameters {W l } L l=1 by locally minimizing some objective function DISPLAYFORM0 First-order methods are typically employed due to the complexity of the objective function in equation (3).

However, directly minimizing the objective is not practical in the context of deep networks: beyond a certain depth performance quickly deteriorates on both test and training data.

Such a phenomenon does not occur in the presence of skip-connections.

Accordingly, we take inspiration from ResNets and propose to modify equation FORMULA0 in the following manner 2 : DISPLAYFORM1 where α l ∈ [0, 1] n determines the weighting given to the skip-connection.

More specifically, α l is a vector were the entry i dictates the presence and magnitude of a skip-connection for neuron i in layer l. Due to the variable nature of parameters α l in equation FORMULA4 , we refer to networks employing such residual blocks as Variable Activation Networks (VAN).The objective of the proposed method is to train a feed-forward network under the constraint that α l = 1 for all layers, l. When the constraint is satisfied all skip-connections are removed.

The advantage of such a strategy is that we only require α l = 1 at the end of training.

This allows us to initialize α l to some other value, thereby relaxing the optimization problem and obtaining the advantages associated with ResNets during the early stages of training.

In particular, whenever α l = 1 information is allowed to flow through the skip-connections, alleviating issues associated with shattered and vanishing gradients.

As a result of the equality constraint on α l , the proposed activation function effectively does not introduce any additional parameters.

All remaining weights can be trained by solving the following constrained optimization problem: DISPLAYFORM2 The associated Lagrangian takes the following simple form BID3 : DISPLAYFORM3 where each λ l ∈ R n are the Lagrange multipliers associated with the constraints on α l .

In practice, we iteratively update α l via stochastic gradients descent (SGD) steps of the form: DISPLAYFORM4 where η is the step-size parameter for SGD.

Throughout the experiments we will often take the non-linearity in F l to be ReLU.

Although not strictly required, we clip the values α l to ensure they remain in the interval [0, 1] n .From equation FORMULA6 , we have that the gradients with respect to Lagrange multipliers are of the form: DISPLAYFORM5 We note that since we require α l ∈ [0, 1] n , the values of λ l are monotonically decreasing.

As the value of Lagrange multiplier decreases, this in turn pushes α l towards 1 in equation FORMULA7 .

We set the step-size for the Lagrange multipliers, η , to be a fraction of η.

The motivation behind such a choice is to allow the network to adjust as we enforce the constraint on α l .

The purpose of the experiments presented in this section is to demonstrate that the proposed method serves to effectively alleviate the degradation problem in deep networks.

We first demonstrate the capabilities of the proposed method using a simple, non-convolutional architecture on the MNIST and Fashion-MNIST datasets BID23 in Section 4.1.

More extensive comparisons are then considered on the CIFAR datasets BID17 in Section 4.2.

Networks of varying depths were trained on both MNIST and Fashion-MNIST datasets.

Following BID22 the networks employed in this section were thin, with each layer containing 50 hidden units.

In all networks the first layer was a fully connected plain layer followed by l layers or residual blocks (depending on the architecture) and a final softmax layer.

The proposed method is benchmarked against several popular architectures such as ResNets and Highway Networks as well as the recently proposed DiracNets BID25 .

Plain networks without skipconnections are also considered.

Finally, we also considered VAN network where the constraint α l = 1 was not enforced.

This corresponds to the case where λ l = 0 for all l. This comparison is included in order to study the capacity and flexibility of VAN networks without the need to satisfy the constraint to remove skip-connections.

For clarity, refer to such networks as VAN (λ = 0) networks.

For all architectures the ReLU activation function was employed together with batch-normalization.

In the case of ResNets and VAN, the residual function consisted of batch-normalization followed by ReLU and a linear projection.

The depth of the network varied from l = 1 to l = 30 hidden layers.

All networks were trained using SGD with momentum.

The learning rate is fixed at η = 0.001 and the momentum parameter at 0.9.

Training consisted of 50 epochs with a batch-size of 128.

In the case of VAN networks the α l values were initialized to 0 for all layers.

As such, during the initial stages of training VAN networks where equivalent to ResNets.

The step-size parameter for Lagrange multipliers, η , was set to be one half of the SGD step-size, η.

Finally, all Lagrange multipliers, λ l , are initialized to -1.

The results are shown in FIG0 where the test accuracy is shown as a function of the network depth for both the MNIST and Fashion-MNIST datasets.

In both cases we see clear evidence of the degradation effect: the performance of plain networks deteriorates significantly once the network depth exceeds some critical value (approximately 10 layers).

As would be expected, this is not the case for ResNets, Highway Networks and DiracNets as such architectures have been explicitly designed to avoid this behavior.

We note that VAN networks do not suffer such a pronounced degradation as the depth increases.

This provides evidence that the gradual removal of skip-connections via Lagrange multipliers leads to improved generalization performance compared to plain networks.

Finally, we note that VAN networks obtain competitive results across all depths.

Crucially, we note that VAN networks outperform plain networks across all depths, suggesting that the introduction of variable skip-connections may lead to convergence at local optima with better generalization performance.

Finally, we note that VAN (λ = 0) networks, where no constraint is placed on skip-connections, obtain competitive results across all depths.

Mean average test accuracy over 10 independent training sessions is shown.

We note that with the exception of plain networks, the performance of all remaining architectures is stable as the number of layers increases.

As a more challenging benchmark we consider the CIFAR-10 and CIFAR-100 datasets.

These consist of 60000 32×32 pixel color images with 10 and 100 classes respectively.

The datasets are divided into 50000 training images and 10000 test images.

We follow BID9 and train deep convolutional networks consisting of four blocks each consisting of n residual layers.

The residual function is of the form conv-BN-ReLU-conv-BN-ReLU.

This corresponds to the pre-activation function BID10 .

The convolutional layers consist of 3 × 3 filters with downsampling at the beginning of blocks 2, 3 and 4.

The network ends with a fully connected softmax layer, resulting in a depth of 8n + 2.

The architecture is described in TAB0 .Networks were trained using SGD with momentum over 165 epochs.

The learning rate was set to η = 0.1 and divided by 10 at the 82nd and 125th epoch.

The momentum parameter was set to 0.9.

Networks were trained using mini-batches of size 128.

Data augmentation followed BID18 : this involved random cropping and horizontal flips.

Weights were initialized following BID8 .

As in Section 4.1, we initialize α l = 0 for all layers.

Furthermore, we set the step-size parameter for the Lagrange multipliers, η , to be one tenth of η and all Lagrange multipliers, λ l , are initialized to -1.

On CIFAR-10 we ran experiments with n ∈ {1, 2, 3, 4, 5, 6, 8, 10} yielding networks with depths ranging from 10 to 82.

For CIFAR-100 experiments were run with n ∈ {1, 2, 3, 4}. Figure 2 : Left: Results on CIFAR-10 dataset are shown as the depth of networks increase.

We note that the performance of both VAN and plain networks deteriorates as the depth increases, but the effect is far less pronounced for VAN networks.

Right: Training and test error curves are shown for networks with 26 layers.

We also plot the mean α residuals: DISPLAYFORM0 (1 − α l ) 2 on the right axis.

Results for experiments on CIFAR-10 are shown in Figure 2 .

The left panel shows the mean test accuracy over five independent training sessions for ResNets, VAN, VAN (λ = 0) and plain networks.

While plain networks provide competitive results for networks with fewer than 30 layers, their performance quickly deteriorates thereafter.

We note that a similar phenomenon is observed in VAN networks but the effect is not as dramatic.

In particular, the performance of VANs is similar to ResNets for networks with up to 40 layers.

Beyond this depth, ResNets outperform VAN by an increasing margin.

This holds true for both VAN and VAN (λ = 0) networks, however, the difference is reduced in magnitude in the case of VAN (λ = 0) networks.

These results are in line with BID10 , who argue that scalar modulated skip-connections (as is the case in VANs where the scalar is 1 − α l ) will either vanish or explode in very deep networks whenever the scalar is not the identity.

The right panel of Figure 2 shows the training and test error for a 26 layer network.

We note that throughout all iterations, both the test and train accuracy of the VAN network dominates that of the plain network.

The thick gold line indicates the mean residuals of the α l parameters across all layers.

This is defined as DISPLAYFORM0 and is a measure of the extent to which skip-connections are present in the network.

Recall that if all α l values are set to one then all skip-connections are removed (see equation FORMULA4 ).

From Figure 2 , it follows that skip-connections are fully removed from the VAN network at approximately the 120 th iteration.

More detailed traces of Lagrange multipliers and α l are provided in Appendix B.A comparison of the performance of VAN networks in provided in TAB1 .

We note that while VAN networks do not outperform ResNets, they do outperform other alternatives such as Highway networks and FitNets BID20 when networks of similar depths considered.

However, it is important to note that both Highway networks and FitNets did not employ batch-normalization, which is a strong regularizer.

In the case of both VAN and VAN (λ = 0) networks, the best performance is obtained with networks of 26 layers while ResNets continue to improve their performance as depth increases.

Finally, current state-of-the-art performance, obtained by Wide ResNets BID24 and DenseNet Huang et al. (2016a) , are also provided in TAB1 Figure 3 provides results on the CIFAR-100 dataset.

This dataset is considerably more challenging as it consists of a larger number of classes as well as fewer examples per class.

As in the case of CIFAR-10, we observe a fall in the performance of both VAN and plain networks beyond a certain depth; in this case approximately 20 layers for plain networks and 30 layers for VANs.

Despite this drop in performance, TAB1 indicates that the performance of VAN networks with both 18 and 26 layers are competitive with many alternatives proposed in the literature.

Furthermore, we note that the performance of VAN (λ = 0) networks is competitive with ResNets in the context of the CIFAR-100 dataset.

We note that the performance of both VAN and plain networks deteriorates as the depth increases, but the effect is far less pronounced for plain networks.

Right: Training and test error curves are shown for VAN and plain networks with 18 layers.

The mean α residuals, DISPLAYFORM1 2 , are shown in gold along the right axis.

Training curves are shown on the right hand side of FIG1 .

As in the equivalent plot for CIFAR-10, the introduction and subsequent removal of skip-connections during training leads to improvements in generalization error.

This manuscript presents a simple method for training deep feed-forward networks which greatly reduces the degradation problem.

In the past, the degradation issue has been successfully addressed via the introduction of skip-connections.

As such, the goal of this work is to propose a new training regime which retains the optimization benefits associated with ResNets while ultimately phasing out skip-connections.

This is achieved by posing network training as a constrained optimization problem where skip-connections are introduced during the early stages of training and subsequently phased out in a principled manner using Lagrange multipliers.

Throughout a series of experiments we demonstrate that the performance of VAN networks is stable, displaying a far smaller drop in performance as depth increases and thereby largely mitigating the degradation problem.

The original formulation for the VAN residual block was as follows: DISPLAYFORM0 We thank an anonymous reviewer for suggesting that such a formulation may be detrimental to the performance of very deep VAN networks.

The reason for this is that scaling constant within each block is always less than one, implying that the contributions of lower layers vanish exponentially as the depth increases.

This argument is also provided in BID10 who perform similar experiments with ResNets.

In order to validate this hypothesis, we compare the performance of VAN networks employing the residual block described in equation FORMULA4 and the residual block described in equation FORMULA12 .

The results, shown in FIG2 , provide evidence in favor of the proposed hypothesis.

While both formulations for VAN networks obtain similar performances for shallow networks, as the depth of the network increases there is a more pronounced drop in the performance of VAN networks which employ residual blocks described in equation FORMULA12 .In a further experiment, we also studied the performance of ResNets with the following residual block: DISPLAYFORM1 The results in FIG2 demonstrate that ResNets which employ the residual blocks defined in equation (10) show a clear deterioration in performance as the depth of the network increases.

Such a degradation in performance is not present when standard ResNets are employed.

We note that the use of residual blocks with non-identity scaling coefficients leads to a larger drop in performance as the network depth increases.

This drop is attributed to vanishing contributions from lower blocks (as all scalings are less than one).

In this section we provide addition figures demonstrating the evolution of Lagrange multipliers, λ l throughout training.

We note that the updates to Lagrange multipliers are directly modulated by the current value of each α l (see equation FORMULA8 ).

As such, we also visualize the mean residuals of the α l parameters across all layers.

This is defined as DISPLAYFORM0 (1 − α l ) 2 and is a measure of the extent to which skip-connections are present in the network.

Once all skip-connections have been removed, this residual will be zero and the values of Lagrange multipliers will no longer change.

This is precisely what we find in FIG3 .

The left panel plots the mean value of Lagrange multipliers across all layers, while the right panel shows the mean residual of α l .

We observe that for networks of different depths, once the constraint to remove skip-connections is satisfied, the value of Lagrange multipliers remains constant.

This occurs at different times; sooner for more shallow networks whilst later on for deeper networks.

FORMULA8 ).

The right panel shows the mean α l residual.

This residual directly modulates the magnitude of changes in Lagrange multipliers.

@highlight

Phasing out skip-connections in a principled manner avoids degradation in deep feed-forward networks.

@highlight

The authors present a new training strategy, VAN, for training very deep feed-forward networks without skip connections

@highlight

The paper introduces an architecture that linearly interpolates between ResNets and vanilla deep nets without skip connections.