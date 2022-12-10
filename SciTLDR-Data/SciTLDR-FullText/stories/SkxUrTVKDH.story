Over-parameterization is ubiquitous nowadays in training neural networks to benefit both optimization in seeking global optima and generalization in reducing prediction error.

However, compressive networks are desired in many real world applications and direct training of small networks may be trapped in local optima.

In this paper, instead of pruning or distilling over-parameterized models to compressive ones, we propose a new approach based on \emph{differential inclusions of inverse scale spaces}, that generates a family of models from simple to complex ones by coupling gradient descent and mirror descent to explore model structural sparsity.

It has a simple discretization, called the Split Linearized Bregman Iteration (SplitLBI), whose global convergence analysis in deep learning is established that from any initializations, algorithmic iterations converge to a critical point of empirical risks.

Experimental evidence shows that\ SplitLBI may achieve state-of-the-art performance in large scale training on ImageNet-2012 dataset etc., while with \emph{early stopping} it unveils effective subnet architecture with comparable test accuracies to dense models after retraining instead of pruning well-trained ones.

The expressive power of deep neural networks comes from the millions of parameters, which are optimized by Stochastic Gradient Descent (SGD) (Bottou, 2010) and variants like Adam (Kingma & Ba, 2015) .

Remarkably, model over-parameterization helps both optimization and generalization.

For optimization, over-parameterization may simplify the landscape of empirical risks toward locating global optima efficiently by gradient descent method (Mei et al., 2018; Venturi et al., 2018; Allen-Zhu et al., 2018; Du et al., 2018) .

On the other hand, over-parameterization does not necessarily result in a bad generalization or overfitting (Zhang et al., 2017) , especially when some weight-size dependent complexities are controlled (Bartlett, 1997; Bartlett et al., 2017; Golowich et al., 2018; Neyshabur et al., 2019) .

However, compressive networks are desired in many real world applications, e.g. robotics, selfdriving cars, and augmented reality.

Despite that 1 regularization has been applied to deep learning to enforce the sparsity on weights toward compact, memory efficient networks, it sacrifices some prediction performance (Collins & Kohli, 2014) .

This is because that the weights learned in neural networks are highly correlated, and 1 regularization on such weights violates the incoherence or irrepresentable conditions needed for sparse model selection (Donoho & Huo, 2001; Tropp, 2004; Zhao & Yu, 2006) , leading to spurious selections with poor generalization.

On the other hand, 2 regularization is often utilized for correlated weights as some low-pass filtering, sometimes in the form of weight decay (Loshchilov & Hutter, 2019) or early stopping (Yao et al., 2007; Wei et al., 2017) .

Furthermore, group sparsity regularization (Yuan & Lin, 2006) has also been applied to neural networks, such as finding optimal number of neuron groups (Alvarez & Salzmann, 2016) and exerting good data locality with structured sparsity (Wen et al., 2016; Yoon & Hwang, 2017 ).

Yet, without the aid of over-parameterization, directly training a compressive model architecture may meet the obstacle of being trapped in local optima in contemporary experience.

Alternatively, researchers in practice typically start from training a big model using common task datasets like ImageNet, and then prune or distill such big models to small ones without sacrificing too much of the performance (Jaderberg et al., 2014; Han et al., 2015; Zhu et al., 2017; Zhou et al., 2017; Zhang et al., 2016; Li et al., 2017; Abbasi-Asl & Yu, 2017; Yang et al., 2018; Arora et al., 2018) .

In particular, a recent study (Frankle & Carbin, 2019) created the lottery ticket hypothesis based on empirical observations: "dense, randomly-initialized, feed-forward networks contain subnetworks (winning tickets) that -when trained in isolation -reach test accuracy comparable to the original network in a similar number of iterations".

How to effectively reduce an over-parameterized model thus becomes the key to compressive deep learning.

Yet, Liu et al. (2019) raised a question, is it necessary to fully train a dense, over-parameterized model before finding important structural sparsity?

In this paper, we provide a novel answer by exploiting a dynamic approach to deep learning with structural sparsity.

We are able to establish a family of neural networks, from simple to complex, by following regularization paths as solutions of differential inclusions of inverse scale spaces.

Our key idea is to design some dynamics that simultaneously exploit over-parameterized models and structural sparsity.

To achieve this goal, the original network parameters are lifted to a coupled pair, with one weight set W of parameters following the standard gradient descend to explore the over-parameterized model space, while the other set of parameters learning structure sparsity in an inverse scale space, i.e., structural sparsity set Γ. The large-scale important parameters are learned at a fast speed while the small unimportant ones are learned at a slow speed.

The two sets of parameters are coupled in an 2 regularization.

The dynamics enjoys a simple discretization, i.e. the Split Linearized Bregman Iteration (SplitLBI), with provable global convergence guarantee shown in this paper.

Here, SplitLBI is a natural extension of SGD with structural sparsity exploration: SplitLBI reduces to the standard gradient descent method when the coupling regularization is weak, while it leads to a sparse mirror descent when the coupling is strong.

Critically, SplitLBI enjoys a nice property that important subnet architecture can be rapidly learned via the structural sparsity parameter Γ following the iterative regularization path, without fully training a dense network first.

Particularly, the support set of structural sparsity parameter Γ learned in the early stage of this inverse scale space discloses important sparse subnet architectures.

Such architectures can be fine-tuned or retrained to achieve comparable test accuracy as the dense, over-parameterized networks.

As a result, the structural sparsity parameter Γ may enable us to rapidly find "winning tickets" in early training epochs for the "lottery" of identifying successful subnetworks that bear comparable test accuracy to the dense ones.

This point is empirically validated in our experiments.

Historically, the Linearized Bregman Iteration (LBI) was firstly proposed in applied mathematics as iterative regularization paths for image reconstruction and compressed sensing (Osher et al., 2005; Yin et al., 2008) , later applied to logistic regression (Shi et al., 2013) .

The convergence analysis was given for convex problems (Yin et al., 2008; Cai et al., 2009) , yet remaining open for non-convex problems met in deep learning.

Osher et al. (2016) established statistical model selection consistency for high dimensional linear regression under the same irrepresentable condition as Lasso, later extended to generalized linear models (Huang & Yao, 2018) .

To relax such conditions, SplitLBI was proposed by Huang et al. (2016) to learn structural sparsity in linear models under weaker conditions than generalized Lasso, that was successfully applied in medical image analysis (Sun et al., 2017) and computer vision (Zhao et al., 2018) .

In this paper, it is the first time that SplitLBI is exploited to train highly non-convex neural networks with structural sparsity, together with a global convergence analysis based on the Kurdyka-Łojasiewicz framework Łojasiewicz (1963) .

(1) SplitLBI, as an extension of SGD, is applied to deep learning by exploring both over-parameterized models and structural sparsity in the inverse scale space.

(2) Global convergence of SplitLBI in such a nonconvex optimization is established based on the KurdykaŁojasiewicz framework, that the whole iterative sequence converges to a critical point of the empirical loss function from arbitrary initializations.

(3) Stochastic variants of SplitLBI demonstrate the comparable and even better performance than other training algorithms on ResNet-18 in large scale training such as ImageNet-2012, among other datasets, together with additional structural sparsity in successful models for interpretability.

(4) Structural sparsity parameters in SplitLBI provide important information about subnetwork architecture with comparable or even better accuracies than dense models before and after retraining --SplitLBI with early stopping can provide fast "winning tickets" without fully training dense, over-parameterized models.

LetNet-5, trained on MNIST.

The left figure shows the magnitude changes for each filter of the models trained by SplitLBI and SGD, where x-axis and y-axis indicate the training epochs, and filter magnitudes ( 2-norm), respectively.

The SplitLBI path of filters selected in the support of Γ are drawn in blue color, while the red color curves represent the filters that are not important and outside the support of Γ. We visualize the corresponding learned filters by Erhan et al. (2009) at 20 (blue), 40 (green), and 80 (black) epochs, which are shown in the right figure with the corresponding color bounding boxes, i.e., blue, green, and black, respectively.

It shows that our SplitLBI enjoys a sparse selection of filters without sacrificing accuracy (see Table 1 ).

The supervised learning task learns a mapping Φ W : X → Y, from input space X to output space Y, with a parameter W such as weights in neural networks, by minimizing certain loss functions on training samples

, σ i is the nonlinear activation function of the i-th layer.

Differential Inclusion of Inverse Scale Space.

Consider the following dynamics,

where V is a sub-gradient ofΩ(Γ) := Ω λ (Γ) + 1 2κ Γ 2 for some sparsity-enforced regularization Ω λ (Γ) = λΩ 1 (Γ) (λ ∈ R + ) such as Lasso or group Lasso penalties for Ω 1 (Γ), κ > 0 is a damping parameter such that the solution path is continuous, and the augmented loss function is

with ν > 0 controlling the gap admitted between W and Γ. Compared to the original loss function L n (W ), theL (W, Γ) additionally adopt the variable splitting strategy, by lifting the original neural network parameter W to (W, Γ) with Γ modeling the structural sparsity of W .

For simplicity, we assumedL is differentiable with respect to W here, otherwise the gradient in Eq. (1a) is understood as subgradient and the equation becomes an inclusion.

The differential inclusion system (1), called Split Inverse Scale Space (SplitISS), can be understood as a gradient descent flow of W t in the proximity of Γ t and a mirror descent flow (Nemirovski & Yudin, 1983) of Γ t associated with a sparsity enforcement penaltyΩ. In mirror descent flow, gradient descent goes on the dual space consisting of sub-gradients V t , driving the flow in sparse primal space of Γ t .

For a large enough ν, it reduces to the gradient descent method for W t .

Yet the solution path of Γ t exhibits the following property in the separation of scales: starting at the zero, important parameters of large scale will be learned fast, popping up to be nonzeros early, while unimportant parameters of small scale will be learned slowly, appearing to be nonzeros late.

In fact, taking Ω λ (Γ) = Γ 1 and κ → ∞ for simplicity, V t as the subgradient ofΩ t , undergoes a gradient descent flow before reaching the ∞ -unit box, which implies that Γ t = 0 in this stage.

The earlier a component in V t reaches the ∞ -unit box, the earlier a corresponding component in Γ t becomes nonzero and rapidly evolves toward a critical point ofL under gradient flow.

On the other hand, the W t follows the gradient descent with a standard 2 -regularization.

Therefore, W t closely follows dynamics of Γ t whose important parameters are selected.

Such a property is called as the inverse scale space in applied mathematics (Burger et al., 2006) and recently was shown to achieve statistical model selection consistency in high dimensional linear regression (Osher et al., 2016) and general linear models (Huang & Yao, 2018) , with a reduction of bias as κ increases.

In this paper, we shall see that the inverse scale space property still holds empirically for the highly nonconvex neural network training via Eq. (1).

For example, Fig. 1 shows a LeNet trained on MNIST by the discretized dynamics, where important sparse filters are selected in early epochs while the popular SGD returns dense filters.

Compared with directly enforcing a penalty function such as 1 or 2 regularization

SplitISS avoids the parameter correlation problem in over-parameterized models.

In fact, a necessary and sufficient condition for Lasso or 1 -type sparse model selection is the incoherence or irrepresentable conditions (Tropp (2004) ; Zhao & Yu (2006) ) that are violated for highly correlated weight parameters, leading to spurious discoveries.

In contrast, Huang et al. (2018) showed that equipped with such a variable splitting where Γ enjoys an orthogonal design where the restricted Hessian of the augmented loss on Γ is orthogonal, the SplitISS can achieve model selection consistency under weaker conditions than generalized Lasso, relaxing the incoherence or irrepresentable conditions when parameters are highly correlated.

For weight parameter W , instead of directly being imposed with 1 -sparsity, it adopts 2 -regularization in the proximity of the sparse path of Γ that admits simultaneously exploring highly correlated parameters in over-parameterized models and sparsity regularization.

Split Linearized Bregman Iterations.

SplitISS admits an extremely simple discrete approximation, using the Euler forward discretization of dynamics (1):

where V 0 = Γ 0 = 0, W 0 can be small random numbers such as Gaussian distribution in neural networks, for some complex networks it can be initialized as common setting.

The proximal map in Eq. (4c) that controls the sparsity of Γ is given by

We shall call such an iterative procedure as Split Linearized Bregman Iteration (SplitLBI), that was firstly coined in Huang et al. (2016) as an iterative regularization path for sparse modeling in high dimensional statistics.

In the application to neural networks, the loss becomes highly non-convex, the SplitLBI returns a sequence of sparse models from simple to complex ones whose global convergence condition to be shown below, while solving Eq. (3) at various levels of λ might not be tractable except for over-parameterized models.

The sparsity-enforcement penalty used in convolutional neural networks can be chosen as follows.

Our sparsity framework aims at regularizing the groups of weight parameters using group Lasso penalty (Yuan & Lin, 2006) ,

and |Γ g | is the number of weights in Γ g .

Thus Eq. (4c) has a closed form solution

g for the g-th filter.

We treat convolutional and fully connected layers in different ways.

(1) For a convolutional layer, Γ g = Γ g (c in , c out , size) denote the convolutional filters where size denotes the kernel size and c in and c out denote the numbers of input channels and output channels, respectively.

When we regard each group as each convolutional filter, g = c out ; otherwise for weight sparsity, g can be every element in the filter that reduces to the Lasso.

(2) For a fully connected layer, Γ = Γ(c in , c out ) where c in and c out denote the numbers of inputs and outputs of the fully connected layer.

Each group g corresponds to each element (i, j), and the group Lasso penalty degenerates to the Lasso penalty.

We present a theorem that guarantees the global convergence of SplitLBI, i.e. from any intialization, the SplitLBI sequence converges to a critical point ofL. Our treatment extends the block coordinate descent (BCD) studied in Zeng et al. (2019) , with a crucial difference being the mirror descent involved in SplitLBI.

Instead of the splitting loss in BCD (Zeng et al., 2019) , a new Lyapunov function is developed here to meet the Kurdyka-Łojasiewicz property Łojasiewicz (1963) .

Xue & Xin (2018) studied convergence of variable splitting method for single hidden layer networks with Gaussian inputs.

Let P := (W, Γ).

Following Huang & Yao (2018) , the SplitLBI algorithm in Eq. (4a-4c) can be rewritten as the following standard Linearized Bregman Iteration,

where

p k ∈ ∂Ψ(P k ), and B q Ψ is the Bregman divergence associated with convex function Ψ, defined by B

Without loss of generality, consider λ = 1 in the sequel.

One can establish the global convergence of SplitLBI under the following assumptions.

is continuous differentiable and ∇ L n is Lipschitz continuous with a positive constant Lip; (b) L n (W ) has bounded level sets; (c) L n (W ) is lower bounded (without loss of generality, we assume that the lower bound is 0); (d) Ω is a proper lower semi-continuous convex function and has locally bounded subgradients, that is, for every compact set S ⊂ R n , there exists a constant C > 0 such that for all Γ ∈ S and all g ∈ ∂Ω(Γ), there holds g ≤ C; and (e) the Lyapunov function

is a Kurdyka-Łojasiewicz function on any bounded set, where

and Ω * is the conjugate of Ω defined as

Remark 1.

Assumption 1 (a)-(c) are regular in the analysis of nonconvex algorithm (see, Attouch et al. (2013) for instance), while Assumption 1 (d) is also mild including all Lipschitz continuous convex function over a compact set.

Some typical examples satisfying Assumption 1(d) are the 1 norm, group 1 norm, and every continuously differentiable penalties.

By Eq. (9) and the definition of conjugate, the Lyapunov function F can be rewritten as follows,

Now we are ready to present the main theorem.

Theorem 1. [Global Convergence of SplitLBI] Suppose that Assumption 1 holds.

Let (W k , Γ k ) be the sequence generated by SplitLBI (Eq. (4a-4c)) with a finite initialization.

If

,

converges to a critical point ofL defined in Eq. (2), and

Applying to the neural networks, typical examples are summarized in the following corollary.

Corollary 1.

Let {W k , Γ k , g k } be a sequence generated by SLBI (16a-16c) for neural network training where (a) is any smooth definable loss function, such as the square loss (t 2 ), exponential loss (e t ), logistic loss log(1 + e −t ), and cross-entropy loss; (b) σ i is any smooth definable activation, such

−t e t +e −t ), and softplus ( 1 c log(1 + e ct ) for some c > 0) as a smooth approximation of ReLU; (c) Ω is the group Lasso.

Then the sequence {W k } converges to a stationary point of L n (W ) under the conditions of Theorem 1.

Proofs of Theorem 1 and Corollary 1 are given in Appendix A.

We begin with some stochastic variants of SplitLBI and implementations, followed by four groups of experiments demonstrating the utilities of weight parameter W t and structural sparsity parameter Γ t in prediction, interpretability, and capturing effective sparse subnetworks.

Batch Split LBI.

For neural network training with large datasets, stochastic approximation of the gradients in Split LBI over the mini-batch (X, Y) batcht is adopted to update the parameter W ,

SplitLBI with momentum (Mom).

Inspired by the variants of SGD, the momentum term can be also incorporated to the standard Split LBI that leads to the following updates of W by replacing Eq (4a) with,

where τ is the momentum factor, empirically setting as 0.9 in default.

One immediate application of such stochastic algorithms of SplitLBI is to "boost networks", i.e. growing a network from the null to a complex one by sequentially applying our algorithm on subnets with increasing complexities.

SplitLBI with momentum and weight decay (Mom-Wd).

The update formulation is,

where β is set as 1e −4 .

Implementation.

Various algorithms are evaluated over the various backbones -LeNet (LeCun et al., 2015) , AlexNet (Krizhevsky et al., 2012) , VGG (Simonyan & Zisserman, 2014), and ResNet (He et al., 2016) etc., respectively.

For MNIST and Cifar-10, the default hyper-parameters of Split LBI are κ = 1, ν = 10 and α k is set as 0.1, decreased by 1/10 every 30 epochs.

In ImageNet-2012, the Split LBI utilizes κ = 1, ν = 1000, and α k is initially set as 0.1, decays 1/10 every 30 epochs.

We set λ = 1 in Eq. (5) by default, unless otherwise specified.

On MNIST and Cifar-10, the batch size is set as 128; and for all methods, the batch size of ImageNet 2012 is 256.

The standard data augmentation implemented in pytorch is applied to Cifar-10 and ImageNet2012 datasets, as He et al. (2016) .

The weights of all models are initialized as He et al. (2015) .

In the following experiments, we define sparsity as percentage of non-zero parameters, i.e. the number of non-zero weights dividing the total number of weights in consideration, that equals to one minus the pruning rate of the network.

We also have the reproducible source codes 1 .

In SplitLBI, the weight parameter W t explores over-parameterized models that can achieve the state-of-the-art performance in large scale training such as ImageNet-2012 classification.

Experimental Design.

We compare different variants of SGD and Adam in the experiments.

By default, the learning rate of competitors is set as 0.1 for SGD and its variant and 0.001 for Adam and its variants, and gradually decreased by 1/10 every 30 epochs.

In particular, we have, SGD: (1) Naive SGD: the standard SGD with batch input.

(2) SGD with l 1 penalty (Lasso).

The l 1 norm is applied to penalize the weights of SGD by encouraging the sparsity of learned model, with the regularization parameter of the l 1 penalty term being set as 1e −3 (3) SGD with momentum (Mom): we utilize momentum 0.9 in SGD.

(4) SGD with momentum and weight decay (Mom-Wd): we set the momentum 0.9 and the standard l 2 weight decay with the coefficient weight 1e SplitLBI achieves the state-of-the-art performance on ImageNet-2012, etc.

Tab.

1 shows the experimental results on ImageNet-2012, Cifar-10, and MNIST of some classical networks --LeNet, AlexNet and ResNet.

Our SplitLBI variants may achieve comparable or even better performance than SGD variants in 100 epochs, indicating the efficacy in learning dense, over-parameterized models.

In SplitLBI, the structural sparsity parameter Γ t explores important sub-network architectures that contributes significantly to the loss or error reduction in early training stages.

Through the 2 -coupling, structural sparsity parameter Γ t may guide the weight parameter to explore those sparse models in favour of improved interpretabiity.

For example, Fig. 1 visualizes some sparse filters learned by SplitLBI of LeNet-5 trained on MNIST (with κ = 10 and weight decay every 40 epochs), in comparison with dense filters learned by SGD.

The activation pattern of such sparse filters favours high order global correlations between pixels of input images.

To further reveal the insights of learned patterns of SplitLBI, we visualize the first convolutional layer of ResNet-18 on ImageNet-2012 along the training path of our SplitLBI as in Fig. 2 .

The left figure compares the training and validation accuracy of SplitLBI and SGD.

The right figure compares visualizations of the filters learned by SplitLBI and SGD using Springenberg et al. (2014) .

Implementation.

To be specific, denote the weights of an l-layer network as {W 1 , W 2 , · · · , W l }.

For the i−th layer weights W i , denote the j−th channel W i j .

Then we compute the gradient of the sum of the feature map computed from each filter W i j with respect to the input image (here a snake image).

We further conduct the min-max normalization to the gradient image, and generate the final visualization map.

The right figure compares the visualized gradient images of first convolutional layer of 64 filters with 7 × 7 receptive fields.

We visualize the models parameters at 20 (purple), 40 (green), and 60 (black) epochs, respectively, which corresponds to the bounding boxes in the right figure annotated by the corresponding colors, i.e., purple, green, and black.

We order the gradient images produced from 64 filters by the descending order of the magnitude ( 2 -norm) of filters, i.e., images are ordered from the upper left to the bottom right.

For comparison, we also provide the visualized gradient from random initialized weights.

Filters learned by ImageNet prefer to non-semantic texture rather than shape and color.

The filters of high norms mostly focus on the texture and shape information, while color information is with the filters of small magnitudes.

This phenomenon is in accordance with observation of AbbasiAsl & Yu (2017) that filters mainly of color information can be pruned for saving computational cost.

Moreover, among the filters of high magnitudes, most of them capture non-semantic textures while few pursue shapes.

This shows that the first convolutional layer of ResNet-18 trained on ImageNet learned non-semantic textures rather than shape to do image classification tasks, in accordance with recent studies (Geirhos et al., 2019) .

How to enhance the semantic shape invariance learning, is arguably a key to improve the robustness of convolutional neural networks.

We conduct ablation studies based on Cifar-10 dataset with VGG-16 and ResNet-56 to evaluate (i) global convergence ofL; and (ii) the structural sparsity learned by Γ t via exploring test accuracies of sparse models obtained by projecting W t onto the support set of Γ t (mask), by varying two key hyper-parameters κ and ν.

Implementation.

We choose SplitLBI with momentum and weight decay, since it achieves very good performance on large-scale experiments.

Specifically, we have two set of experiments, where each experiment is repeated for 5 times: (1) we fix ν = 100 and vary κ = 1, 2, 5, 10, where sparsity of Γ t and validation accuracies of sparse models are shown in top row of Fig. 4 .

Note that we keep κ · α k = 0.1 in Eq (1a), to make comparable learning rate of each variant, and also consistent with SGD.

Thus the learning rate α k will be adjusted by different κ values.

(2) we fix κ = 1, and validate the results of SplitLBI with ν = 10, 20, 50, 100, 200, 500, 1000, 2000 in the second row of Fig. 4 with the learning rate α k = 0.1.

Moreover, rather than using sparse models associated with Γ t , Fig.  6 in Appendix shows the validation accuracies of full models learned by W t .

SplitLBI converges to Critical Point.

Figure 3 shows the curves of training loss ( L n ) and accuracies, with each point representing the average and variance bar over 5 times.

As shown, both training loss and training accuracy will converge, which validates our theoretical result in Theorem 1.

Besides, larger κ brings in slower convergence, which agrees with the analysis the convergence rate is inversely scale to κ in Lemma A.5.

Sparse subnetworks achieve comparable performance to dense models without fine-tuning or retraining.

From the experiments above, the sparsity of Γ grows as κ and ν increase.

While large κ may cause a small number of important parameters growing rapidly, large ν will decouple W t and Γ t such that the growth of W t does not affect Γ t that may over-sparsify and deteriorate model accuracies.

Thus a moderate choice of κ and ν is preferred in practice.

In all cases, one can see that moderate sparse models can achieve comparable predictive power to dense models, even without fine-tuning or retraining.

This shows that the structural sparsity parameter Γ t can indeed capture important weight parameter W t through their coupling.

Equipped with early stopping, Γ t in early epochs may learn effective subnetworks (i.e. "winning tickets" (Frankle & Carbin, 2019; Liu et al., 2019) ) that achieve comparable or even better performance after retraining than existing pruning strategies by SGD.

Experimental Design.

We adopt a comparison baseline as the one-shot pruning strategy in Frankle & Carbin (2019) , which firstly trains a dense over-parameterized model by SGD for T = 160 epochs and find the sparse structure by pruning weights or filters (Liu et al., 2019) , then secondly retrains the structure from the scratch with T epochs from the same initialization as the first step.

For SplitLBI, instead of pruning weights/filters from dense models, we directly utilize the structural sparsity Γ t at different training epochs to define the subnet architecture, followed by retrain-from-scratch (Fine-tune is shown in Appendix Sec. D with preliminary results).

Experiments are conducted on Cifar-10 dataset where we still use VGG-16, ResNet-50, and ResNet-56 as the networks to make direct comparisons to previous works.

SplitLBI uses momentum and weight decay with hyperparameters shown in Tab.

10 in Appendix.

In particular, we set λ = 0.1, and 0.05 for VGG-16, and ResNet-56 respectively, since ResNet-56 has less parameters than VGG-16.

Furthermore, we introduce another variant of our SplitLBI by using Lasso rather than group lasso penalty for Γ t to sparsify the weights of convolutional filters; and the corresponding models are denoted as VGG-16 (Lasso) and ResNet-50 (Lasso).

Every experiment is repeated for five times and the results are shown in Fig. 5 .

Note that in different runs of SplitLBI, the sparsity of Γ t slightly varies.

Sparse subnets found by early stopping of SplitLBI achieve remarkably good accuracy after retrain from scratch.

In Fig.5 (a-b) , sparse filters discovered by Γ t at different epochs are compared against the methods of Network Slimming (Liu et al., 2017 ), Soft Filter Pruning (Yang et al., 2018 , Scratch-B, and Scratch-E, whose results are reported from Liu et al. (2019) .

At similar sparsity levels, SplitLBI can achieve comparable or even better accuracy than competitors, even with sparse architecture learned from very early epochs (e.g. t = 20 or 10).

Moreover in Fig.5 (c-d) , we can draw the same conclusion for the sparse weights of VGG-16 (Lasso) and ResNet-50 (Lasso), against the results reported in Liu et al. (2019) .

These results shows that the structural sparsity parameter Γ t found by early stopping of SplitLBI already discloses important subnetwork architecture that may achieve remarkably good accuracy after retrain from scratch.

Therefore, it is not necessary to fully train a dense model to find a successful sparse subnet architecture with comparable performance to the dense ones --one can early stop SplitLBI properly where the structural parameter Γ t unveils "winning tickets" (Frankle & Carbin, 2019) .

In this paper, a parsimonious deep learning method is proposed based on differential inclusions of inverse scale spaces.

Implemented by a variable splitting scheme, such a dynamics system can exploit over-parameterized models and structural sparsity simultaneously.

Besides, its simple discretization, i.e., the SplitLBI, has a proven global convergence and hence can be employed to train deep networks.

We have experimentally shown that it can achieve the state-of-the-art performance on many datasets including ImageNet-2012, with better interpretability than SGD.

What's more, equipped with early stopping, such a structural sparsity can unveil the "winning tickets" -the architecture of sub-networks which after re-training can achieve comparable and even better accuracy than original dense networks.

First of all, we reformulate Eq. (6) into an equivalent form.

Without loss of generality, consider Ω = Ω 1 in the sequel.

Denote R(P ) := Ω(Γ), then Eq. (6) can be rewritten as,

where

Thus SplitLBI is equivalent to the following iterations,

Exploiting the equivalent reformulation (16a-16c), one can establish the global convergence of (W k , Γ k , g k ) based on the Kurdyka-Łojasiewicz framework.

In this section, the following extended version of Theorem 1 is actually proved.

be the sequence generated by SplitLBI (Eq. (16a-16c)) with a finite initialization.

If

,

to a critical point of F .

Moreover, {(W k , Γ k )} converges to a stationary point ofL defined in Eq. 2, and {W k } converges to a stationary point of L n (W ).

To introduce the definition of the Kurdyka-Łojasiewicz (KL) property, we need some notions and notations from variational analysis, which can be found in Rockafellar & Wets (1998) .

The notion of subdifferential plays a central role in the following definitions.

For each x ∈ dom(h) := {x ∈ R p : h(x) < +∞}, the Fréchet subdifferential of h at x, written ∂h(x), is the set of vectors v ∈ R p which satisfy lim inf

When x / ∈ dom(h), we set ∂h(x) = ∅. The limiting-subdifferential (or simply subdifferential) of h introduced in Mordukhovich (2006) , written ∂h(x) at x ∈ dom(h), is defined by

and its domain by dom(h) := {x ∈ R p : h(x) < +∞} (resp.

dom(h) := {x ∈ R p : h(x) = ∅}).

When h is a proper function, i.e., when dom(h) = ∅, the set of its global minimizers (possibly empty) is denoted by arg min h := {x ∈ R p : h(x) = inf h}.

The KL property (Łojasiewicz, 1963; 1993; Kurdyka, 1998; Bolte et al., 2007a; b) plays a central role in the convergence analysis of nonconvex algorithms (Attouch et al., 2013; Wang et al., 2019) .

The following definition is adopted from Bolte et al. (2007b) .

Definition 1. [Kurdyka-Łojasiewicz property]

A function h is said to have the Kurdyka-Łojasiewicz (KL) property atū ∈ dom(∂h) := {v ∈ R n |∂h(v) = ∅}, if there exists a constant η ∈ (0, ∞), a neighborhood N ofū and a function φ : [0, η) → R + , which is a concave function that is continuous at 0 and satisfies φ(0) = 0, φ ∈ C 1 ((0, η)), i.e., φ is continuous differentiable on (0, η), and φ (s) > 0 for all s ∈ (0, η), such that for all u ∈ N ∩ {u ∈ R n |h(ū) < h(u) < h(ū) + η}, the following inequality holds

If h satisfies the KL property at each point of dom(∂h), h is called a KL function.

KL functions include real analytic functions, semialgebraic functions, tame functions defined in some o-minimal structures (Kurdyka, 1998; Bolte et al., 2007b) , continuous subanalytic functions (Bolte et al., 2007a) and locally strongly convex functions.

In the following, we provide some important examples that satisfy the Kurdyka-Łojasiewicz property.

One can verify whether a multivariable real function h(x) on R p is analytic by checking the analyticity of g(t) := h(x + ty) for any x, y ∈ R p .

(a) A set D ⊂ R p is called semialgebraic (Bochnak et al., 1998) if it can be represented as

where P ij , Q ij are real polynomial functions for

According to (Łojasiewicz, 1965; Bochnak et al., 1998) and (Shiota, 1997, I .2.9, page 52), the class of semialgebraic sets are stable under the operation of finite union, finite intersection, Cartesian product or complementation.

Some typical examples include polynomial functions, the indicator function of a semialgebraic set, and the Euclidean norm (Bochnak et al., 1998, page 26) .

In the following, we consider the deep neural network training problem.

Consider a l-layer feedforward neural network including l − 1 hidden layers of the neural network.

Particularly, let d i be the number of hidden units in the i-th hidden layer for i = 1, . . .

, l − 1.

Let d 0 and d l be the number of units of input and output layers, respectively.

Let W i ∈ R di×di−1 be the weight matrix between the (i − 1)-th layer and the i-th layer for any i = 1, . . .

l 2 .

According to Theorem 2, one major condition is to verify the introduced Lyapunov function F defined in (9) satisfies the Kurdyka-Łojasiewicz property.

For this purpose, we need an extension of semialgebraic set, called the o-minimal structure (see, for instance Coste (1999) , van den Dries (1986) , Kurdyka (1998), Bolte et al. (2007b) ).

The following definition is from Bolte et al. (2007b) .

Definition 4. [o-minimal structure]

An o-minimal structure on (R, +, ·) is a sequence of boolean algebras O n of "definable" subsets of R n , such that for each n ∈ N (i) if A belongs to O n , then A × R and R × A belong to O n+1 ;

(ii) if Π : R n+1 → R n is the canonical projection onto R n , then for any A in O n+1 , the set Π(A) belongs to O n ; (iii) O n contains the family of algebraic subsets of R n , that is, every set of the form

where p : R n → R is a polynomial function.

(iv) the elements of O 1 are exactly finite unions of intervals and points.

Based on the definition of o-minimal structure, we can show the definition of the definable function.

According to van den Dries & Miller (1996) ; Bolte et al. (2007b) , there are some important facts of the o-minimal structure, shown as follows.

(i) The collection of semialgebraic sets is an o-minimal structure.

Recall the semialgebraic sets are Bollean combinations of sets of the form

where p and q i 's are polynomial functions in R n .

(ii) There exists an o-minimal structure that contains the sets of the form

where f is real-analytic around [−1, 1] n .

(iii) There exists an o-minimal structure that contains simultaneously the graph of the exponential function R x → exp(x) and all semialgebraic sets. (iv) The o-minimal structure is stable under the sum, composition, the inf-convolution and several other classical operations of analysis.

The Kurdyka-Łojasiewicz property for the smooth definable function and non-smooth definable function were established in (Kurdyka, 1998, Theorem 1) and (Bolte et al., 2007b, Theorem 11) , respectively.

Now we are ready to present the proof of Corollary 1.

Proof. [

Proof of Corollary 1] To justify this corollary, we only need to verify the associated Lyapunov function F satisfies Kurdyka-Łojasiewicz inequality.

In this case and by (10), F can be rewritten as follows

Because and σ i 's are definable by assumptions, then L n (W, Γ) are definable as compositions of definable functions.

Moreover, according to Krantz & Parks (2002) , W − Γ 2 and W, g are semi-algebraic and thus definable.

Since the group Lasso Ω(Γ) = g Γ 2 is the composition of 2 and 1 norms, and the conjugate of group Lasso penalty is the maximum of group 2 -norm, i.e. Ω * (Γ) = max g Γ g 2 , where the 2 , 1 , and ∞ norms are definable, hence the group Lasso and its conjugate are definable as compositions of definable functions.

Therefore, F is definable and hence satisfies Kurdyka-Łojasiewicz inequality by (Kurdyka, 1998 , Theorem 1).

The verifications of other cases listed in assumptions can be found in the proof of (Zeng et al., 2019, Proposition 1) .

This finishes the proof of this corollary.

Our analysis is mainly motivated by a recent paper (Benning et al., 2017) , as well as the influential work (Attouch et al., 2013) .

According to (Attouch et al., 2013 , Lemma 2.6), there are mainly four ingredients in the analysis, that is, the sufficient descent property, relative error property, continuity property of the generated sequence and the Kurdyka-Łojasiewicz property of the function.

More specifically, we first establish the sufficient descent property of the generated sequence via exploiting the Lyapunov function F (see, (9)) in Lemma A.4 in Section A.4, and then show the relative error property of the sequence in Lemma A.5 in Section A.5.

The continuity property is guaranteed by the continuity ofL(W, Γ) and the relation lim k→∞ B g k Ω (Γ k+1 , Γ k ) = 0 established in Lemma 1(i) in Section A.4.

Thus, together with the Kurdyka-Łojasiewicz assumption of F , we establish the global convergence of SLBI following by (Attouch et al., 2013 , Lemma 2.6).

Let (W ,Γ,ḡ) be a critical point of F , then the following holds

By the final inclusion and the convexity of Ω, it impliesḡ ∈ ∂Ω(Γ).

Plugging this inclusion into the second inclusion yields αν −1 (Γ −W ) = 0.

Together with the first equality imples

This finishes the proof of this theorem.

In the following, we present the sufficient descent property of Q k along the Lyapunov function F .

Lemma.

Suppose that L n is continuously differentiable and ∇ L n is Lipschitz continuous with a constant Lip > 0.

Let {Q k } be a sequence generated by SLBI with a finite initialization.

If

Proof.

By the optimality condition of (15a) and also the inclusion

where

and by the Lipschitz continuity of ∇ L n (W ) with a constant Lip > 0 implies ∇L is Lipschitz continuous with a constant Lip + ν −1 .

This implies

.

Substituting the above inequality into (20) yields

Adding some terms in both sides of the above inequality and after some reformulations implies

where the final equality holds for

where the final inequality holds for B

Thus, we finish the proof of this lemma.

Based on Lemma A.4, we directly obtain the following lemma.

Lemma 1.

Suppose that assumptions of Lemma A.4 hold.

Suppose further that Assumption 1 (b)-(d) hold.

Then (i) both α{L(P k )} and {F (Q k )} converge to the same finite value, and

is also monotonically decreasing.

By the lower boundedness assumption of L n (W ), both L(P ) and F (Q) are lower bounded by their definitions, i.e., (2) and (9), respectively.

Therefore, both {L(P k )} and {F (Q k )} converge, and it is obvious that lim k→∞ F (Q k ) ≥ lim k→∞ αL(P k ).

By (23),

By the convergence of F (Q k ) and the nonegativeness of B

)

and the above equality, it yields

Since L n (W ) has bounded level sets, then W k is bounded.

By the definition ofL(W, Γ) and the finiteness ofL(W k , Γ k ), Γ k is also bounded due to W k is bounded.

The boundedness of g k is due to g k ∈ ∂Ω(Γ k ), condition (d), and the boundedness of Γ k .

By (24), summing up (24) over k = 0, 1, . . . , K yields

Letting K → ∞ and noting that both P k+1 − P k 2 and D(Γ k+1 , Γ k ) are nonnegative, thus

Again by (25),

In this subsection, we provide the bound of subgradient by the discrepancy of two successive iterates.

By the definition of F (9),

Lemma.

Under assumptions of Lemma 1, then

where ρ 1 := 2κ

Proof.

Note that

By the definition ofL (see (2)),

where the last inequality holds for the Lipschitz continuity of ∇ L n with a constant Lip > 0, and by (16a),

Substituting the above (in)equalities into (27) yields

By (16c), it yields

, and after some simplifications yields

where the last inequality holds for the triangle inequality and κ −1 > αν −1 by the assumption.

By (28), (29), and the definition of H k+1 (26), there holds

By (30) and Lemma 1(iv),

This finishes the proof of this lemma.

Figure 6: Validation curves of dense models W t for different κ and ν.

For SLBI we find that the model accuracy is robust to the hyperparameters both in terms of convergence rate and generalization ability.

Here validation accuracy means the accuracy on test set of Cifar10.

The first one is the result for VGG16 ablation study on κ, the second one is the result for ResNet56 ablation study on κ, the third one is the result for VGG16 ablation study on ν and the forth one is the result for ResNet56 ablation study on ν.

To further study the influence of hyperparameters, we record performance of W t for each epoch t with different combinations of hyperparameters.

The experiments is conducted 5 times each, we show the mean in the table, the standard error can be found in the corresponding figure.

We perform experiments on Cifar10 and two commonly used network VGG16 and ResNet56.

On κ , we keep ν = 100 and try κ = 1, 2, 5, 10, the validation curves of models W t are shown in Fig. 6 and Table 2 summarizes the mean accuracies.

Table 3 summarizes best validation accuracies achieved at some epochs, together with their sparsity rates.

These results show that larger kappa leads to slightly lower validation accuracies, where the numerical results are shown in Table 2 .

We can find that κ = 1 achieves the best test accuracy.

On ν , we keep κ = 1 and try ν = 10, 20, 50, 100, 200, 500, 1000, 2000 the validation curve and mean accuracies are show in Fig. 6 and Table 4 .

Table 5 summarizes best validation accuracies achieved at some epochs, together with their sparsity rates.

By carefully tuning ν we can achieve similar or even better results compared to SGD.

Different from κ, ν has less effect on the generalization performance.

By tuning it carefully, we can even get a sparse model with slightly better performance than SGD trained model.

We further compare the computational cost of different optimizers: SGD (Mom), SplitLBI (Mom) and Adam (Naive).

We test each optimizer on one GPU, and all the experiments are done on one GTX2080.

For computational cost, we judge them from two aspects : GPU memory usage and time needed for one batch.

The batch size here is 64, experiment is performed on VGG-16 as shown in Table 7 : This table shows the sparsity for every layer of Lenet-3.

Here sparsity is defined in Sec. 4, number of weights denotes the total number of parameters in the designated layer.

It is interesting that the Γ tends to put lower sparsity on layer with more parameters.

We design the experiment on MNIST, inspired by Frankle & Carbin (2019) .

Here, we explore the subnet obtained by Γ T after T = 100 epochs of training.

As in Frankle et al. (2019) , we adopt the "rewind" trick: re-loading the subnet mask of Γ 100 at different epochs, followed by fine-tuning.

In particular, along the training paths, we reload the subnet models at Epoch 0, Epoch 30, 60, 90, and 100, and further fine-tune these models by SplitLBI (Mom-Wd).

All the models use the same initialization and hence the subnet model at Epoch 0 gives the retraining with the same random initialization as proposed to find winning tickets of lottery in Frankle & Carbin (2019) .

We will denote the rewinded fine-tuned model at epoch 0 as (Lottery), and those at epoch 30, 60, 90, and 100, as F-epoch30, F-epoch60, F-epoch90, and F-epoch100, respectively.

Three networks are studied here -LeNet-3, Conv-2, and Conv-4.

LeNet-3 removes one convolutional layer of LeNet-5; and it is thus less over-parameterized than the other two networks.

Conv-2 and Conv-4, as the scaled-down variants of VGG family as done in Frankle & Carbin (2019) , have two and four fully-connected layers, respectively, followed by max-pooling after every two convolutional layer.

The whole sparsity for Lenet-3 is 0.055, Conv-2 is 0.0185, and Conv-4 is 0.1378.

Detailed sparsity for every layer of the model is shown in Table 7 , 8, 9.

We find that fc-layers are sparser than conv-layers.

We compare SplitLBI variants to the SGD (Mom-Wd) and SGD (Lottery) (Frankle & Carbin, 2019) in the same structural sparsity and the results are shown in Fig. 7 .

In this exploratory experiment, one can see that for overparameterized networks -Conv-2 and Conv-4, fine-tuned rewinding subnets -F-epoch30, F-epoch60, F-epoch90, and F-epoch100, can produce better results than the full models; while for the less over-parameterized model LeNet-3, fine-tuned subnets may achieve less yet still comparable performance to the dense models and remarkably better than the retrained sparse subnets from beginning (i.e. SplitLBI/SGD (Lottery)).

These phenomena suggest that the subnet architecture disclosed by structural sparsity parameter Γ T is valuable, for fine-tuning sparse models with comparable or even better performance than the dense models of W T .

Here we provide more details on the experiments in Fig. 5 .

Table 10 gives the details on hyperparameter setting.

Moreover, Figure 8 provides .

5) .

We calculate the sparsity in every epoch and repeat five times.

The black curve represents the mean of the sparsity and shaded area shows the standard deviation of sparsity.

The vertical blue line shows the epochs that we choose to early stop.

We choose the log-scale epochs for achieve larger range of sparsity.

@highlight

SplitLBI is applied to deep learning to explore model structural sparsity, achieving state-of-the-art performance in ImageNet-2012 and unveiling effective subnet architecture.

@highlight

Proposes an optimization based algorithm for finding important sparse structures of large-scale neural networks by coupling the learning of weight matrix and sparsity constraints, offering guaranteed convergence on nonconvex optimization problems.