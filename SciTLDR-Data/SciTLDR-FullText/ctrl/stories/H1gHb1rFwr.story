Recent evidence shows that convolutional neural networks (CNNs) are biased towards textures so that CNNs are non-robust to adversarial perturbations over textures, while traditional robust visual features like SIFT (scale-invariant feature transforms) are designed to be robust across a substantial range of affine distortion, addition of noise, etc with the mimic of human perception nature.

This paper aims to leverage good properties of SIFT to renovate CNN architectures towards better accuracy and robustness.

We borrow the scale-space extreme value idea from SIFT, and propose EVPNet (extreme value preserving network) which contains three novel components to model the extreme values: (1) parametric differences of Gaussian (DoG) to extract extrema, (2) truncated ReLU to suppress non-stable extrema and (3) projected normalization layer (PNL) to mimic PCA-SIFT like feature normalization.

Experiments demonstrate that EVPNets can achieve similar or better accuracy than conventional CNNs, while achieving much better robustness on a set of adversarial attacks (FGSM,PGD,etc) even without adversarial training.

Convolutional neural networks (CNNs) evolve very fast ever since AlexNet (Krizhevsky & Hinton, 2012 ) makes a great breakthrough on ImageNet image classification challenge (Deng et al., 2009 ) in 2012.

Various network architectures have been proposed to further boost classification performance since then, including VGGNet (Simonyan & Zisserman, 2015) , GoogleNet , ResNet (He et al., 2016) , DenseNet (Huang et al., 2017) and SENet , etc.

Recently, people even introduce network architecture search to automatically learn better network architectures (Zoph & Le, 2017; Liu et al., 2018) .

However, state-of-the-art CNNs are challenged by their robustness, especially vulnerability to adversarial attacks based on small, human-imperceptible modifications of the input (Szegedy et al., 2014; Goodfellow et al., 2015) .

thoroughly study the robustness of 18 well-known ImageNet models using multiple metrics, and reveals that adversarial examples are widely existent.

Many methods are proposed to improve network robustness, which can be roughly categorized into three perspectives: (1) modifying input or intermediate features by transformation (Guo et al., 2018) , denoising Jia et al., 2019) , generative models (Samangouei et al., 2018; Song et al., 2018) ; (2) modifying training by changing loss functions (Wong & Kolter, 2018; Elsayed et al., 2018; , network distillation (Papernot et al., 2016) , or adversarial training (Goodfellow et al., 2015; Tramer et al., 2018 ) (3) designing robust network architectures Svoboda et al., 2019; Nayebi & Ganguli, 2017) and possible combinations of these basic categories.

For more details of current status, please refer to a recent survey (Akhtar & Mian, 2018) .

Although it is known that adversarial examples are widely existent , some fundamental questions are still far from being well studied like what causes it, and how the factor impacts the performance, etc.

One of the interesting findings in is that model architecture is a more critical factor to network robustness than model size (e.g. number of layers).

Some recent works start to explore much deeper nature.

For instance, both (Geirhos et al., 2019; Baker et al., 2018) show that CNNs are trained to be strongly biased towards textures so that CNNs do not distinguish objects contours from other local or even noise edges, thus perform poorly on shape dominating object instances.

On the contrary, there are no statistical difference for human behaviors on both texture rich objects and global shape dominating objects in psychophysical trials.

Ilyas et al. (2019) further analyze and show that deep convolutional features can be categorized into robust and non-robust features, while non-robust features may even account for good generalization.

However, non-robust features are not expected to have good model interpretability.

It is thus an interesting topic to disentangle robust and non-robust features with certain kinds of human priors in the network designing or training process.

In fact, human priors have been extensively used in handcraft designed robust visual features like SIFT (Lowe, 2004) .

SIFT detects scale-space (Lindeberg, 1994) extrema from input images, and selects stable extrema to build robust descriptors with refined location and orientation, which achieves great success for many matching and recognition based vision tasks before CNN being reborn in 2012 (Krizhevsky & Hinton, 2012) .

The scale-space extrema are efficiently implemented by using a difference-of-Gaussian (DoG) function to search over all scales and image locations, while the DoG operator is believed to biologically mimic the neural processing in the retina of the eye (Young, 1987) .

Unfortunately, there is (at least explicitly) no such scale-space extrema operations in all existing CNNs.

Our motivation is to study the possibility of leveraging good properties of SIFT to renovate CNN networks architectures towards better accuracy and robustness.

In this paper, we borrow the scale-space extrema idea from SIFT, and propose extreme value preserving networks (EVPNet) to separate robust features from non-robust ones, with three novel architecture components to model the extreme values: (1) parametric DoG (pDoG) to extract extreme values in scale-space for deep networks, (2) truncated ReLU (tReLU) to suppress noise or non-stable extrema and (3) projected normalization layer (PNL) to mimic PCA-SIFT (Ke et al., 2004) like feature normalization.

pDoG and tReLU are combined into one block named EVPConv, which could be used to replace all k ?? k (k > 1) conv-layers in existing CNNs.

We conduct comprehensive experiments and ablation studies to verify the effectiveness of each component and the proposed EVPNet.

Figure 1 illustrates a comparison of responses for standard convolution + ReLU and EVPConv in ResNet-50 trained on ImageNet, and shows that the proposed EVPConv produces less noises and more responses around object boundary than standard convolution + ReLU, which demonstrates the capability of EVPConv to separate robust features from non-robust ones.

Our major contribution are:

??? To the best of our knowledge, we are the first to explicitly separate robust features from non-robust ones in deep neural networks from an architecture design perspective.

??? We propose three novel network architecture components to model extreme values in deep networks, including parametric DoG, truncated ReLU, and projected normalization layer, and verify their effectiveness through comprehensive ablation studies.

??? We propose extreme value preserving networks (EVPNets) to combine those three novel components, which are demonstrated to be not only more accurate, but also more robust to a set of adversarial attacks (FGSM, PGD, etc) even for clean model without adversarial training.

Robust visual features.

Most traditional robust visual feature algorithms like SIFT (Lowe, 2004) and SURF (Bay et al., 2006) are based on the scale-space theory (Lindeberg, 1994) , while there is a close link between scale-space theory and biological vision (Lowe, 2004) , since many scalespace operations show a high degree of similarity with receptive field profiles recorded from the mammalian retina and the first stages in the visual cortex.

For instance, DoG computes the difference of two Gaussian blurred images and is believed to mimic the neural processing in the retina (Young, 1987) .

SIFT is one such kind of typical robust visual features, which consists of 4 major stages: (1) scale-space extrema detection with DoG operations; (2) Keypoints localization by their stability; (3) Orientation and scale assignment based on primary local gradient direction; (4) Histogram based keypoint description.

We borrow the scale-space extrema idea from SIFT, and propose three novel and robust architecture components to mimic key stages of SIFT.

Robust Network Architectures.

Many research efforts have been devoted to network robustness especially on defending against adversarial attacks as summarized in Akhtar & Mian (2018) .

However, there are very limited works that tackle this problem from a network architecture design perspective.

A major category of methods focus on designing new layers to perform denoising operations on the input image or the intermediate feature maps.

Most of them are shown effective on black-box attacks, while are still vulnerable to white-box attacks.

Non-local denoising layer proposed in is shown to improve robustness to white-box attack to an extent with adversarial training (Madry et al., 2018) .

Peer sample information is introduced in Svoboda et al. (2019) with a graph convolution layer to improve network robustness.

Biologically inspired protection (Nayebi & Ganguli, 2017) introduces highly non-linear saturated activation layer to replace ReLU layer, and demonstrates good robustness to adversarial attacks, while similar higher-order principal is also used in Krotov & Hopfield (2018) .

However, these methods still lack a systematic architecture design guidance, and many (Svoboda et al., 2019; Nayebi & Ganguli, 2017) are not robust to iterative attack methods like PGD under clean model setting.

In this work, inspired by robust visual feature SIFT, we are able to design a series of innovative architecture components systematically for improving both model accuracy and robustness.

We should stress that extreme value theory is a different concept to scale-space extremes, which tries to model the extreme in data distribution, and is used to design an attack-independent metric to measure robustness of DNNs (Weng et al., 2018) by exploring input data distribution.

Difference-of-Gaussian.

Given an input image I and Gaussian kernel G(x, y, ??) as below

where ?? denotes the variance.

Also, difference of Gaussian (DoG) is defined as

where ??? is the convolution operation, and I 1 = G(x, y, ??) ??? I 0 .

Scale-space DoG repeatedly convolves input images with the same Gaussian kernels, and produces difference-of-Gaussian images by subtracting adjacent image scales.

Scale-space extrema (maxima and minima) are detected in DoG images by comparing a pixel to its 26 neighbors in 3??3 grids at current and two adjacent scales (Lowe, 2004) .

Adversarial Attacks.

We use h(??) to denote the softmax output of classification networks, and h c (??) to denote the prediction probability of class c. Then given a classifier h(x) = y, the goal of adversarial attack is to find x adv such that the output of classifier deviates from the true label y:

Attack Method.

The most simple adversarial attack method is Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2015) , a single-step method which takes the sign of the gradient on the input as the direction of the perturbation.

L(??, ??) denotes the loss function defined by cross entropy.

Specifically, the formation is as follows:

where x is the clean input, y is the label.

is the norm bound (||x ??? x adv || ??? , i.e. -ball) of the adversarial perturbation.

Projected gradient descent (PGD) iteratively applies FGSM with a small step size ?? i (Kurakin et al., 2017a; Madry et al., 2018) with formulation as below:

where i is the iteration number, ?? = /T with T being the number of iterations. '

Proj' is the function to project the image back to -ball every step.

Some advanced and complex attacks are further introduced in DeepFool (Moosavi-Dezfooli et al., 2016), CW (Carlini & Wagner, 2017) , MI-FGSM .

Adversarial Training aims to inject adversarial examples into training procedure so that the trained networks can learn to classify adversarial examples correctly.

Specifically, adversarial training solves the following empirical risk minimization problem:

where A(x) denotes the area around x bounded by L ??? /L 2 norm , and H is the hypothesis space.

In this work, we employ both FGSM and PGD to generate adversarial examples for adversarial training.

Inspired by traditional robust visual feature SIFT, this paper aims to improve model accuracy and robustness by introducing three novel network architecture components to mimic some key components in SIFT: parametric DoG (pDoG), truncated ReLU (tReLU), and projected normalization layer (PNL).

Combining pDoG and tReLU constructs the so-called extreme value preserving convolution (EVPConv) block as shown in Figure 2 (a), which can be used to replace all k ?? k (k > 1) conv-layers in existing CNNs.

PNL is a new and robust layer plugged in to replace global average pooling (GAP) layer as shown in Figure 2 (c).

A network with all the three components is named as extreme value preserving network (EVPNet).

In the following, we will describe these three components in details separately, and elaborate on how they are used to construct the EVPConv block and EVPNet.

Parametric DoG (pDoG) is a network component we design to mimic DoG operation.

Recall DoG in Equation 2, it repeatedly convolves input images with the same Gaussian kernel in which kernel size ?? is designable, and then computes the differences for adjacent Gaussian blurred images.

For CNNs, we mimic DoG with two considerations.

First, we replace the Gaussian kernel with a learnable convolutional filter.

Specifically, we treat each channel of feature map separately as one image, and convolve it with a learnable k ?? k kernel to mimic Gaussian convolution.

Note that the learnable convolution kernel is not required to be symmetric since some recent evidence shows that non-symmetric DoG may perform even better (Einevoll & Plesser, 2012; Winnem??ller, 2011) .

Applying the procedure to all the feature-map channels is equal to a depth-wise (DW) convolution .

Second, we enforce successive depth-wise convolutions in the same block with shared weights since traditional DoG operation uses the same Gaussian kernel.

As CNNs produce full scale information at different stages with a series of convolution and downsampling layers, each pDoG block just focuses on producing extrema for current scale, while not requiring to produce full octave extrema like SIFT.

The shared DW convolution introduces minimum parameter overhead, and avoid "extrema drift" in the pDoG space so that it may help finding accurate local extrema.

Formally, given input feature map f 0 , a minimum of two successive depth-wise convolution is applied as

where DW (; ) is depth-wise convolution with w as the shared weights.

pDoG is thus computed as

It is worth noting that the successive minus operations make the minus sign not able to be absorbed into w for replacing minus into addition operation.

To the best of our knowledge, this is the first time, minus component has been introduced into deep neural networks, which brings totally new element for architecture design/search.

Following SIFT, we compute local extrema (maxima and minimal) across the pDoG images using maxout operations (Goodfellow et al., 2013) :

Note we do not compute local extrema in 3 ?? 3 spatial grids as in SIFT since we do not require keypoint localization in CNNs.

Finally, to keep the module compatible to existing networks, we need ensure the output feature map to be of the same size (number of channels and resolution).

Therefore, a maxout operation is taken over to merge two feature maps and obtain the final output of this block:

Truncated ReLU (tReLU).

The pDoG block keeps all the local extrema in the DoG space, while many local extrema are unstable because of small noise and even contrast changes.

SIFT adopts a local structure fitting procedure to reject those unstable local extrema.

To realize similar effect, we propose truncated ReLU (tReLU) to suppress non-robust local extrema.

The basic idea is to truncate small extrema which correspond to noise and non-stable extrema in the pDoG space.

This can be implemented by modifying the commonly used ReLU function as

where ?? is a learnable truncated parameter.

Note that this function is discontinued at x = ?? and x = ?????.

We make a small modification to obtain a continuous version for easy training as below

Figure 2(b) plots the tReLU function.

Different from the original ReLU, tReLU introduces a threshold parameter ?? and keeps elements with higher magnitude.

?? can be either a block-level parameter (each block has one global threshold) or a channel-level parameter (each channel holds a separate threshold).

By default, we take ?? as a block-level parameter.

tReLU is combined with pDoG not only to suppress non-robust extrema, but also to simplify the operations.

When combining Equation 8 and Equation 9 together, there is nested maxout operation which satisfies commutative law, so that we could rewrite z 0 and z 1 as

where | ?? | is element-wise absolute operation.

With tReLU to suppress non-robust features, we have

Hence, in practice, we use Equation 13 instead of Equation 8 to compute z 0 and z 1 .

Note that tReLU does improve robustness and accuracy for pDoG feature maps, while providing no benefits when replacing ReLU in standard CNNs according to our experiments (see Table 1 ).

Projected Normalization Layer (PNL).

SIFT (Lowe, 2004) computes gradient orientation histogram followed by L2 normalization to obtain final feature representation.

This process does not take gradient pixel relationship into account.

PCA-SIFT (Ke et al., 2004) handles this issue by projecting each local gradient patch into a pre-computed eigen-space using PCA.

We borrow the idea from PCA-SIFT to build projected normalization layer (PNL) to replace global average pooling (GAP) based feature generation in existing CNNs.

Suppose the feature-map data matrix before GAP is X ??? R d??c , where d = w ?? h corresponds to feature map resolution, and c is the number of channels, we obtain column vectors {x i ??? R c } d i=1 from X to represent the i-th pixel values from all channels.

The PNL contains three steps:

(1) We add a 1 ?? 1 conv-layer, which can be viewed as a PCA with learnable projection matrix W ??? R c??p .

The output is u i = W T x i , where u i ??? R p further forms a data matrix U ??? R d??p .

(2) We compute L2 norm for row vectors

(3) To eliminate contrast or scale impact, we normalize v to obtain??? = v/ v p , while ?? p means the p norm.

the normalized vector??? is fed into classification layer for prediction purpose.

It is interesting to note that PNL actually computes a second order pooling similar as (Gao et al., 2019; Yu & Salzmann, 2018) .

Suppose w j ??? R c is the j-th row of W, v j in step-2 can be rewritten as

where

is an auto-correlation matrix.

Figure 2 (c) illustrates the PNL layer.

Theoretically, GAP produces a hyper-cube, while PNL produces a hyper-ball.

This is beneficial for robustness since hyper-ball is more smooth, and a smoothed surface is proven more robust (Cohen et al., 2019) .

Our experiments also verify this point (see Table 1 ).

With these three novel components, we can derive a novel convolution block named EVPConv, and the corresponding networks EVPNet.

In details, EVPConv starts from the pDoG component, and replaces Equation 8 with tReLU as in Equation 13.

In SIFT, the contribution of each pixel is weighted by the gradient magnitude.

This idea can be extended to calibrate contributions from each feature-map channel.

Fortunately, Squeeze-and-Excitation (SE) module proposed in provides the desired capability.

We thus insert the SE block after tReLU, and compute the output of EVPConv as:

where concat(??) means concatenating z 0 and z 1 together for a unified and unbiased calibration, SE(??) is the SE module, s 0 and s 1 are the calibration results corresponding to z 0 and z 1 , and max denotes an element-wise maximum operation.

Figure 2 (a) illustrates the overall structure of EVPConv.

EVPConv can be plugged to replace any k ?? k (k >1) conv-layers in existing CNNs, while the PNL layer can be plugged to replace the GAP layer for feature abstraction.

The network consisting of both EVPConv block and the PNL layer is named as EVPNet.

The EVPConv block introduces very few additional parameters: w for shared depth-wise convolution, ?? for tReLU and parameters for SE module.

Note that we allow each EVPConv block having its own w and ??.

EVPConv brings relatively fewer additional parameters, which is about 7???20% (see Appendix A) (smaller models more relative increasing).

It also increases theoretic computing cost 3???10% for a bunch of parameterfree operations like DoG and maxout.

However, the added computing cost is non-negligible in practice (2?? slower according to our training experiments) due to more memory cost for additional copy of feature-maps.

Near memory computing architecture (Singh et al., 2019) may provide efficient support for this new computing paradigm.

Experimental Setup.

We evaluate the proposed network components and EVPNet on CIFAR10 and SVHN datasets.

CIFAR-10 is a widely used image classification dataset containing 60, 000 images of size 32??32 with 50, 000 for training and 10,000 for testing.

SVHN (Netzer et al., 2011 ) is a digit recognition dataset containing 73,257 training images, 26,032 test images, all with size 32??32.

We introduce our novel components into the well-known and widely used ResNet, and compare to the basic model on both clean accuracy and adversarial robustness.

As the EVPConv block contains a SE module, to make a fair comparison, we set SE-ResNet as our comparison target.

In details, we replace the input conv-layer and the first 3 ?? 3 conv-layer in the residual block with EVPConv, and replace the GAP layer with the proposed PNL layer.

Following (He et al., 2016; Huang et al., 2017) , for CIFAR-10, all the networks are trained with SGD using momentum 0.9, 160 epochs in total.

The initial learning rate is 0.1, divided by 10 at 80 and 120 epochs.

For SVHN, we use the same network architecture as CIFAR-10.

The models are trained for 80 epochs, with initial learning rate 0.1, divided by 10 at 40 and 60 epochs.

For tReLU, the channel-level parameter ?? is initialized by uniformly sampling from [0, 1].

In this work, we consider adversarial perturbations constrained under l ??? norm.

The allowed perturbation norm is 8 pixels (Madry et al., 2018) .

We evaluate non-targeted attack adversarial robustness in three settings: normal training, FGSM adversarial training (Goodfellow et al., 2015; Tramer et al., 2018) and PGD adversarial training (Madry et al., 2018) .

During adversarial training, we use the predicted label to generate adversarial examples to prevent label leaking effect (Kurakin et al., 2017b) .

To avoid gradient masking (Tramer et al., 2018) , we use R-FGSM for FGSM adversarial training, which basically starts from a random point in the ball.

Following Madry et al. (2018) , during training, PGD attacks generate adversarial examples by 7 PGD iterations with 2-pixel step size starting from random points in the allowed ball.

We report accuracy on both whitebox and blackbox attack.

We evaluate a set of well-known whitebox attacks, including FGSM, PGD, DeepFool, CW.

We use 'PGD-N ' to denote attack with N PGD iterations of step size 2 pixels by default.

Specifically, we compare results for PGD-10 and PGD-40.

For blackbox attack, we choose VGG-16 as the source model which is found by to exhibit high adversarial transferability, and choose FGSM as the method to generate adversarial examples from VGG-16 as it is shown to lead to better transferability .

Ablation Study.

This part conducts a thorough ablation study to show the effectiveness of each novel architecture component and how they interact to provide strong adversarial robustness.

We conduct experiments on CIFAR-10 with SE-ResNet-20, which contains one input conv-layer, three residual stage each with two bottleneck residual blocks, and GAP layer followed by a classification layer.

We evaluate the accuracy and robustness for all the possible combinations of the proposed three components under the normal training setting.

For PGD attack, we use two step sizes: 1 pixel as in and 2 pixels as in Madry et al. (2018) .

Table 1 lists full evaluation results of each tested model.

Several observations can be made from the table:

(1) Solely adding pDoG or PNL layer leads to significant robustness improvement.

pDoG even yields clean accuracy improvement, while PNL yields slightly clean accuracy drops.

(2) tReLU does not bring benefit for standard convolution, while yields notable improvement on both clean accuracy and adversarial accuracy, when combining with pDoG. That verifies our previous claim that tReLU is suitable to work for the DoG space.

(3) Combining all the three components together obtains the best adversarial robustness, while still achieve 1.2% clean accuracy improvement over the model without these three components.

Based on these observations, we incorporate all the three components into the CNNs to obtain the so-called EVPNet for the following experiments if not explicitly specified.

As 2-pixels PGD attack is much stronger than 1-pixel PGD attack, we use it as default in the following studies.

Benchmark Results.

We conduct extensive experiments on CIFAR-10 and SVHN to compare the proposed EVPNet with the source networks.

The two sources networks are For fair comparison, we use the SE extended ResNet as our baseline.

Table 2 lists comprehensive comparison results on CIFAR-10.

We list 7 different kinds of accuracies: clean model accuracy, whilebox attack accuracies by FGSM/PGD-10/PGD-40/DeepFool/CW, and blackbox attack accuracy with adversarial examples generated by FGSM on the VGG-16 model.

We can see that under normal training case, EVPNet outperforms baseline by a large margin in terms of robustness with FGSM, PGD, DeepFool, and CW attacks.

Even under the strongest PGD-40 white box attack, our EVPNet still has non-zero accuracy without any adversarial training.

For those cases with adversarial training, our EVPNet consistently beats baseline networks with noticeable margin.

training performs worse on PGD-10/PGD-40 attacks than FGSM adversarial training, and even much worse than normal training on this dataset.

This may be due to the fact that SVHN is a shape/edge dominating digit recognition dataset, which may generate a lot of difficult adversarial samples with broken edges.

And it also coincides with the finding by Baker et al. (2018) .

Our EVPNet shows better robustness on this dataset without adversarial training than CIFAR-10, which may suggest that EVPNet is more robust on shape/edge dominating object instances.

All these evidences prove that the proposed EVPNet is a robust network architecture.

Analysis.

We make some further analysis to compare EVPNet to baseline networks.

First, we plot the test error at different PGD iterations for different evaluated networks under normal training case on both CIFAR-10 and SVHN datasets as shown in Figure 3 .

It can be seen that EVPNet consistently performs significantly better than the corresponding baseline networks under all PGD iterations.

Some may concern that the accuracy of EVPNet on the strongest PGD-40 attack is not satisfied (??? 10%).

We argue that from three aspects: (1) The adversarial result is remarkable as it is by the clean model without using any other tricks like adversarial training, adversarial loss, etc.

(2) The proposed components also brings consistent clean accuracy improvement even on large-scale dataset (see Appendix A).

(3) More importantly, the methodology we developed may shed some light on future studies in network robustness and network architecture design/search.

Second, we further investigate the error amplification effect as .

Specifically, we feed both benign examples and adversarial examples into the evaluated networks, and compute the normalized L 2 distance for each res-block outputs as ?? = x ??? x 2 / x 2 , where x is the response vector of benign example, and x is the response vector for the adversarial example.

We randomly sample 64 images from the test set to generate adversarial examples using PGD-40.

The models evaluated are trained without adversarial training.

Figure 4 illustrates the results.

As we can see, EVPNet has much lower average normalized distance than the baseline models almost on all the blocks.

It is interesting to see that the baseline models have a big jump for the normalized distance at the end of the networks on all the 4 sub-figures.

This urges the adversarial learning researchers to make further investigation on the robustness especially for latter layers around GAP.

Nevertheless, this analysis demonstrates that EVPNet significantly reduces the error amplification effect.

Third, we compare the differences on feature responses between regular convolution + ReLU and EVPConv.

This comparison is made on large-scale and relative high resolution (224 ?? 224) ImageNet dataset for better illustration.

We train ResNet-50 and EVPNet-50 on ImageNet, and visualize their prediction responses for the first corresponding convolution block in Figure 1 .

It clearly shows that ResNet-50 has more noise responses, while EVPNet-50 gives more responses on object boundary.

This demonstrates the capability of EVPConv to separate robust features from non-robust ones.

Full benchmark results on ImageNet are also very promising, see Appendix A for more details.

This paper mimics good properties of robust visual feature SIFT to renovate CNN architectures with some novel architecture components, and proposes the extreme value preserving networks (EVPNet).

Experiments demonstrate that EVPNets can achieve similar or better accuracy over conventional CNNs, while achieving much better robustness to a set of adversarial attacks (FGSM, PGD, etc) even for clean model without any other tricks like adversarial training.

top-1 accuracy to near zero, while the EVP-ResNet variants keep 6???10% top-1 accuracy.

The gap in FGSM attacks is even larger.

This improvement is remarkable considering that it is by clean model without adversarial training.

For the MobileNet case, we also observe notable accuracy and robustness improvement.

Please refer to Table 4 for more details.

In summary, our solid results and attempts may inspire future new ways for robust network architecture design or even automatic search.

In the main paper, we demonstrate the great robustness of the proposed components on ResNet with bottleneck residual blocks.

Here, we extend the proposed components to other state-of-the-art network architectures, and choose Wide-ResNet (Zagoruyko & komodakis, 2017) as an example since it is mostly studied on other adversarial training works (Madry et al., 2018; Cisse et al., 2017; Athalye et al., 2018) .

Wide-ResNet (WRN) has two successive wide-channel 3 ?? 3 conv-layers in residual block instead of the three conv-layer bottleneck structure based residual block.

We use WRN-22-8 as the baseline network with depth 22 and widening factor 8.

It is obvious that WRN-22-8 has much better clean accuracy than ResNet-20 and ResNet-56 used in the main paper.

For EVPNet, We replace input conv-layer and the first 3 ?? 3 conv-layer in each wide residual block with EVPConv, and replace the GAP layer with our PNL layer.

Table 5 shows the comparison on normal training case.

We can see that EVPNet achieves similar clean accuracy, while performing significantly better on adversarial attacks with FGSM/PGD-10/PGD-40.

Note that under PGD-10 and PGD-40 attacks, the baseline model drops accuracy to near 0, while the EVPNet remains a much higher accuracy, considering that no adversarial training is utilized in this study.

This demonstrates the strong robustness of the proposed EVPNet.

<|TLDR|>

@highlight

This paper aims to leverage good properties of robust visual features like SIFT to renovate CNN architectures towards better accuracy and robustness.