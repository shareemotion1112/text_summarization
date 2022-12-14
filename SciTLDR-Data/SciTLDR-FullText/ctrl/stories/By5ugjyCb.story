Deep learning algorithms achieve high classification accuracy at the expense of significant computation cost.

To address this cost, a number of quantization schemeshave been proposed - but most of these techniques focused on quantizing weights, which are relatively smaller in size compared to activations.

This paper proposes a novel quantization scheme for activations during training - that enables neural networks to work well with ultra low precision weights and activations without any significant accuracy degradation.

This technique, PArameterized Clipping acTi-vation (PACT), uses an activation clipping parameter α that is optimized duringtraining to find the right quantization scale.

PACT allows quantizing activations toarbitrary bit precisions, while achieving much better accuracy relative to publishedstate-of-the-art quantization schemes.

We show, for the first time, that both weights and activations can be quantized to 4-bits of precision while still achieving accuracy comparable to full precision networks across a range of popular models and datasets.

We also show that exploiting these reduced-precision computational units in hardware can enable a super-linear improvement in inferencing performance dueto a significant reduction in the area of accelerator compute engines coupled with the ability to retain the quantized model and activation data in on-chip memories.

Deep Convolutional Neural Networks (CNNs) have achieved remarkable accuracy for tasks in a wide range of application domains including image processing (He et al. (2016b) ), machine translation (Gehring et al. (2017) ), and speech recognition (Zhang et al. (2017) ).

These state-of-the-art CNNs use very deep models, consuming 100s of ExaOps of computation during training and GBs of storage for model and data.

This poses a tremendous challenge to widespread deployment, especially in resource constrained edge environments -leading to a plethora of explorations in compressed models that minimize memory footprint and computation while preserving model accuracy as much as possible.

Recently, a whole host of different techniques have been proposed to alleviate these computational costs.

Among them, reducing the bit-precision of key CNN data structures, namely weights and activations, has gained attention due to its potential to significantly reduce both storage requirements and computational complexity.

In particular, several weight quantization techniques (Li & Liu (2016) and Zhu et al. (2017) ) showed significant reduction in the bit-precision of CNN weights with limited accuracy degradation.

However, prior work (Hubara et al. (2016b) ; Zhou et al. (2016) ) has shown that a straightforward extension of weight quantization schemes to activations incurs significant accuracy degradation in large-scale image classification tasks such as ImageNet (Russakovsky et al. (2015) ).

Recently, activation quantization schemes based on greedy layer-wise optimization were proposed (Park et al. (2017) ; Graham (2017) ; Cai et al. (2017) ), but achieve limited accuracy improvement.

In this paper, we propose a novel activation quantization technique, PArameterized Clipping acTivation function (PACT) , that automatically optimizes the quantization scales during model training.

PACT allows significant reductions in the bit-widths needed to represent both weights and activations and opens up new opportunities for trading off hardware complexity with model accuracy.

The primary contributions of this work include: 1) PACT: A new activation quantization scheme for finding the optimal quantization scale during training.

We introduce a new parameter α that is used to represent the clipping level in the activation function and is learnt via back-propagation.

α sets the quantization scale smaller than ReLU to reduce the quantization error, but larger than a conventional clipping activation function (used in previous schemes) to allow gradients to flow more effectively.

In addition, regularization is applied to α in the loss function to enable faster convergence.

We provide reasoning and analysis on the expected effectiveness of PACT in preserving model accuracy.3) Quantitative results demonstrating the effectiveness of PACT on a spectrum of models and datasets.

Empirically, we show that: (a) for extremely low bit-precision (≤ 2-bits for weights and activations), PACT achieves the highest model accuracy compared to all published schemes and (b) 4-bit quantized CNNs based on PACT achieve accuracies similar to single-precision floating point representations.4) System performance analysis to demonstrate the trade-offs in hardware complexity for different bit representations vs. model accuracy.

We show that a dramatic reduction in the area of the computing engines is possible and use it to estimate the achievable system-level performance gains.

The rest of the paper is organized as follows: Section 2 provides a summary of related prior work on quantized CNNs.

Challenges in activation quantization are presented in Section 3.

We present PACT, our proposed solution for activation quantization in Section 4.

In Section 5 we demonstrate the effectiveness of PACT relative to prior schemes using experimental results on popular CNNs.

Overall system performance analysis for a representative hardware system is presented in Section 6 demonstrating the observed trade-offs in hardware complexity for different bit representations.

Recently, a whole host of different techniques have been proposed to minimize CNN computation and storage costs.

One of the earliest studies in weight quantization schemes (Hwang & Sung (2014) and Courbariaux et al. (2015) ) show that it is indeed possible to quantize weights to 1-bit (binary) or 2-bits (ternary), enabling an entire DNN model to fit effectively in resource-constrained platforms (e.g., mobile devices).

Effectiveness of weight quantization techniques has been further improved (Li & Liu (2016) and Zhu et al. (2017) ), by ternarizing weights using statistical distribution of weight values or by tuning quantization scales during training.

However, gain in system performance is limited when only weights are quantized while activations are left in high precision.

This is particularly severe in convolutional neural networks (CNNs) since weights are relatively smaller in convolution layers in comparison to fully-connected (FC) layers.

To reduce the overhead of activations, prior work (Kim & Smaragdis (2015) , Hubara et al. (2016a), and Rastegari et al. (2016) ) proposed the use of fully binarized neural networks where activations are quantized using 1-bit as well.

More recently, activation quantization schemes using more general selections in bit-precision (Hubara et al. (2016b); Zhou et al. (2016; 2017); Mishra et al. (2017); Mellempudi et al. (2017) ) have been studied.

However, these techniques show significant degradation in accuracy (> 1%) for ImageNet tasks (Russakovsky et al. (2015) ) when bit precision is reduced significantly (≤ 2 − bits).

Improvements to previous logarithmic quantization schemes (Miyashita et al. (2016) ) using modified base and offset based on "weighted entropy" of activations have also been studied (Park et al. (2017) ).

Graham (2017) recommends that normalized activation, in the process of batch normalization (Ioffe & Szegedy (2015) , BatchNorm), is a good candidate for quantization.

Cai et al. (2017) further exploits the statistics of activations and proposes variants of the ReLU activation function for better quantization.

However, such schemes typically rely on local (and greedy) optimizations, and are therefore not adaptable or optimized effectively during training.

This is further elaborated in Section 3 where we present a detailed discussion on the challenges in quantizing activations.

Quantization of weights is equivalent to discretizing the hypothesis space of the loss function with respect to the weight variables.

Therefore, it is indeed possible to compensate weight quantization errors during model training (Hwang & Sung, 2014; Courbariaux et al., 2015) .

Traditional activation Figure 1 : (a) Training error, (b) Validation error across epochs for different activation functions (relu and clipping) with and without quantization for the ResNet20 model using the CIFAR10 dataset functions, on the other hand, do not have any trainable parameters, and therefore the errors arising from quantizing activations cannot be directly compensated using back-propagation.

Activation quantization becomes even more challenging when ReLU (the activation function most commonly used in CNNs) is used as the layer activation function (ActFn).

ReLU allows gradient of activations to propagate through deep layers and therefore achieves superior accuracy relative to other activation functions (Nair & Hinton (2010) ).

However, as the output of the ReLU function is unbounded, the quantization after ReLU requires a high dynamic range (i.e., more bit-precision).

In Fig. 1 we present the training and validation errors of ResNet20 with the CIFAR10 dataset using ReLU and show that accuracy is significantly degraded with ReLU quantizations It has been shown that this dynamic range problem can be alleviated by using a clipping activation function, which places an upper-bound on the output (Hubara et al. (2016b); Zhou et al. (2016) ).

However, because of layer to layer and model to model differences -it is difficult to determine a globally optimal clipping value.

In addition, as shown in Fig 1, even though the training error obtained using clipping with quantization is less than that obtained with quantized ReLU, the validation error is still noticeably higher than the baseline.

Recently, this challenge has been partially addressed by applying a half-wave Gaussian quantization scheme to activations (Cai et al. (2017) ).

Based on the observation that activation after BatchNorm normalization is close to a Gaussian distribution with zero mean and unit variance, they used Lloyd's algorithm to find the optimal quantization scale for this Gaussian distribution and use that scale for every layer.

However, this technique also does not fully utilize the strength of backpropagation to optimally learn the clipping level because all the quantization parameters are determined offline and remain fixed throughout the training process.

Building on these insights, we introduce PACT, a new activation quantization scheme in which the ActFn has a parameterized clipping level, α.

α is dynamically adjusted via gradient descent-based training with the objective of minimizing the accuracy degradation arising from quantization.

In PACT, the conventional ReLU activation function in CNNs is replaced with the following: DISPLAYFORM0 where α limits the range of activation to [0, α] .

The truncated activation output is then linearly quantized to k bits for the dot-product computations, where DISPLAYFORM1 With this new activation function, α is a variable in the loss function, whose value can be optimized during training.

For back-propagation, gradient ∂yq ∂α can be computed using the Straight-Through Estimator (STE) (Bengio et al. (2013) ) to estimate ∂yq ∂y as 1.

Thus, DISPLAYFORM2 The larger the α, the more the parameterized clipping function resembles a ReLU Actfn.

To avoid large quantization errors due to a wide dynamic range, we include a L2-regularizer for α in the loss function.

FIG6 illustrates how the value of α changes during full-precision training of CIFAR10-ResNet20 starting with an initial value of 10 and using the L2-regularizer.

It can be observed that α converges to values much smaller than the initial value as the training epochs proceed, thereby limiting the dynamic range of activations and minimizing quantization loss.

To provide further reasoning on why PACT works, we provide in-depth analysis in Appendix A and B. In particular, we show in Appendix A that PACT is as expressive as ReLU when it is used as an activation function.

Further we explain in Appendix B that PACT finds a balancing point between clipping and quantization errors to minimize their impact to classification accuracy.

When activation is quantized, the overall behavior of network parameters is affected by the quantization error during training.

To observe the impact of activation quantization during network training, we sweep the clipping parameter α and record the training loss with and without quantization.

Figs. 3 a,b and 3c show cross-entropy and training loss (cross entropy + regularization), respectively, over a range of α for the pre-trained SVHN network.

The loaded network is trained with the proposed quantization scheme in which ReLU is replaced with the proposed parameterized clipping ActFn for each of its seven convolution layers.

We sweep the value of α one layer at a time, keeping all other parameters (weight (W ), bias (b), BatchNorm parameters (β, γ), and the α of other layers) fixed when computing the cross-entropy and training loss.

The cross-entropy computed via full-precision forward-pass of training is shown in FIG6 .

In this case, the cross-entropy converges to a small value in many layers as α increases, indicating that ReLU is a good activation function when no quantization is applied.

But even for the full-precision case, training clipping parameter α may help reduce the cross-entropy for certain layers; for example, ReLU (i.e., α = ∞) is not optimal for act0 and act6 layers.

Next, the cross-entropy computed with quantization in the forward-pass is shown in FIG1 .

With quantization, the cross-entropy increases in most cases as α increases, implying that ReLU is no longer effective.

We also observe that the optimal α has different ranges for different layers, motivating the need to "learn" the quantization scale via training.

In addition, we observe plateaus of cross-entropy for the certain ranges of α (e.g., act6), leading to difficulties for gradient descent-based training.

Finally, in FIG1 , we show the total training loss including both the cross-entropy discussed above and the cost from α regularization.

The regularization effectively gets rid of the plateaus in the training loss, thereby favoring convergence for gradient-descent based training.

At the same time, α regularization does not perturb the global minimum point.

For example, the solid circles in FIG1 , which are the optimal α extracted from the pre-trained model, are at the minimum of the training loss curves.

The regularization coefficient, λ α , discussed in the next section, is an additional hyper-parameter which controls the impact of regularization on α.

For this new quantization approach, we studied the scope of α, the choice of initial values of α,and the impact of regularizing α.

We briefly summarize our findings below, and present more detailed analysis in Appendix C.From our experiments, the best scope for α was to share α per layer.

This choice also reduces hardware complexity because α needs to be multiplied only once after all multiply-accumulate (MAC) operations in reduced-precision in a layer are completed.

Among initialization choices for α, we found it to be advantageous to initialize α to a larger value relative to typical values of activation, and then apply regularization to reduce it during training.

Finally, we observed that applying L2-regularization for α with the same regularization parameter λ used for weight works reasonably well.

We also observed that, as expected, the optimal value for λ α slightly decreases when higher bit-precision is used because more quantization levels result in higher resolution for activation quantization.

Additionally, we follow the practice of many other quantized CNN studies (e.g., Hubara et al. (2016b) ; Zhou et al. FORMULA0 ), and do not quantize the first and last layers, as these have been reported to significantly impact accuracy.

We implemented PACT in Tensorflow BID0 ) using Tensorpack (Zhou et al. FORMULA0 ).

To demonstrate the effectiveness of PACT, we studied several well-known CNNs.

The following is a summary of the Dataset-Network for the tested CNNs.

More implementation details can be found in Appendix.

D. Note that the baseline networks use the same hyper-parameters and ReLU activation functions as described in the references.

For PACT experiments, we only replace ReLU into PACT but the same hyper-parameters are used.

All the time the networks are trained from scratch.• CIFAR10-ResNet20 (CIFAR10, Krizhevsky & Hinton (2010) ): a convolution (CONV) layer followed by 3 ResNet blocks (16 CONV layers with 3x3 filter) and a final fully-connected (FC) layer.• SVHN-SVHN (SVHN, Netzer et al. FORMULA0 ): 7 CONV layers followed by 1 FC layer.• IMAGENET-AlexNet (AlexNet, Krizhevsky et al. FORMULA0 ): 5 parallel-CONV layers followed by 3 FC layers.

BatchNorm is used before ReLU.• IMAGENET-ResNet18 (ResNet18, He et al. FORMULA0 ): a CONV layer followed by 8 ResNet blocks (16 CONV layers with 3x3 filter) and a final FC layer.

"full pre-activation" ResNet structure (He et al. (2016a) ) is employed.• IMAGENET-ResNet50 (ResNet50, He et al. FORMULA0 ): a CONV layer followed by 16 ResNet "bottleneck" blocks (total 48 CONV layers) and a final FC layer.

"full pre-activation" ResNet structure (He et al. FORMULA0 ) is employed.

For comparisons, we include accuracy results reported in the following prior work: DoReFa (Zhou et al. FORMULA0 FORMULA0 ).

Detailed experimental setting for each of these papers, as well as full comparison of accuracy (top-1 and top5) for AlexNet, ResNet18, ResNet50, can be found in Appendix E. In the following section, we present key results demonstrating the effectiveness of PACT relative to prior work.

We first evaluate our activation quantization scheme using various CNNs.

FIG3 training and validation error of PACT for the tested CNNs.

Overall, the higher the bit-precision, the closer the training/validation errors are to the full-precision reference.

Specifically it can be seen that training using bit-precision higher than 3-bits converges almost identically to the full-precision baseline.

The final validation error has less than 1% difference relative to the full-precision validation error for all cases when the activation bit-precision is at least 4-bits.

We further compare activation quantization performance with 3 previous schemes, DoReFa, LPBN, and HWGQ.

We use accuracy degradation as the quantization performance metric, which is calculated as the difference between full-precision accuracy and the accuracy for each quantization bit-precision.

FIG3 shows accuracy degradation (top-1) for ResNet18 (left) and ResNet50 (right) for increasing activation bit-precision, when the same weight bit-precision is used for each quantization scheme (indicated within the parenthesis).

Overall, we observe that accuracy degradation is reduced as we increase the bit-precision of activations.

For both ResNet18 and ResNet50, PACT achieves consistently lower accuracy degradation compared to the other quantization schemes, demonstrating the robustness of PACT relative to prior quantization approaches.

In this section, we demonstrate that although PACT targets activation quantization, it does not preclude us from using weight quantization as well.

We used PACT to quantize activation of CNNs, and DoReFa scheme to quantize weights.

TAB0 summarizes top-1 accuracy of PACT for the tested CNNs (CIFAR10, SVHN, AlexNet, ResNet18, and ResNet50).

We also show the accuracy of CNNs when both the weight and activation are quantized by DoReFa's scheme.

As can be seen, with 4 bit precision for both weights and activation, PACT achieves full-precision accuracy consistently across the networks tested.

To the best of our knowledge, this is the lowest bit precision for both weights and activation ever reported, that can achieve near (≤ 1%) full-precision accuracy.

We further compare the performance of PACT-based quantized CNNs with 7 previous quantization schemes (DoReFa, BalancedQ, WRPN, FGQ, WEP, LPBN, and HWGQ).

Fig. 5 shows comparison of accuracy degradation (top-1) for AlexNet, ResNet18, and ResNet50.

Overall, the accuracy degradation decreases as bit-precision for activation or weight increases.

For example, in Fig. 5a , the accuracy degradation decreases when activation bit-precision increases given the same weight precision or when weight bit-precision increases given the same activation bit-precision.

PACT outperforms other schemes for all the cases.

In fact, AlexNet even achieves marginally better accuracy (i.e., negative accuracy degradation) using PACT instead of full-precision.

In this section, we demonstrate the gain in system performance as a result of the reduction in bit-precision achieved using PACT-CNN.

To this end, as shown in Fig. 6(a) , we consider a DNN accelerator system comprising of a DNN accelerator chip, comprising of multiple cores, interfaced with an external memory.

Each core consists of a 2D-systolic array of fixed-point multiply-andaccumulate (MAC) processing elements on which DNN layers are executed.

Each core also contains an on-chip memory, which stores the operands that are fed into the MAC processing array.

To estimate system performance at different bit precisions, we studied different versions of the DNN accelerator each comprising the same amount of on-chip memory, external memory bandwidth, and occupying iso-silicon area.

First, using real hardware implementations in a state of the art technology (14 nm CMOS), we accurately estimate the reduction in the MAC area achieved by aggressively scaling bit precision.

As shown in Fig. 6(b) , we achieve ∼14× improvement in density when the bit-precisions of both activations and weights are uniformly reduced from 16 bits to 2 bits.

Next, to translate the reduction in area to improvement in overall performance, we built a precisionconfigurable MAC unit, whose bit precision can be modulated dynamically.

The peak compute capability (FLOPs) of the MAC unit varied such that we achieve iso-area at each precision.

Note that the total on-chip memory and external bandwidth remains constant at all precisions.

We estimate the overall system performance using DeepMatrix, a detailed performance modelling framework for DNN accelerators (Venkataramani et al.) .

Fig. 6(c) shows the gain in inference performance for the ResNet50 DNN benchmark.

We study the performance improvement using different external memory bandwidths, namely, a bandwidth unconstrained system (infinite memory bandwidth) and two bandwidth constrained systems at 32 and 64 GBps.

In the bandwidth unconstrained scenario, the gain in performance is limited by how amenable it is to parallelize the work.

In this case, we see a near-linear increase in performance for upto 4 bits and a small drop at extreme quantization levels (2 bits).Practical systems, whose bandwidths are constrained, (surprisingly) exhibit a super-linear growth in performance with quantization.

For example, when external bandwidth is limited to 64 GBps, quantizing from 16 to 4 bits leads to a 4× increase in peak FLOPs but a 4.5× improvement in performance.

This is because, the total amount of on-chip memory remains constant, and at very low precision some of the data-structures begin to fit within the memory present in the cores, thereby avoiding data transfers from the external memory.

Consequently, in bandwidth limited systems, reducing the amount of data transferred from off-chip can provide an additional boost in system performance beyond the increase in peak FLOPs.

Note that for the 4 and 2 bit precision configurations, we still used 8 bit precision to execute the first and last layers of the DNN.

If we are able to quantize the first and last layers as well to 4 or 2 bits, we estimate an additional 1.24× improvement in performance, motivating the need to explore ways to quantize the first and last layers.

In this paper, we propose a novel activation quantization scheme based on the PArameterized Clipping acTivation function (PACT).

The proposed scheme replaces ReLU with an activation function with a clipping parameter, α, that is optimized via gradient descent based training.

We provide analysis on why PACT outperforms ReLU when quantization is applied during training.

Extensive empirical evaluation using several popular convolutional neural networks, such as CIFAR10, SVHN, AlexNet, ResNet18 and ResNet50, shows that PACT quantizes activations very effectively while simultaneously allowing weights to be heavily quantized.

In comparison to all previous quantization schemes, we show that both weights and activations can be quantized much more aggressively (down to 4-bits) -while achieving near (≤ 1%) full-precision accuracy.

In addition, we have shown that the area savings from using reduced-precision MAC units enable a dramatic increase in the number of accelerator cores in the same area, thereby, significantly improving overall system-performance.

When used as an activation function of the neural network, PACT is as expressive as ReLU.

This is because clipping parameter, α, introduced in PACT, allows flexibility in adjusting the dynamic range of activation for each layer.

We demonstrate in the simple example below that PACT can reach the same solution as ReLU via SGD.Lemma A.1.

Consider a single-neuron network with PACT; x = w · a, y = P ACT (x), where a is input and w is weight.

This network can be trained with SGD to find the output the network with ReLU would produce.

Proof.

Consider a sample of training data (a, y * ).

For illustration purposes consider mean-squareerror (MSE) as the cost function: DISPLAYFORM0 Therefore, when α is updated by SGD, DISPLAYFORM1 where η is a learning rate.

Note that during this update, the weight is not updated as DISPLAYFORM2 From MSE, ∂L ∂y = (y − y * ).

Therefore, if y * > x, α is increased for each update of (5) until α ≥ x, then the PACT network behaves the same as the ReLU network.

Interestingly, if y * ≤ y or y < y * < x, α is decreased or increased to converge to y * .

Note that in this case, ReLU would pass erroneous output x to increase cost function, which needs to be fixed by updating w with ∂L ∂w .

PACT, on the other hand, ignores this erroneous output by directly adapting the dynamic range to match the target output y * .

In this way, the PACT network can be trained to produce output which converges to the same target that the ReLU network would achieve via SGD.In general cases, ∂L ∂α = i ∂L ∂yi , and PACT considers output of neurons together to change the dynamic range.

There are two options: (1) if output x i is not clipped, then the network is trained via back-propagation of gradient to update weight, (2) if output x i is clipped, then α is increased or decreased based on how close the overall output is to the target.

Hence, there are configurations under which SGD would lead to a solution close to the one which the network with ReLU would achieve.

FIG6 demonstrates that ResNet20 with PACT converges almost identical to the network with ReLU.

In Section 3, when we briefly discussed the challenges in activation quantization, we mentioned that there is a trade-off between errors due to clipping and quantization.

As the clipping level increases, larger range of activation can be passed to the next layer of the neural network causing less clipping error (ErrClip i = max(x i − α, 0)).

However, the increased dynamic range incurs larger quantization error, since its magnitude is proportional to the clipping level ( DISPLAYFORM0 , with k-bit quantization).

This imposes the challenge of finding a proper clipping level to balance between clipping and quantization errors.

This trade-off can be better observed in FIG7 , which shows normalized mean-square-error caused by clipping and quantization during training of the CIFAR10-ResNet20 with different clipping levels.

It can be seen that activation functions with large dynamic range, such as ReLU, would suffer quantization errors whose magnitude increases exponentially as the bit-precision k decreases.

This explains why the network with ReLU fails to converge when the activation is quantized (Fig. 1) .PACT can find a balancing point between clipping and quantization errors.

As explained in Section A, PACT adjusts dynamic range based on how close the output is to the target.

As both clipping and quantization errors distort output far from the target, PACT would increase or decrease the dynamic range during training to minimize both clipping and quantization errors.

FIG7 shows how PACT balances the clipping and quantization errors during training.

CIFAR10-ResNet20 is trained with clipping activation function with varying clipping level α from 1 to 16.

When activation is quantized, the network trained with clipping activation shows significant accuracy degradation as α increases.

This is consistent with the trend in quantization error we observed in FIG7 .

In this case, PACT achieves the best accuracy one of the clipping activation could achieve, but without exhaustively sweeping over different clipping levels.

In other words, PACT auto-tunes the clipping level to achieve best accuracy without incurring significant computation overhead.

PACT's auto-tuning of dynamic range is critical in efficient yet robust training of large scale quantized neural networks, especially because it does not increase the burden for hyper-parameter tuning.

In fact, we used the same hyper-parameters as well as the original network structure for all the models we tested, except replacing ReLU to PACT, when we applied activation quantization.

Without quantization, there is a trend that validation error decreases as α increases.

Surprisingly, some of the cases even outperforms the ReLU network.

In this case, PACT also achieves comparable accuracy as ReLU, confirming its expressivity discussed in Section A.

In this section, we present details on the hyper-parameters and design choices studied for PACT.

One of key questions is the optimal scope for α.

In other words, determining which neuron activations should share the same α.

We considered 3 possible choices: (a) Individual α for each neuron activation, (b) Shared α among neurons within the same output channel, and (c) Shared α within a layer.

We empirically studied each of these choices of α (without quantization) using CIFAR10-ResNet20 and determined training and validation error for PACT.

As shown in FIG8 , sharing α per layer is the best choice in terms of accuracy.

This is in fact a preferred option from the perspective of hardware complexity as well, since α needs to be multiplied only once after all multiply-accumulate(MAC) operations in a layer are completed.

The optimization behavior of α can be explained from the formulation of the parameterized clipping function.

From Eq. 3 it is clear that, if α is initialized to a very small value, more activations fall into the range for the nonzero gradient, leading to unstable α in the early epochs, potentially causing accuracy degradation.

On the other hand, if α is initialized to a very large value, the gradient becomes too small and α may be stuck at a large value, potentially suffering more on quantization error.

Therefore, it is intuitive to start with a reasonably large value to cover a wide dynamic range and avoid unstable adaptation of α, but apply regularizer to reduce the value of α so as to alleviate quantization error.

In practice, we found that applying L2-regularization for α while setting its coefficient λ α the same as the L2-regularization coefficient for weight, λ, works well.

FIG9 shows that validation error for PACT-quantized CIFAR10-ResNet20 does not significantly vary for a wide range of λ α .

We also observed that, as expected, the optimal value for λ α slightly decreases when higher bit-precision is used because more quantization levels result in higher resolution for activation quantization.

FORMULA0 ) follow the convention to keep the first and last layer in full precision during training, since quantizing those layers lead to substantial accuracy degradation.

We empirically studied this for the proposed quantization approach for CIFAR10-ResNet20.

In FIG10 , the only difference among the curves is whether input activation and weight of the first convolution layer or the last fully-connected layer are quantized.

As can be seen from the plots, there can be noticeable accuracy degradation if the first or last layers are aggressively quantized.

But computation in floating point is very expensive in hardware.

Therefore, we further studied the option of quantizing the first and last layers with higher quantization bit-precision than the bit-precision of the other layers.

TAB2 shows that independent of the quantization level for the other layers, there is little accuracy degradation if the first and last layer are quantized with 8-bits.

This motivates us to employ reduced precision computation even for the first/last layers.

In this section, we summarize details of our CNN implementation as well as our training settings, which is based on the default networks provided by Tensorpack (Zhou et al. (2016) ).

Unless mentioned otherwise, ReLU following BatchNorm is used for ActFn of the convolution (CONV) .0 12.9 11.1 10.9 17.4 10.0 9.4 8.9 15.9 9.7 9.2 8.9 18.2 9.0 8.4 8.5 FL/M/NQ 21.3 11.5 11.5 10.7 17.6 9.7 9.2 9.0 16.5 9.7 8.7 8.7 16.3 9.3 8.6 8.5 NQ/M/FL 12.1 11.2 11.0 11.5 9.8 8.9 9.2 9.2 8.4 8.4 8.7 8.8 8.5 9.0 8.5 8.5 layers, and Softmax is used for the fully-connected (FC) layer.

Note that the baseline networks use the same hyper-parameters and ReLU activation functions as described in the references.

For PACT experiments, we only replace ReLU into PACT but the same hyper-parameters are used.

All the time the networks are trained from scratch.

The CIFAR10 dataset (Krizhevsky & Hinton (2010) ) is an image classification benchmark containing 32 × 32 pixel RGB images.

It consists of 50K training and 10K test image sets.

We used the "standard" ResNet structure (He et al. (2016a) ) which consists of a CONV layer followed by 3 ResNet blocks (16 CONV layers with 3x3 filter) and a final FC layer.

We used stochastic gradient descent (SGD) with momentum of 0.9 and learning rate starting from 0.1 and scaled by 0.1 at epoch 60, 120.

L2-regularizer with decay of 0.0002 is applied to weight.

The mini-batch size of 128 is used, and the maximum number of epochs is 200.The SVHN dataset (Netzer et al. (2011) ) is a real-world digit recognition dataset containing photos of house numbers in Google Street View images, where the "cropped" 32 × 32 colored images (resized to 40 × 40 as input to the network) centered around a single character are used.

It consists of 73257 digits for training and 26032 digits for testing.

We used a CNN model which contains 7 CONV layers followed by 1 FC layer.

We used ADAM(Kingma & Ba FORMULA0 ) with epsilon 10 −5 and learning rate starting from 10 −3 and scaled by 0.5 every 50 epoch.

L2-regularizer with decay of 10 −7 is applied to weight.

The mini-batch size of 128 is used, and the maximum number of epochs is 200.The IMAGENET dataset (Russakovsky et al. (2015) ) consists of 1000-categories of objects with over 1.2M training and 50K validation images.

Images are first resized to 256 256 and randomly cropped to 224224 prior to being used as input to the network.

We used a modified AlexNet, ResNet18 and ResNet50.We used AlexNet network (Krizhevsky et al. (2012) ) in which local contrast renormalization (RNorm) layer is replaced with BatchNorm layer.

We used ADAM with epsilon 10 −5 and learning rate starting from 10 −4 and scaled by 0.2 at epoch 56 and 64.

L2-regularizer with decay factor of 5 × 10 −6 is applied to weight.

The mini-batch size of 128 is used, and the maximum number of epochs is 100.ResNet18 consists of a CONV layer followed by 8 ResNet blocks (16 CONV layers with 3x3 filter) and a final FC layer.

"full pre-activation" ResNet structure (He et al. (2016a) ) is employed.

ResNet50 consists of a CONV layer followed by 16 ResNet "bottleneck" blocks (total 48 CONV layers) and a final FC layer.

"full pre-activation" ResNet structure (He et al. (2016a) ) is employed.

For both ResNet18 and ResNet50, we used stochastic gradient descent (SGD) with momentum of 0.9 and learning rate starting from 0.1 and scaled by 0.1 at epoch 30, 60, 85, 95.

L2-regularizer with decay of 10 −4 is applied to weight.

The mini-batch size of 256 is used, and the maximum number of epochs is 110.

• DoReFa-Net (DoReFa, Zhou et al. (2016) ): A general bit-precision uniform quantization schemes for weight, activation, and gradient of DNN training.

We compared the experimental results of DoReFa for CIFAR10, SVHN, AlexNet and ResNet18 under the same experimental setting as PACT.

Note that a clipped absolute activation function is used for SVHN in DoReFa.• Balanced Quantization (BalancedQ, Zhou et al. (2017) ): A quantization scheme based on recursive partitioning of data into balanced bins.

We compared the reported top-1/top-5 validation accuracy of their quantization scheme for AlexNet and ResNet18.• Quantization using Wide Reduced-Precision Networks (WRPN, Mishra et al. (2017) ): A scheme to increase the number of filter maps to increase robustness for activation quantization.

We compared the reported top-1 accuracy of their quantization with various weight/activation bit-precision for AlexNet.• Fine-grained Quantization (FGQ, Mellempudi et al. (2017) ): A direct quantization scheme (i.e., little re-training needed) based on fine-grained grouping (i.e., within a small subset of filter maps).

We compared the reported top-1 validation accuracy of their quantization with 2-bit weight and 4-bit activation for AlexNet and ResNet50.• Weighted-entropy-based quantization (WEP, Park et al. (2017) ): A quantization scheme that considers statistics of weight/activation.

We compared the top-1/top-5 reported accuracy of their quantization with various bit-precision for AlexNet, where the first and last layers are not quantized.• Low-precision batch normalization (LPBN, Graham (2017) ): A scheme for activation quantization in the process of batch normalization.

We compared the top-1/top-5 reported accuracy of their quantization with 3-5 bit precision for activation.

The first layer activation is not quantized.• Half-wave Gaussian quantization (HWGQ, Cai et al. (2017) ): A quantization scheme that finds the scale via Lloyd search on Normal distribution.

We compared the top-1/top-5 reported accuracy for their quantization with 1-bit weight and varying activation bit-precision for AlexNet, and 2-bit weight for ResNet18 and ResNet50.

The first and last layers are not quantized.

In this section, we present full comparison of accuracy (top-1 and top-5) of the tested CNNs (AlexNet, ResNet18, ResNet50) for image classification on IMAGENET dataset.

All the data points for PACT and DoReFa are obtained by running experiments on Tensorpack.

All the other data points are accuracy reported in the corresponding papers.

As can be seen, PACT achieves the best accuracy across the board for various flavors of quantization.

We also observe that using PACT for activation quantization enables more aggressive weight quantization without loss in accuracy.

<|TLDR|>

@highlight

A new way of quantizing activation of Deep Neural Network via parameterized clipping which optimizes the quantization scale via stochastic gradient descent.