Network quantization is one of the most hardware friendly techniques to enable the deployment of convolutional neural networks (CNNs) on low-power mobile devices.

Recent network quantization techniques quantize each weight kernel in a convolutional layer independently for higher inference accuracy, since the weight kernels in a layer exhibit different variances and hence have different amounts of redundancy.

The quantization bitwidth or bit number (QBN) directly decides the inference accuracy, latency, energy and hardware overhead.

To effectively reduce the redundancy and accelerate CNN inferences, various weight kernels should be quantized with different QBNs.

However, prior works use only one QBN to quantize each convolutional layer or the entire CNN, because the design space of searching a QBN for each weight kernel is too large.

The hand-crafted heuristic of the kernel-wise QBN search is so sophisticated that domain experts can obtain only sub-optimal results.

It is difficult for even deep reinforcement learning (DRL) DDPG-based agents to find a kernel-wise QBN configuration that can achieve reasonable inference accuracy.

In this paper, we propose a hierarchical-DRL-based kernel-wise network quantization technique, AutoQ, to automatically search a QBN for each weight kernel, and choose another QBN for each activation layer.

Compared to the models quantized by the state-of-the-art DRL-based schemes, on average, the same models quantized by AutoQ reduce the inference latency by 54.06%, and decrease the inference energy consumption by 50.69%, while achieving the same inference accuracy.

Although convolutional neural networks (CNNs) have been the dominant approach Sandler et al. (2018) to solving a wide variety of problems such as computer vision and recommendation systems, it is challenging to deploy CNNs to mobile devices having only limited hardware resources and tight power budgets, due to their huge essential computing overhead, e.g., an inference of MobileNetV2 Sandler et al. (2018) involves 6.9M weights and 585M floating point operations.

Several approaches such as pruning He et al. (2018) and low-rank approximation Denton et al. (2014) are proposed to reduce the inference computing overhead of CNNs.

Network quantization ; becomes one of the most hardware friendly CNN acceleration techniques by approximating real-valued weights and activations to QBN -bit fixed-point representations, and performing inferences using cheaper fixed-point multiple-accumulation (MAC) operations, where QBN is the quantization bit number.

Instead of using one QBN for the whole CNN, the layer-wise network quantization ; Elthakeb et al. (2018) assigns a QBN to the weights of each convolutional layer, and searches another QBN for the activations of the same layer to decrease the inference computing overhead.

But the inference cost of the layer-wise quantized CNNs is still prohibitive for low-power mobile devices powered by batteries.

Recent works Zeng et al. (2019) ; Choukroun et al. (2019b) ; Zhang et al. (2018) ; Li et al. (2019) ; Krishnamoorthi (2018) ; Sasaki et al. (2019) find that various weight kernels of a convolutional layer exhibit different variances shown in Figure 1 and hence have different amounts of redundancy.

Therefore, they quantize each weight kernel independently for higher accuracy by calculating a QBN -element scaling factor vector for each kernel, rather than globally quantize all the kernels of a layer as a whole.

To reduce different amounts of redundancy among different weight kernels, these kernel-wise network quantization techniques should have searched a QBN for each kernel of each layer in a CNN.

However, the search space of choosing a QBN for each weight kernel is too large, so prior kernel-wise network quantization Zeng et al. (2019) ; Choukroun et al. (2019b) ; Zhang et al. (2018) ; Li et al. (2019) ; Krishnamoorthi (2018) ; Sasaki et al. (2019) still uses the same QBN for the entire CNN.

As Figure 2 shows, compared to the layer-wise quantized model, on the same FPGA accelerator Umuroglu et al. (2019a) , the kernel-wise quantized model (assigning a QBN to each weight kernel and choosing a QBN for each activation layer) improves the inference accuracy by ??? 2% with the same computing overhead (inference latency).

How to decide a QBN for each weight kernel is the most important task of the kernel-wise network quantization, since the QBNs have a large impact on the inference accuracy, latency and hardware overhead.

Determining a QBN for each weight kernel via hand-crafted heuristics is so sophisticated that even machine learning experts can obtain only sub-optimal results.

Recent works ; Elthakeb et al. (2018) automatically select a QBN for each layer of a CNN through a deep reinforcement learning (DRL) agent without human intervention.

However, it is still difficult for low-power mobile devices such as drones and smart glasses to adopt the layer-wise quantized CNN models.

These mobile devices are very sensitive to the bit-width of fixed-point MAC operations and memory access during inferences due to their limited battery lifetime and hardware resources.

Kernel-wise network quantization assigning a QBN to each weight kernel and searching a QBN for each activation layer of a CNN becomes a must to enable the efficient deployment of deep CNNs on mobile devices by reducing the inference computing overhead.

Although it is straightforward to perform kernel-wise quantization via DRL, it takes ultra-long time for a DRL agent to find a proper QBN for each weight kernel of a CNN.

As CNN architectures are becoming deeper, it is infeasible to employ rule-based domain expertise or conventional DRL-based techniques to explore the exponentially enlarging search space of kernel-wise network quantization.

In this paper, we propose a hierarchical-DRL-based agent, AutoQ, to automatically and rapidly search a QBN for each weight kernel and choose a QBN for each activation layer of a CNN for accurate kernel-wise network quantization.

AutoQ comprises a high-level controller (HLC) and a low-level controller (LLC).

The HLC chooses a QBN for each activation layer and generates a goal, the average QBN for all weight kernels of a convolutional layer, for each layer.

Based on the goal, the LLC produces an action, QBN, to quantize each weight kernel of the layer.

The HLC and LLC simultaneously learn by trials and errors, i.e., penalizing inference accuracy loss while rewarding a smaller QBN.

We also build a state space, a goal and an action space, an intrinsic reward and an extrinsic reward for AutoQ. Instead of proxy signals including FLOPs, number of memory access and model sizes, we design the extrinsic reward to take the inference latency, energy consumption and hardware cost into consideration.

Quantization.

Recent works Lin et al. (2016) ; Zhou et al. (2017); Jacob et al. (2018); McKinstry et al. (2018); Zhang et al. (2018) quantize the real-valued weights and activations to fixed-point representations, so that the model size is reduced and inferences can use low-cost fixed-point MAC operations.

To further reduce inference computing overhead, prior works Kim & Smaragdis (2016) ; ; ; Tang et al. (2017) ; Rastegari et al. (2016) ; quantize weights and activations into multi-bit binary codes of {-1, +1}s.

Rather than real-valued MACs, inferences of these quantized models depend on bit-wise logic operations, i.e., XNORs and popcounts.

These traditional quantization techniques either simply assign a single QBN to the whole CNN or require domain experts to determine a QBN for each layer of a CNN.

Sasaki et al. (2019) observe various weight kernels of a convolutional layer have different amounts of redundancy, and quantize each weight kernel independently for higher accuracy.

To exploit different amounts of redundancy among different weight kernels, these kernel-wise network quantization techniques should have searched a QBN for each kernel of each convolutional layer, and assigned a QBN for each activation layer in a CNN.

However, the search space size of the kernel-wise network quantization is 33

n layer , where c outi is the number of weight kernels (output channels) of the ith layer.

No prior work tries to search such huge design space.

Stewart & Stalzer (2018) to automatically architect CNNs for higher inference accuracy.

Their network architectures outperform many human-designed neural networks.

The weight channel pruning is automatically conducted by DRL He et al. (2018) and genetic algorithm .

ReLeQ Elthakeb et al. (2018) quantizes only the weights of each layer of a CNN by DRL, while HAQ Wang et al. (2019) performs the layer-wise quantization for both weights and activations via a DRL agent.

No prior quantization or pruning work relies on hierarchical DRL.

Table 2 compares AutoQ against prior DRL-based techniques for quantization and pruning.

AutoQ is the first work to automatically quantize each weight kernel and each activation layer of a pre-trained CNN model for mobile devices by hierarchical DRL.

Overview.

We do not aim to present a new network quantization technique, but we formulate the search of a QBN for each weight kernel and each activation layer as a hierarchical DRL problem.

We propose a two-level hierarchical DRL technique, AutoQ, to automatically quantize the weights in the kernel-wise manner and the activations in the layer-wise fashion.

We build the state space, action and goal space, extrinsic and intrinsic reward functions and a hierarchical DRL agent for AutoQ. Although we use the state-of-the-art learned quantization technique, LQ-Nets Zhang et al. (2018) , to quantize weight kernels and activation layers with the QBNs found by AutoQ, future novel quantization techniques can be easily integrated to AutoQ to improve the inference accuracy of the quantized networks.

In the extrinsic reward, besides the inference latency and energy , AutoQ also considers the FPGA area overhead critical to low-cost mobile devices.

Working Flow.

For an n layer -layer CNN, the weight is defined as W ??? R n layer ??cout??cin??ww??hw , where n layer is the number of layers; c out denotes the number of kernels (output channels); c in means the number of input channels; w w indicates the kernel width; and h w is the kernel height.

The activation is defined as A ??? R n layer ??cin??wa??ha , where w a is the feature map width; and h a means the feature map height.

The working flow of AutoQ is shown in Figure 3 .

AutoQ consists of a high-level controller (HLC) and a low-level controller (LLC).

The HLC quantizes the network layer by layer, while the LLC searches a QBN for each weight kernel in a layer.

At first, AutoQ receives an observation state [Li,Kj ] from the environment that is the quantized network model, where state [Li,Kj ] includes the information of the CNN architecture.

The HLC makes a goal g Li that is the QBN for the activation layer L i .

The flow then jumps to .

Or the HLC generates a goal g Li which is the average QBN of all weight kernels in the layer L i for the LLC.

The LLC produces an action a [Li,Kj ] , QBN, for the weight kernel K j of the layer L i .

For the entire layer L i , the LLC aims to reach the goal g Li of the HLC.

The environment sends the network quantization and hardware configuration to the fast and accuracy machine-learning-based hardware overhead estimator.

The hardware overhead estimator returns the energy consumption, area overhead and inference latency for the current quantization and hardware configuration.

With the hardware overhead and inference accuracy, the environment generates an extrinsic reward eRd [Li,Kj ] for AutoQ to evaluate the LLC action.

Based on all actions of LLC for the layer L i , the HLC provides an intrinsic reward iRd Li to tell how well the goal is implemented by the LLC.

State Space.

A state state [Li,Kj ] (observation) is represented by

where L i is the layer index; K j means the weight kernel index; c in indicates the number of input channels; c out denotes the number of kernels; s kernel is the kernel size; s stride is the stride; s f eature is the input feature map size; b dw binarily indicates depthwise convolution or not; b w/a binarily represents weight or activation; g Li???1 is the goal (average QBN) of the last layer; and a [Li, is the action (QBN) of the last kernel in the L i layer.

For each variable in state [Li,Kj ] , we normalize it to [0, 1] .

If the layer is a fully-connected layer, we set s kernel = 1, s stride = 0, and b dw = 0.

Goal and Action Space.

The HLC produces the average QBN for all weight kernels of each layer or the QBN for each activation layer as a goal, while the LLC generates a QBN for each weight kernel in a layer as an action.

The HLC goal g Li for the L i layer uses a continuous space and can be any real value between 1 and goal max , where goal max is the maximum average QBN for a layer and we set it to 8.

If the L i layer is an activation layer, we round the real-valued g Li to the discrete value of roundup(1 + g Li ?? (goal max ??? 1)).

Although the LLC action is an integer between 0 and action max , it still uses a continuous space to capture the relative order, i.e., 2-bit is more aggressive than 3-bit, where action max is the maximum QBN for a kernel and we set it to 8.

For the K j kernel of the L i layer, the LLC generates the continuous action ra [Li,Kj ] that is in the range of [0, 1], and round it up to the discrete value a [Li,Kj ] = roundup(ra [Li,Kj ] ?? action max ).

Extrinsic Reward.

After an action a [Li,Kj ] is taken, AutoQ arrives at a new state state [Li, Kj+1] and receives an extrinsic reward eRd from the environment.

The HLC aims to maximize the accumulative extrinsic reward eRd = i j ?? i couti+j???1 eRd eRd [Li,Kj ] , where ?? eRd ??? [0, 1) is a decay factor.

The immediate extrinsic reward can be represented by eRd [Li,Kj ]

where N C is the network configuration; HC means the hardware configuration, e.g., memory bandwidth; accuracy(N C) indicates the inference accuracy; lat is the inference latency of the network N C running on the hardware HC; en represents the inference energy of N C running on HC; area is the FPGA area (hardware cost) used by N C on HC; ?? acc , ?? l , ?? e and ?? a are user-defined factors deciding the impact of inference accuracy, latency, energy and FPGA area on the extrinsic reward.

By different values of user-defined factors, AutoQ implements the resource-constrained and accuracy-guaranteed searches.

For resource-constrained applications, e.g., low-power drones, AutoQ sets ?? acc = 1, ?? l = 0, ?? e = 0 and ?? a = 0 to achieve the best accuracy given the maximum amount of hardware resources (latency, energy, and FPGA area).

This extrinsic reward offers no incentive for lower QBNs, so AutoQ reduces the QBN by limiting the action space.

AutoQ allows arbitrary action at the first few layers and starts to limit the action when it finds that the hardware resource budget is insufficient even after using the smallest QBN for all the following layers.

For accuracy-guaranteed applications, e.g., fingerprint locks, AutoQB sets ?? acc = 2, ?? l < 1, ?? e < 1 and ?? a < 1 to obtain the shortest latency, the minimal energy, and the smallest hardware cost with no accuracy loss. [Li,Kj ] , where ?? iRd ??? [0, 1) is a decay factor.

The LLC produces actions to help the HLC to maximize the extrinsic reward, so it should aim to complete the goal of the HLC and to maximize the extrinsic reward.

But at the beginning of the AutoQ training, the extremely low extrinsic reward due to the random goals of the HLC prevents the LLC from efficiently learning from the environment.

We propose a shaped reward as the intrinsic reward for the LLC to take both the goal completion and the extrinsic reward into consideration, and to enable fine-grained low-level behavior learning.

The intrinsic reward can be represented by

where ?? is a user-defined factor dynamically enlarging from 0.1 to 0.8 as the number of training epochs increases.

When ?? is small, the HLC has stronger influence on the LLC.

On the contrary, when ?? = 1, the LLC maximizes only the accumulative extrinsic reward.

Hardware Overhead Estimator.

A recent work estimates the hardware latency and energy by physical FPGA accelerators.

However, a typical synthesis for a CNN model on a FPGA costs > 30 minutes Gopinath et al. (2019) .

Invoking a FPGA synthesis for each action will make AutoQ unacceptably slow.

We adopt fast and accurate FPGA latency, area Liu & Carloni (2013) and power Zhou et al. (2019) models to predict the inference latency, energy and FPGA area for an arbitrary configuration of network and hardware.

These machine-learning-based models are highly accurate and can estimate the hardware overhead to compute the extrinsic reward of AutoQ within several milliseconds.

Hierarchical DRL.

AutoQ uses a HIerarchical Reinforcement learning with Off-policy correction (HIRO) Nachum et al. (2018) , to implement the HLC and the LLC.

The LLC is trained by incorporating g Li into the standard TD3 method Nachum et al. (2018) .

So the low-level Q-value function Q LLC ?? LLC is to minimize the error ?? LLC (state [Li,Kj ] , g Li , a [Li,Kj ] , state [Li, Kj+1] ), which is

where ?? with Gaussian noises by collecting the actions as N (?? [Li, K0] , g Li , eRd [Li, K0:

where a [Li, denotes the sequence of a [Li, K0] ??? a [Li, ; and eRd [Li, K0: Kc out ???1 ] means the sequence of eRd [Li, K0] ??? eRd [Li, .

AutoQ stores these state-goal-reward transitions into the replay buffer.

However, since transitions obtained from the past LLCs do not accurately reflect the actions that would occur if the same goal was used with the current LLC, AutoQ has to introduce a correction translating old transitions into ones that agree with the current LLC.

AutoQ re-labels the high-level transition (s [Li, K0] , g Li , eRd [Li, K0: Zhang et al. (2018) , and finetune the quantized model for ten epochs to recover the accuracy using stochastic gradient descent (SGD) with a fixed learning rate of 10 ???3 and momentum of 0.9.

We randomly select 100 categories from the ImageNet to accelerate the model finetuning.

After the search is done, we quantize the model with the best policy found by AutoQ and finetune it on the full dataset.

Implementation Details.

An AutoQ agent, i.e., HLC or LLC, consists of an actor network and a critic network.

Both share the same architecture, i.e., two hidden layers, each of which has 300 units.

For the actor network, we add an additional sigmoid function producing an output in the range of [0, 1].

We use a fixed learning rate of 10 ???4 for the actor network and 10 ???3 for the critic network.

AutoQ trains the networks with the batch size of 64 and the replay buffer size of 2000.

AutoQ first explores 100 episodes with a constant noise, i.e., ?? a [L i ,K j ] = 0.5 for the LLC and ?? g [L i ] = 0.5 for the HLC, and then exploits 300 episodes with exponentially decayed noise.

Storage Cost.

We need to record a 4-bit QBN ranging from 0 to 8 for each activation layer and each weight kernel of a convolutional layer.

The storage overhead of AutoQ is ??? 0.1% of the size of various CNN models.

For instance, ResNet-18 found by resource-constrained AutoQ requires 8.3MB to store its quantized model in Table 3 .

The storage overhead of AutoQ is only 0.07%.

Experimental Settings.

To evaluate AutoQ, we selected several CNN models including ResNet-18, ResNet-50, SqueezeNetV1 Iandola et al. (2016) and MobileNetV2 Sandler et al. (2018) .

The CNN models are trained on ImageNet including 1.26M training images and tested on 50K test images spanning 1K categories of objects.

We evaluated the inference performance, energy consumption and FPGA area of the CNN models quantized by AutoQ on a Xilinx Zynq-7020 embedded FPGA.

On the FPGA, we implemented a temporal CNN accelerator Umuroglu et al. (2019b) that uses bit-serial multipliers, each of which computes with one-bit digits from multiple weights and their corresponding activations in parallel at one time, and then accumulates their partial products.

Resource-constrained Quantization.

We make AutoQ perform the resource-constrained searches by imposing a latency constraint and setting ?? acc = 1, ?? l = 0, ?? e = 0 and ?? a = 0 in the extrinsic reward.

With such a setting, AutoQ aims to search for the best inference accuracy given the longest latency constraint, which is set to the inference latency of the 4-bit network-wise quantized CNN models.

We compare the kernel-wise AutoQ quantized models against the layer-wise HardwareAware Automated Quantization (HAQ) quantized models and the 4-bit networkwise quantized models in Table 3 .

We used the LQ-Nets quantization Zhang et al. (2018) to quantize and finetune the models in all three schemes.

The network-wise scheme uses 4-bit to quantize the whole models, while the layer-wise scheme searches a QBN for weights of each layer, and chooses another QBN for activations of the same layer.

AutoQ chooses a QBN for each weight kernel, and selects another QBN for each activation layer of a CNN.

In Table 3 , the average QBN of weights (W-QBN) can be calculated by n layer Li=1 ccouti Kj =1 W eight QBN [Li,Kj ] n layer i=1

c couti (7) where c outi is the number of output channels in the layer L i and W eight QBN [Li,Kj ] is the QBN for the K j th weight kernel in the layer L i .

The average QBN of activations (A-QBN) is computed

, where Act QBN Li is the QBN for all activations of the layer L i .

Compared to the layer-wise quantization, AutoQ improves the top-1 inference accuracy by > 1.25% when spending almost the same inference latency.

Compared to the 16-bit full-precision models, the models quantized by AutoQ degrade the inference accuracy by at most only 0.41%, but reduce the inference latency by 71.2% on average.

Accuracy-guaranteed Quantization.

We run AutoQ to do the accuracy-guaranteed searches by setting ?? acc = 2, ?? l = 0.5, ?? e = 0 and ?? a = 0 in the extrinsic reward.

Such an extrinsic reward drives AutoQ to quantize the models to achieve the shortest inference latency without significant accuracy loss.

Compared to the layer-wise scheme, AutoQ substantially reduces the inference latency by 42.2% while achieving a similar (averagely -0.1%) top-1 inference accuracy.

Compared to ResNet-18 and ResNet50, the compact models such as SqueezeNetV1 suffer from larger top-1 accuracy degradation, i.e., -0.3% in a accuracy-guaranteed search of AutoQ.

Kernel-wise Search.

AutoQ can assign a QBN to each kernel of a convolutional layer.

The average weight QBN and the average activation QBN of each ResNet-18 layer found by an accuracyguaranteed AutoQ search are shown in Figure 4 .

Both the network-wise and layer-wise quantization techniques use only one QBN to quantize all weight kernels in a convolutional layer, and quantize all activations of the layer by another QBN.

On the contrary, AutoQ searches a QBN for each weight kernel.

Compared to a CNN model quantized by the network-wise or layer-wise quantization technique, the same model quantized by the kernel-wise AutoQ can achieve similar inference accuracy but with a smaller average QBN in each layer.

We also show the weight kernel QBNs of the L 14 layer of ResNet-18 produced by resource-constrained AutoQ searches in Figure 5 .

AutoQ automatically identifies which weight kernel has a smaller (larger) variance and thus less (more) redundancy, so that it can assign a larger (smaller) QBN to the weight kernel.

For instance, as Figure 1 shows, compared to the 53th weight kernel (top-right), the 52th weight kernel (top-left) of ResNet-18 has a smaller weight distribution variance.

Therefore, in Figure 5 , AutoQ assigns a smaller QBN to the 52th weight kernel but provides the 53th weight kernel a larger QBN.

Hierarchical DRL Agent with Shaped Intrinsic Reward.

We evaluated and compared our hierarchical-DRL-based AutoQ against the traditional one-level DDPG-based DRL adopted by a recent layer-wise quantization technique, HAQ Wang et al. (2019) .

The reward comparison of different techniques during the kernel-wise quantization on MobileNetV2 is shown in Figure 6 .

HAQ and AutoQ both support resource-constrained searches, but HAQ cannot support accuracy-guaranteed searches.

So their rewards are just the inference accuracy.

Through the goals of the HLC and the actions of the LLC, AutoQ can find a QBN for each weight kernel and achieve > 70% accuracy much faster than the DDPG-based DRL, i.e., it reaches ??? 70% accuracy after only 200 episodes.

However, the DDPG-based DRL is stuck with 20% inference accuracy until 250 episodes.

The hierarchical-DRL-based AutoQ significantly accelerates the search space exploration of the kernel-wise network quantization.

Although AutoQ uses a prior hierarchical DRL agent HIRO Nachum et al. (2018) to search a QBN for each weight kernel, we propose a novel shaped intrinsic reward considering both the completion of the HLC goals and the extrinsic reward to accelerate the search.

The intrinsic reward of HIRO takes only the completion of the HLC goals into consideration.

The LLC of HIRO cannot directly learn from the environment.

Therefore, compared to AutoQ, it takes extra 200 episodes for HIRO to reach only 60% accuracy as shown in Figure 6 .

Extrinsic Reward.

Unlike the reward of the DDPG-based layer-wise HAQ Wang et al. (2019) considering only the inference accuracy, the extrinsic reward of AutoQ can balance the trade-off between the inference accuracy, latency, energy consumption and FPGA area by enabling the accuracyguaranteed search.

By setting ?? acc = 2, ?? l = 0.5, ?? e = 0.5 and ?? a = 0.5, AutoQ takes the inference accuracy, latency, energy and FPGA area into consideration during an accuracy-guaranteed search.

For instance, AutoQ can find two kernel-wise QBN configurations having similar inference accuracy, latency and energy for MobileNetV2.

We cannot differentiate these two configurations by using only the HAQ reward.

However, the first configuration consumes 94% of the FPGA area, while the other configuration occupies 85% of the FPGA area.

AutoQ can identify the second QBN configuration as a better choice via its extrinsic reward.

Quantization Granularity.

Besides the temporal CNN accelerator Umuroglu et al. (2019b) , the kernel-wise quantized models found by the accuracy-guaranteed AutoQ can reduce the inference latency on a spatial CNN accelerator, BitFunsion Sharma et al. (2018) , that relies on a 2D systolic array of the fusion units spatially summing the shifted partial products of weights and activations.

As Figure 7 shows, compared to the layer-wise quantized models, on average, the kernel-wise quantized models reduce the inference latency by 39.04% and decrease the inference energy by 33.34% on the spatial CNN accelerator.

Therefore, the kernel-wise quantized models greatly reduce the inference latency and energy on both the temporal and spatial CNN accelerators.

Prior works Mellempudi et al. (2017); Choukroun et al. (2019a) suggest it is possible to divide a weight kernel into several subkernels and quantize each sub-kernel independently.

We also use AutoQ to search a QBN for each weight sub-kernel.

As Figure 7 shows, the sub-kernel-wise quantized models cannot improve the inference latency or energy on the spatial CNN accelerator consisting of systolic computing arrays.

Each dot-product operation of a sub-kernel-wise quantized model has to be split into several dotproduct operations to be accumulated together.

A systolic computing array still has to be designed to accommodate the weight sub-kernel with the largest QBN in a kernel.

Therefore, we can see that it is difficult for the fine-grained quantization schemes choosing a QBN for each weight unit that is a part of a kernel to further reduce the inference latency or energy on both the temporal and the spatial CNN accelerators.

In this paper, we propose a hierarchical-DRL-based kernel-wise network quantization technique, AutoQ, consisting of a HLC and a LLC.

The HLC automatically searches an average weight QBN and an average activation QBN for each convolutional layer.

Based on the average weight QBN, the LLC generates a QBN for each weight kernel in each layer.

We also create a state space, a goal and action space, an intrinsic reward and an extrinsic reward to support AutoQ. Particularly, our shaped intrinsic reward enables the LLC to learn efficiently from the environment by considering both the HLC goal completion and the environment extrinsic reward.

Moreover, the extrinsic reward of AutoQ can balance the inference accuracy, latency, energy consumption and FPGA area.

Compared to the models quantized by the state-of-the-art DRL-based schemes, on average, the same models quantized by AutoQ reduce the inference latency by 54.06%, and decrease the inference energy consumption by 50.69%, while achieving the same inference accuracy.

<|TLDR|>

@highlight

Accurate, Fast and Automated Kernel-Wise Neural Network Quantization with Mixed Precision using Hierarchical Deep Reinforcement Learning

@highlight

A method for quantizing neural network weights and activations that uses deep reinforcement learning to select bitwidth for individual kernels in a layer and that achieves better performance, or latency, than prior approaches.

@highlight

This paper proposes to automatically search quantization schemes for each kernel in the neural network, using hierarchial RL to guide the search. 