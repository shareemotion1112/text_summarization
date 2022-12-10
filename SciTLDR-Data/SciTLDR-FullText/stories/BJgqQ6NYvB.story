We present FasterSeg, an automatically designed semantic segmentation network with not only state-of-the-art performance but also faster speed than current methods.

Utilizing neural architecture search (NAS), FasterSeg is discovered from a novel and broader search space integrating multi-resolution branches, that has been recently found to be vital in manually designed segmentation models.

To better calibrate the balance between the goals of high accuracy and low latency, we propose a decoupled and fine-grained latency regularization, that effectively overcomes our observed phenomenons that the searched networks are prone to "collapsing" to low-latency yet poor-accuracy models.

Moreover, we seamlessly extend FasterSeg to a new collaborative search (co-searching) framework, simultaneously searching for a teacher and a student network in the same single run.

The teacher-student distillation further boosts the student model’s accuracy.

Experiments on popular segmentation benchmarks demonstrate the competency of FasterSeg.

For example, FasterSeg can run over 30% faster than the closest manually designed competitor on Cityscapes, while maintaining comparable accuracy.

Semantic segmentation predicts pixel-level annotations of different semantic categories for an image.

Despite its performance breakthrough thanks to the prosperity of convolutional neural networks (CNNs) (Long et al., 2015) , as a dense structured prediction task, segmentation models commonly suffer from heavy memory costs and latency, often due to stacking convolutions and aggregating multiple-scale features, as well as the increasing input image resolutions.

However, recent years witness the fast-growing demand for real-time usage of semantic segmentation, e.g., autonomous driving.

Such has motivated the enthusiasm on designing low-latency, more efficient segmentation networks, without sacrificing accuracy notably (Zhao et al., 2018; Yu et al., 2018a) .

The recent success of neural architecture search (NAS) algorithms has shed light on the new horizon in designing better semantic segmentation models, especially under latency of other resource constraints.

Auto-DeepLab (Liu et al., 2019a) first introduced network-level search space to optimize resolutions (in addition to cell structure) for segmentation tasks.

and Li et al. (2019) adopted pre-defined network-level patterns of spatial resolution, and searched for operators and decoders with latency constraint.

Despite a handful of preliminary successes, we observe that the successful human domain expertise in designing segmentation models appears to be not fully integrated into NAS frameworks yet.

For example, human-designed architectures for real-time segmentation (Zhao et al., 2018; Yu et al., 2018a) commonly exploit multi-resolution branches with proper depth, width, operators, and downsample rates, and find them contributing vitally to the success: such flexibility has not been unleashed by existing NAS segmentation efforts.

Furthermore, the trade-off between two (somewhat conflicting) goals, i.e., high accuracy and low latency, also makes the search process unstable and prone to "bad local minima" architecture options.

As the well-said quote goes: "those who do not learn history are doomed to repeat it".

Inheriting and inspired by the successful practice in hand-crafted efficient segmentation, we propose a novel NAS framework dubbed FasterSeg, aiming to achieve extremely fast inference speed and competitive accuracy.

We designed a special search space capable of supporting optimization over multiple branches of different resolutions, instead of a single backbone.

These searched branches are adaptively aggregated for the final prediction.

To further balance between accuracy versus latency and avoiding collapsing towards either metric (e.g., good latency yet poor accuracy), we design a decoupled and fine-grained latency regularization, that facilitates a more flexible and effective calibration between latency and accuracy.

Moreover, our NAS framework can be easily extended to a collaborative search (co-searching), i.e., jointly searching for a complex teacher network and a light-weight student network in a single run, whereas the two models are coupled by feature distillation in order to boost the student's accuracy.

We summarize our main contributions as follows:

• A novel NAS search space tailored for real-time segmentation, where multi-resolution branches can be flexibility searched and aggregated.

• A novel decoupled and fine-grained latency regularization, that successfully alleviates the "architecture collapse" problem in the latency-constrained search.

• A novel extension to teacher-student co-searching for the first time, where we distill the teacher to the student for further accuracy boost of the latter.

• Extensive experiments demonstrating that FasterSeg achieves extremely fast speed (over 30% faster than the closest manually designed competitor on CityScapes) and maintains competitive accuracy.

Human-designed CNN architectures achieve good accuracy performance nowadays (He et al., 2016; .

However, designing architectures to balance between accuracy and other resource constraints (latency, memory, FLOPs, etc.) requires more human efforts.

To free human experts from this challenging trade-off, neural architecture search (NAS) has been recently introduced and drawn a booming interest (Zoph & Le, 2016; Brock et al., 2017; Pham et al., 2018; Liu et al., 2018a; Chen et al., 2018a; Bender et al., 2018; Chen et al., 2018c; .

These works optimize both accuracy and resource utilization, via a combined loss function , or a hybrid reward signal for policy learning Cheng et al., 2018) , or a constrained optimization formulation .

Most existing resource-aware NAS efforts focus on classification tasks, while semantic segmentation has higher requirements for preserving details and rich contexts, therefore posing more dilemmas for efficient network design.

Fortunately, previous handcrafted architectures for real-time segmentation have identified several consistent and successful design patterns.

ENet (Paszke et al., 2016) adopted early downsampling, and ICNet (Zhao et al., 2018) further incorporated feature maps from multiresolution branches under label guidance.

BiSeNet (Yu et al., 2018a ) fused a context path with fast downsampling and a spatial path with smaller filter strides.

More works target on segmentation efficiency in terms of computation cost Marin et al., 2019) and memory usage .

Their multi-resolution branching and aggregation designs ensure sufficiently large receptive fields (contexts) while preserving high-resolution fine details, providing important clues on how to further optimize the architecture.

There have been recent studies that start pointing NAS algorithms to segmentation tasks.

AutoDeepLab (Liu et al., 2019a) pioneered in this direction by searching the cells and the networklevel downsample rates, to flexibly control the spatial resolution changes throughout the network.

and Li et al. (2019) introduced resource constraints into NAS segmentation.

A multi-scale decoder was also automatically searched .

However, compared with manually designed architectures, those search models still follow a single-backbone design and did not fully utilize the prior wisdom (e.g., multi-resolution branches) in designing their search spaces.

Lastly, we briefly review knowledge distillation (Hinton et al., 2015) , that aims to transfer learned knowledge from a sophisticated teacher network to a light-weight student, to improve the (more efficient) student's accuracy.

For segmentation, Liu et al. (2019b) and Nekrasov et al. (2019) proposed to leverage knowledge distillation to improve the accuracy of the compact model and speed-up convergence.

There was no prior work in linking distillation with NAS yet, and we will introduce the extension of FasterSeg by integrating teacher-student model collaborative search for the first time.

3 FASTERSEG: FASTER REAL-TIME SEGMENTATION Our FasterSeg is discovered from an efficient and multi-resolution search space inspired by previous manual design successes.

A fine-grained latency regularization is proposed to overcome the challenge of "architecture collapse" (Cheng et al., 2018) .

We then extend our FasterSeg to a teacherstudent co-searching framework, further resulting in a lighter yet more accurate student network.

The core motivation behind our search space is to search multi-resolution branches with overall low latency, which has shown effective in previous manual design works (Zhao et al., 2018; Yu et al., 2018a) .

Our NAS framework automatically selects and aggregates branches of different resolutions, based on efficient cells with searchable superkernels.

Figure 1 : The multi-resolution branching search space for FasterSeg, where we aim to optimize multiple branches with different output resolutions.

These outputs are progressively aggregated together in the head module.

Each cell is individually searchable and may have two inputs and two outputs, both of different downsampling rates (s).

Inside each cell, we enable searching for expansion ratios within a single superkernel. (Zhao et al., 2018) .

Bottom: BiSeNet (Yu et al., 2018a) Inspired by (Liu et al., 2019a) , we enable searching for spatial resolutions within the L-layer cells ( Figure  1 ), where each cell takes inputs from two connected predecessors and outputs two feature maps of different resolutions.

Hand-crafted networks for real-time segmentation found multi-branches of different resolutions to be effective (Zhao et al., 2018; Yu et al., 2018a) .

However, architectures explored by current NAS algorithms are restricted to a single backbone.

Our goal is to select b (b > 1) branches of different resolutions in this L-layer framework.

Specifically, we could choose b different final output resolutions for the last layer of cells, and decode each branch via backtrace (section 3.4).

This enables our NAS framework to explore b individual branches with different resolutions, which are progressively "learned to be aggregated" by the head module ( Figure 1 ).

We follow the convention to increase the number of channels at each time of resolution downsampling.

To enlarge the model capacity without incurring much latency, we first downsample the input image to 1 8 original scale with our stem module, and then set our searchable downsample rates s ∈ {8, 16, 32}. Figure 2 shows that our multi-resolution search space is able to cover existing human-designed networks for real-time segmentation.

See Appendix B for branch selection details.

As we aim to boost the inference latency, the speed of executing an operator is a direct metric (rather than indirect metrics like FLOPs) for selecting operator candidates O. Meanwhile, as we previously discussed, it is also important to ensure sufficiently large receptive field for spatial contexts.

We analyze typical operators, including their common surrogate latency measures (FLOPs, parameter numbers), and their real-measured latency on an NVIDIA 1080Ti GPU with TensorRT library, and their receptive fields, as summarized in Table 1 .

Compared with standard convolution, group convolution is often used for reducing FLOPs and number of parameters (Sandler et al., 2018; Ma et al., 2018) .

Convolving with two groups has the same receptive field with a standard convolution but is 13% faster, while halving the parameter amount (which might not be preferable as it reduces the model learning capacity).

Dilated convolution has an enlarged receptive field and is popular in dense predictions Dai et al., 2017) .

However, as shown in Table 1 (and as widely acknowledged in engineering practice), dilated convolution (with dilation rate 2) suffers from dramatically higher latency, although that was not directly reflected in FLOPs nor parameter numbers.

In view of that, we design a new variant called "zoomed convolution", where the input feature map is sequentially processed with bilinear downsampling, standard convolution, and bilinear upsampling.

This special design enjoys 40% lower latency and 2 times larger receptive field compared to standard convolution.

Our search space hence consists of the following operators:

• skip connection

• 3×3 conv.

• 3×3 conv.

×2

• "zoomed conv.": bilinear downsampling + 3×3 conv.

+ bilinear upsampling • "zoomed conv.

×2": bilinear downsampling + 3×3 conv.

×2 + bilinear upsampling As mentioned by Ma et al. (2018) , network fragmentation can significantly hamper the degree of parallelism, and therefore practical efficiency.

Therefore, we choose a sequential search space (rather than a directed acyclic graph of nodes ), i.e., convolutional layers are sequentially stacked in our network.

In Figure 1 , each cell is differentiable 2019a) and will contain only one operator, once the discrete architecture is derived (section 3.4).

It is worth noting that we allow each cell to be individually searchable across the whole search space.

We further give each cell the flexibility to choose different channel expansion ratios.

In our work, we search for the width of the connection between successive cells.

That is however non-trivial due to the exponentially possible combinations of operators and widths.

To tackle this problem, we propose a differentiably searchable superkernel, i.e., directly searching for the expansion ratio χ within a single convolutional kernel which supports a set of ratios X ⊆ N + .

Inspired by (Yu et al., 2018c) and (Stamoulis et al., 2019) , from slim to wide our connections incrementally take larger subsets of input/output dimensions from the superkernel.

During the architecture search, for each superkernel, only one expansion ratio is sampled, activated, and back-propagated in each step of stochastic gradient descent.

This design contributes to a simplified and memory-efficient super network and is implemented via the renowned "Gumbel-Softmax" trick (see Appendix C for details).

To follow the convention to increase the number of channels as resolution downsampling, in our search space we consider the width = χ × s, where s ∈ {8, 16, 32}. We allow connections between each pair of successive cells flexibly choose its own expansion ratio, instead of using a unified single expansion ratio across the whole search space.

Denote the downsample rate as s and layer index as l. To facilitate the search of spatial resolutions, we connect each cell with two possible predecessors' outputs with different downsample rates:

Each cell could have at most two outputs with different downsample rates into its successors:

The expansion ratio χ j s,l is sampled via "Gumbel-Softmax" trick according to p(χ = χ j s,l ) = γ j s,l .

Here, α, β, and γ are all normalized scalars, associated with each operator O k ∈ O, each predecessor's output O l−1 , and each expansion ratio χ ∈ X , respectively (Appendix D).

They encode the architectures to be optimized and derived.

Low latency is desirable yet challenging to optimize.

Previous works (Cheng et al., 2018; observed that during the search procedure, the supernet or search policy often fall into bad "local minimums" where the generated architectures are of extremely low latency but with poor accuracy, especially in the early stage of exploration.

In addition, the searched networked tend to use more skip connections instead of choosing low expansion ratios (Shaw et al., 2019) .

This problem is termed as "architecture collapse" in our paper.

The potential reason is that, finding architectures with extremely low latency (e.g. trivially selecting the most light-weight operators) is significantly easier than discovering meaningful compact architectures of high accuracy.

To address this "architecture collapse" problem, we for the first time propose to leverage a fine-grained, decoupled latency regularization.

We first achieve the continuous relaxation of latency similar to the cell operations in section 3.1.4, via replacing the operator O in Eqn.

1 and 2 with the corresponding latency.

We build a latency lookup table that covers all possible operators to support the estimation of the relaxed latency.

Figure 3 demonstrates the high correlation of 0.993 between the real and estimated latencies (see details in appendix E).

We argue that the core reason behind the "architecture collapse" problem is the different sensitivities of supernet to operator O, downsample rate s, and expansion ratio χ.

Operators like "3×3 conv.

×2" and "zoomed conv." have a huge gap in latency.

Similar latency gap (though more moderate) exists between slim and wide expansion ratios.

However, downsample rates like "8" and "32" do not differ much, since resolution downsampling also brings doubling of the number of both input and output channels.

We quantitatively compared the influence of O, s, and χ towards the supernet latency, by adjusting one of the three aspects and fixing the other two.

Taking O as the example, we first uniformly initialize β and γ, and calculate ∆Latency(O) as the gap between the supernet which dominantly takes the slowest operators and the one adopts the fastest.

Similar calculations were performed for s and χ.

Values of ∆Latency in Table 2 indicate the high sensitivity of the supernet's latency to operators and expansion ratios, while not to resolutions.

Figure 4 (a) shows that the unregularized latency optimization will bias the supernet towards light-weight operators and slim expansion ratios to quickly minimize the latency, ending up with problematic architectures with low accuracy.

Based on this observation, we propose a regularized latency optimization leveraging different granularities of our search space.

We decouple the calculation of supernet's latency into three granularities of our search space (O, s, χ), and regularize each aspect with a different factor:

where we by default set w 1 = 0.001, w 2 = 0.997, w 3 = 0.002 1 .

This decoupled and fine-grained regularization successfully addresses this "architecture collapse" problem, as shown in Figure 4 Knowledge Distillation is an effective approach to transfer the knowledge learned by a large and complex network (teacher T ) into a much smaller network (student S).

In our NAS framework, we can seamlessly extend to teacher-student cosearching, i.e., collaboratively searching for two networks in a single run ( Figure 5 ).

Specifically, we search a complex teacher and light-weight student simultaneously via adopting two sets of architectures in one supernet: (α T , β T ) and (α S , β S , γ S ).

Note that the teacher does not search the expansion ratios and always select the widest one.

This extension does not bring any overhead in memory usage or size of supernet since the teacher and student share the same supernet weights W during the search process.

Two sets of architectures are iteratively optimized during search (please see details in Appendix F), and we apply the latency constraint only on the student, not on the teacher.

Therefore, our searched teacher is a sophisticated network based on the same search space and supernet weights W used by the student .

During training from scratch, we apply a distillation loss from teacher T to student S:

)

KL denotes the KL divergence.

q s i and q t i are predicted logit for pixel i from S and T , respectively.

Equal weights (1.0) are assigned to the segmentation loss and this distillation loss.

Once the search is completed, we derive our discrete architecture from α, β, and γ: • α, γ: We select the optimum operators and expansion ratios by taking the argmax of α and γ.

We shrink the operator "skip connection" to obtain a shallower architecture with less cells.

• β: Different from (Liu et al., 2019a) , for each cell s,l we consider β 0 and β 1 as probabilities of two outputs from cell s 2 ,l−1 and cell s,l−1 into cell s,l .

Therefore, by taking the l * = argmax l (β 0 s,l ), we find the optimum position (cell s,l * ) where to downsample the current resolution (

It is worth noting that, the multi-resolution branches will share both cell weights and feature maps if their cells are of the same operator type, spatial resolution, and expansion ratio.

This design contributes to a faster network.

Once cells in branches diverge, the sharing between the branches will be stopped and they become individual branches (See Figure 6) .

1 These values are obtained by solving equations derived from Table 2 in order to achieve balanced sensitivities on different granularities: 10.42 × w1 = 0.01 × w2 = 5.54 × w1, s.t.

w1 + w2 + w3 = 1.

2 For a branch with two searchable downsampling positions, we consider the argmax over the joint proba-

We use the Cityscapes (Cordts et al., 2016) as a testbed for both our architecture search and ablation studies.

After that, we report our final accuracy and latency on Cityscapes, CamVid (Brostow et al., 2008) , and BDD (Yu et al., 2018b) .

In all experiments, the class mIoU (mean Intersection over Union per class) and FPS (frame per second) are used as the metrics for accuracy and speed, respectively.

Please see Appendix G for dataset details.

In all experiments, we use Nvidia Geforce GTX 1080Ti for benchmarking the computing power.

We employ the high-performance inference framework TensorRT v5.1.5 and report the inference speed.

During this inference measurement, an image of a batch size of 1 is first loaded into the graphics memory, then the model is warmed up to reach a steady speed, and finally, the inference time is measured by running the model for six seconds.

All experiments are performed under CUDA 10.0 and CUDNN V7.

Our framework is implemented with PyTorch.

The search, training, and latency measurement codes are available at https://github.com/TAMU-VITA/FasterSeg.

We consider a total of L = 16 layers in the supernet and our downsample rate s ∈ {8, 16, 32}. In our work we use number of branches b = 2 by default, since more branches will suffer from high latency.

We consider expansion ratio χ s,l ∈ X = {4, 6, 8, 10, 12} for any "downsample rate" s and layer l. The multi-resolution branches have 1695 unique paths.

For cells and expansion ratios, we have (1 + 4 × 5) (15+14+13) + 5 3 ≈ 3.4 × 10 55 unique combinations.

This results in a search space in the order of 10 58 , which is much larger and challenging, compared with preliminary studies.

Architecture search is conducted on Cityscapes training dataset.

Figure 6 visualizes the best spatial resolution discovered (FasterSeg).

Our FasterSeg achieved mutli-resolutions with proper depths.

The two branches share the first three operators then diverge, and choose to aggregate outputs with downsample rates of 16 and 32.

Operators and expansion ratios are listed in Table 7 in Appendix I, where the zoomed convolution is heavily used, suggesting the importance of low latency and large receptive field.

We conduct ablation studies on Cityscapes to evaluate the effectiveness of our NAS framework.

More specifically, we examine the impact of operators (O), downsample rate (s), expansion ratios (χ), and also distillation on the accuracy and latency.

When we expand from a single backbone (b = 1) to multi-branches (b = 2), our FPS drops but we gain a much improvement on mIoU, indicating the multiresolution design is beneficial for segmentation task.

By enabling the search for expansion ratios (χ), we discover a faster network with FPS 163.9 without sacrificing accuracy (70.5%), which proves that the searchable superkernel gets the benefit from eliminating redundant channels while maintaining high accuracy.

This is our student network (S) discovered in our co-searching framework (see below).

We further evaluate the efficacy of our teacher-student co-searching framework.

After the collaboratively searching, we obtain a teacher architecture (T ) and a student architecture (S).

As mentioned above, S is searched with searchable expansion ratios (χ), achieving an FPS of 163.9 and an mIoU of 70.5%.

In contrast, when we directly compress the teacher (channel pruning via selecting the slimmest expansion ratio) and train with distillation from the well-trained original cumbersome teacher, it only achieved mIoU = 66.1% with only FPS = 146.7, indicating that our architecture cosearching surpass the pruning based compression.

Finally, when we adopt the knowledge distillation from the well-trained cumbersome teacher to our searched student, we boost the student's accuracy to 73.1%, which is our final network FasterSeg.

This demonstrates that both a student discovered by co-searching and training with knowledge distillation from the teacher are vital for obtaining an accurate faster real-time segmentation model.

In this section, we compare our FasterSeg with other works for real-time semantic segmentation on three popular scene segmentation datasets.

Note that since we target on real-time segmentation, we measure the mIoU without any evaluation tricks like flipping, multi-scale, etc. (Zhao et al., 2018)

67.7 69.5 37.7 1024×2048 BiSeNet (Yu et al., 2018a) 69.0 68.4 105.8 768×1536 CAS 71.6 70.5 108.0 768×1536 Fast-SCNN (Poudel et al., 2019) 68.6 68.0 123.5 1024×2048 DF1-Seg-d8 (Li et al., 2019 73.1 71.5 163.9 1024×2048

Cityscapes: We evaluate FasterSeg on Cityscapes validation and test sets.

We use original image resolution of 1024×2048 to measure both mIoU and speed inference.

In Table 4 , we see the superior FPS (163.9) of our FasterSeg, even under the maximum image resolution.

This high FPS is over 1.3× faster than human-designed networks.

Meanwhile, our FasterSeg still maintains competitive accuracy, which is 73.1% on the validation set and 71.5% on the test set.

This accuracy is achieved with only Cityscapes fine-annotated images, without using any extra data (coarse-annotated images, ImageNet, etc.).

We directly transfer the searched architecture on Cityscapes to train on CamVid.

Table  5 reveals that without sacrificing much accuracy, our FasterSeg achieved an FPS of 398.1.

This extremely high speed is over 47% faster than the closest competitor in FPS (Yu et al., 2018a) , and is over two times faster than the work with the best mIoU .

This impressive result verifies both the high performance of FasterSeg and also the transferability of our NAS framework.

BDD: In addition, we also directly transfer the learned architecture to the BDD dataset.

In Table 6 we compare our FasterSeg with the baseline provided by Yu et al. (2018b) .

Since no previous work has considered real-time segmentation on the BDD dataset, we get 15 times faster than the DRN-D-22 with slightly higher mIoU. Our FasterSeg still preserve the extremely fast speed and competitive accuracy on BDD.

ENet (Paszke et al., 2016) 68.3 61.2 ICNet (Zhao et al., 2018) 67.1 27.8 BiSeNet (Yu et al., 2018a) 65.6 269.1 CAS 71.2 169.0 FasterSeg (ours) 71.1 398.1

We introduced a novel multi-resolution NAS framework, leveraging successful design patterns in handcrafted networks for real-time segmentation.

Our NAS framework can automatically discover FasterSeg, which achieved both extremely fast inference speed and competitive accuracy.

Our search space is intrinsically of low-latency and is much larger and challenging due to flexible searchable expansion ratios.

More importantly, we successfully addressed the "architecture collapse" problem, by proposing the novel regularized latency optimization of fine-granularity.

We also demonstrate that by seamlessly extending to teacher-student co-searching, our NAS framework can boost the student's accuracy via effective distillation.

A STEM AND HEAD MODULE Stem: Our stem module aims to quickly downsample the input image to 1 8 resolution while increasing the number of channels.

The stem module consists of five 3 × 3 convolution layers, where the first, second, and fourth layer are of stride two and double the number of channels.

Head: As shown in Figure 1 , feature map of shape (C 2s × H × W ) is first reduced in channels by a 1 × 1 convolution layer and bilinearly upsampled to match the shape of the other feature map (C s × 2H × 2W ).

Then, two feature maps are concatenated and fused together with a 3 × 3 convolution layer.

Note that we not necessarily have C 2s = 2C s because of the searchable expansion ratios.

Since our searchable downsample rates s ∈ {8, 16, 32} and the number of selected branches b = 2, our supernet needs to select branches of three possible combinations of resolutions: {8, 16}, {8, 32}, and {16, 32}. For each combination, branches of two resolutions will be aggregated by our head module.

Our supernet selects the best b branches based on the criterion used in :

where m is a searched model aggregating b branches, with accuracy ACC(m) and latency LAT (m).

w is the weight factor defined as:

We empirically set α = β = −0.07 and the target latency T = 8.3 ms in our work.

Formally, suppose we have our set of expansion ratios X ⊆ N + , and we want to sample one ratio χ from X .

For each χ i we have an associated probability γ i , where (Gumbel, 1954; Maddison et al., 2014) helps us approximate differentiable sampling.

We first sample a "Gumbel-Noise"

).

We set the temperature parameter τ = 1 in our work.

D NORMALIZED SCALARS α, β, γ α, β, and γ are all normalized scalars and implemented as softmax.

They act as probabilities associating with each operator O k ∈ O, each predecessor's output O l−1 , and each expansion ratio χ ∈ X , respectively:

where s is downsample rate and l is index of the layer in our supernet.

We build a latency lookup table that covers all possible situations and use this lookup table as building blocks to estimate the relaxed latency.

To verify the continuous relaxation of latency, we randomly sample networks of different operators/downsample rates/expansion ratios out of the supernet M, and measured both the real and estimated latency.

We estimate the network latency by accumulating all latencies of operators consisted in the network.

In Figure 3 , we can see the high correlation between the two measurements, with a correlation coefficient of 0.993.

This accurate estimation of network latency benefits from the sequential design of our search space.

Given our supernet M, the overall optimization target (loss) during architecture search is:

We adopt cross-entropy with "online-hard-element-mining" as our segmentation loss L seg .

Lat(M) is the continuously relaxed latency of supernet, and λ is the balancing factor.

We set λ = 0.01 in our work.

As the architecture α, β, and γ are now involved in the differentiable computation graph, they can be optimized using gradient descent.

Similar in (Liu et al., 2019a) , we adopt the first-order approximation ( ), randomly split our training dataset into two disjoint sets trainA and trainB, and alternates the optimization between:

H ARCHITECTURE SEARCH IMPLEMENTATIONS As stated in the second line of Eqn.

2, a stride 2 convolution is used for all s → 2s connections, both to reduce spatial size and double the number of filters.

Bilinear upsampling is used for all upsampling operations.

We conduct architecture search on the Cityscapes dataset.

We use 160 × 320 random image crops from half-resolution (512 × 1024) images in the training set.

Note that the original validation set or test set is never used for our architecture search.

When learning network weights W , we use SGD optimizer with momentum 0.9 and weight decay of 5×10 −4 .

We used the exponential learning rate decay of power 0.99.

When learning the architecture parameters α, β, andγ, we use Adam optimizer with learning rate 3×10 −4 .

The entire architecture search optimization takes about 2 days on one 1080Ti GPU.

In Table 7 we list the operators (O) and expansion ratios (χ) selected by our FasterSeg.

The downsample rates s in Table 7 and Figure 6 match.

We have the number of output channels c out = s × χ.

We observed that the zoomed convolution is heavily used, suggesting the importance of low latency and large receptive field.

@highlight

We present a real-time segmentation model automatically discovered by a multi-scale NAS framework, achieving 30% faster than state-of-the-art models.