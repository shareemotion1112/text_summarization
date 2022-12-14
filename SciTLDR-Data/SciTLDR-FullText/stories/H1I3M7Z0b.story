We present a new approach and a novel architecture, termed WSNet, for learning compact and efficient deep neural networks.

Existing approaches conventionally learn full model parameters independently and then compress them via \emph{ad hoc} processing such as model pruning or filter factorization.

Alternatively, WSNet proposes learning model parameters by sampling from a compact set of learnable parameters, which naturally enforces {parameter sharing} throughout the learning process.

We demonstrate that such a novel weight sampling approach (and induced WSNet) promotes both weights and computation sharing favorably.

By employing this method, we can more efficiently learn much smaller networks with competitive performance compared to baseline networks with equal numbers of convolution filters.

Specifically, we consider learning compact and efficient 1D convolutional neural networks for audio classification.

Extensive experiments on multiple audio classification datasets verify the effectiveness of WSNet.

Combined with weight quantization, the resulted models are up to \textbf{180$\times$} smaller and theoretically up to \textbf{16$\times$} faster than the well-established baselines, without noticeable performance drop.

Despite remarkable successes in various applications, including e.g. audio classification, speech recognition and natural language processing, deep neural networks (DNNs) usually suffer following two problems that stem from their inherent huge parameter space.

First, most of state-of-the-art deep architectures are prone to over-fitting even when trained on large datasets BID42 .

Secondly, DNNs usually consume large amount of storage memory and energy BID17 .

Therefore these networks are difficult to embed into devices with limited memory and power (such as portable devices or chips).

Most existing networks aim to reduce computational budget through network pruning BID16 BID1 BID11 , filter factorization BID23 BID28 , low bit representation BID36 for weights and knowledge transfering BID20 .

In contrast to the above works that ignore the strong dependencies among weights and learn filters independently based on existing network architectures, this paper proposes to explicitly enforce the parameter sharing among filters to more effectively learn compact and efficient deep networks.

In this paper, we propose a Weight Sampling deep neural network (i.e. WSNet) to significantly reduce both the model size and computation cost of deep networks, achieving more than 100?? smaller size and up to 16?? speedup at negligible performance drop or even achieving better performance than the baseline (i.e. conventional networks that learn filters independently).

Specifically, WSNet is parameterized by layer-wise condensed filters from which each filter participating in actual convolutions can be directly sampled, in both spatial and channel dimensions.

Since condensed filters have significantly fewer parameters than independently trained filters as in conventional CNNs, learning by sampling from them makes WSNet a more compact model compared to conventional CNNs.

In addition, to reduce the ubiquitous computational redundancy in convolving the overlapped filters and input patches, we propose an integral image based method to dramatically reduce the computation cost of WSNet in both training and inference.

The integral image method is also advantageous because it enables weight sampling with different filter size and minimizes computational overhead to enhance the learning capability of WSNet.

In order to demonstrate the efficacy of WSNet, we conduct extensive experiments on the challenging acoustic scene classification and music detection tasks.

On each test dataset, including MusicDet200K (a self-collected dataset, as detailed in Section 4), ESC-50 (Piczak, 2015a) , UrbanSound8K BID40 and DCASE BID45 , WSNet significantly reduces the model size of the baseline by 100?? with comparable or even higher classification accuracy.

When compressing more than 180??, WSNet is only subject to negligible accuracy drop.

At the same time, WSNet significantly reduces the computation cost (up to 16??).

Such results strongly establish the capability of WSNet to learn compact and efficient networks.

Although we detailed experiments mostly limited to 1D CNNs in this paper, we will explore how the same approach can be naturally generalized to 2D CNNs in future work.

In this paper we considered Acoustic Scene Classification (ASC) tasks as well as music detection tasks.

ASC aims to classify the surrounding environment where an audio stream is generated given the audio input BID4 .

It can be applied in many different ways such as audio tagging (Cai et al., 2006) , audio collections management (Landone et al., 2007) , robotic navigation BID10 , intelligent wearable interfaces BID50 , context adaptive tasks BID41 , etc.

Music detection is a related task to determine whether or not a small segment of audio is music.

It is usually treated as a binary classification problem given an audio segment as input, i.e., to classify the segment into two categories: music or non-music.

As evident in many other areas, convolutional neural networks (CNN) have been widely used in audio classification tasks BID48 BID39 .

SoundNet BID2 stands out among different CNNs for sound classification due to the following two reasons.

First, it is trained from the large-scale unlabeled sound data using visual information as a bridge, while many other networks are trained with smaller datasets.

Secondly, SoundNet directly takes one dimensional raw wave signals as input so that there is no need to calculate time-consuming audio specific features, e.g. MFCC BID35 BID12 and spectrogram BID15 .

SoundNet has yielded significant performance improvements on state-of-the-art results with standard benchmarks for acoustic scene classification.

In this paper, we demonstrate that the proposed WSNet achieves a comparable or even better performance than SoundNet at a significantly smaller size and faster speed.

Early approaches for deep model compression include BID29 BID18 that prune the connections in networks based on the second order information.

Most recent works in network compression adopt weight pruning BID16 BID11 BID1 Lebedev & Lempitsky, 2016; BID26 BID31 , filter decomposition BID43 BID14 BID23 , hashed networks BID6 BID9 and weight quantization BID17 .

However, although those works reduce model size, they also suffer from large performance drop.

BID5 and Ba & Caruana (2014) are based on student-teacher approches which may be difficult to apply in new tasks since they require training a teacher network in advance.

BID13 predicts parameters based on a few number of weight values.

BID25 proposes an iterative hard thresholding method, but only achieve relatively small compression ratios.

Gong et al. (2014) uses a binning method which can only be applied over fully connected layers.

BID20 compresses deep models by transferring the knowledge from pre-trained larger networks to smaller networks.

In contrast, WSNet is able to learn compact representation for both convolution layers and fully connected layers from scratch.

The deep models learned by WSNet can significantly reduce model size compared to the baselines with comparable or even better performance.

In terms of deep model acceleration, the factorization and quantization methods listed above can also reduce computation latency in inference.

While irregular pruning (as done in most pruning methods BID17 ) incurs computational overhead, grouped pruning (Lebedev & Lempitsky, 2016 ) is able to accelerate networks.

FFT BID32 and LCNN (Bagherinezhad et al., 2016) are also used to speed up computation in pratice.

Comparatively, WSNet is superior because it learns networks that have both smaller model size and faster computation versus baselines.

WSNet presents a class of novel models with the appealing properties of a small model size and small computation cost.

Some recently proposed efficient model architectures include the class of Inception models BID22 BID9 which adopts depthwise separable convolutions, the class of Residual models BID19 BID49 which uses residual path for efficient optimization, and the factorized networks which use fully factorized convolutions.

MobileNet BID21 and Flattened networks BID24 are based on factorization convolutions.

ShuffleNet BID50 uses group convolution and channel shuffle to reduce computational cost.

Compared with above works, WSNet presents a new model design strategy which is more flexible and generalizable: the parameters in deep networks can be obtained conveniently from a more compact representation, e.g. through the weight sampling method proposed in this paper or other more complex methods based on the learned statistic models.

In this section, we describe details of the proposed WSNet for 1D CNNs.

First, the notations are introduced.

Secondly, we elaborate on the core components in WSNet: weight sampling along the spatial dimension and channel dimension.

Thirdly, we introduce the denser weight sampling to enhance the learning capability of WSNet.

Finally, we propose an integral image method for accelerating WSNet in both training and inference.

Before diving into the details, we first introduce the notations used in this paper.

The traditional 1D convolution layer takes as input the feature map F ??? R T ??M and produces an output feature map G ??? R T ??N where (T, M, N ) denotes the spatial length of input, the channel of input and the number of filters respectively.

We assume that the output has the same spatial size as input which holds true by using zero padded convolution.

The 1D convolution kernel K used in the actual convolution of WSNet has the shape of (L, M, N ) where L is the kernel size.

Let k n , n ??? {1, ?? ?? ?? N } denotes a filter and f t , t ??? {1, ?? ?? ?? T } denotes a input patch that spatially spans from t to t + L ??? 1, then the convolution assuming stride one and zero padding is computed as: DISPLAYFORM0 where ?? stands for the vector inner product.

Note we omit the element-wise activation function to simplify the notation.

In WSNet, instead of learning each weight independently, K is obtained by sampling from a learned condensed filter ?? which has the shape of (L * , M * ).

The goal of training WSNet is thus cast to learn more compact DNNs which satisfy the condition of L * M * < LM N .

To quantize the advantage of WSNet in achieving compact networks, we define the compactness of K in a learned layer in WSNet w.r.t.

the conventional layer with independently learned weights as: DISPLAYFORM1 In the following section, we demonstrate WSNet learn compact networks by sampling weights in two dimensions: the spatial dimension and the channel dimension.3.2 WEIGHT SAMPLING 3.2.1 ALONG SPATIAL DIMENSION In conventional CNNs, the filters in a layer are learned independently which presents two disadvantages.

Firstly, the resulted DNNs have a large number of parameters, which impedes their deploy-

Sampling Stride: S

Figure 1: Illustration of WSNet that learns small condensed filters with weight sampling along two dimensions: spatial dimension (the bottom panel) and channel dimension (the top panel).

The figure depicts procedure of generating two continuous filters (in pink and purple respectively) that convolve with input.

In spatial sampling, filters are extracted from the condensed filter with a stride of S. In channel sampling, the channel of each filter is sampled repeatedly for C times to achieve equal with the input channel.

Please refer to Section 3.2 for detailed explanations.

ment in computation resource constrained platforms.

Second, such over-parameterization makes the network prone to overfitting and getting stuck in (extra introduced) local minimums.

To solve these two problems, a novel weight sampling method is proposed to efficiently reuse the weights among filters.

Specifically, in each convolutional layer of WSNet, all convolutional filters K are sampled from the condensed filter ??, as illustrated in Figure 1 .

By scanning the weight sharing filter with a window size of L and stride of S, we could sample out N filters with filter size of L. Formally, the equation between the filter size of the condensed filter and the sampled filters is: DISPLAYFORM0 The compactness along spatial dimension is DISPLAYFORM1 Note that since the minimal value of S is 1, the minimal value of L * (i.e. the minimum spatial length of the condensed filter) is L + N ??? 1 and the maximal achievable compactness is therefore L.

Although it is experimentally verified that the weight sampling strategy could learn compact deep models with negligible loss of classification accuracy (see Section 4), the maximal compactness is limited by the filter size L, as mentioned in Section 3.2.1.In order to seek more compact networks without such limitation, we propose a channel sharing strategy for WSNet to learn by weight sampling along the channel dimension.

As illustrated in Figure 1 (top panel), the actual filter used in convolution is generated by repeating sampling for C times.

The relation between the channels of filters before and after channel sampling is: DISPLAYFORM0 Therefore, the compactness of WSNet along the channel dimension achieves C. As introduced later in Experiments (Section 4), we observe that the repeated weight sampling along the channel dimension significantly reduces the model size of WSNet without significant performance drop.

One notable advantage of channel sharing is that the maximum compactness can be as large as M (i.e. when the condensed filter has channel of 1), which paves the way for learning much more aggressively smaller models (e.g. more than 100?? smaller models than baselines).The above analysis for weight sampling along spatial/channel dimensions can be conveniently generalized from convolution layers to fully connected layers.

For a fully connected layer, we treat its weights as a flattened vector with channel of 1, along which the spatial sampling (ref.

Section 3.2.1) is performed to reduce the size of learnable parameters.

For example, for the fully connected layer "fc1" in the baseline network in Table 1 , its filter size, channel number and filter number are 1536, 1 and 256 respectively.

We can therefore perform spatial sampling for "fc1" to learn a more compact representation.

Compared with convolutional layers which generally have small filter sizes and thus have limited compactnesses along the spatial dimenstion, the fully connected layers can achieve larger compactnesses along the spatial dimension without harming the performance, as demonstrated in experimental results (ref. to Section 4.2).

WSNet is trained from the scratch in a similar way to conventional deep convolutional networks by using standard error back-propagation.

Since every weight K l,m,n in the convolutional kernel K is sampled from the condensed filter ?? along the spatial and channel dimension, the only difference is the gradient of ?? i,j is the summation of all gradients of weights that are tied to it.

Therefore, by simply recording the position mapping M : (i, j) ??? (l, m, n) from ?? i,j to all the tied weights in K, the gradient of ?? i,j is calculated as: DISPLAYFORM0 where L is the conventional cross-entropy loss function.

In open-sourced machine learning libraries which represent computation as graphs, such as TensorFlow BID0 , Equation (4) can be calculated automatically.

The performance of WSNet might be adversely affected when the size of condensed filter is decreased aggressively (i.e. when S and C are large).

To enhance the learning capability of WSNet, we could sample more filters for layers with significantly reduced sizes.

Specifically, we use a smaller sampling strideS (S < S) when performing spatial sampling.

In order to keep the shape of weights unchanged in the following layer, we append a 1??1 convolution layer with the shape of (1,n, n) to reduce the channels of densely sampled filters.

It is experimentally verified that denser weight sampling can effectively improve the performance of WSNet in Section 4.

However, since it also brings extra parameters and computational cost to WSNet, denser weight sampling is only used in lower layers of WSNet whose filter number (n) is small.

Besides, one can also conduct channel sampling on the added 1??1 convolution layers to further reduce their sizes.

According to Equation 1, the computation cost in terms of the number of multiplications and adds (i.e. Mult-Adds) in a conventional convolutional layer is: DISPLAYFORM0 However, as illustrated in FIG0 , since all filters in a layer in WSNet are sampled from a condensed filter ?? with stride S, calculating the results of convolution in the conventional way as in Eq. FORMULA0 incurs severe computational redundance.

Concretely, as can be seen from Eq. (1), one item in the ouput feature map is equal to the summation of L inner products between the row vector of f and the column vector of k. Therefore, when two overlapped filters that are sampled from the condensed filter (e.g. k 1 and k 2 in FIG0 ) convolves with the overlapped input windows (e.g. f 1 and f 2 in FIG0 ), some partially repeated calculations exist (e.g. the calculations highlight in green and indicated by arrow in FIG0 .

To eliminate such redundancy in convolution and speed-up WSNet, we propose a novel integral image method to enable efficient computation via sharing computations.

We first calculate an inner product map P ??? R T ??L * which stores the inner products between each row vector in the input feature map (i.e. F) and each column vector in the condensed filter (i.e. ??): calculates the inner product of each row in F and each column in ?? as in Eq. (6).

The convolution result between a filter k 1 which is sampled from ?? and the input patch f 1 is then the summation of all values in the segment between (u, v) and DISPLAYFORM1 DISPLAYFORM2 is the convolutional filter size).

Since there are repeated calculations when the filter and input patch are overlapped, e.g. the green segment indicated by arrow when performing convolution between k 2 and s 2 , we construct the integral image I using P according to Eq. (7).

Based on I, the convolutional results between any sampled filter and input patch can be retrieved directly in time complexity of O(1) according to Eq. (8), e.g. the results of DISPLAYFORM3 For notation definitions, please refer to Sec. 3.1.

The comparisons of computation costs between WSNet and the baselines using conventional architectures are introduced in Section 3.4.The integral image for speeding-up convolution is denoted as I. It has the same size as P and can be conveniently obtained throught below formulation: DISPLAYFORM4 Based on I, all convolutional results can be obtained in time complexity of O(1) as follows DISPLAYFORM5 Recall that the n-th filter lies in the spatial range of (nS, nS + L ??? 1) in the condensed filter ??. Since G ??? R T ??N , it thus takes T N times of calculating Eq. (8) to get G. In Eq. (6) ??? Eq. FORMULA11 , we omit the case of padding for clear description.

When zero padding is applied, we can freely get the convolutional results for the padded areas even without using Eq. FORMULA11 DISPLAYFORM6 Based on Eq. (6) ??? Eq. (8), the computation cost of the proposed integral image method is DISPLAYFORM7 Note the computation cost of P (i.e. Eq. FORMULA7 ) is the dominating term in Eq. (9).

Based on Eq. (5), Eq. FORMULA13 and Eq. (2), the theoretical acceleration ratio is DISPLAYFORM8 Recall that L is the filter size and S is the pre-defined stride when sampling filters from the condensed filter ?? (ref. to Eq. FORMULA2 ).In practice, we adopt a variant of above method to further boost the computation efficiency of WSNet, as illustrated in FIG2 In Eq. (6), we repeat ?? by C times along the channel dimension to Figure 3: A variant of the integral image method used in practice which is more efficient than that illustrated in FIG0 .

Instead of repeatedly sampling along the channel dimension of ?? to convolve with the input F, we wrap the channels of F by summing up C matrixes that are evenly divided from F along the channels, i.e.

F(i, j) = DISPLAYFORM9 Since the channle ofF is only 1/C of the channel of F, the overall computation cost is reduced as demonstrated in Eq. (10).

make it equal with the channel of the input F. However, we could first wrap the channels of F by accumulating the values with interval of L along its channel dimension to a thinner feature map F ??? R T ??M * which has the same channel number as ??, i.e.

F(i, j) = C???1 c=0 F(i, j + cM * ).

Both Eq. FORMULA10 and Eq. (8) remain the same.

Then the computational cost is reduced to DISPLAYFORM10 where the first item is the computational cost of warping the channels of F to obtainF. Since the dominating term (i.e. Eq. (6)) in Eq (10) is smaller than in Eq. (9), the overall computation cost is thus largely reduced.

By combining Eq. (10) and Eq. (5), the theoretical acceleration compared to the baseline is DISPLAYFORM11 Finally, we note that the integral image method applied in WSNet naturally takes advantage of the property in weight sampling: redundant computations exist between overlapped filters and input patches.

Different from other deep model speedup methods BID43 BID14 which require to solve time-consuming optimization problems and incur performance drop, the integral image method can be seamlessly embeded in WSNet without negatively affecting the final performance.

In this section, we present the details and analysis of the results in our experiments.

Extensive ablation studies are conducted to verify the effectiveness of the proposed WSNet on learning compact and efficient networks.

On all tested datasets, WSNet is able to improve the classification performance over the baseline networks while using 100?? smaller models.

When using even smaller (e.g. 180??) model size, WSNet achieves comparable performance w.r.t the baselines.

In addition, WSNet achieves 2?? ??? 4?? acceleration compared to the baselines with a much smaller model (more than 100?? smaller).

Datasets We collect a large-scale music detection dataset (MusicDet200K) from publicly available platforms (e.g. Facebook, Twitter, etc.) for conducting experiments.

For fair comparison with previous literatures, we also test WSNet on three standard, publicly available datasets, i.e ESC-50, UrbanSound8K and DCASE.

The details of used datasets are as follows.

MusicDet200K aims to assign a sample a binary label to indicate whether it is music or not.

MusicDet200K has overall 238,000 annotated sound clips.

Each has a time duration of 4 seconds and is resampled to 16000 Hz and normalized BID34 .

Among all samples, we use 200,000/20,000/18,000 as train/val/test set.

The samples belonging to "non-music" count for 70% of all samples, which means if we trivially assign all samples to be "non-music", the classification accuracy is 70%.ESC-50 (Piczak, 2015a) is a collection of 2000 short (5 seconds) environmental recordings comprising 50 equally balanced classes of sound events in 5 major groups (animals, natural soundscapes and water sounds, human non-speech sounds, interior/domestic sounds and exterior/urban noises) divided into 5 folds for cross-validation.

Following BID2 , we extract 10 sound clips from each recording with length of 1 second and time step of 0.5 second (i.e. two neighboring clips have 0.5 seconds overlapped).

Therefore, in each cross-validation, the number of training samples is 16000.

In testing, we average over ten clips of each recording for the final classification result.

UrbanSound8K BID40 ) is a collection of 8732 short (around 4 seconds) recordings of various urban sound sources (air conditioner, car horn, playing children, dog bark, drilling, engine idling, gun shot, jackhammer, siren and street music).

As in ESC-50, we extract 8 clips with the time length of 1 second and time step of 0.5 second from each recording.

For those that are less than 1 second, we pad them with zeros and repeat for 8 times (i.e. time step is 0.5 second).DCASE BID45 ) is used in the Detection and Classification of Acoustic Scenes and Events Challenge (DCASE).

It contains 10 acoustic scene categories, 10 training examples per category and 100 testing examples.

Each sample is a 30-second audio recording.

During training, we evenly extract 12 sound clips with time length of 5 seconds and time step of 2.5 seconds from each recording.

Evaluation criteria To demonstrate that WSNet is capable of learning more compact and efficient models than conventional CNNs, three evaluation criteria are used in our experiments: model size, the number of multiply and adds in calculation (mult-adds) and classification accuracy.

Baseline networks To test the scability of WSNet to different network architectures (e.g. whether having fully connected layers or not), two baseline networks are used in comparision.

The baseline network used on MusicDet200K consists of 7 convolutional layers and 2 fully connected layers, using which we demonstrate the effectiveness of WSNet on both convolutional layers and fully connected layers.

For fair comparison with previous literatures, we firstly modify the state-of-theart SoundNet BID2 by applying pooling layers to all but the last convolutional layer.

As can be seen in Table 5 , this modification significantly boosts the performance of original SoundNet.

We then use the modified SoundNet as a baseline on all three public datasets.

The architectures of the two baseline networks are shown in TAB1 respectively.

Weight Quantization Similar to other works BID17 BID36 , we apply weight quantization to further reduce the size of WSNet.

Specifically, the weights in each layer are linearly quantized to q bins where q is a pre-defined number.

By setting all weights in the same bin to the same value, we only need to store a small index of the shared weight for each weight.

The size of each bin is calculated as (max(??) ??? min(??))/q.

Given q bins, we only need log 2 (q) bits to encode the index.

Assuming each weight in WSNet is represented using 4 bytes float number (32 bits) without weight quantization, the ratio of each layer's size before and after weight quantization is DISPLAYFORM0 Recall that L * and M * are the spatial size and the channel number of condensed filter.

Since the condition L * M * q generally holds in most layers of WSNet, weight quantization is able to reduce the model size by a factor of 32 log 2 (q) .

Different from BID17 BID36 which learns the quantization during training, we apply weight quantization to WSNet Table 1 : Baseline-1: configurations of the baseline network used on MusicDet200K.

Each convolutional layer is followed by a nonlinearity layer (i.e. ReLU), batch normalization layer and pooling layer, which are omitted in the table for brevity.

The strides of all pooling layers are 2.

The padding strategies adopted for both convolutional layers and fully connected layers are all "size preserving".

after its training.

In the experiments, we find that such an off-line way is sufficient to reduce model size without losing accuracy.

Implementation details WSNet is implemented and trained from scratch in Tensorflow BID0 .

Following BID2 , the Adam (Kingma & Ba, 2014) optimizer, a fixed learning rate of 0.001, and momentum term of 0.9 and batch size of 64 are used throughout experiments.

We initialized all the weights to zero mean gaussian noise with a standard deviation of 0.01.

In the network used on MusicDet200K, the dropout ratio for the dropout layers BID44 after each fully connected layer is set to be 0.8.

The overall training takes 100,000 iterations.

Ablation analysis Through controlled experiments, we investigate the effects of each component in WSNet on the model size, computational cost and classification accuracy.

The comparative study results of different settings of WSNet are listed in TAB2 .

For clear description, we name WSNets with different settings by the combination of symbols S/C/SC ??? /D/Q. Please refer to the caption of TAB2 for detailed meanings.(1) Spatial sampling.

We test the performance of WSNet by using different sampling stride S in spatial sampling.

As listed in TAB2 , S 2 and S 4 slightly outperforms the classification accuracy of the baseline, possibly due to reducing the overfitting of models.

When the sampling stride is 8, i.e. the compactness in spatial dimension is 8 (ref. to Section 3.2.1), the classification accuracy of S 8 only drops slightly by 0.6%.

Note that the maximum compactness along the spatial dimension is equal to the filter size, thus for the layer "conv7" which has a filter size of 4, its compactness is limited by 4 (highlighted by underline in spatial sampling enables WSNet to learn significantly smaller model with comparable accuracies w.r.t.

the baseline.(2) Channel sampling.

Three different compactness along the channel dimension, i.e. 2, 4 and 8 are tested by comparing with baslines.

It can be observed from TAB2 that C 2 and C 4 and C 8 have linearly reduced model size without incurring noticeable drop of accuracy.

In fact, C 2 can even improve the accuracy upon the baseline, demonstrating the effectiveness of channel sampling in WSNet.

When learning more compact models, C 8 demonstrates better performance compared to S 8 tha has the same compactness in the spatial dimension, which suggests we should focus on the channel sampling when the compactness along the spatial dimension is high.

We then simultaneously perform weight sampling on both the spatial and channel dimensions.

As demonstrated by the results of S 4 C 4 SC ??? 4 and S 8 C 8 SC ??? 8 , WSNet can learn highly compact models (more than 20?? smaller than baselines) without noticeable performance drop (less than 0.5%).(3) Denser weight sampling.

Denser weight sampling is used to enhance the learning capability of WSNet with aggressive compactness (i.e. when S and C are large) and make up the performance loss caused by sharing too much parameters among filters.

As shown in TAB2 , by sampling 2?? more filters in conv1, conv2 and conv3, S 8 C 8 SC ??? 8 D 2 significantly outperforms the S 8 C 8 SC ???8 .

Above results demonstrate the effectiveness of denser weight sampling to boost the performance of WSNet.(4) Integral image for efficient computation.

As evidenced in the last column in TAB2 , the proposed integral image method consistently reduces the computation cost of WSNet.

For S 8 C 8 SC ??? 8 which is 23?? smaller than the baseline, the computation cost (in terms of #mult-adds) is significantly reduced by 16.4 times.

Due to the extra computation cost brought by the 1??1 convolution in denser TAB2 for the meaning of symbols S/C/D. Since the input lengths for the baseline are different in each dataset, we only provide the #Mult-Adds for UrbanSound8K.

Note that since we use the ratio of baseline's #Mult-Adds versus WSNet's #Mult-Adds for one WSNet, the numbers corresponding to WSNets in the column of #Mult-Adds are the same for all dataset.

Table 5 : Comparison with state-of-the-arts on ESC-50.

All results of WSNet are obtained by 10-folder validation.

Please refer to TAB2 scratch init.; provided data 65.8 ?? 0.25 4?? DISPLAYFORM0 Piczak ConvNet (Piczak, 2015b) scratch init.; provided data 64.5 28M SoundNet BID2 scratch init.; provided data 51.1 13M SoundNet BID2 pre-training; extra data 72.9 13M filter sampling, S 8 C 8 SC ??? 8 D 2 achieves lower acceleration (3.8??).

Group convolution BID49 can be used to alleviate the computation cost of the added 1??1 convolution layers.

We will explore this direction in our future work.(5) Weight quantization.

It can be observed from TAB2 that by using 256 bins to represent each weight by one byte (i.e. 8bits), S 8 C 8 SC ??? 15 A 2 Q 4 is reduced to 1/168 of the baseline's model size while incurring only 0.1% accuracy loss.

The above result demonstrates that the weight quantization is complementary to WSNet and they can be used jointly to effectively reduce the model size of WSNet.

Since we do not use weight quantization to accelerate models in this paper, the WSNets before and after weight quantization have the same computational cost.

The comparison of WSNet with other state-of-the-arts on ESC-50 is listed in Table 5 .

The settings of WSNet used on ESC-50, UrbanSound8K and DCASE are listed in TAB5 .

Compared with the baseline, WSNet is able to significantly reduce the model size of the baseline by 25 times and 45 times, while at the same time improving the accuracy of the baseline by 0.5% and 0.1% respectively.

The computation costs of WSNet are listed in TAB5 , from which one can observe that WSNet achieves higher computational efficiency by reducing the #Mult-Adds of the baseline by 2.3?? and 2.4??, respectively.

Such promising results again demonstrate the effectiveness of WSNet on learning compact and efficient networks.

After applying weight quantization to WSNet, its model size is reduced to only 1/180 of the baseline while the accuracy only slightly drops by 0.2%.

Compared with the SoundNet trained from scratch with provided data, WSNets significantly outperform its classification accuracy by over 10% with more than 100?? smaller models.

Using a transfer learning approach, SoundNet BID2 that is pre-trained using a large number of unlabeled videos achieves better accuracy than WSNet.

However, since the training method is DISPLAYFORM0 RNH BID37 scratch init.

; provided data 77 -Ensemble BID46 scratch init.; provided data 78 -SoundNet BID2 pre-training; extra data 88 13Morthogonal to WSNet, we believe that WSNet can achieve better performance by training in a similar way as SoundNet BID2 on a large amount of unlabeled video data.

We report the comparison results of WSNet with state-of-the-arts on UrbanSound8k in TAB6 .

It is again observed that WSNet significantly reduces the model size of baseline while obtaining comparative results.

Both Piczak (2015b) and BID38 use pre-computed 2D features after log-mel transformation as input.

In comparison, the proposed WSNet simply takes the raw wave of recordings as input, enabling the model to be trained in an end-to-end manner.

As evidenced in TAB7 , WSNet outperforms the classification accuracy of the baseline by 1% with a 100?? smaller model.

When using an even more compact model, i.e. 180?? smaller in model size.

The classification accuracy of WSNet is only one percentage lower than the baseline (i.e. has only one more incorrectly classified sample), verifying the effectiveness of WSNet.

Compared with SoundNet BID2 that utilizes a large number of unlabeled data during training, WSNet (S 8 C 4 D 2 Q 4 ) that is 100?? smaller achieves comparable results only by using the provided data.

In this paper, we present a class of Weight Sampling networks (WSNet) which are highly compact and efficient.

A novel weight sampling method is proposed to sample filters from condensed filters which are much smaller than the independently trained filters in conventional networks.

The weight sampling in conducted in two dimensions of the condensed filters, i.e. by spatial sampling and channel sampling.

TAB2

To further verify WSNet's capacity of learning compact models, we conduct experiments on ESC-50 and MusicDet200K to compare WSNet with baselines compressed in an intuitive way, i.e. reducing the number of filters in each layer.

If #filters in each layer is reduced by T , the overall #parameters in baselines is reduced by T 2 (i.e. the compression ratio of model size is T 2 ).

In Figure 4 and Figure 5 , we plot how baseline accuracy varies with respect to different compression ratios and the accuracies of WSNet with the same model size of compressed baselines.

As shown in Figure 4 and Figure 5 , WSNet outperforms baselines by a large magin across all compression ratios.

Particularly, when the comparison ratios are large (e.g. 45 on ESC-50 and 42 on MusicDet200K), In this paper, we focus on WSNet with 1D convnets.

Comprehensive experiments clearly demonstrate its advantages in learning compact and computation-efficient networks.

We note that WSNet is general and can also be applied to build 2D convnets.

In 2D convnets, each filter has three dimensions including two spatial dimensions (i.e. along X and Y directions) and one channel dimension.

One straightforward extension of WSNet to 2D convnets is as follows: for spatial sampling, each filter is sampled out as a patch (with the same number of channels as in condensed filter) from condensed filter.

Channel sampling remains the same as in 1D convnets, i.e. repeat sampling in the channel dimension of condensed filter.

Following the notations for WSNet with 1D convnets (ref. to Sec. 3.1), we denote the filters in one layer as K ??? R w??h??M ??N where (w, h, M, N ) denote the width and height of each filter, the number of channels and the number of filters respectively.

The condensed filter ?? has the shape of (W, H, M * ).

The relations between the shape of condensed filter and each sampled filter are: DISPLAYFORM0 where Sw and S h are the sampling strides along two spatial dimensions and C is the compactness of WSNet along channel dimension.

The compactness (please refer to Sec. 3.1 for denifinition) of WSNet along spatial dimension is

.

However, such straightforward extension of WSNet to 2D convnets is not optimum due to following two reasons: (1) Compared to 1D filters, 2D filters present stronger spatial dependencies between the two spatial dimensions.

Nave extension may fail to capture such dependencies.

(2) It is not easy to use the integral image method for speeding up WSNet in 2D convnets as in 1D convnets.

Because of above problems, we believe there are more sophisticated and effective methods for applying WSNet to 2D convnets and we would like to explore in our future work.

Nevertheless, we conduct following preliminary experiments on 2D convents using above intuitive extension and verify the effectiveness of WSNet in image classification tasks (on MNIST and CIFAR10).

Since both WSNet and HashNet BID6 BID9 explore weights tying, we compare them on MNIST and CIFAR10.

For fair comparison, we use the same baselines used in BID6 BID9 .

The baseline used for MNIST is a 3-layer fully connected network with a single hidden layer containing 1,000 hidden units.

The configuration of the baseline network used for CIFAR10 is listed in TAB10 .

All hyperparameters used training including learning rate, momentum, drop out and so on follow BID6 BID9 .

For each dataset, we hold out 20% of the whole training samples to form a validation set.

The comparison results between WSNet and HashNet on MNIST/CIFAR10 are listed in TAB11 , respectively.

As one can observe in TAB11 , when learning networks with the same sizes, WSNet achieves significant lower error rates than HashNet on both datasets.

Above results clearly demonstrate the advantages of WSNet in learning compact models.

Furthermore, we also conduct experiment on CIFAR10 with the state-of-the-art ResNet18 BID19 network as baseline.

Both the network architecture and training hyperparameters follow BID19 .

WSNet is able to achieve 20?? smaller model size with slight performance drop (0.6%).

Such promising results further demonstrate the effectiveness of WSNet.

@highlight

We present a novel network architecture for learning compact and efficient deep neural networks