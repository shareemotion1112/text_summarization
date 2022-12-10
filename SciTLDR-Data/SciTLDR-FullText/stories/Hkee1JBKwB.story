Long-term video prediction is highly challenging since it entails simultaneously capturing spatial and temporal information across a long range of image frames.

Standard recurrent models are ineffective since they are prone to error propagation and cannot effectively capture higher-order correlations.

A potential solution is to extend to higher-order spatio-temporal recurrent models.

However, such a model requires  a  large number of parameters and operations, making it intractable  to learn in practice and is prone to overfitting.

In this work, we propose convolutional tensor-train LSTM (Conv-TT-LSTM), which  learns higher-orderConvolutional LSTM (ConvLSTM) efficiently using convolutional  tensor-train decomposition (CTTD).

Our proposed model naturally incorporates higher-order spatio-temporal information at a small cost of memory and computation by using efficient low-rank tensor representations.

We evaluate our model on Moving-MNIST and KTH datasets and show improvements over standard ConvLSTM and better/comparable results to other ConvLSTM-based approaches, but with much fewer parameters.

Understanding dynamics of videos and performing long-term predictions of the future is a highly challenging problem.

It entails learning complex representation of real-world environment without external supervision.

This arises in a wide range of applications, including autonomous driving, robot control , or other visual perception tasks like action recognition or object tracking (Alahi et al., 2016) .

However, long-term video prediction remains an open problem due to high complexity of the video contents.

Therefore, prior works mostly focus on next or first few frames prediction (Lotter et al., 2016; Finn et al., 2016; Byeon et al., 2018) .

Many recent video models use Convolutional LSTM (ConvLSTM) as a basic block (Xingjian et al., 2015) , where spatio-temporal information is encoded as a tensor explicitly in each cell.

In ConvL-STM networks, each cell is a first-order recurrent model, where the hidden state is updated based on its immediate previous step.

Therefore, they cannot easily capture higher-order temporal correlations needed for long-term prediction.

Moreover, they are highly prone to error propagation.

Various approaches have been proposed to augment ConvLSTM, either by modifying networks to explicitly modeling motion (Finn et al., 2016) , or by integrating spatio-temporal interaction in ConvLSTM cells (Wang et al., 2017; 2018a) .

These approaches are often incapable of capturing longterm dependencies and produce blurry prediction.

Another direction to augment ConvLSTM is to incorporate a higher-order RNNs (Soltani & Jiang, 2016) inside each LSTM cell, where its hidden state is updated using multiple past steps.

However, a higher-order model for high-dimensional data (e.g. video) requires a huge number of model parameters, and the computation grows exponentially with the order of the RNNs.

A principled approach to address the curse of dimensionality is tensor decomposition, where a higher-order tensor is compressed into smaller core tensors (Anandkumar et al., 2014) .

Tensor representations are powerful since they retain rich expressivity even with a small number of parameters.

In this work, we propose a novel convolutional tensor decomposition, which allows for compact higher-order ConvLSTM.

Contributions.

We propose Convolutional Tensor-Train LSTM (Conv-TT-LSTM), a modification of ConvLSTM, to build a higher-order spatio-temporal model.

(1) We introduce Convolutional Tensor-Train Decomposition (CTTD) that factorizes a large convolutional kernel into a chain of

Figure 1: Illustration of (a) convolutional tensor-train (Eqs. (5) and (6)) and the difference between convolutional tensor-train LSTM (b) Fixed window version (Eqs. (11a) and (10)) and (c) Sliding window version (Eqs. (11b) and (10) and 1c ), and we found that the SW version performs better than the FW one.

(4) We found that training higher-order tensor models is not straightforward due to gradient instability.

We present several approaches to overcome this such as good learning schedules and gradient clipping.

(5) In the experiments, we show our proposed Conv-TT-LSTM consistently produces sharp prediction over a long period of time for both Moving-MNIST-2 and KTH action datasets.

Conv-TT-LSTM outperforms the state-of-the-art PredRNN++ (Wang et al., 2018a) in LPIPS (Zhang et al., 2018) by 0.050 on the Moving-MNIST-2 and 0.071 on the KTH action dataset, with 5.6 times fewer parameters.

Thus, we obtain best of both worlds: better long-term prediction and model compression.

Tensor Decomposition In machine learning, tensor decompositions, including CP decomposition (Anandkumar et al., 2014) , Tucker decomposition (Kolda & Bader, 2009) , and tensor-train decomposition (Oseledets, 2011), are widely used for dimensionality reduction (Cichocki et al., 2016) and learning probabilistic models (Anandkumar et al., 2014) .

In deep learning, prior works focused on their application in model compression, where the parameters tensors are factorized into smaller tensors.

This technique has been used in compressing convolutional networks (Lebedev et al., 2014; Kim et al., 2015; Novikov et al., 2015; Su et al., 2018; Kossaifi et al., 2017; Kolbeinsson et al., 2019; , recurrent networks (Tjandra et al., 2017; Yang et al., 2017) and transformers (Ma et al., 2019) .

Specifically, Yang et al. (2017) demonstrates that the accuracy of video classification increases if the parameters in recurrent networks are compressed by tensor-train decomposition (Oseledets, 2011 ).

Yu et al. (2017 used tensor-train decomposition to constrain the complexity of higher-order LSTM, where each next step is computed based on the outer product of previous steps.

While this work only considers vector input at each step, we extend their approach to higher-order ConvLSTM, where each step also encodes spatial information.

Video Prediction Prior works on video prediction have focused on several directions: predicting short-term video (Lotter et al., 2016; Byeon et al., 2018) , decomposing motion and contents (Finn et al., 2016; Villegas et al., 2017; Denton et al., 2017; Hsieh et al., 2018) , improving the objective function Mathieu et al. (2015) , and handling diversity of the future (Denton & Fergus, 2018; Babaeizadeh et al., 2017; Lee et al., 2018) .

Many of these works use Convolutional LSTM (ConvL-STM) (Xingjian et al., 2015) as a base module, which deploys 2D convolutional operations in LSTM to efficiently exploit spatio-temporal information.

Finn et al. (2016) used ConvLSTM to model pixel motion.

Some works modified the standard ConvLSTM to better capture spatio-temporal correlations (Wang et al., 2017; 2018a) .

Wang et al. (2018b) integrated 3D convolutions into ConvLSTM.

In addition, current cell states are combined with its historical records using self-attention to efficiently recall the history information.

Byeon et al. (2018) applied ConvLSTM in all possible directions to capture full contexts in video and also demonstrated strong performance using a deep ConvLSTM network as a baseline.

This baseline is adapted to obtain the base architecture in the present paper.

The goal of tensor decomposition is to represent a higher-order tensor as a set of smaller and lowerorder core tensors, with fewer parameters while preserve essential information.

In Yu et al. (2017), tensor-train decomposition (Oseledets, 2011) is used to reduce both parameters and computations in higher-order recurrent models, which we review in the first part of this section.

However, the approach in Yu et al. (2017) only considers recurrent models with vector inputs and cannot cope with image inputs directly.

In the second part, we extend the standard tensor-train decomposition to convolutional tensor-train decomposition (CTTD).

With CTTD, a large convolutional kernel is factorized into a chain of smaller kernels.

We show that such decomposition can reduce both parameters and operations of higher-order spatio-temporal recurrent models.

Standard Tensor-train decomposition Given an m-order tensor T ∈ R I1×···×Im , where I l is the dimension of its l-th order, a standard tensor-train decomposition (TTD) (Oseledets, 2011) factorizes the tensor T into a set of m core tensors

where tensor-train ranks {R l } m l=0 (with R 0 = R m = 1) control the number of parameters in the tensor-train format Eq.(1).

With TTD, the original tensor T of size (

entries, which grows linearly with the order m (assuming R l 's are constant).

Therefore, TTD is commonly used to approximate higher-order tensors with fewer parameters.

The sequential structure in tensor-train decomposition makes it particularly suitable for sequence modeling (Yu et al., 2017) .

Consider a higher-order recurrent model that predicts a scalar output v ∈ R based on the outer product of a sequence of input vectors {u

according to:

This model is intractable in practice since the number of parameters in T ∈ R I1×···Im (and therefore computational complexity of Eq. (2)) grows exponentially with the order m. Now suppose T takes a tensor-train format as in Eq. (1), we prove in Appendix A that (2) can be efficiently computed as

where the vectors {v

are the intermediate steps, with v (0) ∈ R initialized as v (0) = 1, and final output v = v (m) .

Notice that the higher-order tensor T is never reconstructed in the sequential process in Eq. (3), therefore both space and computational complexities grow linearly (not exponentially compared to Eq. (2))with the order m assuming all tensor-train ranks are constants.

Convolutional Tensor-Train Decomposition A convolutional layer in neural network is typically parameterized by a 4-th order tensor T ∈ R K×K×Rm×R0 , where K is the kernel size, R m and R 0 are the number of input and output channels respectively.

Suppose the kernel size K takes the form K = m(k − 1) + 1 (e.g. K = 7 and m = 3, k = 3), a convolutional tensor-train decomposition (CTTD) factorizes T into a set of m core tensors

:,:,r1,r0 * T

where * denotes convolution between 2D-filters, and {R l } m l=1 are the convolutional tensor-train ranks that control the complexity of the convolutional tensor-train format in Eq. (4).

With CTTD, the number of parameters in the decomposed format reduces from

Similar to standard TTD, its convolutional counterpart can also be used to compress higher-order spatio-temporal recurrent models with convolutional operations.

Consider a model that predicts a 3-rd order feature V ∈ R H×W ×R0 based on a sequence of 3-rd features

(where H, W are height/width of the features and R l is the number of channels in U (l) ) such that

where

is the corresponding weights tensor for U (l) .

Suppose each W (l) takes a convolutional tensor-train format in Eq. (4), we prove in Appendix A that the model in Eq. (5) can be computed sequentially similarly without reconstructing the original W (l) 's:

where

are intermediate results of the sequential process, where V (m) ∈ R H×W ×Rm is initialized as all zeros and final prediction V = V (0) .

The operations in Eq. (5) is illustrated in Figure 1a .

In this paper, we denote the Eq. (5)

).

Convolutional LSTM is a basic block for most recent video forecasting models (Xingjian et al., 2015) , where the spatial information is encoded explicitly as tensors in the LSTM cells.

In a ConvLSTM network, each cell is a first-order Markov model, i.e. the hidden state is updated based on its previous step.

In this section, we propose convolutional tensor-train LSTM, where convolutional tensor-train is incorporated to model multi-steps spatio-temporal correlation explicitly.

Notations.

In this section, the symbol * is overloaded to denote convolution between higher-order tensors.

For instance, given a 4-th order weights tensor W ∈ R K×K×S×C and a 3-rd order input tensor X ∈ R H×W ×S , Y = W * X computes a 3-rd output tensor Y ∈ R H×W ×T as Y :,:,c = s=1 W :,:,s,c * X :,:,s .

The symbol • is used to denote element-wise product between two tensors, and σ represents a function that performs element-wise (nonlinear) transformation on a tensor.

Xingjian et al. (2015) extended fully-connected LSTM (FC-LSTM) to Convolutional LSTM (ConvLSTM) to model spatio-temporal structures within each recurrent unit, where all features are encoded as 3-rd order tensors with dimensions (height × width × channels) and matrix multiplications are replaced by convolutions between tensors.

In a ConvLSTM cell, the parameters are characterized by two 4-th order tensors W ∈ R K×K×S×4C and T ∈ R K×K×C×4C , where K is the kernel size of all convolutions and S and C are the numbers of channels of the input X (t) ∈ R H×W ×S and hidden states H (t) ∈ R H×W ×C respectively.

At each time step t, a ConvLSTM cell updates its hidden states H (t) ∈ R H×W ×C based on the previous step H (t−1) and the current input X (t) , where H and W are the height/width that are the same for X (t) and H (t) .

where σ(·) applies sigmoid on the input gate I (t) , forget gate F (t) , output gate O (t) , and hyperbolic tangent on memory cellC (t) .

Note that all tensors

Convolutional Tensor-Train LSTM In Conv-TT-LSTM, we introduce a higher-order recurrent unit to capture multi-steps spatio-temporal correlations in LSTM, where the hidden state H (t) is updated based on its n previous steps

with an m-order convolutional tensor-train (CTT) as in Eq. (5).

Concretely, suppose the parameters in CTT are characterized by m tensors of 4-th order

, Conv-TT-LSTM replaces Eq. (7) in ConvLSTM by two equations:

(10)

, ·) takes a sequence of m tensors as inputs, the first step in Eq. (9) maps the n inputs

, ·) and compute the gates according to Eq. (10).

We propose two realizations of Eq. (9), where the first realization uses a fixed window of

to compute eachH (t,o) , while the second one adopts a sliding window strategy.

At each step, the Conv-TT-LSTM model computes H (t) by replacing Eq. (9) by either Eq. (11a) or (11b).

Conv-TT-LSTM-SW:

In the fixed window version, the previous steps {H (l) } n l=1 are concatenated into a 3-rd order tensor H (t,o) ∈ R H×W ×nC , which is then mapped to a tensorH (t,o) ∈ R H×W ×R by 2D-convolution with a kernel K (l) ∈ R k×k×nC×R .

And in the sliding window version, {H (l) } n l=1 are concatenated into a 4-th order tensorĤ (t,o) ∈ R H×W ×D×C (with D = n − m + 1), which is mapped toH (t,o) ∈ R H×W ×R by 3D-convolution with a kernel K (l) ∈ R k×k×D×R .

For later reference, we name the model with Eqs.(11a) and (10)

We first evaluate our approach extensively on the synthetic Moving-MNIST-2 dataset (Srivastava et al., 2015) .

In addition, we use KTH human action dataset (Laptev et al., 2004) to test the performance of our models in more realistic scenario.

Model Architecture All experiments use a stack of 12-layers of ConvLSTM or Conv-TT-LSTM with 32 channels for the first and last 3 layers, and 48 channels for the 6 layers in the middle.

A convolutional layer is applied on top of all LSTM layers to compute the predicted frames.

Following Byeon et al. (2018) , two skip connections performing concatenation over channels are added between (3, 9) and (6, 12) layers.

Illustration of the network architecture is included in the appendix.

All parameters are initialized by Xavier's normalized initializer (Glorot & Bengio, 2010) and initial states in ConvLSTM or Conv-TT-LSTM are initialized as zeros.

Evaluation Metrics We use two traditional metrics MSE (or PSNR) and SSIM (Wang et al., 2004) , and a recently proposed deep-learning based metric LPIPS (Zhang et al., 2018) , which measures the similarity between deep features.

Since MSE (or PSNR) is based on pixel-wise difference, it favors vague and blurry predictions, which is not a proper measurement of perceptual similarity.

While SSIM was originally proposed to address the problem, Zhang et al. (2018) shows that their proposed LPIPS metric aligns better to human perception.

Learning Strategy All models are trained with ADAM optimizer (Kingma & Ba, 2014) with L 1 + L 2 loss.

Learning rate decay and scheduled sampling (Bengio et al., 2015) are used to ease training.

Scheduled sampling is started once the model does not improve in 20 epochs (in term of validation loss), and the sampling ratio is decreased linearly from 1 until it reaches zero (by 2 × 10 −4 each epoch for Moving-MNIST-2 and 5 × 10 −4 for KTH).

Learning rate decay is further activated if the loss does not drop in 20 epochs, and the rate is decreased exponentially by 0.98 every 5 epochs.

We perform a wide range of hyper-parameters search for Conv-TT-LSTM to identify the best model, and of 10 −3 is found for the models of kernel size 3 and 10 −4 for the models of kernel size 5.

We found that Conv-TT-LSTM models suffer from exploding gradients when learning rate is high (e.g. 10 −3 in our experiments), therefore we also explore various gradient clipping values and select 1 for all Conv-TT-LSTM models.

All hyper-parameters are selected using the best validation performance.

The Moving-MNIST-2 dataset is generated by moving two digits of size 28 × 28 in MNIST dataset within a 64 × 64 black canvas.

These digits are placed at a random initial location, and move with constant velocity in the canvas and bounce when they reach the boundary.

Following Wang et al. (2018a) , we generate 10,000 videos for training, 3,000 for validation, and 5,000 for test with default parameters in the generator 1 .

All our models are trained to predict 10 frames given 10 input frames.

All our models use kernel size 5: Conv-TT-LSTM-FW has hyperparameters as (order 1, steps 3, ranks 8), and Conv-TT-LSTM-SW has hyperparameters as (order 3, steps 3, ranks 8).

Figure 2: Frame-wise comparison in MSE, SSIM and PIPS on Moving-MNIST-2.

For MSE and LPIPS, lower curves denote higher quality; while for SSIM, higher curves imply better quality.

Table 2 reports the average statistics for 10 and 30 frames prediction, and Figure 2 shows comparison of per-frame statistics for PredRNN++ model, ConvLSTM baseline and our proposed Conv-TT-LSTM models.

(1) Our Conv-TT-LSTM models consistently outperform the 1 https://github.com/jthsieh/DDPAE-video-prediction/blob/master/data/moving_mnist.py 2 The results are cited from the original paper, where the miscalculation of MSE is corrected in the table.

3 The results are reproduced from https://github.com/Yunbo426/predrnn-pp with the same datasets in this paper.

The original implementation crops each frame into patches as the input to the model.

We find out such pre-processing is unnecessary and the performance is better than the original paper.

12-layer ConvLSTM baseline for both 10 and 30 frames prediction with fewer parameters; (2) The Conv-TT-LSTMs outperform previous approaches in terms of SSIM and LPIPS (especially on 30 frames prediction), with less than one fifth of the model parameters.

We reproduce the PredRNN++ model (Wang et al., 2018a ) from their source code 2 , and we find that (1) The PredRNN++ model tends to output vague and blurry results in long-term prediction (especially after 20 steps).

(2) and our Conv-TT-LSTMs are able to produce sharp and realistic digits over all steps.

An example of comparison for different models is shown in Figure 3 .

The visualization is consistent with the results in Ablation Study To understand whether our proposed Conv-TT-LSTM universally improves upon ConvLSTM (i.e. not tied to specific architecture, loss function and learning schedule), we perform three ablation studies: (1) Reduce the number of layers from 12 layers to 4 layers (same as Xingjian et al. (2015) and Wang et al. (2018a) ); (2) Change the loss function from L 1 + L 2 to L 1 only; (3) Disable the scheduled sampling and use teacher forcing during training process.

We evaluate the ConvLSTM baseline and our proposed Conv-TT-LSTM in these three settings, and summarize their comparisons in Table 3 .

The results show that our proposed Conv-TT-LSTM outperforms ConvLSTM consistently for all settings, i.e. the Conv-TT-LSTM model improves upon ConvLSTM in a board range of setups, which is not limited to the certain setting used in our paper.

These ablation studies further show that our setup is optimal for predictive learning in Moving-MNIST-2.

KTH action dataset (Laptev et al., 2004) contains videos of 25 individuals performing 6 types of actions on a simple background.

Our experimental setup follows Wang et al. (2018a) , which uses persons 1-16 for training and 17-25 for testing, and each frame is resized to 128 × 128 pixels.

All our models are trained to predict 10 frames given 10 input frames.

During training, we randomly select 20 contiguous frames from the training videos as a sample and group every 10,000 samples into one epoch to apply the learning strategy as explained at the beginning of this section. (Villegas et al., 2017) 25.95 0.804 -----E3D-LSTM (Wang et al., 2018b) 29 Results In Table 4 , we report the evaluation on both 20 and 40 frames prediction.

(1) Our models are consistently better than the ConvLSTM baseline for both 20 and 40 frames prediction.

(2) While our proposed Conv-TT-LSTMs achieve lower SSIM value compared to the state-of-the-art models in 20 frames prediction, they outperform all previous models in LPIPS for both 20 and 40 frames prediction.

An example of the predictions by the baseline and Conv-TT-LSTMs is shown in Figure 3 .

In this paper, we proposed convolutional tensor-train decomposition to factorize a large convolutional kernel into a set of smaller core tensors.

We applied this technique to efficiently construct convolutional tensor-train LSTM (Conv-TT-LSTM), a high-order spatio-temporal recurrent model whose parameters are represented in tensor-train format.

We empirically demonstrated that our proposed Conv-TT-LSTM outperforms standard ConvLSTM and produce better/comparable results compared to other state-of-the-art models with fewer parameters.

Utilizing the proposed model for high-resolution videos is still challenging due to gradient vanishing or explosion.

Future direction will include investigating other training strategies or a model design to ease the training process.

In this section, we prove the sequential algorithms in Eq. (3) for tensor-train decomposition (1) and Eq. (6) for convolutional tensor-train decomposition (4) both by induction.

Proof of Eq. (3) For simplicity, we denote the standard tensor-train decomposition in Eq. (1) as

), then Eq. (2) can be rewritten as Eq. (12) since R 0 = 1 and v

i1,r0,r1 · · · T

where R 0 = 1, v

@highlight

we propose convolutional tensor-train LSTM,  which learns higher-order Convolutional LSTM efficiently using convolutional tensor-train decomposition. 