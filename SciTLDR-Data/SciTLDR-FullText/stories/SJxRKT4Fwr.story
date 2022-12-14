Many real-world applications involve multivariate, geo-tagged time series data: at each location, multiple sensors record corresponding measurements.

For example, air quality monitoring system records PM2.5, CO, etc.

The resulting time-series data often has missing values due to device outages or communication errors.

In order to impute the missing values, state-of-the-art methods are built on Recurrent Neural Networks (RNN), which process each time stamp sequentially, prohibiting the direct modeling of the relationship between distant time stamps.

Recently, the self-attention mechanism has been proposed for sequence modeling tasks such as machine translation, significantly outperforming RNN because the relationship between each two time stamps can be modeled explicitly.

In this paper, we are the first to adapt the self-attention mechanism for multivariate, geo-tagged time series data.

In order to jointly capture the self-attention across different dimensions (i.e. time, location and sensor measurements) while keep the size of attention maps reasonable, we propose a novel approach called Cross-Dimensional Self-Attention (CDSA) to process each dimension sequentially, yet in an order-independent manner.

On three real-world datasets, including one our newly collected NYC-traffic dataset, extensive experiments demonstrate the superiority of our approach compared to state-of-the-art methods for both imputation and forecasting tasks.

Various monitoring applications, such as those for air quality (Zheng et al. (2015) ), health-care (Silva et al. (2012) ) and traffic (Jagadish et al. (2014) ), widely use networked observation stations to record multivariate, geo-tagged time series data.

For example, air quality monitoring systems employ a collection of observation stations at different locations; at each location, multiple sensors concurrently record different measurements such as PM2.5 and CO over time.

Such time series are important for advanced investigation and also are useful for future forecasting.

However, due to unexpected sensor damages or communication errors, missing data is unavoidable.

It is very challenging to impute the missing data because of the diversity of the missing patterns: sometimes almost random while sometimes following various characteristics.

Traditional data imputation methods usually suffer from imposing strong statistical assumptions.

For example, Scharf & Demeure (1991) and Friedman et al. (2001) fit a smooth curve on observations in either time series (Ansley & Kohn (1984) ; Shumway & Stoffer (1982) ) or spatial distribution (Friedman et al. (2001); Stein (2012) ).

Deep learning methods (Li et al. (2018) ; Che et al. (2018); Cao et al. (2018) ; Luo et al. (2018a) ) have been proposed to capture temporal relationship based on RNN (Cho et al. (2014b) ; Hochreiter & Schmidhuber (1997) ; Cho et al. (2014a) ).

However, due to the constraint of sequential computation over time, the training of RNN cannot be parallelized and thus is usually time-consuming.

Moreover, the relationship between each two distant time stamps cannot be directly modeled.

Recently, the self-attention mechanism as shown in Fig. 1(b) has been proposed by the seminal work of Transformer (Vaswani et al. (2017) ) to get rid of the limitation of sequential processing, accelerating the training time substantially and improving the performance significantly on seq-to-seq tasks in Natural Language Processing (NLP) because the relevance between each two time stamps is captured explicitly.

In this paper, we are the first to adapt the self-attention mechanism to impute missing data in multivariate time series, which cover multiple geo-locations and contain multiple measurements as Figure 1: (a) Illustration of the multivariate, geo-tagged time series imputation task: the input data has three dimensions (i.e. time, location, measurement) with some missing values (indicated by the orange dot); the output is of same shape as the input while the missing values have been imputed (indicated by the red dot).

(b) Self-attention mechanism: the Attention Map is first computed using every pair of Query vector and Key vector and then guides the updating of Value vectors via weighted sum to take into account contextual information.

(c) Traditional Self-Attention mechanism updates Value vector along the temporal dimension only vs. Cross-Dimensional Self-Attention mechanism updates Value vector according to data across all dimensions.

shown in Fig. 1(a) .

In order to impute a missing value in such unique multi-dimensional data, it is very useful to look into available data in different dimensions (i.e. time, location and measurement), as shown in Fig. 1(c) , to capture the intra-correlation individually.

To this end, we investigate several choices of modeling self-attention across different dimensions.

In particular, we propose a novel Cross-Dimensional Self-Attention (CDSA) mechanism to capture the attention crossing all dimension jointly yet in a decomposed manner.

In summary, we make the following contributions:

(i) We are the first to apply the self-attention mechanism to the multivariate, geo-tagged time series data imputation task, replacing the conventional RNN-based models to speed up training and directly model the relationship between each two data values in the input data.

(ii) For such unique time series data of multiple dimensions (i.e. time, location, measurement), we comprehensively study several choices of modeling self-attention crossing different dimensions.

Our proposed CDSA mechanism models self-attention crossing all dimensions jointly yet in a dimension-wise decomposed way, preventing the size of attention maps from being too large to be tractable.

We show that CDSA is independent with the order of processing each dimension.

(iii) We extensively evaluate on two standard benchmarks and our newly collected traffic dataset.

Experimental results show that our model outperforms the state-of-the-art models for both data imputation and forecasting tasks.

We visualize the learned attention weights which validate the capability of CDSA to capture important cross-dimensional relationships.

Statistical data imputation methods.

Statistical methods (Ansley & Kohn (1984) (GAN) .

Nevertheless, the spatiotemporal and measurements correlation are mixed and indistinguishable.

so that the mediate back propagation from loss of available observation can contribute to the missing value updating.

Nevertheless, these RNN-based models fundamentally suffer from the constraint of sequential processing, which leads to long training time and prohibits the direct modeling of the relationship between two distant data values.

Self-attention.

Recently, Vaswani et al. (2017) introduced the Transformer framework which relies on self-attention, learning the association between each two words in a sentence.

Then self-attention has been widely applied in seq-to-seq tasks such as machine translation, image generation (Yang et al. (2016); Zhang et al. (2018a) ) and graph-structured data (Veli??kovi?? et al. (2017) ).

In this paper, we are the first to apply self-attention for multi-dimensional data imputation and specifically we investigate several choices of modeling self-attention crossing different data dimensions.

In Sec. 3.1, we first review the conventional self-attention mechanism in NLP.

In Sec. 3.2, we propose three methods for computing attention map cross different dimension.

In Sec. 3.3 and 3.4, we present details of using CDSA for missing data imputation.

As shown in Fig. 1(b) , for language translation task in NLP, given an input sentence, each word x i is mapped into a Query vector q i of d-dim, a Key vector k i of d-dim, and a Value vector v i of v-dim.

The attention from word x j to word x i is effectively the scaled dot-product of q i and k j after Softmax, which is defined as A(i, j) = exp(S(i, j)) T j=1 exp(S(q, j)) ???1 where Then, v i is updated to v i as a weighted sum of all the Value vectors, defined as v i = T j=1 A(i, j)v j , after which each v i is mapped to the layer output x i of the same size as x i .

In order to adapt the self-attention from NLP to our multivariate, geo-tagged time series data, a straightforward way is to view all data in a time stamp as one word embedding and model the self-attention over time.

In order to model Cross-Dimensional Self-Attention (CDSA), in this section we propose three solutions: (1) model attention within each dimension independently and perform late fusion; (2) model attention crossing all dimension jointly; (3) model attention crossing all dimension in a joint yet decomposed manner.

We assume the input X ??? R T ??L??M has three dimensions corresponding time, location, measurement.

X can be reshaped into 2-D matrices (i.e.

.

Similarly, this subscript may be applied on the Query, Key and Value, e.g.,

As shown in Fig. 2(a) , the input X is reshaped into three input matrices X T , X L and X M .

Three streams of self-attention layers are built to process each input matrix in parallel.

Such as the first layer in stream on X L , each vector X L (l, :) of M T -dim is viewed as a word vector in NLP.

Following the steps in Sec. 3.1,

The output of every stream's last layer are fused through element-wise addition,

where the weights ?? T , ?? L and ?? M are trainable parameters.

Besides, the hyper-parameters for each stream such as the number of layers, are set separately.

As shown in Fig. 2(b) , the three-dimensional input X is reshaped as to X. Each unit X(p) is mapped to Q(p, :) and K(p, :) of d-dim as well as V (p, :) of v-dim, where p = p(t, l, m) denotes the index mapping from the 3-D cube to the vector form.

In this way, an attention map of dimension T LM ?? T LM is built to directly model the cross-dimensional interconnection.

The Independent manner sets multiple attention sub-layers in each stream to model the dimensionspecific attention but fail in modeling cross-dimensional dependency.

In contrast, the Joint manner learns the cross-dimensional relationship between units directly but results in huge computation workload.

To capture both the dimension-specific and cross-dimensional attention in a distinguishable way, we propose a novel Decomposed manner.

As shown in Fig. 2(c) , the input X is reshaped as input matrices X T , X L , X M and X. Each unit X(p) is mapped to vector V (p, :) of v-dim as in the Joint while X T , X L and X M are used for building attention map A T , A L ,A M individually as in the Independent.

The attention maps are applied on Value vector in order as,

The attention map with is reshaped from the original attention map and consistent with the calculation in (1), e.g., A T ??? R T LM ??T LM is reshaped from A T ??? R T ??T .

More specifically,

where ??? denotes tensor product and I is the Identity matrix where the subscript indicates the size, e.g., I T ??? R T ??T .

Although the three reshaped attention maps are applied with a certain order, according to (2), we show that each unit in A is effectively calculated as

where

.

Following the associativity of tensor product, we demonstrate

where ?? = ??(T,L,M) denotes the arbitrary arrangement of sequence (T,L,M), e.g., ?? =(L,T,M) and ??(T) = L. Effectively, the arrangement ?? is the order of attention maps to update V .

As (3)-(4) shows that the weight in A is decomposed as the product of weights in three dimension-specific attention maps, the output and gradient back propagation are order-independent.

Furthermore, we show in Supp.

B that the cross-dimensional attention map has the following property: In summary, the Independent builds attention stream for each dimension while the Joint directly model the attention map among all the units.

Our proposed CDSA is based on the Decomposed, which forms a cross-dimensional attention map, out of three dimension-specific maps.

As an alternative of the Decomposed, the Shared maps unit X(p) to Q(p, :) and K(p, :) of d-dim and calculates all three dimension-specific attention map, e.g., Table 1 , by using Tensorflow profile and fixing the hyper-parameters with detailed explanations in Supp., the Decomposed significantly decreases the FLoating point OPerations (FLOPs) compared to the Joint and requires less variables than the Independent.

Detailed comparisons are reported in Sec. 4.3.

Fig. 3(a) , we apply our CDSA mechanism in a Transformer Encoder, a stack of N = 8 identical layers with residual connection (He et al. (2016) ) and normalization (Lei Ba et al. (2016) ) as employed by Vaswani et al. (2017) .

To reconstruct the missing (along with other) values of the input, we apply a fully connected Feed Forward network on the final Value tensor, which is trained jointly with the rest of the model.

As shown in Fig. 3(b) , we apply our CDSA mechanism in Transformer framework where we set N = 9 for both encoder and decoder.

Similar to imputation, we use a fully connected feed forward network to generate the predicted values.

We normalize each measurement of the input by subtracting the mean and dividing by standard deviation across training data.

Then the entries with missed value are set 0.

We use the Adam optimizer (Kingma & Ba (2014) ) to minimize the Root Mean Square Error (RMSE) between the prediction and ground truth.

The model is trained on a single NVIDIA GTX 1080 Ti GPU.

More details (e.g., network hyper-parameters, learning rate and batch size) can be found in Supp.

NYC-Traffic.

New York City Department of Transportation has set up various street cameras 1 .

Each camera keeps taking a snapshot every a few seconds.

The is collected around 1-month data for 186 cameras on Manhattan from 12/03/2015 to 12/26/2015.

For each snapshot, we apply our trained faster-RCNN (Ren et al. (2015) ) vehicle detection model to detect the number of vehicles (#vehicle) in each snapshot.

To aggregate such raw data into time series, for every non-overlapping 5-minute window, we averaged #vehicle from each snapshot to obtain the average #vehicle as the only measurement.

Finally, we obtained 186 time series and the gap between two consecutive time stamps is 5 minutes.

The natural missing rate of the whole dataset is 8.43%.

In order to simulate experiments for imputation, we further remove some entries and hold them as ground truth for evaluation.

The imputation task is to estimate values of these removed entries.

To mimic the natural data missing pattern, we model our manual removal as a Burst Loss, which means at certain location the data is continuously missed for a certain period.

More details about vehicle detection and burst loss are be found in Supp.

To simulate various data missing extents, we vary the final missing rate after removal from 20% to 90%.

For each missing rate, we randomly select 432 consecutive time slots to train our model and evaluate the average RMSE of 5 trials.

The dataset will be released publicly. (2018)) is an Air Quality and Meteorology dataset recorded hourly.

As indicated in Luo et al. (2018a) , 11 locations and 12 measurements are selected.

The natural missing rate is 6.83%.

In order to simulate experiments for imputation, we follow Luo et al. (2018a) to split the data to every 48 hours, randomly hold values of some available entries and vary the missing rate from 20% to 90%.

Mean Squared Error (MSE) is used for evaluation.

METR-LA (Jagadish et al. (2014)).

We follow Li et al. (2018) to use this dataset for traffic speed forecasting.

This dataset contains traffic speed at 207 locations recorded every 5 minutes for 4 months.

Following Li et al. (2018), 80% of data at the beginning of these 4 months is used for training and the remaining 20% is for testing.

In order to simulate the forecasting scenario, within either training or testing set, every time series of consecutive 2 hours are enumerated.

For each time series, data in the first hour is treated as input and data in the second hour is to be predicted.

We respectively evaluate the forecasting results at 15-th, 30-th, 60-th minutes in the second 1 hour and also evalaute the average evaluation results within the total 1 hour.

We use RMSE, Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) as evaluation metrics.

Table 2 , our CDSA consistently outperforms traditional methods (i.e., Auto Regressive, Kriging expo, Kriging linear) and recent RNN-based methods (i.e. MTSI, BRITS, DCRNN) over a wide range of missing rate.

Because CDSA leverages the self-attention mechanism to avoid sequential processing of RNN and directly model the relationship between distant data.

Table 3 shows that our method again achieves significant improvements on cross-dimensional data imputation task.

Detailed overview of baselines can be found in Supp.

Forecasting (METR-LA).

Table 4 shows that for the forecasting task, our CDSA method outperforms previous methods in most cases.

In particular, our method demonstrates clear improvement at long-term forecasting such as 60 min.

This again confirms that our CDSA cthe effectiveness of directly modeling the relationship between every two data values (could from different dimensions and of far distance).

But RNN-based methods and methods that sequentially conduct spatial conv and temporal conv fail to model the distant spatio-temporal relationship explicitly.

The effects of different training losses: For the forecasting task in METR-LA, we compare the performance by setting different training loss in Table 5 and we can see the performance with RMSE as loss metric achieves the best performance.

Ablation study of different cross-dimensional self-attention manners: We compare the performance for different solutions in CDSA mechanism on the three datasets listed above.

1) The way of attention modeling determines the computational complexity.

As shown in Table 1 , since the Independent calculates dimension-specific Value vectors in parallel, the number of variables and FLOPs are larger than those of the Decomposed.

As the Joint and the Shared both share the variables for each dimension, the number of variables is small and basically equals with each other.

As the Joint builds a huge attention map, its FLOPs is much larger than others.

Since the Decomposed draws attention maps like the Independent but shares Value like the Joint, it reduces the computational complexity significantly.

2) As shown in Table 6 -8, we evaluate these methods on three datasets and the Decomposed always achieves the best performance thanks to the better learning ability compared to the Joint and Shared.

More discussions can be found in Supp.

Study of using the imputed time series for forecasting.

On NYC-Traffic of missing rate 50%, we impute missing values in historical data (using statistical methods and our CDSA respectively) and Attention Map Visualization: Fig. 4 shows an PM10 imputation example in location fangshan at t 2 .

Since the pattern of PM2.5 around t 2 is similar to that at t 1 , the attention in orange box is high.

As we can see that PM2.5 and PM10 are strongly correlated , in order to impute PM10 at t 2 , our model utilizes PM10 at t 1 (green arrow) and PM2.5 at t 1 (blue arrow), which crosses dimensions.

More visualization examples can be found in Supp.

In this paper, we have proposed a cross-dimensional self-attention mechanism to impute the missing values in multivariate, geo-tagged time series data.

We have proposed and investigated three methods to model the cross-dimensional self-attention.

Experiments show that our proposed model achieves superior results to the state-of-the-art methods on both imputation and forecasting tasks.

Given the encouraging results, in the future we plan to extend our CDSA mechanism from multivariate, geo-tagged time series to the input that has higher dimension and involves multiple data modalities.

A MODEL ARCHITECTURE

Under the NLP scenario, each word is embedded as a vector and normalized individually.

However, in the Cross-Dimensional scenario, the normalization applied on each individual unit will always lead to a zero-output.

As shown in Fig. 5 , different measurements may exhibit different correlation, i.e., PM2.5 and PM 10 are significantly positively correlated (?? PM2.5, PM 10 = 0.8278) while NO 2 and O 3 are negatively correlated (?? NO2,O3 = ???0.5117).

As discussed in Fig. 4(b) in the paper, different measurement may be used as reference for imputation of other measurements.

As such, the normalization cross multiple measurements is unreasonable and we choose to apply normalization for each measurement in parallel which presumes that the time series inside the spatial network is essentially drawn from a standard normal distribution.

We subsequently add the trainable scalar and bias to scale the normalized value and the scalars (biases) for different measurements are trained individually.

Making use of the approximation property of multi-linear layer Hornik et al. (1989) , a fully connected feed-forward network (FFN) is applied to each unit separately and identically.

This FFN consists of three fully connected layer while RuLU is set as the activation function.

During experiment, since the FFN is applied on each unit individually, we found the improvement by simply increasing the size of weight and bias of each layer is not obvious while increasing the depth of FFN will lead to obvious improvement.

For the self-attention sub-layer in imputation task NYC and KDD-2018, we modify the attention map with mask in (7) to prevent unit of available observation from contributing to the estimation of itself.

where q i and k j are d-dim vectors.

This masking, combined with fact that there is no offset between input and output, ensures that the estimation of unit X(p(t, l, m)) depends on all the units except for itself, including both available and complemented units.

In this way, the gradient back-propagation can be used to update the missed value effectively.

The Table 9 shows the performance improvement of imputation mask applied in our model and demonstrate that mask prevents the estimation of itself and improve the inference ability of the model.

Joint As shown in Fig. 6 , when we build the attention among different units in the Joint, two different kernels will be used to map each input unit X(p) = X (t, l, m) to an 1-D Query vector and an 1-D Key vector individually.

As attention map is a scaled dot-product between Query and Key (Fig. 6 Left) after Softmax, each value of attention map in Joint is essentially the scaled numerical multiplication between each two units of input (Fig. 6 Right) after Softmax.

As such, the multiple parameters inside that kernels only perform as a single scalar and the learning ability/relationship representation in Joint is limited.

Decomposed According to Sec. 3.2.3, to calculate the dimension-specific attention map, e.g., the attention map of Time A T , the input X will be reshaped into matrix X T .

Thus, the units corresponding to the same timestamp, reshaped into one vector X T (t, :), will be mapped into dimension-specific Query vector Q T (t, :) and Key vector K T (t, :).

As more parameters will be introduced into such vector-vector mapping, each dimensional-specific map can learn the intra-correlation and the crossdimensional attention map can model the relationship among each input units effectively.

Shared Same with the Decomposed, the Shared will calculate the dimension-specific attention maps individually.

Like Joint, the Shared will map each unit X(p) into vector.

To calculate the dimensionspecific attention map (e.g., A T ), the "vector-vector mapping" is essentially the summation of the units corresponding to the same timestamp while the multiple parameters introduced in the mapping still perform as a single scalar.

As a result, the learning ability of the intra-correlation is limited so that the cross-dimensional attention map cannot model the relationship among input units effectively.

As described in (2) in the paper, the original attention maps A T , A L and A M are reshaped to A T , A L and A M .

By setting T = L = M = 2 as an example, we draw the attention maps before and after reshape in Fig. 7 .

Making use of the matrix structure, we have

where

Besides, the reshape operation for attention map on different dimension is only determined by the index mapping function, p = p(t, l, m) = LM t + M l + m, from the 3-dimensional cube to the vector form.

C CDSA FRAMEWORK FOR DATA FORECASTING Different from time series imputation task, the time series forecasting task use the current observation to estimate the time series in the future.

To begin with, We first introduce the framework for time series forecasting and we compare the performance for two different types of input.

As shown in Fig. 8 , we apply our CDSA mechanism in Transformer framework and use the same Feed Forward structure as in Sec. 3.3 in the paper.

Notably, we set N = 9 for both encoder and decoder and no CDSA module is used to derive a complement input for prediction task where the missing value is replaced with global mean.

The architecture detail is shown in Fig. 9 The decoder in NLP task originally sets the shifted output as input.

Take the German-to-English translation scenario as an example where the embedded word vectors of German are set as the Encoder input, the model will first send a [GO] vector into the decoder and generate the first word vector of the translated English sequence, then the predicted vector will be sent into the decoder to predict the next word vector and the decoder will complete the sentence translation by repeating this operation until the end.

Mapping directly from this model setting in NLP task to our series forecasting scenario, we can also use the shifted ground truth as the decoder input, i.e., to forecast the speed of the next T time stamps given the speed of the first T time stamps, the data of T ??? t ??? 2T ??? 1 are sent into the decoder.

Consequently, the Casual Mask in Vaswani et al. (2017) need to be modified to make sure that the leftward information flow is prevented.

For data forecasting by CDSA in the Decomposed, the masking on Attention Map on Time is simply masking out (setting to ??????) the values in the input of Softmax which corresponds to the leftward information flow.

Same with Vaswani et al. (2017) , the masking is only adopted in the Multi-head CDSA layer labeled as (Mask TLM) in Fig. 9 .

However, as shown in Fig. 10 , to calculate the Attention Map of Location and Measurement for data forecasting at 2T ??? 1, all illegal units of input corresponding to t ??? 2T ??? 1 have to be masked out (setting as 0).

Then, the Masked Input are mapped to Query, Key and Value to build the Attention Map and calculate the Updated Value.

Besides, the decoder generates predictions given previous ground truth observations during training while the ground truth observations are replaced by predictions generated by the model itself during testing.

As, the discrepancy between the input distributions of training and testing can cause degraded performance, We adopt the integrated sampling Bengio et al. (2015) as in Li et al. (2018) to mitigate this impact while this method is very time-consuming for the Transformer framework.

During testing, In summary, by setting shifted output as the Decoder input, multiple Attention Map are calculated for forecasting value of different time stamps which requires huge memory usage.

Still, integrated sampling makes this framework suffer from an exhausted training time, since we need to send the predicted output back to decoder (Run) and repeat this Run for T times.

During testing, we can use the output corresponding to its own Run (Step) as the predicted result, as well as the output of the last run (Final).

As shown in the first 2 columns in Table 10 , the performance of outputs in the last run (Final) is better than that of Step mode, which means the leftward information flow still exists to break the auto-regressive property in data forecasting even though the mask is adopted on the input data.

For fair comparison, the models for testing are trained in one GPU and the training time are all less than 50 hours.

Typically, missing value still exists in the original dataset.

During experiment, we use the global mean to replace the missing value (Mean).

We also compare the prediction performance between the input with Mean Filling and Complemented Input of Sec. 3.3.

and the results in Table 10 shows the Complemented Input does not lead to performance improvement but increase the training workload.

Consequently, we make encoder and decoder share the same input to reduce the memory usage and training time while our model achieves better performance for long-term prediction.

D DATASET DESCRIPTION D.1 DATA AGGREGATION ON KDD 2018 DATASET et al. (2018a) select 11 common locations between the two datasets and the measurements of paired locations are concatenated.

The location pairs are described in Table 11 .

Since the unit of some measurements are label-based, e.g., the measurement weather denotes the types of weather including sunny, rainy and etc, these label are replaced with value such as 1, 2, ...

Burst loss simulation: As shown in Fig. 11 , we term the loss area marked as blue as the Burst loss area where for a certain camera, the data is continuously missed for a ?? time slots.

After statistics and analysis, we found the the length of time slots 2 ??? ?? ??? 134.

Then, for those time slots of burst loss, we calculate the mean ?? = 6.350773 and standard deviation ?? = 9.809643.

With the mean and standard deviation, We model the generation of burst loss as Gaussian process.

We provide the attention map examples extracted from the last CDSA layer.

As shown in Fig. 4(b) , correlation exists between different measurement, i.e., PM2.5 and PM10 are highly correlated.

As shown in Fig. 12(a) , the estimation of PM2.5 and PM10 is also highly relied on each other, i.e., for the estimation of PM2.5, the color in second unit, representing the weight of PM10, is darker than the rest in the first row.

As shown in Fig. 12(b) and Fig. 13 , the arrow/unit with deeper color indicates a higher weight and the index of location can be found in Table 11 .

According to the map in Fig. 13 , in most cases, neighbouring locations often share higher attention weight, e.g., the estimation at location 1 is mainly relied on the available data from location 2, location 3 and location 4.

However, the estimation of location 11 is not relied on its neighbor (location 6), instead, it is mainly relied on the location 8.

We think this relation is induced since both location 11 and location 8 are the center of express way while they are away from the urban area.

Thus, the air condition from those two location my highly correlated.

Timeslots ( Besides the sample in Fig. 4 where the missing value can be estimated from the cross-dimensional available data, Fig. 14 visualizes another example and further shows that when predicting missing value A, our model pays strong attention to available values C and D while also some attention to another missing value B.

E.3 RUNNING TIME COMPARISON Following the model hyper-parameter setting in Tabls.

1, we further compare the average running time for one segment during testing.

As the way of attention modeling determines the computational efficiency, computation method with higher FLOPs also leads to longer running time.

As shown in Table.

12, the running time of Joint is much higher than the rest 3 methods.

Since the computation schemes of Shared is similar with the Decomposed, while the number of trainable variables of Shared is much less that of Decomposed, the average processing time of Shared is a bit smaller than the running time of the Decomposed.

As described in the main paper, we use the 23-day data of NYC-Traffic for further forecasting.

We split the data into two segments, one segment contained the data of the first 20 days and the other contained the data of the rest 3 days.

We used the imputed data from the first segment to forecast the value of the second segment.

To provide comprehensive comparison, according to different missing ratio (i.e., 30%, 50% 70%), we remove the value of some units in the first segment according to burst loss and then feed the segment with missing value into the data statistical imputation model (i.e., Mean filling, Kriging expo, Kriging linear) and deep learning methods (MTSI, DCRNN, BRITS, CDSA(ours)).

Then, we feed the imputed data in to prediction model (ARIMA, Random Forest Friedman et al. (2001) ) and evaluate the forecasting performance in terms of RMSE.

According to the Fig.15 , we can find our proposed model always outperforms than other method.

We compare our method with both deep learning based methods and statistical methods while all statistical methods are adopted for each measurement individually.

??? Mean Filling: Replace the missing data with global mean.

??? Auto Regressive Akaike (1969) ??? Multi-Imputation by Chained Equation ( The dataset in METR-LA also has missing data while the missing rate is 91%.

Thus, the segment sample whose all units are zero, i.e., all-zero sample, exists.

During training, the all-zero sample (in training set) essentially has no contribution for the model training.

During testing and validation, the evaluation metric will of such samples will not be counted.

Data Pre-processing We apply Z-score normalization on each measurement as (9) respectively and fill the missing value with 0.

Optimizer We use the Adam optimizer Kingma & Ba (2014) while the initial learning rate in each epoch is set as lr(e) = r 0 ?? ?? ceil(max(0,e???d)/i) .

F.3 KDD-2015

For KDD-2018 , Luo et al. (2018a adopts content loss in a GAN-based model to train the random noise and then estimate the missing value, i.e., for one data segment, according to the specified missing rate, some available data will be held to evaluate the imputation performance, while the remaining available data will be set as groundtruth to calculate the content loss.

Our experiment on NYC-Traffic follows the same experiment setup as KDD-2018 while the noise is replaced with the remaining available data and the model parameter is trained according to the content loss.

Thus, there is no division of training, validation or testing since the training loss is not calculated from the held available data.

To comprehensively develop our experiment, we also adopting our method on KDD-2015 and follow the experiment setup in Yi et al. (2016); Cao et al. (2018) while the available data will be trained to predict the held data directly.

-2015 - (Zheng et al. (2015 In order to simulate experiments for imputation, besides the natural missing data, for PM2.5 we follow the strategy used in (Yi et al. (2016); Cao et al. (2018); Zhou & Huang (2018) ) to further manually remove entries and hold the corresponding value as ground truth.

The imputation task is to predict values of these manually removed entries.

For Temperature and Humidity, we follow Zhou & Huang (2018) to randomly hold 20% of available data.

Table 14 shows that for PM2.5, our method outperforms the traditional methods significantly and achieves comparable MAE as IIN (Zhou & Huang (2018) ) while better MRE than IIN (Zhou & Huang (2018) ).

For Temperature and Humidity, our method consistently outperforms state-of-the-art methods.

Since the Decomposed draws attention maps as the Independent but shares Value as the Joint, it reduces the computational complexity significantly.

As shown in Table 15 , we also evaluate these methods on KDD-2015 datasets and the Decomposed achieves the best performance.

Since there are missing data (Naturally missing data) in the original dataset, to evaluate the model performance, we manually remove some of the available observation (Manually removed data) and hold those entries' value as ground truth for evaluation.

The rest data are termed as Available data.

Thus, as a counterpart of Naturally missing data in the original dataset, the Naturally available data consists of Manually removed data and Available data.

NYC: Like KDD-2018, the imputation task assumes the completed data has no label.

For the estimated data, the loss is calculated on the Available data part while the evaluation metric is calculated on the Manually removed data part.

Following the setting in Li et al. (2018), we split the data into training set and testing set where this prediction task assumes the predicted data has labels.

For model structure, we set (d T , d L , v) as (14, 6, 3) and there are 16 heads in each layer.

During training, we set (r 0 , ??, d, batch size) as (0.008, 0.5, 40, 16).

@highlight

A novel self-attention mechanism for multivariate, geo-tagged time series imputation.

@highlight

This paper proposes the problem of applying the transformer network to spatiotemporal data in a compuationally efficient way, and investigates ways of implementing 3D attention.

@highlight

This paper empirically studies the effectiveness of transformer models for time series data imputation across dimensions of the input.