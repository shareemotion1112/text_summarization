State-of-the-art neural machine translation methods employ massive amounts of parameters.

Drastically reducing computational costs of such methods without affecting performance has been up to this point unsolved.

In this work, we propose a quantization strategy tailored to the Transformer architecture.

We evaluate our method on the WMT14 EN-FR and WMT14 EN-DE translation tasks and achieve state-of-the-art quantization results for the Transformer, obtaining no loss in BLEU scores compared to the non-quantized baseline.

We further compress the Transformer by showing that, once the model is trained, a good portion of the nodes in the encoder can be removed without causing any loss in BLEU.

The idea of using neural networks for machine translation was proposed only recently (Kalchbrenner & Blunsom, 2013; Sutskever et al., 2014; .

Nonetheless, the approach has reached impressive levels of translation. (Ahmed et al., 2017; .

A key element of this success was to allow the decoder to attend to all hidden states of the encoder .

A few variations to this additive attention mechanism were proposed, such as multiplicative attention and self-attention (Luong et al., 2015; Cheng et al., 2016; Lin et al., 2017) .

The latter formed the basis of the Transformer network (Vaswani et al., 2017) , which achieved stateof-the-art machine translation.

Inspiring a new wave of work, numerous natural language processing tasks reached new heights (Devlin et al., 2018; Liu et al., 2019) .

Unfortunately, these models make use of an enormous amount of parameters.

Inference on resource-limited hardware such as edgedevices is thus impractical.

A solution to reduce the computational burden of these neural networks is to lower numerical precision.

Consequently, numerical values can be represented using fewer bits (Tang & Kwan, 1993; Marchesi et al., 1993) .

This method called quantization has the advantage of providing good compression rates with minimal loss in accuracy.

It is also conveniently supported by most hardware.

Properly quantizing the Transformer would allow computational speed gains at inference, as well as deployment on more constrained devices.

In this work, we propose a custom quantization strategy of the entire Transformer architecture, where quantization is applied during the training process.

Our method is easy to implement and results are consistent with the full-precision Transformer.

We test our approach on multiple translation tasks such as WMT14 EN-FR and WMT14 EN-DE and obtain state-of-the-art quantization results.

On most tasks, our quantized models score equal or higher BLEU compared to full-precision.

We are, to the best of our knowledge, the first to fully quantize the Transformer architecture without impairing translation quality.

In this section, we review a broad spectrum of quantization and pruning methods for neural network compression.

Quantization has also been jointly used with other compression methods.

Han et al. (2015) combine pruning, quantization, weight sharing and Huffman coding, while Polino et al. (2018) use quantization with knowledge distillation (Hinton et al., 2015) for higher compression rates.

LeCun et al. (1990) were the first to propose a Hessian based method to prune neural net weights, with Hassibi et al. (1994) later improving the method.

More recently, See et al. (2016) show that pruning a fully trained model and then retraining it can increase performance over the original nonpruned model.

Gradually pruning in tandem with training has also been shown to increase performance (Zhu & Gupta, 2017) .

prune nodes instead of weights by applying a penalty in the loss on the γ parameters of batch normalization layers.

Narang et al. (2017b) make better use of hardware by applying pruning and weight decay in blocks to minimize the number of loaded weight matrix chunks.

Chen et al. (2018) combine quantization with block based low-rank matrix approximation of embeddings.

Pruning methods have also been adapted to specific architectures.

Liu et al. (2015) propose an efficient sparse matrix multiplication algorithm for CNNs.

For RNNs, Narang et al. (2017a) show sparse pruning to work well on the architecture.

In order to maintain dimension consistency, Wen et al. (2017) propose to prune all basic LSTM structures concurrently.

Park et al. (2018) introduce simple recurrent units (SRUs) for easy pruning of RNNs.

3 QUANTIZATION STRATEGY 3.1 QUANTIZATION METHOD Our quantization methodology was chosen to be uniform, meaning that the step size between two quantized values is constant.

This choice, which is an additional constraint, was made for practical reasons.

It indeed simplifies all computations required during inference, enabling the exploitation of hardware resources more efficiently.

If the performance with uniform quantization is already on par with full-precision, then more weighty methods are unnecessary.

The uniform quantization scheme employed is further described by Jacob et al. (2017) .

Given an element x of a tensor X, we apply the quantization function Q:

where x min and x max defines the endpoints of the quantization interval.

When quantization is applied to weights, these values are respectively min(X) and max(X).

However, when quantization is applied to activations, those values are running estimates.

The latter are computed during training, where for every forward pass, the x min and x max variables are updated via an exponential moving average with a momentum of 0.9.

The value k is the bit precision.

For example, in the context of 8-bit quantization, k = 8.

At training time, we simulate quantization by first quantizing and then rescaling to the original domain:

where the clamp function associates all the values outside the [x min , x max ] range to the closest endpoint and · represents rounding to the nearest integer.

During backpropagation, we use the straight-through estimator (Hinton, 2012) and set the gradients of clamped values to zero.

The only exception is for the LayerNorm's denominator, where values can still be clamped, but gradients are never zeroed.

Once training is finished, s and x min are frozen along with the weights.

We choose to quantize all operations which can provide a computational speed gain at inference.

In this regard, we quantize all matrix multiplications, meaning that the inputs and weights of MatMuls will both be k-bit quantized.

The other operations we quantize are divisions, but only if both the numerator and denominator are second or higher rank tensors.

For all other operations, such as sums, the computational cost added by the quantization operation outweighs the benefit of performing the operation with reduced precision.

Hence, we do not quantize such operations.

More precisely, we quantize all weights of the Transformer, excluding biases.

The latter are summed with the INT32 output of matrix multiplications and thus provide no additional computational efficiency from being quantized.

Furthermore, the memory space of biases is insignificant in comparison to the weight matrices, representing less than 0.1% of total weights.

For positional embeddings, memory gain is also minimal, but since these will be summed with the quantized input embeddings, we likewise quantize them.

These embeddings also stay fixed, we can thus quantize them once prior to training.

The γ weights of LayerNorms are also quantized.

As for activations, we quantize the sum of the input embeddings with the positional encodings in both the encoder and decoder.

In the Multi-Head Attention, we quantize the (Q, K, V ) input, the softmax's numerator, the softmax's denominator, the softmax's output and the Scaled Dot-Product Attention's output.

For the position-wise feed-forward networks, we quantize the output of the ReLUs and of the feed-forward networks themselves.

Finally, for all LayerNorms, we quantize the numerator x−µ, the denominator √ σ 2 + , their quotient and the output of the LayerNorm.

Instead of using a single set of (s, x min ) per quantized tensor, we can quantize subsets of the latter with each its own set of (s, x min ) (Alistarh et al., 2016) .

Even though this adds more scalars, the memory cost is insignificant overall.

Furthermore, the added flexibility can greatly alleviate the precision loss resulting from trying to fit all values of a tensor into a single domain with lower numerical precision.

We use this bucketing method for all weight matrices, with the number of subset equal to the output dimension.

For activations, we use bucketing when quantizing: the sum of input embeddings with the positional encoding, the Q, K, V inputs, the Scaled Dot-Product Attention's output, the feed-forward's output, the LayerNorm's numerator, the LayerNorm's quotient and the LayerNorm's output.

Unlike Jacob et al. (2017), we do not nudge the domain so that the zero value gets perfectly mapped.

The only zero values which we have to deal with are the padding, the output of ReLU layers and dropouts.

Since padding has no effect on the final output, we completely ignore these values when quantizing.

For ReLUs, we fix the x min estimate of those quantization layers to 0, which guarantees the perfect mapping of the value.

Finally, quantization is applied before any dropout operation.

Indeed, even though the zeros added to the output of the quantization layer might not be part of the domain, this only happens during training.

Recently, simple quantization solutions have been applied to the Transformer.

Cheong & Daniel (2019) apply k-means quantization and binarization with two centroids over the weights of the network.

For both methods, a look up table associated with each quantized layer is used to map indices to their corresponding centroids.

Similarly, Fan (2019) compares binary, 4 and 8-bit uniform quantization of the Transformer weights.

A big disadvantage with quantizing only the weights of a network is that operations must still be performed in full-precision.

Even though there is a reduced memory usage of parameters, these constantly have to be casted back to full-precision.

Achieving quantization of both weights and activations is thus much more beneficial.

The first attempt at doing so for the Transformer applied 8-bit quantization on weights and inputs of feed forward layers and binarizes the (Q, K) input of the Multi-Head Attention (Tierno, 2019) .

The scaling factor √ d k is approximated by a constant which can be computed as a right bitshift.

The method though results in huge drop in translation accuracy.

Achieving better performance, Bhandare et al. (2019) quantize certain MatMul operations and use the KL divergence to estimate the most suited parameters for each quantization range.

They restrain from quantizing all MatMuls, reporting that this resulted in poor accuracy.

All of these methods omit quantizing the whole Transformer architecture, resulting in suboptimal computational efficiency.

Furthermore, these solutions all fail to avoid impairing translation quality.

Our method achieves both.

In this section, we present the results of our full quantization scheme on various tasks.

We first compare our method on a machine translation setup.

We then present the results of numerous ablation studies.

We also compare the impact of delaying quantization on translation quality.

Finally, we evaluate our method on two language model tasks.

We apply our quantization strategy on both the base and big Transformer (Vaswani et al., 2017) .

The training setup of all presented models is the same as in the original paper, with the exception that the dropout ratio is set to 0.1 in all cases.

We refer readers to the original paper for experimental details.

Our models were first evaluated on the WMT 2014 / 2017 English-to-German and WMT 2014 English-to-French translation tasks.

Section A contains results for additional languages.

Reported perplexity is per token and BLEU was measured with multi-bleu.pl 1 on the newstest2014 2 test set.

We used beam search with a beam size of 4 and a length penalty of 0.6, as in (Vaswani et al., 2017) .

Another difference is that no checkpoint averaging was performed.

We compare our results with the original Transformer and other 8-bit quantization methods in Table  1 .

All models are base Transformers.

Original uncompressed size is the same in all cases.

Most work do not report their compressed model size.

For those, we give lower bounds based on their reports.

Our BLEU score was computed on the test set using the checkpoint with the highest validation accuracy of 2 million training steps.

Validation was computed every training epoch.

In Table 2 , we show performance of our method on the WMT14 EN-DE and WMT14 EN-FR for a fixed amount of training steps.

We compare our results with two full-precision Transformers: base and big variants.

Training the quantized models was about twice as slow as training the baselines.

We also compare with two other quantization approaches.

The first one is the "default" approach, which is to naively quantize every possible operation.

The second approach applies our quantization strategy post-training (see section 5.3 for details).

In all cases except for post-quantization, BLEU was computed on the test set using the checkpoint which scored the highest accuracy on the validation set.

Towards the end of training, we ran one validation epoch for every 100 training steps.

As for post-training quantization, the BLEU score was computed on the test set using the best scoring BLEU score on the validation set out of 20 trials.

The latter varied by about 0.2 BLEU.

For the big Transformer variants, best results were obtained when not bucketing the Scaled Dot-Product Attention's output and the sum of the decoder's input embeddings with the positional encoding.

The reason for the default approach's nan in the EN-FR task is because quantizing every operation causes numerical instability in the LayerNorm's denominator, normally provided by the .

Generally, fully quantizing the Transformer seems to result in no loss in translation accuracy.

We believe the reason for this might be the lower numerical precision acting as a regularization effect.

Looking for such an effect in the training and validation curves, differences were too subtle for any conclusions to be made.

All models use full-precision biases, s and x min .

This amounts to 6.52 Mb in the base models and 13.04 Mb in the big models.

Without bucketing, this would amount to 2.17 Mb and 4.33 Mb respectively.

All in all, these represent less than 2% of the total size of our quantized models.

We believe the small increase in model size is worth it in the case of bucketing.

We show in section 5.2 that training without leads to poorer translation.

Although 6-bit quantization seems to perform well, the compression advantage over 8-bit is usually lost.

Most hardware store INT6 using either 8 or 32 bits.

Dedicated hardware is needed to get the full compression advantage.

Unless 6-bit quantization results in better models, sticking to 8-bit seems like the best choice for most hardware.

To compare the effect of bucketing and better understand which operation is more sensitive to quantization, we evaluate the effect of quantizing to 8-bit single operations of the Transformer, with and without bucketing.

By single operation, we mean quantizing the operation of a module for all Transformer layers.

Table 3 shows results on the WMT14 EN-FR translation task.

BLEU was computed on the test set after 100k steps of training.

The only operations underperforming our full-precision baseline of 38.36 BLEU are the LayerNorm's numerator when not bucketed and the denominator.

The latter cannot be bucketed because all dimensions of the variance tensor vary per batch.

Solely quantizing the LayerNorm's denominator with no bucketing works, but results are poor.

To successfully quantize this element without causing performance loss, we suspect quantizing other elements in the network helps.

To further validate our quantization scheme, we evaluated four models trained with alterations to our design choices.

Results are presented in Table 4 ).

All models are 8-bit quantized base Transformers, trained in the same fashion as in section 5.1 on the WMT14 EN-FR task.

Our method's goal is to increase computational efficiency when inferring with the Transformer.

To this end, our quantization scheme only requires us to learn s and x min .

Although we do so with our quantization scheme throughout the whole training, this is not a necessity.

Quantization could also be applied later on while training.

Results for different starting points are compared in Table 5 .

The earliest we start quantizing is at 100 steps, since we need at least a few steps to assess the x min and x max running estimates.

We consider this to be training with quantization "from scratch".

Posttraining quantization is also an option, where once the model is fully trained, we keep the weights fixed, but compute the s, x min and x max over a few hundred steps.

All models were evaluated on the WMT14 EN-DE and WMT14 EN-FR translation tasks.

BLEU was measured on the test set using the checkpoint which scored the highest accuracy on the validation set during training.

Validation was computed every 100 training steps towards the end of training.

From our observed results, quantizing the model early on seems preferable.

This reinforces our belief that quantization helps training via regularization.

Learning quantization parameters adds a significant computational cost during training.

A major advantage to delaying quantization is to perform more training steps in the same given amount of time.

Therefore, when training time is a constraint, a possible strategy is to train a model without quantization, perform more training steps and finally post-quantize the model.

Another advantage of post-quantization is that the method is quick to perform.

This makes it easy to run many iterations and search for the best performing candidate.

To evaluate if our quantization scheme generalizes well to other tasks, we evaluate it on a language modeling benchmark.

As the setup, we use PyTorch's language modeling toy example 3 on the WikiText-2 and WikiText-103 corpus.

The task consists of predicting the sequence {x t+1 , · · · , x t+n+1 } from the input sequence {x t , · · · , x t+n }.

We trained four Transformer models, one full precision and three with our quantization method.

In each case, the model consists of two Transformer encoder layers, with the embedding and hidden size set to 200.

Multi-Head Atten-tion has two heads with keys and values of size 64.

The final word projection layer's weights are shared with the embedding layer.

Models were trained for 10 epochs with a batch size of 20 and sequence length of 35.

Learning rate is set to 5, dropout to 0.2 and gradient clipping to 0.25.

Loss is computed on every element of the output sequence.

We refer readers to the PyTorch example (see 3) for any extra details.

Results are presented in Table 6 .

Validation loss was computed every epoch to determine the best candidate.

Loss and perplexity are computed on the test set and averaged over 10 trials for WikiText-2 and 3 trials for WikiText-3.

We proposed a quantization strategy for the Transformer, quantizing all operations which could provide a computational speed gain, for a fully quantized architecture.

All of our design decisions were aimed at maximizing computational efficiency while making sure our method would be compatible with as many different types of hardware as possible.

With our method, we achieve higher BLEU scores than all other quantization methods for the Transformer on multiple translation tasks and avoid any loss in BLEU compared to full-precision.

Specifically, out of 41 experiments, 8-bit quantization performed equal or better to full-precision in 36 cases.

We are very excited about the possibilities this work opens and plan on applying our method to other tasks.

We also intend to extend our work to variations of the Transformer, as well as further exploring the compression of these networks.

We evaluated our quantization method on additional translation datasets (see Table 7 ).

All models are trained following the same setup as in section 5.1, except the big model was only trained for one epoch.

Vocabulary size is set to 30k for all models.

Since there is no test set for WMT14 ES-EN, we used the validation set as a test set and omitted computing any validation epochs during training.

We propose an additional compression method for the Transformer, which is independent of our quantization method.

Both though can be used conjointly to further compress the Transformer.

Once the model is fully trained and quantized, we can further compress it by removing useless nodes.

By useless, we mean nodes which do not cause any loss in translation quality when removed.

We choose to prune nodes instead of independently pruning weights, to avoid the need of any special hardware or software, which is usually the case when trying to leverage sparse weight matrices obtained by the latter method.

Pruning nodes results in concretely shrunken models.

When getting rid of a node, we remove its corresponding set of weights from the layer outputting it and the following layer receiving the node as input.

The only nodes of the Transformer which can be removed without causing alterations to other components of the network are the nodes in between the two layers of each feed-forward network.

Fortunately, these consist of a substantial portion of the model's weights.

In the case of the base Transformer, for a respective vocabulary of size 37000 and 32000, 39.96% and 41.65% of the total weights are owned by the feed-foward networks.

This number grows to 47.03% and 48.18% in the big Transformer, for again, a respective vocabulary of size 37000 and 32000.

To evaluate which nodes can be safely pruned without affecting translation quality, we estimate the maximum value x max for each node of the ReLU output over a few hundred steps.

This is done on the training set, using the fully trained model and keeping all other weights frozen.

These x max are computed before quantizing the ReLU output and do not replace the ones used by the quantization process.

Figure 1 shows the histogram of these running estimates for one ReLU layer in the encoder and one in the decoder.

All other ReLU layers share the same pattern, where in the encoder there are always multiple x max close to 0, while this not being the case for the decoder.

Once the running estimates are computed, we prune its corresponding node if x max < zσ where z is a hyperparameter and σ the standard deviation of the x max of the layer.

We empirically found z = 0.025 to work well, with higher thresholds causing BLEU to quickly decay.

No retraining of the model is performed after nodes have been pruned.

Using this pruning method, we can further compress the Transformer without affecting BLEU scores.

Table 8 shows results of our pruning method.

Our approach has the advantage of being adaptive, meaning the number of nodes pruned per layer will differ as opposed to a fixed pruning ratio method.

For example, in the case of the big Transformer trained on WMT14 EN-FR, 169 nodes were pruned in the first ReLU of the encoder, while in the second, 1226 were pruned.

Nodes in the decoder rarely got pruned, at most 4 in the whole decoder.

The threshold allows us to find the right ratio of nodes to prune per layer instead of the usual: decide the ratio first and then prune.

We compared with two such methods, where for each task, we fix the ratio to the global percentage of nodes pruned in the encoder by our method.

The first fixed pruning method uses L1-norm to sort nodes to prune in ascending order, while the second sorts nodes using the x max , also in ascending order.

Since x max is a running estimate, results varied per trial.

The method only takes a few hundred training steps to perform though, so running many trials is not an issue.

We evaluated our method on the validation set a few times until accuracy increased over the non-pruned model.

Perplexity and BLEU were then computed on the test set.

Reported results are averaged over a few trials.

When accuracy was not improving, BLEU would either be the same or decreased by about 0.01−0.02.

<|TLDR|>

@highlight

We fully quantize the Transformer to 8-bit and improve translation quality compared to the full precision model.

@highlight

An 8-bit quantization method to quantize the machine translation model Transformer, proposing to use uniform min-max quantization during inference and bucketing weigts before quantization to reduce quantization error.

@highlight

A method for reducing the required memory space by a quantization technique, focused on reducing it for Transformer architecture.